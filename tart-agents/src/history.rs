use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use async_openai::types::responses::{
    EasyInputContent, EasyInputMessageArgs, FunctionCallOutput, FunctionCallOutputItemParam,
    FunctionToolCall, InputItem, Item, ReasoningItem, ReasoningItemContent, Role,
};

use crate::Progress;

/// The system prompt for *tart*, included in every conversation.
const SYSTEM: &str = include_str!("data/SYSTEM.md");

/// An append-only conversation record for one tart session.
///
/// Clones share one record, so the agent loop writes reasoning, tool exchanges, and the
/// final answer to the callers transcript. After accumulation, the transcript is passed
/// back into the model and the conversation continues.
#[derive(Clone, Debug)]
pub struct Transcript {
    /// Every item, oldest first, starting with the system prompt.
    items: Arc<Mutex<Vec<InputItem>>>,
    /// A mode reminder appended to the end of the input at request time.
    ///
    /// This is designed to mirror Codex's instruction handling, as we can't
    /// guarantee that arbitrary endpoints support a dedicated instructions
    /// channel. Trailing the record keeps the cached prefix up to the turn on which
    /// plan mode was triggered intact.
    reminder: Option<InputItem>,
}

impl Transcript {
    /// The record under its lock.
    ///
    /// If a worker panics mid turn and poisons the lock, we recover the original record
    /// rather than killing the session.
    fn items(&self) -> MutexGuard<'_, Vec<InputItem>> {
        self.items.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// A transcript opening with the tart system prompt.
    #[inline]
    pub fn new() -> anyhow::Result<Self> {
        Ok(Self {
            items: Arc::new(Mutex::new(vec![input_message(Role::System, SYSTEM.to_string())?])),
            reminder: None,
        })
    }

    /// A transcript opening with the tart system prompt, followed by the
    /// agent's `instructions` as a second system message, when non-empty.
    #[inline]
    pub fn with_instructions(instructions: String) -> anyhow::Result<Self> {
        let transcript = Self::new()?;
        if !instructions.is_empty() {
            transcript
                .items()
                .push(input_message(Role::System, instructions)?);
        }
        Ok(transcript)
    }

    /// A transcript over `items`, as a session file restored them.
    #[inline]
    pub(crate) fn from_items(items: Vec<InputItem>) -> Self {
        Self {
            items: Arc::new(Mutex::new(items)),
            reminder: None,
        }
    }

    /// Drop the conversation, keeping the leading system items.
    #[inline]
    pub fn clear(&self) {
        let mut items = self.items();
        let systems = items
            .iter()
            .take_while(|item| matches!(item, InputItem::EasyMessage(m) if m.role == Role::System))
            .count();
        items.truncate(systems);
    }

    /// Record the user's turn.
    #[inline]
    pub fn push_user(&self, text: String) -> anyhow::Result<()> {
        self.items().push(input_message(Role::User, text)?);
        Ok(())
    }

    /// Record the assistant's final answer for the current turn.
    #[inline]
    pub fn push_assistant(&self, text: String) -> anyhow::Result<()> {
        self.items().push(input_message(Role::Assistant, text)?);
        Ok(())
    }

    /// Record the reasoning that preceded a round's tool calls.
    ///
    /// This is critical for `DeepSeek`'s thinking mode, which breaks on concurrent tool
    /// calls without it.
    pub(crate) fn push_reasoning(&self, item: ReasoningItem) {
        self.items().push(InputItem::Item(item.into()));
    }

    /// Record one round of tool exchanges.
    pub(crate) fn push_tool_round(&self, round: Vec<(FunctionToolCall, String)>) {
        let mut items = self.items();
        let mut outputs = Vec::with_capacity(round.len());
        for (call, output) in round {
            outputs.push(FunctionCallOutputItemParam {
                call_id: call.call_id.clone(),
                output: FunctionCallOutput::Text(output),
                id: None,
                status: None,
            });
            items.push(InputItem::Item(Item::FunctionCall(call)));
        }
        items.extend(
            outputs
                .into_iter()
                .map(Item::FunctionCallOutput)
                .map(InputItem::Item),
        );
    }

    /// Append `text` to the end of the input on every subsequent request, or clear with `None`.
    ///
    /// # Errors
    ///
    /// Propagates the API's argument validation, which a non-empty `text` can't fail
    #[inline]
    pub fn set_reminder(&mut self, text: Option<&str>) -> anyhow::Result<()> {
        self.reminder = match text {
            Some(text) => Some(input_message(Role::System, text.to_string())?),
            None => None,
        };
        Ok(())
    }

    /// How many items the stored record holds.
    #[inline]
    #[must_use]
    pub(crate) fn len(&self) -> usize {
        self.items().len()
    }

    /// The stored items from `start` on, oldest first, as `Session` persists
    /// and replays.
    ///
    /// A cursor read: only the tail past what a session has already flushed is
    /// cloned, so recording a turn costs its new lines, not the whole record.
    /// A `start` past the end yields nothing.
    #[inline]
    #[must_use]
    pub(crate) fn items_after(&self, start: usize) -> Vec<InputItem> {
        self.items().get(start..).map_or_else(Vec::new, ToOwned::to_owned)
    }

    /// The input items for the next request: the stored record with the
    /// reminder, when one is set, appended after it.
    #[inline]
    #[must_use]
    pub(crate) fn request_items(&self) -> Vec<InputItem> {
        let mut items = self.items().clone();
        items.extend(self.reminder.clone());
        items
    }

    /// The progress stream that renders this record, in live order, for replay.
    ///
    /// Tool exchanges replay headers only, with coloring skipped for simplicity.
    #[inline]
    pub fn replay(&self) -> Vec<Progress> {
        let items = self.items();
        items
            .iter()
            .chain(self.reminder.as_ref())
            .flat_map(Self::replay_events)
            .collect()
    }

    /// The events one recorded item replays as, in live order.
    fn replay_events(item: &InputItem) -> Vec<Progress> {
        match item {
            InputItem::EasyMessage(message) => match (&message.role, &message.content) {
                (Role::User, EasyInputContent::Text(text)) => vec![Progress::User(text.clone())],
                (Role::Assistant, EasyInputContent::Text(text)) => {
                    vec![Progress::Answer(text.clone())]
                }
                _ => Vec::new(),
            },
            InputItem::Item(Item::Reasoning(reasoning)) => {
                let text: String = reasoning
                    .content
                    .iter()
                    .flatten()
                    .map(|ReasoningItemContent::ReasoningText(part)| part.text.as_str())
                    .collect();
                if text.is_empty() {
                    Vec::new()
                } else {
                    vec![Progress::Thinking(text)]
                }
            }
            InputItem::Item(Item::FunctionCall(call)) => {
                // Just show the header, exits/output are not super necessary
                vec![
                    Progress::ToolStart {
                        id: call.call_id.clone(),
                        name: call.name.clone(),
                        arguments: call.arguments.clone(),
                    },
                    Progress::ToolOutput {
                        id: call.call_id.clone(),
                        output: String::new(),
                        exit: Some(0),
                    },
                ]
            }
            _ => Vec::new(),
        }
    }
}

/// One message in the conversation, as the Responses API sends it.
fn input_message(role: Role, text: String) -> anyhow::Result<InputItem> {
    Ok(EasyInputMessageArgs::default()
        .role(role)
        .content(text)
        .build()?
        .into())
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, reason = "test assertions")]

    use super::*;
    use async_openai::types::responses::{ReasoningItemContent, ReasoningTextContent};

    /// A finished `bash` call, as the agent loop would reconstruct it.
    fn bash_call() -> FunctionToolCall {
        FunctionToolCall {
            namespace: None,
            name: "bash".to_string(),
            arguments: r#"{"command":"ls"}"#.to_string(),
            call_id: "call_0".to_string(),
            id: Some("item_0".to_string()),
            status: None,
        }
    }

    /// A reasoning item, as the agent loop captures one.
    fn reasoning_item() -> ReasoningItem {
        ReasoningItem {
            id: Some("rs_0".to_string()),
            summary: Vec::new(),
            content: Some(vec![ReasoningItemContent::ReasoningText(ReasoningTextContent {
                text: "thinking".to_string(),
            })]),
            encrypted_content: None,
            status: None,
        }
    }

    #[test]
    fn transcript_opens_with_the_system_prompt() {
        let items = serde_json::to_value(Transcript::new().unwrap().request_items()).unwrap();

        assert_eq!(items[0]["role"], "system");
        assert_eq!(items[0]["content"], SYSTEM);
    }

    #[test]
    fn instructions_follow_the_system_prompt() {
        let transcript = Transcript::with_instructions("be terse".to_string()).unwrap();
        let items = serde_json::to_value(transcript.request_items()).unwrap();

        assert_eq!(items[1]["role"], "system");
        assert_eq!(items[1]["content"], "be terse");
    }

    #[test]
    fn empty_instructions_are_skipped() {
        let transcript = Transcript::with_instructions(String::new()).unwrap();
        let items = serde_json::to_value(transcript.request_items()).unwrap();

        assert_eq!(items.as_array().unwrap().len(), 1);
        assert_eq!(items[0]["role"], "system");
        assert_eq!(items[0]["content"], SYSTEM);
    }

    /// A reminder trails the record on every request, once, and doesn't hit record.
    #[test]
    fn a_reminder_trails_the_record_once() {
        let mut transcript = Transcript::with_instructions("be terse".to_string()).unwrap();
        transcript.push_user("look at the auth flow".to_string()).unwrap();

        // Without a reminder the request is exactly the stored record.
        assert_eq!(transcript.request_items().len(), transcript.items_after(0).len());

        transcript.set_reminder(Some("plan mode is on")).unwrap();
        let request = serde_json::to_value(transcript.request_items()).unwrap();
        let request = request.as_array().unwrap();
        // After the whole record: the last thing the model reads, and the turn
        // it answers, sit just before it.
        let last = request.len() - 1;
        assert_eq!(request[last]["role"], "system");
        assert_eq!(request[last]["content"], "plan mode is on");
        assert_eq!(request[last - 1]["role"], "user");

        // The stored record is an exact prefix of the request, so neither
        // arming nor clearing the reminder invalidates it.
        let stored = serde_json::to_value(transcript.items_after(0)).unwrap();
        for (sent, kept) in request.iter().zip(stored.as_array().unwrap()) {
            assert_eq!(sent, kept, "the record leads the request unchanged");
        }

        // It stays one copy as turns accrue, and never reaches the record.
        let record = serde_json::to_string(&transcript.items_after(0)).unwrap();
        assert!(!record.contains("plan mode is on"), "never stored: {record}");
        transcript.push_user("and the tests?".to_string()).unwrap();
        let with_two_turns = serde_json::to_string(&transcript.request_items()).unwrap();
        assert_eq!(
            with_two_turns.matches("plan mode is on").count(),
            1,
            "one copy however long the session: {with_two_turns}"
        );

        // Clearing it restores the record exactly, and still moves nothing.
        transcript.set_reminder(None).unwrap();
        assert_eq!(
            serde_json::to_value(transcript.request_items()).unwrap(),
            serde_json::to_value(transcript.items_after(0)).unwrap()
        );
    }

    /// The approval handover should leave no reminders.
    #[test]
    fn approval_leaves_no_reminder_behind() {
        let mut transcript = Transcript::new().unwrap();
        transcript
            .push_user("plan the auth refactor".to_string())
            .unwrap();
        // What the last planning request sent as its prefix: the record alone.
        let sent = serde_json::to_value(transcript.items_after(0)).unwrap();
        transcript.set_reminder(Some("plan mode is on")).unwrap();

        // The plan lands, the mode leaves, the approval turn is recorded.
        transcript
            .push_assistant("1. add a session table".to_string())
            .unwrap();
        transcript.set_reminder(None).unwrap();
        transcript
            .push_user("The plan above is approved: implement it now.".to_string())
            .unwrap();

        let request = serde_json::to_value(transcript.request_items()).unwrap();
        assert!(
            !request.to_string().contains("plan mode is on"),
            "no residue: {request}"
        );
        // The implementing request ends in the approval turn, and everything
        // the planning request sent sits in front of it unchanged.
        let request = request.as_array().unwrap();
        assert_eq!(request.last().unwrap()["role"], "user");
        for (item, cached) in request.iter().zip(sent.as_array().unwrap()) {
            assert_eq!(item, cached, "the cached prefix survives the handover");
        }
    }

    #[test]
    fn pushed_turns_serialize_in_order() {
        let transcript = Transcript::new().unwrap();
        transcript.push_user("hello".to_string()).unwrap();
        transcript.push_assistant("hi there".to_string()).unwrap();

        let items = serde_json::to_value(transcript.request_items()).unwrap();

        assert_eq!(items.as_array().unwrap().len(), 3);
        assert_eq!(items[1]["role"], "user");
        assert_eq!(items[1]["content"], "hello");
        assert_eq!(items[2]["role"], "assistant");
        assert_eq!(items[2]["content"], "hi there");
    }

    #[test]
    fn consecutive_user_messages_replay_in_order() {
        let transcript = Transcript::new().unwrap();
        transcript.push_user("run it".to_string()).unwrap();
        transcript.push_assistant("partial".to_string()).unwrap();
        transcript.push_user("actually, go faster".to_string()).unwrap();

        // An interrupted round's user/partial/user shape replays as recorded.
        let items = serde_json::to_value(transcript.request_items()).unwrap();
        assert_eq!(items[1]["role"], "user");
        assert_eq!(items[2]["role"], "assistant");
        assert_eq!(items[2]["content"], "partial");
        assert_eq!(items[3]["role"], "user");
        assert_eq!(items[3]["content"], "actually, go faster");
    }

    #[test]
    fn tool_rounds_replay_calls_grouped_before_outputs() {
        let transcript = Transcript::new().unwrap();
        let mut second = bash_call();
        second.call_id = "call_1".to_string();
        transcript.push_tool_round(vec![
            (bash_call(), "one\n".to_string()),
            (second, "two\n".to_string()),
        ]);

        let items = serde_json::to_value(transcript.request_items()).unwrap();

        // All calls first, then all outputs: interleaving would split the
        // round into several assistant messages on the provider side.
        assert_eq!(items[1]["type"], "function_call");
        assert_eq!(items[1]["call_id"], "call_0");
        assert_eq!(items[2]["type"], "function_call");
        assert_eq!(items[2]["call_id"], "call_1");
        assert_eq!(items[3]["type"], "function_call_output");
        assert_eq!(items[3]["output"], "one\n");
        assert_eq!(items[4]["type"], "function_call_output");
        assert_eq!(items[4]["output"], "two\n");
    }

    #[test]
    fn clones_share_one_record() {
        let transcript = Transcript::new().unwrap();
        let worker = transcript.clone();
        worker.push_user("hello".to_string()).unwrap();
        worker.push_tool_round(vec![(bash_call(), "one\n".to_string())]);

        // What the agent loop records lands in the caller's transcript.
        let items = serde_json::to_value(transcript.request_items()).unwrap();
        assert_eq!(items.as_array().unwrap().len(), 4);
        assert_eq!(items[1]["role"], "user");
        assert_eq!(items[2]["type"], "function_call");
        assert_eq!(items[3]["type"], "function_call_output");
    }

    #[test]
    fn a_recorded_turn_replays_in_spec_order() {
        let transcript = Transcript::new().unwrap();
        transcript.push_user("run it".to_string()).unwrap();
        transcript.push_reasoning(reasoning_item());
        transcript.push_tool_round(vec![(bash_call(), "one\n".to_string())]);
        transcript.push_assistant("done".to_string()).unwrap();

        let items = serde_json::to_value(transcript.request_items()).unwrap();

        // message, reasoning, call, output, message, each call paired with its output
        assert_eq!(items[1]["role"], "user");
        assert_eq!(items[2]["type"], "reasoning");
        assert_eq!(items[3]["type"], "function_call");
        assert_eq!(items[4]["type"], "function_call_output");
        assert_eq!(items[3]["call_id"], items[4]["call_id"]);
        assert_eq!(items[5]["role"], "assistant");
        assert_eq!(items[5]["content"], "done");
    }

    #[test]
    fn reasoning_replays_as_a_reasoning_item() {
        let transcript = Transcript::new().unwrap();
        transcript.push_reasoning(reasoning_item());

        let items = serde_json::to_value(transcript.request_items()).unwrap();

        assert_eq!(items[1]["type"], "reasoning");
        assert_eq!(items[1]["content"][0]["type"], "reasoning_text");
        assert_eq!(items[1]["content"][0]["text"], "thinking");
    }

    #[test]
    fn clear_keeps_only_the_leading_system_items() {
        let transcript = Transcript::with_instructions("be terse".to_string()).unwrap();
        transcript.push_user("hello".to_string()).unwrap();
        transcript.push_tool_round(vec![(bash_call(), "one\n".to_string())]);
        transcript.push_assistant("hi".to_string()).unwrap();

        transcript.clear();

        let items = serde_json::to_value(transcript.request_items()).unwrap();
        assert_eq!(items.as_array().unwrap().len(), 2);
        assert_eq!(items[0]["content"], SYSTEM);
        assert_eq!(items[1]["content"], "be terse");

        // Without instructions, only the prompt survives.
        let plain = Transcript::new().unwrap();
        plain.push_user("hello".to_string()).unwrap();
        plain.clear();
        let items = serde_json::to_value(plain.request_items()).unwrap();
        assert_eq!(items.as_array().unwrap().len(), 1);
        assert_eq!(items[0]["content"], SYSTEM);
    }

    #[test]
    fn request_items_is_a_copy() {
        let transcript = Transcript::new().unwrap();
        let mut items = transcript.request_items();
        items.clear();

        assert_eq!(transcript.request_items().len(), 1);
    }

    /// The cursor read yields exactly the unseen tail, never the flushed prefix.
    #[test]
    fn items_after_returns_the_unseen_tail_only() {
        let transcript = Transcript::new().unwrap();
        transcript.push_user("hello".to_string()).unwrap();
        transcript.push_assistant("hi".to_string()).unwrap();

        // From zero the whole record reads back, matching a full request.
        let whole = serde_json::to_value(transcript.items_after(0)).unwrap();
        assert_eq!(whole, serde_json::to_value(transcript.request_items()).unwrap());

        // Past the system prompt: only the turn items follow.
        let tail = serde_json::to_value(transcript.items_after(1)).unwrap();
        let tail = tail.as_array().unwrap();
        assert_eq!(tail.len(), 2);
        assert_eq!(tail[0]["role"], "user");
        assert_eq!(tail[1]["role"], "assistant");

        // Reading to the end, and past it, yields nothing without panicking:
        // a cleared record can end before a session's flushed prefix.
        assert!(transcript.items_after(transcript.len()).is_empty());
        assert!(transcript.items_after(transcript.len() + 3).is_empty());
    }

    #[test]
    fn items_round_trip_through_jsonl_lines() {
        let transcript = Transcript::new().unwrap();
        transcript.push_user("run it".to_string()).unwrap();
        transcript.push_reasoning(reasoning_item());
        transcript.push_tool_round(vec![(bash_call(), "one\n".to_string())]);
        transcript.push_assistant("done".to_string()).unwrap();

        // Every shape the harness records is one JSON line that reparses to itself.
        for item in transcript.request_items() {
            let line = serde_json::to_string(&item).unwrap();
            assert!(!line.contains('\n'), "{line}");
            assert_eq!(serde_json::from_str::<InputItem>(&line).unwrap(), item);
        }
    }

    #[test]
    fn replay_renders_words_and_tool_headers_only() {
        use Progress::{Answer, Thinking, ToolOutput, ToolStart, User};

        // System items replay as nothing.
        let transcript = Transcript::with_instructions("be terse".to_string()).unwrap();
        assert!(transcript.replay().is_empty());

        transcript.push_user("run it".to_string()).unwrap();
        transcript.push_reasoning(reasoning_item());
        let mut second = bash_call();
        second.call_id = "call_1".to_string();
        transcript.push_tool_round(vec![
            (bash_call(), "one\n".to_string()),
            (second, "[exit 2]\ntwo\n".to_string()),
        ]);
        transcript.push_assistant("done".to_string()).unwrap();

        let events = transcript.replay();

        // Each call replays as its header, finished empty: no recorded output,
        // no exit derived from its framing.
        assert!(matches!(
            events.as_slice(),
            [
                User(text),
                Thinking(thinking),
                ToolStart {
                    id,
                    name,
                    arguments,
                },
                ToolOutput {
                    output,
                    exit: Some(0),
                    ..
                },
                ToolStart {
                    id: second_id, ..
                },
                ToolOutput {
                    output: second_output,
                    exit: Some(0),
                    ..
                },
                Answer(answer),
            ] if text == "run it"
                && thinking == "thinking"
                && id == "call_0"
                && name == "bash"
                && arguments == r#"{"command":"ls"}"#
                && output.is_empty()
                && second_id == "call_1"
                && second_output.is_empty()
                && answer == "done"
        ));
    }
}
