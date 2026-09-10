use std::sync::{Arc, Mutex, MutexGuard};

use async_openai::types::responses::{
    EasyInputContent, EasyInputMessageArgs, FunctionCallOutput, FunctionCallOutputItemParam,
    FunctionToolCall, InputItem, Item, ReasoningItem, ReasoningItemContent, Role,
};

use time::OffsetDateTime;

use crate::Progress;

/// The system prompt for *tart*, included in every conversation.
const SYSTEM: &str = include_str!("data/SYSTEM.md");

/// The moment as `YYYY-MM-DD HH:MM:SS` UTC, fixed-width and lexically
/// sortable: the stamp a line records itself under.
#[inline]
pub(crate) fn stamp_utc_at(moment: OffsetDateTime) -> String {
    format!(
        "{:04}-{:02}-{:02} {:02}:{:02}:{:02}",
        moment.year(),
        u8::from(moment.month()),
        moment.day(),
        moment.hour(),
        moment.minute(),
        moment.second()
    )
}

/// The current instant as [`stamp_utc_at`] renders it.
#[inline]
fn now_stamp() -> String {
    stamp_utc_at(OffsetDateTime::now_utc())
}

/// One session line's durable metadata, preserved by every rewrite.
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct LineMetadata {
    /// When the line was first written, `YYYY-MM-DD HH:MM:SS` UTC; minted at
    /// write time when absent.
    pub(crate) timestamp: Option<String>,
}

/// One recorded item and the metadata its session line carries.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct RecordLine {
    /// The item itself, as the record and a request send it.
    pub(crate) item: InputItem,
    /// The metadata its session line writes beside it.
    pub(crate) meta: LineMetadata,
}

/// An append-only conversation record for one tart session.
///
/// Clones share one record, so the agent loop writes reasoning, tool exchanges, and the
/// final answer to the callers transcript. After accumulation, the transcript is passed
/// back into the model and the conversation continues.
#[derive(Clone, Debug)]
pub struct Transcript {
    /// The lines, oldest first, opening with the system prompt.
    lines: Arc<Mutex<Vec<RecordLine>>>,
    /// A mode reminder appended to the end of the input at request time.
    ///
    /// This is designed to mirror Codex's instruction handling, as we can't
    /// guarantee that arbitrary endpoints support a dedicated instructions
    /// channel. Trailing the record keeps the cached prefix up to the turn on which
    /// plan mode was triggered intact.
    reminder: Option<InputItem>,
}

impl Transcript {
    /// The lines under their lock.
    ///
    /// If a worker panics mid turn and poisons the lock, we recover the original record
    /// rather than killing the session.
    fn lines(&self) -> MutexGuard<'_, Vec<RecordLine>> {
        crate::locked(&self.lines)
    }

    /// A transcript opening with the tart system prompt.
    #[inline]
    pub fn new() -> anyhow::Result<Self> {
        Ok(Self {
            lines: Arc::new(Mutex::new(vec![RecordLine {
                item: input_message(Role::System, SYSTEM.to_string())?,
                meta: LineMetadata::default(),
            }])),
            reminder: None,
        })
    }

    /// A transcript over `lines`, as a session file restored them.
    #[inline]
    pub(crate) fn from_lines(lines: Vec<RecordLine>) -> Self {
        Self {
            lines: Arc::new(Mutex::new(lines)),
            reminder: None,
        }
    }

    /// Drop the conversation, keeping the leading system items.
    #[inline]
    pub fn clear(&self) {
        self.rewind(0);
    }

    /// Cut the record back to `start`, not including the system prompt.
    #[inline]
    pub fn rewind(&self, start: usize) {
        let mut lines = self.lines();
        let systems = lines
            .iter()
            .take_while(
                |line| matches!(&line.item, InputItem::EasyMessage(m) if m.role == Role::System),
            )
            .count();
        lines.truncate(start.max(systems));
    }

    /// The user messages in the record, oldest first, each with the index its turn begins at.
    #[inline]
    #[must_use]
    pub fn user_turns(&self) -> Vec<(usize, String)> {
        self.lines()
            .iter()
            .map(|line| &line.item)
            .enumerate()
            .filter_map(|(index, item)| match item {
                InputItem::EasyMessage(message) => match (&message.role, &message.content) {
                    (Role::User, EasyInputContent::Text(text)) => Some((index, text.clone())),
                    // Prompts, answers, and the content lists this harness doesn't save
                    // acan be skipped, we wil lnever rewind to them
                    _ => None,
                },
                _ => None,
            })
            .collect()
    }

    /// Record the user's turn.
    #[inline]
    pub fn push_user(&self, text: String) -> anyhow::Result<()> {
        self.push(input_message(Role::User, text)?);
        Ok(())
    }

    /// Record the assistant's final answer for the current turn.
    #[inline]
    pub fn push_assistant(&self, text: String) -> anyhow::Result<()> {
        self.push(input_message(Role::Assistant, text)?);
        Ok(())
    }

    /// Record the answer a round streamed, when it streamed one at all.
    pub(crate) fn push_answer(&self, answer: &str) -> anyhow::Result<()> {
        if answer.is_empty() {
            return Ok(());
        }
        self.push_assistant(answer.to_string())
    }

    /// Append one item, stamped with the moment of its recording.
    fn push(&self, item: InputItem) {
        self.lines().push(RecordLine {
            item,
            meta: LineMetadata { timestamp: Some(now_stamp()) },
        });
    }

    /// Record the reasoning that preceded a round's tool calls.
    ///
    /// This is critical for `DeepSeek`'s thinking mode, which breaks on concurrent tool
    /// calls without it.
    pub(crate) fn push_reasoning(&self, item: ReasoningItem) {
        self.push(InputItem::Item(item.into()));
    }

    /// Record one round of tool exchanges.
    pub(crate) fn push_tool_round(&self, round: Vec<(FunctionToolCall, String)>) {
        let mut lines = self.lines();
        let mut outputs = Vec::with_capacity(round.len());
        for (call, output) in round {
            outputs.push(FunctionCallOutputItemParam {
                call_id: call.call_id.clone(),
                output: FunctionCallOutput::Text(output),
                id: None,
                status: None,
            });
            lines.push(RecordLine {
                item: InputItem::Item(Item::FunctionCall(call)),
                meta: LineMetadata { timestamp: Some(now_stamp()) },
            });
        }
        for output in outputs {
            lines.push(RecordLine {
                item: InputItem::Item(Item::FunctionCallOutput(output)),
                meta: LineMetadata { timestamp: Some(now_stamp()) },
            });
        }
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
        self.lines().len()
    }

    /// The stored items and their metadata from `start` on, oldest first.
    ///
    /// A cursor read: only the tail past what a session has already flushed
    /// is cloned, each item paired with the metadata its session line carries.
    /// A `start` past the end yields nothing.
    #[inline]
    #[must_use]
    pub(crate) fn recorded_after(&self, start: usize) -> Vec<RecordLine> {
        self.lines().get(start..).map_or_else(Vec::new, ToOwned::to_owned)
    }

    /// The input items for the next request: the stored record with the
    /// reminder, when one is set, appended after it.
    #[inline]
    #[must_use]
    pub(crate) fn request_items(&self) -> Vec<InputItem> {
        let mut items: Vec<_> = self.lines().iter().map(|line| line.item.clone()).collect();
        items.extend(self.reminder.clone());
        items
    }

    /// The progress stream that renders this record, in live order, for replay.
    ///
    /// Tool exchanges replay headers only, with coloring skipped for simplicity.
    #[inline]
    pub fn replay(&self) -> Vec<Progress> {
        self.lines()
            .iter()
            .map(|line| &line.item)
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

    impl Transcript {
        /// A transcript over `items`, as a session file restored them: the
        /// lines arrive metadata-free, so a record seeded this way mints
        /// stamps at its next write. Resume restores full lines with
        /// [`Transcript::from_lines`].
        #[inline]
        fn from_items(items: Vec<InputItem>) -> Self {
            Self::from_lines(
                items
                    .into_iter()
                    .map(|item| RecordLine { item, meta: LineMetadata::default() })
                    .collect(),
            )
        }

        /// The stored items from `start` on, oldest first, as tests replay
        /// them; [`Session`](crate::Session) reads the same tail as lines,
        /// metadata attached, in [`Transcript::recorded_after`].
        #[inline]
        #[must_use]
        fn items_after(&self, start: usize) -> Vec<InputItem> {
            self.recorded_after(start)
                .into_iter()
                .map(|line| line.item)
                .collect()
        }
    }

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

    /// A reminder trails the record on every request, once, and doesn't hit record.
    #[test]
    fn a_reminder_trails_the_record_once() {
        let mut transcript = Transcript::new().unwrap();
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
    fn push_answer_records_only_real_answers() {
        let transcript = Transcript::new().unwrap();
        transcript.push_user("hello".to_string()).unwrap();

        transcript.push_answer("").unwrap();
        assert_eq!(
            transcript.request_items().len(),
            2,
            "an empty answer records nothing"
        );

        transcript.push_answer("hi").unwrap();
        assert_eq!(transcript.request_items().len(), 3);
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
        // A record restored from an older tart's session file may open with a
        // second system item; the whole leading block survives a clear.
        let transcript = Transcript::from_items(vec![
            input_message(Role::System, SYSTEM.to_string()).unwrap(),
            input_message(Role::System, "be terse".to_string()).unwrap(),
            input_message(Role::User, "hello".to_string()).unwrap(),
        ]);
        transcript.push_tool_round(vec![(bash_call(), "one\n".to_string())]);
        transcript.push_assistant("hi".to_string()).unwrap();

        transcript.clear();

        let items = serde_json::to_value(transcript.request_items()).unwrap();
        assert_eq!(items.as_array().unwrap().len(), 2);
        assert_eq!(items[0]["content"], SYSTEM);
        assert_eq!(items[1]["content"], "be terse");

        // Without a second item, only the prompt survives.
        let plain = Transcript::new().unwrap();
        plain.push_user("hello".to_string()).unwrap();
        plain.clear();
        let items = serde_json::to_value(plain.request_items()).unwrap();
        assert_eq!(items.as_array().unwrap().len(), 1);
        assert_eq!(items[0]["content"], SYSTEM);
    }

    /// A rewind cuts the record back to the turn it names, response and all.
    #[test]
    fn rewind_keeps_the_prefix_before_the_cut() {
        let transcript = Transcript::new().unwrap();
        transcript.push_user("one".to_string()).unwrap();
        transcript.push_assistant("1".to_string()).unwrap();
        transcript.push_user("two".to_string()).unwrap();
        transcript.push_reasoning(reasoning_item());
        transcript.push_tool_round(vec![(bash_call(), "one\n".to_string())]);
        transcript.push_assistant("2".to_string()).unwrap();

        transcript.rewind(transcript.user_turns()[1].0);

        let items = serde_json::to_value(transcript.request_items()).unwrap();
        let items = items.as_array().unwrap();
        assert_eq!(items.len(), 3, "system, then the first turn only");
        assert_eq!(items[0]["role"], "system");
        assert_eq!(items[1]["role"], "user");
        assert_eq!(items[1]["content"], "one");
        assert_eq!(items[2]["role"], "assistant");
        assert_eq!(items[2]["content"], "1");
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
        let transcript = Transcript::new().unwrap();
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
