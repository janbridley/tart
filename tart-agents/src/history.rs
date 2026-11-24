use std::collections::HashSet;
use std::sync::{Arc, Mutex, MutexGuard};

use serde::{Deserialize, Serialize};
use time::OffsetDateTime;

use crate::Progress;
use crate::backends::{
    EasyInputContent, FunctionToolCall, InputItem, Item, MessageItem, OutputMessageContent,
    ReasoningItemContent, Role, call_output, message,
};

/// The system prompt for *tart*, included in every conversation.
const SYSTEM: &str = include_str!("data/SYSTEM.md");

/// The moment as `YYYY-MM-DD HH:MM:SS` UTC, fixed-width and lexically
/// sortable: the stamp a line records itself under.
#[inline]
pub(crate) fn stamp_utc_at(moment: OffsetDateTime) -> String {
    let (y, m, d) = (moment.year(), u8::from(moment.month()), moment.day());
    let (hr, min, sec) = (moment.hour(), moment.minute(), moment.second());
    format!("{y:04}-{m:02}-{d:02} {hr:02}:{min:02}:{sec:02}")
}

/// One line of the record: an item and the stamp its session file wrote beside
/// it. The item's own fields flatten in beside the stamp, so the line is one
/// JSON object: `{"timestamp":"…","type":"reasoning","id":"rs_…",…}`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub(crate) struct RecordLine {
    /// When the line was recorded, `YYYY-MM-DD HH:MM:SS` UTC
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "stamp"
    )]
    pub(crate) timestamp: Option<String>, // Stamped when it enters record
    /// The item itself, as the record and a request send it.
    #[serde(flatten)]
    pub(crate) item: InputItem,
}

/// The stamp as written, or absent when the sibling holds anything but text.
fn stamp<'de, D: serde::Deserializer<'de>>(deserializer: D) -> Result<Option<String>, D::Error> {
    Ok(
        Option::<serde_json::Value>::deserialize(deserializer)?.and_then(|value| match value {
            serde_json::Value::String(stamp) => Some(stamp),
            _ => None,
        }),
    )
}

impl RecordLine {
    /// One line as the record appends it, stamped with the moment it is
    /// recorded.
    #[inline]
    fn fresh(item: InputItem) -> Self {
        Self {
            item,
            timestamp: Some(stamp_utc_at(OffsetDateTime::now_utc())),
        }
    }
}

/// An append-only conversation record for one tart session.
///
/// Clones share one record, so the agent loop writes reasoning, tool exchanges, and the
/// final answer to the callers transcript. After accumulation, the transcript is passed
/// back into the model and the conversation continues.
///
/// A turn's items replay in the order they were recorded, the system prompt leading:
///
/// ```
/// use tart_agents::Progress;
/// use tart_agents::Transcript;
/// # fn main() -> anyhow::Result<()> {
/// let transcript = Transcript::new()?;
/// transcript.push_user("hello".to_string())?;
/// transcript.push_assistant("hi there".to_string())?;
///
/// assert!(matches!(
///     transcript.replay().as_slice(),
///     [Progress::User(text), Progress::Answer(answer)]
///         if text == "hello" && answer == "hi there"
/// ));
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug)]
pub struct Transcript {
    /// The lines, oldest first, opening with the system prompt.
    lines: Arc<Mutex<Vec<RecordLine>>>,
    /// The repository's AGENTS.md block, sent ahead of the conversation.
    ///
    /// see [`Transcript::set_preamble`]
    preamble: Option<InputItem>,
    /// A mode reminder appended to the end of the input at request time.
    ///
    /// This is designed to mirror Codex's instruction handling, as we can't
    /// guarantee that arbitrary endpoints support a dedicated instructions
    /// channel. Trailing the record keeps the cached prefix up to the turn on which
    /// plan mode was triggered intact.
    reminder: Option<InputItem>,
}

/// Whether `item` is a plain message carrying `role`.
pub(crate) fn is_message(item: &InputItem, role: Role) -> bool {
    matches!(item, InputItem::EasyMessage(message) if message.role == role)
}

impl Transcript {
    /// The lines under their lock, poison recovered as [`crate::locked`] does.
    fn locked(&self) -> MutexGuard<'_, Vec<RecordLine>> {
        crate::locked(&self.lines)
    }

    /// A transcript opening with the tart system prompt.
    #[inline]
    pub fn new() -> anyhow::Result<Self> {
        Self::new_with(SYSTEM)
    }

    /// A transcript opening with `system` instead of the tart prompt.
    #[inline]
    pub fn new_with(system: &str) -> anyhow::Result<Self> {
        Ok(Self {
            lines: Arc::new(Mutex::new(vec![RecordLine::fresh(message(
                Role::System,
                system.to_string(),
            ))])),
            preamble: None,
            reminder: None,
        })
    }

    /// A transcript over `lines`, as a session file restored them.
    #[inline]
    pub(crate) fn from_lines(lines: Vec<RecordLine>) -> Self {
        Self {
            lines: Arc::new(Mutex::new(lines)),
            preamble: None,
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
        let mut lines = self.locked();
        let systems = lines
            .iter()
            .take_while(|line| is_message(&line.item, Role::System))
            .count();
        lines.truncate(start.max(systems));
    }

    /// The user messages in the record, oldest first, each with the index its turn begins at.
    #[inline]
    #[must_use]
    pub fn user_turns(&self) -> Vec<(usize, String)> {
        self.locked()
            .iter()
            .enumerate()
            .filter_map(|(index, line)| match &line.item {
                InputItem::EasyMessage(message) => match (&message.role, &message.content) {
                    (Role::User, EasyInputContent::Text(text)) => Some((index, text.clone())),
                    // Prompts, answers, and content lists the harness never
                    // rewinds to can be skipped
                    _ => None,
                },
                _ => None,
            })
            .collect()
    }

    /// Record the user's turn.
    #[inline]
    pub fn push_user(&self, text: String) -> anyhow::Result<()> {
        self.push(message(Role::User, text));
        Ok(())
    }

    /// Record the assistant's final answer for the current turn.
    #[inline]
    pub fn push_assistant(&self, text: String) -> anyhow::Result<()> {
        self.push(message(Role::Assistant, text));
        Ok(())
    }

    /// Record the answer a round streamed, when it streamed one at all.
    pub(crate) fn push_answer(&self, answer: &str) {
        if !answer.is_empty() {
            self.push(message(Role::Assistant, answer.to_string()));
        }
    }

    /// Append one item, stamped with the moment of its recording.
    fn push(&self, item: InputItem) {
        self.locked().push(RecordLine::fresh(item));
    }

    /// Record one finished round under one lock: the provider's output items
    /// exactly as it minted them, in stream order — reasoning with its
    /// ciphertext, calls with their ids, assistant messages with their
    /// structure; then the streamed `answer`, which stands in for the message
    /// item only when the stream delivered none; then each call's output in
    /// call order.
    ///
    /// Provider items are never rewritten, which keeps the model's reasoning
    /// intact between rounds and the growing record a stable prefix. The one
    /// thing left out is a call that did not run (e.g. when a cancel skips the
    /// rest of its round): paired with no output, the provider refuses the record.
    pub(crate) fn push_round(
        &self,
        round: Vec<InputItem>,
        answer: &str,
        outputs: Vec<(FunctionToolCall, String)>,
    ) {
        let messaged = round
            .iter()
            .any(|item| matches!(item, InputItem::Item(Item::Message(_))));
        let ran: HashSet<&str> = outputs.iter().map(|(call, _)| call.call_id.as_str()).collect();
        let mut lines = self.locked();
        lines.extend(round.into_iter().filter(|item| {
            !matches!(item, InputItem::Item(Item::FunctionCall(call)) if !ran.contains(call.call_id.as_str()))
        }).map(RecordLine::fresh));
        if !messaged && !answer.is_empty() {
            lines.push(RecordLine::fresh(message(Role::Assistant, answer.to_string())));
        }
        lines.extend(
            outputs
                .into_iter()
                .map(|(call, output)| RecordLine::fresh(call_output(&call, output))),
        );
    }

    /// Send `text` as a user message behind the system prompt on every request or clear
    #[inline]
    pub fn set_preamble(&mut self, text: Option<&str>) -> anyhow::Result<()> {
        self.preamble = text.map(|text| message(Role::User, text.to_string()));
        Ok(())
    }

    /// Append `text` to the end of the input on every subsequent request, or clear with `None`.
    #[inline]
    pub fn set_reminder(&mut self, text: Option<&str>) -> anyhow::Result<()> {
        self.reminder = text.map(|text| message(Role::System, text.to_string()));
        Ok(())
    }

    /// The stored lines, oldest first: the session file's contents.
    #[inline]
    #[must_use]
    pub(crate) fn lines(&self) -> Vec<RecordLine> {
        self.locked().clone()
    }

    /// The input items for the next request: the stored record, the AGENTS.md preamble
    /// spliced in behind the leading system messages, and the reminder, if set.
    #[inline]
    #[must_use]
    pub fn request_items(&self) -> Vec<InputItem> {
        let mut items: Vec<_> = self.locked().iter().map(|line| line.item.clone()).collect();
        if let Some(preamble) = &self.preamble {
            let systems = items
                .iter()
                .take_while(|item| is_message(item, Role::System))
                .count();
            items.insert(systems, preamble.clone());
        }
        items.extend(self.reminder.clone());
        items
    }

    /// The progress stream that renders this record, in live order, for replay.
    ///
    /// Tool exchanges replay headers only, with coloring skipped for simplicity.
    #[inline]
    pub fn replay(&self) -> Vec<Progress> {
        self.locked()
            .iter()
            .flat_map(|line| replay_events(&line.item))
            .collect()
    }
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
        // The provider's own words: an assistant message's text, or the
        // reasoning it shows, empty when it kept its thinking to itself.
        InputItem::Item(Item::Message(MessageItem::Output(message))) => {
            let text: String = message
                .content
                .iter()
                .filter_map(|part| match part {
                    OutputMessageContent::OutputText(text) => Some(text.text.as_str()),
                    OutputMessageContent::Refusal(_) => None,
                })
                .collect();
            (!text.is_empty())
                .then_some(Progress::Answer(text))
                .into_iter()
                .collect()
        }
        InputItem::Item(Item::Reasoning(reasoning)) => {
            let text: String = reasoning
                .content
                .iter()
                .flatten()
                .map(|ReasoningItemContent::ReasoningText(part)| part.text.as_str())
                .collect();
            (!text.is_empty())
                .then_some(Progress::Thinking(text))
                .into_iter()
                .collect()
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

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, reason = "test assertions")]

    use super::*;

    impl Transcript {
        /// A transcript over `items`, as a session file restored them: the
        /// lines arrive unstamped, and stay so. Resume restores full lines with
        /// [`Transcript::from_lines`].
        #[inline]
        fn from_items(items: Vec<InputItem>) -> Self {
            Self::from_lines(
                items
                    .into_iter()
                    .map(|item| RecordLine { item, timestamp: None })
                    .collect(),
            )
        }

        /// The stored items from `start` on, oldest first, as tests read them;
        /// [`Session`](crate::Session) reads the same tail as lines, stamps
        /// attached, in [`Transcript::lines`].
        #[inline]
        #[must_use]
        fn items_after(&self, start: usize) -> Vec<InputItem> {
            self.lines().get(start..).map_or_else(Vec::new, |lines| {
                lines.iter().map(|line| line.item.clone()).collect()
            })
        }
    }

    /// A finished `bash` call, as the provider mints one.
    fn bash_call() -> FunctionToolCall {
        FunctionToolCall {
            namespace: None,
            name: "bash".to_string(),
            arguments: r#"{"command":"ls"}"#.to_string(),
            call_id: "call_0".to_string(),
            id: Some("item_0".to_string()),
            status: None,
            caller: None,
            r#async: None,
        }
    }

    fn call(call: FunctionToolCall) -> InputItem {
        InputItem::Item(Item::FunctionCall(call))
    }

    /// A reasoning item as the provider mints one: visible text and its
    /// ciphertext, under its minted id.
    fn reasoning(id: &str) -> InputItem {
        serde_json::from_value(serde_json::json!({
            "type": "reasoning",
            "id": id,"summary": [],
            "content": [{ "type": "reasoning_text", "text": "thinking" }],
            "encrypted_content": "gAAAAABciphertext",
        }))
        .unwrap()
    }

    /// An assistant message as the provider mints one.
    fn answer() -> InputItem {
        serde_json::from_value(serde_json::json!({
            "type": "message",
            "id": "msg_0",
            "role": "assistant",
            "status": "completed",
            "phase": "final_answer",
            "content": [
                {"type": "output_text", "text": "All ", "annotations": []},
                {"type": "output_text", "text": "done.", "annotations": []},
            ],
        }))
        .unwrap()
    }

    /// Record one round as the loop does: the provider's items in stream
    /// order, then the outputs of its calls.
    fn round(
        transcript: &Transcript,
        items: Vec<InputItem>,
        outputs: Vec<(FunctionToolCall, String)>,
    ) {
        transcript.push_round(items, "", outputs);
    }

    #[test]
    fn a_round_cut_short_records_only_the_calls_that_ran() {
        let transcript = Transcript::new().unwrap();
        let mut skipped = bash_call();
        skipped.call_id = "call_1".into();
        skipped.id = Some("item_1".into());
        transcript.push_round(
            vec![call(bash_call()), call(skipped)],
            "checking",
            vec![(bash_call(), "one\n".to_string())],
        );
        let items = serde_json::to_value(transcript.request_items()).unwrap();
        assert_eq!(items[1]["call_id"], "call_0");
        assert_eq!(items[2]["role"], "assistant");
        assert_eq!(items[3]["type"], "function_call_output");
        assert_eq!(items.as_array().unwrap().len(), 4, "{items}");
    }

    #[test]
    fn a_round_records_items_then_the_answer_then_outputs() {
        let transcript = Transcript::new().unwrap();
        transcript.push_user("run it".to_string()).unwrap();
        transcript.push_round(
            vec![reasoning("rs_0"), reasoning("rs_1"), call(bash_call())],
            "running ls\n",
            vec![(bash_call(), "one\n".to_string())],
        );
        transcript.push_round(vec![reasoning("rs_2"), answer()], "All done.", Vec::new());
        transcript.push_round(Vec::new(), "", Vec::new());

        assert_eq!(
            transcript.items_after(0),
            [
                message(Role::System, SYSTEM.to_string()),
                message(Role::User, "run it".to_string()),
                reasoning("rs_0"),
                reasoning("rs_1"),
                call(bash_call()),
                message(Role::Assistant, "running ls\n".to_string()),
                call_output(&bash_call(), "one\n".to_string()),
                reasoning("rs_2"),
                answer(),
            ]
        );
    }

    #[test]
    fn preamble_rides_behind_the_system_prompt_unrecorded() {
        let mut transcript = Transcript::new_with("system").unwrap();
        transcript.set_preamble(Some("# AGENTS.md\n\ndoc")).unwrap();
        transcript.push_user("hello".to_string()).unwrap();

        // Sent on every request, right behind the system prompt.
        let items = serde_json::to_value(transcript.request_items()).unwrap();
        assert_eq!(items[0]["role"], "system");
        assert_eq!(items[1]["role"], "user");
        assert_eq!(items[1]["content"], "# AGENTS.md\n\ndoc");
        assert_eq!(items[2]["content"], "hello");

        // But never recorded: replay shows only the user's own words.
        let users: Vec<String> = transcript
            .replay()
            .into_iter()
            .filter_map(|event| match event {
                Progress::User(text) => Some(text),
                _ => None,
            })
            .collect();
        assert_eq!(users, ["hello"]);

        // Clearing it stops the send.
        transcript.set_preamble(None).unwrap();
        assert_eq!(transcript.request_items().len(), 2);
    }

    #[test]
    fn new_with_opens_with_the_given_prompt_and_survives_clear() {
        let transcript = Transcript::new_with("chat preamble").unwrap();
        let items = serde_json::to_value(transcript.request_items()).unwrap();
        assert_eq!(items[0]["role"], "system");
        assert_eq!(items[0]["content"], "chat preamble");

        // A clear keeps the leading system block, whatever prompt it holds.
        transcript.push_user("hello".to_string()).unwrap();
        transcript.clear();
        let items = serde_json::to_value(transcript.request_items()).unwrap();
        assert_eq!(items.as_array().map(Vec::len), Some(1));
        assert_eq!(items[0]["content"], "chat preamble");

        assert_ne!(SYSTEM, "chat preamble");
        assert_eq!(
            serde_json::to_value(Transcript::new().unwrap().request_items()).unwrap()[0]["content"],
            SYSTEM
        );
    }

    /// A reminder trails the record on every request, once, and never reaches
    /// the record; clearing it, as the approval handover does, leaves the
    /// cached prefix exactly as the planning request sent it.
    #[test]
    fn a_reminder_trails_the_record_once_and_leaves_no_residue() {
        let mut transcript = Transcript::new().unwrap();
        transcript.push_user("look at the auth flow".to_string()).unwrap();

        // Without a reminder the request is exactly the stored record.
        assert_eq!(transcript.request_items(), transcript.items_after(0));

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
        let sent = serde_json::to_value(transcript.items_after(0)).unwrap();
        for (item, kept) in request.iter().zip(sent.as_array().unwrap()) {
            assert_eq!(item, kept, "the record leads the request unchanged");
        }

        // The plan lands, the mode leaves, the approval turn is recorded: the
        // implementing request ends in the approval, the cached prefix intact.
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
        let request = request.as_array().unwrap();
        assert_eq!(request.last().unwrap()["role"], "user");
        for (item, cached) in request.iter().zip(sent.as_array().unwrap()) {
            assert_eq!(item, cached, "the cached prefix survives the handover");
        }
    }

    /// A recorded round replays in spec order — message, reasoning, call,
    /// output, message — and every shape the harness records is one JSON line
    /// that reparses to itself.
    #[test]
    fn a_recorded_turn_keeps_spec_order_and_round_trips_through_jsonl() {
        let transcript = Transcript::new().unwrap();
        transcript.push_user("run it".to_string()).unwrap();
        round(
            &transcript,
            vec![reasoning("rs_0"), call(bash_call())],
            vec![(bash_call(), "one\n".to_string())],
        );
        transcript.push_assistant("done".to_string()).unwrap();

        let items = serde_json::to_value(transcript.request_items()).unwrap();
        assert_eq!(items[1]["role"], "user");
        assert_eq!(items[2]["type"], "reasoning");
        assert_eq!(items[2]["id"], "rs_0");
        assert_eq!(items[2]["content"][0]["text"], "thinking");
        assert_eq!(items[2]["encrypted_content"], "gAAAAABciphertext");
        assert_eq!(items[3]["type"], "function_call");
        assert_eq!(items[4]["type"], "function_call_output");
        assert_eq!(items[3]["id"], "item_0", "the call keeps its minted id");
        assert_eq!(items[3]["call_id"], items[4]["call_id"]);
        assert_eq!(items[5]["role"], "assistant");
        assert_eq!(items[5]["content"], "done");

        for item in transcript.request_items() {
            let line = serde_json::to_string(&item).unwrap();
            assert!(!line.contains('\n'), "{line}");
            assert_eq!(serde_json::from_str::<InputItem>(&line).unwrap(), item);
        }
    }

    /// A round's calls precede their outputs, each output paired to its call:
    /// interleaving would split the round into several assistant messages on
    /// the provider side. What the loop records lands in the caller's
    /// transcript, clones sharing one record.
    #[test]
    fn tool_rounds_replay_calls_grouped_before_outputs() {
        let transcript = Transcript::new().unwrap();
        let worker = transcript.clone();
        worker.push_user("hello".to_string()).unwrap();
        let mut second = bash_call();
        second.call_id = "call_1".to_string();
        round(
            &worker,
            vec![call(bash_call()), call(second.clone())],
            vec![(bash_call(), "one\n".to_string()), (second, "two\n".to_string())],
        );

        let items = serde_json::to_value(transcript.request_items()).unwrap();
        assert_eq!(items[1]["role"], "user");
        assert_eq!(items[2]["type"], "function_call");
        assert_eq!(items[2]["call_id"], "call_0");
        assert_eq!(items[3]["type"], "function_call");
        assert_eq!(items[3]["call_id"], "call_1");
        assert_eq!(items[4]["type"], "function_call_output");
        assert_eq!(items[4]["output"], "one\n");
        assert_eq!(items[5]["type"], "function_call_output");
        assert_eq!(items[5]["output"], "two\n");
        assert_eq!(items.as_array().unwrap().len(), 6);
    }

    #[test]
    fn clear_keeps_only_the_leading_system_items() {
        // A record restored from an older tart's session file may open with a
        // second system item; the whole leading block survives a clear.
        let transcript = Transcript::from_items(vec![
            message(Role::System, SYSTEM.to_string()),
            message(Role::System, "be terse".to_string()),
            message(Role::User, "hello".to_string()),
        ]);
        transcript.push_round(Vec::new(), "", vec![(bash_call(), "one\n".to_string())]);
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
        assert_eq!(plain.request_items().len(), 1);
    }

    /// A rewind cuts the record back to the turn it names, response and all.
    #[test]
    fn rewind_keeps_the_prefix_before_the_cut() {
        let transcript = Transcript::new().unwrap();
        transcript.push_user("one".to_string()).unwrap();
        transcript.push_assistant("1".to_string()).unwrap();
        transcript.push_user("two".to_string()).unwrap();
        round(
            &transcript,
            vec![reasoning("rs_0"), call(bash_call())],
            vec![(bash_call(), "one\n".to_string())],
        );
        transcript.push_assistant("2".to_string()).unwrap();

        transcript.rewind(transcript.user_turns()[1].0);

        let items = serde_json::to_value(transcript.request_items()).unwrap();
        let items = items.as_array().unwrap();
        assert_eq!(items.len(), 3, "system, then the first turn only");
        assert_eq!(items[1]["role"], "user");
        assert_eq!(items[1]["content"], "one");
        assert_eq!(items[2]["role"], "assistant");
        assert_eq!(items[2]["content"], "1");
    }

    /// The cursor read yields exactly the unseen tail, never the flushed prefix.
    #[test]
    fn items_after_returns_the_unseen_tail_only() {
        let transcript = Transcript::new().unwrap();
        transcript.push_user("hello".to_string()).unwrap();
        transcript.push_assistant("hi".to_string()).unwrap();

        // From zero the whole record reads back, matching a full request.
        assert_eq!(transcript.items_after(0), transcript.request_items());

        // Past the system prompt: only the turn items follow.
        let tail = transcript.items_after(1);
        assert_eq!(tail.len(), 2);
        assert_eq!(serde_json::to_value(&tail[0]).unwrap()["role"], "user");

        // Reading to the end, and past it, yields nothing without panicking:
        // a cleared record can end before a session's flushed prefix.
        assert_eq!(transcript.items_after(transcript.lines().len()), []);
        assert_eq!(transcript.items_after(transcript.lines().len() + 3), []);
    }

    #[test]
    fn replay_renders_words_and_tool_headers_only() {
        use Progress::{Answer, Thinking, ToolOutput, ToolStart, User};

        // System items replay as nothing.
        let transcript = Transcript::new().unwrap();
        assert!(transcript.replay().is_empty());

        transcript.push_user("run it".to_string()).unwrap();
        let mut second = bash_call();
        second.call_id = "call_1".to_string();
        round(
            &transcript,
            vec![reasoning("rs_0"), call(bash_call()), call(second.clone())],
            vec![
                (bash_call(), "one\n".to_string()),
                (second, "[exit 2]\ntwo\n".to_string()),
            ],
        );
        round(&transcript, vec![answer()], vec![]);

        let events = transcript.replay();

        // Each call replays as its header, finished empty: no recorded output,
        // no exit derived from its framing. The provider's message replays as
        // the answer, its parts joined.
        assert!(matches!(
            events.as_slice(),
            [
                User(text),
                Thinking(thinking),
                ToolStart { id, name, arguments },
                ToolOutput { output, exit: Some(0), .. },
                ToolStart { id: second_id, .. },
                ToolOutput { output: second_output, exit: Some(0), .. },
                Answer(answer),
            ] if text == "run it"
                && thinking == "thinking"
                && id == "call_0"
                && name == "bash"
                && arguments == r#"{"command":"ls"}"#
                && output.is_empty()
                && second_id == "call_1"
                && second_output.is_empty()
                && answer == "All done."
        ));
    }

    /// A line is the item's own fields with the stamp flattened in beside
    /// them: it parses back to itself, an unstamped line writes no stamp, and
    /// a stamp of another shape, or an unknown sibling, reads as no stamp.
    #[test]
    fn lines_flatten_the_stamp_beside_the_item() {
        // A line as a hand-written file might hold it: no stamp at all.
        let line = RecordLine {
            timestamp: None,
            item: message(Role::System, "system".to_string()),
        };
        let stamped = RecordLine {
            timestamp: Some("2026-11-24 15:19:34".to_string()),
            ..line.clone()
        };
        let text = serde_json::to_string(&stamped).unwrap();
        assert_eq!(
            text,
            r#"{"timestamp":"2026-11-24 15:19:34","type":"message","role":"system","content":"system"}"#
        );
        assert_eq!(serde_json::from_str::<RecordLine>(&text).unwrap(), stamped);
        let unstamped = serde_json::to_string(&line).unwrap();
        assert!(!unstamped.contains("timestamp"), "{unstamped}");

        let mangled = r#"{"content":"hi","role":"assistant","timestamp":123,"usage":"lots","type":"message"}"#;
        let parsed = serde_json::from_str::<RecordLine>(mangled).unwrap();
        assert_eq!(parsed.timestamp, None);
        assert!(matches!(
            &parsed.item,
            InputItem::EasyMessage(message) if message.content == EasyInputContent::Text("hi".into())
        ));
        // A null stamp is absent too, not damage: the line still parses.
        let nulled = mangled.replace("123", "null");
        assert_eq!(serde_json::from_str::<RecordLine>(&nulled).unwrap(), parsed);
    }
}
