use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use async_openai::types::responses::{
    EasyInputMessageArgs, FunctionCallOutput, FunctionCallOutputItemParam, FunctionToolCall,
    InputItem, Item, ReasoningItem, Role,
};

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
        })
    }

    /// A transcript opening with the tart system prompt, followed by the
    /// agent's `instructions` as a second system message.
    #[inline]
    pub fn with_instructions(instructions: String) -> anyhow::Result<Self> {
        let transcript = Self::new()?;
        transcript
            .items()
            .push(input_message(Role::System, instructions)?);
        Ok(transcript)
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

    /// Reset such that the last turn (everything since the user message) disappears.
    #[inline]
    pub fn drop_last_turn(&self) {
        let mut items = self.items();
        // TODO: can we somehow combine this and clear?
        let at = items
            .iter()
            .rposition(|item| matches!(item, InputItem::EasyMessage(m) if m.role == Role::User))
            .unwrap_or(0);
        items.truncate(at);
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
        for output in outputs {
            items.push(InputItem::Item(Item::FunctionCallOutput(output)));
        }
    }

    /// The input items for the next request, cloned from the record.
    #[inline]
    #[must_use]
    pub(crate) fn request_items(&self) -> Vec<InputItem> {
        self.items().clone()
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

        // message, reasoning, call, output, message — each call paired with
        // its output by `call_id`.
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
    fn drop_last_turn_restores_the_pre_turn_state() {
        let transcript = Transcript::new().unwrap();
        transcript.push_user("first".to_string()).unwrap();
        transcript.push_assistant("kept".to_string()).unwrap();

        // A second turn is unwound back to the end of the first
        transcript.push_user("cancel me".to_string()).unwrap();
        transcript.push_reasoning(reasoning_item());
        transcript.push_tool_round(vec![(bash_call(), "one\n".to_string())]);
        transcript.push_assistant("partial".to_string()).unwrap();
        transcript.drop_last_turn();

        let items = serde_json::to_value(transcript.request_items()).unwrap();
        assert_eq!(items.as_array().unwrap().len(), 3);
        assert_eq!(items[1]["content"], "first");
        assert_eq!(items[2]["content"], "kept");

        // The conversation continues cleanly from the rewound state.
        transcript.push_user("next".to_string()).unwrap();
        let items = serde_json::to_value(transcript.request_items()).unwrap();
        assert_eq!(items[3]["content"], "next");
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
}
