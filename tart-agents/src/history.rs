use async_openai::types::responses::{
    EasyInputMessageArgs, FunctionCallOutput, FunctionCallOutputItemParam, FunctionToolCall,
    InputItem, Item, ReasoningItem, Role,
};

/// The system prompt for *tart*, included in every conversation.
const SYSTEM: &str = include_str!("data/SYSTEM.md");

/// An append-only conversation record for one tart session.
///
/// Tool exchanges are recorded by the agent loop for the turn that made them
/// and do not persist into later requests.
#[derive(Clone, Debug)]
pub struct Transcript {
    /// Every item, oldest first, starting with the system prompt.
    items: Vec<InputItem>,
}

impl Transcript {
    /// A transcript opening with the tart system prompt.
    #[inline]
    pub fn new() -> anyhow::Result<Self> {
        Ok(Self {
            items: vec![input_message(Role::System, SYSTEM.to_string())?],
        })
    }

    /// Record the user's turn.
    #[inline]
    pub fn push_user(&mut self, text: String) -> anyhow::Result<()> {
        self.items.push(input_message(Role::User, text)?);
        Ok(())
    }

    /// Record the assistant's final answer for the current turn.
    #[inline]
    pub fn push_assistant(&mut self, text: String) -> anyhow::Result<()> {
        self.items.push(input_message(Role::Assistant, text)?);
        Ok(())
    }

    /// Record the reasoning that preceded a round's tool calls.
    ///
    /// This is critical for `DeepSeek`'s thinking mode, which breaks on concurrent tool
    /// calls without it.
    pub(crate) fn push_reasoning(&mut self, item: ReasoningItem) {
        self.items.push(InputItem::Item(item.into()));
    }

    /// Record one round of tool exchanges.
    pub(crate) fn push_tool_round(&mut self, round: Vec<(FunctionToolCall, String)>) {
        let mut outputs = Vec::with_capacity(round.len());
        for (call, output) in round {
            outputs.push(FunctionCallOutputItemParam {
                call_id: call.call_id.clone(),
                output: FunctionCallOutput::Text(output),
                id: None,
                status: None,
            });
            self.items.push(InputItem::Item(Item::FunctionCall(call)));
        }
        for output in outputs {
            self.items.push(InputItem::Item(Item::FunctionCallOutput(output)));
        }
    }

    /// The input items for the next request, cloned from the record.
    #[inline]
    #[must_use]
    pub(crate) fn request_items(&self) -> Vec<InputItem> {
        self.items.clone()
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

    #[test]
    fn transcript_opens_with_the_system_prompt() {
        let items = serde_json::to_value(Transcript::new().unwrap().request_items()).unwrap();

        assert_eq!(items[0]["role"], "system");
        assert_eq!(items[0]["content"], SYSTEM);
    }

    #[test]
    fn pushed_turns_serialize_in_order() {
        let mut transcript = Transcript::new().unwrap();
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
        let mut transcript = Transcript::new().unwrap();
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
    fn reasoning_replays_as_a_reasoning_item() {
        let mut transcript = Transcript::new().unwrap();
        transcript.push_reasoning(ReasoningItem {
            id: Some("rs_0".to_string()),
            summary: Vec::new(),
            content: Some(vec![ReasoningItemContent::ReasoningText(ReasoningTextContent {
                text: "thinking".to_string(),
            })]),
            encrypted_content: None,
            status: None,
        });

        let items = serde_json::to_value(transcript.request_items()).unwrap();

        assert_eq!(items[1]["type"], "reasoning");
        assert_eq!(items[1]["content"][0]["type"], "reasoning_text");
        assert_eq!(items[1]["content"][0]["text"], "thinking");
    }

    #[test]
    fn request_items_is_a_copy() {
        let transcript = Transcript::new().unwrap();
        let mut items = transcript.request_items();
        items.clear();

        assert_eq!(transcript.request_items().len(), 1);
    }
}
