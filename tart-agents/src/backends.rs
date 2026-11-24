//! Requests and events shared by the agent loop and the Responses adapter.

pub mod openai_responses;

use crate::usage::TokenUsage;

/// The record and tool types exchanged with the loop.
pub(crate) use self::openai_responses::{
    EasyInputContent, FunctionToolCall, InputItem, Item, MessageItem, OutputMessageContent,
    ReasoningItemContent, Role, Tool, call_output, message, tool,
};

/// How hard the model reasons: the provider's own vocabulary, spelled as the
/// config file and the wire spell it.
pub use async_openai::types::responses::ReasoningEffort;

/// The events of one round's stream, in arrival order.
pub(crate) enum Event {
    /// A delta of the answer text.
    Answer(String),
    /// A delta of the model's visible reasoning.
    Thinking(String),
    /// A tool call started; its arguments are still streaming.
    CallStarted,
    /// A finished output item as the next request replays it: reasoning with its
    /// ciphertext, tool calls with their ids, and assistant messages in stream
    /// order.
    ItemDone(InputItem),
    /// The round completed, with its usage when the provider reported one.
    Completed(Option<TokenUsage>),
    /// The round failed; the reason text.
    Failed(String),
    /// The round ended truncated; the reason text.
    Incomplete(String),
}
