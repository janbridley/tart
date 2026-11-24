//! Requests and events shared by the agent loop and the Responses adapter.

pub mod openai_responses;

use serde::{Deserialize, Serialize};

use crate::usage::TokenUsage;

/// The record and tool types exchanged with the loop.
pub(crate) use self::openai_responses::{
    EasyInputContent, FunctionToolCall, InputItem, Item, ReasoningItem, ReasoningItemContent, Role,
    Tool, call_output, message, tool,
};

/// How hard the model reasons.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
#[allow(
    clippy::exhaustive_enums,
    reason = "tart owns this closed set: a new variant should break every match on it"
)]
#[allow(
    missing_docs,
    reason = "levels are self-explanatory and their meaning varies with provider"
)]
pub enum ReasoningEffort {
    None,
    Minimal,
    Low,
    #[default]
    Medium,
    High,
    Xhigh,
    Max,
}

/// The events of one round's stream, in arrival order.
pub(crate) enum Event {
    /// A delta of the answer text.
    Answer(String),
    /// A delta of the model's visible reasoning.
    Thinking(String),
    /// A tool call started; its arguments are still streaming.
    CallStarted,
    /// A finished reasoning item, replayed on later rounds.
    Reasoning(ReasoningItem),
    /// A finished tool call.
    Call(FunctionToolCall),
    /// The round completed, with its usage when the provider reported one.
    Completed(Option<TokenUsage>),
    /// The round failed; the reason text.
    Failed(String),
    /// The round ended truncated; the reason text.
    Incomplete(String),
}
