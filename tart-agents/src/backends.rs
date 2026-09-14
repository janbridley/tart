//! The provider seam: the crate speaks only the vocabulary in this module,
//! and a backend translates it to and from its wire protocol.
//! `openai_responses` is the backend tart runs today; an Anthropic or Chat
//! Completions backend would implement the same [`Backend`] contract beside it.

pub mod openai_responses;

use std::future::Future;
use std::pin::Pin;

use futures::Stream;
use serde::{Deserialize, Serialize};

use crate::usage::TokenUsage;

/// The record and tool types every backend exchanges with the loop.
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

/// One round's event stream; `Err` items are transport failures.
pub(crate) type RoundStream = Pin<Box<dyn Stream<Item = anyhow::Result<Event>> + Send>>;

/// A provider that runs rounds of the tool loop.
pub(crate) trait Backend: Clone + Send + Sync + 'static {
    /// Connect one round: the model, the effort, the record, and the tools.
    fn round(
        &self,
        model: &str,
        effort: Option<ReasoningEffort>,
        items: Vec<InputItem>,
        tools: Vec<Tool>,
    ) -> impl Future<Output = anyhow::Result<RoundStream>> + Send;
}
