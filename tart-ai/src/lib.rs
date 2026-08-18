pub mod openai;

use std::ops::AddAssign;

use serde::Serialize;

use openai::Message;

/// Linear, append-only conversation transcript
#[derive(Default)]
pub struct ContextHistory {
    messages: Vec<Message>,
    /// Token usage accumulated across recorded turns.
    usage: Usage,
}

impl ContextHistory {
    /// Push a message into the history, taking ownership of it.
    pub fn append_message(&mut self, msg: Message) {
        self.messages.push(msg);
    }

    /// The transcript so far.
    pub fn as_slice(&self) -> &[Message] {
        &self.messages
    }

    /// Token usage accumulated across recorded turns.
    pub fn usage(&self) -> Usage {
        self.usage
    }

    /// Record a turn's token usage into the session total.
    pub fn record_usage(&mut self, usage: Usage) {
        self.usage += usage;
    }
}

impl From<Message> for ContextHistory {
    fn from(message: Message) -> Self {
        Self {
            messages: vec![message],
            usage: Usage::default(),
        }
    }
}

/// How hard a model reasons before answering.
///
/// Variants ascend in effort, with `None` disabling thinking entirely.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ReasoningEffort {
    None,
    Low,
    Medium,
    High,
    Max,
}

/// Token accounting for one completion.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Usage {
    /// Tokens in the prompt.
    pub prompt_tokens: u64,
    /// Tokens generated for the answer.
    pub completion_tokens: u64,
    /// Tokens generated for chain-of-thought reasoning.
    pub reasoning_tokens: u64,
    /// Input tokens served from a provider cache.
    pub cached_tokens: u64,
    /// Input tokens newly written to a provider cache.
    pub cache_write_tokens: u64,
}

impl AddAssign for Usage {
    fn add_assign(&mut self, rhs: Self) {
        self.prompt_tokens += rhs.prompt_tokens;
        self.completion_tokens += rhs.completion_tokens;
        self.reasoning_tokens += rhs.reasoning_tokens;
        self.cached_tokens += rhs.cached_tokens;
        self.cache_write_tokens += rhs.cache_write_tokens;
    }
}
