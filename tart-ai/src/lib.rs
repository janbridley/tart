pub mod openai;

use serde::Serialize;

use openai::Message;

/// Linear, append-only conversation transcript
#[derive(Default)]
pub struct ContextHistory {
    messages: Vec<Message>,
}

impl ContextHistory {
    /// Initialize an empty history.
    #[inline]
    pub fn new() -> Self {
        Self {
            messages: Vec::new(),
        }
    }

    /// Push a message into the history, taking ownership of it.
    #[inline]
    pub fn append_message(&mut self, msg: Message) {
        self.messages.push(msg);
    }

    /// The transcript so far.
    #[inline]
    pub fn as_slice(&self) -> &[Message] {
        &self.messages
    }
}

impl From<Message> for ContextHistory {
    #[inline]
    fn from(message: Message) -> Self {
        Self {
            messages: vec![message],
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
