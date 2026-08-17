//! OpenAI Chat Completions interface.

use std::{borrow::Cow, path::PathBuf};

pub const SYSTEM: &str = include_str!("data/SYSTEM.md");

mod completions;
pub use completions::{ChatCompletions, ChatCompletionsClient};

/// Valid `role` entries for a [`Message`]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Role {
    Assistant,
    System,
    Developer,
    User,
}

/// One unit of data passed from the user to the model or vice versa.
struct Message {
    /// The ,
    pub(crate) role: Role,
    /// The text associated with this message.
    pub(crate) content: String,
}
impl Message {
    /// Initialize a `Role::System` message from a markdown file.
    #[inline]
    fn from_system_prompt_markdown_file(filename: PathBuf) -> anyhow::Result<Self> {
        let is_markdown = filename
            .extension()
            .and_then(|ext| ext.to_str())
            .is_some_and(|ext| ext.eq_ignore_ascii_case("md"));

        if !is_markdown {
            anyhow::bail!("Expected a Markdown file (.md), but received: {filename:?}");
        }

        Ok(Self {
            role: Role::System,
            content: std::fs::read_to_string(&filename)?.to_string(),
        })
    }

    #[inline]
    fn system() -> Self {
        Self {
            role: Role::System,
            content: SYSTEM.to_string(),
        }
    }
}

impl Into<ContextHistory> for Message {
    fn into(self) -> ContextHistory {
        ContextHistory {
            messages: vec![self],
        }
    }
}

/// Container of history for an LLM session.
pub struct ContextHistory {
    messages: Vec<Message>,
}

impl ContextHistory {
    /// Push a message into context, taking ownership of it.
    #[inline]
    fn append_message(&mut self, msg: Message) {
        self.messages.push(msg);
    }
}

#[cfg(test)]
mod test {
    use std::path::PathBuf;

    use crate::openai::{ChatCompletions, ChatCompletionsClient, ContextHistory, Message};

    #[test]
    fn completions_can_send_message() {
        let client = ChatCompletionsClient::default();
        let system = Message::system();
        client.create("glm-5.3", &system.into())
    }
}
