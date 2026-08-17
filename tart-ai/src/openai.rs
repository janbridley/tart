//! OpenAI Chat Completions interface.

use std::{borrow::Cow, path::PathBuf};


mod completions;

/// Valid `role` entries for a [`Message`]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Role {
    Assistant,
    System,
    Developer,
    User,
}

/// One unit of data passed from the user to the model or vice versa.
struct Message<'a> {
    /// The ,
    pub(crate) role: Role,
    /// The text associated with this message.
    pub(crate) content: Cow<'a, str>,
}

/// Container of history for an LLM session.
pub struct Context<'a> {
    responses: Vec<Message<'a>>,
}

impl<'a> Context<'a> {
    /// Push a message into context, taking ownership of it.
    #[inline]
    fn append_message(&mut self, msg: Message<'a>) {
        self.responses.push(msg);
    }

    #[inline]
    fn from_system_prompt_file(&mut self, filename: PathBuf) -> anyhow::Result<()> {
        let is_markdown = filename
            .extension()
            .and_then(|ext| ext.to_str())
            .is_some_and(|ext| ext.eq_ignore_ascii_case("md"));

        if !is_markdown {
            anyhow::bail!("Expected a Markdown file (.md), but received: {filename:?}");
        }

        self.append_message(Message {
            role: Role::System,
            content: Cow::from(std::fs::read_to_string(&filename)?),
        });
        Ok(())
    }
}

#[cfg(test)]
mod test {
    use crate::openai::{ChatCompletions, ChatCompletionsClient, Context};

    #[test]
    fn completions_can_send_message() {
        let client = ChatCompletionsClient::default();
        // let messages = Context::client.create("glm-5.3");
    }
}
