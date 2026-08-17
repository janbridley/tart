//! OpenAI Chat Completions interface.

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
    role: Role,
    /// The text associated with this message.
    content: &'a str, // NOTE: should be different container?
    /// An identifier for the message.
    id: String, // Owned identifier for the message
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
}
