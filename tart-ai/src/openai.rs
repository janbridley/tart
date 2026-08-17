//! OpenAI Chat Completions interface.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Role {
    Assistant,
    System,
    Developer,
    User,
}

struct Message<'a> {
    /// The ,
    role: Role,
    /// The text associated with this message.
    content: &'a str, // NOTE: should be different container?
    /// An identifier for the message.
    id: String, // Owned identifier for the message
}

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
