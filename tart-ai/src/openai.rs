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

impl Context<'a> {
    #[inline]
    fn append_message(&mut self, msg: &Message) {
        self.responses.push(msg);
    }
}
