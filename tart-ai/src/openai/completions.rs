use crate::openai::ContextHistory;

pub trait ChatCompletions {
    fn create(&self, model: &str, messages: &ContextHistory);
}

#[derive(Default)]
pub struct ChatCompletionsClient {}

impl ChatCompletions for ChatCompletionsClient {
    fn create(&self, _model: &str, _messages: &ContextHistory) {
        todo!()
    }
}
