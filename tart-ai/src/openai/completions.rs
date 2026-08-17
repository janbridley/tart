use crate::openai::Context;

pub trait ChatCompletions {
    fn create(&self, model: &str, messages: Context);
}

#[derive(Default)]
pub struct ChatCompletionsClient {}

impl ChatCompletions for ChatCompletionsClient {
    fn create(&self, _model: &str, _messages: Context) {
        todo!()
    }
}
