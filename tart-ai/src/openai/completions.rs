use std::ops::Index;

use crate::ModelConfiguration;
use crate::openai::{ContextHistory, Message};

pub trait ChatCompletions {
    /// Request a response from the model endpoint.
    fn create(&self, model: &str, messages: &ContextHistory) -> Message;
}

#[derive(Default)]
pub struct ChatCompletionsClient {
    /// An OpenAI Chat-Completions URL.
    completions_url: String,
    api_key: String,
    model: String,
}

impl ModelConfiguration for ChatCompletionsClient {
    fn url(&self) -> url::Url {
        url::Url::parse(&self.completions_url).expect("Completions URL must be parseable!")
    }
    fn api_key(&self) -> String {
        self.api_key.clone()
    }
    fn model(&self) -> String {
        self.model.clone()
    }
}

impl ChatCompletions for ChatCompletionsClient {
    fn create(&self, _model: &str, _messages: &ContextHistory) -> Message {
        todo!()
    }
}
