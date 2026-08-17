use std::ops::Index;

use crate::ModelConfiguration;
use crate::openai::{ContextHistory, Message};

/// Chat Completions supports multiple completions per request. This is almost never
/// used, so we fix it to be 1.
const COMPLETION_MAX_CHOICES_PER_REQUEST: usize = 1;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum FinishReason {
    Stope,
    Length,
    ToolCalls,
    ContentFilter,
    FunctionCall,
}

#[derive(Clone, PartialEq, Eq)]
pub struct Choices {
    choices: [Message; COMPLETION_MAX_CHOICES_PER_REQUEST],
    finish_reason: FinishReason,
    // NOTE: we do not include logprobs or index, we don't really need them.
}

/// Interface for OpenAI ChatCompletions apis.
pub trait ChatCompletions {
    /// Request a response from the model endpoint.
    fn create(&self, model: &str, messages: &ContextHistory) -> Choices;
}

impl Choices {
    /// Return a reference to the sole choice in a message.
    pub fn get_single_choice(&self) -> (&Message, FinishReason) {
        // This fails to compile if COMPLETION_MAX_CHOICES_PER_REQUEST, preventing users
        // from accidentally discarding information, or mixing requests due to missing
        // index.
        let [choice] = &self.choices;

        (choice, self.finish_reason)
    }
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
    fn create(&self, _model: &str, _messages: &ContextHistory) -> Choices {
        todo!()
    }
}
