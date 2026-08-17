use std::ops::Index;

use serde::{Deserialize, Serialize};

use crate::ModelConfiguration;
use crate::openai::{ContextHistory, Message};

/// Chat Completions supports multiple completions per request. This is almost never
/// used, so we fix it to be 1.
const COMPLETION_MAX_CHOICES_PER_REQUEST: usize = 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    Stop,
    Length,
    ToolCalls,
    ContentFilter,
    FunctionCall,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Choices {
    choices: [Message; COMPLETION_MAX_CHOICES_PER_REQUEST],
    finish_reason: FinishReason,
    // NOTE: we do not include logprobs or index, we don't really need them.
}

/// Interface for OpenAI ChatCompletions apis.
pub trait ChatCompletions {
    /// Request a response from the model endpoint.
    fn create(&self, model: &str, messages: &ContextHistory) -> anyhow::Result<Choices>;
}

/// The JSON body of a chat completion request.
#[derive(Serialize)]
struct CompletionRequest<'a> {
    /// A model name the provider recognizes.
    model: &'a str,
    /// The conversation so far.
    messages: &'a [Message],
}

/// The JSON body of a chat completion response, limited to the fields we use.
#[derive(Deserialize)]
struct CompletionResponse {
    choices: Vec<CompletionChoice>,
}

/// One entry of `choices` in a [`CompletionResponse`].
#[derive(Deserialize)]
struct CompletionChoice {
    message: Message,
    finish_reason: FinishReason,
}

impl Choices {
    /// Return a reference to the sole choice in a message.
    #[inline]
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

impl ChatCompletionsClient {
    /// Configure a client for an OpenAI-compatible Chat Completions endpoint.
    #[inline]
    pub fn new(
        completions_url: impl Into<String>,
        api_key: impl Into<String>,
        model: impl Into<String>,
    ) -> Self {
        Self {
            completions_url: completions_url.into(),
            api_key: api_key.into(),
            model: model.into(),
        }
    }
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
    fn create(&self, model: &str, messages: &ContextHistory) -> anyhow::Result<Choices> {
        let request = CompletionRequest {
            model,
            messages: &messages.messages,
        };

        let response = match ureq::post(self.url().as_str())
            .set("Authorization", &format!("Bearer {}", self.api_key()))
            .send_json(&request)
        {
            Ok(response) => response,
            // Surface the provider's error body, which explains what went wrong.
            Err(ureq::Error::Status(code, response)) => anyhow::bail!(
                "chat completions endpoint returned {code}: {}",
                response
                    .into_string()
                    .unwrap_or_else(|_| "<no body>".to_string())
            ),
            Err(error) => return Err(error.into()),
        };

        let completion: CompletionResponse = response.into_json()?;
        let [choice] = &completion.choices[..] else {
            anyhow::bail!(
                "expected exactly one choice, but the endpoint returned {}",
                completion.choices.len()
            );
        };

        Ok(Choices {
            choices: [choice.message.clone()],
            finish_reason: choice.finish_reason,
        })
    }
}
