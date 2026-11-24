//! The `OpenAI` Responses backend: the only module that touches `async-openai`.

use async_openai::{
    Client,
    config::OpenAIConfig,
    error::OpenAIError,
    types::responses::{
        CreateResponse, EasyInputMessage, FunctionCallOutput, FunctionCallOutputItemParam,
        FunctionToolParam, IncludeEnum, InputParam, NamespaceToolParam, NamespaceToolParamTool,
        Reasoning, ResponseErrorCode, ResponseStream, ResponseUsage,
    },
};
use futures::{Stream, StreamExt};

use crate::debug;
use crate::usage::TokenUsage;

use super::{Event, ReasoningEffort};

/// The record vocabulary the loop exchanges with the backend, plus
/// the raw wire events for tests that serve a stream by hand.
pub(crate) use async_openai::types::responses::{
    EasyInputContent, FunctionTool as Tool, FunctionToolCall, InputItem, Item, MessageItem,
    OutputItem, OutputMessageContent, ReasoningItemContent, ResponseStreamEvent, Role,
    Tool as WireTool,
};
/// Wire types stream fixtures are built from.
#[cfg(test)]
pub(crate) use async_openai::types::responses::{
    ResponseOutputItemDoneEvent, ResponseTextDeltaEvent,
};

/// Credentials and endpoint for the Responses API.
#[derive(Clone)]
#[non_exhaustive]
pub enum Provider {
    /// An API key for an OpenAI-compatible endpoint.
    ApiKey {
        /// The endpoint's base URL.
        base_url: String,
        /// The key sent to the endpoint.
        api_key: String,
    },
    /// The saved `ChatGPT` account, scoped to the `OpenAI` endpoint.
    ChatGpt,
}

impl std::fmt::Debug for Provider {
    #[inline]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::ApiKey { .. } => "ApiKey",
            Self::ChatGpt => "ChatGpt",
        })
    }
}

impl Provider {
    /// An API key for an OpenAI-compatible endpoint.
    #[inline]
    pub fn api_key<U: Into<String>, K: Into<String>>(base_url: U, api_key: K) -> Self {
        Self::ApiKey {
            base_url: base_url.into(),
            api_key: api_key.into(),
        }
    }
}

#[derive(Clone)]
pub(crate) enum Responses {
    ApiKey(Client<OpenAIConfig>),
    ChatGpt(reqwest::Client),
    #[cfg(test)]
    ChatGptFixture(Client<OpenAIConfig>),
}

impl Responses {
    pub(crate) fn new(provider: Provider) -> Self {
        // Reqwest uses rustls without a default cryptography provider.
        let _ = rustls::crypto::ring::default_provider().install_default();
        match provider {
            Provider::ApiKey { base_url, api_key } => Self::ApiKey(Client::with_config(
                OpenAIConfig::new().with_api_base(base_url).with_api_key(api_key),
            )),
            Provider::ChatGpt => Self::ChatGpt(
                reqwest::Client::builder()
                    .https_only(true)
                    .redirect(reqwest::redirect::Policy::none())
                    .build()
                    .expect("ChatGPT HTTP client did not initialize"),
            ),
        }
    }

    pub(crate) const fn requires_completion(&self) -> bool {
        !matches!(self, Self::ApiKey(_))
    }

    #[cfg(test)]
    pub(crate) fn use_chatgpt_wire_format_for_test(&mut self) {
        if let Self::ApiKey(client) = self {
            *self = Self::ChatGptFixture(client.clone());
        }
    }

    pub(crate) async fn round(
        &self,
        model: &str,
        effort: Option<ReasoningEffort>,
        items: Vec<InputItem>,
        tools: Vec<Tool>,
    ) -> anyhow::Result<impl Stream<Item = anyhow::Result<Event>> + Unpin + use<>> {
        let request = request_of(model, effort, items, tools, self.requires_completion());
        debug::log_json("round request", || serde_json::to_string(&request));
        let stream = match self {
            Self::ApiKey(client) => client.responses().create_stream(request).await?,
            Self::ChatGpt(http) => chatgpt_stream(http, request).await?,
            #[cfg(test)]
            Self::ChatGptFixture(client) => client.responses().create_stream(request).await?,
        };
        Ok(stream.filter_map(|item| {
            std::future::ready(match item {
                Ok(event) => translate(event).map(Ok),
                Err(error) => Some(Err(error.into())),
            })
        }))
    }
}

async fn chatgpt_stream(
    http: &reqwest::Client,
    request: CreateResponse,
) -> anyhow::Result<ResponseStream> {
    use crate::chatgpt::{RECONNECT, RESOURCE, access_token, invalidate_access_token};

    let token = tokio::task::spawn_blocking(access_token).await??;
    let client = Client::build(
        http.clone(),
        OpenAIConfig::new()
            .with_api_base(RESOURCE)
            .with_api_key(token.clone())
            .with_org_id("")
            .with_project_id(""),
    );
    let error = match client.responses().create_stream(request).await {
        Err(OpenAIError::ApiError(error))
            if error.status_code == reqwest::StatusCode::UNAUTHORIZED =>
        {
            error
        }
        other => return Ok(other?),
    };
    match error.api_error.code.as_deref() {
        Some("token_revoked" | "refresh_token_invalidated") => {
            tokio::task::spawn_blocking(move || invalidate_access_token(&token)).await??;
            anyhow::bail!("ChatGPT session was revoked. {RECONNECT}.");
        }
        Some("subscription_sharing_invalid_user" | "invalid_token") => {
            anyhow::bail!("{error}. {RECONNECT}.");
        }
        _ => Err(OpenAIError::ApiError(error).into()),
    }
}

/// The wire request one round sends.
fn request_of(
    model: &str,
    effort: Option<ReasoningEffort>,
    items: Vec<InputItem>,
    tools: Vec<Tool>,
    chatgpt: bool,
) -> CreateResponse {
    CreateResponse {
        model: Some(model.into()),
        stream: Some(true),
        store: chatgpt.then_some(false),
        // Encrypted reasoning carries the model's thinking between rounds;
        // asking on every request keeps replay possible on stateful endpoints too.
        include: Some(vec![IncludeEnum::ReasoningEncryptedContent]),
        reasoning: Some(Reasoning { effort, ..Reasoning::default() }),
        input: InputParam::Items(items.into_iter().map(|item| render(item, chatgpt)).collect()),
        tools: (!tools.is_empty()).then(|| {
            if chatgpt {
                vec![WireTool::Namespace(NamespaceToolParam {
                    name: "tart".into(),
                    description: "Tools provided by Tart".into(),
                    tools: tools
                        .into_iter()
                        .map(|tool| {
                            NamespaceToolParamTool::Function(FunctionToolParam {
                                name: tool.name,
                                description: tool.description,
                                parameters: tool.parameters,
                                strict: tool.strict,
                                defer_loading: tool.defer_loading,
                                r#async: tool.r#async,
                                output_schema: tool.output_schema,
                                allowed_callers: tool.allowed_callers,
                            })
                        })
                        .collect(),
                })]
            } else {
                tools.into_iter().map(WireTool::Function).collect()
            }
        }),
        ..CreateResponse::default()
    }
}

/// One recorded item as the endpoint takes it.
///
/// Provider items replay as they arrived, ids and encrypted reasoning included:
/// the item carries its own state, so no lookup has to resolve it. The one
/// rendering is the `ChatGPT` backend's, which takes a `developer` message
/// where others take `system`.
fn render(mut item: InputItem, chatgpt: bool) -> InputItem {
    if chatgpt
        && let InputItem::EasyMessage(message) = &mut item
        && message.role == Role::System
    {
        message.role = Role::Developer;
    }
    item
}

/// Translate one provider event; `None` when tart has no use for it.
fn translate(event: ResponseStreamEvent) -> Option<Event> {
    match event {
        ResponseStreamEvent::ResponseOutputTextDelta(delta) => Some(Event::Answer(delta.delta)),
        ResponseStreamEvent::ResponseReasoningTextDelta(delta) => {
            Some(Event::Thinking(delta.delta))
        }
        ResponseStreamEvent::ResponseOutputItemAdded(added) => {
            matches!(added.item, OutputItem::FunctionCall(_)).then_some(Event::CallStarted)
        }
        // Finished items are the record: an output item is a valid input item
        // as the provider minted it, and the crate's conversion is total.
        ResponseStreamEvent::ResponseOutputItemDone(done) => {
            Some(Event::ItemDone(done.item.into()))
        }
        ResponseStreamEvent::ResponseCompleted(completed) => {
            Some(Event::Completed(completed.response.usage.as_ref().map(usage_of)))
        }
        ResponseStreamEvent::ResponseFailed(failed) => {
            debug::log_json("response failed event", || serde_json::to_string(&failed));
            Some(Event::Failed(failed.response.error.map_or_else(
                || "response failed".to_string(),
                |error| format!("{}: {}", code_text(&error.code), error.message),
            )))
        }
        ResponseStreamEvent::ResponseError(error) => {
            debug::log_json("response error event", || serde_json::to_string(&error));
            Some(Event::Failed(format!(
                "{}: {}",
                error.code.unwrap_or_else(|| "error".to_string()),
                error.message
            )))
        }
        ResponseStreamEvent::ResponseIncomplete(incomplete) => {
            debug::log_json("response incomplete event", || serde_json::to_string(&incomplete));
            Some(Event::Incomplete(
                incomplete
                    .response
                    .incomplete_details
                    .map_or_else(|| "unknown reason".to_string(), |details| details.reason),
            ))
        }
        _ => None,
    }
}

/// The usage a completed response reported: `cached` from `input_tokens_details`,
/// `reasoning` from `output_tokens_details`.
fn usage_of(usage: &ResponseUsage) -> TokenUsage {
    TokenUsage {
        input: u64::from(usage.input_tokens),
        cached: u64::from(usage.input_tokens_details.cached_tokens),
        output: u64::from(usage.output_tokens),
        reasoning: u64::from(usage.output_tokens_details.reasoning_tokens),
        total: u64::from(usage.total_tokens),
    }
}

/// An error code as text: the raw string for passthrough codes, the name otherwise.
fn code_text(code: &ResponseErrorCode) -> String {
    match code {
        ResponseErrorCode::Other(code) => code.clone(),
        named => format!("{named:?}"),
    }
}

/// One message in the conversation, as the backend sends it.
pub(crate) fn message(role: Role, text: String) -> InputItem {
    EasyInputMessage {
        role,
        content: text.into(),
        ..EasyInputMessage::default()
    }
    .into()
}

pub(crate) fn tool(name: &str, description: &str, parameters: serde_json::Value) -> Tool {
    Tool {
        name: name.into(),
        description: Some(description.into()),
        parameters: Some(parameters),
        ..Tool::default()
    }
}

/// A tool call's output, paired back to its call.
pub(crate) fn call_output(call: &FunctionToolCall, output: String) -> InputItem {
    InputItem::Item(Item::FunctionCallOutput(FunctionCallOutputItemParam {
        call_id: Some(call.call_id.clone()),
        output: FunctionCallOutput::Text(output),
        id: None,
        status: None,
        name: None,
        namespace: call.namespace.clone(),
        caller: None,
    }))
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, reason = "test assertions")]

    use super::*;

    /// The request's shape on both backends: streaming; stateless on `ChatGPT`,
    /// with namespaced tools, `developer` system messages, and encrypted
    /// reasoning asked for; no `tools` field when nothing is offered.
    #[test]
    fn the_request_renders_for_its_endpoint() {
        let system = message(Role::System, "Instructions".into());
        let read = tool("read", "Read a file", serde_json::json!({"type": "object"}));
        let chatgpt =
            serde_json::to_value(request_of("model", None, vec![system], vec![read], true))
                .unwrap();
        assert_eq!(chatgpt["store"], false);
        assert_eq!(chatgpt["stream"], true);
        assert_eq!(chatgpt["input"][0]["role"], "developer");
        assert_eq!(chatgpt["tools"][0]["type"], "namespace");
        assert_eq!(chatgpt["tools"][0]["tools"][0]["name"], "read");
        assert_eq!(
            chatgpt["include"],
            serde_json::json!(["reasoning.encrypted_content"])
        );

        let bare =
            serde_json::to_value(request_of("model", None, Vec::new(), Vec::new(), false)).unwrap();
        assert!(bare.get("tools").is_none(), "no tools, no field: {bare}");
        let armed = serde_json::to_value(request_of(
            "model",
            None,
            Vec::new(),
            vec![crate::tools::bash()],
            false,
        ))
        .unwrap();
        assert_eq!(armed["tools"][0]["type"], "function");
        assert_eq!(armed["tools"][0]["name"], "bash");
    }

    /// The replay contract: a recorded item goes back as it arrived, minted ids
    /// and encrypted reasoning included, on both backends.
    #[test]
    fn the_request_replays_recorded_items_verbatim() {
        let items = [
            serde_json::json!({
                "type": "reasoning",
                "id": "rs_0",
                "summary": [],
                "content": [{"type": "reasoning_text", "text": "pondering"}],
                "encrypted_content": "ciphertext",
            }),
            serde_json::json!({
                "type": "function_call",
                "id": "fc_0",
                "call_id": "call_0",
                "name": "bash",
                "arguments": "{}",
                "status": "completed",
            }),
            serde_json::json!({
                "type": "message",
                "id": "msg_0",
                "role": "assistant",
                "status": "completed",
                "phase": "final_answer",
                "content": [
                    {"type": "output_text", "text": "All done.", "annotations": [], "logprobs": []},
                ],
            }),
        ];
        for value in items {
            let item: InputItem = serde_json::from_value(value.clone()).unwrap();
            for chatgpt in [false, true] {
                let request = request_of("model", None, vec![item.clone()], Vec::new(), chatgpt);
                let back = &serde_json::to_value(&request).unwrap()["input"][0];
                for (key, _) in value.as_object().unwrap() {
                    assert_eq!(back[key], value[key], "{} keeps {key}: {back}", value["type"]);
                }
            }
        }
    }
}
