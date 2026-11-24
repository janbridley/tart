//! The `OpenAI` Responses backend: the only module that touches `async-openai`.

use std::future::Future;

use async_openai::{
    Client,
    config::OpenAIConfig,
    error::OpenAIError,
    types::responses::{
        CreateResponse, CreateResponseArgs, EasyInputMessageArgs, FunctionCallOutput,
        FunctionCallOutputItemParam, FunctionTool, InputParam, Reasoning,
        ReasoningEffort as ResponsesEffort, ResponseErrorCode, ResponseStream, ResponseUsage,
    },
};
use futures::StreamExt;

use crate::debug;
use crate::usage::TokenUsage;

use super::{Backend, Event, ReasoningEffort, RoundStream};

/// The record and tool vocabulary the loop exchanges with any backend, plus
/// the raw wire events for tests that serve a stream by hand.
pub(crate) use async_openai::types::responses::{
    EasyInputContent, FunctionToolCall, InputItem, Item, OutputItem, ReasoningItem,
    ReasoningItemContent, ResponseStreamEvent, Role, Tool,
};
/// Wire and content types only test fixtures require. Can't be below for some reason?
#[cfg(test)]
pub(crate) use async_openai::types::responses::{
    ReasoningTextContent, ResponseOutputItemDoneEvent, ResponseTextDeltaEvent,
};

/// A Responses-API endpoint.
#[derive(Clone, Debug)]
pub struct Responses {
    client: Client<OpenAIConfig>,
    chatgpt_http: Option<reqwest::Client>,
}

pub(crate) fn chatgpt_backend() -> Responses {
    let mut backend = backend(crate::chatgpt::RESOURCE, "");
    backend.chatgpt_http = Some(
        reqwest::Client::builder()
            .https_only(true)
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .expect("ChatGPT HTTP client did not initialize"),
    );
    backend
}

/// A backend handle for an OpenAI-compatible Responses endpoint.
pub(crate) fn backend<U: Into<String>, K: Into<String>>(base_url: U, api_key: K) -> Responses {
    // Register a cryptography backend, otherwise reqwest rustls-no-provider panics.
    let _ = rustls::crypto::ring::default_provider().install_default();
    let config = OpenAIConfig::new()
        .with_api_base(base_url.into())
        .with_api_key(api_key.into());
    Responses {
        client: Client::with_config(config),
        chatgpt_http: None,
    }
}

impl Backend for Responses {
    fn requires_completion(&self) -> bool {
        self.chatgpt_http.is_some()
    }

    fn round(
        &self,
        model: &str,
        effort: Option<ReasoningEffort>,
        items: Vec<InputItem>,
        tools: Vec<Tool>,
    ) -> impl Future<Output = anyhow::Result<RoundStream>> + Send {
        let model = model.to_string();
        async move {
            let mut request = request_of(&model, effort, items, tools)?;
            if self.requires_completion() {
                configure_chatgpt_request(&mut request);
            }
            debug::log_json("round request", || serde_json::to_string(&request));
            let stream = if let Some(http) = &self.chatgpt_http {
                chatgpt_stream(http, request).await?
            } else {
                self.client.responses().create_stream(request).await?
            };
            Ok(Box::pin(stream.filter_map(|item| async {
                match item {
                    Ok(event) => translate(event).map(Ok),
                    Err(error) => Some(Err(anyhow::Error::new(error))),
                }
            })) as RoundStream)
        }
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
) -> anyhow::Result<CreateResponse> {
    // The builder's setters borrow it, so chain off the binding.
    let mut args = CreateResponseArgs::default();
    args.model(model)
        .stream(true)
        .reasoning(Reasoning {
            effort: effort.map(effort_of),
            summary: None,
            context: None,
            mode: None,
        })
        .input(InputParam::Items(
            items.into_iter().map(strip_server_side_state).collect(),
        ));
    if !tools.is_empty() {
        args.tools(tools);
    }
    Ok(args.build()?)
}

fn configure_chatgpt_request(request: &mut CreateResponse) {
    use async_openai::types::responses::{
        FunctionToolParam, NamespaceToolParam, NamespaceToolParamTool,
    };

    request.store = Some(false);
    if let InputParam::Items(items) = &mut request.input {
        for item in items {
            if let InputItem::EasyMessage(message) = item
                && message.role == Role::System
            {
                message.role = Role::Developer;
            }
        }
    }
    if let Some(tools) = &mut request.tools {
        let mut functions = Vec::new();
        tools.retain(|tool| {
            let Tool::Function(function) = tool else {
                return true;
            };
            functions.push(NamespaceToolParamTool::Function(FunctionToolParam {
                name: function.name.clone(),
                description: function.description.clone(),
                parameters: function.parameters.clone(),
                strict: function.strict,
                defer_loading: function.defer_loading,
                r#async: function.r#async,
                output_schema: function.output_schema.clone(),
                allowed_callers: function.allowed_callers.clone(),
            }));
            false
        });
        if !functions.is_empty() {
            tools.push(Tool::Namespace(NamespaceToolParam {
                name: "tart".into(),
                description: "Tools provided by Tart".into(),
                tools: functions,
            }));
        }
    }
}

/// One record item as a plain, self-contained item.
///
/// We must drop server-side data (`id`s for stored items, as well as their
/// `encrypted_content`), or proxies like `Azure OpenAI` kill the stream when the
/// backing resource changes. We keep track of dropped data on our end for posterity.
fn strip_server_side_state(mut item: InputItem) -> InputItem {
    match &mut item {
        InputItem::Item(Item::Reasoning(reasoning)) => {
            reasoning.id = None;
            reasoning.encrypted_content = None;
        }
        InputItem::Item(Item::FunctionCall(call)) => call.id = None,
        _ => {} // Nothing else the record holds names stored state.
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
        ResponseStreamEvent::ResponseOutputItemDone(done) => match done.item {
            OutputItem::Reasoning(item) => Some(Event::Reasoning(item)),
            OutputItem::FunctionCall(call) => Some(Event::Call(call)),
            _ => None,
        },
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

/// The wire spelling of a reasoning effort; the seam an upstream rename breaks.
const fn effort_of(effort: ReasoningEffort) -> ResponsesEffort {
    match effort {
        ReasoningEffort::None => ResponsesEffort::None,
        ReasoningEffort::Minimal => ResponsesEffort::Minimal,
        ReasoningEffort::Low => ResponsesEffort::Low,
        ReasoningEffort::Medium => ResponsesEffort::Medium,
        ReasoningEffort::High => ResponsesEffort::High,
        ReasoningEffort::Xhigh => ResponsesEffort::Xhigh,
        ReasoningEffort::Max => ResponsesEffort::Max,
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
pub(crate) fn message(role: Role, text: String) -> anyhow::Result<InputItem> {
    Ok(EasyInputMessageArgs::default()
        .role(role)
        .content(text)
        .build()?
        .into())
}

/// A function tool with the given name, description, and JSON-schema parameters.
pub(crate) fn tool(name: &str, description: &str, parameters: serde_json::Value) -> Tool {
    Tool::Function(FunctionTool {
        defer_loading: None,
        name: name.to_string(),
        description: Some(description.to_string()),
        parameters: Some(parameters),
        strict: None,
        r#async: None,
        output_schema: None,
        allowed_callers: None,
    })
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

    #[test]
    fn chatgpt_uses_stateless_requests_and_namespaced_tools() {
        let mut request = request_of(
            "model",
            None,
            vec![message(Role::System, "Instructions".into()).unwrap()],
            vec![tool("read", "Read a file", serde_json::json!({"type": "object"}))],
        )
        .unwrap();
        configure_chatgpt_request(&mut request);
        let wire = serde_json::to_value(request).unwrap();
        assert_eq!(wire["store"], false);
        assert_eq!(wire["stream"], true);
        assert_eq!(wire["input"][0]["role"], "developer");
        assert_eq!(wire["tools"][0]["type"], "namespace");
        assert_eq!(wire["tools"][0]["name"], "tart");
        assert_eq!(wire["tools"][0]["tools"][0]["name"], "read");
    }

    #[test]
    fn the_request_omits_an_empty_tools_list() {
        let bare = serde_json::to_value(request_of("model", None, Vec::new(), Vec::new()).unwrap())
            .unwrap();
        assert!(bare.get("tools").is_none(), "no tools, no field: {bare}");

        let armed = serde_json::to_value(
            request_of("model", None, Vec::new(), vec![crate::tools::bash()]).unwrap(),
        )
        .unwrap();
        assert_eq!(
            armed["tools"].as_array().map(Vec::len),
            Some(1),
            "the offered tool rides along: {armed}"
        );
    }

    #[test]
    fn the_request_replays_items_without_provider_minted_ids() {
        let reasoning = InputItem::Item(Item::Reasoning(ReasoningItem {
            id: Some("rs_0".to_string()),
            encrypted_content: Some("ciphertext".to_string()),
            content: Some(vec![ReasoningItemContent::ReasoningText(ReasoningTextContent {
                text: "pondering".to_string(),
            })]),
            ..ReasoningItem::default()
        }));
        let call = InputItem::Item(Item::FunctionCall(FunctionToolCall {
            id: Some("fc_0".to_string()),
            call_id: "call_0".to_string(),
            name: "bash".to_string(),
            arguments: "{}".to_string(),
            ..FunctionToolCall::default()
        }));
        let request = request_of("model", None, vec![reasoning, call], Vec::new()).unwrap();

        let body = serde_json::to_value(&request).unwrap();
        let input = &body["input"];
        // An id is either stripped or serialized as null, whichever the type
        // does with a `None`; both leave nothing for a resource to look up.
        assert!(input[0]["id"].is_null(), "reasoning id gone: {input}");
        assert!(
            input[0]["encrypted_content"].is_null(),
            "ciphertext gone: {input}"
        );
        assert!(input[1]["id"].is_null(), "call id gone: {input}");
        // What carries meaning for the next round stays.
        assert_eq!(input[0]["content"][0]["text"], "pondering");
        assert_eq!(input[1]["call_id"], "call_0");
        assert_eq!(input[1]["name"], "bash");
    }
}
