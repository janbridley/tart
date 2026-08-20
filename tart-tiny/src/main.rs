use async_openai::{
    Client,
    config::OpenAIConfig,
    traits::EventType,
    types::responses::{
        CreateResponseArgs, EasyInputMessage, FunctionCallOutput, FunctionCallOutputItemParam,
        FunctionTool, FunctionToolCall, InputItem, InputParam, Item, OutputItem,
        ResponseStreamEvent, Tool,
    },
};
use futures::StreamExt;
use std::{
    collections::HashMap,
    env,
    io::{Write, stdout},
};

/// What sandbox?
fn run_bash(command: &str) -> String {
    std::process::Command::new("bash")
        .arg("-c")
        .arg(command)
        .output()
        .map_or_else(
            |e| format!("error: {e}"),
            |o| {
                format!(
                    "{}{}",
                    String::from_utf8_lossy(&o.stdout),
                    String::from_utf8_lossy(&o.stderr)
                )
            },
        )
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let api_key = env::var("DEEPSEEK_API_KEY").expect("DEEPSEEK_API_KEY not set!");
    let config = OpenAIConfig::new()
        .with_api_base("https://api.deepseek.com")
        .with_api_key(&api_key);
    let client = Client::with_config(config);

    let tools = vec![Tool::Function(FunctionTool {
        defer_loading: None,
        name: "bash".to_string(),
        description: Some("Run a bash command and return its stdout/stderr".to_string()),
        parameters: Some(serde_json::json!({
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The bash command to run"}
            },
            "required": ["command"]
        })),
        strict: None,
    })];

    let mut input_items: Vec<InputItem> =
        vec![EasyInputMessage::from(
        "Write a haiku about programming, then use the bash tool to write it to ./haiku.txt",
    )
    .into()];

    let request = CreateResponseArgs::default()
        .model("deepseek-v4-flash")
        .stream(true)
        .input(InputParam::Items(input_items.clone()))
        .tools(tools.clone())
        .build()?;

    let mut stream = client.responses().create_stream(request).await?;
    let mut lock = stdout().lock();

    // Track function-call metadata (name, call_id) and streamed arguments by item_id
    let mut call_meta: HashMap<String, (String, String)> = HashMap::new();
    let mut call_args: HashMap<String, String> = HashMap::new();
    let mut function_call: Option<FunctionToolCall> = None;

    while let Some(result) = stream.next().await {
        let event = result?;
        match &event {
            ResponseStreamEvent::ResponseOutputTextDelta(delta) => {
                write!(lock, "{}", delta.delta)?;
            }
            ResponseStreamEvent::ResponseOutputItemAdded(added) => {
                if let OutputItem::FunctionCall(fc) = &added.item {
                    let item_id = fc.id.clone().unwrap_or_default();
                    call_meta.insert(item_id, (fc.name.clone(), fc.call_id.clone()));
                    writeln!(lock, "\n[tool call: {}]", fc.name)?;
                }
            }
            ResponseStreamEvent::ResponseFunctionCallArgumentsDelta(delta) => {
                call_args
                    .entry(delta.item_id.clone())
                    .or_default()
                    .push_str(&delta.delta);
            }
            ResponseStreamEvent::ResponseFunctionCallArgumentsDone(done) => {
                if let Some((name, call_id)) = call_meta.get(&done.item_id) {
                    function_call = Some(FunctionToolCall {
                        namespace: None,
                        name: name.clone(),
                        arguments: call_args.remove(&done.item_id).unwrap_or_default(),
                        call_id: call_id.clone(),
                        id: Some(done.item_id.clone()),
                        status: None,
                    });
                }
            }
            _ => {
                writeln!(lock, "\n{}: skipping", event.event_type())?;
            }
        }
        stdout().flush()?;
    }
    writeln!(lock)?;

    let Some(call) = function_call else {
        return Ok(());
    };

    // Execute the tool
    let args: serde_json::Value = serde_json::from_str(&call.arguments)?;
    let command = args["command"].as_str().ok_or("missing 'command' argument")?;
    writeln!(lock, "[running] {command}")?;
    let output = run_bash(command);
    writeln!(lock, "[output]\n{output}")?;

    // Replay the call + its output, then stream the final answer
    input_items.push(InputItem::Item(Item::FunctionCall(call.clone())));
    input_items.push(InputItem::Item(Item::FunctionCallOutput(
        FunctionCallOutputItemParam {
            call_id: call.call_id.clone(),
            output: FunctionCallOutput::Text(output),
            id: None,
            status: None,
        },
    )));

    let request = CreateResponseArgs::default()
        .model("deepseek-v4-flash")
        .stream(true)
        .input(InputParam::Items(input_items))
        .tools(tools)
        .build()?;

    let mut stream = client.responses().create_stream(request).await?;
    while let Some(result) = stream.next().await {
        if let ResponseStreamEvent::ResponseOutputTextDelta(delta) = result? {
            write!(lock, "{}", delta.delta)?;
            stdout().flush()?;
        }
    }
    writeln!(lock)?;

    Ok(())
}
