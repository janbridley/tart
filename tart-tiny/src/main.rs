use async_openai::{
    Client,
    config::OpenAIConfig,
    traits::EventType,
    types::responses::{CreateResponseArgs, ResponseStreamEvent},
};
use futures::StreamExt;
use std::{
    env,
    io::{Write, stdout},
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let api_key = env::var("DEEPSEEK_API_KEY").expect("DEEPSEEK_API_KEY not set!");
    let config = OpenAIConfig::new()
        .with_api_base("https://api.deepseek.com")
        .with_api_key(&api_key);
    let client = Client::with_config(config);

    let request = CreateResponseArgs::default()
        .model("deepseek-v4-flash")
        .stream(true)
        .input("Write a haiku about programming.")
        .build()?;

    let mut stream = client.responses().create_stream(request).await?;

    let mut lock = stdout().lock();

    while let Some(result) = stream.next().await {
        match result {
            Ok(response_event) => match &response_event {
                ResponseStreamEvent::ResponseOutputTextDelta(delta) => {
                    write!(lock, "{}", delta.delta)?;
                }
                _ => {
                    writeln!(lock, "\n{}: skipping\n", response_event.event_type())?;
                }
            },
            Err(e) => {
                eprintln!("\n{e:#?}");
            }
        }
        stdout().flush()?;
    }

    Ok(())
}
