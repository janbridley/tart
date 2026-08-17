//! Manual bin for testing request output.

use std::io::Write;

use tart_ai::openai::{ChatCompletionsClient, Delta, Message, Role};
use tart_ai::{ContextHistory, ReasoningEffort};

/// How hard the model reasons before answering.
const REASONING_EFFORT: ReasoningEffort = ReasoningEffort::Low;

fn main() -> anyhow::Result<()> {
    let api_key = std::env::var("DEEPSEEK_API_KEY")?;
    let client = ChatCompletionsClient::new(
        "https://api.deepseek.com/chat/completions",
        api_key,
        "deepseek-v4-flash",
    )
    .reasoning_effort(REASONING_EFFORT);

    let mut history = ContextHistory::from(Message::system());
    history.append_message(Message {
        role: Role::User,
        content: "Who are you?".to_string(),
    });

    let stream = client.create(&history)?;
    let (message, finish_reason) = stream.complete(|delta| {
        match delta {
            // Dim the chain-of-thought; it precedes the answer.
            Delta::Thinking(text) => print!("\x1b[2m{text}\x1b[0m"),
            Delta::Answer(text) => print!("{text}"),
        }
        // Tokens carry no newlines so we flush each delta manually
        let _ = std::io::stdout().flush();
    })?;
    println!();
    println!("finish_reason: {finish_reason:?}");
    println!("assembled: {}", message.content);

    Ok(())
}
