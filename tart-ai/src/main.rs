//! Manual bin for testing request output.

use std::io::Write;

use tart_ai::openai::{ChatCompletionsClient, Delta, Message};
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
    history.append_message(Message::user("Who are you?".to_string()));

    let mut stream = client.create(&history)?;
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

    // Read usage from the stream into our history
    if let Some(u) = stream.usage() {
        history.record_usage(u)
    }
    // Append the message to our context
    history.append_message(message);
    let usage = history.usage();
    println!(
        "input tokens:  {}\noutput tokens: {} completion + {} reasoning ({} cached)",
        usage.prompt_tokens, usage.completion_tokens, usage.reasoning_tokens, usage.cached_tokens
    );

    Ok(())
}
