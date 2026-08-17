//! Manual bin for testing request output.

use std::io::Write;

use tart_ai::ContextHistory;
use tart_ai::openai::{ChatCompletionsClient, Message, Role};

fn main() -> anyhow::Result<()> {
    let api_key = std::env::var("DEEPSEEK_API_KEY")?;
    let client = ChatCompletionsClient::new(
        "https://api.deepseek.com/chat/completions",
        api_key,
        "deepseek-v4-flash",
    );

    let history = ContextHistory::from(Message {
        role: Role::User,
        content: "Give a 100 word story.".to_string(),
    });

    let stream = client.create(&history)?;
    let (message, finish_reason) = stream.complete(|delta| {
        print!("{delta}");
        // Tokens carry no newlines so we flush each delta manually
        let _ = std::io::stdout().flush();
    })?;
    println!();
    println!("finish_reason: {finish_reason:?}");
    println!("assembled: {}", message.content);

    Ok(())
}
