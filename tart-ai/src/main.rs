//! Manual bin for testing request output.

use tart_ai::openai::{ChatCompletionsClient, Message, Role};

fn main() -> anyhow::Result<()> {
    let api_key = std::env::var("DEEPSEEK_API_KEY")?;
    let client = ChatCompletionsClient::new(
        "https://api.deepseek.com/chat/completions",
        api_key,
        "deepseek-v4-flash",
    );

    let messages = [Message {
        role: Role::User,
        content: "Say hi and then a random number.".to_string(),
    }];

    let (message, finish_reason) = client.create(&messages)?;
    println!("finish_reason: {finish_reason:?}");
    println!("content: {}", message.content);

    Ok(())
}
