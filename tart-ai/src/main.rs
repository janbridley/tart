//! Manual bin for testing request output.

use tart_ai::openai::{ChatCompletions, ChatCompletionsClient, ContextHistory, Message, Role};

fn main() -> anyhow::Result<()> {
    let api_key = std::env::var("DEEPSEEK_API_KEY")?;
    let client = ChatCompletionsClient::new(
        "https://api.deepseek.com/chat/completions",
        api_key,
        "deepseek-chat",
    );

    let prompt: ContextHistory = Message {
        role: Role::User,
        content: "Say hi in one word.".to_string(),
    }
    .into();

    let choices = client.create("deepseek-v4-flash", &prompt)?;
    let (message, finish_reason) = choices.get_single_choice();
    println!("finish_reason: {finish_reason:?}");
    println!("content: {}", message.content);

    Ok(())
}
