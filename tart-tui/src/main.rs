//! A terminal chat front end for the `async-openai` Responses API.
//!
//! ```text
//! │ transcript (wraps, auto-tails)          │
//! │ ❯ hello                                 │
//! ├─────────────────────────────────────────┤
//! │ ❯ ▊ prompt, grows with its content      │
//! └─────────────────────────────────────────┘
//! ```

mod clipboard;
mod file_mentions;
mod keybinds;
mod pane;
mod tmux_override;

#[cfg(test)]
mod testutil;

use std::io::stdout;
use std::sync::mpsc::{self, RecvTimeoutError};
use std::time::Duration;

use async_compat::Compat;
use async_openai::{
    Client,
    config::OpenAIConfig,
    types::responses::{
        CreateResponse, CreateResponseArgs, EasyInputMessageArgs, InputItem, InputParam, Reasoning,
        ReasoningEffort, ResponseStreamEvent, Role,
    },
};
use futures::{StreamExt, executor::block_on};
use ratatui::DefaultTerminal;
use ratatui::crossterm::event::{
    self, DisableBracketedPaste, EnableBracketedPaste, Event, KeyCode, KeyEvent, KeyModifiers,
};
use ratatui::crossterm::execute;
use ratatui::text::Span;

use pane::{DIM_STYLE, Pane, PaneEvent};
use tmux_override::{override_shift_up, restore_tmux};

/// The personality tart brings to every conversation.
const SYSTEM: &str = include_str!("data/SYSTEM.md");

pub const DRAW_INTERVAL_MS: u64 = 100;

fn main() -> anyhow::Result<()> {
    install_panic_hook();
    let mut terminal = ratatui::try_init()?;
    execute!(stdout(), EnableBracketedPaste)?;
    // The alternate screen is live, so the conditional rebind takes effect.
    let _tmux = override_shift_up();
    let result = run(&mut terminal);
    ratatui::try_restore()?;
    execute!(stdout(), DisableBracketedPaste)?;
    terminal.show_cursor()?;
    result
}

/// Leave a normal terminal (and the tmux binding restored) even on panic.
fn install_panic_hook() {
    let hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        restore_tmux();
        let _ = ratatui::try_restore();
        let _ = execute!(stdout(), DisableBracketedPaste);
        hook(info);
    }));
}

/// One wake source for the event loop.
enum Wake {
    /// Terminal input, read on its own thread.
    Input(Event),
    /// Progress from the background generation.
    Generation(Progress),
}

/// Progress from one background generation.
enum Progress {
    /// A fragment of the model's chain-of-thought reasoning.
    Thinking(String),
    /// A fragment of the final answer.
    Answer(String),
    /// The stream ended; the assembled answer, if any arrived.
    Done { message: Option<String> },
    /// The request or stream failed.
    Failed(String),
}

/// One message in the conversation, as the Responses API sends it.
fn input_message(role: Role, text: String) -> anyhow::Result<InputItem> {
    Ok(EasyInputMessageArgs::default()
        .role(role)
        .content(text)
        .build()?
        .into())
}

/// Drive one streaming generation to completion, reporting progress to `on_progress`.
///
/// Fragments arrive as the model works, and the generation always ends with exactly
/// one terminal event: [`Progress::Done`] with the assembled answer (`None` if nothing
/// arrived), or [`Progress::Failed`] on a request or stream error. A stream that closes
/// without a terminal event still reports `Done` with whatever the model produced.
///
/// Blocks the current thread until the generation finishes.
fn generate<F: Fn(Progress) + Send + 'static>(
    client: &Client<OpenAIConfig>,
    request: CreateResponse,
    on_progress: F,
) {
    // `Compat` enters the global tokio runtime and exposes `futures` blocking control.
    let mut stream = match block_on(Compat::new(client.responses().create_stream(request))) {
        Ok(stream) => stream,
        Err(error) => return on_progress(Progress::Failed(error.to_string())),
    };

    let mut answer = String::new();
    while let Some(item) = block_on(stream.next()) {
        match item {
            Ok(ResponseStreamEvent::ResponseOutputTextDelta(delta)) => {
                answer.push_str(&delta.delta);
                on_progress(Progress::Answer(delta.delta));
            }
            Ok(ResponseStreamEvent::ResponseReasoningTextDelta(delta)) => {
                on_progress(Progress::Thinking(delta.delta));
            }
            Ok(ResponseStreamEvent::ResponseFailed(failed)) => {
                return on_progress(Progress::Failed(
                    failed
                        .response
                        .error
                        .map_or_else(|| "response failed".to_string(), |error| error.message),
                ));
            }
            Ok(_) => {}
            Err(error) => return on_progress(Progress::Failed(error.to_string())),
        }
    }
    on_progress(Progress::Done {
        message: (!answer.is_empty()).then_some(answer),
    });
}

fn run(terminal: &mut DefaultTerminal) -> anyhow::Result<()> {
    let mut pane = Pane::default();
    pane.push(Span::styled(
        "tart · Enter sends text · Alt+Enter for newline · Shift+↑ to enter \
        scrollback",
        DIM_STYLE,
    ));

    let api_key = std::env::var("DEEPSEEK_API_KEY")?;
    let config = OpenAIConfig::new()
        .with_api_base("https://api.deepseek.com")
        .with_api_key(&api_key);
    let client = Client::with_config(config);
    let mut history: Vec<InputItem> = vec![input_message(Role::System, SYSTEM.to_string())?];

    // Forward terminal input onto the wake channel so the event loop has a single wait point.
    let (wake, wake_receiver) = mpsc::channel();
    std::thread::spawn({
        let sender = wake.clone();
        move || -> Option<()> {
            loop {
                let event = event::read().ok()?;
                sender.send(Wake::Input(event)).ok()?;
            }
        }
    });

    let mut generating = false;
    let mut quit = false;
    while !quit {
        terminal.draw(|frame| pane.render(frame, frame.area()))?;
        match wake_receiver.recv_timeout(Duration::from_millis(DRAW_INTERVAL_MS)) {
            Ok(Wake::Input(Event::Key(key))) => match on_key(&mut pane, key) {
                Some(PaneEvent::Quit) => quit = true,
                // Copy the selected text when we exit copy mode.
                Some(PaneEvent::Copy(text)) => clipboard::copy(&text)?,
                Some(PaneEvent::Submit(line)) => match line.trim() {
                    "/clear" => pane.clear(),
                    "/quit" | "/exit" => quit = true,
                    // Don't submit text while the model is generating
                    _ if generating => {}
                    _ => {
                        // New response clears the thinking box for the previous one
                        pane.begin_response();
                        history.push(input_message(Role::User, line)?);
                        let request = CreateResponseArgs::default()
                            .model("deepseek-v4-flash")
                            .stream(true)
                            .reasoning(Reasoning {
                                effort: Some(ReasoningEffort::High),
                                summary: None,
                            })
                            .input(InputParam::Items(history.clone()))
                            .build()?;
                        generating = true;
                        let client = client.clone();
                        let sender = wake.clone();
                        // Run the generation on its own thread
                        std::thread::spawn(move || {
                            generate(&client, request, move |progress| {
                                let _ = sender.send(Wake::Generation(progress));
                            });
                        });
                    }
                },
                None => {}
            },
            Ok(Wake::Input(Event::Paste(text))) => pane.on_paste(&text),
            // Resizes are handled at render time (see Pane::render); the redraw
            // timer just loops around and draws again.
            Ok(Wake::Input(_)) | Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => anyhow::bail!("event channel closed"),
            // Update the pane when we recieve new text
            Ok(Wake::Generation(Progress::Thinking(text))) => {
                pane.append_thinking(&Span::styled(text, DIM_STYLE));
            }
            Ok(Wake::Generation(Progress::Answer(text))) => pane.append(&Span::raw(text)),
            // When the model is done, carry the turn into the next request
            Ok(Wake::Generation(Progress::Done { message })) => {
                generating = false;
                if let Some(text) = message {
                    history.push(input_message(Role::Assistant, text)?);
                }
            }
            // If the model *fails* for some reason, show the error.
            Ok(Wake::Generation(Progress::Failed(error))) => {
                generating = false;
                pane.append(&Span::styled(error, DIM_STYLE));
            }
        }
    }
    Ok(())
}

/// Esc leaves copy mode; everything else is the pane's.
fn on_key(pane: &mut Pane, key: KeyEvent) -> Option<PaneEvent> {
    if key.code == KeyCode::Esc && key.modifiers == KeyModifiers::NONE {
        pane.escape();
        return None;
    }
    pane.on_key(key)
}
