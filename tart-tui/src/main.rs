//! A terminal chat front end for the `tart-ai` client.
//!
//! ```text
//! │ transcript (wraps, auto-tails)          │
//! │ ❯ hello                                 │
//! ├─────────────────────────────────────────┤
//! │ ❯ ▊ prompt, grows with its content      │
//! └─────────────────────────────────────────┘
//! ```

mod file_mentions;
mod pane;
mod tmux_override;

#[cfg(test)]
mod testutil;

use std::io::stdout;
use std::sync::mpsc::{self, RecvTimeoutError};
use std::time::Duration;

use ratatui::DefaultTerminal;
use ratatui::crossterm::event::{
    self, DisableBracketedPaste, EnableBracketedPaste, Event, KeyCode, KeyEvent, KeyModifiers,
};
use ratatui::crossterm::execute;
use ratatui::text::Span;

use pane::{DIM_STYLE, Pane, PaneEvent};
use tmux_override::{override_shift_up, restore_tmux};

use tart_ai::{
    ContextHistory, ReasoningEffort,
    openai::{ChatCompletionsClient, Delta, GenerationEvent, Message},
};

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
    Generation(GenerationEvent),
}

fn run(terminal: &mut DefaultTerminal) -> anyhow::Result<()> {
    let mut pane = Pane::default();
    pane.push(Span::styled(
        "tart · Enter sends text · Alt+Enter for newline · Shift+↑ to enter \
        scrollback",
        DIM_STYLE,
    ));

    let api_key = std::env::var("DEEPSEEK_API_KEY")?;
    let client = ChatCompletionsClient::new(
        "https://api.deepseek.com/chat/completions",
        api_key,
        "deepseek-v4-flash",
    )
    .reasoning_effort(ReasoningEffort::Max);
    let mut history = ContextHistory::from(Message::system());

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

    let mut quit = false;
    while !quit {
        terminal.draw(|frame| pane.render(frame, frame.area()))?;
        #[allow(
            clippy::match_same_arms,
            reason = "different wake sources that happen to need no handling"
        )]
        match wake_receiver.recv_timeout(Duration::from_millis(DRAW_INTERVAL_MS)) {
            Ok(Wake::Input(Event::Key(key))) => match on_key(&mut pane, key) {
                Some(PaneEvent::Quit) => quit = true,
                Some(PaneEvent::Submit(line)) => match line.trim() {
                    "/clear" => pane.clear(),
                    "/quit" | "/exit" => quit = true,
                    // Don't submit text while the agent is generating code
                    _ if client.is_generating() => {}
                    _ => {
                        // New response clears the thinking box for the previous one
                        pane.begin_response();
                        history.append_message(Message::user(line));
                        let sender = wake.clone();
                        client.spawn(0, &history, move |event| {
                            // Run the generation on its own thread
                            let _ = sender.send(Wake::Generation(event));
                        })?;
                    }
                },
                None => {}
            },
            Ok(Wake::Input(Event::Paste(text))) => pane.on_paste(&text),
            // Resizes are handled at render time (see Pane::render); the redraw
            // timer just loops around and draws again.
            Ok(Wake::Input(_)) | Err(RecvTimeoutError::Timeout) => {}
            // Update the pane when we recieve new text
            Ok(Wake::Generation(GenerationEvent::Delta(delta))) => match delta {
                // Dim the chain-of-thought preceding the answer
                Delta::Thinking(text) => pane.append_thinking(&Span::styled(text, DIM_STYLE)),
                Delta::Answer(text) => pane.append(&Span::raw(text)),
                // `Delta` is non-exhaustive, nothing else exists yet.
                _ => {}
            },

            // When the model is done generating, record data
            Ok(Wake::Generation(GenerationEvent::Done { message, usage })) => {
                if let Some(usage) = usage {
                    history.record_usage(usage);
                }
                if let Some(message) = message {
                    history.append_message(message);
                }
            }
            // If the model *fails* for some reason, show the error.
            Ok(Wake::Generation(GenerationEvent::Failed(error))) => {
                pane.append(&Span::styled(error.to_string(), DIM_STYLE));
            }
            // `GenerationEvent` is non-exhaustive, nothing else exists yet.
            Ok(Wake::Generation(_)) => {}
            Err(RecvTimeoutError::Disconnected) => anyhow::bail!("event channel closed"),
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
