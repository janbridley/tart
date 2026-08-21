//! A terminal chat front end for the tart agent harness.
//!
//! ```text
//! │ transcript (wraps, auto-tails)          │
//! │ ❯ hello                                 │
//! ├─────────────────────────────────────────┤
//! │ ❯ ▊ prompt, grows with its content      │
//! └─────────────────────────────────────────┘
//! ```

mod cli;
mod clipboard;
mod config;
mod file_mentions;
mod keybinds;
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
use tart_agents::{Agent, Progress, Transcript, sandbox::Policy};
use tmux_override::{override_shift_up, restore_tmux};

pub const DRAW_INTERVAL_MS: u64 = 100;

fn main() -> anyhow::Result<()> {
    let path = cli::agents_path();
    let agent_config = config::Config::load(&path)?.default_agent()?;
    let label = agent_config.to_string();
    let policy = Policy::new(std::env::current_dir()?)?.exclude_git();
    let mut agent = Agent::new(
        agent_config.base_url,
        agent_config.api_key,
        agent_config.model,
        policy,
    );
    if let Some(effort) = agent_config.effort {
        agent = agent.reasoning_effort(effort);
    }
    let transcript = match agent_config.instructions {
        Some(instructions) => Transcript::with_instructions(instructions)?,
        None => Transcript::new()?,
    };

    install_panic_hook();
    let mut terminal = ratatui::try_init()?;
    execute!(stdout(), EnableBracketedPaste)?;
    // The alternate screen is live, so the conditional rebind takes effect.
    let _tmux = override_shift_up();
    let result = run(&mut terminal, &agent, transcript, &label);
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

fn run(
    terminal: &mut DefaultTerminal,
    agent: &Agent,
    mut transcript: Transcript,
    label: &str,
) -> anyhow::Result<()> {
    let mut pane = Pane::default();
    pane.push(Span::styled(
        format!(
            "tart · {label} · Enter sends text · Alt+Enter for newline · \
            Shift+↑ to enter scrollback"
        ),
        DIM_STYLE,
    ));

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
        #[allow(
            clippy::match_same_arms,
            reason = "different wake sources that happen to need no handling"
        )]
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
                        transcript.push_user(line)?;
                        generating = true;
                        let sender = wake.clone();
                        // The agent loop runs on its own thread
                        agent.spawn(&transcript, move |progress| {
                            let _ = sender.send(Wake::Generation(progress));
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
            // Show what the model ran, and what came back, dimmed like thinking
            Ok(Wake::Generation(Progress::Command(command))) => {
                pane.push(Span::styled(format!("$ {command}"), DIM_STYLE));
            }
            Ok(Wake::Generation(Progress::CommandOutput(output))) => {
                for line in output.split('\n') {
                    pane.push(Span::styled(line.to_string(), DIM_STYLE));
                }
            }
            // When the model is done, carry the turn into the next request
            Ok(Wake::Generation(Progress::Done { message })) => {
                generating = false;
                if let Some(text) = message {
                    transcript.push_assistant(text)?;
                }
            }
            // If the model *fails* for some reason, show the error.
            Ok(Wake::Generation(Progress::Failed(error))) => {
                generating = false;
                pane.append(&Span::styled(error, DIM_STYLE));
            }
            // `Progress` is non-exhaustive; later variants need no handling yet.
            Ok(Wake::Generation(_)) => {}
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
