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
mod perf;
mod session_picker;
mod tmux_override;

#[cfg(test)]
mod testutil;

use std::io::stdout;
use std::sync::mpsc::{self, RecvTimeoutError};
use std::time::{Duration, Instant};

use ratatui::DefaultTerminal;
use ratatui::crossterm::event::{
    self, DisableBracketedPaste, EnableBracketedPaste, Event, KeyboardEnhancementFlags,
    PopKeyboardEnhancementFlags, PushKeyboardEnhancementFlags,
};
use ratatui::crossterm::execute;
use ratatui::text::Span;

use pane::{DIM_STYLE, Pane, PaneEvent};
use perf::Perf;
use tart_agents::{
    Agent, Progress, ReasoningEffort, SESSIONS_ROOT, Session, Transcript, sandbox::Policy,
};
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
    let root = &SESSIONS_ROOT;
    let cwd = std::env::current_dir()?;
    let mut session = Session::start(root, &cwd);
    // A fresh conversation opens on the configured prompt and instructions.
    let transcript = match agent_config.instructions {
        Some(instructions) => Transcript::with_instructions(instructions)?,
        None => Transcript::new()?,
    };
    install_panic_hook();
    let mut terminal = ratatui::try_init()?;
    execute!(stdout(), EnableBracketedPaste)?;
    execute!(
        stdout(),
        PushKeyboardEnhancementFlags(KeyboardEnhancementFlags::DISAMBIGUATE_ESCAPE_CODES)
    )?;
    // The alternate screen is live, so the conditional rebind takes effect.
    let _tmux = override_shift_up();
    let mut pane = Pane::default();
    pane.set_session_dir(SESSIONS_ROOT.clone(), cwd);
    pane.push(Span::styled(
        format!(
            "tart · {label} · Enter sends text · Alt+Enter for newline · \
            Shift+↑ to enter scrollback"
        ),
        DIM_STYLE,
    ));
    if let Some(tokens) = agent_config.context_tokens {
        pane.set_context_tokens(tokens);
    }
    let result = run(&mut terminal, &mut agent, transcript, &mut session, &mut pane);
    ratatui::try_restore()?;
    execute!(stdout(), PopKeyboardEnhancementFlags)?;
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

/// Parse a `/effort` argument.
fn effort_of(name: &str) -> Option<ReasoningEffort> {
    match name {
        "none" => Some(ReasoningEffort::None),
        "minimal" => Some(ReasoningEffort::Minimal),
        "low" => Some(ReasoningEffort::Low),
        "medium" => Some(ReasoningEffort::Medium),
        "high" => Some(ReasoningEffort::High),
        "xhigh" => Some(ReasoningEffort::Xhigh),
        _ => None,
    }
}

#[allow(
    clippy::too_many_lines,
    reason = "the event loop reads best as one straight-line function"
)]
fn run(
    terminal: &mut DefaultTerminal,
    agent: &mut Agent,
    mut transcript: Transcript,
    session: &mut Session,
    pane: &mut Pane,
) -> anyhow::Result<()> {
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
    let mut perf_on = false;
    let mut perf = Perf::default();
    // Whether Esc cancelled the turn in flight; when it ends, the turn is unwound
    let mut cancelled = false;
    while !quit {
        let t0 = Instant::now();
        let done = terminal.draw(|frame| pane.render(frame, frame.area()))?;
        if perf_on {
            pane.set_perf(Some(perf.frame(t0.elapsed(), done.buffer)));
        } else {
            pane.set_perf(None);
        }
        match wake_receiver.recv_timeout(Duration::from_millis(DRAW_INTERVAL_MS)) {
            Ok(Wake::Input(Event::Key(key))) => match pane.on_key(key) {
                Some(PaneEvent::Quit) => quit = true,
                // Esc with nothing open and running turn aborts the stream and resets.
                Some(PaneEvent::Cancel) => {
                    agent.cancel();
                    cancelled = true;
                }
                // Copy the selected text when we exit copy mode.
                Some(PaneEvent::Copy(text)) => clipboard::copy(&text)?,
                Some(PaneEvent::Submit(line)) => match line.trim() {
                    // Clear the display and the model's memory of the session
                    "/clear" => {
                        pane.clear();
                        transcript.clear();
                        // The abandoned file stays as history; the next turn
                        // starts a fresh one.
                        session.reset();
                    }
                    "/quit" | "/exit" => quit = true,
                    "/perf" => {
                        perf_on = !perf_on;
                        perf = Perf::default();
                    }
                    // A submitted `/resume` line means the chooser was closed;
                    // it opens by itself while the line is being typed.
                    _ if line.trim().starts_with("/resume") => pane.push(Span::styled(
                        "type /resume and pick a session as you type",
                        DIM_STYLE,
                    )),
                    // Set how hard the model reasons.
                    _ if let Some(arg) = line.trim().strip_prefix("/effort") => {
                        let arg = arg.trim();
                        match effort_of(arg) {
                            Some(effort) => {
                                agent.set_reasoning_effort(effort);
                                pane.push(Span::styled(
                                    format!("reasoning effort: {arg}"),
                                    DIM_STYLE,
                                ));
                            }
                            // Bare and unknown arguments both show the usage.
                            None => pane.push(Span::styled(
                                "usage: /effort none|minimal|low|medium|high|xhigh",
                                DIM_STYLE,
                            )),
                        }
                    }
                    _ => {
                        // New response clears the thinking box for the previous one
                        pane.begin_response();
                        transcript.push_user(line)?;
                        pane.set_generating(true);
                        let sender = wake.clone();
                        // The agent loop runs on its own thread
                        agent.spawn(&transcript, move |progress| {
                            let _ = sender.send(Wake::Generation(progress));
                        });
                    }
                },
                // A session picked in the `/resume` chooser swaps the conversation
                //
                // We flush the full abandoned file so we can later resume.
                Some(PaneEvent::Resume(path)) => match session.reopen(&path) {
                    Ok((restored, resumed)) => {
                        let history = restored.replay();
                        *session = resumed;
                        transcript = restored;
                        pane.clear();
                        let name = path.file_stem().map_or_else(
                            || path.display().to_string(),
                            |stem| stem.to_string_lossy().into_owned(),
                        );
                        pane.push(Span::styled(format!("resumed {name}"), DIM_STYLE));
                        pane.extend(history);
                    }
                    // A file too damaged to open just puts the error into our pane.
                    Err(error) => pane.push(Span::styled(error.to_string(), DIM_STYLE)),
                },
                None => {}
            },
            Ok(Wake::Input(Event::Paste(text))) => pane.on_paste(&text),
            // Resizes are handled at render time (see Pane::render); the redraw
            // timer just loops around and draws again.
            Ok(Wake::Input(_)) | Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => anyhow::bail!("event channel closed"),
            // Update the pane as progress arrives.
            Ok(Wake::Generation(progress)) => match &progress {
                // When the turn ends the worker has already recorded the entire turn
                Progress::Done { .. } | Progress::Failed(_) => {
                    pane.set_generating(false);
                    if cancelled {
                        // Reset TUI and context as if the last turn never happened.
                        transcript.drop_last_turn();
                        cancelled = false;
                        pane.cancel_turn();
                    }
                    // A failure also resolves anything still running, then
                    // shows the error.
                    if let Progress::Failed(error) = &progress {
                        pane.fail_pending(error);
                        pane.append(&Span::styled(error.clone(), DIM_STYLE));
                    }
                    session.record(&transcript)?;
                }
                _ => pane.apply(&progress),
            },
        }
    }
    // A quit mid-generation keeps the partial turn unless it was cancelled. In-flight
    // requests (if present) are reconstructed or repaired in the transcript.
    if cancelled {
        transcript.drop_last_turn();
    }
    session.record(&transcript)?;
    Ok(())
}
