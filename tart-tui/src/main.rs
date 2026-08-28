//! A terminal chat front end for the tart agent harness.
//!
//! ```text
//! │ transcript (wraps, auto-tails)          │
//! │ ❯ hello                                 │
//! ├─────────────────────────────────────────┤
//! │ ❯ ▊ prompt, grows with its content      │
//! └─────────────────────────────────────────┘
//! ```

mod attachments;
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

use pane::{DIM_STYLE, Mode, Pane, PaneEvent};
use perf::Perf;
use tart_agents::{
    Agent, CancelToken, ChatMode, Progress, ReasoningEffort, SESSIONS_ROOT, Session, Transcript,
    manual_command, prompts, sandbox::Policy,
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
    pane.set_control(agent.control());
    pane.note(format!("tart · {label}"));
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
    /// A finished manual command's framed output.
    Command(String),
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

/// The user message recording one manual command and its framed output, so the
/// model reads the run like pasted text rather than a tool result.
///
/// The fence is four backticks long, so a fence marker inside a command or its
/// output cannot close it early.
fn manual_message(command: &str, framed: &str) -> String {
    format!(
        "I ran this command myself, outside the sandbox:\n\n\
         ````console\n$ {command}\n{}\n````",
        framed.trim_end()
    )
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
    // The sandbox's grant root, for deciding which mentions need attaching.
    let cwd = std::env::current_dir()?;
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
    let control = agent.control();
    // A manual command's cancel lever, held while one runs; Esc and quit set it.
    let mut manual_cancel: Option<(CancelToken, std::thread::JoinHandle<()>)> = None;
    // State-mutation time since the last frame.
    let mut work = Duration::ZERO;
    while !quit {
        let t0 = Instant::now();
        let done = terminal.draw(|frame| pane.render(frame, frame.area()))?;
        let frame = t0.elapsed() + std::mem::take(&mut work);
        if perf_on {
            pane.set_perf(Some(perf.frame(frame, done.buffer)));
        } else {
            pane.set_perf(None);
        }
        match wake_receiver.recv_timeout(Duration::from_millis(DRAW_INTERVAL_MS)) {
            Ok(Wake::Input(Event::Key(key))) => match pane.on_key(key) {
                Some(PaneEvent::Quit) => quit = true,
                // Esc with nothing open aborts whatever is in flight: the turn
                // (a no-op when idle) and any manual command.
                Some(PaneEvent::Cancel) => {
                    control.cancel();
                    if let Some((token, _)) = &manual_cancel {
                        token.cancel();
                    }
                }
                // Copy the selected text when we exit copy mode.
                Some(PaneEvent::Copy(text)) => clipboard::copy(&text)?,
                // Run the user's command unsandboxed on its own thread.
                Some(PaneEvent::Command(command)) => {
                    pane.manual_running(Some(command.clone()));
                    let token = CancelToken::new();
                    let runner = {
                        let token = token.clone();
                        let sender = wake.clone();
                        std::thread::spawn(move || {
                            let framed = manual_command(&command, &token);
                            let _ = sender.send(Wake::Command(framed));
                        })
                    };
                    manual_cancel = Some((token, runner));
                }
                // Shift+Tab: toggle plan mode, exactly as `/plan` does.
                Some(PaneEvent::Plan) => {
                    let on = !pane.is_plan();
                    pane.set_plan(agent, &mut transcript, on)?;
                }
                // Enter approved the drafted plan: leave plan mode and start
                // the implementing turn, which may now write.
                Some(PaneEvent::Approve) => {
                    pane.set_mode(Mode::Default);
                    agent.set_mode(ChatMode::Default);
                    transcript.set_reminder(None)?;
                    pane.note("plan approved · implementing");
                    pane.echo(prompts::PLAN_APPROVAL);
                    transcript.push_user(prompts::PLAN_APPROVAL.to_string())?;
                    pane.begin_response();
                    pane.set_generating(true);
                    let sender = wake.clone();
                    agent.spawn(&transcript, move |progress| {
                        let _ = sender.send(Wake::Generation(progress));
                    });
                }
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
                    _ if line.trim().starts_with("/resume") => {
                        pane.note("type /resume and pick a session as you type");
                    }
                    // Set how hard the model reasons.
                    _ if let Some(arg) = line.trim().strip_prefix("/effort") => {
                        let arg = arg.trim();
                        match effort_of(arg) {
                            Some(effort) => {
                                agent.set_reasoning_effort(effort);
                                pane.note(format!("reasoning effort: {arg}"));
                            }
                            // Bare and unknown arguments both show the usage.
                            None => pane.note("usage: /effort none|minimal|low|medium|high|xhigh"),
                        }
                    }
                    // Toggle plan mode: read-only research and planning.
                    "/plan" => {
                        let on = !pane.is_plan();
                        pane.set_plan(agent, &mut transcript, on)?;
                    }
                    _ => {
                        // Steering message is emptied on the same iteration it sends.
                        if let Some(text) = control.take_steer() {
                            pane.echo(&text);
                            let (message, notes) = attachments::attach_mentions(&text, &cwd);
                            for note in notes {
                                pane.note(note);
                            }
                            transcript.push_user(message)?;
                        }
                        // New response clears the thinking box for the previous one
                        pane.begin_response();
                        let (message, notes) = attachments::attach_mentions(&line, &cwd);
                        for note in notes {
                            pane.note(note);
                        }
                        transcript.push_user(message)?;
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
                        // The restored record carries no reminder; plan mode
                        // outlives a resume, so re-arm it on the new transcript.
                        transcript
                            .set_reminder(pane.is_plan().then_some(prompts::PLAN_REMINDER))?;
                        pane.clear();
                        let name = path.file_stem().map_or_else(
                            || path.display().to_string(),
                            |stem| stem.to_string_lossy().into_owned(),
                        );
                        pane.note(format!("resumed {name}"));
                        pane.extend(history);
                    }
                    // A file too damaged to open just puts the error into our pane.
                    Err(error) => pane.note(error.to_string()),
                },
                None => {}
            },
            Ok(Wake::Input(Event::Paste(text))) => pane.on_paste(&text),
            // Resizes are handled at render time (see Pane::render); the redraw
            // timer just loops around and draws again.
            Ok(Wake::Input(_)) | Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => anyhow::bail!("event channel closed"),
            // A manual command finished: echo it, then record the exchange as user msg
            Ok(Wake::Command(framed)) => {
                manual_cancel = None;
                if let Some(command) = pane.manual_done(&framed) {
                    transcript.push_user(manual_message(&command, &framed))?;
                    session.record(&transcript)?;
                }
            }
            // Update the pane as progress arrives and time it into `work`.
            Ok(Wake::Generation(progress)) => {
                let ping = Instant::now();
                match &progress {
                    // When the turn ends the worker has already recorded the entire turn
                    Progress::Done { .. } | Progress::Failed(_) | Progress::Cancelled => {
                        pane.set_generating(false);
                        // A finished plan in plan mode is ready for Enter to approve
                        pane.set_plan_ready(matches!(&progress, Progress::Done { .. }));
                        // A failure also resolves anything still running, then
                        // shows the error.
                        if let Progress::Failed(error) = &progress {
                            pane.fail_pending(error);
                            pane.append_span(&Span::styled(error.clone(), DIM_STYLE));
                        }
                        // A cancelled turn keeps its streamed partial message + notify.
                        // (The pane already spilled any queued steering when
                        // Esc landed. A cancel is a take-back.)
                        if matches!(progress, Progress::Cancelled) {
                            pane.note("⎋ cancelled");
                        } else if control.steering().is_some() {
                            // A steer that outlived its turn starts a fresh one
                            if matches!(progress, Progress::Done { .. }) {
                                let text = control.take_steer().expect("checked above");
                                pane.echo(&text);
                                let (message, notes) = attachments::attach_mentions(&text, &cwd);
                                for note in notes {
                                    pane.note(note);
                                }
                                transcript.push_user(message)?;
                                pane.begin_response();
                                pane.set_generating(true);
                                let sender = wake.clone();
                                agent.spawn(&transcript, move |progress| {
                                    let _ = sender.send(Wake::Generation(progress));
                                });
                            } else {
                                pane.spill_steer();
                            }
                        }
                        session.record(&transcript)?;
                    }
                    _ => pane.apply(&progress),
                }
                work += ping.elapsed();
            }
        }
    }
    // A quit mid-generation keeps the partial turn; in-flight requests (if
    // any) are reconstructed or repaired in the transcript.
    session.record(&transcript)?;
    // A quit mid-manual-command kills it rather than orphaning the group.
    if let Some((token, runner)) = manual_cancel {
        token.cancel();
        let _ = runner.join();
    }
    Ok(())
}
