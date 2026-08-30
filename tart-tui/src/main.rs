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
use std::time::{Duration, Instant};

use futures::executor::LocalPool;
use futures::select;
use futures::stream::StreamExt;
use ratatui::DefaultTerminal;
use ratatui::crossterm::event::{
    DisableBracketedPaste, EnableBracketedPaste, Event, EventStream, KeyboardEnhancementFlags,
    PopKeyboardEnhancementFlags, PushKeyboardEnhancementFlags,
};
use ratatui::crossterm::execute;
use ratatui::text::Span;

use pane::{DIM_STYLE, Mode, Pane, PaneEvent};
use perf::Perf;
use tart_agents::{
    Agent, CancelToken, ChatMode, Progress, ReasoningEffort, SESSIONS_ROOT, Session, Transcript,
    TurnTask, manual_command, prompts, sandbox::Policy,
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
    // The session's progress stream, carrying every turns' events.
    let (progress, progress_rx) = futures::channel::mpsc::unbounded::<Progress>();
    // A finished manual command's output, framed with context.
    let (manual_done, manual_rx) = futures::channel::mpsc::unbounded::<String>();

    // We don't want to restrict generation to a single thread, so we create an owned
    // `smol` executor with an extra driver thread shared by `spawn` and `run`.
    let executor = std::sync::Arc::new(smol::Executor::new());
    let _driver = {
        let executor = std::sync::Arc::clone(&executor);
        std::thread::spawn(move || {
            futures::executor::block_on(executor.run(std::future::pending::<()>()));
        })
    };
    let drive = move |task: TurnTask| {
        executor.spawn(task).detach();
    };

    let mut quit = false;
    let mut perf_on = false;
    let mut perf = Perf::default();
    // A manual command's cancel lever, held while one runs; Esc and quit set it.
    let mut manual_cancel: Option<(CancelToken, std::thread::JoinHandle<()>)> = None;
    // A plan switch deferred by a running turn, applied when the turn ends.
    let mut pending_plan: Option<bool> = None;
    // State-mutation time since the last frame.
    let mut work = Duration::ZERO;

    // One wait point over every wake source.
    let mut pool = LocalPool::new();
    pool.run_until(async {
        let mut input = EventStream::new().fuse();
        let mut progress_rx = progress_rx.fuse();
        let mut manual_rx = manual_rx.fuse();
        while !quit {
            let t0 = Instant::now();
            let done = terminal.draw(|frame| pane.render(frame, frame.area()))?;
            let frame = t0.elapsed() + std::mem::take(&mut work);
            if perf_on {
                pane.set_perf(Some(perf.frame(frame, done.buffer)));
            } else {
                pane.set_perf(None);
            }
            // The tick resets every iteration to ensure we don't accumulate timers
            // while streaming.
            let mut ticked =
                futures::FutureExt::fuse(smol::Timer::after(Duration::from_millis(
                    DRAW_INTERVAL_MS,
                )));
            select! {
                ev = input.select_next_some() => {
                    let ev = ev?;
                    if let Event::Key(key) = ev {
                        match pane.on_key(key) {
                            Some(PaneEvent::Quit) => quit = true,
                            // Esc with nothing open aborts whatever is in flight: the turn
                            // (a no-op when idle) and any manual command.
                            Some(PaneEvent::Cancel) => {
                                pane.cancel_turn();
                                // Esc also cancels a plan switch still waiting for the turn.
                                pending_plan = None;
                                if let Some((token, _)) = &manual_cancel {
                                    token.cancel();
                                }
                            }
                            // Copy the selected text when we exit copy mode.
                            Some(PaneEvent::Copy(text)) => clipboard::copy(&text)?,
                            // Run the user's command unsandboxed on its own thread
                            Some(PaneEvent::Command(command)) => {
                                pane.manual_running(Some(command.clone()));
                                let token = CancelToken::new();
                                let runner = {
                                    let token = token.clone();
                                    let sender = manual_done.clone();
                                    std::thread::spawn(move || {
                                        let framed = manual_command(&command, &token);
                                        let _ = sender.unbounded_send(framed);
                                    })
                                };
                                manual_cancel = Some((token, runner));
                            }
                            // Shift+Tab: toggle plan mode, exactly as `/plan` does.
                            Some(PaneEvent::Plan) => {
                                let on = !pane.is_plan();
                                pending_plan = pane.set_plan(agent, &mut transcript, on)?;
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
                                pane.start_turn(agent, &transcript, &progress, &drive);
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
                                        None => {
                                            pane.note(
                                                "usage: /effort none|minimal|low|medium|high|xhigh",
                                            );
                                        }
                                    }
                                }
                                // Toggle plan mode: read-only research and planning.
                                "/plan" => {
                                    let on = !pane.is_plan();
                                    pending_plan = pane.set_plan(agent, &mut transcript, on)?;
                                }
                                _ => {
                                    // Steering message is emptied on the same iteration it sends.
                                    if let Some(text) = pane.take_steer() {
                                        pane.echo(&text);
                                        pane.submit_text(&transcript, &text, &cwd)?;
                                    }
                                    pane.submit_text(&transcript, &line, &cwd)?;
                                    pane.start_turn(agent, &transcript, &progress, &drive);
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
                        }
                    } else if let Event::Paste(text) = ev {
                        pane.on_paste(&text);
                    }
                    // Resizes and everything else are handled at render time
                    // (see Pane::render); the tick just loops around and draws.
                }
                // A manual command finished: echo it, then record the exchange as user msg
                framed = manual_rx.select_next_some() => {
                    manual_cancel = None;
                    if let Some(command) = pane.manual_done(&framed) {
                        transcript.push_user(manual_message(&command, &framed))?;
                        session.record(&transcript)?;
                    }
                }
                // Update the pane as progress arrives and time it into `work`.
                event = progress_rx.select_next_some() => {
                    let ping = Instant::now();
                    match &event {
                        // When the turn ends the worker has already recorded the entire turn
                        Progress::Done { .. } | Progress::Failed(_) | Progress::Cancelled => {
                            pane.set_generating(false);
                            // A finished plan in plan mode is ready for Enter to approve
                            pane.set_plan_ready(matches!(&event, Progress::Done { .. }));
                            // A plan switch queued mid-turn takes effect now, ahead of
                            // any steer that starts the next turn.
                            if let Some(on) = pending_plan.take() {
                                pane.set_plan(agent, &mut transcript, on)?;
                            }
                            // A failure also resolves anything still running, then
                            // shows the error.
                            if let Progress::Failed(error) = &event {
                                pane.fail_pending(error);
                                pane.append_span(&Span::styled(error.clone(), DIM_STYLE));
                            }
                            // A cancelled turn keeps its streamed partial message + notify.
                            // (The pane already spilled any queued steering when
                            // Esc landed. A cancel is a take-back.)
                            if matches!(event, Progress::Cancelled) {
                                pane.note("⎋ cancelled");
                            } else if pane.steering().is_some() {
                                // A steer that outlived its turn starts a fresh one
                                if matches!(event, Progress::Done { .. }) {
                                    let text = pane.take_steer().expect("checked above");
                                    pane.echo(&text);
                                    pane.submit_text(&transcript, &text, &cwd)?;
                                    pane.start_turn(agent, &transcript, &progress, &drive);
                                } else {
                                    pane.spill_steer();
                                }
                            }
                            session.record(&transcript)?;
                        }
                        _ => pane.apply(&event),
                    }
                    work += ping.elapsed();
                }
                // Redraw on the next loop-around if ticked.
                _ = ticked => {}
            }
        }
        Ok::<(), anyhow::Error>(())
    })?;
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
