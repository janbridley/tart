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
mod model_picker;
mod pane;
mod perf;
mod session_picker;
mod tmux_override;

#[cfg(test)]
mod testutil;

use std::io::stdout;
use std::path::Path;
use std::sync::mpsc::{self, RecvTimeoutError, Sender};
use std::time::{Duration, Instant};

use ratatui::DefaultTerminal;
use ratatui::crossterm::event::{
    self, DisableBracketedPaste, EnableBracketedPaste, Event, KeyboardEnhancementFlags,
    PopKeyboardEnhancementFlags, PushKeyboardEnhancementFlags,
};
use ratatui::crossterm::execute;
use ratatui::text::Span;

use pane::{DIM_STYLE, Mode, Pane, PaneEvent, Wake};
use perf::Perf;
use tart_agents::{
    AGENT_TOOL, Agent, AgentId, Agents, CancelToken, ChatMode, MAIN, Outcome, Progress,
    ReasoningEffort, SESSIONS_ROOT, Session, Transcript, manual_command, prompts, sandbox::Policy,
};

use crate::config::Models;

use tmux_override::{override_shift_up, restore_tmux};

pub const DRAW_INTERVAL_MS: u64 = 100;

fn main() -> anyhow::Result<()> {
    let path = cli::agents_path();
    let config = config::Config::load(&path)?;
    let agent_config = config.default_agent()?;
    let label = agent_config.to_string();
    let context_tokens = agent_config.context_tokens;
    let policy = Policy::new(std::env::current_dir()?)?.exclude_git();
    let (mut agent, current) = agent_config.into_agent(policy);
    let root = &SESSIONS_ROOT;
    let cwd = std::env::current_dir()?;
    let mut session = Session::start(root, &cwd);
    let transcript = Transcript::new()?;
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
    pane.set_control(agent.handle());
    pane.note(format!("tart · {label}"));
    pane.set_context_tokens(context_tokens);
    pane.set_models(config.agents());
    let models = Models { config, current };
    let result = run(
        &mut terminal,
        &mut agent,
        transcript,
        &mut session,
        &mut pane,
        models,
    );
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
    mut models: Models,
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
    // The registry every conversation registers in: the main turn's lever at startup,
    // each subagent's as the model spawns it.
    let agents = Agents::new({
        let sender = wake.clone();
        move |id, progress| {
            let _ = sender.send(Wake::Generation(id, progress));
        }
    });
    agent.set_subagents(std::sync::Arc::new(agents.clone()));
    agents.adopt(agent.handle());
    // A manual command's cancel lever, held while one runs; Esc and quit set it.
    let mut manual_cancel: Option<CancelToken> = None;
    // A plan switch deferred by a running turn, applied when the turn ends.
    let mut pending_plan: Option<bool> = None;
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
                // Esc with nothing open aborts whatever is in flight.
                Some(PaneEvent::Cancel) => {
                    agents.cancel_all();
                    // Esc also cancels a plan switch still waiting for the turn.
                    pending_plan = None;
                    if let Some(token) = &manual_cancel {
                        token.cancel();
                    }
                }
                // Copy the selected text when we exit copy mode.
                Some(PaneEvent::Copy(text)) => clipboard::copy(&text)?,
                // Run the user's command unsandboxed on its own thread.
                Some(PaneEvent::Command(command)) => {
                    pane.manual_running(Some(command.clone()));
                    let token = CancelToken::new();
                    {
                        let token = token.clone();
                        let sender = wake.clone();
                        std::thread::spawn(move || {
                            let framed = manual_command(&command, &token);
                            let _ = sender.send(Wake::Command(framed));
                        });
                    }
                    manual_cancel = Some(token);
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
                    pane.start_turn(agent, &transcript, &wake);
                }
                Some(PaneEvent::Submit(line)) => match line.trim() {
                    // Clear the display and the model's memory of the session
                    "/clear" => {
                        pane.clear();
                        transcript.clear();
                        // The abandoned file stays as history; the next turn
                        // starts a fresh one.
                        session.reset();
                        // Its subagents die with it, reports included.
                        agents.clear();
                    }
                    "/quit" | "/exit" => quit = true,
                    "/perf" => {
                        perf_on = !perf_on;
                        perf = Perf::default();
                    }
                    // Stop one subagent without stopping anything else.
                    _ if let Some(arg) = line.trim().strip_prefix("/stop") => {
                        match arg.trim().parse::<u64>() {
                            Ok(id) => {
                                agents.cancel(AgentId::from(id));
                                pane.note(format!("stop sent to subagent {id}"));
                            }
                            Err(_) => pane.note("usage: /stop <id> · /agents to list"),
                        }
                    }
                    // List the running subagents.
                    "/agents" => {
                        let running = agents.running();
                        let listed = if running.is_empty() {
                            "none".to_string()
                        } else {
                            running
                                .into_iter()
                                .map(|(id, task)| format!("{id}: {task}"))
                                .collect::<Vec<_>>()
                                .join(" · ")
                        };
                        pane.note(format!("subagents: {listed}"));
                    }
                    // A submitted `/resume` line means the chooser was closed;
                    // it opens by itself while the line is being typed.
                    _ if line.trim().starts_with("/resume") => {
                        pane.note("type /resume and pick a session as you type");
                    }
                    // A submitted `/model` line means the chooser was closed;
                    // it opens by itself while the line is being typed.
                    _ if line
                        .trim()
                        .strip_prefix("/model")
                        .is_some_and(|rest| rest.is_empty() || rest.starts_with(' ')) =>
                    {
                        pane.note("type /model and pick an agent as you type");
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
                        pending_plan = pane.set_plan(agent, &mut transcript, on)?;
                    }
                    _ if line.trim().starts_with('/') => pane.note(format!(
                        "unknown command {} · /clear /resume /model /plan /effort /agents /stop /perf /quit",
                        line.split_whitespace().next().unwrap_or_default()
                    )),
                    _ => {
                        // A queued message drains into the record ahead of the
                        // fresh submit, joining its turn.
                        pane.drain_queued(&transcript, &cwd)?;
                        pane.submit_text(&transcript, &line, &cwd)?;
                        pane.start_turn(agent, &transcript, &wake);
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
                        // The resumed conversation's subagents are not its:
                        // they die with the one it replaced.
                        agents.clear();
                        let name = path.file_stem().map_or_else(
                            || path.display().to_string(),
                            |stem| stem.to_string_lossy().into_owned(),
                        );
                        pane.note(format!("resumed {name}"));
                        pane.extend(history);
                    }
                    // A file too damaged to open just puts the error into our pane.
                    Err(error) => pane.note(error.to_string()),
                }
                // An agent picked in the `/model` chooser: swap the endpoint,
                // the model, and the effort for the next turn.
                Some(PaneEvent::Model(choice)) => models.swap(&choice, agent, pane),
                None => {}
            },
            Ok(Wake::Input(Event::Paste(text))) => pane.on_paste(&text),
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
            Ok(Wake::Generation(id, progress)) => {
                let ping = Instant::now();
                if id == MAIN {
                    on_main_event(
                        pane,
                        agent,
                        &mut transcript,
                        session,
                        &agents,
                        &cwd,
                        &wake,
                        &mut pending_plan,
                        &progress,
                    )?;
                } else {
                    on_child_event(pane, agent, &agents, &transcript, &wake, id, &progress)?;
                }
                work += ping.elapsed();
            }
            // Resizes are handled at render time (see Pane::render); the redraw
            // timer just loops around and draws again.
            Ok(Wake::Input(_)) | Err(RecvTimeoutError::Timeout) => {}
        }
    }
    // A quit mid-generation keeps the partial turn; in-flight requests (if
    // any) are reconstructed or repaired in the transcript.
    session.record(&transcript)?;
    // A quit mid-manual-command kills it rather than orphaning the group.
    if let Some(token) = manual_cancel {
        token.cancel();
    }
    Ok(())
}

/// Apply one MAIN event: the pane, the record, the session. A terminal event
/// ends the turn, delivers pending subagent reports, and requeues.
#[allow(
    clippy::too_many_arguments,
    reason = "the event touches every arm of the loop's state"
)]
fn on_main_event(
    pane: &mut Pane,
    agent: &mut Agent,
    transcript: &mut Transcript,
    session: &mut Session,
    agents: &Agents,
    cwd: &Path,
    wake: &Sender<Wake>,
    pending_plan: &mut Option<bool>,
    progress: &Progress,
) -> anyhow::Result<()> {
    // When the turn ends the worker has already recorded the entire turn
    if progress.is_terminal() {
        pane.set_generating(false);
        // A finished plan in plan mode is ready for Enter to approve
        pane.set_plan_ready(matches!(progress, Progress::Done { .. }));
        // A plan switch queued mid-turn takes effect now, ahead of
        // any queued message that starts the next turn.
        if let Some(on) = pending_plan.take() {
            pane.set_plan(agent, transcript, on)?;
        }
        // A failure also resolves anything still running, then
        // shows the error.
        if let Progress::Failed(error) = progress {
            pane.fail_pending(error);
            pane.append_span(&Span::styled(error.clone(), DIM_STYLE));
        }
        let requeued = if matches!(progress, Progress::Done { .. } | Progress::Cancelled) {
            // Pending subagent reports deliver together as
            // one turn; a queued user message follows when
            // that turn ends.
            pane.deliver_reports(agents, agent, transcript, wake)?
                || pane.requeue(agent, transcript, cwd, wake)?
        } else {
            pane.spill_queued();
            false
        };
        // A cancelled turn keeps its streamed partial message + notify.
        // (The pane already spilled any queued message when
        // Esc landed. A cancel is a take-back.)
        if matches!(progress, Progress::Cancelled) && !requeued {
            pane.note("⎋ cancelled");
        }
        session.record(transcript)?;
    } else {
        pane.apply(progress);
    }
    Ok(())
}

/// Apply one child event: its box, its spend, and — once it has ended with no
/// turn running — the delivery of its report.
fn on_child_event(
    pane: &mut Pane,
    agent: &Agent,
    agents: &Agents,
    transcript: &Transcript,
    wake: &Sender<Wake>,
    id: AgentId,
    progress: &Progress,
) -> anyhow::Result<()> {
    match progress {
        Progress::ToolStart { name, arguments, .. } => {
            if name == AGENT_TOOL {
                pane.start_agent(id, arguments);
            } else {
                pane.touch_agent(id, name, arguments);
            }
        }
        _ if progress.is_terminal() => {
            // The peek resolves the box and queues the id; the
            // claim happens at delivery, so a `wait` inside this
            // turn can still take the report for itself.
            match agents.outcome(id) {
                Some(outcome) => {
                    pane.report(id);
                    pane.finish_agent(
                        id,
                        outcome.report(),
                        matches!(outcome, Outcome::Done(_)).then_some(0),
                    );
                }
                None => pane.finish_agent(id, "report delivered through wait".to_string(), Some(0)),
            }
            // Idle: the reports start their turn now. Busy: the
            // running turn's end delivers them.
            if !pane.is_generating() {
                pane.deliver_reports(agents, agent, transcript, wake)?;
            }
        }
        // A child's spend meters into the status line's agent total.
        Progress::Usage { output, .. } => pane.add_child_output(*output),
        _ => {}
    }
    Ok(())
}
