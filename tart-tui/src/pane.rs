//! `Pane` object stores data and rendering logic for the terminal interface.

mod copy;
mod editor;
mod markdown;
mod transcript;
mod wrap;

use ratatui::buffer::Buffer;
use ratatui::crossterm::event::{KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::{Frame, symbols};
use std::path::PathBuf;
use std::time::Instant;
use tart_agents::{Progress, TurnControl};
use unicode_segmentation::UnicodeSegmentation;

pub(crate) use editor::{Editor, g_to_byte, graphemes};

use crate::clipboard::Selection;
use crate::file_mentions::{self, FilePopup};
use crate::session_picker::{SessionPopup, derive_query as session_query};
use copy::{CopyCursor, clamp_cell, moved, window_top};
use transcript::Transcript;
use wrap::wrap_draft;

pub const PROMPT: &str = "❯ ";
/// Cells before the editor starts (the prompt symbol's width).
const GUTTER: u16 = 2;

pub const DIM_STYLE: Style = Style::new().fg(Color::DarkGray);
/// The highlight for the transcript's actionable hints.
pub(crate) const HIGHLIGHT_STYLE: Style = Style::new().fg(Color::Blue);
const PROMPT_STYLE: Style = Style::new().fg(Color::Cyan).add_modifier(Modifier::BOLD);
/// The prompt glyph while the `!` manual-command mode is on: also 2 cells.
const BANG_PROMPT: &str = "! ";
/// The bang glyph's color, matching the frame it announces.
const BANG_STYLE: Style = Style::new().fg(Color::Magenta).add_modifier(Modifier::BOLD);
/// The rules' color while the mode is on or a run is in flight; see [`reframe`].
const BANG_RULE: Color = Color::Magenta;
/// The copy cursor and the editor caret are the cell under them, inverted.
const CURSOR_STYLE: Style = Style::new().add_modifier(Modifier::REVERSED);
/// Frames for the statusline spinner.
const SPINNER_FRAMES: [&str; 6] = ["·  ", "·· ", "···", " ··", "  ·", "   "];
/// Milliseconds per spinner frame.
const SPINNER_MS: u128 = 200;

/// Key results the pane cannot handle itself.
#[derive(Debug, PartialEq)]
pub enum PaneEvent {
    Submit(String),
    /// A manual command the user types, ready to run unsandboxed.
    Command(String),
    /// Text chosen in copy mode, ready for the clipboard.
    Copy(String),
    /// A session picked in the `/resume` chooser, ready to swap to.
    Resume(PathBuf),
    /// Esc with nothing open, with a turn or a manual command in flight.
    Cancel,
    Quit,
}

/// The popup over the prompt, when one is open.
pub(crate) enum Popup {
    /// The `@file` typeahead over the working directory.
    Files(FilePopup),
    /// The `/resume` chooser over this project's sessions.
    Sessions(SessionPopup),
}

impl Popup {
    /// The list machinery under this popup kind.
    fn list(&mut self) -> &mut FilePopup {
        match self {
            Self::Files(popup) => popup,
            Self::Sessions(sessions) => &mut sessions.popup,
        }
    }

    /// Move the highlight up one row.
    fn select_prev(&mut self) {
        self.list().select_prev();
    }

    /// Move the highlight down one row.
    fn select_next(&mut self) {
        self.list().select_next();
    }

    /// Point the popup at a new query, refiltering when it changed.
    fn set_query(&mut self, query: String) {
        self.list().set_query(query);
    }
}

/// One finished response's token usage, as shown on the status line.
#[derive(Clone, Copy)]
struct Usage {
    /// All input tokens, cached included.
    input: u64,
    /// The input tokens served from the prompt cache.
    cached: u64,
    /// The tokens the model generated.
    output: u64,
}

/// One manual command in flight.
struct ManualCommand {
    /// The text of the command
    command: String,
    /// A timestamp for when the command started
    started: Instant,
}

/// The TUI interface.
#[derive(Default)]
pub struct Pane {
    /// Input interface for the `Pane`.
    prompt: Editor,
    /// Conversation history.
    transcript: Transcript,
    /// A `CopyCursor` in copymode, or `None` otherwise.
    copy: Option<CopyCursor>,
    /// The `@file` typeahead or the `/resume` chooser, while either is present
    popup: Option<Popup>,
    /// Where `/resume` lists sessions from: the sessions root and project.
    session_dir: Option<(PathBuf, PathBuf)>,
    /// The `/perf` stats line, shown on the bottom rule row; `None` when off.
    perf: Option<String>,
    /// The last response's token usage.
    usage: Option<Usage>,
    /// The model's context window, from the agents file.
    context_tokens: Option<u64>,
    /// When the generating turn started; `None` when the model is idle.
    /// Enter keeps the draft while set, and the status rule's spinner runs
    /// off the elapsed time for a steady frame rate.
    spin: Option<Instant>,
    /// Whether the `!` manual-command mode is enabled
    bang: bool,
    /// A manual command in flight, holding the purple frame until its output lands.
    manual: Option<ManualCommand>,
    /// Control parameters for steering or interrupting the model.
    control: TurnControl,
}

impl Pane {
    pub fn on_key(&mut self, key: KeyEvent) -> Option<PaneEvent> {
        // Press and Repeat drive the app, Release is not important/useful.
        if !matches!(key.kind, KeyEventKind::Press | KeyEventKind::Repeat) {
            return None;
        }
        let event = self.route(key);
        // The popups track the draft, except in copy mode.
        if self.copy.is_none() {
            self.sync_popup(Some(&key));
        }
        event
    }

    /// Update a popup based on input keystrokes (e.g. `@` opens, `Esc` closes).
    fn sync_popup(&mut self, key: Option<&KeyEvent>) {
        // Bang mode completes the argument under the caret as a file, bash's
        // default for arguments, so neither prefix popup applies: an `@` or a
        // leading `/resume` is just text in a command.
        if self.bang {
            // An open list tracks the word as it changes and closes when the
            // word ends; only Tab opens one (see the Tab arm in `route`).
            let query =
                file_mentions::derive_argument(&self.prompt).filter(|(query, _)| !query.is_empty());
            file_mentions::update(&mut self.popup, query, false);
            return;
        }
        match session_query(&self.prompt) {
            Some(query) => {
                if let Some(popup @ Popup::Sessions(_)) = &mut self.popup {
                    popup.set_query(query);
                } else if key.is_some_and(|key| key.code != KeyCode::Esc) {
                    self.popup = self.session_dir.as_ref().map(|(root, project)| {
                        Popup::Sessions(SessionPopup::new(root, project, query))
                    });
                }
            }
            None => {
                file_mentions::update(
                    &mut self.popup,
                    file_mentions::derive_query(&self.prompt),
                    key.is_some_and(file_mentions::rearm),
                );
            }
        }
    }

    /// One key the open popup owns: arrows move the highlight, Tab/Enter accepts.
    fn popup_key(&mut self, key: KeyEvent) -> Option<PaneEvent> {
        let popup = self.popup.as_mut()?;
        match key.code {
            KeyCode::Up => popup.select_prev(),
            KeyCode::Down => popup.select_next(),
            KeyCode::Tab | KeyCode::Enter => return self.accept_popup(),
            _ => {}
        }
        None
    }

    /// Close the open popup and apply its highlighted row.
    fn accept_popup(&mut self) -> Option<PaneEvent> {
        match self.popup.take() {
            Some(Popup::Files(popup)) => {
                // A mention replaces its `@` word, or a completion its argument.
                if self.bang {
                    popup.accept_argument(&mut self.prompt);
                } else {
                    popup.accept(&mut self.prompt);
                }
            }
            Some(Popup::Sessions(sessions)) => {
                if self.spin.is_none()
                    && let Some(path) = sessions.selected_path()
                {
                    self.clear_prompt();
                    return Some(PaneEvent::Resume(path));
                }
            }
            None => {}
        }
        None
    }

    /// Key routing without the popup bookkeeping.
    fn route(&mut self, key: KeyEvent) -> Option<PaneEvent> {
        // Copy mode takes every key first, so the draft is immutable while scrolling
        if let Some(cursor) = self.copy {
            match key.code {
                // q or Esc leaves copy mode.
                KeyCode::Char('q' | 'Q') | KeyCode::Esc => self.copy = None,
                // Space (unconditionally) begins a selection at the current position
                KeyCode::Char(' ') => {
                    self.copy = Some(CopyCursor {
                        anchor: Some((cursor.row, cursor.col)),
                        ..cursor
                    });
                }
                // Enter copies the selection to clipboard and exits copy mode.
                KeyCode::Enter => {
                    let text = Selection::between(cursor.anchor, (cursor.row, cursor.col))
                        .map(|selection| selection.text(self.transcript.rows()))
                        .filter(|text| !text.is_empty());
                    self.copy = None;
                    if let Some(text) = text {
                        return Some(PaneEvent::Copy(text));
                    }
                }
                _ => self.copy = Some(moved(self.transcript.rows(), cursor, key.code)),
            }
            return None;
        }
        if key.modifiers.contains(KeyModifiers::CONTROL) {
            match key.code {
                KeyCode::Char('c' | 'd') => return Some(PaneEvent::Quit),
                KeyCode::Char('u') => self.clear_prompt(),
                // To the line start / end, as readline
                KeyCode::Char('a') => self.prompt.home(),
                KeyCode::Char('e') => self.prompt.end(),
                // Toggle the thinking run's visibility (only in normal mode)
                KeyCode::Char('t') => self.transcript.toggle_thinking(),
                // Toggle expanding the tool outputs' collapsed middles
                KeyCode::Char('o') => self.transcript.toggle_expand(),
                _ => {}
            }
            return None;
        }
        // macOS word/line bindings (Option/Cmd + arrows/Backspace); unclaimed
        // Option-chars fall through to `insert_char`.
        if crate::keybinds::mac_modifiers(&mut self.prompt, &key) {
            return None;
        }
        // Popups take the arrow keys, Tab, and Enter before the events hit the pane
        let claimed = match key.code {
            KeyCode::Up | KeyCode::Down | KeyCode::Tab => true,
            KeyCode::Enter => !self.bang,
            _ => false,
        };
        if self.popup.is_some() && !key.modifiers.contains(KeyModifiers::ALT) && claimed {
            return self.popup_key(key);
        }
        // Option+Up moves the queued steering into the composer for editing.
        // Enter re-queues the edited draft as a bew message.
        if key.code == KeyCode::Up
            && key.modifiers.contains(KeyModifiers::ALT)
            && self.control.steering().is_some()
        {
            self.spill_steer();
            return None;
        }
        match key.code {
            KeyCode::Enter if key.modifiers.contains(KeyModifiers::ALT) => self.prompt.new_line(),
            // Classify manual commands BEFORE steering
            KeyCode::Enter if self.bang => return self.submit_bang(),
            // A draft mid-generation must steer the running turn, but slash commands
            // still have to wait. Only one message can be queued at once
            KeyCode::Enter if self.spin.is_some() => {
                let text = self.prompt.text();
                if !text.trim().is_empty() && !text.trim().starts_with('/') {
                    if self.control.steer(text) {
                        self.clear_prompt();
                    } else {
                        self.note("message queued: Option+Up to edit");
                    }
                }
            }
            KeyCode::Enter => return self.submit(),
            // Esc with nothing to close cancels the turn or a manual command.
            KeyCode::Esc => {
                if !self.escape() && (self.spin.is_some() || self.manual.is_some()) {
                    if self.spin.is_some() {
                        self.spill_steer();
                    }
                    return Some(PaneEvent::Cancel);
                }
            }
            KeyCode::Up if key.modifiers.contains(KeyModifiers::SHIFT) => {
                self.copy = Some(CopyCursor::enter(self.transcript.rows().len()));
            }
            // Tab completes the argument under the caret as a file
            KeyCode::Tab if self.bang => self.open_completion(),
            // `!` on an empty draft enters the manual-command mode; anywhere
            // else (or with the mode already on) it is an ordinary character.
            KeyCode::Char('!') if !self.bang && self.draft_is_empty() => self.bang = true,
            KeyCode::Char(c) => self.prompt.insert_char(c),
            KeyCode::Tab => self.prompt.insert_char('\t'),
            // An empty draft's backspace leaves the mode instead of deleting.
            KeyCode::Backspace if self.bang && self.draft_is_empty() => self.bang = false,
            KeyCode::Backspace => self.prompt.backspace(),
            KeyCode::Left => self.prompt.left(),
            KeyCode::Right => self.prompt.right(),
            KeyCode::Up => self.prompt.up(),
            KeyCode::Down => self.prompt.down(),
            KeyCode::Home => self.prompt.home(),
            KeyCode::End => self.prompt.end(),
            _ => {}
        }
        None
    }

    /// Pastes end up in the prompt box, newlines included.
    pub fn on_paste(&mut self, text: &str) {
        if self.copy.is_none() {
            // A pasted command starting with the marker enters the mode.
            if !self.bang
                && self.draft_is_empty()
                && let Some(command) = text.strip_prefix('!')
            {
                self.bang = true;
                self.prompt.insert_str(command);
            } else {
                self.prompt.insert_str(text);
            }
            // A paste refilters an open popup but never opens one.
            self.sync_popup(None);
        }
    }

    /// Close the file popup, else leave copy mode. Returns whether anything was closed.
    pub fn escape(&mut self) -> bool {
        self.popup.take().is_some() || self.copy.take().is_some()
    }

    /// Paint one event into the tui
    pub fn apply(&mut self, progress: &Progress) {
        match progress {
            Progress::User(text) => {
                self.echo(text);
                self.begin_response();
            }
            Progress::Thinking(text) => self.append_thinking(text),
            Progress::Answer(text) => self.append_answer(text),
            Progress::ToolStart { id, name, digest } => {
                self.start_tool(id.clone(), name, digest.clone());
            }
            Progress::ToolOutput { id, output, exit } => {
                self.finish_tool(id, output.clone(), *exit);
            }
            Progress::Usage { input, cached, output } => {
                self.set_usage(*input, *cached, *output);
            }
            // A steering message aborted that response: retire its thinking
            // run (the record dropped it too), then echo like a submitted one.
            Progress::Steered(text) => {
                self.begin_response();
                self.echo(text);
            }
            // `Progress` is non-exhaustive; later variants need no handling yet.
            _ => {}
        }
    }

    pub fn push<L: Into<Line<'static>>>(&mut self, line: L) {
        self.transcript.push(line);
    }

    /// Append a dim system line to the UI that is excluded from the sessions record.
    pub fn note<S: Into<String>>(&mut self, text: S) {
        self.push(Span::styled(text.into(), DIM_STYLE));
    }

    /// Append a fragment of the model's streamed answer; see [`Transcript::append`].
    pub fn append_answer(&mut self, text: &str) {
        self.transcript.append(text);
    }

    /// Append a styled streaming fragment outside the answer.
    pub fn append_span(&mut self, span: &Span<'static>) {
        self.transcript.append_span(span);
    }

    /// Append a streaming chain-of-thought fragment to the current thinking block
    pub fn append_thinking(&mut self, fragment: &str) {
        self.transcript.append_thinking(fragment);
    }

    /// Retire the previous response's thinking; see [`Transcript::begin_response`].
    pub fn begin_response(&mut self) {
        self.transcript.begin_response();
    }

    /// Record a tool invocation's start; see [`Transcript::start_tool`].
    pub fn start_tool(&mut self, id: String, name: &'static str, digest: String) {
        self.transcript.start_tool(id, name, digest);
    }

    /// Fill in the pending invocation; see [`Transcript::finish_tool`].
    pub fn finish_tool(&mut self, id: &str, output: String, exit: Option<i32>) {
        self.transcript.finish_tool(id, output, exit);
    }

    /// Resolve still-running invocations; see [`Transcript::fail_pending`].
    pub fn fail_pending(&mut self, reason: &str) {
        self.transcript.fail_pending(reason);
    }

    /// Record one response's token usage for the status line.
    pub fn set_usage(&mut self, input: u64, cached: u64, output: u64) {
        self.usage = Some(Usage { input, cached, output });
    }

    /// Record the model's context window, from the agents file.
    pub fn set_context_tokens(&mut self, tokens: u64) {
        self.context_tokens = Some(tokens);
    }

    /// Name the directory `/resume` picks sessions from.
    pub fn set_session_dir(&mut self, root: PathBuf, project: PathBuf) {
        self.session_dir = Some((root, project));
    }

    /// The status line's text: the context size against the model's window,
    /// with the cache share when the provider reports one.
    fn status_text(&self) -> Option<String> {
        let usage = self.usage?;
        let context = usage.input + usage.output;
        let mut text = match self.context_tokens {
            Some(window) => format!("{} / {}", token_count(context), token_count(window)),
            None => token_count(context),
        };
        if usage.cached > 0 && usage.input > 0 {
            let percent = usage.cached * 100 / usage.input;
            text.push_str(" · ");
            text.push_str(&percent.to_string());
            text.push_str("% cached");
        }
        Some(text)
    }

    /// The status rule's current spinner frame, or None while idle: it spins
    /// for a generating turn first, then for a manual command.
    fn spinner(&self) -> Option<&'static str> {
        let started = self
            .spin
            .or_else(|| self.manual.as_ref().map(|manual| manual.started))?;
        let elapsed = started.elapsed().as_millis();
        Some(SPINNER_FRAMES[(elapsed / SPINNER_MS) as usize % SPINNER_FRAMES.len()])
    }

    /// The bottom rule's badge while a manual command runs: `! <command>`.
    fn manual_command_badge(&self, width: u16) -> Option<String> {
        self.manual.as_ref().map(|manual| {
            // The badge reads `[ ! <command> ]` after the spinner's three cells.
            let budget = width.saturating_sub(9) as usize;
            let command = manual.command.replace('\n', " ");
            format!("! {}", ellipsize(&command, budget))
        })
    }

    /// The top rule's queued-steering snippet, if a message waits.
    fn queued_text(&self, width: u16) -> Option<String> {
        queued_rule_text(&self.control.steering()?, width as usize)
    }

    pub fn clear(&mut self) {
        self.transcript.clear();
        // The abandoned conversation's usage leaves with it.
        self.usage = None;
    }

    /// Update the `/perf` stats line; `None` restores the bottom rule.
    pub fn set_perf(&mut self, perf: Option<String>) {
        self.perf = perf;
    }

    /// Mark the model busy; Enter keeps the draft instead of submitting it.
    pub fn set_generating(&mut self, generating: bool) {
        self.spin = generating.then(Instant::now);
    }

    /// Wire the pane to the agent's turn control.
    pub fn set_control(&mut self, control: TurnControl) {
        self.control = control;
    }

    /// Move the queued steering message into the composer as editable text
    pub fn spill_steer(&mut self) {
        if let Some(text) = self.control.take_steer() {
            if !self.prompt.text().is_empty() {
                self.prompt.new_line();
            }
            self.prompt.insert_str(&text);
        }
    }

    /// Open the file completion for the argument under the caret.
    fn open_completion(&mut self) {
        let query = file_mentions::derive_argument(&self.prompt);
        file_mentions::update(&mut self.popup, query, true);
    }

    /// Submit the manual-command draft, or explain why now is not the time.
    fn submit_bang(&mut self) -> Option<PaneEvent> {
        let refusal = if self.manual.is_some() {
            "a manual command is already running"
        } else if self.spin.is_some() {
            "manual commands wait for the turn: Esc to cancel"
        } else if self.prompt.text().trim().is_empty() {
            "! alone runs nothing"
        } else {
            return self.submit();
        };
        self.note(refusal);
        None
    }

    /// Echo the draft into the transcript and clear it.
    fn submit(&mut self) -> Option<PaneEvent> {
        let text = self.prompt.text();
        if text.trim().is_empty() {
            return None;
        }
        let bang = self.bang;
        self.bang = false;
        self.prompt.clear();
        if bang {
            // A command's echo waits for its run: what the transcript shows
            // and what it records both land with the framed output.
            return Some(PaneEvent::Command(text));
        }
        self.echo(&text);
        Some(PaneEvent::Submit(text))
    }

    /// Echo a submitted line into the transcript, as [`Pane::submit`] renders it.
    pub fn echo(&mut self, text: &str) {
        self.echo_styled(PROMPT, PROMPT_STYLE, text);
    }

    /// [`Pane::echo`] with the glyph swapped, so a manual command's line reads
    /// exactly like the prompt row that launched it.
    fn echo_styled(&mut self, glyph: &'static str, style: Style, text: &str) {
        let mut rows = text.split('\n');
        self.transcript.push(Line::from(vec![
            Span::styled(glyph, style),
            Span::raw(rows.next().unwrap_or_default().to_string()),
        ]));
        for continuation in rows {
            self.transcript.push(Line::from(format!("  {continuation}")));
        }
    }

    /// Mark a manual command in flight: the frame holds the bang styling and
    /// the status rule carries `! <command>` until [`Pane::manual_done`].
    pub fn manual_running(&mut self, command: Option<String>) {
        self.manual = command.map(|command| ManualCommand { command, started: Instant::now() });
    }

    /// Finish the in-flight manual command by echoing its line and the framed
    /// output, and hand back the command that ran.
    pub fn manual_done(&mut self, framed: &str) -> Option<String> {
        let manual = self.manual.take()?;
        self.echo_styled(BANG_PROMPT, BANG_STYLE, &manual.command);
        // The framing already says how it ended, so the output renders dim.
        self.transcript
            .append_span(&Span::styled(framed.to_string(), DIM_STYLE));
        Some(manual.command)
    }

    /// Whether every line of the draft is empty, i.e. nothing is typed yet.
    fn draft_is_empty(&self) -> bool {
        self.prompt.lines.iter().all(String::is_empty)
    }

    /// Empty the draft, leaving the manual-command mode with it.
    fn clear_prompt(&mut self) {
        self.prompt.clear();
        self.bang = false;
    }

    /// Render into the granted area: transcript, rule, prompt, rule.
    pub fn render(&mut self, frame: &mut Frame, area: Rect) {
        if area.height == 0 || area.width == 0 {
            return;
        }
        // The prompt grows with its wrapped content, always leaving space for the
        // transcript and the two rules. Copy mode swaps the prompt for a one-row hint,
        // so the draft is only wrapped when the prompt actually renders.
        let cap = area.height.saturating_sub(4).max(1) as usize;
        let (prompt_height, layout) = if self.copy.is_some() {
            (1, None)
        } else {
            let layout = wrap_draft(
                &self.prompt.lines,
                (self.prompt.line, self.prompt.g),
                area.width.saturating_sub(GUTTER) as usize,
            );
            (layout.rows.len().min(cap).max(1) as u16, Some(layout))
        };
        let [transcript, bar_top, prompt_area, bar_bottom] = Layout::vertical([
            Constraint::Min(1),
            Constraint::Length(1),
            Constraint::Length(prompt_height),
            Constraint::Length(1),
        ])
        .areas(area);

        self.render_transcript(transcript, bar_top, bar_bottom, frame);
        // No layout means copy mode: show the scrollback hint instead.
        let Some(layout) = layout else {
            Self::render_scrollback_hint(frame, prompt_area);
            return;
        };

        if prompt_area.height == 0 || prompt_area.width < GUTTER {
            return;
        }
        let buf = frame.buffer_mut();
        // Live prompt: the "❯ " gutter (or "! " in the manual-command mode),
        // then the wrapped draft. The inverted caret cell marks the cursor, so
        // the terminal cursor stays hidden and the prompt viewport can never
        // misplace it.
        let width = prompt_area.width - GUTTER;
        let (glyph, glyph_style) = if self.bang {
            (BANG_PROMPT, BANG_STYLE)
        } else {
            (PROMPT, PROMPT_STYLE)
        };
        buf.set_span(
            prompt_area.x,
            prompt_area.y,
            &Span::styled(glyph, glyph_style),
            GUTTER,
        );
        self.prompt.top = window_top(
            layout.rows.len(),
            prompt_area.height as usize,
            Some((layout.caret_row, self.prompt.top)),
        );
        let top = self.prompt.top;
        let shown = (prompt_area.height as usize).min(layout.rows.len().saturating_sub(top));
        for i in 0..shown {
            buf.set_line(
                prompt_area.x + GUTTER,
                prompt_area.y + i as u16,
                &layout.rows[top + i],
                width,
            );
        }
        // A caret at the end of a full row inverts the row's last cell.
        let col = layout.caret_col.min(width.saturating_sub(1) as usize) as u16;
        let pos = (
            prompt_area.x + GUTTER + col,
            prompt_area.y + (layout.caret_row - top) as u16,
        );
        if let Some(cell) = buf.cell_mut(pos) {
            cell.set_style(CURSOR_STYLE);
        }
        // The popup overlays the transcript, anchored above the top rule..
        let hint = if self.bang {
            "↑↓ select · Tab insert · Esc close"
        } else {
            "↑↓ select · Tab/Enter insert · Esc close"
        };
        match self.popup.as_mut() {
            Some(Popup::Files(popup)) => popup.render(frame, bar_top, "files", hint),
            Some(Popup::Sessions(sessions)) => sessions.render(frame, bar_top),
            None => {}
        }
    }

    /// Sync, window, and paint the transcript, then draw its two rules.
    fn render_transcript(
        &mut self,
        area: Rect,
        bar_top: Rect,
        bar_bottom: Rect,
        frame: &mut Frame,
    ) {
        // Queued snippet reads the steer queue before `sync` borrows the transcript
        let queued = self.queued_text(bar_top.width);
        // Wrap only what is new at an unchanged width, or rewrap if width changed.
        let rows = self.transcript.sync(area.width as usize);
        // Clamp to the wrapped rows, moving the cursor to (0, 0) when empty.
        if let Some(cursor) = &mut self.copy {
            (cursor.row, cursor.col) = clamp_cell(rows, cursor.row, cursor.col);
            cursor.anchor = cursor.anchor.map(|(row, col)| clamp_cell(rows, row, col));
        }
        let visible = area.height as usize;
        let top = window_top(rows.len(), visible, self.copy.map(|c| (c.row, c.top)));
        if let Some(cursor) = &mut self.copy {
            cursor.top = top;
            cursor.visible = visible;
        }
        let buf = frame.buffer_mut();
        let shown = visible.min(rows.len().saturating_sub(top));
        rows[top..top + shown].iter().zip(area.y..).for_each(|(row, y)| {
            buf.set_line(area.x, y, row, area.width);
        });
        if let Some(cursor) = self.copy {
            if let Some(selection) = Selection::between(cursor.anchor, (cursor.row, cursor.col)) {
                selection.paint(buf, rows, area, top, shown);
            }
            let pos = (area.x + cursor.col as u16, area.y + (cursor.row - top) as u16);
            if let Some(cell) = buf.cell_mut(pos) {
                cell.set_style(CURSOR_STYLE);
            }
        }
        queued_rule(buf, bar_top, queued.as_deref());
        if let Some(perf) = &self.perf {
            // Replace the statusline with the perf counters
            let line = Line::from(Span::styled(format!("{perf} · {} rows", rows.len()), DIM_STYLE));
            // Layout overflow can park a zero-height bar past the last row.
            if bar_bottom.y < buf.area.height {
                buf.set_line(bar_bottom.x, bar_bottom.y, &line, bar_bottom.width);
            }
        } else {
            let status = self
                .manual_command_badge(bar_bottom.width)
                .or_else(|| self.status_text());
            status_rule(buf, bar_bottom, status.as_deref(), self.spinner());
        }
        // The manual-command mode colors the frame around the composer.
        if self.bang || self.manual.is_some() {
            reframe(buf, bar_top, BANG_RULE);
            reframe(buf, bar_bottom, BANG_RULE);
        }
    }

    /// In copy mode the prompt area shows the scrollback keybindings.
    fn render_scrollback_hint(frame: &mut Frame, prompt_area: Rect) {
        if prompt_area.height > 0 {
            frame.buffer_mut().set_line(
                prompt_area.x,
                prompt_area.y,
                &Line::from(Span::styled(
                    "▲ scrollback · ←↑↓→/PgUp/Home/End · Space select · Enter copy · q to exit",
                    Style::new().fg(Color::Yellow),
                )),
                prompt_area.width,
            );
        }
    }
}

/// Replay events into the pane, painting each as the live stream does.
///
/// Tool exchanges arrive as headers only (see `Transcript::replay`).
impl Extend<Progress> for Pane {
    fn extend<T: IntoIterator<Item = Progress>>(&mut self, iter: T) {
        for event in iter {
            self.apply(&event);
        }
    }
}

/// A full-width dim rule row, drawn cell by cell.
fn rule(buf: &mut Buffer, area: Rect) {
    for col in area.columns() {
        if let Some(cell) = buf.cell_mut((col.x, area.y)) {
            cell.set_symbol(symbols::line::HORIZONTAL).set_style(DIM_STYLE);
        }
    }
}

/// Recolor one already-drawn rule row.
fn reframe(buf: &mut Buffer, area: Rect, color: Color) {
    for col in area.columns() {
        if let Some(cell) = buf.cell_mut((col.x, area.y)) {
            cell.set_fg(color);
        }
    }
}

/// A token count in the status line's compact style: `843`, `45k`, `1.2 M`.
fn token_count(tokens: u64) -> String {
    match tokens {
        0..=999 => tokens.to_string(),
        1_000..=999_999 => format!("{}k", tokens / 1_000),
        _ => format!("{}.{} M", tokens / 1_000_000, tokens % 1_000_000 / 100_000),
    }
}

/// `text` as-is when it fits `budget` cells, else its clipped start and an ellipsis
fn ellipsize(text: &str, budget: usize) -> String {
    let cells = |text: &str| {
        text.graphemes(true)
            .map(|g| Span::raw(g).width().max(1))
            .sum::<usize>()
    };
    if cells(text) <= budget {
        return text.to_string();
    }
    // One cell stays free so a cut always shows its ellipsis.
    let mut cut = String::new();
    let mut used = 0;
    for grapheme in text.graphemes(true) {
        let spent = Span::raw(grapheme).width().max(1);
        if used + spent >= budget {
            break;
        }
        cut.push_str(grapheme);
        used += spent;
    }
    cut.push('…');
    cut
}

/// A full-width dim rule row with the status badge set into it:
/// `───[ status ]─────…`; while generating, a spinner takes the leading
/// dashes: `···[ status ]─────…`.
fn status_rule(buf: &mut Buffer, area: Rect, status: Option<&str>, spinner: Option<&str>) {
    rule(buf, area);
    // The bar can be parked past the last row.
    if area.y >= buf.area.height {
        return;
    }
    if let Some(spinner) = spinner {
        buf.set_stringn(area.x, area.y, spinner, 3, DIM_STYLE);
    }
    if let Some(status) = status {
        buf.set_stringn(
            area.x + 3,
            area.y,
            format!("[ {status} ]"),
            area.width.saturating_sub(3) as usize,
            DIM_STYLE,
        );
    }
}

/// The top rule's steering snippet: `queued: '<message>'`, ellipsized to fit
/// `width` cells; `None` when the rule is too narrow to hold one.
fn queued_rule_text(message: &str, width: usize) -> Option<String> {
    // The preview reads as one line; wide graphemes pay their cell widths.
    let message = message.replace('\n', " ");
    // `queued: ''` wraps the snippet, and a dash of rule leads it.
    if width <= 11 {
        return None;
    }
    Some(format!("queued: '{}'", ellipsize(&message, width - 10)))
}

/// A full-width dim rule row with the queued-steering snippet set into its
/// right end: `────────queued: 'fix the login'`.
fn queued_rule(buf: &mut Buffer, area: Rect, text: Option<&str>) {
    rule(buf, area);
    let Some(text) = text else { return };
    // The bar can be parked past the last row.
    if area.y >= buf.area.height {
        return;
    }
    let line = Line::from(Span::styled(text, DIM_STYLE));
    let x = area.x + area.width.saturating_sub(line.width() as u16);
    buf.set_line(x, area.y, &line, area.width);
}

#[cfg(test)]
mod tests {
    #![allow(clippy::panic, clippy::unwrap_used, reason = "test assertions")]

    use super::*;
    use crate::testutil::{draw, draw_backgrounds, draw_styles};

    fn key(code: KeyCode, modifiers: KeyModifiers) -> KeyEvent {
        KeyEvent::new(code, modifiers)
    }

    fn render(pane: &mut Pane, size: (u16, u16)) -> String {
        draw(|frame, area| pane.render(frame, area), size)
    }

    #[test]
    fn token_count_formats_compactly() {
        assert_eq!(token_count(0), "0");
        assert_eq!(token_count(843), "843");
        assert_eq!(token_count(999), "999");
        assert_eq!(token_count(1_000), "1k");
        assert_eq!(token_count(45_600), "45k");
        assert_eq!(token_count(999_999), "999k");
        assert_eq!(token_count(1_000_000), "1.0 M");
        assert_eq!(token_count(1_234_567), "1.2 M");
        assert_eq!(token_count(999_999_999), "999.9 M");
    }

    /// The bottom rule carries the token gauge; plain until usage arrives,
    /// and `perf` still overrides it.
    #[test]
    fn the_bottom_rule_carries_the_usage_badge() {
        let mut pane = Pane::default();
        pane.push(Line::from("text"));
        assert!(!render(&mut pane, (60, 8)).contains('['));

        pane.set_context_tokens(200_000);
        pane.set_usage(45_000, 40_000, 3_000);
        let gauge = render(&mut pane, (60, 8));
        assert!(gauge.contains("───[ 48k / 200k · 88% cached ]"), "{gauge}");

        // The perf line replaces the badge when both are set.
        pane.set_perf(Some(" fps 60 ".into()));
        let perf = render(&mut pane, (60, 8));
        assert!(perf.contains("rows"));
        assert!(!perf.contains("48k"), "{perf}");
        pane.set_perf(None);

        // No cache reported drops the suffix.
        pane.set_usage(1_000, 0, 0);
        assert!(render(&mut pane, (60, 8)).contains("[ 1k / 200k ]"));

        // A generating turn spins in the leading dashes
        pane.set_generating(true);
        let spinning = render(&mut pane, (60, 8));
        assert!(spinning.contains("·  [ 1k / 200k ]"), "{spinning}");
        pane.set_generating(false);
        assert!(render(&mut pane, (60, 8)).contains("───[ 1k / 200k ]"));

        // No window configured drops the ratio.
        let mut bare = Pane::default();
        bare.push(Line::from("text"));
        bare.set_usage(45_000, 0, 3_000);
        assert!(render(&mut bare, (60, 8)).contains("[ 48k ]"));
    }

    #[test]
    fn perf_line_replaces_the_bottom_rule() {
        let mut pane = Pane::default();
        let plain = render(&mut pane, (60, 10));
        pane.set_perf(Some(" fps 60 ".into()));
        let perf = render(&mut pane, (60, 10));
        assert!(!plain.contains("rows"));
        assert!(perf.contains("fps 60"));
        assert!(perf.contains("rows"));
    }

    /// Enter mid-generation queues the draft as steering; slash commands wait
    #[test]
    fn enter_while_generating_queues_steering() {
        let mut pane = Pane::default();
        for c in ['h', 'i'] {
            pane.on_key(key(KeyCode::Char(c), KeyModifiers::NONE));
        }
        pane.set_generating(true);
        // Enter steers and clears; nothing echoes into the transcript.
        assert_eq!(pane.on_key(key(KeyCode::Enter, KeyModifiers::NONE)), None);
        assert_eq!(pane.prompt.text(), "");
        assert_eq!(pane.control.steering(), Some("hi".to_string()));
        assert!(pane.transcript.message_texts().is_empty());

        // A second message waits: the queue takes one at a time, and the
        // draft stays in the composer with a note pointing at Option+Up.
        pane.on_paste("another");
        assert_eq!(pane.on_key(key(KeyCode::Enter, KeyModifiers::NONE)), None);
        assert_eq!(pane.prompt.text(), "another");
        assert_eq!(pane.control.steering(), Some("hi".to_string()));
        let screen = render(&mut pane, (60, 10));
        assert!(screen.contains("message queued: Option+Up to edit"), "{screen}");
        pane.prompt.clear();

        // Alt+Enter still edits the fresh draft.
        pane.on_key(key(KeyCode::Enter, KeyModifiers::ALT));
        assert_eq!(pane.prompt.lines.len(), 2);

        // A slash command mid-generation stays in the composer.
        pane.prompt.clear();
        pane.on_paste("/effort high");
        assert_eq!(pane.on_key(key(KeyCode::Enter, KeyModifiers::NONE)), None);
        assert_eq!(pane.prompt.text(), "/effort high");

        // Once the model is done, Enter submits the draft as a normal turn.
        pane.prompt.clear();
        pane.on_paste("done waiting");
        pane.set_generating(false);
        assert_eq!(
            pane.on_key(key(KeyCode::Enter, KeyModifiers::NONE)),
            Some(PaneEvent::Submit("done waiting".into()))
        );
    }

    /// Esc cancels the turn and spills the queued steering into the composer;
    /// a second submission while one waits is refused.
    #[test]
    fn esc_spills_queued_steering_into_the_composer() {
        let mut pane = Pane::default();
        pane.set_generating(true);
        assert!(pane.control.steer("one".to_string()));
        assert!(!pane.control.steer("two".to_string())); // one at a time
        assert_eq!(
            pane.on_key(key(KeyCode::Esc, KeyModifiers::NONE)),
            Some(PaneEvent::Cancel)
        );
        assert_eq!(pane.prompt.text(), "one");
        assert_eq!(pane.control.steering(), None);
    }

    /// Option+Up moves the queued steering into the composer for editing;
    /// Enter re-queues the edited draft.
    #[test]
    fn alt_up_spills_the_queue_into_the_composer() {
        let mut pane = Pane::default();
        pane.set_generating(true);
        assert!(pane.control.steer("one".to_string()));
        pane.on_key(key(KeyCode::Up, KeyModifiers::ALT));
        assert_eq!(pane.prompt.text(), "one");
        assert_eq!(pane.control.steering(), None);

        // An edit, then Enter: the draft re-queues as the steering message.
        pane.on_paste(" now two");
        assert_eq!(pane.on_key(key(KeyCode::Enter, KeyModifiers::NONE)), None);
        assert_eq!(pane.control.steering(), Some("one now two".to_string()));
        assert_eq!(pane.prompt.text(), "");

        // A spill onto a non-empty draft lands on a fresh line, and the chord
        // reaches through an open popup (bare arrows stay the popup's).
        pane.on_paste("also this");
        pane.popup = Some(Popup::Files(FilePopup::from_files(vec![], String::new())));
        pane.on_key(key(KeyCode::Up, KeyModifiers::ALT));
        assert_eq!(pane.prompt.text(), "also this\none now two");
    }

    /// A steered message retires the aborted response's thinking run, then
    /// echoes into the transcript like a submitted one.
    #[test]
    fn steered_messages_echo_into_the_transcript() {
        let mut pane = Pane::default();
        pane.begin_response();
        pane.append_thinking("doomed reasoning");
        pane.append_answer("partial");
        pane.transcript.toggle_thinking(); // reveal the run
        assert!(render(&mut pane, (60, 10)).contains("doomed"));
        pane.apply(&Progress::Steered("go faster".to_string()));
        pane.append_answer(" more");
        let screen = render(&mut pane, (60, 10));
        assert!(screen.contains("❯ go faster"), "{screen}");
        assert!(screen.contains("partial"), "{screen}");
        assert!(!screen.contains("doomed"), "{screen}");
    }

    /// `!` on an empty draft enters the manual-command mode: the glyph swaps to
    /// `!`, both rules turn magenta, and the marker never reaches the editor
    #[test]
    fn bang_on_an_empty_draft_swaps_the_glyph_and_the_rules() {
        let mut pane = Pane::default();
        pane.push(Line::from("text"));
        let idle_styles = draw_styles(|frame, area| pane.render(frame, area), (40, 8));
        let idle: Vec<&str> = idle_styles.lines().collect();
        // Rows: five of transcript, top rule, prompt, bottom rule.
        assert_eq!(idle[5], "d".repeat(40), "{idle_styles}");
        assert_eq!(idle[7], "d".repeat(40), "{idle_styles}");

        pane.on_key(key(KeyCode::Char('!'), KeyModifiers::NONE));
        assert!(pane.bang, "the mode is on");
        assert_eq!(pane.prompt.text(), "", "the marker is swallowed");
        let screen = render(&mut pane, (40, 8));
        assert!(screen.contains("! "), "{screen}");
        assert!(!screen.contains("❯"), "{screen}");
        let bang = draw_styles(|frame, area| pane.render(frame, area), (40, 8));
        let bang: Vec<&str> = bang.lines().collect();
        assert_eq!(bang[5], "m".repeat(40), "the top rule is magenta");
        assert_eq!(bang[7], "m".repeat(40), "the bottom rule is magenta");
        assert!(bang[6].starts_with('B'), "the glyph is bold: {bang:?}");

        // Backspace on the still-empty draft leaves the mode; typing first
        // would make the backspace an ordinary deletion.
        pane.on_key(key(KeyCode::Backspace, KeyModifiers::NONE));
        assert!(!pane.bang);
        let screen = render(&mut pane, (40, 8));
        assert!(screen.contains("❯ "), "{screen}");
        let left = draw_styles(|frame, area| pane.render(frame, area), (40, 8));
        let left: Vec<&str> = left.lines().collect();
        assert_eq!(left[5], "d".repeat(40));
        assert_eq!(left[7], "d".repeat(40));
    }

    /// The marker only enters the mode at the start of an empty draft
    #[test]
    fn a_literal_bang_elsewhere_stays_text() {
        let mut pane = Pane::default();
        pane.on_paste("hi");
        pane.on_key(key(KeyCode::Char('!'), KeyModifiers::NONE));
        assert!(!pane.bang);
        assert_eq!(pane.prompt.text(), "hi!");

        let mut literal = Pane::default();
        literal.on_paste(" !ls");
        assert!(!literal.bang);
        assert_eq!(literal.prompt.text(), " !ls");

        let mut cleared = Pane::default();
        cleared.on_key(key(KeyCode::Char('!'), KeyModifiers::NONE));
        cleared.on_paste("ls");
        cleared.on_key(key(KeyCode::Char('u'), KeyModifiers::CONTROL));
        assert!(!cleared.bang);
        assert_eq!(cleared.prompt.text(), "");
    }

    /// Enter ships the command as its own event
    #[test]
    fn bang_enter_ships_the_command_and_waits_to_echo() {
        let mut pane = Pane::default();
        pane.on_key(key(KeyCode::Char('!'), KeyModifiers::NONE));
        for c in "ls -la".chars() {
            pane.on_key(key(KeyCode::Char(c), KeyModifiers::NONE));
        }
        assert_eq!(
            pane.on_key(key(KeyCode::Enter, KeyModifiers::NONE)),
            Some(PaneEvent::Command("ls -la".to_string()))
        );
        assert!(!pane.bang, "submit leaves the mode");
        assert_eq!(pane.prompt.text(), "", "submit clears the draft");
        assert!(pane.transcript.message_texts().is_empty(), "nothing echoed yet");

        // The run holds the frame and claims the status badge, clipped to fit.
        pane.manual_running(Some("x".repeat(60)));
        let screen = render(&mut pane, (40, 8));
        // A 40-cell rule leaves the command 31 cells after the frame.
        assert!(screen.contains(&format!("[ ! {}… ]", "x".repeat(30))), "{screen}");
        let running = draw_styles(|frame, area| pane.render(frame, area), (40, 8));
        let running: Vec<&str> = running.lines().collect();
        assert_eq!(running[5], "m".repeat(40), "the frame stays bang while running");
        assert_eq!(running[7], "m".repeat(40));

        // Landing: the line echoes in the bang style, the output dim, and the
        // pane hands back the command that ran.
        assert_eq!(pane.manual_done("[exit 1]\nboom"), Some("x".repeat(60)));
        assert_eq!(
            pane.transcript.message_texts(),
            [
                format!("! {}", "x".repeat(60)),
                "[exit 1]".to_string(),
                "boom".to_string()
            ]
        );
        let done = draw_styles(|frame, area| pane.render(frame, area), (40, 8));
        let done: Vec<&str> = done.lines().collect();
        assert_eq!(done[5], "d".repeat(40), "the frame relaxes when the run ends");
        assert_eq!(done[7], "d".repeat(40));
        // A second landing has nothing to finish.
        assert_eq!(pane.manual_done("again"), None);

        // A run with a badge renders at every size.
        pane.manual_running(Some("cargo build".to_string()));
        for size in [(60, 8), (6, 3), (2, 1), (1, 0)] {
            render(&mut pane, size);
        }
    }

    /// A multi-line command runs verbatim and echoes every line, indented like
    /// a submitted one.
    #[test]
    fn a_multiline_manual_command_echoes_its_lines() {
        let mut pane = Pane::default();
        pane.on_key(key(KeyCode::Char('!'), KeyModifiers::NONE));
        pane.on_paste("cargo build");
        pane.on_key(key(KeyCode::Enter, KeyModifiers::ALT));
        pane.on_paste("cargo test");
        assert_eq!(pane.prompt.text(), "cargo build\ncargo test");
        assert_eq!(
            pane.on_key(key(KeyCode::Enter, KeyModifiers::NONE)),
            Some(PaneEvent::Command("cargo build\ncargo test".to_string()))
        );

        pane.manual_running(Some("cargo build\ncargo test".to_string()));
        assert_eq!(
            pane.manual_done("done"),
            Some("cargo build\ncargo test".to_string())
        );
        assert_eq!(
            pane.transcript.message_texts(),
            ["! cargo build", "  cargo test", "done"]
        );
    }

    /// The marker alone runs nothing, and the mode stays on for the command.
    #[test]
    fn bang_enter_on_an_empty_command_is_a_noop() {
        let mut pane = Pane::default();
        pane.on_key(key(KeyCode::Char('!'), KeyModifiers::NONE));
        assert_eq!(pane.on_key(key(KeyCode::Enter, KeyModifiers::NONE)), None);
        assert!(pane.bang, "the mode stays on");
        assert_eq!(pane.prompt.text(), "");
        let screen = render(&mut pane, (40, 8));
        assert!(screen.contains("! alone runs nothing"), "{screen}");
        // Whitespace-only is just as empty.
        pane.on_paste("   ");
        assert_eq!(pane.on_key(key(KeyCode::Enter, KeyModifiers::NONE)), None);
        assert!(pane.bang);
    }

    /// A manual command never steers a running turn, and waits for it instead.
    #[test]
    fn bang_enter_waits_for_a_running_turn() {
        let mut pane = Pane::default();
        pane.on_key(key(KeyCode::Char('!'), KeyModifiers::NONE));
        pane.on_paste("cargo build");
        pane.set_generating(true);
        assert_eq!(pane.on_key(key(KeyCode::Enter, KeyModifiers::NONE)), None);
        assert_eq!(pane.prompt.text(), "cargo build", "the draft is kept");
        assert_eq!(pane.control.steering(), None, "a command is not a message");
        let screen = render(&mut pane, (40, 8));
        assert!(screen.contains("manual commands wait for the turn"), "{screen}");

        // Once the turn is done, the same draft ships as a command.
        pane.set_generating(false);
        assert_eq!(
            pane.on_key(key(KeyCode::Enter, KeyModifiers::NONE)),
            Some(PaneEvent::Command("cargo build".to_string()))
        );
    }

    /// One manual command at a time: a second waits rather than mislabeling the
    /// first run's echo.
    #[test]
    fn a_second_manual_command_waits_for_the_first() {
        let mut pane = Pane::default();
        pane.manual_running(Some("first".to_string()));
        pane.on_key(key(KeyCode::Char('!'), KeyModifiers::NONE));
        pane.on_paste("second");
        assert_eq!(pane.on_key(key(KeyCode::Enter, KeyModifiers::NONE)), None);
        assert_eq!(pane.prompt.text(), "second");
        let screen = render(&mut pane, (40, 8));
        assert!(screen.contains("a manual command is already running"), "{screen}");
    }

    /// Esc cancels a manual command.
    #[test]
    fn esc_cancels_a_manual_command_without_spilling_the_queue() {
        let mut pane = Pane::default();
        pane.manual_running(Some("cargo build".to_string()));
        assert!(pane.control.steer("meanwhile".to_string()));
        assert_eq!(
            pane.on_key(key(KeyCode::Esc, KeyModifiers::NONE)),
            Some(PaneEvent::Cancel)
        );
        assert_eq!(pane.control.steering(), Some("meanwhile".to_string()));
        // The run is still in flight; its framed output lands when it does.
        assert!(render(&mut pane, (40, 8)).contains("[ ! cargo build ]"));
    }

    /// A paste starting with the marker enters the mode, the marker stripped.
    #[test]
    fn a_pasted_bang_command_enters_the_mode() {
        let mut pane = Pane::default();
        pane.on_paste("!echo hi");
        assert!(pane.bang);
        assert_eq!(pane.prompt.text(), "echo hi");
        assert_eq!(
            pane.on_key(key(KeyCode::Enter, KeyModifiers::NONE)),
            Some(PaneEvent::Command("echo hi".to_string()))
        );

        // Into a non-empty draft the same paste is ordinary text.
        let mut literal = Pane::default();
        literal.on_paste("run this");
        literal.on_paste("!echo hi");
        assert!(!literal.bang);
        assert_eq!(literal.prompt.text(), "run this!echo hi");
    }
    /// Inside a command, Tab completes the argument under the caret as a file
    /// (bash's default) while neither prefix popup applies: an `@` is just
    /// text, and a leading `/resume` is a path, not the session chooser.
    #[test]
    fn bang_mode_completes_arguments_not_prefixes() {
        let mut pane = Pane::default();
        pane.set_session_dir(PathBuf::from("/tmp/root"), PathBuf::from("/tmp/proj"));
        pane.on_key(key(KeyCode::Char('!'), KeyModifiers::NONE));

        // The command's own word never completes, not even on Tab.
        for c in "cat".chars() {
            pane.on_key(key(KeyCode::Char(c), KeyModifiers::NONE));
        }
        pane.on_key(key(KeyCode::Tab, KeyModifiers::NONE));
        assert!(pane.popup.is_none(), "the command word is not a file");

        // Tab completes the argument over this package; the hint offers Tab
        // alone, since Enter here runs the command.
        for c in " Cargo".chars() {
            pane.on_key(key(KeyCode::Char(c), KeyModifiers::NONE));
        }
        pane.on_key(key(KeyCode::Tab, KeyModifiers::NONE));
        assert!(matches!(pane.popup, Some(Popup::Files(_))));
        let screen = render(&mut pane, (60, 14));
        assert!(screen.contains("Cargo.toml"), "{screen}");
        assert!(screen.contains("Tab insert · Esc close"), "{screen}");
        assert!(!screen.contains("Tab/Enter"), "{screen}");

        // Tab again accepts the highlighted path and closes the list.
        pane.on_key(key(KeyCode::Tab, KeyModifiers::NONE));
        assert_eq!(pane.prompt.text(), "cat Cargo.toml");
        assert!(pane.popup.is_none(), "accepting closes the list");

        // Enter runs the command even while a list is open: only Tab accepts.
        for c in " src/pane".chars() {
            pane.on_key(key(KeyCode::Char(c), KeyModifiers::NONE));
        }
        pane.on_key(key(KeyCode::Tab, KeyModifiers::NONE));
        assert!(matches!(pane.popup, Some(Popup::Files(_))), "the list reopened");
        assert_eq!(
            pane.on_key(key(KeyCode::Enter, KeyModifiers::NONE)),
            Some(PaneEvent::Command("cat Cargo.toml src/pane".to_string()))
        );
        assert!(pane.popup.is_none(), "submitting clears the draft and the list");

        // A leading `/resume` is a path in a command, never the chooser.
        pane.on_key(key(KeyCode::Char('!'), KeyModifiers::NONE));
        pane.on_paste("/resume fix");
        assert!(!matches!(pane.popup, Some(Popup::Sessions(_))));
        assert_eq!(
            pane.on_key(key(KeyCode::Enter, KeyModifiers::NONE)),
            Some(PaneEvent::Command("/resume fix".to_string()))
        );
    }

    /// The list opens only on Tab and follows the word: typing never opens
    /// it, an ended word closes it, and Tab there opens nothing.
    #[test]
    fn completions_open_only_on_tab() {
        let mut pane = Pane::default();
        pane.on_key(key(KeyCode::Char('!'), KeyModifiers::NONE));
        for c in "cat Cargo".chars() {
            pane.on_key(key(KeyCode::Char(c), KeyModifiers::NONE));
        }
        assert!(pane.popup.is_none(), "typing never opens the list");

        // Tab opens it, typing refilters it, Esc closes it.
        pane.on_key(key(KeyCode::Tab, KeyModifiers::NONE));
        assert!(matches!(pane.popup, Some(Popup::Files(_))));
        pane.on_key(key(KeyCode::Char('x'), KeyModifiers::NONE));
        assert!(
            matches!(pane.popup, Some(Popup::Files(_))),
            "an open list refilters"
        );
        pane.on_key(key(KeyCode::Esc, KeyModifiers::NONE));
        assert!(pane.popup.is_none(), "Esc closes the list");

        // Tab re-opens it on the same word; ending the word closes it, and a
        // Tab there has nothing to open.
        pane.on_key(key(KeyCode::Tab, KeyModifiers::NONE));
        assert!(matches!(pane.popup, Some(Popup::Files(_))));
        pane.on_key(key(KeyCode::Char(' '), KeyModifiers::NONE));
        assert!(pane.popup.is_none(), "an ended word closes it");
        pane.on_key(key(KeyCode::Tab, KeyModifiers::NONE));
        assert!(pane.popup.is_none(), "an ended word has nothing to complete");

        // The command's own word opens nothing — and inserts no tab either.
        let mut bare = Pane::default();
        bare.on_key(key(KeyCode::Char('!'), KeyModifiers::NONE));
        bare.on_paste("cargo");
        bare.on_key(key(KeyCode::Tab, KeyModifiers::NONE));
        assert!(bare.popup.is_none());
        assert_eq!(bare.prompt.text(), "cargo");
    }

    /// The top rule carries the queued-steering snippet, right-aligned.
    #[test]
    fn the_top_rule_shows_the_queued_snippet() {
        let mut pane = Pane::default();
        pane.push(Line::from("text"));
        let idle = render(&mut pane, (60, 8));
        assert!(!idle.contains("queued:"), "{idle}");

        assert!(pane.control.steer("update feature XYZ".to_string()));
        let screen = render(&mut pane, (60, 8));
        assert!(screen.contains("queued: 'update feature XYZ'"), "{screen}");
        // The snippet sits in the rule: dashes run right up to it.
        let at = screen.find("queued:").expect("snippet");
        assert!(screen[..at].ends_with('─'), "{screen}");

        // A freed slot takes the next message.
        pane.control.take_steer();
        assert!(pane.control.steer("actually, stop".to_string()));
        let screen = render(&mut pane, (60, 8));
        assert!(screen.contains("queued: 'actually, stop'"), "{screen}");
        assert!(!screen.contains("XYZ"), "{screen}");
    }

    /// The snippet ellipsizes by cells and yields to narrow rules.
    #[test]
    fn queued_rule_text_ellipsizes() {
        assert_eq!(
            queued_rule_text("fix it", 40),
            Some("queued: 'fix it'".to_string())
        );
        // Newlines read as spaces: the preview is one line.
        assert_eq!(
            queued_rule_text("fix the\nlogin flow", 40),
            Some("queued: 'fix the login flow'".to_string())
        );
        // An exactly-in-budget message fits without the ellipsis.
        assert_eq!(
            queued_rule_text("abcdefghij", 20),
            Some("queued: 'abcdefghij'".to_string())
        );
        // A long snippet ellipsizes inside the quotes, cut visible.
        assert_eq!(
            queued_rule_text("update the login flow today", 20),
            Some("queued: 'update th…'".to_string())
        );
        // Wide graphemes pay two cells, not one count: 6 cells into a
        // 5-cell budget cuts after two, into a 3-cell budget after one.
        assert_eq!(
            queued_rule_text("日本語", 15),
            Some("queued: '日本…'".to_string())
        );
        assert_eq!(queued_rule_text("日本語", 13), Some("queued: '日…'".to_string()));
        // Too narrow for even the skeleton: no snippet at all.
        assert_eq!(queued_rule_text("x", 11), None);
    }

    /// A cancelled turn keeps its partial answer; main adds only the marker.
    #[test]
    fn a_cancelled_turn_keeps_its_partial_answer() {
        let mut pane = Pane::default();
        pane.push(Line::from("earlier"));
        pane.begin_response();
        pane.append_answer("earlier answer");

        // The next turn, as main drives it: submit echoes the draft, the
        // response begins, then the stream arrives and Esc lands.
        pane.on_paste("write a story");
        assert_eq!(
            pane.on_key(key(KeyCode::Enter, KeyModifiers::NONE)),
            Some(PaneEvent::Submit("write a story".into()))
        );
        pane.begin_response();
        pane.set_generating(true);
        pane.append_thinking("thinking");
        pane.append_answer("Once upon");
        pane.set_generating(false);
        pane.note("⎋ cancelled");

        let screen = render(&mut pane, (60, 20));
        assert!(screen.contains("earlier answer"), "{screen}");
        assert!(screen.contains("write a story"), "{screen}");
        assert!(screen.contains("Once upon"), "{screen}");
        assert!(screen.contains("⎋ cancelled"), "{screen}");
    }

    #[test]
    fn echo_renders_the_prompt_and_indents_continuations() {
        let mut pane = Pane::default();

        pane.echo("first\nsecond");

        assert_eq!(pane.transcript.message_texts(), ["❯ first", "  second"]);
    }

    /// A `/resume` line opens the session chooser; Enter swaps to the picked
    /// session, and the chooser never swaps under a running turn.
    #[test]
    fn a_resume_line_opens_the_chooser_and_enter_swaps() {
        let root = tempfile::tempdir().unwrap();
        let project = PathBuf::from("/tmp/proj");
        let dir = root.path().join("tmp-proj");
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("20260101-000000.jsonl");
        std::fs::write(
            &file,
            "{\"type\":\"message\",\"role\":\"system\",\"content\":\"s\"}\n\
             {\"type\":\"message\",\"role\":\"user\",\"content\":\"fix the login flow\"}\n",
        )
        .unwrap();
        let mut pane = Pane::default();
        pane.set_session_dir(root.path().to_path_buf(), project);

        for c in "/resume fix".chars() {
            pane.on_key(key(KeyCode::Char(c), KeyModifiers::NONE));
        }
        assert!(matches!(pane.popup, Some(Popup::Sessions(_))));

        // Enter picks the highlighted session and clears the draft.
        assert_eq!(
            pane.on_key(key(KeyCode::Enter, KeyModifiers::NONE)),
            Some(PaneEvent::Resume(file))
        );
        assert_eq!(pane.prompt.text(), "");

        // Esc closes the chooser; the draft survives. Reopened, a generating
        // turn keeps the chooser from swapping.
        for c in "/resume fix".chars() {
            pane.on_key(key(KeyCode::Char(c), KeyModifiers::NONE));
        }
        pane.on_key(key(KeyCode::Esc, KeyModifiers::NONE));
        assert!(pane.popup.is_none());
        assert_eq!(pane.prompt.text(), "/resume fix");
        pane.on_key(key(KeyCode::Char('x'), KeyModifiers::NONE));
        assert!(matches!(pane.popup, Some(Popup::Sessions(_))));
        pane.set_generating(true);
        assert_eq!(pane.on_key(key(KeyCode::Enter, KeyModifiers::NONE)), None);
    }

    /// Replaying a session renders the echoes, tool headers, and answers the
    /// live stream would have shown.
    #[test]
    fn replayed_turns_render_like_live_ones() {
        let mut pane = Pane::default();
        pane.push(Line::from("banner"));
        let replay = [
            Progress::User("run it".to_string()),
            Progress::Thinking("thinking".to_string()),
            // A replayed tool box: the header, finished empty.
            Progress::ToolStart {
                id: "call_0".to_string(),
                name: "Bash",
                digest: "ls -la".to_string(),
            },
            Progress::ToolOutput {
                id: "call_0".to_string(),
                output: String::new(),
                exit: Some(0),
            },
            Progress::Answer("done".to_string()),
        ];
        pane.extend(replay);

        let screen = render(&mut pane, (60, 20));
        assert!(screen.contains("❯ run it"), "{screen}");
        assert!(screen.contains("Bash(ls -la)"), "{screen}");
        assert!(screen.contains("(no output)"), "{screen}");
        assert!(screen.contains("done"), "{screen}");
        // The thinking run rides below the echo, exactly as a live turn.
        let echo_at = screen.find("❯ run it").unwrap();
        let thinking_at = screen.find("Thinking").unwrap();
        assert!(echo_at < thinking_at, "{screen}");
    }

    /// Validate copy mode isn't exited prematurely.
    #[test]
    fn copy_mode_swallows_keys_and_survives_resizes() {
        let mut pane = Pane::default();
        for i in 0..10 {
            pane.push(Line::from(format!("message {i} aaaa bbbb cccc dddd")));
        }
        render(&mut pane, (14, 24)); // narrow: several wrapped rows each
        pane.on_key(key(KeyCode::Up, KeyModifiers::SHIFT));
        for _ in 0..5 {
            pane.on_key(key(KeyCode::Up, KeyModifiers::NONE));
        }
        assert!(pane.copy.expect("copy cursor").row > 10);
        pane.on_key(key(KeyCode::Char('u'), KeyModifiers::CONTROL)); // eaten
        pane.on_key(key(KeyCode::PageUp, KeyModifiers::NONE));

        render(&mut pane, (80, 24)); // grow
        let cursor = pane.copy.expect("resize left copy mode");
        assert_eq!(cursor.row, pane.transcript.rows().len() - 1, "not clamped");

        assert!(pane.escape());
        assert!(render(&mut pane, (40, 10)).contains("❯ ")); // live again
    }

    /// Space anchors, movement reshapes, Enter ships the selection out and
    /// leaves copy mode.
    #[test]
    fn enter_copies_the_selection_and_exits() {
        let mut pane = Pane::default();
        pane.push(Line::from("abc def"));
        render(&mut pane, (20, 8)); // rows exist before the cursor walks them
        pane.on_key(key(KeyCode::Up, KeyModifiers::SHIFT)); // enter, at (0, 0)
        pane.on_key(key(KeyCode::Right, KeyModifiers::NONE)); // to (0, 1)
        pane.on_key(key(KeyCode::Char(' '), KeyModifiers::NONE)); // anchor
        for _ in 1.."abc def".len() {
            pane.on_key(key(KeyCode::Right, KeyModifiers::NONE)); // to row end
        }
        assert_eq!(
            pane.on_key(key(KeyCode::Enter, KeyModifiers::NONE)),
            Some(PaneEvent::Copy("bc def".into()))
        );
        assert!(pane.copy.is_none());
        assert!(render(&mut pane, (40, 10)).contains("❯ ")); // live again
    }

    /// `q`, a bare Enter, and an anchored-but-empty selection all leave
    /// without clobbering the clipboard.
    #[test]
    fn leaving_without_a_selection_copies_nothing() {
        let mut pane = Pane::default();
        pane.push(Line::from("abc"));
        render(&mut pane, (20, 8));
        pane.on_key(key(KeyCode::Up, KeyModifiers::SHIFT));
        pane.on_key(key(KeyCode::Char(' '), KeyModifiers::NONE));
        assert_eq!(pane.on_key(key(KeyCode::Char('q'), KeyModifiers::NONE)), None);
        assert!(pane.copy.is_none());

        pane.on_key(key(KeyCode::Up, KeyModifiers::SHIFT)); // back in
        assert_eq!(pane.on_key(key(KeyCode::Enter, KeyModifiers::NONE)), None);
        assert!(pane.copy.is_none());

        let mut empty = Pane::default(); // anchored, but no rows to select
        render(&mut empty, (20, 8));
        empty.on_key(key(KeyCode::Up, KeyModifiers::SHIFT));
        empty.on_key(key(KeyCode::Char(' '), KeyModifiers::NONE));
        assert_eq!(empty.on_key(key(KeyCode::Enter, KeyModifiers::NONE)), None);
        assert!(empty.copy.is_none());
    }

    /// The band lights exactly the copied cells: light-black over the
    /// selection, nothing anywhere else on screen.
    #[test]
    fn selection_paints_a_dark_gray_band() {
        let mut pane = Pane::default();
        pane.push(Line::from("abc def"));
        render(&mut pane, (20, 8)); // rows exist before the cursor walks them
        pane.on_key(key(KeyCode::Up, KeyModifiers::SHIFT));
        pane.on_key(key(KeyCode::Home, KeyModifiers::NONE));
        pane.on_key(key(KeyCode::Char(' '), KeyModifiers::NONE)); // anchor (0, 0)
        for _ in 1.."abc def".len() {
            pane.on_key(key(KeyCode::Right, KeyModifiers::NONE)); // to (0, 6)
        }
        let grid = draw_backgrounds(|frame, area| pane.render(frame, area), (20, 8));
        let mut lines = grid.lines();
        assert_eq!(
            lines.next(),
            Some(format!("{}{}", "#".repeat(7), ".".repeat(13)).as_str())
        );
        assert!(lines.all(|line| !line.contains('#')));
    }

    /// A rewrap between Space and Enter re-clamps the anchor with the cursor
    #[test]
    fn anchored_selection_survives_a_rewrap() {
        let mut pane = Pane::default();
        pane.push(Line::from("abcdef"));
        pane.push(Line::from("z"));
        render(&mut pane, (40, 8)); // rows: ["abcdef", "z"]
        pane.on_key(key(KeyCode::Up, KeyModifiers::SHIFT)); // (1, 0)
        pane.on_key(key(KeyCode::Up, KeyModifiers::NONE)); // (0, 0)
        for _ in 0..5 {
            pane.on_key(key(KeyCode::Right, KeyModifiers::NONE)); // (0, 5)
        }
        pane.on_key(key(KeyCode::Char(' '), KeyModifiers::NONE)); // anchor
        pane.on_key(key(KeyCode::Down, KeyModifiers::NONE)); // (1, 0)
        render(&mut pane, (3, 8)); // rows: ["abc", "def", "z"]
        assert_eq!(
            pane.on_key(key(KeyCode::Enter, KeyModifiers::NONE)),
            Some(PaneEvent::Copy("c\nd".into()))
        );
    }

    #[test]
    fn tiny_terminal_renders_without_panic() {
        let mut pane = Pane::default();
        pane.push(Line::from("text"));
        render(&mut pane, (6, 3));
        render(&mut pane, (2, 1));
        render(&mut pane, (1, 0));
        // A queued message renders (or skips) at every size too.
        assert!(pane.control.steer("queued up".to_string()));
        render(&mut pane, (6, 3));
        render(&mut pane, (2, 1));
        render(&mut pane, (1, 0));
        // The badge must skip out-of-bounds bars like `rule` does.
        pane.set_usage(45_000, 0, 3_000);
        render(&mut pane, (6, 3));
        // The /perf line must skip out-of-bounds bars like `rule` does.
        pane.set_perf(Some(" fps 60 ".into()));
        render(&mut pane, (6, 3));
        render(&mut pane, (2, 1));
        render(&mut pane, (1, 0));
    }

    #[test]
    fn ctrl_o_expands_tool_output_in_live_mode_only() {
        let mut pane = Pane::default();
        pane.start_tool("call_0".to_string(), "Bash", "seq 20".to_string());
        let mut output = String::new();
        for i in 0..20 {
            output.push_str("line ");
            output.push_str(&i.to_string());
            output.push('\n');
        }
        pane.finish_tool("call_0", output, Some(0));
        let collapsed = render(&mut pane, (60, 30));
        assert!(collapsed.contains("ctrl+o to expand"));

        pane.on_key(key(KeyCode::Char('o'), KeyModifiers::CONTROL));
        assert!(render(&mut pane, (60, 30)).contains("line 10"));

        // Copy mode swallows Ctrl+O before the control branch.
        pane.on_key(key(KeyCode::Up, KeyModifiers::SHIFT));
        pane.on_key(key(KeyCode::Char('o'), KeyModifiers::CONTROL));
        assert!(pane.copy.is_some(), "still in copy mode");

        pane.escape();
        pane.on_key(key(KeyCode::Char('o'), KeyModifiers::CONTROL));
        assert!(!render(&mut pane, (60, 30)).contains("line 10"));
    }

    #[test]
    fn ctrl_t_toggles_in_live_mode_and_is_inert_in_copy_mode() {
        let mut pane = Pane::default();
        pane.push(Line::from("❯ hi"));
        pane.begin_response();
        pane.append_thinking("visible reasoning");
        pane.append_answer("answer");

        // Hidden by default: the placeholder shows, the reasoning does not.
        let hidden = render(&mut pane, (40, 12));
        assert!(hidden.contains("Thinking"));
        assert!(!hidden.contains("visible reasoning"));

        pane.on_key(key(KeyCode::Char('t'), KeyModifiers::CONTROL));
        let shown = render(&mut pane, (40, 12));
        assert!(shown.contains("visible reasoning"));

        // Copy mode swallows Ctrl+T before the control branch (inert by design).
        pane.on_key(key(KeyCode::Up, KeyModifiers::SHIFT));
        pane.on_key(key(KeyCode::Char('t'), KeyModifiers::CONTROL));
        assert!(pane.escape(), "still in copy mode");
        assert!(render(&mut pane, (40, 12)).contains("visible reasoning"));

        pane.on_key(key(KeyCode::Char('t'), KeyModifiers::CONTROL));
        assert!(render(&mut pane, (40, 12)).contains("Thinking"));
    }

    #[test]
    fn answers_render_markdown_end_to_end() {
        let mut pane = Pane::default();
        pane.extend([
            Progress::User("show off".to_string()),
            Progress::Answer("## Done\n\n- item **one**".to_string()),
        ]);
        let screen = render(&mut pane, (40, 12));
        assert!(screen.contains("Done"), "{screen}");
        assert!(screen.contains("• item one"), "{screen}");
        assert!(!screen.contains("**"), "{screen}");
        assert!(!screen.contains("##"), "{screen}");

        // The emphasized word really is styled on screen.
        let styles = draw_styles(|frame, area| pane.render(frame, area), (40, 12));
        assert!(styles.contains("BBB"), "{styles}");

        // Copy mode copies the rendered form: the markers stay gone.
        render(&mut pane, (40, 12));
        pane.on_key(key(KeyCode::Up, KeyModifiers::SHIFT)); // enter at the tail
        pane.on_key(key(KeyCode::Home, KeyModifiers::NONE)); // the top
        pane.on_key(key(KeyCode::Char(' '), KeyModifiers::NONE)); // anchor
        pane.on_key(key(KeyCode::End, KeyModifiers::NONE)); // the bottom row
        for _ in 0.."• item one".len() {
            pane.on_key(key(KeyCode::Right, KeyModifiers::NONE));
        }
        let Some(PaneEvent::Copy(text)) = pane.on_key(key(KeyCode::Enter, KeyModifiers::NONE))
        else {
            panic!("copy mode ships no selection");
        };
        assert!(text.contains("Done"), "{text}");
        assert!(text.contains("• item one"), "{text}");
        assert!(!text.contains("**"), "{text}");
    }
}
