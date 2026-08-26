//! `Pane` object stores data and rendering logic for the terminal interface.

use ratatui::buffer::Buffer;
use ratatui::crossterm::event::{KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::{Frame, symbols};
use std::path::PathBuf;
use std::time::Instant;
use tart_agents::Progress;
use unicode_segmentation::UnicodeSegmentation;

use crate::clipboard::Selection;
use crate::file_mentions::{self, FilePopup};
use crate::session_picker::{SessionPopup, derive_query as session_query};

pub const PROMPT: &str = "❯ ";
/// Cells before the editor starts (the prompt symbol's width).
const GUTTER: u16 = 2;
/// Spaces a tab renders as.
const TAB_WIDTH: usize = 4;

pub const DIM_STYLE: Style = Style::new().fg(Color::DarkGray);
/// Stands in for a hidden thinking run.
const THINKING_HIDDEN: &str = "[Thinking… ctrl+t to toggle]";

/// A tool box's collapsed output keeps this many head and tail lines.
const TOOL_HEAD: usize = 3;
const TOOL_TAIL: usize = 2;
/// Standin for a tool that produced no output.
const TOOL_NO_OUTPUT: &str = "(no output)";
/// The bullet and name of a tool box header: running, succeeded, or failed.
const TOOL_RUNNING: Style = Style::new().fg(Color::Cyan).add_modifier(Modifier::BOLD);
const TOOL_OK: Style = Style::new().fg(Color::Green).add_modifier(Modifier::BOLD);
const TOOL_ERR: Style = Style::new().fg(Color::Red).add_modifier(Modifier::BOLD);
const PROMPT_STYLE: Style = Style::new().fg(Color::Cyan).add_modifier(Modifier::BOLD);
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
    /// Text chosen in copy mode, ready for the clipboard.
    Copy(String),
    /// A session picked in the `/resume` chooser, ready to swap to.
    Resume(PathBuf),
    /// Esc with nothing open and a turn in flight.
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

/// The pane state before a submitted turn, restored if the turn is cancelled.
struct TurnSnapshot {
    /// Transcript entries before the turn's echo and stream.
    entries: usize,
    /// The draft before it was submitted.
    draft: Editor,
}

/// The TUI interface.
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
    /// The state before the submitted turn (in case we have to cancel a message).
    turn: Option<TurnSnapshot>,
}

impl Pane {
    pub fn new() -> Self {
        Self {
            prompt: Editor::default(),
            transcript: Transcript::default(),
            copy: None,
            popup: None,
            session_dir: None,
            perf: None,
            usage: None,
            context_tokens: None,
            spin: None,
            turn: None,
        }
    }

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
                    &self.prompt,
                    &mut self.popup,
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
            Some(Popup::Files(popup)) => popup.accept(&mut self.prompt),
            Some(Popup::Sessions(sessions)) => {
                if self.spin.is_none()
                    && let Some(path) = sessions.selected_path()
                {
                    self.prompt.clear();
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
                KeyCode::Char('u') => self.prompt.clear(),
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
        // All popups take arrow keys, Tab, & Enter before the events hit the main pane
        if self.popup.is_some()
            && matches!(
                key.code,
                KeyCode::Up | KeyCode::Down | KeyCode::Tab | KeyCode::Enter
            )
        {
            return self.popup_key(key);
        }
        match key.code {
            KeyCode::Enter if key.modifiers.contains(KeyModifiers::ALT) => self.prompt.new_line(),
            // A draft cannot go out mid-generation; Enter keeps it for later.
            KeyCode::Enter if self.spin.is_some() => {}
            KeyCode::Enter => return self.submit().map(PaneEvent::Submit),
            // Esc with nothing to close cancels the turn in flight.
            KeyCode::Esc => {
                if !self.escape() && self.spin.is_some() {
                    return Some(PaneEvent::Cancel);
                }
            }
            KeyCode::Up if key.modifiers.contains(KeyModifiers::SHIFT) => {
                self.copy = Some(CopyCursor::enter(self.transcript.rows().len()));
            }
            KeyCode::Char(c) => self.prompt.insert_char(c),
            KeyCode::Tab => self.prompt.insert_char('\t'),
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
            self.prompt.insert_str(text);
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
            Progress::Thinking(text) => {
                self.append_thinking(&Span::styled(text.clone(), DIM_STYLE));
            }
            Progress::Answer(text) => self.append(&Span::raw(text.clone())),
            Progress::ToolStart { id, name, digest } => {
                self.start_tool(id.clone(), name, digest.clone());
            }
            Progress::ToolOutput { id, output, exit } => {
                self.finish_tool(id, output.clone(), *exit);
            }
            Progress::Usage { input, cached, output } => {
                self.set_usage(*input, *cached, *output);
            }
            // `Progress` is non-exhaustive; later variants need no handling yet.
            _ => {}
        }
    }

    pub fn push<L: Into<Line<'static>>>(&mut self, line: L) {
        self.transcript.push(line);
    }

    /// Append a streaming fragment; see [`Transcript::append`].
    pub fn append(&mut self, span: &Span<'static>) {
        self.transcript.append(span);
    }

    /// Append a streaming chain-of-thought fragment to the current thinking block.
    pub fn append_thinking(&mut self, span: &Span<'static>) {
        self.transcript.append_thinking(span);
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

    /// The status rule's current spinner frame, or None while idle.
    fn spinner(&self) -> Option<&'static str> {
        let elapsed = self.spin?.elapsed().as_millis();
        Some(SPINNER_FRAMES[(elapsed / SPINNER_MS) as usize % SPINNER_FRAMES.len()])
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

    /// Restore the pane to before the cancelled turn, adding a dim cancelled label.
    pub fn cancel_turn(&mut self) {
        let Some(turn) = self.turn.take() else {
            return;
        };
        self.transcript.messages.truncate(turn.entries);
        self.transcript.rows.clear();
        self.transcript.cache.1 = 0;
        self.transcript.run = Some(ThinkingRun {
            start: turn.entries,
            end: turn.entries,
        });
        self.transcript.open = false;
        self.prompt = turn.draft;
        self.push(Span::styled("⎋ cancelled", DIM_STYLE));
    }

    /// Echo the draft into the transcript and clear it.
    fn submit(&mut self) -> Option<String> {
        if self.prompt.text().trim().is_empty() {
            return None;
        }
        // Snapshot the pre-turn state: a cancelled turn restores to here.
        self.turn = Some(TurnSnapshot {
            entries: self.transcript.messages.len(),
            draft: self.prompt.clone(),
        });
        let text = self.prompt.text();
        self.echo(&text);
        self.prompt.clear();
        Some(text)
    }

    /// Echo a submitted line into the transcript, as [`Pane::submit`] renders it.
    pub fn echo(&mut self, text: &str) {
        let mut rows = text.split('\n');
        self.transcript.push(Line::from(vec![
            Span::styled(PROMPT, PROMPT_STYLE),
            Span::raw(rows.next().unwrap_or_default().to_string()),
        ]));
        for continuation in rows {
            self.transcript.push(Line::from(format!("  {continuation}")));
        }
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
        // Live prompt: the "❯ " gutter, then the wrapped draft. The inverted caret
        // cell marks the cursor, so the terminal cursor stays hidden and the prompt
        // viewport can never misplace it.
        let width = prompt_area.width - GUTTER;
        buf.set_span(
            prompt_area.x,
            prompt_area.y,
            &Span::styled(PROMPT, PROMPT_STYLE),
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
        // The popup overlays the transcript, anchored above the top rule.
        match self.popup.as_mut() {
            Some(Popup::Files(popup)) => popup.render(
                frame,
                bar_top,
                "files",
                "↑↓ select · Tab/Enter insert · Esc close",
            ),
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
        rule(buf, bar_top);
        if let Some(perf) = &self.perf {
            // Replace the statusline with the perf counters
            let line = Line::from(Span::styled(format!("{perf} · {} rows", rows.len()), DIM_STYLE));
            // Layout overflow can park a zero-height bar past the last row.
            if bar_bottom.y < buf.area.height {
                buf.set_line(bar_bottom.x, bar_bottom.y, &line, bar_bottom.width);
            }
        } else {
            status_rule(buf, bar_bottom, self.status_text().as_deref(), self.spinner());
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

impl Default for Pane {
    fn default() -> Self {
        Self::new()
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

/// A token count in the status line's compact style: `843`, `45k`, `1.2 M`.
fn token_count(tokens: u64) -> String {
    match tokens {
        0..=999 => tokens.to_string(),
        1_000..=999_999 => format!("{}k", tokens / 1_000),
        _ => format!("{}.{} M", tokens / 1_000_000, tokens % 1_000_000 / 100_000),
    }
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

/// The prompt editor: a multi-line draft with a grapheme caret.
#[derive(Clone)]
pub(crate) struct Editor {
    /// Draft lines; always at least one.
    pub(crate) lines: Vec<String>,
    pub(crate) line: usize,
    /// Grapheme index within `lines[line]`.
    pub(crate) g: usize,
    /// First wrapped row shown when the draft outgrows the prompt box;
    /// re-anchored to the caret by render.
    top: usize,
}

impl Default for Editor {
    fn default() -> Self {
        Self {
            lines: vec![String::new()],
            line: 0,
            g: 0,
            top: 0,
        }
    }
}

impl Editor {
    /// The whole draft, lines joined by '\n'.
    pub(crate) fn text(&self) -> String {
        self.lines.join("\n")
    }

    pub(crate) fn clear(&mut self) {
        self.lines = vec![String::new()];
        self.line = 0;
        self.g = 0;
        self.top = 0;
    }

    /// Graphemes on the current line.
    pub(crate) fn line_len(&self) -> usize {
        graphemes(&self.lines[self.line])
    }

    /// Insert one character; controls are ignored except tab.
    fn insert_char(&mut self, c: char) {
        if !c.is_control() || c == '\t' {
            let line = &mut self.lines[self.line];
            line.insert(g_to_byte(line, self.g), c);
            self.g += 1;
        }
    }

    /// Insert pasted text: CRLF normalized, controls dropped except tab,
    /// newlines split the draft.
    pub(crate) fn insert_str(&mut self, text: &str) {
        for (i, part) in text.lines().enumerate() {
            if i > 0 {
                self.new_line();
            }
            let cleaned: String = part.chars().filter(|c| !c.is_control() || *c == '\t').collect();
            let line = &mut self.lines[self.line];
            line.insert_str(g_to_byte(line, self.g), &cleaned);
            self.g += graphemes(&cleaned);
        }
    }

    /// Split the draft at the caret (Alt+Enter).
    fn new_line(&mut self) {
        let line = &mut self.lines[self.line];
        let tail = line.split_off(g_to_byte(line, self.g));
        self.lines.insert(self.line + 1, tail);
        self.line += 1;
        self.g = 0;
    }

    /// Delete the previous grapheme; at a line start, join with the line
    /// above.
    pub(crate) fn backspace(&mut self) {
        if self.g > 0 {
            let line = &mut self.lines[self.line];
            let start = g_to_byte(line, self.g - 1);
            let end = g_to_byte(line, self.g);
            line.replace_range(start..end, "");
            self.g -= 1;
        } else if self.line > 0 {
            let joined = self.lines.remove(self.line);
            self.line -= 1;
            self.g = graphemes(&self.lines[self.line]);
            self.lines[self.line].push_str(&joined);
        }
    }

    /// One grapheme left, joining across lines.
    pub(crate) fn left(&mut self) {
        if self.g > 0 {
            self.g -= 1;
        } else if self.line > 0 {
            self.line -= 1;
            self.g = self.line_len();
        }
    }

    /// One grapheme right, joining across lines.
    fn right(&mut self) {
        if self.g < self.line_len() {
            self.g += 1;
        } else if self.line + 1 < self.lines.len() {
            self.line += 1;
            self.g = 0;
        }
    }

    /// One logical line up; the grapheme index carries over, clamped.
    fn up(&mut self) {
        if self.line > 0 {
            self.line -= 1;
            self.g = self.g.min(self.line_len());
        }
    }

    /// One logical line down; the grapheme index carries over, clamped.
    fn down(&mut self) {
        if self.line + 1 < self.lines.len() {
            self.line += 1;
            self.g = self.g.min(self.line_len());
        }
    }

    /// To the line start.
    pub(crate) fn home(&mut self) {
        self.g = 0;
    }

    /// To the line end.
    pub(crate) fn end(&mut self) {
        self.g = self.line_len();
    }
}

pub(crate) fn graphemes(s: &str) -> usize {
    s.graphemes(true).count()
}

/// Byte offset of grapheme boundary `g` (the string end when `g` is the count).
pub(crate) fn g_to_byte(s: &str, g: usize) -> usize {
    s.grapheme_indices(true).nth(g).map_or(s.len(), |(i, _)| i)
}

/// The row rendered in place of a hidden thinking run.
fn thinking_placeholder() -> Line<'static> {
    Line::from(Span::styled(THINKING_HIDDEN, DIM_STYLE))
}

/// One dim `⎿` output row of a tool box.
fn tool_row(text: &str) -> Line<'static> {
    Line::from(Span::styled(format!("  ⎿ {text}"), DIM_STYLE))
}

/// A status header and then the output rendered for a running tool call
///
/// Running calls show just the header with a dim ellipsis; finished ones add
/// their output rows, collapsed to [`TOOL_HEAD`] head and [`TOOL_TAIL`] tail
/// lines around a count of the hidden middle unless `expanded`. A call a later
/// one superseded only renders its header.
fn tool_lines(tool: &ToolCall, expanded: bool) -> Vec<Line<'static>> {
    let status = match (&tool.output, tool.exit) {
        (None, _) => TOOL_RUNNING,
        (_, Some(0)) => TOOL_OK,
        _ => TOOL_ERR,
    };

    let mut header = vec![
        Span::styled("● ", status),
        Span::styled(tool.name, status),
        Span::styled(format!("({})", tool.digest), DIM_STYLE),
    ];

    let Some(output) = &tool.output else {
        header.push(Span::styled(" …", DIM_STYLE));
        return vec![Line::from(header)];
    };

    if let Some(c) = tool.exit.filter(|&c| c != 0) {
        header.push(Span::styled(format!(" exit {c}"), TOOL_ERR));
    }

    // A superseded box stays folded down to its header; Ctrl+O governs only
    // the boxes still standing.
    if tool.superseded {
        return vec![Line::from(header)];
    }

    let lines: Vec<_> = output.lines().collect();
    let limit = TOOL_HEAD + TOOL_TAIL;

    let body: Vec<Line<'static>> = match lines.as_slice() {
        [] => vec![tool_row(TOOL_NO_OUTPUT)],
        _ if !expanded && lines.len() > limit => {
            let hidden = lines.len() - limit;
            lines[..TOOL_HEAD]
                .iter()
                .copied()
                .map(tool_row)
                .chain(std::iter::once(tool_row(&format!(
                    "… +{hidden} lines (ctrl+o to expand)"
                ))))
                .chain(lines[lines.len() - TOOL_TAIL..].iter().copied().map(tool_row))
                .collect()
        }
        _ => lines.into_iter().map(tool_row).collect(),
    };

    [vec![Line::from(header)], body].concat()
}

/// The display lines a run of entries renders as.
fn entry_lines(entries: &[Entry], expanded: bool) -> Vec<Line<'static>> {
    entries
        .iter()
        .flat_map(|entry| match entry {
            Entry::Text(line) => std::iter::once(line.clone()).collect::<Vec<_>>(),
            Entry::Tool(tool) => tool_lines(tool, expanded),
        })
        .collect()
}

/// One response's chain-of-thought, as a message-index range into `Transcript::messages`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ThinkingRun {
    /// First message index of the run.
    start: usize,
    /// One past the last thinking message; equals `messages.len()` while the run is
    /// still the transcript tail.
    end: usize,
}

/// One tool invocation, updated in place when its output arrives.
#[derive(Clone)]
struct ToolCall {
    /// Pairs the finishing `ToolOutput` with this start.
    id: String,
    /// Display name: `Bash`, `Read`, or `Edit`.
    name: &'static str,
    /// Argument digest, e.g. `ls -la` or `src/main.rs:10-50`.
    digest: String,
    /// Combined output; `None` while the call is still running.
    output: Option<String>,
    /// The process exit code, for the status color.
    exit: Option<i32>,
    /// A later invocation folded this box down to its header line.
    superseded: bool,
}

/// One transcript message: a text line or a live tool invocation.
#[derive(Clone)]
enum Entry {
    Text(Line<'static>),
    Tool(ToolCall),
}

/// One view's message log, kept pre-wrapped to the display width:
///
/// Call `sync` to re-wrap the new content. The current response's thinking lives in
/// [`Transcript::run`]. While hidden its messages render as a placeholder block, and
/// are drained when the next response begins.
#[derive(Default)]
struct Transcript {
    messages: Vec<Entry>,
    /// Whether the last message is still open for `append` runs.
    open: bool,
    rows: Vec<Line<'static>>,
    /// (width, how many *visible* messages `rows` already contains).
    cache: (usize, usize),
    /// The current response's chain-of-thought, if one has begun. Tool starts
    /// rotate it below themselves, so it always rides under the tool boxes.
    run: Option<ThinkingRun>,
    /// Whether the thinking run renders; sticky across turns. Starts hidden.
    show_thinking: bool,
    /// Whether tool outputs render in full; sticky, flipped by Ctrl+O. Starts
    /// collapsed.
    show_tool_output: bool,
}

impl Transcript {
    /// Append a committed line, ending any append-run.
    fn push(&mut self, line: impl Into<Line<'static>>) {
        self.messages.push(Entry::Text(line.into()));
        self.open = false;
    }

    /// Record a tool invocation's start; it renders as a running header until
    /// finished. The call supersedes every finished box before it, folding each
    /// down to its header line, and rotates the thinking run below itself so
    /// the chain-of-thought always renders under the tool boxes.
    fn start_tool(&mut self, id: String, name: &'static str, digest: String) {
        let fold = self.messages.iter().position(
            |entry| matches!(entry, Entry::Tool(tool) if tool.output.is_some() && !tool.superseded),
        );
        // The rotation below also moves the run's messages, so the stale-row
        // span starts at whichever of the two comes first.
        let run_start = self.run.filter(|run| run.start < run.end).map(|run| run.start);
        let stale = [fold, run_start].into_iter().flatten().min();
        // Rewind before flipping the flags: the stale-row count must reflect the
        // bodies the flags are about to hide.
        if let Some(index) = stale {
            self.rewind(index);
        }
        for entry in &mut self.messages {
            if let Entry::Tool(tool) = entry
                && tool.output.is_some()
            {
                tool.superseded = true;
            }
        }
        self.messages.push(Entry::Tool(ToolCall {
            id,
            name,
            digest,
            output: None,
            exit: None,
            superseded: false,
        }));
        // Late thinking fragments then extend the run in place, under the
        // boxes, instead of splicing back above them; an empty run moves only
        // its markers, so thinking that starts after the call still opens
        // below the box.
        if let Some(run) = &mut self.run {
            let span = run.end - run.start;
            self.messages[run.start..].rotate_left(span);
            run.end = self.messages.len();
            run.start = run.end - span;
        }
        self.open = false;
    }

    /// Fill in the pending invocation with `id`, then refold from its box.
    fn finish_tool(&mut self, id: &str, output: String, exit: Option<i32>) {
        let Some(index) = self.messages.iter().rposition(
            |entry| matches!(entry, Entry::Tool(tool) if tool.output.is_none() && tool.id == id),
        ) else {
            return;
        };
        // The box folds differently once tools finished, so the rows `sync` folded
        // `messages[index..]` are stale. Rewind the fold point so the next `sync`
        // refolds just the tail.
        self.rewind(index);
        let Some(Entry::Tool(tool)) = self.messages.get_mut(index) else {
            return;
        };
        tool.output = Some(output);
        tool.exit = exit;
    }

    /// Drop the cached rows of `messages[index..]` so the next `sync` refolds
    /// them. A hidden thinking run inside the span forces a full refold: the
    /// placeholder means the raw row count would not match what `sync` folded.
    fn rewind(&mut self, index: usize) {
        if self.cache.0 == 0 || index >= self.cache.1 {
            return;
        }
        if (index..self.cache.1).any(|i| self.thinking_hidden(i)) {
            // Hidden thinking requires us to refold EVERYTHING for correctness.
            self.rows.clear();
            self.cache.1 = 0;
        } else {
            let stale = wrap_lines(
                &entry_lines(&self.messages[index..self.cache.1], self.show_tool_output),
                self.cache.0,
            )
            .len();
            self.rows.truncate(self.rows.len() - stale);
            self.cache.1 = index;
        }
    }

    /// Resolve every still-running invocation as failed to prevent stuck boxes.
    fn fail_pending(&mut self, reason: &str) {
        let mut failed = false;
        for entry in &mut self.messages {
            if let Entry::Tool(tool) = entry
                && tool.output.is_none()
            {
                tool.output = Some(reason.to_string());
                tool.exit = None;
                failed = true;
            }
        }
        if failed {
            self.rows.clear();
            self.cache.1 = 0;
        }
    }

    /// Expand or collapse every tool output.
    fn toggle_expand(&mut self) {
        if self.messages.iter().any(|entry| matches!(entry, Entry::Tool(_))) {
            self.show_tool_output = !self.show_tool_output;
            self.rows.clear();
            self.cache.1 = 0;
        }
    }

    /// The last message when it is text, for append-run gluing.
    fn text_last(&self) -> Option<&Line<'static>> {
        match self.messages.last() {
            Some(Entry::Text(line)) => Some(line),
            _ => None,
        }
    }

    /// The last message when it is text, mutably, for span extension.
    fn text_last_mut(&mut self) -> Option<&mut Line<'static>> {
        match self.messages.last_mut() {
            Some(Entry::Text(line)) => Some(line),
            _ => None,
        }
    }

    /// Append a streaming fragment, gluing onto the previous fragment while style matches.
    ///
    /// Newlines in the text end the current line.
    fn append(&mut self, span: &Span<'static>) {
        if self.run.is_some_and(|run| run.end == self.messages.len()) {
            self.break_line();
        }
        for (i, part) in span.content.split('\n').enumerate() {
            (i > 0).then(|| self.break_line());
            if !part.is_empty() {
                self.append_fragment(Span::styled(part.to_string(), span.style));
            }
        }
    }

    /// Glue one unbroken fragment onto the transcript.
    fn append_fragment(&mut self, span: Span<'static>) {
        let glue = self.open
            && self
                .text_last()
                .is_some_and(|line| line.spans.last().is_some_and(|last| last.style == span.style));
        if glue {
            // The cache already counted the line being extended: drop its stale rows
            // and hand the message back for the next sync (unless it is hidden
            // thinking, whose rows were never in `rows`)
            if self.cache.1 == self.messages.len()
                && self.cache.0 > 0
                && !self.thinking_hidden(self.messages.len() - 1)
            {
                let stale = wrap_lines(
                    &entry_lines(&self.messages[self.messages.len() - 1..], self.show_tool_output),
                    self.cache.0,
                )
                .len();
                self.rows.truncate(self.rows.len() - stale);
                self.cache.1 -= 1;
            }
            if let Some(last) = self.text_last_mut().and_then(|l| l.spans.last_mut()) {
                // Extend the last matching span if available to save memory.
                last.content.to_mut().push_str(&span.content);
            }
        } else {
            self.messages.push(Entry::Text(Line::from(span)));
            self.open = true;
        }
    }

    /// End the current append-run; later appends start a fresh line.
    fn break_line(&mut self) {
        self.open = false;
    }

    /// Append a chain-of-thought fragment into the current thinking run.
    fn append_thinking(&mut self, span: &Span<'static>) {
        // Open a run if none exists yet (e.g. thinking after a `/clear`).
        let at = self.messages.len();
        self.run.get_or_insert(ThinkingRun { start: at, end: at });
        // The answer already started: move the fragment back above it
        let late = self.run.is_some_and(|run| run.end < self.messages.len());
        if late && span.content.split('\n').all(str::is_empty) {
            // Nothing to splice back: leave the wrap cache and the answer run alone
            return;
        }
        let before = self.messages.len();
        // Skip gluing thinking if we have a late thinking fragment or the run is empty
        if late || self.run.is_some_and(|run| run.start == run.end) {
            self.break_line();
        }
        for (i, part) in span.content.split('\n').enumerate() {
            (i > 0).then(|| self.break_line());
            if !part.is_empty() {
                self.append_fragment(Span::styled(part.to_string(), span.style));
            }
        }
        let Some(run) = &mut self.run else {
            return;
        };
        if late {
            let end = run.end;
            let count = self.messages.len() - before;
            // Rotate the fresh messages back above the tail that followed the run.
            self.messages[end..].rotate_left(before - end);
            run.end = end + count;
            // Rewrap from the splice point, dropping only the rows the already-folded
            // tail occupied.
            if self.cache.0 > 0 && self.cache.1 > end {
                let stale = wrap_lines(
                    &entry_lines(
                        &self.messages[run.end..self.cache.1 + count],
                        self.show_tool_output,
                    ),
                    self.cache.0,
                )
                .len();
                self.rows.truncate(self.rows.len() - stale);
            }
            self.cache.1 = self.cache.1.min(end);
        } else {
            run.end = self.messages.len();
        }
    }

    /// Show or hide the current response's chain-of-thought.
    fn toggle_thinking(&mut self) {
        self.show_thinking = !self.show_thinking;
        if self.run.is_some_and(|run| run.start < run.end) {
            // The run's rows sit mid-`rows`; rebuild from scratch.
            self.rows.clear();
            self.cache.1 = 0;
        }
    }

    /// Retire the previous response's thinking and open a fresh, empty run.
    fn begin_response(&mut self) {
        if let Some(run) = self.run.take()
            && run.start < run.end
        {
            self.messages.drain(run.start..run.end);
        }
        // Rows that included the drained messages are stale; rewrapping once
        // per turn is fine.
        self.rows.clear();
        self.cache.1 = 0;
        // Never glue onto the retired turn's tail.
        self.open = false;
        let at = self.messages.len();
        self.run = Some(ThinkingRun { start: at, end: at });
    }

    /// Whether message `i` belongs to the hidden thinking run.
    fn thinking_hidden(&self, i: usize) -> bool {
        !self.show_thinking && self.run.is_some_and(|run| i >= run.start && i < run.end)
    }

    /// Drop every message and reset our caches, persisting the thinking preference
    fn clear(&mut self) {
        self.messages.clear();
        self.rows.clear();
        self.cache.1 = 0;
        self.open = false;
        self.run = None;
    }

    /// Wrap the visible messages not yet folded into `rows`; a width change rewraps
    /// everything. A hidden thinking run renders as its placeholder.
    fn sync(&mut self, width: usize) -> &[Line<'static>] {
        if self.cache.0 != width {
            self.rows.clear();
            self.cache = (width, 0);
        }
        let expanded = self.show_tool_output;
        let done = self.cache.1;
        if let Some(run) = &self.run
            && !self.show_thinking
            && run.start < run.end
        {
            let (start, end) = (run.start, run.end);
            self.rows.extend(wrap_lines(
                &entry_lines(&self.messages[done.min(start)..start], expanded),
                width,
            ));
            // If the fold point is before the run we need to render, otherwise skip.
            if start >= done {
                self.rows.extend(wrap_lines(&[thinking_placeholder()], width));
            }
            self.rows.extend(wrap_lines(
                &entry_lines(&self.messages[end.max(done)..], expanded),
                width,
            ));
        } else {
            self.rows
                .extend(wrap_lines(&entry_lines(&self.messages[done..], expanded), width));
        }
        self.cache.1 = self.messages.len();
        &self.rows
    }

    /// The wrapped rows; current as of the last `sync`.
    fn rows(&self) -> &[Line<'static>] {
        &self.rows
    }
}

/// Cursor in copy mode: `row`/`col` address a cell in the wrapped rows.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct CopyCursor {
    row: usize,
    col: usize,
    /// Selection start cell; `None` until Space anchors one.
    anchor: Option<(usize, usize)>,
    /// `usize::MAX` means we start at the end, and position is clamped on render
    top: usize,
    /// Rows the last render showed.
    visible: usize,
}

impl CopyCursor {
    /// Enter copy mode with the cursor on the last row.
    fn enter(rows_len: usize) -> Self {
        Self {
            row: rows_len.saturating_sub(1),
            col: 0,
            anchor: None,
            top: usize::MAX,
            visible: 0,
        }
    }
}

/// The cursor after one key step, clamped to the transcript edges.
fn moved(rows: &[Line<'static>], cursor: CopyCursor, key: KeyCode) -> CopyCursor {
    if rows.is_empty() {
        return CopyCursor { row: 0, col: 0, ..cursor };
    }
    let last = rows.len() - 1;
    let mut c = cursor;
    match key {
        KeyCode::Up => c.row = c.row.saturating_sub(1),
        KeyCode::Down => c.row = (c.row + 1).min(last),
        KeyCode::Left => {
            if c.col > 0 {
                c.col -= 1;
            } else if c.row > 0 {
                c.row -= 1;
                c.col = rows[c.row].width().saturating_sub(1);
            }
        }
        KeyCode::Right => {
            if c.col + 1 < rows[c.row].width() {
                c.col += 1;
            } else if c.row < last {
                c.row += 1;
                c.col = 0;
            }
        }
        KeyCode::PageUp => c.row = c.row.saturating_sub(c.visible.max(1)),
        KeyCode::PageDown => c.row = (c.row + c.visible.max(1)).min(last),
        KeyCode::Home => c.row = 0,
        KeyCode::End => c.row = last,
        _ => return cursor,
    }
    c.col = c.col.min(rows[c.row].width().saturating_sub(1));
    c
}

/// First transcript row to render, used to anchor the prompt viewport.
fn window_top(rows_len: usize, visible: usize, anchor: Option<(usize, usize)>) -> usize {
    let max_top = rows_len.saturating_sub(visible);
    anchor.map_or(max_top, |(row, top)| {
        top.min(max_top)
            .min(row)
            .max(row.saturating_add(1).saturating_sub(visible))
    })
}

/// A cell clamped into the wrapped rows: the last row, that row's last cell.
#[inline]
fn clamp_cell(rows: &[Line<'static>], row: usize, col: usize) -> (usize, usize) {
    let row = row.min(rows.len().saturating_sub(1));
    let col = col.min(rows.get(row).map_or(0, |row| row.width().saturating_sub(1)));
    (row, col)
}

/// One wrapped row under construction: (&'a grapheme, style, cell width).
type Row<'a> = Vec<(&'a str, Style, usize)>;

/// Greedy word wrap (break before a word when it fits on the next row,
/// hard-break when it does not), preserving span styles.
struct Wrapper<'a> {
    /// Row cell budget; at least 1.
    width: usize,
    rows: Vec<Line<'static>>,
    row: Row<'a>,
    row_width: usize,
    /// Index into `row` where the current word began.
    word_start: Option<usize>,
}

impl<'a> Wrapper<'a> {
    fn new(width: usize) -> Self {
        Self {
            width: width.max(1),
            rows: Vec::new(),
            row: Vec::new(),
            row_width: 0,
            word_start: None,
        }
    }

    /// Index the row under construction will get once finished.
    fn row_index(&self) -> usize {
        self.rows.len()
    }

    /// Cell column where the next grapheme would land on the current row.
    fn col(&self) -> usize {
        self.row_width
    }

    /// Add one rendered cell; `sym` is a single space for expanded tabs.
    fn push(&mut self, sym: &'a str, style: Style) {
        // Single-byte symbols can be printed without looking up width.
        let gw = if sym.len() == 1 { 1 } else { Span::raw(sym).width() };
        let space = sym == " ";
        if self.row_width + gw > self.width && !self.row.is_empty() && gw > 0 {
            if space {
                // The wrapping space is not carried to the next row.
                self.emit_row();
                self.word_start = None;
                return;
            }
            // Break before the current word so it moves down intact; fall
            // back to a hard break when the word fills the row.
            if let Some(split) = self.word_start.filter(|&ws| ws > 0) {
                let tail = self.row.split_off(split);
                let tail_width = tail.iter().map(|g| g.2).sum();
                self.emit_row();
                self.row = tail;
                self.row_width = tail_width;
            } else {
                self.emit_row();
            }
            self.word_start = Some(0);
            // A word longer than a full row keeps hard-breaking.
            while self.row_width + gw > self.width && !self.row.is_empty() {
                self.emit_row();
            }
        }
        if space {
            self.word_start = None;
        } else if self.word_start.is_none() {
            self.word_start = Some(self.row.len());
        }
        self.row.push((sym, style, gw));
        self.row_width += gw;
    }

    /// End the current row at a logical boundary (newline or message end).
    fn hard_break(&mut self) {
        self.emit_row();
        self.word_start = None;
    }

    /// Drain the row under construction into a `Line`, merging adjacent
    /// graphemes that share a style back into spans.
    fn emit_row(&mut self) {
        let mut spans: Vec<Span<'static>> = Vec::new();
        for (sym, style, _) in std::mem::take(&mut self.row) {
            match spans.last_mut() {
                Some(last) if last.style == style => last.content.to_mut().push_str(sym),
                // Allocate an owned string once per run
                _ => spans.push(Span::styled(sym.to_owned(), style)),
            }
        }
        self.rows.push(Line::from(spans));
        self.row_width = 0;
    }
}

/// Feed one grapheme: a tab becomes `TAB_WIDTH` spaces, other control characters are
/// invisible, anything else renders as itself.
fn feed<'a>(wrapper: &mut Wrapper<'a>, grapheme: &'a str, style: Style) {
    match grapheme {
        "\t" => (0..TAB_WIDTH).for_each(|_| wrapper.push(" ", style)),
        _ if !grapheme.chars().any(char::is_control) => wrapper.push(grapheme, style),
        _ => {} // Ignore unhandled control characters
    }
}

/// Wrap styled transcript lines to display rows.
fn wrap_lines(messages: &[Line<'static>], width: usize) -> Vec<Line<'static>> {
    let mut wrapper = Wrapper::new(width);
    for line in messages {
        for (grapheme, style) in line.spans.iter().flat_map(|span| {
            span.content
                .graphemes(true)
                .map(|g| (g, line.style.patch(span.style)))
        }) {
            feed(&mut wrapper, grapheme, style);
        }
        wrapper.hard_break();
    }
    wrapper.rows
}

/// The draft wrapped for display, plus the caret's cell in it.
struct PromptLayout {
    rows: Vec<Line<'static>>,
    caret_row: usize,
    /// May equal its row's width; paint sites clamp.
    caret_col: usize,
}

/// Wrap the draft and locate the caret's cell
/// The carat should be at the boundary before grapheme `cursor.1` of line `cursor.0`
fn wrap_draft(lines: &[String], cursor: (usize, usize), width: usize) -> PromptLayout {
    let mut wrapper = Wrapper::new(width);
    let mut caret = (0, 0);
    let (cl, mut gc) = cursor;
    let cl = cl.min(lines.len().saturating_sub(1));
    for (li, line) in lines.iter().enumerate() {
        // Only the caret line needs its grapheme count; skip the extra scan elsewhere.
        let mut count = 0;
        if li == cl {
            count = line.graphemes(true).count();
            gc = gc.min(count);
        }
        for (gi, grapheme) in line.graphemes(true).enumerate() {
            if li == cl && gi == gc {
                caret = (wrapper.row_index(), wrapper.col());
            }
            feed(&mut wrapper, grapheme, Style::new());
        }
        if li == cl && gc == count {
            caret = (wrapper.row_index(), wrapper.col());
        }
        wrapper.hard_break();
    }
    PromptLayout {
        rows: wrapper.rows,
        caret_row: caret.0,
        caret_col: caret.1,
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, reason = "test assertions")]

    use super::*;
    use crate::testutil::{draw, draw_backgrounds};

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
        let mut pane = Pane::new();
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
        let mut bare = Pane::new();
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

    #[test]
    fn enter_while_generating_keeps_the_draft() {
        let mut pane = Pane::default();
        for c in ['h', 'i'] {
            pane.on_key(key(KeyCode::Char(c), KeyModifiers::NONE));
        }
        pane.set_generating(true);
        // Enter neither echoes nor clears; Alt+Enter still edits the draft.
        assert_eq!(pane.on_key(key(KeyCode::Enter, KeyModifiers::NONE)), None);
        pane.on_key(key(KeyCode::Enter, KeyModifiers::ALT));
        assert_eq!(pane.prompt.lines.len(), 2);
        assert_eq!(pane.prompt.text(), "hi\n");
        assert!(message_texts(&pane.transcript).is_empty());
        // Once the model is done, Enter submits the intact draft.
        pane.set_generating(false);
        assert_eq!(
            pane.on_key(key(KeyCode::Enter, KeyModifiers::NONE)),
            Some(PaneEvent::Submit("hi\n".into()))
        );
        assert_eq!(pane.prompt.text(), "");
    }

    /// Cancelling a turn restores the pane to before the message.
    #[test]
    fn cancel_turn_restores_the_pre_turn_state() {
        let mut pane = Pane::new();
        pane.push(Line::from("earlier"));
        pane.begin_response();
        pane.append(&Span::raw("earlier answer"));

        // The next turn, as main drives it: submit echoes the draft, the
        // response begins, then the stream and tools arrive.
        pane.on_paste("write a story");
        assert_eq!(
            pane.on_key(key(KeyCode::Enter, KeyModifiers::NONE)),
            Some(PaneEvent::Submit("write a story".into()))
        );
        pane.begin_response();
        pane.set_generating(true);
        pane.append_thinking(&Span::styled("thinking", DIM_STYLE));
        pane.append(&Span::raw("Once upon"));
        pane.start_tool("call_0".to_string(), "Bash", "sleep 5".to_string());

        pane.cancel_turn();

        let screen = render(&mut pane, (60, 20));
        assert!(screen.contains("earlier answer"), "{screen}");
        assert!(screen.contains("⎋ cancelled"), "{screen}");
        assert!(!screen.contains("Once upon"), "{screen}");
        assert!(!screen.contains("sleep 5"), "{screen}");
        assert_eq!(pane.prompt.text(), "write a story");
        // The restored log folds cleanly.
        assert_eq!(
            texts(&pane.transcript.rows),
            texts(&wrap_lines(
                &entry_lines(&pane.transcript.messages, pane.transcript.show_tool_output),
                pane.transcript.cache.0
            ))
        );
    }

    fn texts(lines: &[Line<'static>]) -> Vec<String> {
        lines
            .iter()
            .map(|line| line.spans.iter().map(|span| span.content.as_ref()).collect())
            .collect()
    }

    #[test]
    fn echo_renders_the_prompt_and_indents_continuations() {
        let mut pane = Pane::default();

        pane.echo("first\nsecond");

        assert_eq!(message_texts(&pane.transcript), ["❯ first", "  second"]);
    }

    struct Scratch(PathBuf);

    impl Scratch {
        /// Create the guard under the temp directory.
        fn new(tag: &str) -> Self {
            let path = std::env::temp_dir().join(format!("tart-pane-{tag}-{}", std::process::id()));
            std::fs::create_dir_all(&path).unwrap();
            Self(path)
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    /// A `/resume` line opens the session chooser; Enter swaps to the picked
    /// session, and the chooser never swaps under a running turn.
    #[test]
    fn a_resume_line_opens_the_chooser_and_enter_swaps() {
        let root = Scratch::new("resume");
        let project = PathBuf::from("/tmp/proj");
        let dir = root.0.join("tmp-proj");
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("20260101-000000.jsonl");
        std::fs::write(
            &file,
            "{\"type\":\"message\",\"role\":\"system\",\"content\":\"s\"}\n\
             {\"type\":\"message\",\"role\":\"user\",\"content\":\"fix the login flow\"}\n",
        )
        .unwrap();
        let mut pane = Pane::new();
        pane.set_session_dir(root.0.clone(), project);

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
    /// live stream would have shown — and a later cancel cannot rewind into it.
    #[test]
    fn replayed_turns_render_like_live_ones() {
        let mut pane = Pane::new();
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

        // A live turn that gets cancelled rewinds to its own start, never into
        // the replayed history. (The draft returns to the prompt editor.)
        pane.on_paste("cancel me");
        pane.on_key(key(KeyCode::Enter, KeyModifiers::NONE));
        pane.set_generating(true);
        pane.cancel_turn();
        let screen = render(&mut pane, (60, 20));
        assert!(screen.contains("❯ run it"), "{screen}");
        assert!(screen.contains("done"), "{screen}");
        assert!(
            !message_texts(&pane.transcript)
                .iter()
                .any(|text| text.contains("cancel me"))
        );
    }

    /// The messages the transcript renders: the log with a hidden run
    /// replaced by its placeholder row.
    fn visible(t: &Transcript) -> Vec<Line<'static>> {
        let mut entries = t.messages.clone();
        if let Some(run) = t.run
            && !t.show_thinking
            && run.start < run.end
        {
            entries.splice(
                run.start..run.end,
                std::iter::once(Entry::Text(thinking_placeholder())),
            );
        }
        entry_lines(&entries, t.show_tool_output)
    }

    /// The text of the transcript's plain messages, tool boxes aside.
    fn message_texts(t: &Transcript) -> Vec<String> {
        t.messages
            .iter()
            .filter_map(|entry| match entry {
                Entry::Text(line) => Some(
                    line.spans
                        .iter()
                        .map(|span| span.content.as_ref())
                        .collect::<String>(),
                ),
                Entry::Tool(_) => None,
            })
            .collect()
    }

    /// Start a pending `Bash(echo hi)` invocation, as the pane would on a
    /// `ToolStart` event.
    fn start_bash(t: &mut Transcript, id: &str) {
        t.start_tool(id.to_string(), "Bash", "echo hi".to_string());
    }

    #[test]
    fn wraps_words_hard_breaks_and_keeps_non_ascii_spaces() {
        let wrap =
            |text: &str, width: usize| texts(&wrap_lines(&[Line::from(text.to_string())], width));
        assert_eq!(wrap("aaa bbb ccc", 7), ["aaa bbb", "ccc"]);
        assert_eq!(wrap("aaaaaaaa", 3), ["aaa", "aaa", "aa"]); // hard break
        assert_eq!(wrap("", 10).len(), 1); // an empty message keeps a row
        assert_eq!(wrap("aaa\u{a0}bbb", 4), ["aaa\u{a0}", "bbb"]);
        assert_eq!(wrap("aa\tbb", 5), ["aa   ", "bb"]);
        assert_eq!(wrap("a\rb", 10), ["ab"]);
    }

    #[test]
    fn draft_caret_lands_in_the_wrapped_rows() {
        let at = |g: usize| {
            let layout = wrap_draft(&["hello world".to_string()], (0, g), 5);
            (layout.caret_row, layout.caret_col)
        };
        assert_eq!(at(0), (0, 0));
        assert_eq!(at(7), (1, 1)); // inside "world"
        assert_eq!(at(11), (1, 5)); // brim-full end
        assert_eq!(at(5), (0, 5)); // before the dropped space: row end
    }

    /// Fragments of one streaming run land on one growing line, and a style change
    /// starts a new one without any caller-side state.
    #[test]
    fn appends_glue_by_style() {
        let dim = Style::new().fg(Color::DarkGray);
        let mut transcript = Transcript::default();
        transcript.push(Line::from("prompt"));
        transcript.append(&Span::raw("Hel"));
        transcript.append(&Span::raw("lo "));
        transcript.append(&Span::raw("world"));
        transcript.append(&Span::styled(" (thinking)", dim));
        transcript.append(&Span::styled(" more", dim));
        assert_eq!(
            message_texts(&transcript),
            ["prompt", "Hello world", " (thinking) more"]
        );

        // `push` and `break_line` both end the run.
        transcript.push(Line::from("committed"));
        transcript.append(&Span::raw("after"));
        transcript.break_line();
        transcript.append(&Span::raw("again"));
        assert_eq!(
            message_texts(&transcript),
            [
                "prompt",
                "Hello world",
                " (thinking) more",
                "committed",
                "after",
                "again"
            ]
        );
    }

    /// Whatever pushes, appends, or width changes happen between renders, the cached
    /// rows always equal a full re-wrap
    #[test]
    fn wrap_cache_always_matches_a_full_rewrap() {
        let mut transcript = Transcript::default();
        for i in 0..5 {
            transcript.push(Line::from(format!("message {i} aaaa bbbb cccc dddd")));
        }
        let assert_fresh = |transcript: &Transcript| {
            assert_eq!(
                texts(&transcript.rows),
                texts(&wrap_lines(
                    &entry_lines(&transcript.messages, transcript.show_tool_output),
                    transcript.cache.0
                ))
            );
        };
        transcript.sync(20);
        assert_eq!(transcript.cache, (20, 5));
        assert_fresh(&transcript);

        transcript.push(Line::from("tail")); // between renders
        transcript.sync(20);
        assert_fresh(&transcript);

        transcript.sync(80); // width change rebuilds
        assert_eq!(transcript.cache, (80, 6));
        assert_fresh(&transcript);

        transcript.append(&Span::raw("streaming aaaa bbbb")); // glued run
        transcript.sync(80);
        assert_fresh(&transcript);
        transcript.append(&Span::raw(" cccc dddd"));
        transcript.sync(80);
        assert_fresh(&transcript);

        // Tool boxes mutate mid-log: the running header, then the finished block.
        start_bash(&mut transcript, "call_0");
        transcript.sync(80);
        assert_fresh(&transcript);
        transcript.finish_tool("call_0", "one\ntwo\nthree\n".to_string(), Some(0));
        transcript.sync(80);
        assert_fresh(&transcript);
        transcript.toggle_expand();
        transcript.sync(80);
        assert_fresh(&transcript);

        transcript.clear(); // hidden pane: re-push to the same count
        for i in 0..6 {
            transcript.push(Line::from(format!("fresh {i}")));
        }
        transcript.sync(80);
        assert_fresh(&transcript);
        assert!(!texts(&transcript.rows).iter().any(|row| row.contains("aaaa")));
    }

    #[test]
    fn viewport_and_copy_cursor_math() {
        assert_eq!(window_top(10, 5, None), 5); // live pins the tail
        assert_eq!(window_top(10, 5, Some((9, usize::MAX))), 5); // enter
        assert_eq!(window_top(10, 5, Some((3, 5))), 3); // dragged up
        assert_eq!(window_top(10, 5, Some((8, 2))), 4); // dragged down
        assert_eq!(window_top(10, 5, Some((4, 2))), 2); // inside: stays

        let rows = wrap_lines(&[Line::from("abc"), Line::from("de")], 10);
        let cur = |row, col| CopyCursor {
            row,
            col,
            anchor: None,
            top: 0,
            visible: 2,
        };
        assert_eq!(moved(&rows, cur(0, 0), KeyCode::Right), cur(0, 1));
        assert_eq!(moved(&rows, cur(1, 0), KeyCode::Left), cur(0, 2)); // wraps
        assert_eq!(moved(&rows, cur(1, 0), KeyCode::PageUp), cur(0, 0));
        assert_eq!(moved(&rows, cur(0, 0), KeyCode::PageDown), cur(1, 0));
        assert_eq!(moved(&rows, cur(1, 0), KeyCode::Char('x')), cur(1, 0));

        // A move reshapes a selection; it never drops the anchor.
        let mut anchored = cur(0, 0);
        anchored.anchor = Some((1, 1));
        assert_eq!(moved(&rows, anchored, KeyCode::Right).anchor, Some((1, 1)));
    }

    /// Editing operates on graphemes: a line-start backspace joins lines, the caret
    /// caret rides a join, and control characters never enter the draft.
    #[test]
    fn editing_operates_on_graphemes() {
        let mut pane = Pane::new();
        pane.on_paste("日本\n語");
        pane.on_key(key(KeyCode::Home, KeyModifiers::NONE));
        pane.prompt.backspace(); // joins the lines at the boundary
        assert_eq!(pane.prompt.text(), "日本語");
        assert_eq!((pane.prompt.line, pane.prompt.g), (0, 2));
        pane.prompt.insert_char('\u{7}'); // control: ignored
        assert_eq!(pane.prompt.text(), "日本語");

        // Left/right cross line joins; the family emoji is one step.
        let mut pane = Pane::new();
        pane.on_paste("ab\n🙋‍♂️x");
        pane.prompt.left();
        pane.prompt.left();
        assert_eq!((pane.prompt.line, pane.prompt.g), (1, 0)); // line start
        pane.prompt.left();
        assert_eq!((pane.prompt.line, pane.prompt.g), (0, 2)); // joins above
        pane.prompt.right();
        assert_eq!((pane.prompt.line, pane.prompt.g), (1, 0)); // and back
    }

    /// Validate copy mode isn't exited prematurely.
    #[test]
    fn copy_mode_swallows_keys_and_survives_resizes() {
        let mut pane = Pane::new();
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
        let mut pane = Pane::new();
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
        let mut pane = Pane::new();
        pane.push(Line::from("abc"));
        render(&mut pane, (20, 8));
        pane.on_key(key(KeyCode::Up, KeyModifiers::SHIFT));
        pane.on_key(key(KeyCode::Char(' '), KeyModifiers::NONE));
        assert_eq!(pane.on_key(key(KeyCode::Char('q'), KeyModifiers::NONE)), None);
        assert!(pane.copy.is_none());

        pane.on_key(key(KeyCode::Up, KeyModifiers::SHIFT)); // back in
        assert_eq!(pane.on_key(key(KeyCode::Enter, KeyModifiers::NONE)), None);
        assert!(pane.copy.is_none());

        let mut empty = Pane::new(); // anchored, but no rows to select
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
        let mut pane = Pane::new();
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
        let mut pane = Pane::new();
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
        let mut pane = Pane::new();
        pane.push(Line::from("text"));
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
    fn wrap_cache_matches_a_full_rewrap_while_hidden() {
        let mut t = Transcript::default();
        assert!(!t.show_thinking, "thinking starts hidden");
        let assert_fresh = |t: &Transcript| {
            assert_eq!(texts(&t.rows), texts(&wrap_lines(&visible(t), t.cache.0)));
        };

        t.push(Line::from("❯ echo"));
        t.begin_response();
        t.sync(20);
        assert_fresh(&t);
        t.append_thinking(&Span::styled("hmm aaaa bbbb", DIM_STYLE));
        t.sync(20);
        assert_fresh(&t);
        t.append_thinking(&Span::styled(" cccc dddd", DIM_STYLE)); // glued
        t.sync(20);
        assert_fresh(&t);
        t.append_thinking(&Span::styled("line two\nline three", DIM_STYLE));
        t.sync(20);
        assert_fresh(&t);
        t.append(&Span::raw("the answer aaaa bbbb"));
        t.sync(20);
        assert_fresh(&t);

        // A tool box lands after the answer, finishes, and round-two reasoning
        // splices back above both.
        start_bash(&mut t, "c0");
        t.sync(20);
        assert_fresh(&t);
        t.finish_tool("c0", "out aaaa".to_string(), Some(0));
        t.sync(20);
        assert_fresh(&t);
        t.append_thinking(&Span::styled(" mid", DIM_STYLE));
        t.sync(20);
        assert_fresh(&t);

        t.sync(80); // width change rebuilds
        assert_fresh(&t);
        t.append_thinking(&Span::styled(" late", DIM_STYLE)); // splices above the answer
        t.sync(80);
        assert_fresh(&t);

        t.toggle_thinking(); // reveal
        t.sync(80);
        assert_fresh(&t);
        t.append_thinking(&Span::styled(" more", DIM_STYLE));
        t.sync(80);
        assert_fresh(&t);
        t.toggle_thinking(); // and hide again
        t.sync(80);
        assert_fresh(&t);

        t.begin_response(); // retirement drains the run
        t.sync(80);
        assert_fresh(&t);

        t.clear();
        t.sync(80);
        assert_fresh(&t);
    }

    #[test]
    fn toggle_hides_then_shows_the_latest_thinking() {
        let mut t = Transcript::default();
        t.push(Line::from("❯ echo"));
        t.begin_response();
        // Long enough to wrap: hiding must shrink the row count.
        let reasoning = "secret reasoning ".repeat(6);
        t.append_thinking(&Span::styled(reasoning, DIM_STYLE));
        t.append(&Span::raw("the answer"));

        t.toggle_thinking(); // reveal
        t.sync(40);
        let shown = texts(&t.rows);
        assert!(shown.iter().any(|row| row.contains("secret reasoning")));
        assert!(shown.iter().any(|row| row.contains("the answer")));

        t.toggle_thinking(); // hide: a placeholder replaces the reasoning
        t.sync(40);
        let hidden = texts(&t.rows);
        assert!(hidden.len() < shown.len());
        assert!(hidden.iter().any(|row| row.contains("Thinking")));
        assert!(hidden.iter().all(|row| !row.contains("secret reasoning")));
        assert!(hidden.iter().any(|row| row.contains("the answer")));

        t.toggle_thinking(); // and back: the rewrap is byte-identical
        t.sync(40);
        assert_eq!(texts(&t.rows), shown);
    }

    #[test]
    fn retirement_drains_old_thinking_only() {
        let mut t = Transcript::default();
        t.push(Line::from("❯ one"));
        t.begin_response();
        t.append_thinking(&Span::styled("old reasoning", DIM_STYLE));
        t.append(&Span::raw("old answer"));
        t.push(Line::from("❯ two"));
        t.begin_response();
        let messages = message_texts(&t);
        assert!(messages.iter().all(|m| !m.contains("old reasoning")));
        assert!(messages.iter().any(|m| m.contains("old answer")));
        assert!(messages.iter().any(|m| m.contains("❯ two")));

        t.append_thinking(&Span::styled("new reasoning", DIM_STYLE));
        t.append(&Span::raw("new answer"));
        t.push(Line::from("❯ three"));
        t.begin_response();
        t.sync(40);
        let rows = texts(&t.rows);
        assert!(rows.iter().all(|r| !r.contains("reasoning")));
        assert!(!rows.iter().any(|r| r.contains("Thinking")), "empty run");
        assert!(rows.iter().any(|r| r.contains("old answer")));
        assert!(rows.iter().any(|r| r.contains("new answer")));
    }

    /// The dim error line the `Failed` path appends must never end up inside the drain
    #[test]
    fn error_line_survives_retirement() {
        let mut t = Transcript::default();
        t.begin_response();
        t.append_thinking(&Span::styled("doomed reasoning", DIM_STYLE));
        // Same dim style as the thinking: without the append boundary it
        // would glue into the run and drain away with it.
        t.append(&Span::styled("boom: network down", DIM_STYLE));
        assert_eq!(message_texts(&t), ["doomed reasoning", "boom: network down"]);
        t.begin_response();
        let messages = message_texts(&t);
        assert!(messages.iter().any(|m| m.contains("boom")));
        assert!(messages.iter().all(|m| !m.contains("doomed")));
        t.sync(40);
        assert!(texts(&t.rows).iter().any(|row| row.contains("boom")));
    }

    #[test]
    fn late_thinking_stays_contiguous() {
        let mut t = Transcript::default();
        t.begin_response();
        t.append_thinking(&Span::styled("t1", DIM_STYLE));
        t.append(&Span::raw("a1"));
        t.append_thinking(&Span::styled("t2", DIM_STYLE));
        assert_eq!(message_texts(&t), ["t1", "t2", "a1"]);
        let run = t.run.expect("run");
        assert_eq!((run.start, run.end), (0, 2));

        t.sync(40); // hidden (default): both thinking messages collapse
        let hidden = texts(&t.rows);
        assert_eq!(hidden.len(), 2); // placeholder + answer
        assert!(hidden.iter().any(|r| r.contains("Thinking")));
        assert!(hidden.iter().all(|r| !r.contains("t1") && !r.contains("t2")));

        t.toggle_thinking();
        t.sync(40);
        assert_eq!(texts(&t.rows), texts(&wrap_lines(&visible(&t), 40)));

        t.begin_response();
        assert_eq!(message_texts(&t), ["a1"]);
    }

    /// Gluing a hidden thinking fragment must not truncate the cached rows of
    /// the visible messages around it.
    #[test]
    fn hidden_glue_does_not_eat_visible_rows() {
        let mut t = Transcript::default();
        t.push(Line::from("prompt"));
        t.begin_response();
        t.append_thinking(&Span::styled("abc", DIM_STYLE));
        t.sync(20);
        t.append_thinking(&Span::styled("def", DIM_STYLE)); // glue, cache primed
        t.sync(20);
        t.append(&Span::raw("answer"));
        t.sync(20);
        assert_eq!(
            texts(&t.rows),
            texts(&wrap_lines(
                &[Line::from("prompt"), thinking_placeholder(), Line::from("answer")],
                20
            ))
        );
    }

    /// Make sure late fragments (reasoning AFTER text) rewrap only the spliced data.
    #[test]
    fn late_fragments_rewrap_from_the_splice_point() {
        let mut t = Transcript::default();
        t.push(Line::from("❯ echo"));
        t.begin_response();
        t.append_thinking(&Span::styled("t1", DIM_STYLE));
        t.append(&Span::raw("a1"));
        t.sync(40);
        t.append_thinking(&Span::styled("t2", DIM_STYLE)); // late, tail folded
        assert_eq!(t.cache, (40, 2), "the pre-run messages stay cached");
        t.sync(40);
        assert_eq!(texts(&t.rows), texts(&wrap_lines(&visible(&t), 40)));

        // A second late fragment with an unsynced glued answer in between.
        t.append(&Span::raw(" a2"));
        t.append_thinking(&Span::styled("t3", DIM_STYLE));
        t.sync(40);
        assert_eq!(texts(&t.rows), texts(&wrap_lines(&visible(&t), 40)));
        assert_eq!(message_texts(&t), ["❯ echo", "t1", "t2", "t3", "a1 a2"]);
    }

    /// A late fragment without text shouldn't break the cache or current run.
    #[test]
    fn empty_late_fragments_change_nothing() {
        let mut t = Transcript::default();
        t.push(Line::from("❯ echo"));
        t.begin_response();
        t.append_thinking(&Span::styled("t1", DIM_STYLE));
        t.append(&Span::raw("a1"));
        t.sync(40);
        let (rows, cache) = (texts(&t.rows), t.cache);
        t.append_thinking(&Span::styled("\n\n", DIM_STYLE));
        assert_eq!(t.cache, cache);
        assert_eq!(texts(&t.rows), rows);
        assert_eq!(message_texts(&t), ["❯ echo", "t1", "a1"]);
        // The answer is still open: later text joins its message.
        t.append(&Span::raw(" more"));
        t.sync(40);
        assert_eq!(message_texts(&t), ["❯ echo", "t1", "a1 more"]);
    }

    /// Tool boxes render, update in place, collapse, and color by outcome.
    #[test]
    fn tool_calls_render_update_and_collapse() {
        let mut t = Transcript::default();
        t.push(Line::from("❯ run it"));
        t.begin_response();
        start_bash(&mut t, "call_0");
        t.sync(40);
        assert!(texts(&t.rows).iter().any(|row| row.contains("● Bash(echo hi) …")));

        t.finish_tool("call_0", "hi\n".to_string(), Some(0));
        t.sync(40);
        let rows = texts(&t.rows);
        assert!(rows.iter().any(|row| row.contains("● Bash(echo hi)")));
        assert!(!rows.iter().any(|row| row.contains('…')));
        assert!(rows.iter().any(|row| row.contains("⎿ hi")));

        // A long output collapses to head + count + tail; Ctrl+O expands it.
        start_bash(&mut t, "call_1");
        let mut long = String::new();
        for i in 0..20 {
            long.push_str("line ");
            long.push_str(&i.to_string());
            long.push('\n');
        }
        t.finish_tool("call_1", long, Some(0));
        t.sync(40);
        let collapsed = texts(&t.rows);
        assert!(
            collapsed
                .iter()
                .any(|row| row.contains("… +15 lines (ctrl+o to expand)"))
        );
        assert!(!collapsed.iter().any(|row| row.contains("line 10")));

        t.toggle_expand();
        t.sync(40);
        assert!(texts(&t.rows).iter().any(|row| row.contains("line 10")));
        t.toggle_expand();
        t.sync(40);
        assert!(!texts(&t.rows).iter().any(|row| row.contains("line 10")));

        // A failure shows its code, an empty output says so, and an unknown id
        // finishes nothing.
        start_bash(&mut t, "call_2");
        t.finish_tool("call_2", String::new(), Some(1));
        t.finish_tool("call_404", "nope".to_string(), Some(0));
        t.sync(40);
        let rows = texts(&t.rows);
        assert!(rows.iter().any(|row| row.contains("exit 1")));
        assert!(rows.iter().any(|row| row.contains(TOOL_NO_OUTPUT)));
        assert!(!rows.iter().any(|row| row.contains("nope")));
    }

    /// A new call folds every finished box before it down to its header line,
    /// Ctrl+O or not; the newest box keeps its body until it is folded too.
    #[test]
    fn new_calls_fold_finished_boxes_to_their_headers() {
        let mut t = Transcript::default();
        let assert_fresh = |t: &Transcript| {
            assert_eq!(
                texts(&t.rows),
                texts(&wrap_lines(
                    &entry_lines(&t.messages, t.show_tool_output),
                    t.cache.0
                ))
            );
        };
        t.push(Line::from("❯ run it"));
        t.begin_response();
        start_bash(&mut t, "call_0");
        t.finish_tool("call_0", "one\ntwo\nthree\n".to_string(), Some(0));
        t.sync(40);
        assert_fresh(&t);
        assert!(texts(&t.rows).iter().any(|row| row.contains("⎿ one")));

        // The second call folds the first: only the two headers render.
        t.start_tool("call_1".to_string(), "Bash", "ls -la".to_string());
        t.sync(40);
        assert_fresh(&t);
        let rows = texts(&t.rows);
        assert!(rows.iter().any(|row| row.contains("● Bash(echo hi)")));
        assert!(rows.iter().any(|row| row.contains("● Bash(ls -la) …")));
        assert!(!rows.iter().any(|row| row.contains('⎿')));

        // The newest box keeps its collapsed body; the folded one stays hidden.
        t.finish_tool(
            "call_1",
            "out aaaa\nbbbb\ncccc\ndddd\neeee\nffff\n".to_string(),
            Some(0),
        );
        t.sync(40);
        assert_fresh(&t);
        let rows = texts(&t.rows);
        assert!(rows.iter().any(|row| row.contains("⎿ out aaaa")));
        assert!(!rows.iter().any(|row| row.contains("⎿ dddd")));
        assert!(!rows.iter().any(|row| row.contains("⎿ one")));

        // Ctrl+O expands the standing box only; the folded one stays a header.
        t.toggle_expand();
        t.sync(40);
        assert_fresh(&t);
        let rows = texts(&t.rows);
        assert!(rows.iter().any(|row| row.contains("⎿ dddd")));
        assert!(!rows.iter().any(|row| row.contains("⎿ one")));
        t.toggle_expand();
        t.sync(40);
        assert_fresh(&t);
        assert!(!texts(&t.rows).iter().any(|row| row.contains("⎿ dddd")));
    }

    /// A failed generation resolves its still-running boxes instead of
    /// leaving them pending forever.
    #[test]
    fn failed_generations_resolve_pending_tools() {
        let mut t = Transcript::default();
        start_bash(&mut t, "call_0");
        t.sync(40);
        assert!(texts(&t.rows).iter().any(|row| row.contains('…')));

        t.fail_pending("generation panicked");
        t.sync(40);
        let rows = texts(&t.rows);
        assert!(rows.iter().any(|row| row.contains("generation panicked")));
        assert!(!rows.iter().any(|row| row.contains('…')));
    }

    /// Text streamed after a tool box never glues onto it.
    #[test]
    fn appends_after_a_tool_start_a_new_line() {
        let mut t = Transcript::default();
        t.append(&Span::raw("answer"));
        start_bash(&mut t, "call_0");
        t.append(&Span::raw("more"));
        assert_eq!(message_texts(&t), ["answer", "more"]);
    }

    /// A tool start rotates the thinking run below the new box, and later
    /// fragments extend it there; the cache stays honest through toggles and
    /// width changes.
    #[test]
    fn thinking_rides_below_tool_boxes() {
        let mut t = Transcript::default();
        let assert_fresh = |t: &Transcript| {
            assert_eq!(texts(&t.rows), texts(&wrap_lines(&visible(t), t.cache.0)));
        };
        t.push(Line::from("❯ go"));
        t.begin_response();
        t.append_thinking(&Span::styled("t1", DIM_STYLE));
        t.append(&Span::raw("a1"));
        start_bash(&mut t, "call_0");
        t.finish_tool("call_0", "out\n".to_string(), Some(0));
        t.sync(40);
        assert_fresh(&t);
        // t1 rotated below the box when the call started, placeholder and all.
        assert_eq!(message_texts(&t), ["❯ go", "a1", "t1"]);
        assert!(texts(&t.rows).iter().any(|row| row.contains("Thinking")));

        t.append_thinking(&Span::styled("t2", DIM_STYLE)); // extends the run below the box
        start_bash(&mut t, "call_1"); // a second box stacks above the run
        t.sync(40);
        assert_fresh(&t);
        assert_eq!(message_texts(&t), ["❯ go", "a1", "t1", "t2"]);
        // Every header must precede the placeholder in the rendered rows.
        let rows = texts(&t.rows);
        let last_box = rows
            .iter()
            .rposition(|row| row.contains("Bash"))
            .expect("box headers");
        let think = rows
            .iter()
            .position(|row| row.contains("Thinking"))
            .expect("placeholder");
        assert!(last_box < think, "{rows:?}");

        t.toggle_thinking();
        t.sync(40);
        assert_fresh(&t);
        t.toggle_expand();
        t.sync(40);
        assert_fresh(&t);
        t.sync(80);
        assert_fresh(&t);
    }

    #[test]
    fn ctrl_o_expands_tool_output_in_live_mode_only() {
        let mut pane = Pane::new();
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
        let mut pane = Pane::new();
        pane.push(Line::from("❯ hi"));
        pane.begin_response();
        pane.append_thinking(&Span::styled("visible reasoning", DIM_STYLE));
        pane.append(&Span::raw("answer"));

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
    fn empty_runs_and_clear_reset_state() {
        let mut t = Transcript::default();
        t.push(Line::from("prompt"));
        t.begin_response();
        t.sync(20);
        // An empty run renders no placeholder; toggling it changes nothing.
        let rows = texts(&t.rows);
        assert_eq!(rows, ["prompt"]);
        t.toggle_thinking();
        t.sync(20);
        assert_eq!(texts(&t.rows), rows);

        t.append_thinking(&Span::styled("reasoning", DIM_STYLE));
        t.clear();
        assert!(t.messages.is_empty() && t.rows.is_empty());
        assert!(t.run.is_none());
        // Toggled on above: `clear` keeps the preference rather than the default.
        assert!(t.show_thinking, "clear keeps the sticky preference");

        // Thinking after a clear lazily re-opens a drainable run.
        t.append_thinking(&Span::styled("again", DIM_STYLE));
        assert!(t.run.is_some_and(|run| run.start == 0 && run.end == 1));
        t.begin_response();
        assert!(t.messages.is_empty(), "the re-opened run drains");
    }
}
