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
use tart_agents::Progress;

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
    /// The state before the submitted turn (in case we have to cancel a message).
    turn: Option<TurnSnapshot>,
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
        self.transcript.restore_to(turn.entries);
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
            entries: self.transcript.message_count(),
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
        assert!(pane.transcript.message_texts().is_empty());
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
        let mut pane = Pane::default();
        pane.push(Line::from("earlier"));
        pane.begin_response();
        pane.append_answer("earlier answer");

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
        pane.append_answer("Once upon");
        pane.start_tool("call_0".to_string(), "Bash", "sleep 5".to_string());

        pane.cancel_turn();

        let screen = render(&mut pane, (60, 20));
        assert!(screen.contains("earlier answer"), "{screen}");
        assert!(screen.contains("⎋ cancelled"), "{screen}");
        assert!(!screen.contains("Once upon"), "{screen}");
        assert!(!screen.contains("sleep 5"), "{screen}");
        assert_eq!(pane.prompt.text(), "write a story");
        // The restored log folds cleanly.
        pane.transcript.assert_rows_match_full_rewrap();
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
            !pane
                .transcript
                .message_texts()
                .iter()
                .any(|text| text.contains("cancel me"))
        );
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
        pane.append_thinking(&Span::styled("visible reasoning", DIM_STYLE));
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
