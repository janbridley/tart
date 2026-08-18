//! `Pane` object stores data and rendering logic for the terminal interface.

use ratatui::buffer::Buffer;
use ratatui::crossterm::event::{KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::{Frame, symbols};
use unicode_segmentation::UnicodeSegmentation;

use crate::file_mentions::{self, FilePopup};

pub const PROMPT: &str = "❯ ";
/// Cells before the editor starts (the prompt symbol's width).
const GUTTER: u16 = 2;
/// Spaces a tab renders as.
const TAB_WIDTH: usize = 4;

pub const DIM_STYLE: Style = Style::new().fg(Color::DarkGray);
const PROMPT_STYLE: Style = Style::new().fg(Color::Cyan).add_modifier(Modifier::BOLD);
/// The copy cursor and the editor caret are the cell under them, inverted.
const CURSOR_STYLE: Style = Style::new().add_modifier(Modifier::REVERSED);

/// Key results the pane cannot handle itself.
#[derive(Debug, PartialEq)]
pub enum PaneEvent {
    Submit(String),
    Quit,
}

/// The TUI interface.
pub struct Pane {
    /// Input interface for the `Pane`.
    prompt: Editor,
    /// Conversation history.
    transcript: Transcript,
    /// A `CopyCursor` in copymode, or `None` otherwise.
    copy: Option<CopyCursor>,
    /// The `@file` typeahead, open while an `@` word is being typed.
    popup: Option<FilePopup>,
}

impl Pane {
    pub fn new() -> Self {
        Self {
            prompt: Editor::default(),
            transcript: Transcript::default(),
            copy: None,
            popup: None,
        }
    }

    pub fn on_key(&mut self, key: KeyEvent) -> Option<PaneEvent> {
        // Press and Repeat drive the app, Release is not important/useful.
        if !matches!(key.kind, KeyEventKind::Press | KeyEventKind::Repeat) {
            return None;
        }
        let event = self.route(key);
        // @file popup updates on keystroke
        file_mentions::update(&self.prompt, &mut self.popup, file_mentions::rearm(&key));
        event
    }

    /// Key routing without the popup bookkeeping.
    fn route(&mut self, key: KeyEvent) -> Option<PaneEvent> {
        // Copy mode takes every key first, so the draft is immutable while scrolling
        if let Some(cursor) = self.copy {
            match key.code {
                KeyCode::Char('q' | 'Q') => self.copy = None,
                _ => self.copy = Some(moved(self.transcript.rows(), cursor, key.code)),
            }
            return None;
        }
        if key.modifiers.contains(KeyModifiers::CONTROL) {
            match key.code {
                KeyCode::Char('c' | 'd') => return Some(PaneEvent::Quit),
                KeyCode::Char('u') => self.prompt.clear(),
                _ => {}
            }
            return None;
        }
        // The @file popup is the highest priority UI element in live mode: arrows move
        // the selection, and Tab/Enter insert.
        if let Some(popup) = self.popup.as_mut()
            && matches!(
                key.code,
                KeyCode::Up | KeyCode::Down | KeyCode::Tab | KeyCode::Enter
            )
        {
            match key.code {
                KeyCode::Up => popup.select_prev(),
                KeyCode::Down => popup.select_next(),
                _ => {
                    popup.accept(&mut self.prompt);
                    self.popup = None;
                }
            }
            return None;
        }
        match key.code {
            KeyCode::Enter if key.modifiers.contains(KeyModifiers::ALT) => self.prompt.new_line(),
            KeyCode::Enter => return self.submit().map(PaneEvent::Submit),
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
            file_mentions::update(&self.prompt, &mut self.popup, false);
        }
    }

    /// Close the file popup, else leave copy mode. Returns whether anything was closed.
    pub fn escape(&mut self) -> bool {
        self.popup.take().is_some() || self.copy.take().is_some()
    }

    pub fn push<L: Into<Line<'static>>>(&mut self, line: L) {
        self.transcript.push(line);
    }

    /// Append a streaming fragment; see [`Transcript::append`].
    pub fn append(&mut self, span: &Span<'static>) {
        self.transcript.append(span);
    }

    pub fn clear(&mut self) {
        self.transcript.clear();
    }

    /// Echo the draft into the transcript and clear it.
    fn submit(&mut self) -> Option<String> {
        if self.prompt.text().trim().is_empty() {
            return None;
        }
        let text = self.prompt.text();
        let mut rows = text.split('\n');
        self.transcript.push(Line::from(vec![
            Span::styled(PROMPT, PROMPT_STYLE),
            Span::raw(rows.next().unwrap_or_default().to_string()),
        ]));
        for continuation in rows {
            self.transcript
                .push(Line::from(format!("  {continuation}")));
        }
        self.prompt.clear();
        Some(text)
    }

    /// Render into the granted area: transcript, rule, prompt, rule.
    pub fn render(&mut self, frame: &mut Frame, area: Rect) {
        if area.height == 0 || area.width == 0 {
            return;
        }
        // The prompt grows with its wrapped content, always leaving space for the
        // transcript and the two rules.
        let layout = wrap_draft(
            &self.prompt.lines,
            (self.prompt.line, self.prompt.g),
            area.width.saturating_sub(GUTTER) as usize,
        );
        let cap = area.height.saturating_sub(4).max(1) as usize;
        let prompt_height = if self.copy.is_some() {
            1
        } else {
            layout.rows.len().min(cap).max(1) as u16
        };
        let [transcript, bar_top, prompt_area, bar_bottom] = Layout::vertical([
            Constraint::Min(1),
            Constraint::Length(1),
            Constraint::Length(prompt_height),
            Constraint::Length(1),
        ])
        .areas(area);

        self.render_transcript(transcript, bar_top, bar_bottom, frame);
        if self.copy.is_some() {
            Self::render_scrollback_hint(frame, prompt_area);
            return;
        }

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
        // The @file popup overlays the transcript, anchored above the top rule.
        if let Some(popup) = self.popup.as_mut() {
            file_mentions::render(frame, popup, bar_top);
        }
    }
    /// Sync, window, and paint the transcript, then draw its two rules.
    fn render_transcript(&mut self, area: Rect, bar_top: Rect, bar_bottom: Rect, frame: &mut Frame) {
        // Wrap only what is new at an unchanged width, or rewrap if width changed.
        let rows = self.transcript.sync(area.width as usize);
        if let Some(cursor) = &mut self.copy
            && !rows.is_empty()
        {
            cursor.row = cursor.row.min(rows.len() - 1);
            cursor.col = cursor.col.min(rows[cursor.row].width().saturating_sub(1));
        }
        let visible = area.height as usize;
        let top = window_top(rows.len(), visible, self.copy.map(|c| (c.row, c.top)));
        if let Some(cursor) = &mut self.copy {
            cursor.top = top;
            cursor.visible = visible;
        }
        let buf = frame.buffer_mut();
        let shown = visible.min(rows.len().saturating_sub(top));
        rows[top..top + shown]
            .iter()
            .zip(area.y..)
            .for_each(|(row, y)| {
                buf.set_line(area.x, y, row, area.width);
            });
        if let Some(cursor) = self.copy {
            let pos = (
                area.x + cursor.col as u16,
                area.y + (cursor.row - top) as u16,
            );
            if let Some(cell) = buf.cell_mut(pos) {
                cell.set_style(CURSOR_STYLE);
            }
        }
        rule(buf, bar_top);
        rule(buf, bar_bottom);
    }

    /// In copy mode the prompt area shows the scrollback keybindings.
    fn render_scrollback_hint(frame: &mut Frame, prompt_area: Rect) {
        if prompt_area.height > 0 {
            frame.buffer_mut().set_line(
                prompt_area.x,
                prompt_area.y,
                &Line::from(Span::styled(
                    "▲ scrollback · ←↑↓→ move · PgUp/PgDn/Home/End · q to exit",
                    Style::new().fg(Color::Yellow),
                )),
                prompt_area.width,
            );
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
            cell.set_symbol(symbols::line::HORIZONTAL)
                .set_style(DIM_STYLE);
        }
    }
}

/// The prompt editor: a multi-line draft with a grapheme caret.
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
    fn text(&self) -> String {
        self.lines.join("\n")
    }

    pub(crate) fn clear(&mut self) {
        self.lines = vec![String::new()];
        self.line = 0;
        self.g = 0;
        self.top = 0;
    }

    /// Graphemes on the current line.
    fn line_len(&self) -> usize {
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
            let cleaned: String = part
                .chars()
                .filter(|c| !c.is_control() || *c == '\t')
                .collect();
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
    fn home(&mut self) {
        self.g = 0;
    }

    /// To the line end.
    fn end(&mut self) {
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

/// One view's message log, kept pre-wrapped to the display width:
///
/// Call `sync` to re-wrap the new content.
#[derive(Default)]
struct Transcript {
    messages: Vec<Line<'static>>,
    /// Whether the last message is still open for `append` runs.
    open: bool,
    rows: Vec<Line<'static>>,
    /// (width, how many messages `rows` already contains).
    cache: (usize, usize),
}

impl Transcript {
    /// Append a committed line, ending any append-run.
    fn push(&mut self, line: impl Into<Line<'static>>) {
        self.messages.push(line.into());
        self.open = false;
    }

    /// Append a streaming fragment, gluing onto the previous fragment while style matches.
    ///
    /// Newlines in the text end the current line.
    fn append(&mut self, span: &Span<'static>) {
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
            && self.messages.last().is_some_and(|line| {
                line.spans
                    .last()
                    .is_some_and(|last| last.style == span.style)
            });
        if glue {
            // The cache already counted the line being extended: drop its
            // stale rows and hand the message back for the next sync.
            if self.cache.1 == self.messages.len() && self.cache.0 > 0 {
                let stale =
                    wrap_lines(&self.messages[self.messages.len() - 1..], self.cache.0).len();
                self.rows.truncate(self.rows.len() - stale);
                self.cache.1 -= 1;
            }
            if let Some(last) = self.messages.last_mut().and_then(|l| l.spans.last_mut()) {
                // Extend the last matching span if available to save memory.
                last.content.to_mut().push_str(&span.content);
            }
        } else {
            self.messages.push(Line::from(span));
            self.open = true;
        }
    }

    /// End the current append-run; later appends start a fresh line.
    fn break_line(&mut self) {
        self.open = false;
    }

    /// Drop every message and reset our caches.
    fn clear(&mut self) {
        self.messages.clear();
        self.rows.clear();
        self.cache.1 = 0;
        self.open = false;
    }

    /// Wrap the messages not yet folded into `rows`; a width change rewraps everything.
    fn sync(&mut self, width: usize) -> &[Line<'static>] {
        if self.cache.0 != width {
            self.rows.clear();
            self.cache = (width, 0);
        }
        let done = self.cache.1;
        self.rows.extend(wrap_lines(&self.messages[done..], width));
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
            top: usize::MAX,
            visible: 0,
        }
    }
}

/// The cursor after one key step, clamped to the transcript edges.
fn moved(rows: &[Line<'static>], cursor: CopyCursor, key: KeyCode) -> CopyCursor {
    if rows.is_empty() {
        return CopyCursor {
            row: 0,
            col: 0,
            ..cursor
        };
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
        let gw = Span::raw(sym).width();
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
        let count = line.graphemes(true).count();
        if li == cl {
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
    use super::*;
    use crate::testutil::draw;

    fn key(code: KeyCode, modifiers: KeyModifiers) -> KeyEvent {
        KeyEvent::new(code, modifiers)
    }

    fn render(pane: &mut Pane, size: (u16, u16)) -> String {
        draw(|frame, area| pane.render(frame, area), size)
    }

    fn texts(lines: &[Line<'static>]) -> Vec<String> {
        lines
            .iter()
            .map(|line| {
                line.spans
                    .iter()
                    .map(|span| span.content.as_ref())
                    .collect()
            })
            .collect()
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
            texts(&transcript.messages),
            ["prompt", "Hello world", " (thinking) more"]
        );

        // `push` and `break_line` both end the run.
        transcript.push(Line::from("committed"));
        transcript.append(&Span::raw("after"));
        transcript.break_line();
        transcript.append(&Span::raw("again"));
        assert_eq!(
            texts(&transcript.messages),
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
                texts(&wrap_lines(&transcript.messages, transcript.cache.0))
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

        transcript.clear(); // hidden pane: re-push to the same count
        for i in 0..6 {
            transcript.push(Line::from(format!("fresh {i}")));
        }
        transcript.sync(80);
        assert_fresh(&transcript);
        assert!(
            !texts(&transcript.rows)
                .iter()
                .any(|row| row.contains("aaaa"))
        );
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
            top: 0,
            visible: 2,
        };
        assert_eq!(moved(&rows, cur(0, 0), KeyCode::Right), cur(0, 1));
        assert_eq!(moved(&rows, cur(1, 0), KeyCode::Left), cur(0, 2)); // wraps
        assert_eq!(moved(&rows, cur(1, 0), KeyCode::PageUp), cur(0, 0));
        assert_eq!(moved(&rows, cur(0, 0), KeyCode::PageDown), cur(1, 0));
        assert_eq!(moved(&rows, cur(1, 0), KeyCode::Char('x')), cur(1, 0));
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
        assert!(pane.copy.unwrap().row > 10);
        pane.on_key(key(KeyCode::Char('u'), KeyModifiers::CONTROL)); // eaten
        pane.on_key(key(KeyCode::PageUp, KeyModifiers::NONE));

        render(&mut pane, (80, 24)); // grow
        let cursor = pane.copy.expect("resize left copy mode");
        assert_eq!(cursor.row, pane.transcript.rows().len() - 1, "not clamped");

        assert!(pane.escape());
        assert!(render(&mut pane, (40, 10)).contains("❯ ")); // live again
    }

    #[test]
    fn tiny_terminal_renders_without_panic() {
        let mut pane = Pane::new();
        pane.push(Line::from("text"));
        render(&mut pane, (6, 3));
        render(&mut pane, (2, 1));
        render(&mut pane, (1, 0));
    }
}
