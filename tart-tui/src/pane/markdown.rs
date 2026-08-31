//! Render the model's answer text as styled markdown.
//!
//! Only [`Progress::Answer`](tart_agents::Progress::Answer) fragments are pretty-
//! rendered, all others stay base text. Copy mode reads the rendered rows so that a
//! copied answer carries the styled form, markers stripped. Lines wrap to the
//! display width with their block's hanging indent, so a wrapped list item or
//! quotation keeps its marker column on every row.

use itertools::Itertools;
use pulldown_cmark::{Alignment, Event, Options, Parser, Tag, TagEnd};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};

use super::wrap::wrap_lines;
use super::{DIM_STYLE, SpansExt};

/// H1: blue and bold.
const HEADING1: Style = Style::new().fg(Color::Blue).add_modifier(Modifier::BOLD);
/// H2: cyan and bold.
const HEADING2: Style = Style::new().fg(Color::Cyan).add_modifier(Modifier::BOLD);
/// H3: Same color as H2, but italic as well.
const HEADING3: Style = Style::new()
    .fg(Color::Cyan)
    .add_modifier(Modifier::BOLD)
    .add_modifier(Modifier::ITALIC);
/// H4–H6: bold with default color
const HEADING: Style = Style::new().add_modifier(Modifier::BOLD);
/// Heading styles by level, H1 first.
const HEADING_STYLES: [Style; 6] = [HEADING1, HEADING2, HEADING3, HEADING, HEADING, HEADING];
/// Inline `` `code` ``: set apart by color rather than markers.
const INLINE_CODE: Style = Style::new().fg(Color::LightYellow);
/// Blockquote rails, code-block content, and other quiet chrome.
const QUIET: Style = DIM_STYLE;
/// Blockquote text: quiet like code, but italic.
const QUOTE: Style = Style::new().fg(Color::DarkGray).add_modifier(Modifier::ITALIC);
/// Link text: light blue and underlined.
const LINK: Style = Style::new()
    .fg(Color::LightBlue)
    .add_modifier(Modifier::UNDERLINED);
/// The bullet an unordered list item renders instead of its marker.
const BULLET: &str = "• ";
/// The width of the rule row a horizontal rule renders as
///
/// Should be short, as [`render`] may wrap it.
const HR_WIDTH: usize = 24;

/// Render answer text to styled transcript lines, wrapping each line to
/// `width` cells with its block's hanging indent carried onto every wrapped
/// row; a `width` of 0 leaves each line on one row.
pub(crate) fn render(raw: &str, width: usize) -> Vec<Line<'static>> {
    let mut blocks = Blocks::new(width);
    for event in Parser::new_ext(raw, options()) {
        blocks.event(event);
    }
    blocks.finish()
}

/// Enable `CommonMark` core plus GitHub's tables and strikethroughs.
fn options() -> Options {
    Options::ENABLE_TABLES | Options::ENABLE_STRIKETHROUGH
}

/// The event walk's state: the rows out, the inline row and line context
/// under construction, and the buffered blocks.
#[derive(Default)]
struct Blocks {
    /// The cell width rows wrap at; 0 leaves lines unwrapped.
    width: usize,
    /// The rendered rows.
    out: Vec<Line<'static>>,
    /// The inline row under construction; breaks and block ends flush it.
    inline: Inline,
    /// The context around the current block's lines.
    prefix: Prefix,
    /// Whether a blank row separates the next rendered row from the block above
    pending_gap: bool,
    /// Whether the last rendered row was html and should be appended to.
    in_html: bool,
    /// The open lists, outermost first.
    lists: Vec<List>,
    /// Blockquote nesting depth.
    quote: usize,
    /// The code block under construction, rendered at its closing tag.
    code: Option<String>,
    /// The table under construction, rendered at its closing tag.
    table: Option<Table>,
}

/// One open inline scope (emphasis, strikethrough, a link, an image): its
/// style patch applies until the scope closes. Patches must stay add-only;
/// popping is restoring only because they never subtract.
struct Scope {
    /// The patch this scope applies over the scopes around it.
    patch: Style,
    /// A link's destination and the label text captured so far.
    link: Option<(String, String)>,
}

/// The inline row under construction: its spans, the scopes styling them,
/// and whether the next text trims its leading whitespace.
#[derive(Default)]
struct Inline {
    /// The row's spans so far.
    spans: Vec<Span<'static>>,
    /// The scopes opened and not yet closed, outermost first.
    scopes: Vec<Scope>,
    /// Whether the next text trims its leading whitespace (a continuation
    /// line's indent is insignificant).
    trim: bool,
}

impl Inline {
    /// The current style: the base patched by every open scope.
    fn style(&self) -> Style {
        self.scopes
            .iter()
            .fold(Style::new(), |style, scope| style.patch(scope.patch))
    }

    /// Open a scope whose patch applies until it closes.
    fn open(&mut self, patch: Style) {
        self.scopes.push(Scope { patch, link: None });
    }

    /// Append styled text, gluing onto the last span when the style matches.
    fn push(&mut self, text: &str, style: Style) {
        let text = if self.trim { text.trim_start() } else { text };
        if text.is_empty() {
            return;
        }
        self.trim = false;
        self.spans.push_merged(text, style);
        if let Some((_, label)) = self.scopes.iter_mut().rev().find_map(|scope| scope.link.as_mut())
        {
            label.push_str(text);
        }
    }

    /// Take the row's spans, dropping the source's invisible trailing whitespace.
    fn take(&mut self) -> Vec<Span<'static>> {
        if let Some(last) = self.spans.last_mut() {
            let owned = last.content.to_mut();
            let len = owned.trim_end().len();
            owned.truncate(len);
        }
        std::mem::take(&mut self.spans)
    }

    /// Whether any text is pending.
    fn is_empty(&self) -> bool {
        self.spans.is_empty()
    }
}

/// The context around one block's lines: the spans its first and later
/// lines carry before their text, and the style stamped on the line.
#[derive(Default)]
struct Prefix {
    /// Spans before the block's first line (a bullet, a rail).
    first: Vec<Span<'static>>,
    /// Spans before each later line (indents, a rail).
    rest: Vec<Span<'static>>,
    /// The style stamped on every line of the block.
    style: Style,
    /// Whether the next line is the block's first.
    fresh: bool,
}

impl Prefix {
    /// Build the block's next line: its first or rest prefix, then the
    /// spans, stamped with the block's style.
    fn line(&mut self, mut spans: Vec<Span<'static>>) -> Line<'static> {
        let mut lead = if self.fresh {
            std::mem::take(&mut self.first)
        } else {
            self.rest.clone()
        };
        self.fresh = false;
        lead.append(&mut spans);
        let mut line = lead.into_line();
        line.style = self.style;
        line
    }
}

/// One open list's numbering and indents.
struct List {
    /// The first number of an ordered list; `None` renders bullets.
    ordered: Option<u64>,
    /// The number the next item renders.
    next: u64,
    /// This level's own bullet width in cells; the open levels' `cont`s sum
    /// to the absolute indent.
    cont: String,
}

/// A table buffered until its column widths are known at the closing tag.
struct Table {
    /// The header row's cells, each a run of styled spans.
    head: Vec<Vec<Span<'static>>>,
    /// The body rows.
    rows: Vec<Vec<Vec<Span<'static>>>>,
    /// The pending row's cells.
    row: Vec<Vec<Span<'static>>>,
    /// The per-column alignments, from the delimiter row.
    aligns: Vec<Alignment>,
}

impl Blocks {
    /// Start a walk whose rows wrap to `width` cells, or to none when 0.
    fn new(width: usize) -> Self {
        Self { width, ..Self::default() }
    }

    /// Fold one parser event into the state.
    fn event(&mut self, event: Event<'_>) {
        match event {
            Event::Start(tag) => self.start(tag),
            Event::End(tag) => self.end(tag),
            Event::Text(text) => {
                if let Some(code) = &mut self.code {
                    code.push_str(&text);
                } else {
                    let style = self.inline.style();
                    self.inline.push(&text, style);
                }
            }
            Event::Code(text) => {
                // Inline code renders verbatim: a break's trim never eats
                // its leading space.
                self.inline.trim = false;
                let style = self.inline.style().patch(INLINE_CODE);
                self.inline.push(&text, style);
            }
            Event::SoftBreak | Event::HardBreak => self.line_break(),
            Event::Rule => {
                self.gap();
                let mut spans = self.indents();
                spans.push(Span::styled("─".repeat(HR_WIDTH), QUIET));
                self.push_row(Line::from(spans));
            }
            Event::Html(html) => {
                // The parser splits one block into per-line chunks with trailing \n
                if html.trim().is_empty() {
                    return;
                }
                if !self.in_html {
                    self.gap();
                }
                self.quiet_rows(&html);
                self.in_html = true;
            }
            Event::InlineHtml(html) => {
                let style = self.inline.style();
                self.inline.push(&html, style);
            }
            // Task lists, footnotes, and math don't render specially.
            _ => {}
        }
    }

    /// Open a block or inline scope.
    fn start(&mut self, tag: Tag<'_>) {
        match tag {
            Tag::Paragraph => self.open_block(None),
            Tag::Heading { level, .. } => {
                self.open_block(Some(HEADING_STYLES[level as usize - 1]));
            }
            Tag::BlockQuote(_) => self.quote += 1,
            // The fenced language tag stays unused for now. TODO: support syntax!
            Tag::CodeBlock(_) => {
                self.gap();
                self.code = Some(String::new());
            }
            Tag::List(start) => {
                self.gap();
                let ordered = start.filter(|&n| n > 0);
                self.lists.push(List {
                    next: ordered.unwrap_or(0),
                    ordered,
                    cont: String::new(),
                });
            }
            Tag::Item => {
                // A nested list opening inside this item flushes the item's own text
                if !self.inline.is_empty() {
                    self.flush_line();
                }
                let bullet = match self.lists.last() {
                    Some(list) if list.ordered.is_some() => format!("{}. ", list.next),
                    _ => BULLET.to_owned(),
                };
                // Each level stores only its own bullet's width in cells, so
                // the levels sum to the true indent.
                let pad = " ".repeat(bullet.chars().count());
                let column = self.continuations(self.lists.len() - 1);
                if let Some(list) = self.lists.last_mut() {
                    list.next = list.next.wrapping_add(1);
                    list.cont.clone_from(&pad);
                }
                let style = if self.quote > 0 { QUOTE } else { Style::new() };
                let mut first = self.rail();
                first.push(Span::raw(format!("{column}{bullet}")));
                let mut rest = self.rail();
                let indent = format!("{column}{pad}");
                if !indent.is_empty() {
                    rest.push(Span::raw(indent));
                }
                self.prefix = Prefix { first, rest, style, fresh: true };
            }
            Tag::Table(aligns) => {
                self.gap();
                self.table = Some(Table {
                    head: Vec::new(),
                    rows: Vec::new(),
                    row: Vec::new(),
                    aligns,
                });
            }
            Tag::Strong => self.inline.open(Style::new().add_modifier(Modifier::BOLD)),
            Tag::Emphasis => self.inline.open(Style::new().add_modifier(Modifier::ITALIC)),
            Tag::Strikethrough => {
                self.inline.open(Style::new().add_modifier(Modifier::CROSSED_OUT));
            }
            Tag::Link { dest_url, .. } => self.open_link(&dest_url),
            // An image renders its alt text as the label in brackets.
            Tag::Image { dest_url, .. } => {
                self.open_link(&dest_url);
                let style = self.inline.style();
                self.inline.push("[", style);
            }
            // The table machinery drives on the cell ends; rows buffer here.
            // An unknown tag changing nothing stays safest.
            _ => {}
        }
    }

    /// Open a paragraph or heading: separate it from the block above, take
    /// the line context, and stamp the block's style.
    fn open_block(&mut self, style: Option<Style>) {
        self.gap();
        let style = match style {
            Some(style) => style,
            None if self.quote > 0 => QUOTE,
            None => Style::new(),
        };
        if self.lists.is_empty() {
            let rail = self.rail();
            self.prefix = Prefix {
                first: rail.clone(),
                rest: rail,
                style,
                fresh: true,
            };
        } else if self.prefix.fresh {
            // The item's opening block keeps its bullet context.
            self.prefix.style = style;
        } else {
            let cont = self.indents();
            self.prefix = Prefix {
                first: cont.clone(),
                rest: cont,
                style,
                fresh: true,
            };
        }
    }

    /// Close a block or inline scope.
    fn end(&mut self, tag: TagEnd) {
        match tag {
            TagEnd::Paragraph | TagEnd::Item | TagEnd::Heading(_) => self.end_block(),
            TagEnd::BlockQuote(_) => self.quote = self.quote.saturating_sub(1),
            TagEnd::CodeBlock => {
                let Some(code) = self.code.take() else {
                    return;
                };
                let body = code.strip_suffix('\n').unwrap_or(&code);
                if !body.is_empty() {
                    self.quiet_rows(body);
                }
            }
            TagEnd::List(_) => {
                self.lists.pop();
            }
            TagEnd::Table => self.flush_table(),
            TagEnd::TableHead => {
                if let Some(table) = &mut self.table {
                    table.head = std::mem::take(&mut table.row);
                }
            }
            TagEnd::TableRow => {
                if let Some(table) = &mut self.table {
                    let row = std::mem::take(&mut table.row);
                    table.rows.push(row);
                }
            }
            TagEnd::TableCell => {
                let cell = std::mem::take(&mut self.inline.spans);
                if let Some(table) = &mut self.table {
                    table.row.push(cell);
                }
            }
            TagEnd::Strong | TagEnd::Emphasis | TagEnd::Strikethrough => {
                self.inline.scopes.pop();
            }
            TagEnd::Link => {
                if let Some(scope) = self.inline.scopes.pop()
                    // The destination rides along dimly unless the label finishes it.
                    && let Some((dest, label)) = scope.link
                    && !dest.is_empty()
                    && dest != label
                {
                    self.show_dest(&dest);
                }
            }
            TagEnd::Image => {
                // Close the bracket while the label style still applies.
                let style = self.inline.style();
                self.inline.push("]", style);
                if let Some(scope) = self.inline.scopes.pop()
                    && let Some((dest, _)) = scope.link
                    && !dest.is_empty()
                {
                    self.show_dest(&dest);
                }
            }
            // An unknown end changing nothing stays safest.
            _ => {}
        }
    }

    /// Open a link-like scope capturing its destination and streamed label.
    fn open_link(&mut self, dest: &str) {
        self.inline.open(LINK);
        if let Some(scope) = self.inline.scopes.last_mut() {
            scope.link = Some((dest.to_string(), String::new()));
        }
    }

    /// Echo the destination dimly after the label.
    fn show_dest(&mut self, dest: &str) {
        self.inline.trim = false;
        self.inline.push(&format!(" ({dest})"), QUIET);
    }

    /// The blockquote rail for the current depth, on every line of a quote.
    fn rail(&self) -> Vec<Span<'static>> {
        if self.quote > 0 {
            vec![Span::styled("│ ".repeat(self.quote), QUIET)]
        } else {
            Vec::new()
        }
    }

    /// The rail plus every open list level's indent.
    ///
    /// Buffered rows (code, tables, rules, html) take this because a container's
    /// first block opens no paragraph, so `Prefix` may still be default.
    /// Block text instead uses the `Prefix` frozen at its block's open.
    fn indents(&self) -> Vec<Span<'static>> {
        let mut spans = self.rail();
        let indent = self.continuations(self.lists.len());
        if !indent.is_empty() {
            spans.push(Span::raw(indent));
        }
        spans
    }

    /// The joined continuation indents of the `n` outermost open lists.
    fn continuations(&self, n: usize) -> String {
        self.lists[..n].iter().map(|list| list.cont.as_str()).collect()
    }

    /// Push each line of `text` as a quiet row, carrying the context prefix.
    fn quiet_rows(&mut self, text: &str) {
        let lead = self.indents();
        for line in text.lines() {
            for row in self.wrap(&lead, vec![Span::styled(line.to_owned(), QUIET)]) {
                self.push_row(Line::from([lead.as_slice(), &row].concat()));
            }
        }
    }

    /// A block continues after a break, so indent doesn't matter
    fn line_break(&mut self) {
        if self.table.is_some() {
            // Cell text wraps at the cell, not the row; join with a space.
            let style = self.inline.style();
            self.inline.push(" ", style);
        } else {
            self.inline.trim = true;
            self.flush_line();
        }
    }

    /// Flush the inline row as a line, or as several wrapped to the width.
    fn flush_line(&mut self) {
        let spans = self.inline.take();
        let rows = self.wrap(&self.prefix.rest, spans);
        for spans in rows {
            let line = self.prefix.line(spans);
            self.push_row(line);
        }
    }

    /// Wrap one block line's spans into rows that fit the cells left of `lead`
    fn wrap(&self, lead: &[Span<'static>], spans: Vec<Span<'static>>) -> Vec<Vec<Span<'static>>> {
        if self.width == 0 {
            return vec![spans];
        }
        let budget = self
            .width
            .saturating_sub(lead.iter().map(Span::width).sum())
            .max(1);
        wrap_lines(&[Line::from(spans)], budget)
            .into_iter()
            .map(|row| row.spans)
            .collect()
    }

    /// Flush a block's last line when text is pending.
    fn end_block(&mut self) {
        if !self.inline.is_empty() || self.prefix.fresh {
            self.flush_line();
        }
    }

    /// Mark a block boundary.
    fn gap(&mut self) {
        if !self.out.is_empty() && self.lists.is_empty() {
            self.pending_gap = true;
        }
    }

    /// Push a rendered row, emitting the pending block separator first.
    fn push_row(&mut self, line: Line<'static>) {
        self.in_html = false;
        if self.pending_gap && !self.out.is_empty() {
            self.pending_gap = false;
            let blank = self.rail().into_line();
            self.out.push(blank);
        }
        self.out.push(line);
    }

    /// Render the buffered table with padding, a bold header, and a dim rule.
    #[allow(
        unstable_name_collisions,
        reason = "std's Iterator::intersperse is unstable"
    )]
    fn flush_table(&mut self) {
        let Some(table) = self.table.take() else {
            return;
        };
        let Table { head, rows, aligns, .. } = table;
        let columns = aligns
            .len()
            .max(head.len())
            .max(rows.iter().map(Vec::len).max().unwrap_or_default());
        let mut widths = vec![0_usize; columns];
        for row in std::iter::once(&head).chain(&rows) {
            for (i, cell) in row.iter().enumerate() {
                widths[i] = widths[i].max(cell_width(cell));
            }
        }
        // A short row pads with empty cells and a missing alignment pads left.
        let cells = |row: &[Vec<Span<'static>>]| {
            (0..columns)
                .map(|i| {
                    pad_cell(
                        row.get(i).cloned().unwrap_or_default(),
                        widths[i],
                        aligns.get(i).copied().unwrap_or(Alignment::None),
                    )
                })
                .collect::<Vec<_>>()
        };
        // One dash run per column, joined where the cell separators sit.
        let rule: Vec<Span<'static>> = widths
            .iter()
            .copied()
            .map(|w| Span::styled("─".repeat(w), QUIET))
            .intersperse(Span::styled("─┼─", QUIET))
            .collect();
        let surround = self.indents();
        self.push_row(row_line(surround.clone(), cells(&head), true));
        self.push_row(row_line(surround.clone(), vec![rule], false));
        for row in &rows {
            self.push_row(row_line(surround.clone(), cells(row), false));
        }
    }

    /// Close the walk and render stragglers.
    fn finish(mut self) -> Vec<Line<'static>> {
        if !self.inline.is_empty() {
            self.flush_line();
        }
        self.flush_table();
        self.out
    }
}

/// A cell's width in terminal cells.
fn cell_width(cell: &[Span<'static>]) -> usize {
    cell.iter().map(Span::width).sum()
}

/// One cell padded to its column's width and alignment.
fn pad_cell(mut cell: Vec<Span<'static>>, width: usize, align: Alignment) -> Vec<Span<'static>> {
    let fill = width.saturating_sub(cell_width(&cell));
    let (left, right) = match align {
        Alignment::Left | Alignment::None => (0, fill),
        Alignment::Right => (fill, 0),
        Alignment::Center => (fill / 2, fill - fill / 2),
    };
    let lead = " ".repeat(left);
    if !lead.is_empty() {
        match cell.first_mut() {
            Some(first) if first.style == Style::new() => {
                first.content.to_mut().insert_str(0, &lead);
            }
            _ => cell.insert(0, Span::raw(lead)),
        }
    }
    let tail = " ".repeat(right);
    if !tail.is_empty() {
        cell.push_merged(&tail, Style::new());
    }
    cell
}

/// Assemble one table row: the context prefix, then dim cells, then bold header
#[allow(
    unstable_name_collisions,
    reason = "std's Iterator::intersperse is unstable"
)]
fn row_line(
    prefix: Vec<Span<'static>>,
    cells: Vec<Vec<Span<'static>>>,
    header: bool,
) -> Line<'static> {
    let mut line = Line::from(prefix);
    line.extend(
        cells
            .into_iter()
            .map(|cell| {
                cell.into_iter()
                    .map(|span| {
                        if header {
                            Span::styled(span.content, HEADING.patch(span.style))
                        } else {
                            span
                        }
                    })
                    .collect::<Vec<_>>()
            })
            .intersperse_with(|| vec![Span::styled(" │ ", QUIET)])
            .flatten(),
    );
    line
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::{segments, texts};

    /// The rendered rows as plain text.
    fn rows(raw: &str) -> Vec<String> {
        texts(&render(raw, 0))
    }

    /// Plain text renders as paragraph blocks.
    #[test]
    fn plain_text_renders_as_blocks() {
        assert_eq!(rows(""), Vec::<String>::new());
        assert_eq!(rows("one"), ["one"]);
        assert_eq!(rows("one\n"), ["one"]);
        assert_eq!(rows("one\n\n"), ["one"]);
        assert_eq!(rows("one\ntwo"), ["one", "two"]);
        assert_eq!(rows("one\n\ntwo"), ["one", "", "two"]);
        assert_eq!(rows("one\n\n\ntwo"), ["one", "", "two"]);
        assert_eq!(rows("\n\none"), ["one"]);
        // Prose that trips no construct renders literally: names, sums,
        // plain digits, hyphenated words.
        assert_eq!(
            rows("well-known names stay plain"),
            ["well-known names stay plain"]
        );
        assert_eq!(rows("a + b = c"), ["a + b = c"]);
        assert_eq!(rows("1 item"), ["1 item"]);
        // Indented text is code's territory, whichever digits it carries.
        assert_eq!(rows("summary:\n    1. first"), ["summary:", "1. first"]);
    }

    /// Emphasis styles its text and strips its markers
    #[test]
    fn emphasis_styles_and_strips() {
        let spans = &segments(&render("**bold** and *italic* and ~~gone~~", 0))[0];
        assert_eq!(spans[0], ("bold".into(), HEADING));
        assert_eq!(spans[1], (" and ".into(), Style::new()));
        assert_eq!(
            spans[2],
            ("italic".into(), Style::new().add_modifier(Modifier::ITALIC))
        );
        assert_eq!(
            spans[4],
            ("gone".into(), Style::new().add_modifier(Modifier::CROSSED_OUT))
        );
        // An unclosed marker renders literally; streaming is forgiving.
        assert_eq!(rows("**bold"), ["**bold"]);
        // Intraword underscores stay literal: snake_case is a name.
        assert_eq!(rows("snake_case_name"), ["snake_case_name"]);
    }

    /// Inline code renders verbatim in its own color.
    #[test]
    fn inline_code_colors() {
        let spans = &segments(&render("run `cargo test` now", 0))[0];
        assert_eq!(spans[1], ("cargo test".into(), INLINE_CODE));
        assert_eq!(spans[2], (" now".into(), Style::new()));
        // A break's trim never eats code's leading space: verbatim.
        assert_eq!(rows("a\n` x` b"), ["a", " x b"]);
    }

    /// Headings strip their markers and scale by level.
    #[test]
    fn headings_strip_and_scale() {
        let spans = segments(&render("# One\n## Two\n### Three\n###### Six", 0));
        assert_eq!(spans[0][0], ("One".into(), HEADING1));
        assert_eq!(spans[2][0], ("Two".into(), HEADING2));
        assert_eq!(spans[4][0], ("Three".into(), HEADING3));
        assert_eq!(spans[6][0], ("Six".into(), HEADING));
        // A heading still streaming renders with its partial text.
        assert_eq!(segments(&render("# Hea", 0))[0][0], ("Hea".into(), HEADING1));
    }

    /// Fenced and indented code render verbatim and dim.
    #[test]
    fn code_blocks_render_verbatim_dim() {
        assert_eq!(
            segments(&render("```rust\nlet x = **1**;\n    indented\n```", 0)),
            [
                vec![("let x = **1**;".into(), QUIET)],
                vec![("    indented".into(), QUIET)],
            ]
        );
        assert_eq!(
            segments(&render("`x`\n\n    four spaces are code", 0)),
            [
                vec![("x".to_string(), INLINE_CODE)],
                vec![(String::new(), Style::new())],
                vec![("four spaces are code".to_string(), QUIET)],
            ]
        );
        assert_eq!(
            segments(&render("```\nunclosed", 0)),
            [vec![("unclosed".into(), QUIET)]]
        );
    }

    /// Lists render bullets and numbers with hanging indents.
    #[test]
    fn lists_render_markers_and_indents() {
        assert_eq!(
            rows("- one\n- two\n  - nested\n1. first\n2. second"),
            ["• one", "• two", "  • nested", "", "1. first", "2. second"]
        );
        // Continuation lines of an item align under its text.
        assert_eq!(
            rows("- a long item\n  wrapped here"),
            ["• a long item", "  wrapped here"]
        );
    }

    /// Each nesting level indents by its own bullet's width alone.
    #[test]
    fn nested_lists_indent_by_level() {
        assert_eq!(rows("- a\n  - b\n    - c"), ["• a", "  • b", "    • c"]);
        assert_eq!(rows("- a\n  - b\n    wrapped"), ["• a", "  • b", "    wrapped"]);
    }

    /// Wrapped rows keep the hanging indent so a continuation aligns under its item's
    /// text by the marker's width.
    #[test]
    fn wrapped_rows_keep_the_hanging_indent() {
        // `4. ` is three cells; `10. ` is four, a wider marker a wider indent.
        let wrap = |raw: &str, width: usize| texts(&render(raw, width));
        assert_eq!(
            wrap("4. a long numbered line that wraps at a narrow width", 30),
            ["4. a long numbered line that ", "   wraps at a narrow width"]
        );
        assert_eq!(
            wrap("10. ten is two digits wide so the marker is four cells", 30),
            ["10. ten is two digits wide so ", "    the marker is four cells"]
        );
        assert_eq!(
            wrap("- bullet items keep their two space continuation", 20),
            ["• bullet items keep ", "  their two space ", "  continuation"]
        );
        assert_eq!(
            wrap("> a quoted line that is long enough to wrap here", 20),
            ["│ a quoted line that", "│ is long enough to ", "│ wrap here"]
        );
        // A nested item's continuation aligns under the nested text alone.
        assert_eq!(
            wrap("1. outer\n   1. nested item with a long wrapping text line", 26),
            [
                "1. outer",
                "   1. nested item with a ",
                "      long wrapping text ",
                "      line"
            ]
        );
        // Verbatim rows wrap to the indent they sit at.
        assert_eq!(
            wrap("- item\n\n  ```\n  code line long enough to wrap yes\n  ```", 24),
            ["• item", "  code line long enough ", "  to wrap yes"]
        );
    }

    #[test]
    fn item_continuations_and_buffered_blocks_keep_prefixes() {
        assert_eq!(rows("- a\n\n  b"), ["• a", "  b"]);
        assert_eq!(rows("- a\n  - b\n\n  c"), ["• a", "  • b", "  c"]);
        assert_eq!(rows("- item\n\n  ```\n  code\n  ```"), ["• item", "  code"]);
        assert_eq!(rows("> ```\n> code\n> ```"), ["│ code"]);
        assert_eq!(rows("> ---"), [format!("│ {}", "─".repeat(HR_WIDTH))]);
    }

    /// An HTML block renders as one block
    #[test]
    fn html_and_empty_blocks_render_tight() {
        assert_eq!(
            rows("para\n\n<div>\n  x\n</div>\n\nafter"),
            ["para", "", "<div>", "  x", "</div>", "", "after"]
        );
        assert_eq!(rows("```\n```"), Vec::<String>::new());
        assert_eq!(rows("a\n\n```\n```\n\nb"), ["a", "", "b"]);
    }

    /// Blockquotes render dim and italic behind a rail on every line, nested
    /// by depth; the line's quote style italicizes the rail along with the
    /// text, so a quotation reads apart from verbatim, quiet blocks.
    #[test]
    fn blockquotes_render_a_rail() {
        let lines = segments(&render("> quoted\n> more\n\nafter", 0));
        assert_eq!(lines[0][0], ("│ ".into(), QUOTE));
        assert_eq!(lines[0][1], ("quoted".into(), QUOTE));
        assert_eq!(lines[1][0], ("│ ".into(), QUOTE));
        assert_eq!(lines[1][1], ("more".into(), QUOTE));
        // The separating blank carries no rail: the quote has closed.
        assert_eq!(lines[2], vec![(String::new(), Style::new())]);
        assert_eq!(lines[3][0], ("after".into(), Style::new()));
        assert_eq!(rows(">> deep"), ["│ │ deep"]);
    }

    /// A horizontal rule renders as a dim rule row.
    #[test]
    fn rules_render_dim() {
        let lines = segments(&render("above\n\n---\n\nbelow", 0));
        assert_eq!(lines[2][0], ("─".repeat(HR_WIDTH), QUIET));
    }

    /// Blocks separate by exactly one blank row; none leads or trails.
    #[test]
    fn blocks_separate_by_one_blank() {
        assert_eq!(rows("one\n\ntwo"), ["one", "", "two"]);
        assert_eq!(rows("one\n\n\ntwo"), ["one", "", "two"]);
        assert_eq!(rows("# head\n\npara"), ["head", "", "para"]);
        assert_eq!(rows("para\n\n# head"), ["para", "", "head"]);
        assert_eq!(rows("`c`\n\n"), ["c"]);
    }

    #[test]
    fn breaks_continue_the_line() {
        assert_eq!(rows("one\ntwo"), ["one", "two"]);
        assert_eq!(rows("one  \ntwo"), ["one", "two"]);
    }

    #[test]
    fn links_style_labels() {
        let spans = &segments(&render("see [the docs](https://example.com) now", 0))[0];
        assert_eq!(spans[1], ("the docs".into(), LINK));
        assert_eq!(spans[2], (" (https://example.com)".into(), QUIET));
        assert_eq!(spans[3], (" now".into(), Style::new()));
        assert_eq!(rows("<https://example.com>"), ["https://example.com"]);
        // A break before the close keeps the destination's leading space,
        // and a styled label still dedups against its destination.
        assert_eq!(rows("[a\n](u)"), ["a", " (u)"]);
        assert_eq!(rows("[**x**](x)"), ["x"]);
        // An image renders its alt text in brackets with the destination.
        assert_eq!(rows("![alt text](img.png)"), ["[alt text] (img.png)"]);
    }

    #[test]
    fn tables_render_aligned() {
        let lines = segments(&render("| left | mid | right |\n|---|:-:|--:|\n| a | b | c |", 0));
        assert_eq!(lines[0][0], ("left".into(), HEADING));
        assert_eq!(lines[0][1], (" │ ".into(), QUIET));
        assert_eq!(lines[0][2], ("mid".into(), HEADING));
        assert_eq!(lines[0][4], ("right".into(), HEADING));
        assert_eq!(lines[1][0], ("────".into(), QUIET));
        assert_eq!(lines[1][1], ("─┼─".into(), QUIET));
        assert_eq!(lines[1][2], ("───".into(), QUIET));
        assert_eq!(lines[2][0], ("a   ".into(), Style::new()));
        assert_eq!(lines[2][1], (" │ ".into(), QUIET));
        assert_eq!(lines[2][2], (" b ".into(), Style::new()));
        assert_eq!(lines[2][4], ("    c".into(), Style::new()));
        // A table still streaming renders the rows it has.
        assert_eq!(rows("| a | b |\n|---|---|"), ["a │ b", "──┼──"]);
    }
}
