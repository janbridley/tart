//! Render the model's answer text as styled markdown.
//!
//! Only [`Progress::Answer`](tart_agents::Progress::Answer) fragments are pretty-
//! rendered, all others stay base text. Copy mode reads the rendered rows so that a
//! copied answer carries the styled form, markers stripped.

use pulldown_cmark::{Alignment, Event, HeadingLevel, Options, Parser, Tag, TagEnd};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};

use super::DIM_STYLE;

/// H1: blue and bold.
const HEADING1: Style = Style::new().fg(Color::Blue).add_modifier(Modifier::BOLD);
/// H2: cyan and bold.
const HEADING2: Style = Style::new().fg(Color::Cyan).add_modifier(Modifier::BOLD);
/// H3–H6: bold with default color
const HEADING: Style = Style::new().add_modifier(Modifier::BOLD);
/// Inline `` `code` ``: set apart by color rather than markers.
const INLINE_CODE: Style = Style::new().fg(Color::Yellow);
/// Blockquote rails, code-block content, and other quiet chrome.
const QUIET: Style = DIM_STYLE;
/// Link text: the same blue as the transcript's actionable hints.
const LINK: Style = Style::new().fg(Color::Blue);
/// The bullet an unordered list item renders instead of its marker.
const BULLET: &str = "• ";
/// The width of the rule row a horizontal rule renders as
///
/// Should be short, as [`render`] may wrap it.
const HR_WIDTH: usize = 24;

/// Render answer text to styled transcript lines.
pub(crate) fn render(raw: &str) -> Vec<Line<'static>> {
    let mut blocks = Blocks::default();
    for event in Parser::new_ext(raw, options()) {
        blocks.event(event);
    }
    blocks.finish()
}

/// Enable `CommonMark` core plus GitHub's tables and strikethroughs.
fn options() -> Options {
    Options::ENABLE_TABLES | Options::ENABLE_STRIKETHROUGH
}

/// The event walk's state: the rows built so far and the inline context.
#[derive(Default)]
#[allow(
    clippy::struct_excessive_bools,
    reason = "each bool is one mode bit of the walk"
)]
struct Blocks {
    /// The rendered rows.
    out: Vec<Line<'static>>,
    /// The current block's first-line prefix.
    marker: Vec<Span<'static>>,
    /// The continuation prefix the block's later lines take.
    cont: Vec<Span<'static>>,
    /// Whether the next flushed line is the block's first and therefore takes a marker.
    fresh: bool,
    /// Whether a blank row separates the next rendered row from the block above
    pending_gap: bool,
    /// Whether the last rendered row was html and should be appended to.
    in_html: bool,
    /// The inline row under construction; breaks and block ends flush it.
    spans: Vec<Span<'static>>,
    /// Whether the next inline text trims its leading whitespace.
    trim: bool,
    /// The style of inline text; emphasis and links scope patches onto it.
    style: Style,
    /// The inline scopes opened and not yet closed, outermost first.
    scopes: Vec<Scope>,
    /// The style stamped on a whole flushed line.
    block_style: Option<Style>,
    /// The open lists, outermost first.
    lists: Vec<List>,
    /// Blockquote nesting depth.
    quote: usize,
    /// The code block under construction, rendered at its closing tag.
    code: Option<String>,
    /// The table under construction, rendered at its closing tag.
    table: Option<Table>,
}

/// One open inline scope (emphasis or a link) restoring the outer style when it closes.
struct Scope {
    /// The style to restore at this scope's end.
    style: Style,
    /// A link's destination and the label text captured so far.
    link: Option<(String, String)>,
}

/// One open list's numbering and indents.
struct List {
    /// The first number of an ordered list; `None` renders bullets.
    ordered: Option<u64>,
    /// The number the next item renders.
    next: u64,
    /// The continuation indent of the item now open at this level: the
    /// level's indent plus its bullet's width in cells.
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
    /// Fold one parser event into the state.
    fn event(&mut self, event: Event<'_>) {
        match event {
            Event::Start(tag) => self.start(tag),
            Event::End(tag) => self.end(tag),
            Event::Text(text) => {
                if let Some(code) = &mut self.code {
                    code.push_str(&text);
                } else {
                    self.push_inline(&text, self.style);
                }
            }
            Event::Code(text) => {
                // Inline code renders verbatim: a break's trim never eats
                // its leading space.
                self.trim = false;
                self.push_inline(&text, self.style.patch(INLINE_CODE));
            }
            Event::SoftBreak | Event::HardBreak => self.line_break(),
            Event::Rule => {
                self.gap();
                let mut spans = self.prefix();
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
            Event::InlineHtml(html) => self.push_inline(&html, self.style),
            // Task lists, footnotes, and math don't render specially.
            _ => {}
        }
    }

    /// Open a block or inline scope.
    fn start(&mut self, tag: Tag<'_>) {
        match tag {
            Tag::Paragraph => self.open_block(None),
            Tag::Heading { level, .. } => {
                self.open_block(Some(match level {
                    HeadingLevel::H1 => HEADING1,
                    HeadingLevel::H2 => HEADING2,
                    _ => HEADING,
                }));
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
                if !self.spans.is_empty() {
                    self.flush_line();
                }
                let bullet = match self.lists.last() {
                    Some(list) if list.ordered.is_some() => format!("{}. ", list.next),
                    _ => BULLET.to_owned(),
                };
                // Every outer level's indent with no part of its own bullet
                let column = self.lists[..self.lists.len() - 1]
                    .iter()
                    .map(|list| list.cont.as_str())
                    .collect::<String>();
                // Align line continuations by cell count
                let cont = format!("{column}{}", " ".repeat(bullet.chars().count()));
                if let Some(list) = self.lists.last_mut() {
                    list.next = list.next.wrapping_add(1);
                    list.cont = cont;
                }
                let mut marker = self.rail();
                marker.push(Span::raw(format!("{column}{bullet}")));
                self.marker = marker;
                self.cont = self.prefix();
                self.fresh = true;
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
            Tag::Strong => self.scope(|style| style.add_modifier(Modifier::BOLD)),
            Tag::Emphasis => self.scope(|style| style.add_modifier(Modifier::ITALIC)),
            Tag::Strikethrough => self.scope(|style| style.add_modifier(Modifier::CROSSED_OUT)),
            Tag::Link { dest_url, .. } => {
                self.scope(|style| style.patch(LINK));
                if let Some(scope) = self.scopes.last_mut() {
                    scope.link = Some((dest_url.to_string(), String::new()));
                }
            }
            // An image renders its alt text as the label in brackets.
            Tag::Image { dest_url, .. } => {
                self.scope(|style| style.patch(LINK));
                if let Some(scope) = self.scopes.last_mut() {
                    scope.link = Some((dest_url.to_string(), String::new()));
                }
                self.push_inline("[", self.style);
            }
            // The table machinery drives on the cell ends; rows buffer here.
            // An unknown tag changing nothing stays safest.
            _ => {}
        }
    }

    /// Open a paragraph or heading, separate from the block above.
    fn open_block(&mut self, style: Option<Style>) {
        self.gap();
        self.block_style = style;
        if self.lists.is_empty() {
            let rail = self.rail();
            self.marker = rail.clone();
            self.cont = rail;
        } else if !self.fresh {
            let prefix = self.prefix();
            self.marker = prefix.clone();
            self.cont = prefix;
        }
        self.fresh = true;
    }

    /// Close a block or inline scope.
    fn end(&mut self, tag: TagEnd) {
        match tag {
            TagEnd::Paragraph | TagEnd::Item => self.end_block(),
            TagEnd::Heading(_) => {
                self.end_block();
                self.block_style = None;
            }
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
                let cell = std::mem::take(&mut self.spans);
                if let Some(table) = &mut self.table {
                    table.row.push(cell);
                }
            }
            TagEnd::Strong | TagEnd::Emphasis | TagEnd::Strikethrough => {
                if let Some(scope) = self.scopes.pop() {
                    self.style = scope.style;
                }
            }
            TagEnd::Link => {
                if let Some(scope) = self.scopes.pop() {
                    self.style = scope.style;
                    // The destination rides along dimly unless the label finishes
                    if let Some((dest, label)) = scope.link
                        && !dest.is_empty()
                        && dest != label
                    {
                        self.trim = false;
                        self.push_inline(format!(" ({dest})"), QUIET);
                    }
                }
            }
            TagEnd::Image => {
                // Close the bracket while the label style still applies.
                self.push_inline("]", self.style);
                if let Some(scope) = self.scopes.pop() {
                    self.style = scope.style;
                    if let Some((dest, _)) = scope.link
                        && !dest.is_empty()
                    {
                        self.trim = false;
                        self.push_inline(format!(" ({dest})"), QUIET);
                    }
                }
            }
            // An unknown end changing nothing stays safest.
            _ => {}
        }
    }

    /// Open an inline scope restoring the current style at its end.
    fn scope(&mut self, patch: impl Fn(Style) -> Style) {
        let style = self.style;
        self.scopes.push(Scope { style, link: None });
        self.style = patch(style);
    }

    /// The blockquote rail for the current depth, on every line of a quote.
    fn rail(&self) -> Vec<Span<'static>> {
        if self.quote > 0 {
            vec![Span::styled("│ ".repeat(self.quote), QUIET)]
        } else {
            Vec::new()
        }
    }

    /// The prefix rail + indent every row of the current context carries.
    fn prefix(&self) -> Vec<Span<'static>> {
        let mut spans = self.rail();
        let indent = self
            .lists
            .iter()
            .map(|list| list.cont.as_str())
            .collect::<String>();
        if !indent.is_empty() {
            spans.push(Span::raw(indent));
        }
        spans
    }

    /// Push each line of `text` as a quiet row, carrying the context prefix.
    fn quiet_rows(&mut self, text: &str) {
        for line in text.lines() {
            let mut spans = self.prefix();
            spans.push(Span::styled(line.to_owned(), QUIET));
            self.push_row(Line::from(spans));
        }
    }

    /// Append styled inline text, gluing onto the last span when the style
    /// matches, and capturing it into the nearest open link's label.
    fn push_inline(&mut self, text: impl AsRef<str>, style: Style) {
        let text = if self.trim { text.as_ref().trim_start() } else { text.as_ref() };
        if text.is_empty() {
            return;
        }
        self.trim = false;
        if self.spans.last().is_some_and(|last| last.style == style) {
            if let Some(last) = self.spans.last_mut() {
                last.content.to_mut().push_str(text);
            }
        } else {
            self.spans.push(Span::styled(text.to_owned(), style));
        }
        if let Some((_, label)) = self.scopes.iter_mut().rev().find_map(|scope| scope.link.as_mut())
        {
            label.push_str(text);
        }
    }

    /// A soft or hard break: the block continues on the next line, whose
    /// leading indent is insignificant.
    fn line_break(&mut self) {
        if self.table.is_some() {
            // Cell text wraps at the cell, not the row; join with a space.
            self.push_inline(" ", self.style);
        } else {
            self.trim = true;
            self.flush_line();
        }
    }

    /// Flush the inline row as a line.
    fn flush_line(&mut self) {
        let prefix = if self.fresh {
            std::mem::take(&mut self.marker)
        } else {
            self.cont.clone()
        };
        self.fresh = false;
        // Trailing whitespace from the source is invisible; drop it.
        if let Some(last) = self.spans.last_mut() {
            let owned = last.content.to_mut();
            let len = owned.trim_end().len();
            owned.truncate(len);
        }
        let mut line = Line::from(prefix);
        line.spans.append(&mut self.spans);
        if line.spans.is_empty() {
            line.spans.push(Span::raw(""));
        }
        line.style = match (self.block_style, self.quote > 0) {
            (Some(style), _) => style,
            (None, true) => QUIET,
            (None, false) => Style::new(),
        };
        self.push_row(line);
    }

    /// Flush a block's last line when text is pending.
    fn end_block(&mut self) {
        if !self.spans.is_empty() || self.fresh {
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
            let mut blank = Line::from(self.rail());
            if blank.spans.is_empty() {
                blank.spans.push(Span::raw(""));
            }
            self.out.push(blank);
        }
        self.out.push(line);
    }

    /// Render the buffered table with padding, a bold header, and a dim rule.
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
        let mut rule = Vec::with_capacity(columns * 2);
        for (i, w) in widths.iter().enumerate() {
            if i > 0 {
                rule.push(Span::styled("─┼─", QUIET));
            }
            rule.push(Span::styled("─".repeat(*w), QUIET));
        }
        let prefix = self.prefix();
        self.push_row(row_line(prefix.clone(), cells(&head), true));
        self.push_row(row_line(prefix.clone(), vec![rule], false));
        for row in &rows {
            self.push_row(row_line(prefix.clone(), cells(row), false));
        }
    }

    /// Close the walk and render stragglers.
    fn finish(mut self) -> Vec<Line<'static>> {
        if !self.spans.is_empty() {
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
        match cell.last_mut() {
            Some(last) if last.style == Style::new() => {
                last.content.to_mut().push_str(&tail);
            }
            _ => cell.push(Span::raw(tail)),
        }
    }
    cell
}

/// Assemble one table row: the context prefix, then dim cells, then bold header
fn row_line(
    prefix: Vec<Span<'static>>,
    cells: Vec<Vec<Span<'static>>>,
    header: bool,
) -> Line<'static> {
    let mut spans = prefix;
    for (i, cell) in cells.into_iter().enumerate() {
        if i > 0 {
            spans.push(Span::styled(" │ ", QUIET));
        }
        spans.extend(cell.into_iter().map(|span| {
            if header {
                Span::styled(span.content, HEADING.patch(span.style))
            } else {
                span
            }
        }));
    }
    Line::from(spans)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::{segments, texts};

    /// The rendered rows as plain text.
    fn rows(raw: &str) -> Vec<String> {
        texts(&render(raw))
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
        let spans = &segments(&render("**bold** and *italic* and ~~gone~~"))[0];
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
        let spans = &segments(&render("run `cargo test` now"))[0];
        assert_eq!(spans[1], ("cargo test".into(), INLINE_CODE));
        assert_eq!(spans[2], (" now".into(), Style::new()));
        // A break's trim never eats code's leading space: verbatim.
        assert_eq!(rows("a\n` x` b"), ["a", " x b"]);
    }

    /// Headings strip their markers and scale by level.
    #[test]
    fn headings_strip_and_scale() {
        let spans = segments(&render("# One\n## Two\n### Three\n###### Six"));
        assert_eq!(spans[0][0], ("One".into(), HEADING1));
        assert_eq!(spans[2][0], ("Two".into(), HEADING2));
        assert_eq!(spans[4][0], ("Three".into(), HEADING));
        assert_eq!(spans[6][0], ("Six".into(), HEADING));
        // A heading still streaming renders with its partial text.
        assert_eq!(segments(&render("# Hea"))[0][0], ("Hea".into(), HEADING1));
    }

    /// Fenced and indented code render verbatim and dim.
    #[test]
    fn code_blocks_render_verbatim_dim() {
        assert_eq!(
            segments(&render("```rust\nlet x = **1**;\n    indented\n```")),
            [
                vec![("let x = **1**;".into(), QUIET)],
                vec![("    indented".into(), QUIET)],
            ]
        );
        assert_eq!(
            segments(&render("`x`\n\n    four spaces are code")),
            [
                vec![("x".to_string(), INLINE_CODE)],
                vec![(String::new(), Style::new())],
                vec![("four spaces are code".to_string(), QUIET)],
            ]
        );
        assert_eq!(
            segments(&render("```\nunclosed")),
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

    /// Blockquotes render dim behind a rail on every line, nested by depth.
    #[test]
    fn blockquotes_render_a_rail() {
        let lines = segments(&render("> quoted\n> more\n\nafter"));
        assert_eq!(lines[0][0], ("│ ".into(), QUIET));
        assert_eq!(lines[0][1], ("quoted".into(), QUIET));
        assert_eq!(lines[1][0], ("│ ".into(), QUIET));
        assert_eq!(lines[1][1], ("more".into(), QUIET));
        // The separating blank carries no rail: the quote has closed.
        assert_eq!(lines[2], vec![(String::new(), Style::new())]);
        assert_eq!(lines[3][0], ("after".into(), Style::new()));
        assert_eq!(rows(">> deep"), ["│ │ deep"]);
    }

    /// A horizontal rule renders as a dim rule row.
    #[test]
    fn rules_render_dim() {
        let lines = segments(&render("above\n\n---\n\nbelow"));
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
        let spans = &segments(&render("see [the docs](https://example.com) now"))[0];
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
        let lines = segments(&render("| left | mid | right |\n|---|:-:|--:|\n| a | b | c |"));
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
