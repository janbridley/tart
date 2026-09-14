//! The word-wrap engine: styled transcript lines and the draft to display rows.

use ratatui::style::Style;
use ratatui::text::{Line, Span};
use unicode_segmentation::UnicodeSegmentation;

use super::SpansExt;

/// Spaces a tab renders as.
const TAB_WIDTH: usize = 4;

/// One grapheme's cell width, never less than one.
#[inline]
pub(crate) fn grapheme_width(grapheme: &str) -> usize {
    if grapheme.len() == 1 {
        1
    } else {
        Span::raw(grapheme).width().max(1)
    }
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

    /// Add one rendered cell; `sym` is a single space for expanded tabs.
    fn push(&mut self, sym: &'a str, style: Style) {
        // Single-byte symbols can be printed without looking up width.
        let gw = grapheme_width(sym);
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
            spans.push_merged(sym, style);
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
pub(crate) fn wrap_lines(messages: &[Line<'static>], width: usize) -> Vec<Line<'static>> {
    let mut wrapper = Wrapper::new(width);
    for line in messages {
        for span in &line.spans {
            let style = line.style.patch(span.style);
            for grapheme in span.content.graphemes(true) {
                feed(&mut wrapper, grapheme, style);
            }
        }
        wrapper.hard_break();
    }
    wrapper.rows
}

/// Wrap one line whose continuation rows hang `indent` cells in, keeping the indent.
pub(crate) fn wrap_hanging(
    line: &Line<'static>,
    width: usize,
    indent: usize,
) -> Vec<Line<'static>> {
    let (lead, rest) = split_line(line, indent);
    let mut rows = wrap_lines(std::slice::from_ref(&rest), width.saturating_sub(indent).max(1));
    let mut first = lead;
    first.spans.extend(rows.remove(0).spans);
    rows.insert(0, first);
    for row in &mut rows[1..] {
        row.spans.insert(0, Span::raw(" ".repeat(indent)));
    }
    rows
}

/// Split `line` after its first `at` cells into its lead and the rest, keeping style.
fn split_line(line: &Line<'static>, at: usize) -> (Line<'static>, Line<'static>) {
    let mut lead = Line::default().style(line.style);
    let mut rest = Line::default().style(line.style);
    let mut cells = 0;
    for span in &line.spans {
        let style = line.style.patch(span.style);
        for grapheme in span.content.graphemes(true) {
            // A wide grapheme at the boundary rides `rest` so it doesn't get split
            if cells + grapheme_width(grapheme) <= at {
                cells += grapheme_width(grapheme);
                lead.spans.push_merged(grapheme, style);
            } else {
                rest.spans.push_merged(grapheme, style);
            }
        }
    }
    (lead, rest)
}

/// The draft wrapped for display, plus the caret's cell in it.
pub(crate) struct PromptLayout {
    pub(crate) rows: Vec<Line<'static>>,
    pub(crate) caret_row: usize,
    /// May equal its row's width; paint sites clamp.
    pub(crate) caret_col: usize,
}

/// Wrap the draft to display rows and locate the caret's cell: the boundary
/// before grapheme `cursor.1` of line `cursor.0`, found where that grapheme
/// paints among the finished rows.
pub(crate) fn wrap_draft(lines: &[String], cursor: (usize, usize), width: usize) -> PromptLayout {
    let mut wrapper = Wrapper::new(width);
    // The row each logical line's painting begins on, so boundary matching
    // below knows where one line's cells end and the next's begin.
    let mut starts = Vec::with_capacity(lines.len());
    for line in lines {
        starts.push(wrapper.rows.len());
        for grapheme in line.graphemes(true) {
            feed(&mut wrapper, grapheme, Style::new());
        }
        wrapper.hard_break();
    }
    let caret = (!lines.is_empty()).then(|| {
        let (cl, mut gc) = cursor;
        let cl = cl.min(lines.len() - 1);
        gc = gc.min(lines[cl].graphemes(true).count());
        caret_in_rows(&wrapper.rows, lines, &starts, (cl, gc))
    });
    PromptLayout {
        rows: wrapper.rows,
        caret_row: caret.map_or(0, |c| c.0),
        caret_col: caret.map_or(0, |c| c.1),
    }
}

/// The cursor's cell among already-wrapped rows: walk the painted graphemes
/// beside the draft's, skipping whatever wrapping dropped — spaces at row
/// breaks, control characters — so every boundary rides where its grapheme
/// finally paints, even when a later wrap carried its word to a lower row
/// mid-pass.
fn caret_in_rows(
    rows: &[Line<'static>],
    lines: &[String],
    starts: &[usize],
    target: (usize, usize),
) -> (usize, usize) {
    let (tl, tg) = target;
    // One entry per painted grapheme: (row, starting col, width, text).
    let mut cells: Vec<(usize, usize, usize, &str)> = Vec::new();
    for (ri, row) in rows.iter().enumerate() {
        let mut col = 0;
        for g in row.spans.iter().flat_map(|s| s.content.graphemes(true)) {
            let w = grapheme_width(g);
            cells.push((ri, col, w, g));
            col += w;
        }
    }
    let mut next = 0; // index into `cells`
    let mut last = (0, 0); // just past the last matched grapheme
    for (li, line) in lines.iter().enumerate() {
        // This line paints exactly the rows up to the next line's start;
        // matching never crosses that boundary, so a dropped trailing
        // space cannot swallow the next line's leading one and a tab's
        // expansion cannot eat past its own line.
        let end_row = starts.get(li + 1).copied().unwrap_or(usize::MAX);
        let count = line.graphemes(true).count();
        let mut painted_any = false;
        for (gi, g) in line.graphemes(true).enumerate() {
            // A dropped control paints nowhere; a tab paints as spaces.
            let dropped = g.chars().any(char::is_control) && g != "\t";
            let (wants, max) = if g == "\t" { (" ", TAB_WIDTH) } else { (g, 1) };
            let mut boundary = last;
            if !dropped {
                let mut taken = 0;
                while taken < max
                    && matches!(cells.get(next), Some(&(row, _, _, cg)) if cg == wants && row < end_row)
                {
                    let (row, col, width, _) = cells[next];
                    if taken == 0 {
                        boundary = (row, col); // the grapheme's first cell
                    }
                    last = (row, col + width);
                    next += 1;
                    taken += 1;
                }
                painted_any |= taken > 0;
            }
            if (li, gi) == (tl, tg) {
                return boundary;
            }
        }
        // The line's end boundary: past its last grapheme, or on the empty
        // row an unpainted line still owns.
        let line_end = if painted_any { last } else { (starts[li], 0) };
        if li == tl && tg == count {
            return line_end;
        }
    }
    (0, 0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::texts;

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
    fn continuation_rows_hang_under_the_gutter() {
        let rows = wrap_hanging(&Line::from("❯ one two three four five six"), 14, 2);
        assert_eq!(texts(&rows), ["❯ one two ", "  three four ", "  five six"]);
        // No indent: exactly what `wrap_lines` would produce.
        assert_eq!(
            texts(&wrap_hanging(&Line::from("ab cd".to_string()), 2, 0)),
            texts(&wrap_lines(&[Line::from("ab cd".to_string())], 2))
        );
    }

    #[test]
    fn every_continuation_row_hangs_whatever_the_break_path() {
        let rows = wrap_hanging(&Line::from("❯ one two xyzzy"), 10, 2);
        assert_eq!(texts(&rows), ["❯ one two ", "  xyzzy"]);
        let rows = wrap_hanging(&Line::from("❯ aaaaaa bbbbbbbbbbbb"), 8, 2);
        assert_eq!(texts(&rows), ["❯ aaaaaa", "  bbbbbb", "  bbbbbb"]);
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

        // At a width that keeps the wrapping space, the boundary past it is
        // the next row's first cell, not the row above's brim.
        let kept = |g: usize| {
            let layout = wrap_draft(&["hello world".to_string()], (0, g), 6);
            (layout.caret_row, layout.caret_col)
        };
        assert_eq!(kept(5), (0, 5)); // before the kept space
        assert_eq!(kept(6), (1, 0)); // the "world" row's start

        // A word a later wrap splits mid-pass moves down whole: at width 4
        // the rows are "hell" / "o " / "worl" / "d", so the boundary at `w`
        // rides row 2, where `w` finally paints — not row 1 where it first
        // landed while the row was still growing.
        let split = |g: usize| {
            let layout = wrap_draft(&["hello world".to_string()], (0, g), 4);
            (layout.caret_row, layout.caret_col)
        };
        assert_eq!(split(6), (2, 0)); // `w`, moved down by the later split
        assert_eq!(split(7), (2, 1));
        assert_eq!(split(11), (3, 1)); // the line end, past `d`
    }

    /// Hard breaks own their rows: a line's dropped trailing space leaves the
    /// next line's leading space to that line, a tab's expansion stops at its
    /// own line's rows, and empty lines sit on the rows they own.
    #[test]
    fn boundaries_respect_hard_breaks() {
        // "aa\t" wraps to "aa " / "  " at width 3; " x" paints " x".
        let lines = ["aa\t".to_string(), " x".to_string()];
        let caret = |target: (usize, usize)| {
            let l = wrap_draft(&lines, target, 3);
            (l.caret_row, l.caret_col)
        };
        assert_eq!(caret((0, 3)), (1, 2)); // line 0's end, past the tab
        assert_eq!(caret((1, 0)), (2, 0)); // before line 1's space
        assert_eq!(caret((1, 1)), (2, 1)); // `x`'s own cell
        assert_eq!(caret((1, 2)), (2, 2)); // line 1's end

        // "ab " at width 2 drops the space: the empty line below owns row 2.
        let lines = ["ab ".to_string(), String::new()];
        assert_eq!(wrap_draft(&lines, (1, 0), 2).caret_row, 2);

        // Two empty lines after the dropped-space line each own their row.
        let lines = ["ab ".to_string(), String::new(), String::new()];
        assert_eq!(wrap_draft(&lines, (1, 0), 2).caret_row, 2);
        assert_eq!(wrap_draft(&lines, (2, 0), 2).caret_row, 3);
    }

    /// Every boundary's caret sits on the row where its grapheme actually
    /// paints, whatever the wrap path — word wraps, hard breaks, words a
    /// later wrap moved down mid-pass, tabs, multi-line drafts.
    #[test]
    fn caret_row_matches_painted_row() {
        let drafts = [
            vec!["hello world".to_string()],
            vec!["aaaa bbbb `cccccccccccccccccc` dddd".to_string()],
            vec!["x aaaaaaaaaaaaaaaaaaaaaaaaaaaa y".to_string()],
            vec!["duis id aute     `asdfasdfasasdf `".to_string()],
            vec!["aa\tbb cc".to_string()],
            vec!["one two".to_string(), "three four".to_string()],
            vec!["aaaaaaaaaaaaaaaaaa `bb cc` tail".to_string(), "z".to_string()],
            vec!["aa\t".to_string(), " x".to_string()],
            vec!["ab ".to_string(), String::new()],
            vec![" x".to_string(), "y".to_string()],
        ];
        for lines in &drafts {
            for width in 3..24usize {
                let rows = wrap_draft(lines, (0, 0), width).rows;
                // The painted graphemes in order, each with its row; spaces
                // may be dropped at wraps and tabs expand, so match on the
                // visible letters only.
                let painted: Vec<(usize, &str)> = rows
                    .iter()
                    .enumerate()
                    .flat_map(|(ri, r)| {
                        r.spans
                            .iter()
                            .flat_map(|s| s.content.graphemes(true))
                            .map(move |g| (ri, g))
                    })
                    .filter(|(_, g)| *g != " ")
                    .collect();
                let mut seen = 0usize;
                for (li, line) in lines.iter().enumerate() {
                    for (gi, g) in line.graphemes(true).enumerate() {
                        if g != " " && g != "\t" {
                            assert_eq!(
                                wrap_draft(lines, (li, gi), width).caret_row,
                                painted[seen].0,
                                "lines={lines:?} w={width} boundary ({li},{gi}) {g:?}"
                            );
                            seen += 1;
                        }
                    }
                }
            }
        }
    }
}
