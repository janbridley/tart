//! The word-wrap engine: styled transcript lines and the draft to display rows.

use ratatui::style::Style;
use ratatui::text::{Line, Span};
use unicode_segmentation::UnicodeSegmentation;

use super::SpansExt;

/// Spaces a tab renders as.
const TAB_WIDTH: usize = 4;

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

/// The draft wrapped for display, plus the caret's cell in it.
pub(crate) struct PromptLayout {
    pub(crate) rows: Vec<Line<'static>>,
    pub(crate) caret_row: usize,
    /// May equal its row's width; paint sites clamp.
    pub(crate) caret_col: usize,
}

/// Wrap the draft and locate the caret's cell
/// The carat should be at the boundary before grapheme `cursor.1` of line `cursor.0`
pub(crate) fn wrap_draft(lines: &[String], cursor: (usize, usize), width: usize) -> PromptLayout {
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
}
