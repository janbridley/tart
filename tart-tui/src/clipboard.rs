//! [`Selection`] objects track spans of selected text, which can be copied to the
//! system clipboard.

use std::io;

use crossterm::clipboard::CopyToClipboard;
use crossterm::execute;
use itertools::Itertools;
use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::{Color, Style};
use ratatui::text::{Line, Span};
use unicode_segmentation::UnicodeSegmentation;

/// The selection band's background.
const SELECT_BG: Color = Color::DarkGray;
/// The style painted across a selection: the band, foreground untouched.
const SELECT_STYLE: Style = Style::new().bg(SELECT_BG);

/// A copy-mode selection, indexing two points in the output transcript.
///
/// NOTE: Either end of this can be fixed while the other moves!
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct Selection {
    /// The first selected cell, (row, col).
    start: (usize, usize),
    /// The last selected cell, (row, col).
    end: (usize, usize),
}

impl Selection {
    /// The selection between `anchor` and `cursor`: `None` before space is pressed.
    pub(crate) fn between(anchor: Option<(usize, usize)>, cursor: (usize, usize)) -> Option<Self> {
        anchor.map(|anchor| {
            if anchor <= cursor {
                Self { start: anchor, end: cursor }
            } else {
                Self { start: cursor, end: anchor }
            }
        })
    }

    /// The selected text, including whitespace introduced by the wrapping.
    pub(crate) fn text(self, rows: &[Line<'static>]) -> String {
        if rows.is_empty() {
            return String::new();
        }
        let last = rows.len() - 1;
        let (r0, r1) = (self.start.0.min(last), self.end.0.min(last));
        rows[r0..=r1]
            .iter()
            .enumerate()
            .map(|(i, row)| {
                let a = if i == 0 { self.start.1 } else { 0 };
                let b = if i == r1 - r0 { self.end.1 } else { usize::MAX };
                row_slice(row, a, b)
            })
            .join("\n")
    }

    /// Paint the selection region.
    pub(crate) fn paint(
        self,
        buf: &mut Buffer,
        rows: &[Line<'static>],
        area: Rect,
        top: usize,
        shown: usize,
    ) {
        for row in self.start.0..=self.end.0.min(rows.len().saturating_sub(1)) {
            // Off-window and empty rows paint nothing.
            let w = rows.get(row).map_or(0, Line::width);
            if w == 0 || row < top || row >= top + shown {
                continue;
            }
            let a = if row == self.start.0 { self.start.1.min(w - 1) } else { 0 };
            let b = if row == self.end.0 { (self.end.1 + 1).min(w) } else { w };
            let rect = Rect::new(area.x + a as u16, area.y + (row - top) as u16, (b - a) as u16, 1);
            buf.set_style(rect, SELECT_STYLE);
            // A glyph the same color as the band should be recolored to show up.
            for x in rect.x..rect.right() {
                if let Some(cell) = buf.cell_mut((x, rect.y))
                    && cell.fg == SELECT_BG
                {
                    cell.set_fg(Color::Reset);
                }
            }
        }
    }
}

/// The graphemes of `row` occupying cells `a..=b`, cell widths measured the wrapper.
fn row_slice(row: &Line<'static>, a: usize, b: usize) -> String {
    let mut out = String::new();
    let mut col = 0;
    for g in row.spans.iter().flat_map(|s| s.content.graphemes(true)) {
        let w = Span::raw(g).width();
        if col <= b && col + w.max(1) > a {
            out.push_str(g);
        }
        col += w;
    }
    out
}

/// Store `text` in the terminal clipboard.
pub(crate) fn copy(text: &str) -> io::Result<()> {
    execute!(io::stdout(), CopyToClipboard::to_clipboard_from(text))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rows(texts: &[&str]) -> Vec<Line<'static>> {
        texts.iter().map(|t| Line::from(t.to_string())).collect()
    }

    fn sel(a: (usize, usize), c: (usize, usize)) -> Selection {
        Selection::between(Some(a), c).expect("anchored")
    }

    /// The selection slices cells out of the wrapped rows, either end may have moved
    /// last, and rows join with '\n'.
    #[test]
    fn text_slices_cells_and_joins_rows() {
        let rows = rows(&["abc", "def"]);
        assert_eq!(sel((0, 1), (1, 1)).text(&rows), "bc\nde");
        assert_eq!(sel((1, 1), (0, 1)).text(&rows), "bc\nde");
        assert_eq!(sel((0, 0), (0, 2)).text(&rows), "abc");
        assert_eq!(sel((1, 2), (1, 2)).text(&rows), "f");
        assert_eq!(sel((0, 1), (0, 1)).text(&[]), "");
        assert!(Selection::between(None, (0, 1)).is_none());
    }

    /// A cursor on a wide grapheme's trailing cell selects that grapheme.
    #[test]
    fn text_covers_wide_graphemes() {
        let rows = rows(&["日", "本", "語"]);
        assert_eq!(sel((0, 0), (1, 0)).text(&rows), "日\n本");
        assert_eq!(sel((0, 1), (0, 1)).text(&rows), "日");
    }

    /// Whole middle rows come along, empty rows included as empty lines.
    #[test]
    fn text_spans_middle_and_empty_rows() {
        let rows = rows(&["aa bb", "cc", "", "dd"]);
        assert_eq!(sel((0, 0), (3, 1)).text(&rows), "aa bb\ncc\n\ndd");
    }
}
