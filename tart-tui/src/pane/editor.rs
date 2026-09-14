//! The prompt's draft editor.

use unicode_segmentation::UnicodeSegmentation;

use super::wrap::wrap_draft;

/// ⌃Y history
#[derive(Clone, Default)]
struct Undo {
    /// Snapshots: (lines, line, g) before each edit group.
    steps: Vec<(Vec<String>, usize, usize)>,
    /// Whether the open group is a run of single-character edits.
    chars: bool,
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
    pub(crate) top: usize,
    /// Cell width the draft was last rendered at; row-wise motion wraps to it.
    pub(crate) width: usize,
    /// Prompt snapshot history for ⌃Y undos.
    undo: Undo,
}

impl Default for Editor {
    fn default() -> Self {
        Self {
            lines: vec![String::new()],
            line: 0,
            g: 0,
            top: 0,
            // Unwrapped until the first render: rows then match logical lines.
            width: usize::MAX,
            undo: Undo::default(),
        }
    }
}

impl Editor {
    /// The whole draft, lines joined by '\n'.
    pub(crate) fn text(&self) -> String {
        self.lines.join("\n")
    }

    pub(crate) fn clear(&mut self) {
        // An already-empty draft has nothing to rewind.
        if self.line == 0 && self.g == 0 && self.lines.len() == 1 && self.lines[0].is_empty() {
            return;
        }
        self.checkpoint(false);
        self.lines = vec![String::new()];
        self.line = 0;
        self.g = 0;
        self.top = 0;
    }

    /// Drop the undo history.
    pub(crate) fn forget_undo(&mut self) {
        self.undo.forget();
    }

    /// Graphemes on the current line.
    pub(crate) fn line_len(&self) -> usize {
        graphemes(&self.lines[self.line])
    }

    /// Insert one character; controls are ignored except tab.
    pub(crate) fn insert_char(&mut self, c: char) {
        if insertable(c) {
            self.checkpoint(true);
            let line = &mut self.lines[self.line];
            line.insert(g_to_byte(line, self.g), c);
            self.g += 1;
        }
    }

    /// Insert pasted text: CRLF normalized, controls dropped except tab, split on \n.
    pub(crate) fn insert_str(&mut self, text: &str) {
        if text.is_empty() {
            return; // an empty paste changes nothing
        }
        self.checkpoint(false);
        let mut parts = text.lines();
        self.insert_fragment(parts.next().unwrap_or_default());
        for part in parts {
            self.new_line();
            self.insert_fragment(part);
        }
    }

    /// Splice one line's worth of pasted text, cleaned, in at the caret.
    fn insert_fragment(&mut self, part: &str) {
        let cleaned: String = part.chars().filter(|c| insertable(*c)).collect();
        let line = &mut self.lines[self.line];
        line.insert_str(g_to_byte(line, self.g), &cleaned);
        self.g += graphemes(&cleaned);
    }

    /// Split the draft at the caret (Alt+Enter).
    pub(crate) fn new_line(&mut self) {
        self.checkpoint(false);
        let line = &mut self.lines[self.line];
        let tail = line.split_off(g_to_byte(line, self.g));
        self.lines.insert(self.line + 1, tail);
        self.line += 1;
        self.g = 0;
    }

    /// Delete the previous grapheme; at a line start, join with the line above.
    pub(crate) fn backspace(&mut self) {
        // Nothing to delete at the draft's very start.
        if self.g == 0 && self.line == 0 {
            return;
        }
        self.checkpoint(true);
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
    pub(crate) fn right(&mut self) {
        if self.g < self.line_len() {
            self.g += 1;
        } else if self.line + 1 < self.lines.len() {
            self.line += 1;
            self.g = 0;
        }
    }

    /// One rendered row up, keeping the caret's column where the row allows.
    pub(crate) fn up(&mut self) {
        let layout = wrap_draft(&self.lines, (self.line, self.g), self.width);
        if layout.caret_row > 0 {
            let col = layout.caret_col;
            self.row_home();
            self.left(); // onto the end of the row above
            self.row_home();
            self.seek_col(col);
        }
    }

    /// One rendered row down, keeping the caret's column where the row allows.
    pub(crate) fn down(&mut self) {
        let layout = wrap_draft(&self.lines, (self.line, self.g), self.width);
        if layout.caret_row + 1 < layout.rows.len() {
            let col = layout.caret_col;
            self.row_end();
            self.right(); // onto the start of the row below
            self.seek_col(col);
        }
    }

    /// To the start of the rendered row (Cmd+Left).
    pub(crate) fn row_home(&mut self) {
        self.sweep(false);
    }

    /// To the end of the rendered row (Cmd+Right).
    pub(crate) fn row_end(&mut self) {
        self.sweep(true);
    }

    /// To the rendered row's start (Cmd+Left), stepping up to the previous
    /// row's start when the caret already sits there, as readline does.
    pub(crate) fn home(&mut self) {
        let at = (self.line, self.g);
        self.row_home();
        if (self.line, self.g) == at {
            self.left(); // onto the row above's end, or nowhere at the top
            self.row_home();
        }
    }

    /// To the rendered row's end (Cmd+Right), stepping down to the next row's
    /// end when the caret already sits there, as readline does.
    pub(crate) fn end(&mut self) {
        let at = (self.line, self.g);
        self.row_end();
        if (self.line, self.g) == at {
            self.right(); // onto the row below's start, or nowhere at the bottom
            self.row_end();
        }
    }

    /// The caret's rendered row at the last wrapped width.
    fn row(&self) -> usize {
        wrap_draft(&self.lines, (self.line, self.g), self.width).caret_row
    }

    /// Walk one grapheme at a time while the caret stays on its rendered row,
    /// keeping the last position that did.
    fn sweep(&mut self, forward: bool) {
        let row = self.row();
        let mut last = (self.line, self.g);
        loop {
            let before = (self.line, self.g);
            if forward {
                self.right();
            } else {
                self.left();
            }
            if (self.line, self.g) == before || self.row() != row {
                (self.line, self.g) = last;
                return;
            }
            last = (self.line, self.g);
        }
    }

    /// On the caret's rendered row, advance to the last cell at or before `col`.
    fn seek_col(&mut self, col: usize) {
        let row = self.row();
        loop {
            let before = (self.line, self.g);
            self.right();
            let layout = wrap_draft(&self.lines, (self.line, self.g), self.width);
            if (self.line, self.g) == before || layout.caret_row != row || layout.caret_col > col {
                (self.line, self.g) = before;
                return;
            }
        }
    }

    /// ⌃Y: step the draft back one edit; a run of single characters is one step.
    pub(crate) fn undo(&mut self) {
        if let Some((lines, line, g)) = self.undo.step_back() {
            self.lines = lines;
            self.line = line;
            self.g = g;
        }
    }

    /// Snapshot the draft before an edit; consecutive character edits share one undo step.
    pub(crate) fn checkpoint(&mut self, chars: bool) {
        self.undo.checkpoint((&self.lines, self.line, self.g), chars);
    }
}

impl Undo {
    /// Snapshot the draft before an edit.
    fn checkpoint(&mut self, draft: (&[String], usize, usize), chars: bool) {
        if !chars || !self.chars {
            let (lines, line, g) = draft;
            self.steps.push((lines.to_vec(), line, g));
        }
        self.chars = chars;
    }

    /// The group to step back to, closing the open character run.
    fn step_back(&mut self) -> Option<(Vec<String>, usize, usize)> {
        self.chars = false;
        self.steps.pop()
    }

    /// Drop every group: a draft that shipped out must not come back.
    fn forget(&mut self) {
        self.steps.clear();
    }
}

pub(crate) fn graphemes(s: &str) -> usize {
    s.graphemes(true).count()
}

/// Whether a character may enter the draft: controls are dropped except tab.
#[inline]
fn insertable(c: char) -> bool {
    !c.is_control() || c == '\t'
}

/// Byte offset of grapheme boundary `g` (the string end when `g` is the count).
pub(crate) fn g_to_byte(s: &str, g: usize) -> usize {
    s.grapheme_indices(true).nth(g).map_or(s.len(), |(i, _)| i)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Editing operates on graphemes: a line-start backspace joins lines, the caret
    /// caret rides a join, and control characters never enter the draft.
    #[test]
    fn editing_operates_on_graphemes() {
        let mut editor = Editor::default();
        editor.insert_str("日本\n語");
        editor.row_home(); // no width yet: the logical line's start
        editor.backspace(); // joins the lines at the boundary
        assert_eq!(editor.text(), "日本語");
        assert_eq!((editor.line, editor.g), (0, 2));
        editor.insert_char('\u{7}'); // control: ignored
        assert_eq!(editor.text(), "日本語");

        // Left/right cross line joins; the family emoji is one step.
        let mut editor = Editor::default();
        editor.insert_str("ab\n🙋‍♂️x");
        editor.left();
        editor.left();
        assert_eq!((editor.line, editor.g), (1, 0)); // line start
        editor.left();
        assert_eq!((editor.line, editor.g), (0, 2)); // joins above
        editor.right();
        assert_eq!((editor.line, editor.g), (1, 0)); // and back
    }

    /// A typed run undoes as one step, not one character at a time; pastes
    /// and splits are steps of their own.
    #[test]
    fn undo_steps_back_by_edit_group() {
        let mut editor = Editor::default();
        for c in "hi there".chars() {
            editor.insert_char(c);
        }
        editor.undo();
        assert_eq!(editor.text(), "");
        editor.undo(); // nothing left: a no-op
        assert_eq!(editor.text(), "");

        // Typing and backspacing form one group; the paste after is another.
        let mut editor = Editor::default();
        editor.insert_char('a');
        editor.insert_char('b');
        editor.backspace();
        editor.insert_str("paste");
        editor.undo();
        assert_eq!(editor.text(), "a");
        editor.undo();
        assert_eq!(editor.text(), "");
    }

    /// Arrows move by rendered rows once a wrap width is known, keeping the
    /// column; the clamping carries across shorter rows and empty lines.
    #[test]
    fn arrows_move_by_rendered_rows() {
        // Before any render the rows are the logical lines.
        let mut editor = Editor::default();
        editor.insert_str("ab\ncd");
        editor.line = 1;
        editor.g = 1;
        editor.up();
        assert_eq!((editor.line, editor.g), (0, 1));
        editor.down();
        assert_eq!((editor.line, editor.g), (1, 1));

        // Width 5: rows "hello", "world", "ab", "", "xy".
        editor = Editor::default();
        editor.insert_str("hello world\nab\n\nxy");
        editor.width = 5;
        editor.line = 0;
        editor.g = 3;
        editor.row_end();
        assert_eq!((editor.line, editor.g), (0, 5)); // before the dropped space
        editor.row_end(); // already at the row's end
        assert_eq!((editor.line, editor.g), (0, 5));
        editor.down(); // the column rides down from the row's end
        assert_eq!((editor.line, editor.g), (0, 11));
        editor.up(); // and back up to the row's end
        assert_eq!((editor.line, editor.g), (0, 5));
        editor.g = 3;
        editor.down(); // a mid-row column is kept
        assert_eq!((editor.line, editor.g), (0, 9));
        editor.row_home();
        assert_eq!((editor.line, editor.g), (0, 6)); // the space is not in the row
        editor.down();
        assert_eq!((editor.line, editor.g), (1, 0)); // column 0
        editor.down();
        assert_eq!((editor.line, editor.g), (2, 0)); // the empty row
        editor.down();
        assert_eq!((editor.line, editor.g), (3, 0));
        editor.down(); // no row below: stays put
        assert_eq!((editor.line, editor.g), (3, 0));
        editor.up();
        assert_eq!((editor.line, editor.g), (2, 0));
        editor.up();
        assert_eq!((editor.line, editor.g), (1, 0));
        editor.up();
        assert_eq!((editor.line, editor.g), (0, 6)); // row start, past the space
        editor.up();
        assert_eq!((editor.line, editor.g), (0, 0)); // and on to the first row
        editor.up(); // no row above: stays put
        assert_eq!((editor.line, editor.g), (0, 0));
    }
}
