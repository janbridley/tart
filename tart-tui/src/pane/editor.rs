//! The prompt's draft editor.

use unicode_segmentation::UnicodeSegmentation;

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
    pub(crate) fn insert_char(&mut self, c: char) {
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
    pub(crate) fn new_line(&mut self) {
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
    pub(crate) fn right(&mut self) {
        if self.g < self.line_len() {
            self.g += 1;
        } else if self.line + 1 < self.lines.len() {
            self.line += 1;
            self.g = 0;
        }
    }

    /// One logical line up; the grapheme index carries over, clamped.
    pub(crate) fn up(&mut self) {
        if self.line > 0 {
            self.line -= 1;
            self.g = self.g.min(self.line_len());
        }
    }

    /// One logical line down; the grapheme index carries over, clamped.
    pub(crate) fn down(&mut self) {
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Editing operates on graphemes: a line-start backspace joins lines, the caret
    /// caret rides a join, and control characters never enter the draft.
    #[test]
    fn editing_operates_on_graphemes() {
        let mut editor = Editor::default();
        editor.insert_str("日本\n語");
        editor.home();
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
}
