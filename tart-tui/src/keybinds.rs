//! macOS-style editor keybinds: word moves (Option+←/→), word delete
//! (Option+Backspace), and line jumps (Cmd+←/→/Backspace).

use ratatui::crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use unicode_segmentation::UnicodeSegmentation;

use crate::pane::{Editor, g_to_byte, graphemes};

impl Editor {
    /// Grapheme index of the start of the word before grapheme `g`, or 0.
    fn prev_word_start(&self, g: usize) -> usize {
        let mut boundary = 0;
        let mut count = 0;
        for word in self.lines[self.line].split_word_bounds() {
            let len = graphemes(word);
            if count >= g {
                break;
            }
            if !word.trim().is_empty() {
                boundary = count;
            }
            count += len;
        }
        boundary
    }

    /// Grapheme index just past the end of the first word after `g`, or the
    /// line length.
    fn next_word_end(&self, g: usize) -> usize {
        let mut count = 0;
        let mut boundary = self.line_len();
        for word in self.lines[self.line].split_word_bounds() {
            let len = graphemes(word);
            if count + len > g && !word.trim().is_empty() {
                boundary = count + len;
                break;
            }
            count += len;
        }
        boundary
    }

    /// Option+←: to the previous word start, joining across lines.
    pub(crate) fn word_left(&mut self) {
        if self.g > 0 {
            self.g = self.prev_word_start(self.g);
        } else if self.line > 0 {
            self.line -= 1;
            self.g = self.prev_word_start(self.line_len());
        }
    }

    /// Option+→: past the next word end, joining across lines.
    pub(crate) fn word_right(&mut self) {
        if self.g < self.line_len() {
            self.g = self.next_word_end(self.g);
        } else if self.line + 1 < self.lines.len() {
            self.line += 1;
            self.g = self.next_word_end(0);
        }
    }

    /// Option+Backspace: delete through the previous word start. A no-op at a
    /// line start — no joining, matching macOS editors.
    pub(crate) fn delete_word(&mut self) {
        let start = self.prev_word_start(self.g);
        if start < self.g {
            let line = &mut self.lines[self.line];
            line.replace_range(g_to_byte(line, start)..g_to_byte(line, self.g), "");
            self.g = start;
        }
    }

    /// Cmd+Backspace: delete from the caret to the line start.
    pub(crate) fn delete_to_line_start(&mut self) {
        let line = &mut self.lines[self.line];
        line.replace_range(..g_to_byte(line, self.g), "");
        self.g = 0;
    }
}

/// Whether `key` is one of the macOS word/line bindings, applied to `prompt`.
///
/// The Option-modifier fall-through matters: option+letter arrives as
/// `ALT + Char(c)` and must keep reaching `insert_char`.
pub(crate) fn mac_modifiers(prompt: &mut Editor, key: &KeyEvent) -> bool {
    if key.modifiers.contains(KeyModifiers::ALT) {
        match key.code {
            KeyCode::Left => prompt.word_left(),
            KeyCode::Right => prompt.word_right(),
            KeyCode::Backspace => prompt.delete_word(),
            _ => return false,
        }
        return true;
    }
    if key.modifiers.contains(KeyModifiers::SUPER) {
        match key.code {
            KeyCode::Left => prompt.home(),
            KeyCode::Right => prompt.end(),
            KeyCode::Backspace => prompt.delete_to_line_start(),
            _ => return false,
        }
        return true;
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    fn editor(text: &str, line: usize, g: usize) -> Editor {
        let mut e = Editor::default();
        e.insert_str(text);
        e.line = line;
        e.g = g;
        e
    }

    #[test]
    fn word_hops_skip_whitespace_runs_and_join_lines() {
        let mut e = editor("hello   world", 0, 9); // inside "world"
        e.word_left();
        assert_eq!(e.g, 8);
        e.word_left();
        assert_eq!(e.g, 0);
        e.word_right(); // past "hello"
        assert_eq!(e.g, 5);
        e.word_right(); // past "world"
        assert_eq!(e.g, 14);

        // Cross-line joins in both directions.
        let mut e = editor("ab\ncd", 1, 0);
        e.word_left();
        assert_eq!((e.line, e.g), (0, 0));
        e.word_right();
        e.word_right();
        assert_eq!((e.line, e.g), (1, 2));
    }

    #[test]
    fn delete_word_and_line_start() {
        let mut e = editor("one two", 0, 7);
        e.delete_word();
        assert_eq!(e.text(), "one ");
        e.delete_word();
        assert_eq!(e.text(), "");

        // A no-op at a line start: no joining, later lines stay put.
        let mut e = editor("ab\ncd", 1, 0);
        e.delete_word();
        assert_eq!((e.line, e.g), (1, 0));
        assert_eq!(e.lines, ["ab", "cd"]);

        let mut e = editor("ab cd", 0, 5);
        e.delete_to_line_start();
        assert_eq!(e.text(), "");
        assert_eq!(e.g, 0);
    }

    /// The word boundary scans are clamped at both edges.
    #[test]
    fn word_boundaries_clamp() {
        let e = editor("hi", 0, 0);
        assert_eq!(e.prev_word_start(0), 0);
        let e = editor("hi", 0, 2);
        assert_eq!(e.next_word_end(2), 2);
    }

    /// Only Option+arrows/Backspace are claimed; option-chars fall through, and
    /// the Command bindings map to line jumps.
    #[test]
    fn modifier_routing() {
        let mut e = editor("a b", 0, 3);
        assert!(mac_modifiers(&mut e, &KeyEvent::new(KeyCode::Left, KeyModifiers::ALT)));
        assert_eq!(e.g, 2);
        assert!(!mac_modifiers(
            &mut e,
            &KeyEvent::new(KeyCode::Char('f'), KeyModifiers::ALT)
        ));

        assert!(mac_modifiers(&mut e, &KeyEvent::new(KeyCode::Left, KeyModifiers::SUPER)));
        assert_eq!(e.g, 0);
        assert!(mac_modifiers(&mut e, &KeyEvent::new(KeyCode::Backspace, KeyModifiers::SUPER)));
        assert_eq!(e.text(), "");
    }
}
