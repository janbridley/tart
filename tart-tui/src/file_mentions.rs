//! The `@file` mention popup.
//!
//! Typing a word-start `@` in the prompt opens a typeahead file selector over
//! the working directory (gitignored and hidden entries skipped).

use ignore::WalkBuilder;
use nucleo_matcher::pattern::{CaseMatching, Normalization, Pattern};
use nucleo_matcher::{Config, Matcher};
use ratatui::Frame;
use ratatui::crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::layout::Rect;
use ratatui::style::{Modifier, Style};
use ratatui::text::Line;
use ratatui::widgets::{Block, Clear, List, ListItem, ListState};

use crate::pane::{Editor, Popup, g_to_byte, graphemes};

/// Bound the matcher's output; the list only renders its visible window.
const MAX_SHOWN: usize = 256;
const HIGHLIGHT: Style = Style::new().add_modifier(Modifier::REVERSED);

/// The suffix and byte offset of the last word-start `@` on the caret's line.
pub(crate) fn derive_query(editor: &Editor) -> Option<(String, usize)> {
    let line = &editor.lines[editor.line];
    let prefix = &line[..g_to_byte(line, editor.g)];
    let at = prefix.rfind('@')?;
    let word_start = prefix[..at].chars().next_back().is_none_or(char::is_whitespace);
    word_start.then(|| (prefix[at + 1..].to_string(), at))
}

/// All non-ignored files under the current directory, as sorted relative paths.
fn walk_current_directory() -> Vec<String> {
    let root = std::env::current_dir().unwrap_or_default();
    let mut files: Vec<String> = WalkBuilder::new(&root)
        .build()
        .filter_map(std::result::Result::ok)
        .filter(|e| e.file_type().is_some_and(|ft| ft.is_file()))
        .filter_map(|e| {
            e.into_path()
                .strip_prefix(&root)
                .ok()
                .map(|p| p.to_string_lossy().into_owned())
        })
        .collect();
    files.sort();
    files
}

pub struct FilePopup {
    /// Cached directory snapshot from when the popup opened.
    files: Vec<String>,
    query: String,
    /// Matched paths, sorted by relevance and capped at `MAX_SHOWN`.
    matches: Vec<String>,
    /// Total hits before the cap (for the title's `+`).
    total: usize,
    state: ListState,
    matcher: Matcher,
}

impl FilePopup {
    fn new(query: String) -> Self {
        Self::from_files(walk_current_directory(), query)
    }

    /// A popup over `files`, filtered by `query`.
    pub(crate) fn from_files(files: Vec<String>, query: String) -> Self {
        let mut popup = Self {
            files,
            query,
            matches: Vec::new(),
            total: 0,
            state: ListState::default(),
            matcher: Matcher::new(Config::DEFAULT.match_paths()),
        };
        popup.refilter();
        popup
    }

    pub(crate) fn select_prev(&mut self) {
        self.state.select_previous();
    }

    pub(crate) fn select_next(&mut self) {
        self.state.select_next();
    }

    /// Point the popup at a new query, refiltering when it changed.
    pub(crate) fn set_query(&mut self, query: String) {
        if self.query != query {
            self.query = query;
            self.refilter();
        }
    }

    /// The highlighted row, if any.
    pub(crate) fn selected(&self) -> Option<&str> {
        self.state
            .selected()
            .and_then(|i| self.matches.get(i).map(String::as_str))
    }

    /// Replace the word after the `@` with the selection, quoting paths w/ whitespace
    pub(crate) fn accept(&self, editor: &mut Editor) {
        let (Some((_, at)), Some(path)) = (derive_query(editor), self.selected()) else {
            return;
        };
        let text = if path.contains(char::is_whitespace) {
            format!("\"{path}\"")
        } else {
            path.to_string()
        };

        let line = &mut editor.lines[editor.line];
        // Replace the whole word after the `@`
        let start = at + 1;
        let end = line[start..]
            .find(char::is_whitespace)
            .map_or(line.len(), |i| start + i);
        line.replace_range(start..end, &text);
        editor.g = graphemes(&line[..=at]) + graphemes(&text);
    }

    /// Re-match the query against the file snapshot, reseeding the selection.
    fn refilter(&mut self) {
        let pattern = Pattern::parse(&self.query, CaseMatching::Smart, Normalization::Smart);
        let hits = pattern.match_list(self.files.iter(), &mut self.matcher);
        self.total = hits.len();
        self.matches = hits
            .into_iter()
            .take(MAX_SHOWN)
            .map(|(path, _)| path.clone())
            .collect();
        self.state.select((!self.matches.is_empty()).then_some(0));
    }
}

/// (Re)open the popup when the `@` word may have changed: caret moves and
/// deletions can shift or eat it, and a fresh `@` always re-arms.
pub(crate) fn rearm(key: &KeyEvent) -> bool {
    let modified = key
        .modifiers
        .intersects(KeyModifiers::CONTROL | KeyModifiers::ALT);
    match key.code {
        KeyCode::Left | KeyCode::Right | KeyCode::Home | KeyCode::End | KeyCode::Backspace => true,
        KeyCode::Char('@') => !modified,
        _ => false,
    }
}

/// Derive an `@query` from the draft and decide the typeahead's fate.
///
/// - A changed query refilters the list
/// - A vanished query closes any popup.
pub(crate) fn update(editor: &Editor, popup: &mut Option<Popup>, rearm: bool) {
    // Closed and not re-arming -> skip the update.
    if popup.is_none() && !rearm {
        return;
    }
    let Some((query, _)) = derive_query(editor) else {
        *popup = None;
        return;
    };
    match popup {
        Some(Popup::Files(p)) if p.query == query => {}
        Some(Popup::Files(p)) => {
            p.query = query;
            p.refilter();
        }
        Some(Popup::Sessions(_)) | None if rearm => {
            *popup = Some(Popup::Files(FilePopup::new(query)));
        }
        _ => {}
    }
}

/// Draw the popup anchored above `anchor` (the top rule), overlaying the transcript.
///
/// `label` names the list in the title and `hint` the keys in the bottom rule,
/// since the popup fronts both the `@file` typeahead and the session picker.
pub(crate) fn render(
    frame: &mut Frame,
    popup: &mut FilePopup,
    anchor: Rect,
    label: &str,
    hint: &str,
) {
    // Borders plus at least one row; never taller than the space above.
    let h = (popup.matches.len() as u16 + 2).min(anchor.y).max(3);
    let area = Rect {
        x: anchor.x + 1,
        y: anchor.y.saturating_sub(h),
        width: anchor.width.saturating_sub(2),
        height: h,
    };
    let items: Vec<ListItem> = if popup.matches.is_empty() {
        vec![ListItem::new(format!("no matches for `{}`", popup.query))]
    } else {
        popup
            .matches
            .iter()
            .map(|path| ListItem::new(path.as_str()))
            .collect()
    };
    // "+" if we have more files than fit, empty otherwise
    let more = if popup.total > popup.matches.len() { "+" } else { "" };
    let list = List::new(items)
        .block(
            Block::bordered()
                .title(format!(" {label} · {}{} ", popup.matches.len(), more))
                .title_bottom(Line::from(format!(" {hint} "))),
        )
        .highlight_style(HIGHLIGHT)
        .highlight_symbol("❯ ");
    frame.render_widget(Clear, area);
    frame.render_stateful_widget(list, area, &mut popup.state);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The query derives from a word-start `@` and matches subsequences;
    /// an empty query lists everything.
    #[test]
    fn queries_derive_and_match() {
        let mut editor = Editor::default();
        editor.insert_str("fix @ma");
        assert_eq!(derive_query(&editor), Some(("ma".into(), 4)));
        editor.clear();
        editor.insert_str("user@host");
        assert_eq!(derive_query(&editor), None);
        editor.clear();
        editor.insert_str("see @");
        assert_eq!(derive_query(&editor), Some((String::new(), 4)));

        let files: Vec<String> = ["src/main.rs", "docs/manual.md", "Cargo.toml"]
            .into_iter()
            .map(String::from)
            .collect();
        let mut popup = FilePopup::from_files(files.clone(), "srm".into());
        assert_eq!(popup.matches, ["src/main.rs"]);
        popup.query = String::new();
        popup.refilter();
        assert_eq!(popup.matches, files);
    }

    #[test]
    fn accept_keeps_the_at_and_rearm_reopens() {
        let mut editor = Editor::default();
        editor.insert_str("read @ma");
        let popup = FilePopup::from_files(vec!["src/main.rs".to_string()], "ma".into());
        popup.accept(&mut editor);
        assert_eq!(editor.lines, ["read @src/main.rs"]);
        assert_eq!(editor.g, 17); // parked after the inserted path

        // Closed after completion: plain typing keeps it closed, while a deletion,
        // a caret move, or a fresh `@` re-arm it.
        let mut slot = None;
        editor.insert_str(" more");
        update(&editor, &mut slot, rearm(&KeyEvent::from(KeyCode::Char('x'))));
        assert!(slot.is_none());
        editor.backspace();
        update(&editor, &mut slot, rearm(&KeyEvent::from(KeyCode::Backspace)));
        assert!(slot.is_some());
        slot = None;
        editor.left();
        update(&editor, &mut slot, rearm(&KeyEvent::from(KeyCode::Left)));
        assert!(slot.is_some());
        slot = None;
        editor.insert_str(" @ne");
        update(&editor, &mut slot, rearm(&KeyEvent::from(KeyCode::Char('@'))));
        assert!(slot.is_some());
    }
}
