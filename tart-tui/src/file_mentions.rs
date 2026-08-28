//! The file-completion popup.
//!
//! Two front ends share one typeahead over the working directory (gitignored
//! and hidden entries skipped): typing a word-start `@` in the prompt mentions
//! a file, and an argument in a `!` shell command completes as a file.

use ignore::WalkBuilder;
use nucleo_matcher::pattern::{CaseMatching, Normalization, Pattern};
use nucleo_matcher::{Config, Matcher};
use ratatui::Frame;
use ratatui::crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::layout::Rect;
use ratatui::style::{Modifier, Style};
use ratatui::text::Line;
use ratatui::widgets::{Block, Clear, List, ListItem, ListState};
use std::path::{Path, PathBuf};

use crate::pane::{Editor, Popup, g_to_byte, graphemes};

/// Bound the matcher's output; the list only renders its visible window.
const MAX_SHOWN: usize = 256;
const HIGHLIGHT: Style = Style::new().add_modifier(Modifier::REVERSED);

/// The suffix and byte offset of the last word-start `@` on the caret's line,
/// while the caret is still inside that `@`-word.
pub(crate) fn derive_query(editor: &Editor) -> Option<(String, usize)> {
    let line = &editor.lines[editor.line];
    let prefix = &line[..g_to_byte(line, editor.g)];
    let at = prefix.rfind('@')?;
    let word_start = prefix[..at].chars().next_back().is_none_or(char::is_whitespace);
    // Whitespace between the `@` and the caret (usually) means the caret left the word.
    let inside = !prefix[at + 1..].contains(char::is_whitespace);
    (word_start && inside).then(|| (prefix[at + 1..].to_string(), at))
}

/// The argument under the caret, as a file to complete.
pub(crate) fn derive_argument(editor: &Editor) -> Option<(String, usize)> {
    let line = &editor.lines[editor.line];
    let prefix = &line[..g_to_byte(line, editor.g)];
    let start = prefix.rfind(char::is_whitespace)? + 1;
    (!prefix[start..].is_empty()).then(|| (prefix[start..].to_string(), start))
}

/// The directory a completion token names, as typed
/// `Some("")` is the root and `Some("~")` the home directory. `None` for bare words
fn dir_named(token: &str) -> Option<String> {
    token.rsplit_once('/').map(|(dir, _)| dir.to_string())
}

/// Expand directories to readable paths.
fn readable_dir(dirpart: &str) -> Option<PathBuf> {
    if dirpart.starts_with('~') {
        return expand_tilde(dirpart);
    }
    Some(match dirpart {
        "" => PathBuf::from("/"),
        dir => PathBuf::from(dir),
    })
}

/// A path with a leading `~` expanded to the home directory, as typed otherwise
pub(crate) fn expand_tilde(path: &str) -> Option<PathBuf> {
    let rest = path.strip_prefix('~')?;
    if !rest.is_empty() && !rest.starts_with('/') {
        return None;
    }
    let home = PathBuf::from(std::env::var_os("HOME")?);
    Some(home.join(rest.trim_start_matches('/')))
}

/// One directory's entries as completion candidates, each prefixed by `dirpart`
fn list_directory(dirpart: &str, dir: &Path, base: &str) -> Vec<String> {
    let mut entries: Vec<String> = WalkBuilder::new(dir)
        .max_depth(Some(1))
        .hidden(false)
        .build()
        .filter_map(std::result::Result::ok)
        .filter(|entry| entry.depth() == 1)
        .filter_map(|entry| {
            let name = entry.file_name().to_string_lossy().into_owned();
            (name.starts_with('.') == base.starts_with('.')).then(|| {
                if entry.path().is_dir() {
                    format!("{dirpart}/{name}/")
                } else {
                    format!("{dirpart}/{name}")
                }
            })
        })
        .collect();
    entries.sort();
    entries
}

/// The candidates for completing shell path `token`, and the directory it named
fn path_source(token: &str) -> (Option<String>, Vec<String>) {
    let Some((dir, base)) = token.rsplit_once('/') else {
        return (None, walk_current_directory());
    };
    let dirpart = dir.to_string();
    let files =
        readable_dir(&dirpart).map_or_else(Vec::new, |dir| list_directory(&dirpart, &dir, base));
    (Some(dirpart), files)
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
    /// The candidates, as they were sourced.
    files: Vec<String>,
    query: String,
    /// This popup completes shell paths, re-sourcing when the query's directory moves
    paths: bool,
    /// The directory the candidates came from, as the query typed it.
    dir: Option<String>,
    /// Matched paths, sorted by relevance and capped at `MAX_SHOWN`.
    matches: Vec<String>,
    /// Total hits before the cap (for the title's `+`).
    total: usize,
    state: ListState,
    matcher: Matcher,
}

impl FilePopup {
    /// A popup completing shell path `token`, whose named directory is the
    /// candidate source: `../`, `~/…`, and absolute paths all complete.
    fn complete(token: &str) -> Self {
        let mut popup = Self::from_files(Vec::new(), token.to_string());
        popup.paths = true;
        popup.refile();
        popup.refilter();
        popup
    }

    /// A popup over `files`, filtered by `query`.
    pub(crate) fn from_files(files: Vec<String>, query: String) -> Self {
        let mut popup = Self {
            files,
            query,
            paths: false,
            dir: None,
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

    /// Point the popup at a new query, refiltering when it changed. A path
    /// completion whose query names a new directory re-sources first, so a
    /// typed `/` lists the directory it just named.
    pub(crate) fn set_query(&mut self, query: String) {
        if self.query != query {
            self.query = query;
            if self.paths && self.dir != dir_named(&self.query) {
                self.refile();
            }
            self.refilter();
        }
    }

    /// Re-source the candidates for the current query, whose named directory
    /// changed.
    fn refile(&mut self) {
        let (dir, files) = path_source(&self.query);
        self.dir = dir;
        self.files = files;
    }

    /// The highlighted row, if any.
    pub(crate) fn selected(&self) -> Option<&str> {
        self.state
            .selected()
            .and_then(|i| self.matches.get(i).map(String::as_str))
    }

    /// Replace the `@` word with the selection, quoting paths with whitespace.
    /// Returns whether it named a directory, so completing can keep going.
    pub(crate) fn accept(&self, editor: &mut Editor) -> bool {
        self.insert(editor, derive_query(editor).map(|(query, at)| (query, at + 1)))
    }

    /// Replace the argument under the caret with the selection, the bang-mode
    /// twin of [`FilePopup::accept`].
    pub(crate) fn accept_argument(&self, editor: &mut Editor) -> bool {
        self.insert(editor, derive_argument(editor))
    }

    /// Overwrite the word starting at byte `start` with the chosen path,
    /// quoting paths with whitespace and parking the caret after it.
    fn insert(&self, editor: &mut Editor, word: Option<(String, usize)>) -> bool {
        let (Some((_, start)), Some(path)) = (word, self.selected()) else {
            return false;
        };
        let text = if path.contains(char::is_whitespace) {
            format!("\"{path}\"")
        } else {
            path.to_string()
        };

        let line = &mut editor.lines[editor.line];
        // Replace the whole word, to its end.
        let end = line[start..]
            .find(char::is_whitespace)
            .map_or(line.len(), |i| start + i);
        line.replace_range(start..end, &text);
        editor.g = graphemes(&line[..start]) + graphemes(&text);
        // Directories complete with a trailing slash, for continuing into them.
        path.ends_with('/')
    }

    /// Draw the popup anchored above `anchor` (the top rule), overlaying the transcript.
    ///
    /// `label` names the list in the title and `hint` the keys in the bottom rule,
    /// since the popup fronts both the `@file` typeahead and the session picker.
    pub(crate) fn render(&mut self, frame: &mut Frame, anchor: Rect, label: &str, hint: &str) {
        // Borders plus at least one row; never taller than the space above.
        let h = (self.matches.len() as u16 + 2).min(anchor.y).max(3);
        let area = Rect {
            x: anchor.x + 1,
            y: anchor.y.saturating_sub(h),
            width: anchor.width.saturating_sub(2),
            height: h,
        };
        let items: Vec<ListItem> = if self.matches.is_empty() {
            let text = if self.files.is_empty() {
                // An empty source list just shows a nice message.
                format!("no {label}")
            } else {
                format!("no matches for `{}`", self.query)
            };
            vec![ListItem::new(text)]
        } else {
            self.matches
                .iter()
                .map(|path| ListItem::new(path.as_str()))
                .collect()
        };
        // "+" if we have more files than fit, empty otherwise
        let more = if self.total > self.matches.len() { "+" } else { "" };
        let list = List::new(items)
            .block(
                Block::bordered()
                    .title(format!(" {label} · {}{} ", self.matches.len(), more))
                    .title_bottom(Line::from(format!(" {hint} "))),
            )
            .highlight_style(HIGHLIGHT)
            .highlight_symbol("❯ ");
        frame.render_widget(Clear, area);
        frame.render_stateful_widget(list, area, &mut self.state);
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

/// Open the path completion for `token`, replacing any open popup. The
/// command's own word and an ended word open nothing.
pub(crate) fn open_path(popup: &mut Option<Popup>, token: Option<(String, usize)>) {
    if let Some((token, _)) = token {
        *popup = Some(Popup::Files(FilePopup::complete(&token)));
    }
}

/// Derive a query from the draft and decide the typeahead's fate.
///
/// - A changed query refilters the list
/// - A vanished query closes any popup.
pub(crate) fn update(popup: &mut Option<Popup>, query: Option<(String, usize)>, rearm: bool) {
    // Closed and not re-arming -> skip the update.
    if popup.is_none() && !rearm {
        return;
    }
    let Some((query, _)) = query else {
        *popup = None;
        return;
    };
    match popup {
        // An unchanged query refilters nothing: set_query no-ops on it.
        Some(Popup::Files(p)) => p.set_query(query),
        Some(Popup::Sessions(_)) | None if rearm => {
            *popup = Some(Popup::Files(FilePopup::complete(&query)));
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, reason = "test assertions")]

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
        editor.clear();
        editor.insert_str("read @Cargo.toml please");
        assert_eq!(derive_query(&editor), None); // caret past the mention word

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

    /// The directory a token names, and where it reads from.
    #[test]
    fn directories_derive_from_tokens() {
        assert_eq!(dir_named("Cargo"), None, "a bare word names none");
        assert_eq!(dir_named("src/ma"), Some("src".to_string()));
        assert_eq!(dir_named("../"), Some("..".to_string()));
        assert_eq!(dir_named("/Users/jenna"), Some("/Users".to_string()));
        assert_eq!(dir_named("/Users"), Some(String::new()), "the root names none");
        assert_eq!(dir_named("~/Downloads"), Some("~".to_string()));

        assert_eq!(readable_dir("src"), Some(PathBuf::from("src")));
        assert_eq!(readable_dir(""), Some(PathBuf::from("/")));
        let home = PathBuf::from(std::env::var_os("HOME").unwrap());
        assert_eq!(readable_dir("~"), Some(home.clone()));
        assert_eq!(readable_dir("~/Downloads"), Some(home.join("Downloads")));
    }

    /// A token naming a directory completes from it: entries keep the prefix
    /// as typed, directories trail a slash, and dotfiles appear only for a
    /// prefix that is one itself.
    #[test]
    fn tokens_naming_a_directory_list_it() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("main.rs"), "").unwrap();
        std::fs::write(tmp.path().join(".hidden"), "").unwrap();
        std::fs::create_dir(tmp.path().join("sub")).unwrap();
        let tmp = tmp.path().display().to_string();

        let popup = FilePopup::complete(&format!("{tmp}/ma"));
        assert_eq!(popup.matches, [format!("{tmp}/main.rs")]);

        let dot = FilePopup::complete(&format!("{tmp}/."));
        assert_eq!(dot.matches, [format!("{tmp}/.hidden")], "{:?}", dot.matches);

        let all = FilePopup::complete(&format!("{tmp}/"));
        assert!(
            all.matches.contains(&format!("{tmp}/main.rs")),
            "{:?}",
            all.matches
        );
        assert!(all.matches.contains(&format!("{tmp}/sub/")), "{:?}", all.matches);
        assert!(
            !all.matches.iter().any(|m| m.ends_with(".hidden")),
            "{:?}",
            all.matches
        );
    }

    /// A typed `/` names a new directory, and an open list re-sources, not refilters
    #[test]
    fn a_slash_in_the_query_resources_the_list() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::create_dir(tmp.path().join("sub")).unwrap();
        std::fs::write(tmp.path().join("sub").join("inner.rs"), "").unwrap();
        let tmp = tmp.path().display().to_string();

        let mut popup = FilePopup::complete(&format!("{tmp}/s"));
        assert_eq!(popup.matches, [format!("{tmp}/sub/")]);

        popup.set_query(format!("{tmp}/sub/"));
        assert_eq!(
            popup.matches,
            [format!("{tmp}/sub/inner.rs")],
            "{:?}",
            popup.matches
        );
    }

    /// Arguments derive from any word after whitespace, up to the caret.
    #[test]
    fn arguments_derive_after_whitespace() {
        let mut editor = Editor::default();
        editor.insert_str("cat src/main.rs");
        assert_eq!(derive_argument(&editor), Some(("src/main.rs".into(), 4)));
        // The caret ends the query, so completing mid-word sees its prefix.
        for _ in 0..3 {
            editor.left();
        }
        assert_eq!(derive_argument(&editor), Some(("src/main".into(), 4)));

        editor.clear();
        editor.insert_str("cargo");
        assert_eq!(
            derive_argument(&editor),
            None,
            "the command word is not an argument"
        );

        editor.clear();
        editor.insert_str("cat ");
        assert_eq!(derive_argument(&editor), None, "an ended word is no word");

        editor.clear();
        editor.insert_str("one  two");
        assert_eq!(derive_argument(&editor), Some(("two".into(), 5)));
    }

    /// A completion overwrites the whole argument word, tail and all, quoting
    /// paths with whitespace.
    #[test]
    fn accept_argument_replaces_the_word() {
        let mut editor = Editor::default();
        editor.insert_str("cat src/ma");
        let popup = FilePopup::from_files(vec!["src/main.rs".to_string()], "ma".into());
        popup.accept_argument(&mut editor);
        assert_eq!(editor.lines, ["cat src/main.rs"]);
        assert_eq!(editor.g, 15); // parked after the inserted path

        // Mid-word: the word's tail beyond the caret goes with it.
        let mut editor = Editor::default();
        editor.insert_str("echo one two three");
        for _ in 0..6 {
            editor.left();
        }
        let popup = FilePopup::from_files(vec!["two words.txt".to_string()], "tw".into());
        popup.accept_argument(&mut editor);
        assert_eq!(editor.lines, ["echo one \"two words.txt\" three"]);
        assert_eq!(editor.g, 24);
    }

    #[test]
    fn accept_keeps_the_at_and_rearm_reopens() {
        let mut editor = Editor::default();
        editor.insert_str("read @ma");
        let popup = FilePopup::from_files(vec!["src/main.rs".to_string()], "ma".into());
        popup.accept(&mut editor);
        assert_eq!(editor.lines, ["read @src/main.rs"]);
        assert_eq!(editor.g, 17); // parked after the inserted path

        // Closed after completion: plain typing keeps it closed, and re-arming
        // keys stay closed too while the caret is outside the mention word.
        let mut slot = None;
        editor.insert_str(" more");
        update(
            &mut slot,
            derive_query(&editor),
            rearm(&KeyEvent::from(KeyCode::Char('x'))),
        );
        assert!(slot.is_none());
        editor.backspace();
        update(
            &mut slot,
            derive_query(&editor),
            rearm(&KeyEvent::from(KeyCode::Backspace)),
        );
        assert!(slot.is_none());
        slot = None;
        editor.left();
        update(
            &mut slot,
            derive_query(&editor),
            rearm(&KeyEvent::from(KeyCode::Left)),
        );
        assert!(slot.is_none());
        slot = None;
        editor.insert_str(" @ne");
        update(
            &mut slot,
            derive_query(&editor),
            rearm(&KeyEvent::from(KeyCode::Char('@'))),
        );
        assert!(slot.is_some());
    }
}
