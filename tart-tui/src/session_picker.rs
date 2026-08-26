//! Session picker window, triggered via `/resume`.

use std::path::{Path, PathBuf};

use ratatui::Frame;
use ratatui::layout::Rect;

use crate::file_mentions::FilePopup;
use crate::pane::Editor;
use tart_agents::session;

/// `/resume` typeahead over one project's sessions.
pub(crate) struct SessionPopup {
    /// Fuzzy matcher and file popup over the project's sessions.
    popup: FilePopup,
    /// Listed sessions, their lines the popup's rows.
    sessions: Vec<(PathBuf, String)>,
}

impl SessionPopup {
    /// Create a selection menut over `project`'s sessions in `root`, newest first.
    pub(crate) fn new(root: &Path, project: &Path, query: String) -> Self {
        let sessions = session::list(root, project).unwrap_or_default();
        let lines: Vec<String> = sessions.iter().map(|(_, line)| line.clone()).collect();
        Self {
            popup: FilePopup::from_files(lines, query),
            sessions,
        }
    }

    /// Point the chooser at a new query, refiltering when it changed.
    pub(crate) fn set_query(&mut self, query: String) {
        self.popup.set_query(query);
    }

    pub(crate) fn select_prev(&mut self) {
        self.popup.select_prev();
    }

    pub(crate) fn select_next(&mut self) {
        self.popup.select_next();
    }

    /// The file behind the highlighted row, if any.
    pub(crate) fn selected_path(&self) -> Option<PathBuf> {
        let line = self.popup.selected()?;
        self.sessions
            .iter()
            .find_map(|(path, label)| (label == line).then(|| path.clone()))
    }

    /// Draw the chooser, anchored above `anchor` like the `@file` popup.
    pub(crate) fn render(&mut self, frame: &mut Frame, anchor: Rect) {
        self.popup.render(
            frame,
            anchor,
            "sessions",
            "↑↓ select · Enter to resume · Esc to close popup",
        );
    }
}

/// Everything preceeding a `/resume` line, used to filter results.
///
/// `/resume` alone opens the chooser unfiltered and `/resume fix` filters results
pub(crate) fn derive_query(editor: &Editor) -> Option<String> {
    let line = &editor.lines[editor.line];
    let rest = line.strip_prefix("/resume")?;
    (rest.is_empty() || rest.starts_with(' ')).then(|| rest.trim_start().to_string())
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, reason = "test assertions")]

    use super::*;

    /// A scratch directory removed when the guard drops, so parallel tests dont collide
    struct Scratch(PathBuf);

    impl Scratch {
        /// Create the guard under the temp directory.
        fn new(tag: &str) -> Self {
            let path =
                std::env::temp_dir().join(format!("tart-picker-{tag}-{}", std::process::id()));
            std::fs::create_dir_all(&path).unwrap();
            Self(path)
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn editor(text: &str) -> Editor {
        let mut editor = Editor::default();
        editor.insert_str(text);
        editor
    }

    #[test]
    fn the_query_derives_from_a_leading_resume_word() {
        assert_eq!(derive_query(&editor("/resume")), Some(String::new()));
        assert_eq!(derive_query(&editor("/resume fix")), Some("fix".into()));
        assert_eq!(derive_query(&editor("/resume  spaced")), Some("spaced".into()));
        // A longer word, or a `/resume` mid-line, is not the chooser.
        assert_eq!(derive_query(&editor("/resumefoo")), None);
        assert_eq!(derive_query(&editor("fix /resume")), None);
        assert_eq!(derive_query(&editor("hello")), None);
    }

    #[test]
    fn the_chooser_lists_and_picks_this_projects_sessions() {
        let root = Scratch::new("list");
        let project = Path::new("/tmp/proj");
        let dir = root.0.join("tmp-proj");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("20260101-000000.jsonl"),
            "{\"type\":\"message\",\"role\":\"system\",\"content\":\"s\"}\n\
             {\"type\":\"message\",\"role\":\"user\",\"content\":\"fix the login flow\"}\n",
        )
        .unwrap();

        // Opened with a query, the match is already highlighted.
        let mut chooser = SessionPopup::new(&root.0, project, "login".to_string());
        assert_eq!(chooser.selected_path(), Some(dir.join("20260101-000000.jsonl")));

        // A project with no sessions opens empty; a query with no match picks nothing
        let empty = SessionPopup::new(&root.0, Path::new("/tmp/elsewhere"), String::new());
        assert_eq!(empty.selected_path(), None);
        chooser.set_query("nomatch".to_string());
        assert_eq!(chooser.selected_path(), None);
    }
}
