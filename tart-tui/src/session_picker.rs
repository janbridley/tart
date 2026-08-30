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
    pub(crate) popup: FilePopup,
    /// Listed sessions paired with the label the popup's rows show.
    sessions: Vec<(PathBuf, String)>,
}

impl SessionPopup {
    /// Create a selection menut over `project`'s sessions in `root`, newest first.
    pub(crate) fn new(root: &Path, project: &Path, query: String) -> Self {
        let sessions = session::list(root, project)
            .unwrap_or_default()
            .into_iter()
            .map(|(path, opening)| {
                let label = label(&path, &opening);
                (path, label)
            })
            .collect::<Vec<_>>();
        let lines: Vec<String> = sessions.iter().map(|(_, line)| line.clone()).collect();
        Self {
            popup: FilePopup::from_files(lines, query),
            sessions,
        }
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

/// One picker row: the session's filename stamp, then its opening request
/// capped with an ellipsis; a session without messages says so.
fn label(path: &Path, opening: &str) -> String {
    let name = path
        .file_name()
        .map_or_else(String::new, |name| name.to_string_lossy().into_owned());
    let stamp = name.strip_suffix(".jsonl").unwrap_or(&name);
    let opening = if opening.is_empty() {
        "(no messages)".to_string()
    } else {
        capped(opening)
    };
    format!("{stamp}  {opening}")
}

/// `text` capped at 60 characters, plus an ellipsis when it runs past.
fn capped(text: &str) -> String {
    let mut capped = text.chars().take(60).collect::<String>();
    if text.chars().nth(60).is_some() {
        capped.push('…');
    }
    capped
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

    /// Labels pair the file's stamp with its opening request, capped.
    #[test]
    fn labels_pair_the_stamp_with_a_capped_opening_request() {
        let path = Path::new("/root/proj/20260102-000000.jsonl");
        assert_eq!(
            label(path, "fix the login flow"),
            "20260102-000000  fix the login flow"
        );
        // A long opening keeps 60 characters plus the ellipsis.
        assert_eq!(
            label(path, &"x".repeat(80)),
            format!("20260102-000000  {}…", "x".repeat(60))
        );
        // A session without messages says so, and a foreign name keeps its
        // whole stem.
        assert_eq!(label(path, ""), "20260102-000000  (no messages)");
        assert_eq!(label(Path::new("notes.txt"), "hi"), "notes.txt  hi");
    }

    #[test]
    fn the_chooser_lists_and_picks_this_projects_sessions() {
        let root = tempfile::tempdir().unwrap();
        let project = Path::new("/tmp/proj");
        let dir = root.path().join("tmp-proj");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("20260101-000000.jsonl"),
            "{\"type\":\"message\",\"role\":\"system\",\"content\":\"s\"}\n\
             {\"type\":\"message\",\"role\":\"user\",\"content\":\"fix the login flow\"}\n",
        )
        .unwrap();

        // Opened with a query, the match is already highlighted.
        let mut chooser = SessionPopup::new(root.path(), project, "login".to_string());
        assert_eq!(chooser.selected_path(), Some(dir.join("20260101-000000.jsonl")));

        // A project with no sessions opens empty; a query with no match picks nothing
        let empty = SessionPopup::new(root.path(), Path::new("/tmp/elsewhere"), String::new());
        assert_eq!(empty.selected_path(), None);
        chooser.popup.set_query("nomatch".to_string());
        assert_eq!(chooser.selected_path(), None);
    }
}
