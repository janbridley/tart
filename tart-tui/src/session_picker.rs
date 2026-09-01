//! Session picker window, triggered via `/resume`.

use std::path::{Path, PathBuf};

use crate::file_mentions::{Picker, command_query};
use crate::pane::Editor;
use tart_agents::session;

/// `/resume` typeahead over one project's sessions: the rows it lists and the
/// query prefix that opens it. One picker row: the session's filename stamp,
/// then its opening request capped with an ellipsis; a session without
/// messages says so.
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

/// Open the session chooser over `project`'s sessions in `root`, filtered by
/// `query`: the rows paired with the path each selects, or `None` when there
/// are no sessions to show.
pub(crate) fn session_picker(
    root: &Path,
    project: &Path,
    query: String,
) -> Option<Picker<PathBuf>> {
    let sessions = session::list(root, project)
        .unwrap_or_default()
        .into_iter()
        .map(|(path, opening)| (path.clone(), label(&path, &opening)))
        .collect::<Vec<_>>();
    if sessions.is_empty() {
        return None;
    }
    Some(Picker::from_picks(sessions, query))
}

/// Everything preceding a `/resume` command, used to filter results.
///
/// `/resume` alone opens the chooser unfiltered and `/resume fix` filters results
pub(crate) fn derive_query(editor: &Editor) -> Option<String> {
    command_query(editor, "/resume")
}
