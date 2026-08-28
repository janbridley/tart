//! Attachments expand `@path` mentions the sandbox denies into useful context

use std::io::ErrorKind;
use std::path::{Component, Path, PathBuf};

use tart_agents::{CONTENT_CAP, head_cap};

use crate::file_mentions;

/// The `@path` tokens in a submitted line, as the typeahead would complete them
fn mentions(line: &str) -> Vec<String> {
    let mut found: Vec<String> = Vec::new();
    for mention in line.split_whitespace().filter_map(|word| word.strip_prefix('@')) {
        if !mention.is_empty() && !found.iter().any(|seen| seen == mention) {
            found.push(mention.to_string());
        }
    }
    found
}

/// Resolve `.` and `..` components lexically, without touching the filesystem.
fn lexical(path: &Path) -> PathBuf {
    let mut resolved = PathBuf::new();
    for part in path.components() {
        match part {
            Component::CurDir => {}
            Component::ParentDir => {
                resolved.pop();
            }
            other => resolved.push(other.as_os_str()),
        }
    }
    resolved
}

/// Whether a mentioned path stays inside the sandbox's read grants.
fn inside_grant(mention: &str, cwd: &Path) -> bool {
    let expanded = file_mentions::expand_tilde(mention).unwrap_or_else(|| PathBuf::from(mention));
    let resolved = lexical(&cwd.join(expanded));
    resolved.starts_with(cwd)
        || resolved.starts_with("/tmp")
        || resolved.starts_with(std::env::temp_dir())
}

/// A fence longer than any run of backticks in `text`, so a file can't close its block
fn fence_len(text: &str) -> usize {
    let (mut longest, mut run) = (0, 0);
    for c in text.chars() {
        run = if c == '`' { run + 1 } else { 0 };
        longest = longest.max(run);
    }
    longest.max(3) + 1
}

/// One outside-the-sandbox mention's attachment block and its note for the pane.
fn attach(mention: &str) -> (String, String) {
    let path = file_mentions::expand_tilde(mention).unwrap_or_else(|| PathBuf::from(mention));
    let why = |error: &std::io::Error| {
        match error.kind() {
            ErrorKind::NotFound => "not found",
            ErrorKind::IsADirectory => "a directory",
            ErrorKind::InvalidData => "not UTF-8 text",
            _ => "unreadable",
        }
        .to_string()
    };
    match std::fs::read_to_string(&path) {
        Ok(contents) => {
            let kept = head_cap(&contents, CONTENT_CAP);
            let note = if kept.len() < contents.len() {
                format!("attached {mention} (truncated to {} KiB)", CONTENT_CAP / 1024)
            } else {
                format!("attached {mention} ({} bytes)", contents.len())
            };
            let fence = "`".repeat(fence_len(&kept));
            (format!("{fence}\n{kept}\n{fence}"), note)
        }
        Err(error) => {
            let reason = why(&error);
            (reason.clone(), format!("{mention}: {reason}, not attached"))
        }
    }
}

/// Expand a submitted line's `@path` mentions outside the sandbox into attached content
pub(crate) fn attach_mentions(line: &str, cwd: &Path) -> (String, Vec<String>) {
    let mut blocks = Vec::new();
    let mut notes = Vec::new();
    for mention in mentions(line) {
        if !inside_grant(&mention, cwd) {
            let (block, note) = attach(&mention);
            blocks.push(format!("`{mention}`:\n{block}"));
            notes.push(note);
        }
    }
    if blocks.is_empty() {
        return (line.to_string(), notes);
    }
    let message = format!(
        "{line}\n\nAttached from outside the sandbox: your tools cannot read or \
         edit these, so work from the contents below.\n\n{}\n",
        blocks.join("\n\n")
    );
    (message, notes)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, reason = "test assertions")]

    use super::*;

    /// Mentions scan as the typeahead completes them, once each.
    #[test]
    fn mentions_scan_word_start_tokens() {
        assert_eq!(
            mentions("fix @src/pane.rs and @~/notes.md"),
            ["src/pane.rs", "~/notes.md"]
        );
        assert_eq!(mentions("mail user@host about @/etc/hosts"), ["/etc/hosts"]);
        assert_eq!(mentions("no mentions here"), Vec::<String>::new());
        assert_eq!(mentions("dup @Cargo.toml @Cargo.toml"), ["Cargo.toml"]);
        assert_eq!(mentions("a bare @ does nothing"), Vec::<String>::new());
    }

    /// Only paths staying inside the sandbox's grants stay plain references.
    #[test]
    fn only_paths_inside_the_grant_stay_references() {
        let cwd = std::env::current_dir().unwrap();
        assert!(inside_grant("src/pane.rs", &cwd));
        assert!(inside_grant("./src/x", &cwd));
        assert!(inside_grant("src/../src/x", &cwd));
        assert!(inside_grant(&cwd.display().to_string(), &cwd));
        assert!(inside_grant("/tmp/build.log", &cwd));
        assert!(!inside_grant("../tart-agents/Cargo.toml", &cwd));
        assert!(!inside_grant("src/../../elsewhere/x", &cwd));
        assert!(!inside_grant("~/notes.md", &cwd));
        assert!(!inside_grant("/Users/elsewhere/x", &cwd));
    }

    /// A submitted line keeps its inside-the-sandbox mentions as typed, and
    /// expands outside ones into attached contents the model can work from.
    #[test]
    fn outside_mentions_attach_verbatim() {
        let cwd = std::env::current_dir().unwrap();

        // Inside the grant: the model reads it with its own tools.
        let (message, notes) = attach_mentions("fix @src/pane.rs please", &cwd);
        assert_eq!(message, "fix @src/pane.rs please");
        assert!(notes.is_empty());

        // Outside: the contents ride along in a fence that outlasts any run of
        // backticks in the file, with a note for the pane.
        let (message, notes) = attach_mentions("see @../tart-agents/Cargo.toml", &cwd);
        assert!(
            message.starts_with("see @../tart-agents/Cargo.toml\n\nAttached from outside"),
            "{message}"
        );
        // A file with no backticks gets the four-backtick fence.
        assert!(
            message.contains("`../tart-agents/Cargo.toml`:\n````\n[package]"),
            "{message}"
        );
        assert_eq!(notes.len(), 1);
        assert!(
            notes[0].starts_with("attached ../tart-agents/Cargo.toml ("),
            "{notes:?}"
        );

        // A mention that will not read attaches its reason instead.
        let (message, notes) = attach_mentions("look at @/no/such/path", &cwd);
        assert!(message.contains("`/no/such/path`:\nnot found"), "{message}");
        assert_eq!(notes, ["/no/such/path: not found, not attached".to_string()]);
    }

    /// The fence outlasts any run of backticks an attached file carries.
    #[test]
    fn fences_outlast_their_contents() {
        assert_eq!(fence_len("plain text"), 4);
        assert_eq!(fence_len("a ``` fence\n```"), 4);
        assert_eq!(fence_len("```` nested"), 5);
    }
}
