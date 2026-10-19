//! Slash-command completion in the prompt box.

use itertools::Itertools;

use crate::pane::{Editor, graphemes};

/// Every command, sorted.
const COMMANDS: &[&str] = &[
    "/agent", "/agents", "/clear", "/effort", "/exit", "/gpu", "/model", "/perf", "/plan", "/quit",
    "/resume", "/rewind", "/stop",
];
/// Chat mode's subset: no plan mode, subagents, or sandbox.
const CHAT_COMMANDS: &[&str] = &[
    "/clear", "/effort", "/exit", "/model", "/perf", "/quit", "/resume", "/rewind",
];

/// The live `/`-command completion: the typed word and the commands it can
/// still become, shared by the prompt's ghost and its `Tab`.
pub(crate) struct TabComplete {
    /// The typed `/`-word: the draft's whole text, live only while the caret
    /// rides its end.
    word: String,
    /// Whether chat mode's smaller command table applies.
    chat: bool,
}

impl TabComplete {
    /// The completion live in `editor`, if a `/`-word is being typed.
    pub(crate) fn live(editor: &Editor, chat: bool) -> Option<Self> {
        let [line] = editor.lines.as_slice() else {
            return None;
        };
        (editor.g == graphemes(line))
            .then_some(line.as_str())
            .filter(|word| word.starts_with('/') && !word.contains(char::is_whitespace))
            .map(|word| Self { word: word.to_string(), chat })
    }

    /// The commands the typed word can still become.
    fn candidates(&self) -> impl Iterator<Item = &'static str> + '_ {
        let table = if self.chat { CHAT_COMMANDS } else { COMMANDS };
        table.iter().copied().filter(|c| c.starts_with(&self.word))
    }

    /// The dim ghost: each candidate's untyped remainder, comma-joined when
    /// the completion is ambiguous. `None` when nothing completes.
    pub(crate) fn ghost(&self) -> Option<String> {
        let remainders = self
            .candidates()
            .map(|c| &c[self.word.len()..])
            .filter(|rest| !rest.is_empty())
            .collect_vec();
        (!remainders.is_empty()).then(|| remainders.join(","))
    }

    /// `Tab` on the typed word: insert the unique candidate's remainder, or multiple.
    ///
    /// Returns whether the word was a real cmd, so an unknown one keeps `Tab` literal.
    pub(crate) fn complete(self, editor: &mut Editor) -> bool {
        let mut candidates = self.candidates();
        let Some(first) = candidates.next() else {
            return false;
        };
        let insert = match candidates.next() {
            // Unique: the whole remainder, readying the command's argument.
            None => format!("{} ", &first[self.word.len()..]),
            Some(_) => shared(self.candidates())[self.word.len()..].to_string(),
        };
        if !insert.trim().is_empty() {
            editor.insert_str(&insert);
        }
        true
    }
}

/// The commands the unknown-command hint prints, so it cannot drift from
/// the tables.
pub(crate) fn hint(chat: bool) -> String {
    let table = if chat { CHAT_COMMANDS } else { COMMANDS };
    table.join(" ")
}

/// The longest prefix every word shares, borrowed from the first.
fn shared<'a>(words: impl Iterator<Item = &'a str>) -> &'a str {
    words
        .reduce(|a, b| {
            // How many leading chars agree, as their byte offset in `a`.
            let agreed = a.chars().zip(b.chars()).take_while(|(x, y)| x == y).count();
            &a[..a.char_indices().nth(agreed).map_or(a.len(), |(i, _)| i)]
        })
        .unwrap_or("")
}

#[cfg(test)]
mod tests {
    #![allow(clippy::panic, reason = "test assertions")]

    use super::*;

    fn editor(text: &str) -> Editor {
        let mut editor = Editor::default();
        editor.insert_str(text);
        editor
    }

    #[test]
    fn ghost_shows_the_untyped_remainders() {
        let ghost = |text, chat| TabComplete::live(&editor(text), chat).and_then(|c| c.ghost());
        assert_eq!(ghost("/mod", false).as_deref(), Some("el"));
        assert_eq!(ghost("/re", false).as_deref(), Some("sume,wind"));
        assert_eq!(
            ghost("/", false).as_deref(),
            Some("agent,agents,clear,effort,exit,gpu,model,perf,plan,quit,resume,rewind,stop")
        );
        // A whole command, an argument, and an unknown word all end the ghost.
        assert_eq!(ghost("/resume", false), None);
        assert_eq!(ghost("/resume fix", false), None);
        assert_eq!(ghost("/nope", false), None);
        // A word that is itself a command still ghosts its longer kin.
        assert_eq!(ghost("/agent", false).as_deref(), Some("s"));
        // Chat mode has no subagents, plan mode, or sandbox.
        assert_eq!(ghost("/pl", true), None);
        assert_eq!(ghost("/cl", true).as_deref(), Some("ear"));
        assert_eq!(ghost("/ag", true), None);
        assert_eq!(ghost("/ex", true).as_deref(), Some("it"));
    }

    #[test]
    fn only_a_caret_at_the_words_end_is_live() {
        let mut caret_moved = editor("/re");
        caret_moved.left();
        let multi_line = editor("fix\n/re");
        assert!(TabComplete::live(&caret_moved, false).is_none());
        assert!(TabComplete::live(&multi_line, false).is_none());
        let mut editor = editor("/р");
        let handled = TabComplete::live(&editor, false).is_some_and(|c| c.complete(&mut editor));
        assert!(!handled, "the unknown word is not completed");
        assert_eq!(editor.text(), "/р");
    }

    /// The tables stay sorted, which the ghost's determinism rests on.
    #[test]
    fn the_tables_stay_sorted() {
        for table in [COMMANDS, CHAT_COMMANDS] {
            assert!(table.windows(2).all(|pair| pair[0] < pair[1]));
        }
        assert_eq!(hint(true), CHAT_COMMANDS.join(" "));
    }

    /// The shared prefix cuts on char boundaries, multibyte included, and an
    /// empty run of words shares nothing.
    #[test]
    fn the_shared_prefix_cuts_on_char_boundaries() {
        assert_eq!(shared(["日本語", "日本人", "日x"].into_iter()), "日");
        assert_eq!(shared(["resume"].into_iter()), "resume");
        assert_eq!(shared(std::iter::empty::<&str>()), "");
    }

    /// Tab completes the unique remainder with a space for the argument, the
    /// shared prefix when ambiguous, and leaves other drafts to the tab key.
    #[test]
    fn tab_completes_the_shared_prefix() {
        let tab = |text, chat| {
            let mut editor = editor(text);
            let handled = TabComplete::live(&editor, chat).is_some_and(|c| c.complete(&mut editor));
            (handled, editor.text())
        };
        // Unique: the remainder and a space, readying the argument.
        assert_eq!(tab("/mod", false), (true, "/model ".to_string()));
        // Ambiguous: only as far as they agree, then the ghost narrows.
        assert_eq!(tab("/a", false), (true, "/agent".to_string()));
        // A whole command already typed swallows the Tab without a space.
        assert_eq!(tab("/plan", false), (true, "/plan".to_string()));
        // No `/`-word live: Tab keeps its old meaning.
        assert_eq!(tab("hello", false), (false, "hello".to_string()));
        assert_eq!(tab("/nope", false), (false, "/nope".to_string()));
        assert_eq!(tab("/model ", false), (false, "/model ".to_string()));
        // Chat mode has no subagents, plan mode, or sandbox.
        assert_eq!(tab("/st", true), (false, "/st".to_string()));
        assert_eq!(tab("/cl", true), (true, "/clear ".to_string()));
    }
}
