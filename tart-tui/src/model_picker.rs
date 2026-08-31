//! Model picker window, triggered via `/model`.

use ratatui::Frame;
use ratatui::layout::Rect;

use crate::config::AgentChoice;
use crate::file_mentions::FilePopup;
use crate::pane::Editor;

/// `/model` typeahead over the agents the agents file defines.
pub(crate) struct ModelPopup {
    /// Fuzzy matcher and file popup over the defined agents.
    pub(crate) popup: FilePopup,
    /// Listed choices paired with the labels the popup's rows show.
    choices: Vec<AgentChoice>,
}

impl ModelPopup {
    /// Create a chooser over `choices`, filtered by `query`.
    pub(crate) fn new(choices: &[AgentChoice], query: String) -> Self {
        let rows = choices.iter().map(ToString::to_string).collect();
        Self {
            popup: FilePopup::from_files(rows, query),
            choices: choices.to_vec(),
        }
    }

    /// The choice behind the highlighted row, if any.
    pub(crate) fn selected_choice(&self) -> Option<&AgentChoice> {
        let row = self.popup.selected()?;
        self.choices.iter().find(|choice| choice.to_string() == row)
    }

    /// Draw the chooser, anchored above `anchor` like the `@file` popup.
    pub(crate) fn render(&mut self, frame: &mut Frame, anchor: Rect) {
        self.popup.render(
            frame,
            anchor,
            "models",
            "↑↓ select · Enter to switch · Esc to close popup",
        );
    }
}

/// Everything preceding a `/model` line, used to filter results.
///
/// `/model` alone opens the chooser unfiltered and `/model glm` filters results.
pub(crate) fn derive_query(editor: &Editor) -> Option<String> {
    let line = &editor.lines[editor.line];
    let rest = line.strip_prefix("/model")?;
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

    fn choices() -> Vec<AgentChoice> {
        [
            ("zai", "Coding Specialist", "glm-5.3"),
            ("openrouter", "Ling Flash", "inclusionai/ling-3.0-flash"),
        ]
        .into_iter()
        .map(|(provider, name, model)| AgentChoice {
            provider: provider.to_string(),
            name: name.to_string(),
            model: model.to_string(),
        })
        .collect()
    }

    #[test]
    fn the_query_derives_from_a_leading_model_word() {
        assert_eq!(derive_query(&editor("/model")), Some(String::new()));
        assert_eq!(derive_query(&editor("/model glm")), Some("glm".into()));
        assert_eq!(derive_query(&editor("/model  spaced")), Some("spaced".into()));
        // A longer word, or a `/model` mid-line, is not the chooser.
        assert_eq!(derive_query(&editor("/models")), None);
        assert_eq!(derive_query(&editor("fix /model")), None);
        assert_eq!(derive_query(&editor("hello")), None);
    }

    /// Rows round-trip: the highlighted label picks its choice back out, and a
    /// query with no match picks nothing.
    #[test]
    fn the_chooser_lists_and_picks_the_defined_agents() {
        let mut chooser = ModelPopup::new(&choices(), "laguna".to_string());
        assert_eq!(chooser.selected_choice(), None, "no agent matches");

        chooser.popup.set_query(String::new());
        assert_eq!(
            chooser.selected_choice().map(ToString::to_string),
            Some("zai · Coding Specialist · glm-5.3".to_string())
        );

        // The openrouter model id is searchable through the row's model part.
        chooser.popup.set_query("ling".to_string());
        assert_eq!(
            chooser
                .selected_choice()
                .map(|choice| (choice.provider.as_str(), choice.name.as_str())),
            Some(("openrouter", "Ling Flash"))
        );
    }
}
