//! Rewind-point picker, triggered via `/rewind`.

use tart_agents::Transcript as Conversation;

use crate::file_mentions::Picker;
use crate::pane::ellipsize;
use crate::recorded::synthetic;

/// One turn's row: its number, counted from the session's first, beside its
/// opening line capped with an ellipsis.
///
/// The numbers keep rows unique where two messages open alike, which is how
/// [`Picker::selected`] recovers the pick behind a row; they join the fuzzy
/// query too, so digits alone find their turn.
fn label(number: usize, text: &str) -> String {
    let opening = text.split_once('\n').map_or(text, |(line, _)| line);
    format!("{number}  {}", ellipsize(opening, 60))
}

/// Open the rewind chooser over `conversation`'s user turns, filtered by `query`.
///
/// Newest messages are at the top, then the start of the session, with each pick
/// holding a numeric tag for its turn index.
pub(crate) fn rewind_picker(
    conversation: &Conversation,
    query: String,
) -> Option<Picker<(usize, String)>> {
    let turns = conversation.user_turns();
    // The record's first user message sits just past the system prompt, so
    // rewinding there empties the conversation: a `/clear` into a fresh file.
    let start_row = (turns.first()?.0, String::new());
    let mut picks: Vec<_> = turns
        .iter()
        .enumerate()
        .filter(|(_, (_, text))| !synthetic(text))
        .map(|(number, (start, text))| ((*start, text.clone()), label(number + 1, text)))
        .collect();
    // Newest first: the closest rewind point leads the unfiltered list.
    picks.reverse();
    picks.push((start_row, "(start of session)".to_string()));
    Some(Picker::from_picks(picks, query))
}
