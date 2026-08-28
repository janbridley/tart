//! The plan-mode reminder the front end frames around user turns.
//!
//! [`PLAN_REMINDER`] rides along on every user turn while plan mode is on. It is
//! wrapped in tags and stripped again for display, so a session record replays as
//! the user wrote it.

/// The reminder prefixed to every user turn while plan mode is on.
pub const PLAN_REMINDER: &str = include_str!("data/PLAN.md");

/// The user turn that follows an approved plan, sent to start implementing it.
pub const PLAN_APPROVAL: &str = "The plan above is approved: implement it now. Edit files \
     as needed, follow the numbered steps in order, and say what changed when you finish.";

/// The tags wrapping a reminder inside a user turn.
const REMINDER_OPEN: &str = "<system-reminder>";
const REMINDER_CLOSE: &str = "</system-reminder>";

/// `text` with the plan-mode reminder in front of it, as it is sent to the model.
#[inline]
#[must_use]
pub fn with_reminder(text: &str) -> String {
    format!("{REMINDER_OPEN}\n{PLAN_REMINDER}\n{REMINDER_CLOSE}\n\n{text}")
}

/// The leading reminder block in `text`, if this turn carried one.
#[inline]
#[must_use]
pub fn reminder_span(text: &str) -> Option<&str> {
    let body = text.strip_prefix(REMINDER_OPEN)?;
    let end = REMINDER_OPEN.len() + body.find(REMINDER_CLOSE)? + REMINDER_CLOSE.len();
    Some(&text[..end])
}

/// The user's own words in `text`: a leading reminder block is dropped.
///
/// The inverse of [`with_reminder`] for display; anything the caller framed differently
/// comes back untouched.
#[inline]
#[must_use]
pub fn without_reminder(text: &str) -> &str {
    let Some(span) = reminder_span(text) else {
        return text;
    };
    text[span.len()..].trim_start_matches('\n')
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, reason = "test assertions")]

    use super::*;

    #[test]
    fn a_reminder_round_trips_to_the_text_it_wrapped() {
        let sent = with_reminder("fix the login flow");

        assert_eq!(
            reminder_span(&sent).unwrap(),
            reminder_span(&with_reminder("x")).unwrap()
        );
        assert_eq!(without_reminder(&sent), "fix the login flow");
        // And back the other way: re-wrapping the visible text reproduces it.
        assert_eq!(with_reminder(without_reminder(&sent)), sent);
    }

    #[test]
    fn text_without_a_reminder_passes_through_untouched() {
        for plain in ["hello", "", "<system-reminder> opened but never closed"] {
            assert_eq!(without_reminder(plain), plain);
            assert!(reminder_span(plain).is_none());
        }
    }

    /// A reminder the user's own text happens to repeat is not mistaken for the frame.
    #[test]
    fn only_the_leading_block_is_stripped() {
        let sent = with_reminder("see <system-reminder> in the docs\n</system-reminder>");
        assert_eq!(
            without_reminder(&sent),
            "see <system-reminder> in the docs\n</system-reminder>"
        );
        // The span ends at the first close tag, inside the frame.
        assert!(reminder_span(&sent).unwrap().ends_with(REMINDER_CLOSE));
        assert!(!reminder_span(&sent).unwrap().contains("docs"));
    }

    #[test]
    fn the_canned_prompts_read_as_written() {
        assert!(PLAN_REMINDER.contains("Plan mode is on"));
        assert!(PLAN_APPROVAL.contains("approved"));
        // The frame wraps the whole reminder, tags included.
        let sent = with_reminder("x");
        assert!(sent.starts_with("<system-reminder>\n"));
        assert!(sent.contains("\n</system-reminder>\n\nx"));
    }
}
