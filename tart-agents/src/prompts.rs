//! The canned prompts plan mode adds.
//!
//! [`PLAN_REMINDER`] is the mode's standing instruction and [`PLAN_APPROVAL`] is the
//! user turn that starts implementing an approved plan. Both reach the model through
//! the transcript, with the reminder sent as asystem message spliced in behind the
//! leading system block while the mode is on (see [`Transcript::set_reminder`]). Neither is ever stored inside a user message,
//! so a session record replays as the user wrote it.
//!
//! [`Transcript::set_reminder`]: crate::Transcript::set_reminder

/// The standing instruction for every request while plan mode is on.
pub const PLAN_REMINDER: &str = include_str!("data/PLAN.md");

/// The user turn that follows an approved plan, sent to start implementing it.
pub const PLAN_APPROVAL: &str = "The plan above is approved: implement it now. Edit files \
     as needed, follow the numbered steps in order, and say what changed when you finish.";

/// The system prompt for `--chat` sessions.
pub const CHAT: &str = include_str!("data/CHAT.md");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_canned_prompts_read_as_written() {
        assert!(PLAN_REMINDER.contains("Plan mode is on"));
        assert!(
            PLAN_REMINDER.contains("read-only"),
            "the reminder names the contract"
        );
        assert!(PLAN_APPROVAL.contains("approved"));
        assert!(
            CHAT.contains("no shell and no filesystem"),
            "chat owns its limits"
        );
        assert!(
            !CHAT.contains("spawn_agent"),
            "the minimal prompt does not advertise withheld tools"
        );
    }
}
