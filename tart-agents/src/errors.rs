//! How a failed provider exchange is answered: classified, retried, or ended.

use std::time::Duration;

use crate::Progress;

/// The most consecutive rounds a dropped stream retries before failing.
const MAX_STREAM_RETRIES: usize = 4;

/// The pause before the first retry; it doubles on each further one
/// (2s, 4s, 8s, 16s).
#[cfg(not(test))]
const RETRY_PAUSE: Duration = Duration::from_secs(2);
/// A test's pause is nothing, so budget tests run in milliseconds.
#[cfg(test)]
const RETRY_PAUSE: Duration = Duration::from_millis(1);

/// Count one dropped round, reporting the retry and pausing while one remains.
///
/// Returns whether the round should re-run. `false` means the caller fails the
/// turn with the reason. A completed round resets the count, so four *consecutive*
/// drops end the turn.
pub(crate) fn retry_dropped<F: Fn(Progress)>(
    on_progress: &F,
    reason: &str,
    retries: &mut usize,
) -> bool {
    *retries += 1;
    if *retries > MAX_STREAM_RETRIES {
        return false;
    }
    on_progress(Progress::Note(format!(
        "{reason}; retry {retries}/{MAX_STREAM_RETRIES}"
    )));
    std::thread::sleep(RETRY_PAUSE * (1u32 << (*retries - 1)));
    true
}

/// How a provider error reads: what the round should do with it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ProviderError {
    /// Transient trouble -> retry with exponential backoff
    Transient,
    /// The request is refused as-is, so no backoff could recover anything.
    Permanent,
    /// The transcript is over the model's context window, so the turn ends immediately.
    Overflow,
}

/// Classify a provider error by its text, whichever error type it arrived in.
pub(crate) fn provider_error(reason: &str) -> ProviderError {
    let reason = reason.to_ascii_lowercase();
    let hits = |file: &'static str| needles(file).iter().any(|needle| reason.contains(needle));
    if hits(include_str!("data/provider-overflow.txt")) {
        ProviderError::Overflow
    } else if hits(include_str!("data/provider-permanent.txt")) {
        ProviderError::Permanent
    } else {
        ProviderError::Transient
    }
}

/// The needle lines of one data file: trimmed, `#` comments aside.
fn needles(file: &'static str) -> Vec<&'static str> {
    file.lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .collect()
}

/// Answer a provider error with the round's next move: `None` retries it
/// along the backoff, and a terminal [`Progress`] ends the turn.
pub(crate) fn absorb_provider_error<F: Fn(Progress)>(
    on_progress: &F,
    reason: &str,
    retries: &mut usize,
) -> Option<Progress> {
    match provider_error(reason) {
        ProviderError::Transient => {
            if retry_dropped(on_progress, reason, retries) {
                return None;
            }
            Some(Progress::Failed(reason.to_string()))
        }
        // An over-window record cannot be retried into fitting: the turn
        // ends saying the drastic action instead.
        ProviderError::Overflow => Some(Progress::Failed(format!(
            "the transcript exceeds the model's context window: /clear or trim the session: {reason}"
        ))),
        ProviderError::Permanent => Some(Progress::Failed(reason.to_string())),
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, reason = "test assertions")]

    use super::*;
    use std::sync::Mutex;

    /// Provider errors class by their own text, whichever lane they arrive
    /// in: transient trouble rides the backoff, refusals end at once, and an
    /// over-window transcript is its own class.
    #[test]
    fn provider_errors_class_by_text() {
        use ProviderError::*;

        let cases = [
            ("rate limit exceeded, retry later", Transient),
            ("internal server error", Transient),
            ("connection reset by peer", Transient),
            // An input-token *rate limit* mentions input tokens; it must not
            // read as the context window.
            (
                "Rate limit reached for input tokens per minute: Limit 150000, Used 90000",
                Transient,
            ),
            ("", Transient),
            ("Unauthorized: invalid credentials", Permanent),
            ("Incorrect API key provided", Permanent),
            ("your billing quota is exceeded", Permanent),
            ("content policy violation", Permanent),
            // The providers' own documented wording, pinned case by case.
            (
                "Insufficient balance or no resource package. Please recharge.",
                Permanent,
            ),
            (
                "System detected potentially unsafe or sensitive content in input or generation.",
                Permanent,
            ),
            ("Your GLM Coding Plan package has expired.", Permanent),
            (
                "Your account or API key has insufficient credits. Add more credits and retry.",
                Permanent,
            ),
            ("content_policy_violation", Permanent),
            ("credit_balance_exhausted", Permanent),
            ("organization_spend_limit_exceeded", Permanent),
            ("This model's maximum context length is 262144 tokens", Overflow),
            ("the prompt is too long: 390000 > 262144 tokens", Overflow),
            // z.ai's phrasing carries no "is".
            ("Prompt too long", Overflow),
            // vLLM's pre-check names its own limit.
            (
                "The engine prompt length 390000 exceeds the max_model_len 262144. Please reduce prompt.",
                Overflow,
            ),
            (
                "payload_too_large: The request body exceeds the maximum allowed size.",
                Overflow,
            ),
            // Machine identifiers arrive underscored in the error code.
            ("context_length_exceeded: requested 390000", Overflow),
            ("prompt_too_long", Overflow),
        ];
        for (reason, class) in cases {
            assert_eq!(provider_error(reason), class, "{reason}");
        }
    }

    /// The retry budget: each drop reports its retry, and the fifth
    /// consecutive one ends the turn instead.
    #[test]
    fn four_consecutive_drops_then_give_up() {
        let notes = Mutex::new(Vec::new());
        let mut retries = 0;
        // Keep dropping until `retry_dropped` stops offering a retry.
        while retry_dropped(
            &|progress| {
                if let Progress::Note(text) = progress {
                    notes.lock().unwrap().push(text);
                }
            },
            "stream ended without output",
            &mut retries,
        ) {}

        let notes = notes.lock().unwrap();
        assert_eq!(
            *notes,
            vec![
                "stream ended without output; retry 1/4",
                "stream ended without output; retry 2/4",
                "stream ended without output; retry 3/4",
                "stream ended without output; retry 4/4",
            ],
        );
        // The count a completed round resets.
        assert_eq!(retries, 5);
    }
}
