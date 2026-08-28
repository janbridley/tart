//! Rendering tool-call digests for display.

use std::collections::BTreeMap;
use std::ops::RangeInclusive;

use itertools::Itertools;
use rangemap::RangeInclusiveSet;

/// Merge a run of calls with the same name to consolidate tui output.
///
/// - Read digests coalesce per file-line counts for easier reading.
/// - Edit digests count repeated changes to the same file as `path +N edits`
/// - Anything past [`crate::session::one_line`]'s cap truncates so commands stay readable
#[inline]
pub fn merge_digests(name: &str, digests: &[String]) -> String {
    if let [one] = digests
        && name != "Bash"
    {
        return one.clone();
    }
    let merged = match name {
        "Read" => merge_reads(digests).unwrap_or_else(|| digests.join(", ")),
        "Edit" => merge_edits(digests),
        _ => digests.join(", "),
    };
    crate::session::one_line(&merged)
}

/// Coalesce file reads into a nice set of ranges.
fn merge_reads(digests: &[String]) -> Option<String> {
    let mut files: BTreeMap<&str, RangeInclusiveSet<u64>> = BTreeMap::new();
    for digest in digests {
        let (path, range) = read_range(digest)?;
        files.entry(path).or_default().insert(range);
    }
    Some(
        files
            .iter()
            .format_with(", ", |(path, ranges), f| match (whole_file(ranges), path) {
                (true, p) => f(&p),
                (false, p) => f(&format_args!("{p}:{}", bounds_list(ranges))),
            })
            .to_string(),
    )
}

/// The inverse of `read_digest` in the tools module.
///
/// A missing bound clamps to 0 or `u64::MAX`; `None` when the digest is not parseable.
fn read_range(digest: &str) -> Option<(&str, RangeInclusive<u64>)> {
    // A digest with no bounds reads the file whole.
    let Some((path, bounds)) = digest.rsplit_once(':') else {
        return Some((digest, 0..=u64::MAX));
    };
    let (lo, hi) = bounds.split_once('-')?;
    let bound = |text: &str, default: u64| {
        Some(if text.is_empty() { default } else { text.parse().ok()? })
    };
    let start = bound(lo, 0)?;
    let end = bound(hi, u64::MAX)?;
    (start <= end).then_some((path, start..=end))
}

/// Whether a file's ranges cover all lines.
fn whole_file(ranges: &RangeInclusiveSet<u64>) -> bool {
    ranges
        .iter()
        .any(|range| *range.start() == 0 && *range.end() == u64::MAX)
}

/// The ranges of one file as digest bounds: `400-530,565-570,590`.
fn bounds_list(ranges: &RangeInclusiveSet<u64>) -> String {
    ranges
        .iter()
        .map(|r| match (*r.start(), *r.end()) {
            (s, e) if s == e => s.to_string(),
            (0, e) => format!("-{e}"),
            (s, u64::MAX) => format!("{s}-"),
            (s, e) => format!("{s}-{e}"),
        })
        .collect::<Vec<_>>()
        .join(",")
}

/// Merge repeated consecutive file edits into `+N edits`.
fn merge_edits(digests: &[String]) -> String {
    let edits: BTreeMap<&str, usize> =
        digests.iter().map(String::as_str).counts().into_iter().collect();

    edits
        .iter()
        .map(|(path, count)| match *count {
            1 => (*path).to_string(),
            2 => format!("{path} +1 edit"),
            _ => format!("{path} +{} edits", count - 1),
        })
        .collect::<Vec<_>>()
        .join(", ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merge_digests_coalesces_counts_and_caps() {
        let digests = |parts: &[&str]| parts.iter().map(ToString::to_string).collect::<Vec<_>>();

        // The four-line gap stays split; the overlap unions; ascending order.
        assert_eq!(
            merge_digests(
                "Read",
                &digests(&["a.rs:485-560", "a.rs:420-480", "a.rs:555-600"])
            ),
            "a.rs:420-480,485-600"
        );
        // A single line renders as its bare number.
        assert_eq!(
            merge_digests("Read", &digests(&["a.rs:590-590", "a.rs:400-400"])),
            "a.rs:400,590"
        );
        // An open tail absorbs whatever it covers.
        assert_eq!(
            merge_digests("Read", &digests(&["a.rs:5-", "a.rs:10-20"])),
            "a.rs:5-"
        );
        // A whole-file read subsumes the bounded ones; files sort by path.
        assert_eq!(
            merge_digests("Read", &digests(&["b.rs:1-2", "a.rs", "a.rs:10-20"])),
            "a.rs, b.rs:1-2"
        );
        // A lone digest renders verbatim.
        assert_eq!(merge_digests("Read", &digests(&["a.rs:1-9"])), "a.rs:1-9");

        // Edits count repeats beyond the first; other files ride along.
        assert_eq!(
            merge_digests("Edit", &digests(&["x.py", "x.py", "x.py"])),
            "x.py +2 edits"
        );
        assert_eq!(
            merge_digests("Edit", &digests(&["b.rs", "a.rs", "a.rs"])),
            "a.rs +1 edit, b.rs"
        );

        // Bash caps to its first line: 60 kept chars plus the ellipsis.
        let multiline = vec![format!("{}\nsecond line", "x".repeat(70))];
        let capped = merge_digests("Bash", &multiline);
        assert_eq!(capped.chars().count(), 61);
        assert!(capped.ends_with('…'), "{capped}");

        // Reversed bounds would panic the range set; they degrade to a join.
        assert_eq!(
            merge_digests("Read", &digests(&["a.rs:30-10", "a.rs:1-5"])),
            "a.rs:30-10, a.rs:1-5"
        );
    }
}
