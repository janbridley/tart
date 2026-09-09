//! Terminal spinners for the statusline and agent statuses.
//!
//! Many of these are re-implemented from patterns in the `bash-cli-spinners` gallery,
//! but with different cadences and custom closed-form expressions.

/// The draw loop's cadence, which every agent table shares.
const CADENCE: u128 = crate::DRAW_INTERVAL_MS as u128;

/// The braille cell for an eight-dot `mask`.
fn braille(mask: u8) -> char {
    char::from_u32(0x2800 + u32::from(mask)).unwrap_or('\u{2800}')
}

/// Dot-bit positions: up the left column, across the top, down the right one
const ARCH: [u8; 6] = [2, 1, 0, 3, 4, 5];

/// Dot-bit positions: down the left column, across the bottom, up the right one
const RIM: [u8; 8] = [0, 1, 2, 6, 7, 5, 4, 3];

/// `sand`: falling pips that pile and dissolve on two independent schedules
const SAND: [u8; 35] = [
    0x01, 0x02, 0x04, 0x40, 0x48, 0x50, 0x60, 0xc0, 0xc1, 0xc2, 0xc4, 0xcc, 0xd4, 0xe4, 0xe5, 0xe6,
    0xee, 0xf6, 0xf7, 0xff, 0x7f, 0x3f, 0x9f, 0x1f, 0x5b, 0x1b, 0x2b, 0x8b, 0x0b, 0x0d, 0x49, 0x09,
    0x11, 0x21, 0x81,
];
#[inline]
fn sand(i: usize) -> char {
    braille(SAND[i])
}
/// Three-pip spinner that walks up and down the character.
#[inline]
fn climb(i: usize) -> char {
    const T: [u8; 6] = [0x01, 0x01, 0x09, 0x19, 0x1A, 0x12];
    let j = i.min(28 - i);
    let x = T[j % 6] << (j / 6);
    braille(x ^ (((x ^ x >> 3) & 7) * 9 * u8::from(i >= 15)))
}

/// Four-pip snake moving clockwise around the character.
#[inline]
fn snake(i: usize) -> char {
    const PATH: [u8; 8] = [0, 3, 4, 1, 2, 5, 7, 6];
    braille((0..3).map(|step| 1 << PATH[(i + step) % 8]).sum())
}

/// Two pips hopping around the corners of the character.
#[inline]
fn hops(i: usize) -> char {
    braille(1 << ARCH[i - (i + 4) / 7] | 0x80 >> ((i + 4) / 7))
}

/// Eight-bit counter using all pips of character.
#[inline]
fn counter(i: usize) -> char {
    braille((i & 0x87 | i >> 1 & 0x38 | (i & 8) << 3) as u8)
}

/// A four-pip comet bouncing back and forth along the top of the character.
#[inline]
fn bounce(i: usize) -> char {
    let head = i.min(14 - i);
    let window = head.saturating_sub(2)..=head.min(5);
    braille(ARCH[window].iter().map(|bit| 1 << bit).sum())
}

/// A single pip orbiting a full filled column.
#[inline]
fn orbit(i: usize) -> char {
    braille(if i < 4 { 0xB8 } else { 0x47 } | 1 << RIM[i])
}

/// A single pip orbiting the empty character.
#[inline]
fn solo(i: usize) -> char {
    braille(1 << RIM[i])
}

/// A terminal spinner with its frame count and transition function.
pub struct Spinner {
    /// Frames in one period of the cycle.
    period: usize,
    /// The cell at `index` into the period.
    cell: fn(usize) -> char,
}

impl Spinner {
    /// The cell showing at `millis` past the cycle's start.
    #[inline]
    pub fn frame(&self, millis: u128) -> char {
        (self.cell)((millis / CADENCE) as usize % self.period)
    }
}

/// The subagent block's spinners, one per slot, taken in arrival order:
pub const AGENTS: [Spinner; 8] = [
    Spinner { period: 35, cell: sand },
    Spinner { period: 8, cell: snake },
    Spinner { period: 7, cell: hops },
    Spinner { period: 256, cell: counter },
    Spinner { period: 14, cell: bounce },
    Spinner { period: 8, cell: orbit },
    Spinner { period: 8, cell: solo },
    Spinner { period: 29, cell: climb },
];

/// MAIN's dot frame showing at `millis` past its cycle's start.
#[inline]
pub fn main_dots(millis: u128) -> &'static str {
    const DOTS: [&str; 6] = ["·  ", "·· ", "···", " ··", "  ·", "   "];
    DOTS[(millis / (2 * CADENCE)) as usize % DOTS.len()]
}

#[cfg(test)]
mod tests {
    #![allow(clippy::panic, reason = "test assertions")]

    use super::*;

    /// Every table's literal sequence, one character per frame.
    const GOLDEN: [&str; 8] = [
        "⠁⠂⠄⡀⡈⡐⡠⣀⣁⣂⣄⣌⣔⣤⣥⣦⣮⣶⣷⣿⡿⠿⢟⠟⡛⠛⠫⢋⠋⠍⡉⠉⠑⠡⢁",
        "⠙⠚⠖⠦⢤⣠⣁⡉",
        "⢄⢂⢁⡁⡈⡐⡠",
        concat!(
            "⠀⠁⠂⠃⠄⠅⠆⠇⡀⡁⡂⡃⡄⡅⡆⡇⠈⠉⠊⠋⠌⠍⠎⠏⡈⡉⡊⡋⡌⡍⡎⡏⠐⠑⠒⠓⠔⠕⠖⠗⡐⡑⡒⡓⡔⡕⡖⡗",
            "⠘⠙⠚⠛⠜⠝⠞⠟⡘⡙⡚⡛⡜⡝⡞⡟⠠⠡⠢⠣⠤⠥⠦⠧⡠⡡⡢⡣⡤⡥⡦⡧⠨⠩⠪⠫⠬⠭⠮⠯⡨⡩⡪⡫⡬⡭⡮⡯",
            "⠰⠱⠲⠳⠴⠵⠶⠷⡰⡱⡲⡳⡴⡵⡶⡷⠸⠹⠺⠻⠼⠽⠾⠿⡸⡹⡺⡻⡼⡽⡾⡿⢀⢁⢂⢃⢄⢅⢆⢇⣀⣁⣂⣃⣄⣅⣆⣇",
            "⢈⢉⢊⢋⢌⢍⢎⢏⣈⣉⣊⣋⣌⣍⣎⣏⢐⢑⢒⢓⢔⢕⢖⢗⣐⣑⣒⣓⣔⣕⣖⣗⢘⢙⢚⢛⢜⢝⢞⢟⣘⣙⣚⣛⣜⣝⣞⣟",
            "⢠⢡⢢⢣⢤⢥⢦⢧⣠⣡⣢⣣⣤⣥⣦⣧⢨⢩⢪⢫⢬⢭⢮⢯⣨⣩⣪⣫⣬⣭⣮⣯⢰⢱⢲⢳⢴⢵⢶⢷⣰⣱⣲⣳⣴⣵⣶⣷",
            "⢸⢹⢺⢻⢼⢽⢾⢿⣸⣹⣺⣻⣼⣽⣾⣿",
        ),
        "⠄⠆⠇⠋⠙⠸⠰⠠⠰⠸⠙⠋⠇⠆",
        "⢹⢺⢼⣸⣇⡧⡗⡏",
        "⠁⠂⠄⡀⢀⠠⠐⠈",
        "⠁⠁⠉⠙⠚⠒⠂⠂⠒⠲⠴⠤⠄⠄⠤⠠⠠⠤⠦⠖⠒⠐⠐⠒⠓⠋⠉⠈⠈",
    ];

    /// Every table generates exactly its literal sequence with one cell per frame.
    #[test]
    fn tables_generate_their_literal_sequences() {
        for (slot, (golden, spinner)) in GOLDEN.into_iter().zip(AGENTS).enumerate() {
            let made: String = (0..spinner.period).map(spinner.cell).collect();
            assert_eq!(made, golden, "slot {slot}");
        }
    }

    /// MAIN's dots cycle their padded cells.
    #[test]
    fn main_dots_cycle() {
        let frames = [0, 1, 4, 6].map(|i| main_dots(i * 2 * CADENCE));
        assert_eq!(frames, ["·  ", "·· ", "  ·", "·  "]);
    }

    #[test]
    fn frames_start_and_wrap() {
        let frames = [0, 1, 8].map(|i| AGENTS[1].frame(i * CADENCE));
        assert_eq!(frames, ['⠙', '⠚', '⠙']);
    }

    /// One spinner per concurrent subagent the registry can run, and no more.
    #[test]
    fn every_concurrent_subagent_has_a_spinner() {
        assert_eq!(AGENTS.len(), tart_agents::MAX_SUBAGENTS);
    }

    /// `dots8Bit` visits every braille pattern exactly once.
    #[test]
    fn counter_enumerates_every_braille_pattern_once() {
        let seen: std::collections::HashSet<_> = (0..256).map(counter).collect();
        assert_eq!(seen.len(), 256);
    }
}
