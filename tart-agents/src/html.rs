//! Render an HTML document down to the text a model actually wants to read.
//!
//! Pages are mostly markup: navigation, scripts, styles, and attributes that carry no
//! prose. [`to_text`] strips all of that while keeping the two things a model needs —
//! the visible words, and where the links point — without pulling in an HTML parser
//! dependency.
//!
//! The reader is a hand-written scanner rather than a tree builder: it never has to be
//! right about the document, only about the text. Malformed markup degrades the same
//! way a browser's recovery does, by skipping to the next tag.

/// Elements whose entire contents are dropped: no browser renders their text.
const DROP: [&str; 12] = [
    "script", "style", "noscript", "template", "svg", "math", "iframe", "canvas", "object",
    "embed", "audio", "video",
];

/// Elements that start a new line: the browser's block-level set, plus a few that only
/// ever hold chrome.
const BLOCK: [&str; 39] = [
    "address", "article", "aside", "blockquote", "body", "caption", "dd", "div", "dl", "dt",
    "fieldset", "figcaption", "figure", "footer", "form", "h1", "h2", "h3", "h4", "h5", "h6",
    "header", "hr", "html", "legend", "main", "nav", "ol", "p", "pre", "section", "summary",
    "table", "tbody", "td", "tfoot", "th", "thead", "tr", "ul",
];

/// Longest entity we look for, counting the `&` and `;`: `&hellip;` and `&#x27;` fit.
const ENTITY_LIMIT: usize = 12;

/// A small, common subset of the named entities, plus the numeric forms.
///
/// The named set covers what survives in real page text; anything else is left as
/// written rather than guessed at.
fn named_entity(name: &str) -> Option<&'static str> {
    Some(match name {
        "amp" => "&",
        "lt" => "<",
        "gt" => ">",
        "quot" => "\"",
        "apos" | "rsquo" => "'",
        "lsquo" => "\u{2018}",
        "ldquo" => "\u{201c}",
        "rdquo" => "\u{201d}",
        "nbsp" => " ",
        "hellip" => "\u{2026}",
        "mdash" => "\u{2014}",
        "ndash" => "\u{2013}",
        "bull" | "middot" => "\u{2022}",
        "copy" => "\u{a9}",
        "reg" => "\u{ae}",
        "trade" => "\u{2122}",
        "deg" => "\u{b0}",
        "plusmn" => "\u{b1}",
        "times" => "\u{d7}",
        "divide" => "\u{f7}",
        "minus" => "\u{2212}",
        "prime" => "\u{2032}",
        "larr" => "\u{2190}",
        "uarr" => "\u{2191}",
        "rarr" => "\u{2192}",
        "darr" => "\u{2193}",
        "harr" => "\u{2194}",
        "infin" => "\u{221e}",
        "asymp" | "equiv" => "\u{2248}",
        "ne" => "\u{2260}",
        "le" => "\u{2264}",
        "ge" => "\u{2265}",
        "pound" => "\u{a3}",
        "euro" => "\u{20ac}",
        "yen" => "\u{a5}",
        "cent" => "\u{a2}",
        "sect" => "\u{a7}",
        "para" => "\u{b6}",
        "laquo" => "\u{ab}",
        "raquo" => "\u{bb}",
        _ => return None,
    })
}

/// Decode one character reference starting at `&`, returning it and the input consumed.
fn entity_at(text: &[u8], start: usize) -> Option<(String, usize)> {
    let end = text[start + 1..]
        .iter()
        .take(ENTITY_LIMIT)
        .position(|byte| *byte == b';')
        .map(|offset| start + 1 + offset)?;
    let name = std::str::from_utf8(&text[start + 1..end]).ok()?;
    if let Some(decimal) = name.strip_prefix('#') {
        let digits = decimal.strip_prefix('x').map(str::to_ascii_lowercase).map_or_else(
            || decimal.to_string(),
            |hex| format!("0x{hex}"),
        );
        // Bases are checked so a leading `+` cannot sneak past `from_str_radix`.
        if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_hexdigit() || b == b'x') {
            return None;
        }
        let radix = if digits.starts_with("0x") { 16 } else { 10 };
        let code = u32::from_str_radix(digits.trim_start_matches("0x"), radix).ok()?;
        return char::from_u32(code).map(|c| (c.to_string(), end - start + 1));
    }
    named_entity(name).map(|text| (text.to_string(), end - start + 1))
}

/// Accumulates output text, collapsing whitespace the way a browser's renderer does.
struct Renderer {
    /// The rendered text so far.
    out: String,
    /// Depth of open `pre`/`textarea` elements; their whitespace is preserved.
    preformatted: usize,
}

impl Renderer {
    /// Append page text, collapsing runs of whitespace unless preformatted.
    fn text(&mut self, text: &str) {
        for token in text.split_whitespace() {
            if !self.out.is_empty() && !self.out.ends_with(['\n', ' ']) {
                self.out.push(' ');
            }
            self.out.push_str(token);
        }
    }

    /// Append preformatted text verbatim.
    fn raw(&mut self, text: &str) {
        self.out.push_str(text);
    }

    /// Start a new line, at most once in a row.
    fn newline(&mut self) {
        if !self.out.ends_with('\n') && !self.out.is_empty() {
            self.out.push('\n');
        }
    }
}

/// One scanned tag: its name, whether it closes, and its attribute text.
struct Tag<'a> {
    /// Lowercased element name.
    name: String,
    /// True for a closing tag (`</a>`).
    closing: bool,
    /// The attributes as written, for pulling out `href`.
    attributes: &'a str,
}

/// Scan the next tag out of `html`, returning it and the offset just past its `>`.
fn tag_at(html: &str) -> Option<(Tag<'_>, usize)> {
    let bytes = html.as_bytes();
    let open = bytes.iter().position(|byte| *byte == b'<')?;
    let rest = &html[open..];
    let closing = rest.starts_with("</");
    let name_start = if closing { 2 } else { 1 };
    let name_end = rest[name_start..]
        .find(|c: char| c.is_whitespace() || c == '>' || c == '/')
        .map_or(rest.len(), |offset| name_start + offset);
    if name_end <= name_start {
        return None;
    }
    let name = rest[name_start..name_end].to_ascii_lowercase();
    let after = rest[name_end..].find('>').map_or(rest.len(), |offset| name_end + offset + 1);
    let attributes = &rest[name_end..after.min(rest.len())];
    Some((
        Tag {
            name,
            closing,
            attributes: attributes.trim_end_matches(['>', '/']).trim(),
        },
        open + after,
    ))
}

/// The `href` of an attribute list, if present and not obviously relative junk.
fn href(attributes: &str) -> Option<String> {
    let value = attributes
        .split_ascii_whitespace()
        .find(|attr| attr.len() > 5 && attr[..5].eq_ignore_ascii_case("href="))?;
    let url = value[5..].trim_matches(['"', '\'']);
    (url.starts_with("http://") || url.starts_with("https://")).then(|| url.to_string())
}

/// Render `html` to readable plain text.
///
/// Scripts, styles, and non-visual elements are dropped; block boundaries become
/// newlines; whitespace collapses; anchor targets are appended in `[…]` when the link
/// text does not already spell them out.
pub(crate) fn to_text(html: &str) -> String {
    let mut renderer = Renderer {
        out: String::with_capacity(html.len() / 2),
        preformatted: 0,
    };
    // The byte offset of the pending anchor's text, so `</a>` can attribute it.
    let mut anchor: Option<(String, usize)> = None;
    let mut rest = html;

    while let Some((tag, consumed)) = tag_at(rest) {
        let (head, tail) = rest.split_at(consumed);
        renderer.text(head);
        rest = tail;
        let name = tag.name.as_str();

        if DROP.contains(&name) {
            // Raw-text elements swallow anything up to their own closing tag, so a `<`
            // inside a script body cannot be mistaken for markup.
            skip_closed(&mut rest, name);
            continue;
        }
        if tag.closing {
            if name == "pre" || name == "textarea" {
                renderer.preformatted = renderer.preformatted.saturating_sub(1);
            }
            if name == "a"
                && let Some((url, mark)) = anchor.take()
            {
                let gained = renderer.out.len() - mark;
                let text = renderer.out[mark..].to_string();
                if gained > 0 && gained < 200 && !text.contains(&url) {
                    renderer.raw(format!(" <{url}>"));
                }
            }
            continue;
        }
        if name == "pre" || name == "textarea" {
            renderer.preformatted += 1;
            renderer.newline();
        } else if name == "br" {
            renderer.newline();
        } else if name == "li" {
            renderer.newline();
            renderer.raw("- ");
        } else if BLOCK.contains(&name) {
            renderer.newline();
        } else if name == "title" {
            renderer.newline();
        }
        if name == "a" && let Some(url) = href(tag.attributes) {
            anchor = Some((url, renderer.out.len()));
        }
    }
    renderer.text(rest);
    tidy(&renderer.out)
}

/// Advance `rest` past the closing tag for a dropped element, dropping its contents.
fn skip_closed(rest: &mut &str, name: &str) {
    let closer = format!("</{name}");
    match rest.to_ascii_lowercase().find(&closer) {
        Some(found) => *rest = &rest[found + closer.len()..],
        // Unterminated: the rest of the document is that element's content.
        None => *rest = "",
    }
}

/// Collapse blank-line runs, strip line-trailing spaces, and trim the ends.
fn tidy(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut blanks = 0;
    for line in text.lines() {
        let trimmed = line.trim_end();
        if trimmed.is_empty() {
            blanks += 1;
        } else {
            blanks = 0;
        }
        if blanks > 1 {
            continue;
        }
        out.push_str(trimmed);
        out.push('\n');
    }
    out.trim().to_string()
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, reason = "test assertions")]

    use super::*;

    #[test]
    fn strips_scripts_styles_and_comments() {
        let html = r#"<html><head><title>T</title><style>a{color:red}</style>
<script>if (1 < 2) { document.write("<p>nope</p>") }</script></head>
<body><!-- gone --><p>kept</p></body></html>"#;

        assert_eq!(to_text(html), "T\nkept");
    }

    #[test]
    fn block_elements_become_newlines_and_lists_get_bullets() {
        let html = "<h1>Title</h1><p>One</p><p>Two</p><ul><li>a</li><li>b</li></ul>";

        assert_eq!(to_text(html), "Title\nOne\nTwo\n- a\n- b");
    }

    #[test]
    fn anchor_targets_are_appended_when_the_text_differs() {
        assert_eq!(
            to_text(r##"<p>See <a href="https://example.com/docs">the docs</a> now.</p>"##),
            "See the docs <https://example.com/docs> now."
        );
        // The link text already carries the target, so it is not repeated.
        assert_eq!(
            to_text(r#"<a href="https://example.com">https://example.com</a>"#),
            "https://example.com"
        );
        // Relative and non-http targets carry no information worth keeping.
        assert_eq!(to_text(r#"<a href="/about">About</a>"#), "About");
    }

    #[test]
    fn entities_decode_including_numeric_forms() {
        assert_eq!(to_text("<p>a &amp; b &lt;c&gt; &#65;&#x42;</p>"), "a & b <c> AB");
        assert_eq!(to_text("<p>&nbsp;&hellip;&mdash;</p>"), "\u{2026}\u{2014}");
        // Unknown or malformed references pass through untouched.
        assert_eq!(to_text("<p>&nope; & 5 &amp</p>"), "&nope; & 5 &amp");
    }

    #[test]
    fn preformatted_text_keeps_its_layout() {
        let html = "<pre>fn main() {\n    println!(\"hi\");\n}</pre>";

        assert_eq!(to_text(html), "fn main() {\n    println!(\"hi\");\n}");
    }

    #[test]
    fn malformed_markup_degrades_to_its_text() {
        assert_eq!(to_text("<p>unclosed <b>bold"), "unclosed bold");
        assert_eq!(to_text("no tags at all"), "no tags at all");
        assert_eq!(to_text("<"), "<");
        assert_eq!(to_text(""), "");
    }

    #[test]
    fn blank_line_runs_collapse_to_one() {
        let html = "<p>a</p><br><br><br><p>b</p>";

        assert_eq!(to_text(html), "a\nb");
    }

    #[test]
    fn entity_at_decodes_the_prefix_and_reports_its_length() {
        let text = b"&amp; rest";
        let (decoded, consumed) = entity_at(text, 0).unwrap();

        assert_eq!(decoded, "&");
        assert_eq!(consumed, "&amp;".len());
        assert_eq!(&text[consumed..], b" rest");
    }
}
