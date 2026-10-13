//! Surface the LaTeX that math-rendered pages carry in their DOM.
//!
//! Nearly every math renderer embeds the TeX source in the HTML it serves:
//! `MathML` annotations (`KaTeX`, `MathJax` v3, `MediaWiki`/Parsoid),
//! `LaTeXML` `@alttext` attributes, `MathJax` v2 scripts, and
//! equation-image alt texts. The extractor prunes the wrappers around the actual tex
//! data so it bubbles up through the later markdown postprocessing .

use dom_query::{Document, Selection};

/// Surface the LaTeX a page carries as plain `$...$` text.
///
/// Four rules, each swapping an element for `` ` $tex$ ` `` plain text:
///
/// - **`MathML` annotations** (`KaTeX`, `MathJax` v3, `MediaWiki`/Parsoid):
///   replace the whole math wrapper or bare `<math>`
/// - **`LaTeXML` `@alttext`** (DLMF, ar5iv): read the attribute from bare `<math>`
///   elements with no annotation
/// - **`MathJax` v2 scripts**: replace `<script type="math/tex">` with its text content
/// - **Equation images** (`WordPress` `latex.php`, `MathWorld` `SVG`s):
///   replace `<img>` with its alt text when the alt or class/src looks like math
///
/// Pages without math markup pass through unmodified.
pub(super) fn math_to_text(html: &str) -> String {
    let doc = Document::from(html);
    // Every rule reduces to "swap this element for ` $tex$ ` plain text".
    let swap = |element: &Selection, tex: &str| {
        let text = doc.tree.new_text(format!(" ${tex}$ "));
        element.replace_with_selection(&text.into());
    };
    let annotation = r#"annotation[encoding="application/x-tex"]"#;
    for wrapper in doc
        .select(
            "span.mwe-math-element, span.katex, mjx-container, span.MathJax, span.math-container",
        )
        .iter()
    {
        let tex = wrapper.select(annotation).first();
        let text = tex.text();
        if !text.trim().is_empty() {
            swap(&wrapper, &text);
        }
    }
    for math in doc.select("math").iter() {
        let annotation = math.select(annotation).first();
        let tex = if annotation.is_empty() {
            math.attr("alttext").unwrap_or_default() // LaTeXML: DLMF, ar5iv
        } else {
            annotation.text()
        };
        if !tex.trim().is_empty() {
            swap(&math, &tex);
        }
    }
    doc.select(r#"script[type^="math/tex"]"#)
        .iter()
        .for_each(|script| swap(&script, &script.text()));
    for img in doc.select("img").iter() {
        let alt = img.attr("alt").unwrap_or_default();
        let class = img.attr("class").unwrap_or_default();
        let src = img.attr("src").unwrap_or_default();
        let math_ish = alt.contains('^')
            || alt.contains('\\')
            || class.contains("latex")
            || class.contains("formula")
            || class.contains("equation")
            || src.contains("latex.php");
        if !alt.trim().is_empty() && math_ish {
            swap(&img, alt.trim());
        }
    }
    doc.html().into()
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, reason = "test assertions")]

    use super::math_to_text;

    #[test]
    fn leaves_pages_without_math_untouched() {
        // dom_query's serializer normalizes (adds <head></head>), so compare
        // content: the same text and elements survive, nothing injected.
        let html = "<html><body><main><p>plain prose</p><pre><code>a = b</code></pre></main></body></html>";
        let out = math_to_text(html);
        assert!(out.contains("plain prose"), "{out}");
        assert!(out.contains("a = b"), "{out}");
        assert!(!out.contains('$'), "{out}");
    }

    #[test]
    fn swaps_katex_wrappers_for_their_annotation_tex() {
        let html = r#"<html><body><main><p>x <span class="katex"><span class="katex-mathml"><math><semantics><mrow><msup><mi>x</mi><mn>2</mn></msup></mrow><annotation encoding="application/x-tex">x^2</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span>x²</span></span></span> y</p></main></body></html>"#;
        let out = math_to_text(html);
        assert!(out.contains("$x^2$"), "{out}");
        assert!(!out.contains("katex-html"), "{out}");
    }

    #[test]
    fn swaps_mediawiki_wrappers_for_their_annotation_tex() {
        let html = r#"<html><body><main><p><span class="mwe-math-element"><span class="mwe-math-mathml-inline" style="display:none"><math><semantics><mrow><mi>E</mi></mrow><annotation encoding="application/x-tex">E=mc^2</annotation></semantics></math></span><img src="/wiki/Special:MathShowImage" alt="rendered equation"></span></p></main></body></html>"#;
        let out = math_to_text(html);
        assert!(out.contains("$E=mc^2$"), "{out}");
    }

    #[test]
    fn swaps_bare_math_with_alttext() {
        let html = r#"<html><body><main><p><math class="ltx_Math" alttext="\alpha+\beta">α+β</math></p></main></body></html>"#;
        let out = math_to_text(html);
        assert!(out.contains(r"$\alpha+\beta$"), "{out}");
    }

    #[test]
    fn swaps_mathjax_v2_scripts_for_their_text() {
        let html = r#"<html><body><main><p>Phase <script type="math/tex">\phi(x)</script> evolves.</p></main></body></html>"#;
        let out = math_to_text(html);
        assert!(out.contains(r"$\phi(x)$"), "{out}");
    }

    #[test]
    fn swaps_mathjax_v3_mjx_containers() {
        let html = r#"<html><body><main><p><mjx-container><mjx-assistive-mml><math><semantics><mrow><mi>E</mi></mrow><annotation encoding="application/x-tex">E=mc^2</annotation></semantics></math></mjx-assistive-mml></mjx-container></p></main></body></html>"#;
        let out = math_to_text(html);
        assert!(out.contains("$E=mc^2$"), "{out}");
        assert!(!out.contains("mjx-container"), "{out}");
    }

    #[test]
    fn swaps_equation_images_by_alt_text_on_markers() {
        let html = r#"<html><body><main><p><img src="/equations/eq1.svg" class="numberedequation" alt="\sum_{i=1}^n i"></p></main></body></html>"#;
        let out = math_to_text(html);
        assert!(out.contains(r"$\sum_{i=1}^n i$"), "{out}");
    }

    #[test]
    fn swaps_equation_images_by_class_and_src() {
        // WordPress latex.php: alt carries the TeX un-encoded, no markers
        let html = r#"<html><body><main><p><img src="https://s0.wp.com/latex.php?latex=%7Bx%3Ey%7D&bg=fff" class="latex" alt="{x>y}"></p></main></body></html>"#;
        let out = math_to_text(html);
        // the serializer escapes > in the text node; the extractor un-escapes
        assert!(out.contains(" ${x&gt;y}$ "), "{out}");
    }

    #[test]
    fn no_ops_on_math_containers_without_annotations() {
        // Stack Exchange: the wrapper matches but holds literal $…$ text
        let html = r#"<html><body><main><p><span class="math-container">$\frac{a}{b}$</span></p></main></body></html>"#;
        let out = math_to_text(html);
        assert!(out.contains(r"$\frac{a}{b}$"), "{out}");
        assert_eq!(out.matches('$').count(), 2, "exactly the original two");
    }

    #[test]
    fn no_ops_on_images_without_math_markers() {
        let html = r#"<html><body><main><img src="/logo.png" alt="logo"><img src="/photo.jpg" alt="a photo"></main></body></html>"#;
        let out = math_to_text(html);
        assert!(out.contains("logo"), "{out}");
        assert!(out.contains("a photo"), "{out}");
        assert!(!out.contains('$'), "{out}");
    }

    #[test]
    fn nested_wrapper_torture_case() {
        let html = r#"<html><body><main><p>pre <span class="math-container"><span class="katex"><math><semantics><mrow><mi>E</mi></mrow><annotation encoding="application/x-tex">E=mc^2</annotation></semantics></math></span></span> mid <img src="https://s0.wp.com/latex.php?latex=x" class="latex" alt="x^{2}"> post</p></main></body></html>"#;
        let out = math_to_text(html);
        assert!(out.contains("$E=mc^2$"), "{out}");
        assert!(out.contains("$x^{2}$"), "{out}");
    }

    #[test]
    fn mathml_without_annotation_passes_through() {
        // The one accepted loss: no TeX in the DOM, nothing to surface
        let html = "<html><body><main><p><math><mrow><mi>x</mi><mo>+</mo><mi>y</mi></mrow></math></p></main></body></html>";
        let out = math_to_text(html);
        assert!(!out.contains('$'), "no $ injected: {out}");
    }
}
