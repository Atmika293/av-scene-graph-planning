"""Shared HTML snippet for the published report's math-notation rendering.

Used by benchmark.py and build_correctness_report.py (each embeds one copy
in its own fragment so the fragment is viewable standalone), and by
build_report.py, which strips every embedded copy back out and keeps a
single one in the combined page's <head> -- the same pattern already used
for each fragment's embedded plotly.js CDN <script> tag.

Chart-note text that includes a formula should wrap it in \\( ... \\) for
inline math or \\[ ... \\] for a centered, standalone equation (KaTeX's
auto-render delimiters, configured below).
"""

KATEX_VERSION = "0.16.11"

KATEX_CDN_HTML = (r"""<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@VERSION/dist/katex.min.css">
<script src="https://cdn.jsdelivr.net/npm/katex@VERSION/dist/katex.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/katex@VERSION/dist/contrib/auto-render.min.js"></script>
<script>
document.addEventListener("DOMContentLoaded", function() {
    renderMathInElement(document.body, {
        delimiters: [
            {left: "\\(", right: "\\)", display: false},
            {left: "\\[", right: "\\]", display: true}
        ]
    });
});
</script>
""").replace("VERSION", KATEX_VERSION)
