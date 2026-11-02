"""Assemble the published report from whichever section fragments exist.

Combines docs/_performance_section.html (benchmark.py, Phase 2) and
docs/_correctness_section.html (build_correctness_report.py, Phase 5) into
one docs/index.html: a single <head> (one plotly.js CDN load and one KaTeX
CDN load, not one per fragment -- each fragment embeds its own copy of both
so it's independently viewable while being built/debugged in isolation, but
the combined page only needs one of each), a short nav linking the section
anchors each fragment already defines
(<section id="performance">/<section id="correctness">), and the fragments
themselves, unmodified except for stripping their embedded CDN tags.

Either fragment can be absent (e.g. only one phase has been run so far) --
the page is built from whichever exist, skipping missing ones and their
nav link.

Run with: python scripts/build_report.py (from the repo root, after
benchmark.py and/or build_correctness_report.py have produced their
fragments).
"""

import os
import re

from av_data_pipeline.report_html import KATEX_CDN_HTML

REPORT_DIR = "docs"
OUTPUT_FILE = os.path.join(REPORT_DIR, "index.html")

# (anchor matching the fragment's own <section id="...">, nav label, fragment path)
SECTIONS = [
    ("performance", "Performance", os.path.join(REPORT_DIR, "_performance_section.html")),
    ("correctness", "Correctness", os.path.join(REPORT_DIR, "_correctness_section.html")),
]

# Matches a fragment's embedded plotly.js CDN <script> tag regardless of
# version, so this doesn't go stale if the installed plotly library (and so
# the version plotly.io embeds) is upgraded later.
PLOTLYJS_SCRIPT_RE = re.compile(r'<script charset="utf-8" src="https://cdn\.plot\.ly/[^"]+"[^>]*></script>')


def load_fragments() -> list[tuple[str, str, str]]:
    """Read every section fragment that currently exists under docs/.

    Returns:
        (anchor, label, html) for each SECTIONS entry whose file exists, in
        SECTIONS order; entries whose file is missing are skipped.
    """
    fragments = []
    for anchor, label, path in SECTIONS:
        if not os.path.exists(path):
            print(f"Skipping missing section: {path}")
            continue
        with open(path) as fp:
            fragments.append((anchor, label, fp.read()))

    return fragments


def extract_plotlyjs_script(fragments: list[tuple[str, str, str]]) -> str:
    """Find one fragment's embedded plotly.js CDN script tag, to hoist into <head>.

    Args:
        fragments: Loaded fragments, as returned by load_fragments.

    Returns:
        The first plotly.js CDN <script> tag found across the fragments.

    Raises:
        SystemExit: If no fragment contains one -- the combined page's
            charts wouldn't render at all.
    """
    for _, _, html in fragments:
        match = PLOTLYJS_SCRIPT_RE.search(html)
        if match:
            return match.group(0)

    raise SystemExit("No plotly.js CDN <script> tag found in any section fragment -- "
                     "cannot assemble a page whose charts would render.")


def build_page(fragments: list[tuple[str, str, str]]) -> str:
    """Build the combined page: one <head>, a nav, then every section in order.

    Args:
        fragments: Loaded fragments, as returned by load_fragments.

    Returns:
        The full docs/index.html contents.
    """
    plotlyjs_script = extract_plotlyjs_script(fragments)

    nav_links = " ".join(f'<a href="#{anchor}">{label}</a>' for anchor, label, _ in fragments)
    # Each fragment keeps its own embedded copy of both CDN resources for
    # standalone viewing; strip them here since the combined page loads
    # plotly.js and KaTeX once each, in <head>. KATEX_CDN_HTML is a known,
    # exact, shared constant (unlike plotly's version-varying script tag),
    # so a plain string removal is enough -- no regex needed.
    sections_html = "\n".join(
        PLOTLYJS_SCRIPT_RE.sub("", html, count=1).replace(KATEX_CDN_HTML, "")
        for _, _, html in fragments
    )

    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>av-data-pipeline: performance &amp; correctness report</title>
{plotlyjs_script}
{KATEX_CDN_HTML}
<style>
 body {{ font-family: system-ui, sans-serif; margin: 0 auto; max-width: 1100px; padding: 24px; }}
 nav {{ margin-bottom: 24px; }}
 nav a {{ margin-right: 16px; font-weight: 600; text-decoration: none; }}
 h2 {{ border-bottom: 1px solid #ccc; padding-bottom: 4px; }}
 table {{ border-collapse: collapse; margin: 12px 0; }}
 th, td {{ border: 1px solid #ccc; padding: 4px 10px; text-align: right; }}
 figure {{ display: inline-block; vertical-align: top; }}
 .chart-note {{ max-width: 720px; color: #555; font-size: 0.92em; line-height: 1.5; margin: 16px 0 4px 0; }}
</style>
</head>
<body>
<nav>{nav_links}</nav>
{sections_html}
</body>
</html>
"""


if __name__ == "__main__":
    fragments = load_fragments()
    if not fragments:
        raise SystemExit("No section fragments found under docs/ -- nothing to assemble. "
                         "Run benchmark.py and/or build_correctness_report.py first.")

    os.makedirs(REPORT_DIR, exist_ok=True)
    with open(OUTPUT_FILE, "w") as outfile:
        outfile.write(build_page(fragments))

    labels = ", ".join(label for _, label, _ in fragments)
    print(f"Wrote {OUTPUT_FILE} ({len(fragments)} section(s): {labels})")
