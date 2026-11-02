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

# Narrative context for the quantitative sections below: what was built, in
# what order, and why -- so a reader lands on the charts already knowing
# what question each one is answering. Deliberately static prose (not
# computed from report data) since it describes the overall process rather
# than any one run's results; update it by hand as new phases land.
OVERVIEW_HTML = """<section id="overview">
<h2>Overview</h2>
<p class="chart-note">This project applies autonomous-vehicle perception techniques to the
<a href="https://www.nuscenes.org/">nuScenes</a> dataset, building
up a data pipeline one validated stage at a time rather than assembling it all at once and debugging
from the end. The two sections below are the quantitative output of that process so far; this
overview gives the story that connects them.</p>

<h3>What's been built, in order</h3>
<ul class="chart-note" style="margin:4px 0 8px 20px; padding:0;">
<li><strong>Foundations.</strong> Before writing any pipeline code: camera calibration /
projective geometry (how a 3D point maps to a pixel) and SE(3) rigid-body transforms
(representing and composing 3D poses/rotations) -- the math every later stage depends on.</li>
<li><strong>Schema exploration + single-machine ingestion.</strong> Walked the nuScenes schema
(scenes, samples, sensor calibration, ego pose) via the devkit, then synced the 6 camera streams
per sample by nearest timestamp, thresholded + visualized the resulting skew, and wrote the
synced records plus ego pose/calibration out to JSON.</li>
<li><strong>Ray-parallelized ingestion.</strong> Re-ran that ingestion under
<a href="https://www.ray.io/">Ray</a> across a range of worker counts, to see whether -- and how
much -- parallelizing it actually helps, and where the cost of parallelism itself shows up. The
<a href="#performance">Performance</a> section below is that benchmark.</li>
<li><strong>Pixel &rarr; ego &rarr; world projection, validated.</strong> Implemented the SE(3)
chain that projects a 3D point from camera space through ego space into world space (and its
inverse, pixel &rarr; 3D), then checked it against ground truth by reprojecting each labeled 3D box
back into every camera it's visible in and comparing the result to nuScenes' own 2D box via IOU.
The <a href="#correctness">Correctness</a> section below is that validation.</li>
</ul>

<h3>What this has shown so far</h3>
<ul class="chart-note" style="margin:4px 0 8px 20px; padding:0;">
<li><strong>Parallelism has a floor, not just a ceiling.</strong> Throughput improves with more
workers only up to a point set by the machine's actual core count and the fixed per-worker cost
(interpreter startup, imports, <code>ray.init()</code>) -- see the Performance section's
per-chart Observations for the specific numbers on this machine.</li>
<li><strong>The projection math checks out.</strong> On real, non-degenerate boxes the reprojected
box agrees with nuScenes' reference box to well under a pixel (IOU far above 0.99). Where IOU drops,
it's almost always because the box itself is tiny or edge-clipped (a few px&sup2;), where IOU is
inherently sensitive to sub-pixel differences -- not a flaw in the projection. That's also why the
area cutoff used elsewhere in this pipeline is a configurable setting
(<code>config.toml</code>'s <code>min_bbox_area</code>) rather than a hardcoded constant: the right
floor depends on the dataset's own area distribution.</li>
<li><strong>Measure before explaining.</strong> Every number quoted in the sections below --
startup overhead, speedup, wait-time spread, CPU utilization, IOU by category/camera/size -- is
computed from the pipeline's own run data each time this report is rebuilt, not hand-picked, so the
explanations stay accurate if re-run on different hardware or a different sweep.</li>
</ul>
</section>"""

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

    nav_links = " ".join(
        [f'<a href="#overview">Overview</a>']
        + [f'<a href="#{anchor}">{label}</a>' for anchor, label, _ in fragments]
    )
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
{OVERVIEW_HTML}
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
