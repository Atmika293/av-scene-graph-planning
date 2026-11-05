"""Shared HTML for the published site: the KaTeX snippet, the site-wide styles, and the page shell.

KATEX_CDN_HTML is used by benchmark.py and build_correctness_report.py (each
embeds one copy in its own fragment so the fragment is viewable standalone),
and by report.site (via build_report.py), which strips every embedded copy
back out and keeps a single one in each page's <head> -- the same pattern
already used for each fragment's embedded plotly.js CDN <script> tag.

Chart-note text that includes a formula should wrap it in \\( ... \\) for
inline math or \\[ ... \\] for a centered, standalone equation (KaTeX's
auto-render delimiters, configured below).

Light theme only, deliberately: every Plotly figure and the BEV raster's
empty cells assume a white background, so a dark page would put the charts
in white boxes rather than actually theme them.
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

SITE_NAME = "AV Perception: nuScenes Pipeline"
REPO_URL = "https://github.com/Atmika293/av-scene-graph-planning"

SITE_CSS = """
:root {
  --ink: #1a1d21; --muted: #5b6470; --faint: #8a929c;
  --line: #e3e6ea; --surface: #f6f7f9; --bg: #ffffff;
  --accent: #1f5fbf; --accent-soft: #e8f0fb;
  --pass: #1d7a46; --pass-soft: #e5f4ec;
  --info: #8a5a00; --info-soft: #fbf1dc;
  --todo: #6b7280; --todo-soft: #eef0f2;
  /* A ceiling, not a target width: used as max-width (not width) below, so
     a block still fills 100% of its container and shrinks to fit a narrow
     screen exactly like the browser's own default behavior, only capping
     once a screen is wide enough to hit this value. An earlier version
     used width: min(94vw, 1400px) here, which reserved a proportional
     margin on every screen, including phones that never needed one --
     confirmed by comparing rendered widths at common phone sizes, not just
     assumed. Shared by the header/main/footer so they can't drift apart. */
  --page-width: 1400px;
}
* { box-sizing: border-box; }
html { -webkit-text-size-adjust: 100%; }
body { margin: 0; background: var(--bg); color: var(--ink);
  font: 16px/1.6 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; }
a { color: var(--accent); }
code { font: 0.88em ui-monospace, SFMono-Regular, Menlo, monospace; background: var(--surface);
  padding: 1px 5px; border-radius: 4px; }

.site-header { border-bottom: 1px solid var(--line); background: var(--bg); position: sticky; top: 0; z-index: 10; }
.site-header .inner { max-width: var(--page-width); margin: 0 auto; padding: 12px 16px;
  display: flex; flex-wrap: wrap; align-items: baseline; gap: 6px 24px; }
.brand { font-weight: 700; color: var(--ink); text-decoration: none; letter-spacing: -0.01em; }
nav.pages { display: flex; flex-wrap: wrap; gap: 4px 18px; }
nav.pages a { color: var(--muted); text-decoration: none; font-weight: 500; padding: 2px 0; }
nav.pages a:hover { color: var(--ink); }
nav.pages a[aria-current] { color: var(--ink); border-bottom: 2px solid var(--accent); }

main { max-width: var(--page-width); margin: 0 auto; padding: 28px 16px 48px; }
nav.sections { margin: 0 0 24px; display: flex; flex-wrap: wrap; gap: 4px 16px; font-size: 0.92em; }
nav.sections::before { content: "On this page:"; color: var(--faint); }
nav.sections a { text-decoration: none; font-weight: 600; }

h1 { font-size: 1.9em; line-height: 1.2; letter-spacing: -0.02em; margin: 0 0 12px; }
h2 { font-size: 1.4em; line-height: 1.3; margin: 40px 0 12px; padding-bottom: 6px; border-bottom: 1px solid var(--line); }
h3 { font-size: 1.12em; margin: 28px 0 8px; }
h4 { font-size: 1em; margin: 20px 0 6px; }
section { scroll-margin-top: 72px; }
h2, h3 { scroll-margin-top: 72px; }

/* Hand-written (Markdown) content: readable measure, left-aligned tables. */
.prose { max-width: 760px; }
.prose p, .prose li { color: var(--ink); }
.prose .lede { font-size: 1.12em; color: var(--muted); }
.prose table { width: 100%; }
.prose th, .prose td { text-align: left; }
.prose blockquote { margin: 16px 0; padding: 4px 16px; border-left: 3px solid var(--line); color: var(--muted); }
.prose img { max-width: 100%; }
@media (max-width: 640px) { .prose table { display: block; overflow-x: auto; } }
/* position: relative so KaTeX's absolutely-positioned (visually hidden) MathML copy is clipped too. */
.katex-display { position: relative; overflow-x: auto; overflow-y: hidden; padding: 2px 0; }

/* Generated fragments (benchmark.py, build_correctness_report.py, ...). */
table { border-collapse: collapse; margin: 12px 0; font-size: 0.94em; }
th, td { border: 1px solid var(--line); padding: 6px 10px; text-align: right; vertical-align: top; }
th { background: var(--surface); font-weight: 600; }
figure { display: inline-block; vertical-align: top; }
img { max-width: 100%; height: auto; }
.chart-note { max-width: 760px; color: var(--muted); font-size: 0.92em; line-height: 1.55; margin: 16px 0 4px 0; }
.js-plotly-plot, .plotly-graph-div { max-width: 100%; }

/* Small shared components used from the Markdown pages. */
.meta { color: var(--faint); font-size: 0.9em; margin: -4px 0 12px; }
.badge { display: inline-block; font-size: 0.75em; font-weight: 600; padding: 1px 8px; border-radius: 999px;
  vertical-align: middle; margin-left: 6px; white-space: nowrap; }
.badge.done, .badge.pass { background: var(--pass-soft); color: var(--pass); }
.badge.progress, .badge.diagnostic { background: var(--info-soft); color: var(--info); }
.badge.todo, .badge.notrun { background: var(--todo-soft); color: var(--todo); }
.badge.fail { background: #fdecec; color: #b42318; }

.skill-pills { display: flex; flex-wrap: wrap; gap: 8px; margin: 18px 0 6px; }
.skill-pills .pill { display: inline-block; padding: 5px 14px; border-radius: 999px;
  background: var(--accent-soft); color: var(--accent); font-size: 0.88em; font-weight: 600; }

.tags { margin: 4px 0 0; }
.tags .tag { display: inline-block; padding: 1px 9px; border-radius: 999px; border: 1px solid var(--line);
  color: var(--muted); font-size: 0.78em; font-weight: 600; margin: 2px 4px 2px 0; }

.callout { background: var(--surface); border: 1px solid var(--line); border-radius: 8px; padding: 12px 16px; margin: 16px 0; }
.callout > :first-child { margin-top: 0; } .callout > :last-child { margin-bottom: 0; }

.stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; margin: 20px 0 28px; max-width: none; }
.stat { border: 1px solid var(--line); border-radius: 10px; padding: 14px 16px; }
.stat .value { font-size: 1.7em; font-weight: 700; letter-spacing: -0.02em; line-height: 1.1; font-variant-numeric: tabular-nums; }
.stat .label { color: var(--muted); font-size: 0.88em; margin-top: 6px; line-height: 1.4; }

.cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 12px; margin: 16px 0; }
.card { display: block; border: 1px solid var(--line); border-radius: 10px; padding: 14px 16px; text-decoration: none; color: var(--ink); }
.card:hover { border-color: var(--accent); background: var(--accent-soft); }
.card strong { display: block; color: var(--accent); margin-bottom: 4px; }
.card span { color: var(--muted); font-size: 0.92em; line-height: 1.45; display: block; }

.progress-strip { display: grid; grid-template-columns: repeat(7, 1fr); gap: 4px; margin: 16px 0 6px; }
.progress-strip div { font-size: 0.78em; text-align: center; padding: 8px 4px; border-radius: 6px;
  background: var(--todo-soft); color: var(--todo); line-height: 1.25; }
.progress-strip .done { background: var(--pass-soft); color: var(--pass); }
.progress-strip .current { background: var(--accent); color: #fff; font-weight: 600; }
@media (max-width: 640px) { .progress-strip { grid-template-columns: repeat(4, 1fr); } }

/* Validation page: chain of trust + per-stage check tables. */
.chain { display: flex; flex-wrap: wrap; align-items: stretch; gap: 8px; margin: 20px 0 8px; }
.chain .link { flex: 1 1 180px; border: 1px solid var(--line); border-radius: 10px; padding: 12px 14px; text-decoration: none; color: var(--ink); }
.chain .link:hover { border-color: var(--accent); }
.chain .link .n { color: var(--faint); font-size: 0.8em; font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em; }
.chain .link .name { font-weight: 700; margin: 2px 0; }
.chain .link .sub { color: var(--muted); font-size: 0.88em; }
.chain .arrow { align-self: center; color: var(--faint); font-size: 1.3em; }
@media (max-width: 640px) { .chain .arrow { display: none; } }
.checks { width: 100%; }
.checks th, .checks td { text-align: left; }
.checks td.result { font-variant-numeric: tabular-nums; }
.table-scroll { overflow-x: auto; }

.site-footer { border-top: 1px solid var(--line); color: var(--faint); font-size: 0.88em; }
.site-footer .inner { max-width: var(--page-width); margin: 0 auto; padding: 16px; }
"""


def page_shell(title: str, page_nav: str, section_nav: str, body: str, head_extra: str) -> str:
    """Wrap one page's body in the site-wide <head>, header, and footer.

    Args:
        title: The page's <title>.
        page_nav: The site-wide page links (<a> tags), already marked with
            aria-current for this page.
        section_nav: This page's in-page anchor links, or "" if it has fewer
            than two sections.
        body: The page's sections, in order.
        head_extra: Extra <head> content, e.g. a hoisted plotly.js tag.

    Returns:
        The page's full HTML document.
    """
    section_nav_html = f'<nav class="sections">{section_nav}</nav>\n' if section_nav else ""
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
{head_extra}
{KATEX_CDN_HTML}
<style>{SITE_CSS}</style>
</head>
<body>
<header class="site-header"><div class="inner">
<a class="brand" href="index.html">{SITE_NAME}</a>
<nav class="pages">{page_nav}</nav>
</div></header>
<main>
{section_nav_html}{body}
</main>
<footer class="site-footer"><div class="inner">
Source: <a href="{REPO_URL}">{REPO_URL.removeprefix("https://")}</a> &middot;
every number on this site is computed from the pipeline's own run output each time it's rebuilt.
</div></footer>
</body>
</html>
"""
