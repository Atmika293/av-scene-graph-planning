"""Assemble the published site (docs/*.html) from hand-written content and generated fragments.

The site is a list of PAGES. Each page is a sequence of sections, each one
of three kinds:

    content    a hand-written content/*.md page (always present), rendered
               via report.markup -- prose, with {{stat:...}}/{{dates:...}}
               directives filled in from the latest run data and git history
    fragment   a generated docs/_*.html section from a script that needs the
               dataset to run (benchmark.py, build_correctness_report.py,
               build_bev_report.py) -- optional: skipped if not built yet
    render     generated at site-build time from reports/ (e.g.
               report.validation.render, the Validation page's check table)

Every page gets the same single <head> (one plotly.js CDN load and one KaTeX
CDN load, not one per fragment -- each fragment embeds its own copy of both
so it's independently viewable while being built/debugged in isolation, but
the combined page only needs one of each), the site header and page nav,
an in-page nav when it has more than one section, and its sections in order,
fragments unmodified except for stripping their embedded CDN tags.

A page is built if at least one of its non-intro sections is available, so
a results page whose fragments haven't been generated yet is left out (and
unlinked) rather than published as an intro with nothing under it.

Hand-written content lives next to this module rather than under docs/, so
docs/ holds only generated output and can be rebuilt from scratch.
"""

import os
import re
from typing import Callable, NamedTuple

from report import validation
from report.html import KATEX_CDN_HTML, page_shell
from report.markup import render_markdown

REPORT_DIR = "docs"
CONTENT_DIR = os.path.join(os.path.dirname(__file__), "content")


class Section(NamedTuple):
    """One block of a page. Exactly one of content/fragment/render is set."""
    anchor: str
    label: str
    content: str | None = None  # file name under CONTENT_DIR
    fragment: str | None = None  # path under REPORT_DIR; optional
    render: Callable[[], str] | None = None
    intro: bool = False  # doesn't count toward whether the page gets built


class Page(NamedTuple):
    filename: str
    label: str
    title: str
    sections: list[Section]


PAGES = [
    Page("index.html", "Home", "AV Perception: nuScenes Pipeline", [
        Section("home", "Home", content="home.md"),
    ]),
    Page("learning.html", "Learning", "Learning log · AV Perception", [
        Section("learning", "Learning log", content="learning.md"),
    ]),
    Page("build.html", "Build log", "Build log · AV Perception", [
        Section("build", "Build log", content="build.md"),
    ]),
    Page("validation.html", "Validation", "Validation · AV Perception", [
        Section("validation-intro", "Approach", content="validation.md", intro=True),
        Section("checks", "Checks", render=validation.render),
        Section("validation-notes", "Design notes", content="validation_notes.md", intro=True),
    ]),
    Page("results.html", "Results", "Ingestion & projection results · AV Perception", [
        Section("results-intro", "About", content="results.md", intro=True),
        Section("performance", "Performance", fragment=os.path.join(REPORT_DIR, "_performance_section.html")),
        Section("correctness", "Correctness", fragment=os.path.join(REPORT_DIR, "_correctness_section.html")),
    ]),
    Page("bev.html", "BEV", "Bird's-eye-view fusion · AV Perception", [
        Section("bev-intro", "About", content="bev.md", intro=True),
        Section("bev", "Views", fragment=os.path.join(REPORT_DIR, "_bev_section.html")),
    ]),
    Page("roadmap.html", "Roadmap", "Roadmap · AV Perception", [
        Section("roadmap", "Roadmap", content="roadmap.md"),
    ]),
]

# Matches a fragment's embedded plotly.js CDN <script> tag regardless of
# version, so this doesn't go stale if the installed plotly library (and so
# the version plotly.io embeds) is upgraded later.
PLOTLYJS_SCRIPT_RE = re.compile(r'<script charset="utf-8" src="https://cdn\.plot\.ly/[^"]+"[^>]*></script>')


def load_section(section: Section) -> str | None:
    """One section's HTML, or None if it's a fragment that hasn't been generated yet."""
    if section.content:
        with open(os.path.join(CONTENT_DIR, section.content)) as fp:
            body = render_markdown(fp.read())
        return f'<section id="{section.anchor}" class="prose">\n{body}\n</section>'
    if section.render:
        return section.render()
    if not os.path.exists(section.fragment):
        print(f"Skipping missing section: {section.fragment}")
        return None
    with open(section.fragment) as fp:
        return fp.read()


def extract_plotlyjs_script(sections_html: list[str]) -> str:
    """Find one section's embedded plotly.js CDN script tag, to hoist into <head>.

    Args:
        sections_html: The page's loaded sections.

    Returns:
        The first plotly.js CDN <script> tag found across the sections, or ""
        if no section uses Plotly (e.g. a prose-only page).

    Raises:
        SystemExit: If a section uses Plotly but none contains the CDN tag --
            the combined page's charts wouldn't render at all.
    """
    for html in sections_html:
        match = PLOTLYJS_SCRIPT_RE.search(html)
        if match:
            return match.group(0)

    if any("Plotly." in html for html in sections_html):
        raise SystemExit("A section fragment uses Plotly but none embeds its plotly.js CDN "
                         "<script> tag -- cannot assemble a page whose charts would render.")
    return ""


def build_page(page: Page, sections: list[tuple[Section, str]], built_pages: list[Page]) -> str:
    """Build one page: the shared shell around its available sections, in order.

    Args:
        page: The page being built.
        sections: (section, html) for each of the page's available sections.
        built_pages: Every page being built this run, for the page-level nav.

    Returns:
        The page's full HTML contents.
    """
    sections_html = [html for _, html in sections]
    plotlyjs_script = extract_plotlyjs_script(sections_html)

    page_nav = " ".join(
        f'<a href="{p.filename}" aria-current="page">{p.label}</a>' if p == page
        else f'<a href="{p.filename}">{p.label}</a>'
        for p in built_pages
    )
    section_nav = ""
    if len(sections) > 1:
        section_nav = " ".join(f'<a href="#{s.anchor}">{s.label}</a>' for s, _ in sections)

    # Each fragment keeps its own embedded copy of both CDN resources for
    # standalone viewing; strip them here since the combined page loads
    # plotly.js and KaTeX once each, in <head>. KATEX_CDN_HTML is a known,
    # exact, shared constant (unlike plotly's version-varying script tag),
    # so a plain string removal is enough -- no regex needed.
    body = "\n".join(
        PLOTLYJS_SCRIPT_RE.sub("", html, count=1).replace(KATEX_CDN_HTML, "")
        for html in sections_html
    )

    return page_shell(page.title, page_nav, section_nav, body, plotlyjs_script)


def build_site() -> dict[str, str]:
    """Build every page that has at least one non-intro section available.

    Returns:
        {output path under REPORT_DIR: page HTML}, in PAGES order.
    """
    loaded = []
    for page in PAGES:
        sections = [(s, html) for s in page.sections if (html := load_section(s)) is not None]
        if any(not s.intro for s, _ in sections):
            loaded.append((page, sections))
    built_pages = [page for page, _ in loaded]

    return {
        os.path.join(REPORT_DIR, page.filename): build_page(page, sections, built_pages)
        for page, sections in loaded
    }
