"""Render the hand-written content/*.md pages, expanding two {{...}} directives first.

    {{stat:projection.iou_min|.4f}}   a number from report.stats.all_stats(),
                                      looked up by dotted path and formatted
                                      with a format spec (after the "|"), so
                                      prose never quotes a hand-typed number
    {{dates:path/a path/b ...}}       the date range of the feat:/fix: commits
                                      touching any of those paths

Dates come only from feat:/fix: commits, so a refactor: (e.g. moving files
between directories) or a docs: commit (republishing the report) doesn't
stretch a milestone's range. List paths that belong to that milestone
alone -- a file several milestones keep editing (transform.py,
ingestion_keys.py) would stretch each one to the last edit -- and include a
file's pre-rename path too, since git log on a path doesn't follow renames
(e.g. src/av_scene_graph_planning/ before the av_data_pipeline rename).

An unknown stat path or a malformed directive raises rather than rendering
blank, so a typo fails the build instead of publishing an empty number. A
stat whose report hasn't been generated yet renders as "n/a".

HTML comments (<!-- ... -->) are stripped before rendering, so content files
can carry editorial notes (e.g. "DRAFT: check this") that never reach the
published page source.
"""

import re
import subprocess
from datetime import date

import markdown

from report.stats import all_stats

DIRECTIVE_RE = re.compile(r"\{\{(stat|dates):\s*([^}]*?)\s*\}\}")
COMMENT_RE = re.compile(r"<!--.*?-->\n?", re.DOTALL)
MARKDOWN_EXTENSIONS = ["tables", "attr_list", "md_in_html", "fenced_code", "toc"]


def _stat(arg: str) -> str:
    path, _, spec = arg.partition("|")
    value = all_stats()
    for key in path.split("."):
        if value is None:
            return "n/a"
        if key not in value:
            raise SystemExit(f"Unknown stat path in content: {{{{stat:{arg}}}}}")
        value = value[key]
    return "n/a" if value is None else format(value, spec)


def _format_range(first: date, last: date) -> str:
    if first == last:
        return f"{first:%b} {first.day}, {first.year}"
    if first.year == last.year:
        return f"{first:%b} {first.day} – {last:%b} {last.day}, {last.year}"
    return f"{first:%b} {first.day}, {first.year} – {last:%b} {last.day}, {last.year}"


def _dates(arg: str) -> str:
    paths = arg.split()
    if not paths:
        raise SystemExit("{{dates:}} needs at least one path")
    log = subprocess.run(
        ["git", "log", "-E", "--grep=^(feat|fix)(\\(|:)", "--format=%ad", "--date=short", "--", *paths],
        capture_output=True, text=True, check=True,
    ).stdout.split()
    if not log:
        return "Not yet committed"
    days = sorted(date.fromisoformat(d) for d in log)
    return _format_range(days[0], days[-1])


def expand_directives(text: str) -> str:
    """Replace every {{stat:...}}/{{dates:...}} directive with its value."""
    def replace(match: re.Match) -> str:
        kind, arg = match.groups()
        return _stat(arg) if kind == "stat" else _dates(arg)
    return DIRECTIVE_RE.sub(replace, text)


def render_markdown(text: str) -> str:
    """Strip comments, expand directives, then render Markdown (tables, {: .class} attributes, Markdown inside HTML blocks)."""
    return markdown.markdown(expand_directives(COMMENT_RE.sub("", text)), extensions=MARKDOWN_EXTENSIONS)
