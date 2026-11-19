"""The Papers series: one registry entry per paper, and the index page's cards.

Each paper with notes gets its own page (paper-<slug>.html), built by
report.site from PAPERS: a hand-written content/<content> page that opens with
a summary of the paper and then my own notes. The Papers index lists every
entry as a card, grouped by the phase I read it for, so the reading stays tied
to what I was building at the time. A paper I haven't started has no content
yet: its card shows on the index but links straight to the paper instead.
"""

from html import escape
from typing import NamedTuple

# The phases papers can belong to, in index order: (anchor, heading).
PHASES = [
    ("phase-2", "Phase 2: Ingestion + fusion"),
]

# Status badge per paper status: (badge class, label), matching the Learning page's badges.
STATUS_BADGES = {
    "done": ("done", "Read"),
    "reading": ("progress", "Reading"),
    "todo": ("todo", "Up next"),
}


class Paper(NamedTuple):
    """One paper, in the order it should appear within its phase."""
    slug: str  # page is paper-<slug>.html
    title: str  # the paper's short name, e.g. "Lift-Splat-Shoot"
    authors: str  # e.g. "Philion & Fidler"
    venue: str  # e.g. "ECCV 2020"
    url: str  # the paper itself; "" if no link yet
    phase: str  # anchor from PHASES
    status: str  # key of STATUS_BADGES
    summary: str  # one sentence for the index card: what the paper does
    content: str | None = None  # page file name under report/content/; None until I have notes

    @property
    def filename(self) -> str:
        """The paper's page file name."""
        return f"paper-{self.slug}.html"


PAPERS = [
    Paper(
        slug="lss",
        title="Lift-Splat-Shoot",
        authors="Philion & Fidler",
        venue="ECCV 2020",
        url="https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123590188.pdf",
        phase="phase-2",
        status="reading",
        summary="A bird's-eye view from cameras alone: each pixel is lifted along its ray with a "
                "predicted depth distribution, then splatted onto a BEV grid.",
        content="paper_lss.md",
    ),
    Paper(
        slug="bevformer",
        title="BEVFormer",
        authors="Li et al.",
        venue="ECCV 2022",
        url="",
        phase="phase-2",
        status="todo",
        summary="Camera-only BEV built with attention over space and time instead of explicit depth.",
    ),
    Paper(
        slug="bevfusion",
        title="BEVFusion",
        authors="Liang et al.",
        venue="NeurIPS 2022",
        url="",
        phase="phase-2",
        status="todo",
        summary="Camera and LiDAR features fused in a shared BEV space.",
    ),
    Paper(
        slug="bev-survey",
        title="Delving into the Devils of Bird's-eye-view Perception",
        authors="Li et al.",
        venue="T-PAMI 2023",
        url="https://arxiv.org/abs/2209.05324",
        phase="phase-2",
        status="todo",
        summary="A survey of BEV perception from cameras, LiDAR and both.",
    ),
]


def badge(paper: Paper) -> str:
    """The paper's status badge."""
    css, label = STATUS_BADGES[paper.status]
    return f'<span class="badge {css}">{label}</span>'


def render_index() -> str:
    """The index page's cards, one group per phase, as one <section>.

    A paper with a page links to it; one without links to the paper itself, or isn't a link
    at all if there's no URL yet.
    """
    groups = []
    for anchor, heading in PHASES:
        cards = []
        for paper in (p for p in PAPERS if p.phase == anchor):
            inner = (f"<strong>{escape(paper.title)} {badge(paper)}</strong>"
                     f"<span>{escape(paper.summary)}</span>"
                     f'<span class="meta">{escape(paper.authors)} · {escape(paper.venue)}</span>')
            href = paper.filename if paper.content else paper.url
            cards.append(f'<a class="card" href="{escape(href)}">{inner}</a>' if href
                         else f'<div class="card">{inner}</div>')
        if cards:
            groups.append(f'<h2 id="{anchor}">{escape(heading)}</h2>\n<div class="cards">\n'
                          + "\n".join(cards) + "\n</div>")
    return '<section id="papers-index" class="prose">\n' + "\n".join(groups) + "\n</section>"
