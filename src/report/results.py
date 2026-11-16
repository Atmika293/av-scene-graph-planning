"""The Results series: one registry entry per result page, and the index page's cards.

Each result is its own page (result-<slug>.html), built by report.site from
RESULTS: a short hand-written content/<content> header (title, links, a
one-paragraph summary) followed by the result's generated fragment under
docs/. The Results index lists every entry as a card. Same pattern as
report.experiments, but for measurements of the pipeline as built rather
than questions about a decision.
"""

from html import escape
from typing import NamedTuple

from report.markup import expand_directives


class Result(NamedTuple):
    """One result page, in the order it should appear on the index."""
    slug: str  # page is result-<slug>.html
    title: str  # short name of what was measured
    summary: str  # one or two sentences for the index card (may use {{stat:...}})
    content: str  # header file name under report/content/
    dates: str  # paths for {{dates:...}}: the result's date range from feat:/fix: commits
    fragment: str  # generated section, path under docs/
    anchor: str  # the fragment's own <section id>, for the in-page nav
    # Hand-written date shown instead of {{dates:...}}, for when a later commit
    # to the same files belongs to something else; "" to use dates.
    date: str = ""

    @property
    def date_html(self) -> str:
        """The date line for the card and the page header."""
        return escape(self.date) if self.date else expand_directives(f"{{{{dates: {self.dates}}}}}")

    @property
    def filename(self) -> str:
        """The result's page file name."""
        return f"result-{self.slug}.html"


RESULTS = [
    Result(
        slug="ray-scaling",
        title="Ray scaling benchmark",
        summary="How ingestion's wall time changes with the number of Ray workers: at best "
                "{{stat:benchmark.best_speedup|.2f}}× faster, at {{stat:benchmark.best_workers}} "
                "workers, and why it stops there.",
        content="result_ray_scaling.md",
        dates="benchmark.py",
        fragment="_performance_section.html",
        anchor="performance",
        # benchmark.py's later feat: commit is the task-layout experiment's.
        date="Nov 1, 2026",
    ),
    Result(
        slug="projection-accuracy",
        title="Projection accuracy",
        summary="How closely the 3D to 2D box projection reproduces nuScenes' reference boxes: "
                "every one of {{stat:projection.iou_count|,}} annotation and camera pairs at IoU "
                "{{stat:projection.iou_min|.4f}} or higher.",
        content="result_projection_accuracy.md",
        dates="scripts/build_correctness_report.py scripts/report/build_correctness_report.py",
        fragment="_correctness_section.html",
        anchor="correctness",
    ),
]


def render_index() -> str:
    """The index page's cards, in registry order, as one <section>."""
    cards = []
    for result in RESULTS:
        date = result.date_html
        cards.append(
            f'<a class="card" href="{result.filename}"><strong>{escape(result.title)}</strong>'
            f"<span>{expand_directives(result.summary)}</span>"
            f'<span class="meta">{date}</span></a>'
        )
    return '<section id="results-index">\n<div class="cards">\n' + "\n".join(cards) + "\n</div>\n</section>"
