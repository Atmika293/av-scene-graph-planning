"""The Experiments series: one registry entry per article, and the index page's cards.

Each article is its own page (experiment-<slug>.html), built by report.site
from EXPERIMENTS: a hand-written content/<content> article, plus an optional
generated data fragment under docs/. The Experiments index page lists every
entry here as a card, so adding an article is one markdown file plus one
entry below; the index can't fall out of step with the pages.

See .claude/skills/experiment-article/SKILL.md for how to write one.
"""

from html import escape
from typing import NamedTuple

from report.markup import expand_directives


class Experiment(NamedTuple):
    """One article in the series, in the order it should appear on the index."""
    slug: str  # page is experiment-<slug>.html
    title: str  # the question the article asks, as a short heading
    summary: str  # one or two sentences for the index card (may use {{stat:...}})
    content: str  # article file name under report/content/
    dates: str  # paths for {{dates:...}}: the article's date range from feat:/fix: commits
    outcome: str = ""  # optional short verdict for the card, e.g. "Negligible"; "" for none
    fragment: str | None = None  # optional generated data section, path under docs/

    @property
    def filename(self) -> str:
        """The article's page file name."""
        return f"experiment-{self.slug}.html"


EXPERIMENTS = [
    Experiment(
        slug="ray-task-layout",
        title="Did consolidating Ray tasks matter?",
        summary="Two tasks per batch instead of four: at most {{stat:task_layout.diff_pct_max|.1%}} "
                "faster after startup. Nearly all task time is in bbox projection, which neither "
                "layout touches.",
        content="experiments_task_layout.md",
        dates="benchmark_task_layout.py",
        outcome="Negligible",
        fragment="_task_layout_section.html",
    ),
]


def render_index() -> str:
    """The index page's cards, newest-registered last, as one <section>."""
    cards = []
    for experiment in EXPERIMENTS:
        outcome = (f' <span class="badge diagnostic">{escape(experiment.outcome)}</span>'
                   if experiment.outcome else "")
        date = expand_directives(f"{{{{dates: {experiment.dates}}}}}")
        cards.append(
            f'<a class="card" href="{experiment.filename}"><strong>{escape(experiment.title)}{outcome}</strong>'
            f"<span>{expand_directives(experiment.summary)}</span>"
            f'<span class="meta">{date}</span></a>'
        )
    return '<section id="experiments-index">\n<div class="cards">\n' + "\n".join(cards) + "\n</div>\n</section>"
