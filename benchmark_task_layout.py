"""Benchmark: 2 tasks per batch (consolidated) vs. 4 tasks per batch (split).

partition.py writes each batch with two Ray tasks: one for all three core
tables, one for the projected bboxes. The earlier design gave every output
table its own task (four per batch). This script measures whether that
consolidation actually helped, under the same protocol as benchmark.py: it
reuses benchmark.run_partition_once (same subprocess isolation, same
BATCH_SIZE/MAX_IN_FLIGHT overrides, same timing and CPU sampling), changing
only TASK_LAYOUT.

Layouts are interleaved within each (worker count x repeat), and which one
goes first alternates by repeat, so slow drift over the sweep (thermal
throttling, background load) lands on both layouts equally instead of
favouring whichever ran first.

Writes one JSON line per run to reports/task_layout_results.jsonl (kept
separate from benchmark.py's reports/benchmark_results.jsonl, which the
site's headline numbers read), then BENCHMARK_REPORT_DIR/_task_layout_section.html,
an HTML fragment for the site's Experiments page: a comparison table, a chart,
and a table of where the task compute goes. The raw results are also copied
to BENCHMARK_REPORT_DIR/experiments_data/ so the page can link them (reports/
itself isn't published).

Run with: python benchmark_task_layout.py (from the repo root).
For a tighter comparison: BENCHMARK_REPEATS=5 python benchmark_task_layout.py
To rebuild the fragment from the last sweep without rerunning it:
python benchmark_task_layout.py --report-only
"""

import json
import os
import shutil
import statistics
import sys

import plotly.graph_objects as go

from av_data_pipeline.config import (
    BENCHMARK_WORKER_COUNTS, BENCHMARK_REPEATS, BENCHMARK_REPORT_DIR,
)
from benchmark import run_partition_once
from report.html import REPO_URL

RESULTS_PATH = "reports/task_layout_results.jsonl"
PUBLISHED_DATA_DIR = "experiments_data"  # under BENCHMARK_REPORT_DIR
LAYOUTS = ("consolidated", "split")
LAYOUT_LABELS = {"consolidated": "2 tasks/batch (current)", "split": "4 tasks/batch (earlier design)"}


def run_sweep() -> list[dict]:
    """Run every (worker count x repeat x layout), writing results incrementally.

    Returns:
        Every run's result record, in the order they completed.
    """
    os.makedirs("reports", exist_ok=True)
    results = []

    with open(RESULTS_PATH, "w") as results_file:
        for worker_count in BENCHMARK_WORKER_COUNTS:
            for repeat in range(BENCHMARK_REPEATS):
                # Alternate which layout goes first, so neither one always
                # runs on a machine the other has just warmed up.
                order = LAYOUTS if repeat % 2 == 0 else LAYOUTS[::-1]
                for layout in order:
                    print(f"Running worker_count={worker_count} repeat={repeat} layout={layout}...")
                    result = run_partition_once(worker_count, repeat, task_layout=layout)
                    results.append(result)

                    results_file.write(json.dumps(result) + "\n")
                    results_file.flush()

    return results


def load_results() -> list[dict]:
    """The last sweep's result records, read back from RESULTS_PATH."""
    with open(RESULTS_PATH) as results_file:
        return [json.loads(line) for line in results_file if line.strip()]


def _mean_stdev(values: list[float]) -> tuple[float, float]:
    """Mean and sample stdev (0 for a single value)."""
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def is_bbox_task(task: dict) -> bool:
    """Whether a task_timings entry is a batch's bbox task (labels end in the task name)."""
    return task["label"].endswith("write_bboxes")


def aggregate(results: list[dict]) -> dict[str, dict[int, dict]]:
    """Summarize every successful run, grouped by layout then worker count.

    Args:
        results: Every run's result record, as produced by run_sweep.

    Returns:
        layout -> worker_count -> {n, wall (mean, stdev), work (mean, stdev),
        bbox_compute, core_compute}. "work" is wall time minus startup
        overhead: the part of a run after the first task starts, which is
        the only part the task layout can affect. bbox_compute and
        core_compute are the mean per-run compute seconds of the bbox task(s)
        and of the core-table task(s), from each task's own label.
    """
    summary: dict[str, dict[int, dict]] = {layout: {} for layout in LAYOUTS}

    for layout in LAYOUTS:
        for worker_count in BENCHMARK_WORKER_COUNTS:
            runs = [r for r in results if r["task_layout"] == layout
                    and r["worker_count"] == worker_count and r["success"]
                    and r["startup_overhead"] is not None]
            if not runs:
                continue

            summary[layout][worker_count] = {
                "n": len(runs),
                "wall": _mean_stdev([r["wall_time"] for r in runs]),
                "work": _mean_stdev([r["wall_time"] - r["startup_overhead"] for r in runs]),
                "bbox_compute": statistics.mean(
                    sum(t["compute_time"] for t in r["task_timings"] if is_bbox_task(t)) for r in runs),
                "core_compute": statistics.mean(
                    sum(t["compute_time"] for t in r["task_timings"] if not is_bbox_task(t)) for r in runs),
            }

    return summary


def _worker_counts(summary: dict[str, dict[int, dict]]) -> list[int]:
    """Worker counts with successful runs under both layouts."""
    return sorted(set(summary["consolidated"]) & set(summary["split"]))


def build_bar_chart(summary: dict[str, dict[int, dict]], metric: str, title: str) -> go.Figure:
    """Grouped bar of one metric ('wall' or 'work') by worker count, one bar per layout, stdev error bars."""
    worker_counts = _worker_counts(summary)
    fig = go.Figure()
    for layout in LAYOUTS:
        fig.add_bar(name=LAYOUT_LABELS[layout], x=worker_counts,
                   y=[summary[layout][wc][metric][0] for wc in worker_counts],
                   error_y={"type": "data", "array": [summary[layout][wc][metric][1] for wc in worker_counts]})

    fig.update_layout(barmode="group", title=title,
                      xaxis_title="Worker count", yaxis_title="Seconds (mean, stdev bars)")
    # Sweep values, not a continuous range -- see benchmark.py's breakdown chart.
    fig.update_xaxes(type="category")
    return fig


def _verdict(cons: dict, split: dict) -> str:
    """Which layout was faster after startup, or "within noise".

    "Within noise" means the difference is smaller than the two layouts'
    stdevs added together: a rough rule of thumb for a handful of repeats,
    not a significance test, and stated as such on the page.
    """
    diff = split["work"][0] - cons["work"][0]
    if abs(diff) < cons["work"][1] + split["work"][1]:
        return "within noise"
    return "2 tasks faster" if diff > 0 else "4 tasks faster"


def build_summary_table_html(summary: dict[str, dict[int, dict]]) -> str:
    """Both layouts' time after startup, the 4-task layout's difference, and a verdict per worker count."""
    rows = ""
    for wc in _worker_counts(summary):
        cons, split = summary["consolidated"][wc], summary["split"][wc]
        diff = split["work"][0] - cons["work"][0]
        rows += (f"<tr><td>{wc}</td>"
                 f"<td>{cons['work'][0]:.2f} ± {cons['work'][1]:.2f}</td>"
                 f"<td>{split['work'][0]:.2f} ± {split['work'][1]:.2f}</td>"
                 f"<td>{diff:+.2f} ({diff / cons['work'][0]:+.1%})</td>"
                 f"<td>{_verdict(cons, split)}</td></tr>")
    return ("<div class=\"table-scroll\"><table><thead><tr><th>Workers</th>"
            "<th>2 tasks (s)</th><th>4 tasks (s)</th><th>4 tasks vs. 2</th><th>Verdict</th>"
            f"</tr></thead><tbody>{rows}</tbody></table></div>")


def build_compute_table_html(summary: dict[str, dict[int, dict]]) -> str:
    """Per-run compute of the bbox task(s) vs. the core-table task(s), under each layout."""
    rows = ""
    for wc in _worker_counts(summary):
        cells = ""
        for layout in LAYOUTS:
            entry = summary[layout][wc]
            share = entry["bbox_compute"] / (entry["bbox_compute"] + entry["core_compute"])
            cells += (f"<td>{entry['bbox_compute']:.2f}</td><td>{entry['core_compute']:.2f}</td>"
                      f"<td>{share:.0%}</td>")
        rows += f"<tr><td>{wc}</td>{cells}</tr>"
    return ("<div class=\"table-scroll\"><table><thead>"
            "<tr><th rowspan=\"2\">Workers</th><th colspan=\"3\">2 tasks/batch</th>"
            "<th colspan=\"3\">4 tasks/batch</th></tr>"
            "<tr><th>Bbox (s)</th><th>Core tables (s)</th><th>Bbox share</th>"
            "<th>Bbox (s)</th><th>Core tables (s)</th><th>Bbox share</th></tr>"
            f"</thead><tbody>{rows}</tbody></table></div>")


def build_report(summary: dict[str, dict[int, dict]]) -> None:
    """Write the comparison fragment, and publish a copy of the raw results next to it."""
    os.makedirs(BENCHMARK_REPORT_DIR, exist_ok=True)
    data_dir = os.path.join(BENCHMARK_REPORT_DIR, PUBLISHED_DATA_DIR)
    os.makedirs(data_dir, exist_ok=True)
    shutil.copy(RESULTS_PATH, data_dir)
    data_link = f"{PUBLISHED_DATA_DIR}/{os.path.basename(RESULTS_PATH)}"

    n = min(summary[layout][wc]["n"] for layout in LAYOUTS for wc in _worker_counts(summary))
    chart = build_bar_chart(summary, "work", "Time after the first task starts, by task layout")

    section_html = (
        "<section id=\"task-layout-data\">\n"
        "<h2>Data: 2 vs. 4 Ray tasks per batch</h2>\n"
        "<h3>Time after startup</h3>\n"
        "<div class=\"chart-note\">Wall time minus startup overhead (interpreter, imports, "
        "<code>ray.init()</code>, loading the devkit's tables), which is identical under both "
        "layouts. Mean ± sample standard deviation over "
        f"{n} successful runs per layout and worker count. \"Within noise\" means the "
        "difference is smaller than the two standard deviations added together: a rule of "
        "thumb for this few repeats, not a significance test.</div>\n"
        f"{build_summary_table_html(summary)}\n"
        f"{chart.to_html(full_html=False, include_plotlyjs='cdn')}\n"
        "<h3>Where the task compute goes</h3>\n"
        "<div class=\"chart-note\">Compute seconds per run, summed over every batch's bbox "
        "task and over its core-table task(s), from the same runs. Read from each task's own "
        "timing, so queue wait and startup are excluded.</div>\n"
        f"{build_compute_table_html(summary)}\n"
        f"<p class=\"chart-note\">Raw results, one JSON line per run: "
        f"<a href=\"{data_link}\"><code>{os.path.basename(RESULTS_PATH)}</code></a>. "
        f"Produced by <a href=\"{REPO_URL}/blob/main/benchmark_task_layout.py\"><code>benchmark_task_layout.py</code></a>.</p>\n"
        "</section>\n"
    )

    output_path = os.path.join(BENCHMARK_REPORT_DIR, "_task_layout_section.html")
    with open(output_path, "w") as fragment_file:
        fragment_file.write(section_html)

    print(f"Wrote {output_path} and {os.path.join(data_dir, os.path.basename(RESULTS_PATH))}")


if __name__ == "__main__":
    summary = aggregate(load_results() if "--report-only" in sys.argv else run_sweep())

    if not _worker_counts(summary):
        print("No worker count has successful runs under both layouts -- nothing to report.")
        sys.exit(1)

    build_report(summary)
