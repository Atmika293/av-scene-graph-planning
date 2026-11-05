"""Benchmark harness: compares partition.py's performance across worker counts.

Runs partition.py as an isolated subprocess (not in-process ray.shutdown()/
ray.init() cycling -- a fresh process per run is needed for a clean 1-vs-N
comparison) once per (worker count x repeat), overriding NUM_CPUS/BATCH_SIZE/
MAX_IN_FLIGHT in its environment so there are enough independent per-scene
tasks to actually exercise however many workers are under test. Each run
collects:
  - wall_time: the subprocess' total wall-clock duration (time.time(), so
    it's directly comparable to partition.py's own internal timestamps).
  - startup_overhead: time from subprocess launch to the first Ray task
    actually starting -- interpreter/import/ray.init()/NuScenes-load cost,
    which doesn't shrink as worker count grows.
  - task_timings: partition.py's own reports/partition_timing.json, i.e.
    every task's wait/compute/write breakdown for that run.
  - cpu_samples: whole-machine CPU utilization sampled throughout the run.

Writes one JSON line per run to reports/benchmark_results.jsonl as it goes
(so a crash partway through the sweep doesn't lose earlier runs), then
aggregates everything into 4 charts in
BENCHMARK_REPORT_DIR/_performance_section.html -- an HTML *fragment*, not a
full page (see build_report.py, which will combine this with the
correctness section into one docs/index.html).

Run with: python benchmark.py (from the repo root, same as partition.py).
"""

import json
import os
import statistics
import subprocess
import sys
import threading
import time

import psutil
import plotly.graph_objects as go

from av_data_pipeline.config import (
    BENCHMARK_WORKER_COUNTS, BENCHMARK_REPEATS,
    BENCHMARK_CPU_SAMPLE_INTERVAL, BENCHMARK_REPORT_DIR,
)
from report.html import KATEX_CDN_HTML

RESULTS_PATH = "reports/benchmark_results.jsonl"
PARTITION_TIMING_PATH = "reports/partition_timing.json"


class CPUSampler:
    """Samples whole-machine CPU utilization on a background thread.

    psutil.cpu_percent(interval=X) blocks for X seconds and returns the
    average utilization over that window, so one call per loop iteration
    is itself the sampling cadence -- no separate sleep() is needed.
    """

    def __init__(self, interval: float):
        self.interval = interval
        self.samples: list[tuple[float, float]] = []  # (elapsed_seconds, cpu_percent)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._start_time = None

    def start(self):
        self._start_time = time.perf_counter()
        self._thread.start()

    def _run(self):
        while not self._stop.is_set():
            pct = psutil.cpu_percent(interval=self.interval)
            self.samples.append((time.perf_counter() - self._start_time, pct))

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=self.interval + 1)


def run_partition_once(worker_count: int, repeat: int) -> dict:
    """Run partition.py once as a subprocess with NUM_CPUS=worker_count.

    BATCH_SIZE/MAX_IN_FLIGHT are overridden (to 1 and 32) so there are
    enough independent per-scene tasks to actually exercise worker_count
    workers concurrently, regardless of config.toml's own partition.py
    defaults (which are tuned for a real export run, not a benchmark).

    Args:
        worker_count: Value to set NUM_CPUS to for this run.
        repeat: Which repeat of this worker_count this is (0-indexed; only
            carried through into the result record, not used for behavior).

    Returns:
        One result record: worker_count, repeat, success, wall_time,
        startup_overhead (None if the run failed before any task started),
        task_timings (partition.py's own per-task breakdown), cpu_samples.
    """
    env = os.environ.copy()
    env["NUM_CPUS"] = str(worker_count)
    env["BATCH_SIZE"] = "1"
    env["MAX_IN_FLIGHT"] = "32"

    # Stale from a previous run (or absent); either way, only this run's
    # fresh file -- written unconditionally by partition.py -- should count.
    if os.path.exists(PARTITION_TIMING_PATH):
        os.remove(PARTITION_TIMING_PATH)

    sampler = CPUSampler(BENCHMARK_CPU_SAMPLE_INTERVAL)
    sampler.start()

    launched_at = time.time()
    proc = subprocess.run([sys.executable, "partition.py"], env=env,
                         capture_output=True, text=True)
    wall_time = time.time() - launched_at

    sampler.stop()

    success = proc.returncode == 0
    task_timings = []
    if os.path.exists(PARTITION_TIMING_PATH):
        with open(PARTITION_TIMING_PATH) as timing_file:
            task_timings = json.load(timing_file)

    startup_overhead = None
    if task_timings:
        # received_at is time.time(), the same clock as launched_at (same
        # machine), so this cross-process subtraction is valid.
        earliest_received_at = min(t["received_at"] for t in task_timings)
        startup_overhead = earliest_received_at - launched_at

    if not success:
        print(f"  worker_count={worker_count} repeat={repeat}: partition.py "
             f"failed (exit {proc.returncode}):\n{proc.stderr[-2000:]}")

    return {
        "worker_count": worker_count,
        "repeat": repeat,
        "success": success,
        "wall_time": wall_time,
        "startup_overhead": startup_overhead,
        "task_timings": task_timings,
        "cpu_samples": sampler.samples,
    }


def run_sweep() -> list[dict]:
    """Run every (worker count x repeat) combination, writing results incrementally.

    Returns:
        Every run's result record, in the order they completed.
    """
    os.makedirs("reports", exist_ok=True)
    results = []

    with open(RESULTS_PATH, "w") as results_file:
        for worker_count in BENCHMARK_WORKER_COUNTS:
            for repeat in range(BENCHMARK_REPEATS):
                print(f"Running worker_count={worker_count} repeat={repeat}...")
                result = run_partition_once(worker_count, repeat)
                results.append(result)

                results_file.write(json.dumps(result) + "\n")
                results_file.flush()

    return results


def aggregate_by_worker_count(results: list[dict]) -> dict[int, dict]:
    """Summarize every successful run, grouped by worker count.

    Args:
        results: Every run's result record, as produced by run_sweep.

    Returns:
        worker_count -> {wall_times, mean_wall_time, stdev_wall_time,
        mean_startup_overhead, mean_wait/compute/write (per task, averaged
        across every task in every repeat), all_wait_times/all_compute_times/
        all_write_times (flattened, for the distribution chart), cpu_samples
        (one representative repeat's samples, for the utilization-over-time
        chart)}.
    """
    by_worker_count: dict[int, dict] = {}

    for worker_count in BENCHMARK_WORKER_COUNTS:
        runs = [r for r in results if r["worker_count"] == worker_count and r["success"]]
        if not runs:
            continue

        wall_times = [r["wall_time"] for r in runs]
        startup_overheads = [r["startup_overhead"] for r in runs if r["startup_overhead"] is not None]
        all_tasks = [task for r in runs for task in r["task_timings"]]

        by_worker_count[worker_count] = {
            "wall_times": wall_times,
            "mean_wall_time": statistics.mean(wall_times),
            "stdev_wall_time": statistics.stdev(wall_times) if len(wall_times) > 1 else 0.0,
            "mean_startup_overhead": statistics.mean(startup_overheads) if startup_overheads else 0.0,
            "mean_wait": statistics.mean(t["wait_time"] for t in all_tasks) if all_tasks else 0.0,
            "mean_compute": statistics.mean(t["compute_time"] for t in all_tasks) if all_tasks else 0.0,
            "mean_write": statistics.mean(t["write_time"] for t in all_tasks) if all_tasks else 0.0,
            "all_wait_times": [t["wait_time"] for t in all_tasks],
            "all_compute_times": [t["compute_time"] for t in all_tasks],
            "all_write_times": [t["write_time"] for t in all_tasks],
            # First successful repeat's samples only -- repeats can differ
            # slightly in duration/sample count, and averaging across them
            # would need resampling that isn't worth it for a diagnostic chart.
            "cpu_samples": runs[0]["cpu_samples"],
        }

    return by_worker_count


def build_breakdown_chart(by_worker_count: dict[int, dict]) -> go.Figure:
    """Grouped bar: average per-task time composition by worker count.

    Grouped, not stacked: these four quantities are per-task averages that
    overlap in real wall-clock time once more than one worker is active, so
    stacking them would visually imply a sum that isn't real (see the
    speedup chart for the actual measured wall-clock improvement). A log
    y-axis is used because the components span orders of magnitude -- disk
    write (milliseconds) was invisible next to startup overhead (seconds)
    on a linear scale; a log-scaled *stacked* bar would have been worse,
    since log(a+b) != log(a)+log(b), so the stacking itself would no longer
    represent a true sum.
    """
    worker_counts = sorted(by_worker_count)
    fig = go.Figure()
    for segment, label in [("mean_startup_overhead", "Startup overhead"),
                           ("mean_wait", "Wait (avg/task)"),
                           ("mean_compute", "Compute (avg/task)"),
                           ("mean_write", "Disk write (avg/task)")]:
        fig.add_bar(name=label, x=worker_counts,
                   y=[by_worker_count[wc][segment] for wc in worker_counts])

    fig.update_layout(barmode="group", title="Average per-task time breakdown by worker count",
                      xaxis_title="Worker count", yaxis_title="Seconds (log scale)",
                      yaxis_type="log")
    # worker_counts are sweep values (1, 2, 4, 8, ...), not an evenly-spaced
    # range -- a numeric (linear) x-axis would space the bars by their actual
    # value and auto-generate intermediate ticks (e.g. "6") that have no bar
    # under them. Category type spaces every worker count evenly instead,
    # matching how the box-plot charts below already treat it.
    fig.update_xaxes(type="category")
    return fig


def build_breakdown_observations_html(by_worker_count: dict[int, dict]) -> str:
    """Observations sub-section for the breakdown chart: startup overhead's flatness.

    Computed from the actual run, not hardcoded, so it stays accurate if the
    sweep is re-run on different hardware or with different worker counts.
    """
    startups = [by_worker_count[wc]["mean_startup_overhead"] for wc in by_worker_count]
    lo, hi = min(startups), max(startups)
    return (
        "<h4>Observations</h4>"
        f"<div class=\"chart-note\">Startup overhead stays within roughly "
        f"\\([{lo:.1f}, {hi:.1f}]\\) seconds across every worker count tested here. "
        "That's expected: it's the cost of interpreter startup, imports, "
        "ray.init(), and loading the NuScenes devkit's tables. All of it "
        "happens once per subprocess, before any task is submitted, so adding "
        "more workers has nothing to parallelize away.</div>"
    )


def build_speedup_chart(by_worker_count: dict[int, dict]) -> go.Figure:
    """Line chart: measured speedup vs. an ideal-linear-scaling reference.

    Speedup is relative to the smallest worker count actually benchmarked
    (not hardcoded to 1), so this still works if BENCHMARK_WORKER_COUNTS
    doesn't include 1.
    """
    worker_counts = sorted(by_worker_count)
    baseline_wc = worker_counts[0]
    baseline_time = by_worker_count[baseline_wc]["mean_wall_time"]

    actual = [baseline_time / by_worker_count[wc]["mean_wall_time"] for wc in worker_counts]
    ideal = [wc / baseline_wc for wc in worker_counts]

    fig = go.Figure()
    fig.add_scatter(x=worker_counts, y=actual, mode="lines+markers", name="Measured speedup")
    fig.add_scatter(x=worker_counts, y=ideal, mode="lines", name="Ideal (linear) speedup",
                   line=dict(dash="dash"))
    fig.update_layout(title=f"Speedup vs. worker count (relative to {baseline_wc} worker(s))",
                      xaxis_title="Worker count", yaxis_title=f"Speedup (x{baseline_wc}-worker time)")
    return fig


def build_speedup_observations_html(by_worker_count: dict[int, dict]) -> str:
    """Observations sub-section for the speedup chart: why it flattens (or reverses).

    Computed from the actual run: finds the fastest worker count, and if a
    larger one tested here was actually slower, explains that using the
    machine's own logical core count and the measured compute-time increase
    -- rather than asserting it as a fixed claim that might not hold on
    different hardware or a different sweep.
    """
    worker_counts = sorted(by_worker_count)
    best_wc = min(worker_counts, key=lambda wc: by_worker_count[wc]["mean_wall_time"])
    slowest_larger_wcs = [wc for wc in worker_counts
                          if wc > best_wc and
                          by_worker_count[wc]["mean_wall_time"] > by_worker_count[best_wc]["mean_wall_time"]]

    text = (f"The fastest run here used {best_wc} worker(s) "
           f"({by_worker_count[best_wc]['mean_wall_time']:.1f}s mean wall time).")

    if slowest_larger_wcs:
        worst_wc = max(slowest_larger_wcs, key=lambda wc: by_worker_count[wc]["mean_wall_time"])
        cpu_count = os.cpu_count()
        text += (f" {worst_wc} workers was slower "
                f"({by_worker_count[worst_wc]['mean_wall_time']:.1f}s) despite the extra parallelism.")
        if cpu_count and worst_wc >= cpu_count:
            text += (f" This machine has {cpu_count} logical cores, so at {worst_wc} workers the workers "
                    "alone saturate every core, leaving none spare for the driver process and Ray's own "
                    "scheduling; that contention shows up as a higher average compute time per task "
                    f"({by_worker_count[best_wc]['mean_compute']:.3f}s at {best_wc} workers vs. "
                    f"{by_worker_count[worst_wc]['mean_compute']:.3f}s at {worst_wc}), which outweighs the "
                    "reduced queueing and flattens (here, slightly reverses) the speedup curve.")
        else:
            text += (" Compute time per task also rose there "
                    f"({by_worker_count[best_wc]['mean_compute']:.3f}s at {best_wc} workers vs. "
                    f"{by_worker_count[worst_wc]['mean_compute']:.3f}s at {worst_wc}), suggesting contention "
                    "for some resource other than CPU core count.")
    else:
        text += " Speedup kept improving through the largest worker count tested here."

    return f"<h4>Observations</h4><div class=\"chart-note\">{text}</div>"


def build_cpu_utilization_chart(by_worker_count: dict[int, dict]) -> go.Figure:
    """Line chart: whole-machine CPU utilization over time, one trace per worker count."""
    fig = go.Figure()
    for worker_count in sorted(by_worker_count):
        samples = by_worker_count[worker_count]["cpu_samples"]
        fig.add_scatter(x=[s[0] for s in samples], y=[s[1] for s in samples],
                        mode="lines", name=f"{worker_count} worker(s)")

    fig.update_layout(title="CPU utilization over time (one representative repeat per worker count)",
                      xaxis_title="Elapsed seconds", yaxis_title="CPU utilization (%)")
    return fig


def build_cpu_observations_html(by_worker_count: dict[int, dict]) -> str:
    """Observations sub-section for the CPU utilization chart.

    psutil.cpu_percent() reports whole-machine utilization normalized so
    100% means every logical core is fully busy -- so the right reference
    point is this machine's own core count, not a fixed threshold.
    """
    worker_counts = sorted(by_worker_count)
    lo_wc, hi_wc = worker_counts[0], worker_counts[-1]
    lo_vals = [v for _, v in by_worker_count[lo_wc]["cpu_samples"]]
    hi_vals = [v for _, v in by_worker_count[hi_wc]["cpu_samples"]]
    cpu_count = os.cpu_count()

    text = (f"Mean utilization rises from {statistics.mean(lo_vals):.0f}% at {lo_wc} worker(s) to "
           f"{statistics.mean(hi_vals):.0f}% at {hi_wc}, with the peak reaching "
           f"{max(hi_vals):.0f}% there (vs. {max(lo_vals):.0f}% at {lo_wc}). ")
    if cpu_count:
        text += (f"This machine has {cpu_count} logical cores, so 100% here means every core is fully busy "
                f"at once: at {lo_wc} worker(s), utilization stays low because most cores sit idle while "
                f"only {lo_wc} task(s) run at a time; by {hi_wc} workers, utilization is sustained much "
                "higher and repeatedly approaches that ceiling. That's the same saturation that shows up as "
                "increased per-task compute time in the Speedup chart's observations above.")
    else:
        text += ("Rising utilization with worker count is expected: more workers means more cores doing "
                "work simultaneously, up to the point where it saturates the machine's available cores.")

    return f"<h4>Observations</h4><div class=\"chart-note\">{text}</div>"


def build_time_distribution_chart(by_worker_count: dict[int, dict]) -> go.Figure:
    """Grouped box plot: per-task wait/compute/write time distributions by worker count.

    One group of 3 boxes (wait, compute, write) per worker count, rather
    than 3 separate charts, so their relative spread is comparable at a
    glance. Log y-axis for the same reason as the breakdown chart: write
    time is typically milliseconds, wait/compute typically seconds.
    """
    fig = go.Figure()
    for metric, label in [("all_wait_times", "Wait"),
                          ("all_compute_times", "Compute"),
                          ("all_write_times", "Write")]:
        x, y = [], []
        for worker_count in sorted(by_worker_count):
            values = by_worker_count[worker_count][metric]
            x += [worker_count] * len(values)
            y += values
        fig.add_box(x=x, y=y, name=label)

    fig.update_layout(boxmode="group", title="Per-task time distribution by worker count",
                      xaxis_title="Worker count", yaxis_title="Seconds (log scale)",
                      yaxis_type="log")
    return fig


def build_wait_observations_html(by_worker_count: dict[int, dict]) -> str:
    """Observations sub-section for the time-distribution chart: wait time narrowing.

    Computed from the smallest and largest worker counts actually tested,
    not hardcoded worker-count values.
    """
    worker_counts = sorted(by_worker_count)
    lo_wc, hi_wc = worker_counts[0], worker_counts[-1]
    lo_waits = by_worker_count[lo_wc]["all_wait_times"]
    hi_waits = by_worker_count[hi_wc]["all_wait_times"]

    return (
        "<h4>Observations</h4>"
        f"<div class=\"chart-note\">Average wait time drops from "
        f"{statistics.mean(lo_waits):.2f}s at {lo_wc} worker(s) to "
        f"{statistics.mean(hi_waits):.2f}s at {hi_wc}, and its spread (stdev) narrows from "
        f"{statistics.pstdev(lo_waits):.2f}s to {statistics.pstdev(hi_waits):.2f}s over the same range. "
        f"With only {lo_wc} worker(s), a task that isn't running yet has to queue behind whichever "
        "tasks got there first, so its wait depends heavily on its position in that queue, which "
        "varies widely from task to task. With more workers, a newly-submitted task is far more likely "
        "to find one immediately free, so both its average wait and the variability of that wait shrink "
        "together.</div>"
    )


def build_summary_table_html(by_worker_count: dict[int, dict]) -> str:
    """Small HTML table: mean wall time +/- stdev per worker count, with the formulas used."""
    rows = "\n".join(
        f"<tr><td>{wc}</td><td>{by_worker_count[wc]['mean_wall_time']:.2f}</td>"
        f"<td>{by_worker_count[wc]['stdev_wall_time']:.2f}</td></tr>"
        for wc in sorted(by_worker_count)
    )
    formula_note = (
        "<div class=\"chart-note\">Mean wall time is the average measured subprocess "
        "duration across every successful repeat at that worker count: "
        "\\[\\bar{x} = \\frac{1}{n}\\sum_{i=1}^{n} x_i\\] "
        "Stdev is the sample standard deviation of the same values (0 if fewer than "
        "2 repeats succeeded): \\[s = \\sqrt{\\frac{1}{n-1}\\sum_{i=1}^{n} "
        "(x_i - \\bar{x})^2}\\] "
        "Here \\(x_i\\) is one repeat's measured wall time, \\(n\\) is how many "
        "repeats succeeded at that worker count, and \\(\\bar{x}\\) (\"x-bar\") is "
        "the mean from the first formula, reused directly in the second.</div>"
    )
    return (formula_note +
           f"<table><thead><tr><th>Worker count</th><th>Mean wall time (s)</th>"
           f"<th>Stdev (s)</th></tr></thead><tbody>{rows}</tbody></table>")


def build_report(by_worker_count: dict[int, dict]) -> None:
    """Write the 4 charts + summary table as an HTML fragment.

    This is a fragment (a <section>, not a full <html> document) so
    build_report.py (not yet built) can combine it with the correctness
    section into one docs/index.html sharing a single plotly.js load.
    include_plotlyjs='cdn' is used here anyway so the fragment also renders
    correctly if opened directly, ahead of that final assembly step.
    """
    os.makedirs(BENCHMARK_REPORT_DIR, exist_ok=True)

    charts = [
        ("Per-task time breakdown",
         build_breakdown_chart(by_worker_count),
         "Average time a single task spends in each phase, by worker count. Bars are "
         "grouped, not stacked: these phases overlap once more than one worker runs, "
         "so this chart shows composition, not total duration (see Speedup for that). "
         "Log scale because the phases span orders of magnitude."
         "<ul style=\"margin:4px 0 8px 20px; padding:0;\">"
         "<li><strong>Startup overhead</strong>: time from subprocess launch to the "
         "first Ray task actually starting: interpreter startup, imports, "
         "ray.init(), and loading the NuScenes devkit's tables. Paid once per "
         "subprocess, before any task runs, so it doesn't shrink as worker count "
         "grows.</li>"
         "<li><strong>Wait (avg/task)</strong>: how long a task sits queued before a "
         "worker is free to run it. Expected to shrink as worker count grows, since more "
         "workers means less queueing.</li>"
         "<li><strong>Compute (avg/task)</strong>: time actually spent building and "
         "projecting a batch's rows. Each task does the same fixed work regardless of "
         "concurrency, so this should stay roughly flat, unless worker count starts "
         "exceeding the machine's available cores.</li>"
         "<li><strong>Disk write (avg/task)</strong>: time spent inside "
         "pq.write_table. Consistently the smallest component here; not CPU-bound, so "
         "it isn't expected to change with worker count.</li>"
         "</ul>",
         build_breakdown_observations_html(by_worker_count)),
        ("Speedup vs. worker count",
         build_speedup_chart(by_worker_count),
         "Measured wall-clock speedup against an ideal linear-scaling reference line, "
         "relative to the smallest worker count benchmarked (\\(N_{baseline}\\)): "
         "\\[\\text{Speedup}(N) = \\frac{T_{baseline}}{T_N} \\qquad "
         "\\text{Speedup}_{ideal}(N) = \\frac{N}{N_{baseline}}\\] "
         "The closer the measured line tracks the ideal line, the more effectively "
         "extra workers are being used; a line flattening below ideal means "
         "diminishing returns.",
         build_speedup_observations_html(by_worker_count)),
        ("CPU utilization over time",
         build_cpu_utilization_chart(by_worker_count),
         "Whole-machine CPU utilization sampled over one representative run per worker "
         "count. Higher, more sustained utilization means the workers are actually "
         "kept busy; a run that finishes faster but with low utilization suggests the "
         "bottleneck isn't CPU.",
         build_cpu_observations_html(by_worker_count)),
        ("Per-task time distribution",
         build_time_distribution_chart(by_worker_count),
         "Full distribution (not just the mean) of each task's wait/compute/write "
         "time, grouped by worker count. Box plots surface spread and outliers an "
         "average alone would hide. For example, wait time's spread narrowing as worker "
         "count grows is a stronger signal of reduced queueing than the mean alone.",
         build_wait_observations_html(by_worker_count)),
    ]
    chart_html = "\n".join(
        f"<h3>{heading}</h3>\n<div class=\"chart-note\">{note}</div>\n" +
        fig.to_html(full_html=False, include_plotlyjs="cdn" if i == 0 else False) +
        (observations or "")
        for i, (heading, fig, note, observations) in enumerate(charts)
    )

    section_html = (
        "<section id=\"performance\">\n"
        f"{KATEX_CDN_HTML}\n"
        "<h2>Performance: Ray worker-count benchmark</h2>\n"
        f"{build_summary_table_html(by_worker_count)}\n"
        f"{chart_html}\n"
        "</section>\n"
    )

    output_path = os.path.join(BENCHMARK_REPORT_DIR, "_performance_section.html")
    with open(output_path, "w") as fragment_file:
        fragment_file.write(section_html)

    print(f"Wrote {output_path}")


if __name__ == "__main__":
    results = run_sweep()
    by_worker_count = aggregate_by_worker_count(results)

    if not by_worker_count:
        print("No successful runs -- nothing to report.")
        sys.exit(1)

    build_report(by_worker_count)
