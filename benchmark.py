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
  - cpu_samples: CPU utilization sampled throughout the run, as a share of
    the CPUs this machine (or container) may use; see CPUSampler.

Each sweep is a *series*: one dataset version (NUSCENES_VERSION) on one
machine (BENCHMARK_MACHINE). It writes one JSON line per run to its own
reports/benchmark_results_<version>_<machine>.jsonl as it goes (so a crash
partway through doesn't lose earlier runs, and a sweep of one dataset or
machine never overwrites another's). Every run writes its Parquet output to
an emptied data/benchmark_output/<version>_<machine>/, so runs don't pile
up output and each one starts from the same state.

The report is then built from every series' results file into one
BENCHMARK_REPORT_DIR/_performance_section.html (an HTML *fragment*, combined
into the site by scripts/report/build_report.py): one speedup line per
series, and the other charts once per series.

Run with: python benchmark.py (from the repo root, same as partition.py).
On another dataset or machine:
    NUSCENES_VERSION=v1.0-trainval BENCHMARK_MACHINE=runpod python benchmark.py
To rebuild the fragment from saved results without rerunning anything:
    python benchmark.py --report-only
"""

import argparse
import glob
import json
import os
import shutil
import statistics
import subprocess
import sys
import threading
import time

import psutil
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from av_data_pipeline.config import (
    NUSCENES_VERSION, BENCHMARK_WORKER_COUNTS, BENCHMARK_REPEATS,
    BENCHMARK_CPU_SAMPLE_INTERVAL, BENCHMARK_REPORT_DIR, BENCHMARK_MACHINE,
)
from av_data_pipeline.machine import available_cpus, cpu_usage_seconds
from report.html import KATEX_CDN_HTML

REPORTS_DIR = "reports"
RESULTS_PREFIX = "benchmark_results_"
PARTITION_TIMING_PATH = "reports/partition_timing.json"
# The series this process sweeps: one dataset version on one machine.
SERIES = f"{NUSCENES_VERSION}_{BENCHMARK_MACHINE}"
# Parent of each series' benchmark output folder; kept apart from OUTPUT_ROOT,
# where a real export goes, since each run empties its folder first.
BENCHMARK_OUTPUT_DIR = "data/benchmark_output"

# Each machine gets its own results page: comparisons are only drawn within one
# machine. The laptop's single-dataset sweep is the v1.0-mini page; a machine that
# swept several datasets gets a page comparing them (see build_comparison_report).
LAPTOP_MACHINE = "laptop"
LAPTOP_FRAGMENT = "_performance_section.html"

# One colour per dataset on the comparison page, smallest dataset first (Plotly's
# first default colours, matching the v1.0-mini page); startup is drawn in grey.
DATASET_COLORS = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA"]
STARTUP_COLOR = "#B6BBC4"


def results_path(prefix: str, series: str = SERIES) -> str:
    """A series' results file under reports/, e.g. reports/benchmark_results_v1.0-mini_laptop.jsonl."""
    return os.path.join(REPORTS_DIR, f"{prefix}{series}.jsonl")


def load_all_series(prefix: str) -> dict[str, list[dict]]:
    """Every series' saved result records, keyed by series name, in name order."""
    series = {}
    for path in sorted(glob.glob(os.path.join(REPORTS_DIR, f"{prefix}*.jsonl"))):
        name = os.path.basename(path)[len(prefix):-len(".jsonl")]
        with open(path) as results_file:
            series[name] = [json.loads(line) for line in results_file if line.strip()]
    return series


def series_label(series: str) -> str:
    """A series name as readers see it: "v1.0-mini_laptop" -> "v1.0-mini on laptop"."""
    version, _, machine = series.rpartition("_")
    return f"{version} on {machine}"


class CPUSampler:
    """Samples CPU utilization on a background thread, as a share of the CPUs available.

    On Linux it reads this cgroup's own CPU time (machine.cpu_usage_seconds),
    because in a container psutil.cpu_percent() reads the host's /proc/stat
    and would count every other tenant's work too. Utilization is the CPU
    time used over each window divided by window length x available_cpus(),
    so 100% means every CPU this container may use was busy, matching what
    it means on the laptop.

    Elsewhere (macOS) it uses psutil.cpu_percent(interval=X), which blocks for
    X seconds and returns the average over that window, so one call per loop
    iteration is itself the sampling cadence.
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
        if cpu_usage_seconds() is None:
            while not self._stop.is_set():
                pct = psutil.cpu_percent(interval=self.interval)
                self.samples.append((time.perf_counter() - self._start_time, pct))
            return

        cpus = available_cpus()
        last_wall, last_usage = time.perf_counter(), cpu_usage_seconds()
        while not self._stop.wait(self.interval):
            wall, usage = time.perf_counter(), cpu_usage_seconds()
            pct = 100 * (usage - last_usage) / ((wall - last_wall) * cpus)
            self.samples.append((wall - self._start_time, min(pct, 100.0)))
            last_wall, last_usage = wall, usage

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=self.interval + 1)


def in_flight_cap(worker_count: int) -> int:
    """MAX_IN_FLIGHT for a run with worker_count workers: max(32, 2 x worker_count).

    The driver builds each batch's manifest itself, one at a time, so a cap of
    exactly worker_count would let a worker that just finished sit idle until
    the next batch is built; twice the worker count keeps a queue waiting.
    The floor of 32 is the fixed cap every earlier sweep used, so runs at up to
    16 workers (all of the laptop's) still run exactly as those did.
    """
    return max(32, 2 * worker_count)


def run_partition_once(worker_count: int, repeat: int, task_layout: str | None = None) -> dict:
    """Run partition.py once as a subprocess with NUM_CPUS=worker_count.

    BATCH_SIZE/MAX_IN_FLIGHT are overridden (to 1 and in_flight_cap()) so
    there are enough independent per-scene tasks to actually exercise
    worker_count workers concurrently, regardless of config.toml's own
    partition.py defaults (which are tuned for a real export run, not a
    benchmark).

    Args:
        worker_count: Value to set NUM_CPUS to for this run.
        repeat: Which repeat of this worker_count this is (0-indexed; only
            carried through into the result record, not used for behavior).
        task_layout: TASK_LAYOUT to run partition.py with ("consolidated" or
            "split"); None leaves it to config.toml/the environment. Used by
            benchmark_task_layout.py to compare the two under this same harness.

    Returns:
        One result record: worker_count, repeat, task_layout, success, wall_time,
        startup_overhead (None if the run failed before any task started),
        task_timings (partition.py's own per-task breakdown), cpu_samples.
    """
    env = os.environ.copy()
    env["NUM_CPUS"] = str(worker_count)
    env["BATCH_SIZE"] = "1"
    env["MAX_IN_FLIGHT"] = str(in_flight_cap(worker_count))
    # An emptied folder per run: earlier runs' files neither pile up nor sit
    # in the folder a later run writes to.
    output_root = os.path.join(BENCHMARK_OUTPUT_DIR, SERIES)
    shutil.rmtree(output_root, ignore_errors=True)
    env["OUTPUT_ROOT"] = output_root
    if task_layout is not None:
        env["TASK_LAYOUT"] = task_layout

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
        "task_layout": task_layout,
        "series": SERIES,
        # Recorded per run, so a report built on another machine still
        # describes the one the sweep ran on.
        # The CPUs this machine (or container) may use, not the host's.
        "cpu_count": available_cpus(),
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
    os.makedirs(REPORTS_DIR, exist_ok=True)
    results = []

    with open(results_path(RESULTS_PREFIX), "w") as results_file:
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

    for worker_count in sorted({r["worker_count"] for r in results}):
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
            # Runs from before cpu_count was recorded all came from the
            # machine that has always built the report, so fall back to it.
            "cpu_count": runs[0].get("cpu_count") or os.cpu_count(),
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
        "<h3>Observations</h3>"
        f"<div class=\"chart-note\">Startup overhead stays within roughly "
        f"\\([{lo:.1f}, {hi:.1f}]\\) seconds across every worker count tested here. "
        "That's expected: it's the cost of interpreter startup, imports, "
        "ray.init(), and loading the NuScenes devkit's tables. All of it "
        "happens once per subprocess, before any task is submitted, so adding "
        "more workers has nothing to parallelize away.</div>"
    )


def build_speedup_chart(series: dict[str, dict[int, dict]]) -> go.Figure:
    """Line chart: measured speedup per series vs. an ideal-linear-scaling reference.

    Each series' speedup is relative to its own smallest worker count actually
    benchmarked (not hardcoded to 1), so this still works if a sweep doesn't
    include 1. With one series the line is just "Measured speedup"; with
    several, each line is named after its series.
    """
    fig = go.Figure()
    baselines = set()
    for name, by_worker_count in series.items():
        worker_counts = sorted(by_worker_count)
        baseline_wc = worker_counts[0]
        baselines.add(baseline_wc)
        baseline_time = by_worker_count[baseline_wc]["mean_wall_time"]
        actual = [baseline_time / by_worker_count[wc]["mean_wall_time"] for wc in worker_counts]
        label = "Measured speedup" if len(series) == 1 else f"Measured: {series_label(name)}"
        fig.add_scatter(x=worker_counts, y=actual, mode="lines+markers", name=label)

    baseline_wc = min(baselines)
    all_worker_counts = sorted({wc for by_worker_count in series.values() for wc in by_worker_count})
    fig.add_scatter(x=all_worker_counts, y=[wc / baseline_wc for wc in all_worker_counts], mode="lines",
                    name="Ideal (linear) speedup", line=dict(dash="dash"))
    relative_to = (f"{baseline_wc} worker(s)" if len(baselines) == 1
                   else "each series' smallest worker count")
    fig.update_layout(title=f"Speedup vs. worker count (relative to {relative_to})",
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
        cpu_count = by_worker_count[worst_wc]["cpu_count"]
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

    return f"<h3>Observations</h3><div class=\"chart-note\">{text}</div>"


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
    cpu_count = by_worker_count[hi_wc]["cpu_count"]

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

    return f"<h3>Observations</h3><div class=\"chart-note\">{text}</div>"


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
        "<h3>Observations</h3>"
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
    """Small HTML table: mean wall time +/- stdev per worker count."""
    rows = "\n".join(
        f"<tr><td>{wc}</td><td>{by_worker_count[wc]['mean_wall_time']:.2f}</td>"
        f"<td>{by_worker_count[wc]['stdev_wall_time']:.2f}</td></tr>"
        for wc in sorted(by_worker_count)
    )
    return (f"<table><thead><tr><th>Worker count</th><th>Mean wall time (s)</th>"
            f"<th>Stdev (s)</th></tr></thead><tbody>{rows}</tbody></table>")


def build_wall_time_note_html() -> str:
    """The wall-time section's heading and the formulas behind its table(s)."""
    return (
        "<h2>Wall time by worker count</h2>\n<div class=\"chart-note\">Mean wall time is the average measured subprocess "
        "duration across every successful repeat at that worker count: "
        "\\[\\bar{x} = \\frac{1}{n}\\sum_{i=1}^{n} x_i\\] "
        "Stdev is the sample standard deviation of the same values (0 if fewer than "
        "2 repeats succeeded): \\[s = \\sqrt{\\frac{1}{n-1}\\sum_{i=1}^{n} "
        "(x_i - \\bar{x})^2}\\] "
        "Here \\(x_i\\) is one repeat's measured wall time, \\(n\\) is how many "
        "repeats succeeded at that worker count, and \\(\\bar{x}\\) (\"x-bar\") is "
        "the mean from the first formula, reused directly in the second.</div>"
    )


def build_report(series: dict[str, dict[int, dict]], fragment_name: str = LAPTOP_FRAGMENT) -> None:
    """Write the summary table(s) and 4 charts as an HTML fragment.

    This is a fragment (a <section>, not a full <html> document) that
    scripts/report/build_report.py combines into the site, sharing a single
    plotly.js load. include_plotlyjs='cdn' is used here anyway so the
    fragment also renders correctly if opened directly.

    Args:
        series: Series name -> its aggregate_by_worker_count summary. With one
            series the fragment has a single table and chart per section; with
            several, the speedup chart draws one line per series and every other
            table and chart repeats once per series, labelled with its name.
        fragment_name: File name under BENCHMARK_REPORT_DIR to write.
    """
    os.makedirs(BENCHMARK_REPORT_DIR, exist_ok=True)
    several = len(series) > 1

    # (heading, per-series chart builder or None for the shared speedup chart,
    #  note, per-series observations builder)
    charts = [
        ("Per-task time breakdown",
         build_breakdown_chart,
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
         build_breakdown_observations_html),
        ("Speedup vs. worker count",
         None,
         "Measured wall-clock speedup against an ideal linear-scaling reference line, "
         "relative to the smallest worker count benchmarked (\\(N_{baseline}\\)): "
         "\\[\\text{Speedup}(N) = \\frac{T_{baseline}}{T_N} \\qquad "
         "\\text{Speedup}_{ideal}(N) = \\frac{N}{N_{baseline}}\\] "
         "The closer the measured line tracks the ideal line, the more effectively "
         "extra workers are being used; a line flattening below ideal means "
         "diminishing returns.",
         build_speedup_observations_html),
        ("CPU utilization over time",
         build_cpu_utilization_chart,
         "Whole-machine CPU utilization sampled over one representative run per worker "
         "count. Higher, more sustained utilization means the workers are actually "
         "kept busy; a run that finishes faster but with low utilization suggests the "
         "bottleneck isn't CPU.",
         build_cpu_observations_html),
        ("Per-task time distribution",
         build_time_distribution_chart,
         "Full distribution (not just the mean) of each task's wait/compute/write "
         "time, grouped by worker count. Box plots surface spread and outliers an "
         "average alone would hide. For example, wait time's spread narrowing as worker "
         "count grows is a stronger signal of reduced queueing than the mean alone.",
         build_wait_observations_html),
    ]
    plotlyjs_included = False

    def figure_html(fig: go.Figure) -> str:
        """One chart's HTML; only the fragment's first chart embeds the plotly.js CDN tag."""
        nonlocal plotlyjs_included
        html = fig.to_html(full_html=False, include_plotlyjs=False if plotlyjs_included else "cdn")
        plotlyjs_included = True
        return html

    def labelled(observations: str, name: str) -> str:
        """Observations for one series, headed with its name when there are several."""
        return (observations.replace("<h3>Observations</h3>", f"<h3>Observations: {series_label(name)}</h3>")
                if several else observations)

    blocks = []
    for heading, build_chart, note, build_observations in charts:
        block = f"<h2>{heading}</h2>\n<div class=\"chart-note\">{note}</div>\n"
        if build_chart is None:
            block += figure_html(build_speedup_chart(series))
            block += "".join(labelled(build_observations(summary), name) for name, summary in series.items())
        else:
            for name, summary in series.items():
                fig = build_chart(summary)
                if several:
                    fig.update_layout(title_text=f"{fig.layout.title.text}: {series_label(name)}")
                block += figure_html(fig) + labelled(build_observations(summary), name)
        blocks.append(block)

    tables = "".join(
        (f"<h3>{series_label(name)}</h3>\n" if several else "") + build_summary_table_html(summary)
        for name, summary in series.items()
    )
    section_html = (
        "<section id=\"performance\">\n"
        f"{KATEX_CDN_HTML}\n"
        f"{build_wall_time_note_html()}{tables}\n"
        + "\n".join(blocks) + "\n"
        "</section>\n"
    )

    output_path = os.path.join(BENCHMARK_REPORT_DIR, fragment_name)
    with open(output_path, "w") as fragment_file:
        fragment_file.write(section_html)

    print(f"Wrote {output_path} from {len(series)} series: {', '.join(series)}")


def scene_count(results: list[dict]) -> int:
    """Scenes one run processed: its distinct batches, since every benchmark run uses BATCH_SIZE=1."""
    run = next(r for r in results if r["success"])
    return len({task["label"].split(":")[0] for task in run["task_timings"]})


def scaling_summary(by_worker_count: dict[int, dict]) -> dict:
    """One dataset's scaling numbers, from its aggregate_by_worker_count summary.

    "Work" is a run's wall time minus its startup overhead: the part extra
    workers can actually shrink. Speedups are relative to the smallest worker
    count benchmarked.
    """
    worker_counts = sorted(by_worker_count)
    base = worker_counts[0]
    wall = {wc: by_worker_count[wc]["mean_wall_time"] for wc in worker_counts}
    startup = {wc: by_worker_count[wc]["mean_startup_overhead"] for wc in worker_counts}
    work = {wc: wall[wc] - startup[wc] for wc in worker_counts}
    speedup = {wc: wall[base] / wall[wc] for wc in worker_counts}
    work_speedup = {wc: work[base] / work[wc] for wc in worker_counts}
    return {
        "worker_counts": worker_counts, "base": base, "max": worker_counts[-1],
        "wall": wall, "startup": startup, "work": work,
        "speedup": speedup, "work_speedup": work_speedup,
        "best": min(worker_counts, key=wall.get),
        "best_work": max(worker_counts, key=work_speedup.get),
        "repeats": min(len(by_worker_count[wc]["wall_times"]) for wc in worker_counts),
        "cpu_count": by_worker_count[worker_counts[-1]]["cpu_count"],
    }


def build_comparison_report(raw: dict[str, list[dict]], fragment_name: str) -> None:
    """Write the page fragment comparing several datasets swept on one machine.

    Every series in raw must come from the same machine, so the only thing that
    differs between them is the dataset. Unlike build_report's per-series
    charts, every chart and note here is about the comparison: wall-clock
    speedup against speedup after startup, how each run splits into startup and
    the time after it, the average per-task wait/compute/write, and CPU use. The notes are computed from the results, so
    they stay true if the sweeps are rerun.

    Args:
        raw: Series name -> its saved result records, all from one machine.
        fragment_name: File name under BENCHMARK_REPORT_DIR to write.
    """
    os.makedirs(BENCHMARK_REPORT_DIR, exist_ok=True)
    summaries = {name: aggregate_by_worker_count(results) for name, results in raw.items()}
    scenes = {name: scene_count(results) for name, results in raw.items()}
    names = sorted(summaries, key=scenes.get)  # smallest dataset first
    stats = {name: scaling_summary(summaries[name]) for name in names}
    version = {name: name.rpartition("_")[0] for name in names}
    color = {name: DATASET_COLORS[i % len(DATASET_COLORS)] for i, name in enumerate(names)}
    small, large = names[0], names[-1]
    cpus = stats[large]["cpu_count"]
    base = stats[large]["base"]
    all_worker_counts = sorted({wc for name in names for wc in stats[name]["worker_counts"]})
    worker_list = ", ".join(str(wc) for wc in all_worker_counts)

    plotlyjs_included = False

    def figure_html(fig: go.Figure) -> str:
        """One chart's HTML; only the fragment's first chart embeds the plotly.js CDN tag."""
        nonlocal plotlyjs_included
        html = fig.to_html(full_html=False, include_plotlyjs=False if plotlyjs_included else "cdn")
        plotlyjs_included = True
        return html

    def note(text: str) -> str:
        return f'<div class="chart-note">{text}</div>\n'

    def observations(items: list[str]) -> str:
        return ("<h3>Observations</h3>\n" + note(
            '<ul style="margin:4px 0 8px 20px; padding:0;">'
            + "".join(f"<li>{item}</li>" for item in items) + "</ul>"))

    # At a glance: one row per dataset.
    rows = "".join(
        f"<tr><td>{version[n]}</td><td>{scenes[n]:,}</td>"
        f"<td>{stats[n]['wall'][stats[n]['base']]:.1f} s</td>"
        f"<td>{stats[n]['wall'][stats[n]['best']]:.1f} s at {stats[n]['best']} workers</td>"
        f"<td>{stats[n]['speedup'][stats[n]['best']]:.2f}×</td>"
        f"<td>{stats[n]['work_speedup'][stats[n]['best_work']]:.2f}× at {stats[n]['best_work']} workers</td></tr>"
        for n in names)
    glance = (
        "<h2>At a glance</h2>\n"
        + note(f"Each dataset's mean wall time at {base} worker{'s' if base > 1 else ''} and at its fastest "
               "worker count, and how much faster the fastest was. The last column leaves out each "
               "run's fixed startup cost (see <a href=\"#scaling-timing\">Wall time and time after "
               "startup</a>).")
        + "<table><thead><tr><th>Dataset</th><th>Scenes</th>"
        f"<th>{base} worker{'s' if base > 1 else ''}</th><th>Fastest</th><th>Wall-clock speedup</th>"
        f"<th>Speedup after startup</th></tr></thead><tbody>{rows}</tbody></table>\n")

    method = (
        "<h2>How it was measured</h2>\n"
        + note('<ul style="margin:4px 0 8px 20px; padding:0;">'
               f"<li><strong>One machine:</strong> every run of both datasets on the same Runpod CPU pod "
               f"with {cpus} CPUs, so the dataset is the only thing that differs.</li>"
               f"<li><strong>Runs:</strong> each worker count ({worker_list}) run {stats[large]['repeats']} "
               "times as a fresh process, startup included, with one scene per batch and at most "
               "max(32, 2 × workers) tasks in flight. The harness is the same as for the "
               "<a href=\"result-ray-scaling.html\">v1.0-mini page</a>.</li>"
               f"<li><strong>CPU use:</strong> read from the pod's own share of the host, so 100% "
               f"means all {cpus} of its CPUs busy.</li>"
               "<li><strong>Not compared with the laptop:</strong> the v1.0-mini page's results come "
               "from a different machine, so its times are not set against these.</li></ul>"))

    # Wall time and time after startup: what each is, how it's averaged, and a table per dataset.
    timing_tables = "".join(
        f"<h3>{version[n]}</h3>\n<table><thead><tr><th>Worker count</th><th>Mean wall time (s)</th>"
        "<th>Stdev (s)</th><th>Mean startup (s)</th><th>Mean time after startup (s)</th></tr></thead><tbody>"
        + "".join(f"<tr><td>{wc}</td><td>{stats[n]['wall'][wc]:.1f}</td>"
                  f"<td>{summaries[n][wc]['stdev_wall_time']:.1f}</td>"
                  f"<td>{stats[n]['startup'][wc]:.1f}</td><td>{stats[n]['work'][wc]:.1f}</td></tr>"
                  for wc in stats[n]["worker_counts"])
        + "</tbody></table>\n"
        for n in names)
    timing = (
        '<h2 id="scaling-timing">Wall time and time after startup</h2>\n'
        + note("Every run is a fresh <code>partition.py</code> process, timed by the benchmark from "
               "launch to exit and split where the first task starts on a worker:"
               '<ul style="margin:4px 0 8px 20px; padding:0;">'
               "<li><strong>Wall time \\(T\\):</strong> from launching the process to its exit. This is "
               "what you actually wait for.</li>"
               "<li><strong>Startup \\(S\\):</strong> from launch to the first task starting: Python "
               "and imports, ray.init() starting the workers, loading the devkit's tables, and building "
               "the first batch's input. It happens once per run, before anything is parallel, so more "
               "workers can't shrink it.</li>"
               "<li><strong>Time after startup \\(T - S\\):</strong> the rest of the run. Mostly the "
               "tasks running across the workers, but also the driver building each later batch's "
               "input one at a time, collecting every task's result, and shutting down.</li></ul>"
               "Each worker count runs several times, and the tables give means over its successful "
               "runs. For a worker count with \\(n\\) runs, where \\(T_i\\) and \\(S_i\\) are run "
               "\\(i\\)'s wall time and startup: "
               "\\[\\bar{T} = \\frac{1}{n}\\sum_{i=1}^{n} T_i \\qquad "
               "\\bar{S} = \\frac{1}{n}\\sum_{i=1}^{n} S_i \\qquad "
               "\\text{time after startup} = \\bar{T} - \\bar{S}\\] "
               "Stdev is the sample standard deviation of the wall times, "
               "\\(s = \\sqrt{\\frac{1}{n-1}\\sum_{i=1}^{n} (T_i - \\bar{T})^2}\\): how much the "
               "repeats differ from each other.")
        + timing_tables)

    # Speedup: wall clock and after startup, both axes log so ideal scaling is a diagonal.
    fig = go.Figure()
    for n in names:
        s = stats[n]
        fig.add_scatter(x=s["worker_counts"], y=[s["speedup"][wc] for wc in s["worker_counts"]],
                        mode="lines+markers", name=f"{version[n]}: wall clock",
                        line=dict(color=color[n], width=2), marker=dict(size=8))
        fig.add_scatter(x=s["worker_counts"], y=[s["work_speedup"][wc] for wc in s["worker_counts"]],
                        mode="lines+markers", name=f"{version[n]}: after startup",
                        line=dict(color=color[n], width=2, dash="dot"),
                        marker=dict(size=8, symbol="circle-open"))
    fig.add_scatter(x=all_worker_counts, y=[wc / base for wc in all_worker_counts], mode="lines",
                    name="Ideal (linear)", line=dict(color="#888888", width=1.5, dash="dash"))
    ticks = dict(type="log", tickvals=all_worker_counts, ticktext=[str(wc) for wc in all_worker_counts])
    fig.update_layout(title=dict(text="Speedup vs. worker count (both axes log scale)"),
                      xaxis=dict(title=dict(text="Worker count"), **ticks),
                      yaxis=dict(title=dict(text=f"Speedup (× the {base}-worker time)"), **ticks))

    speed_items = []
    for n in names:
        s = stats[n]
        best, top = s["best"], s["max"]
        item = (f"<strong>{version[n]}</strong> was fastest at {best} workers, "
                f"{s['speedup'][best]:.2f}× the {s['base']}-worker time")
        if best == top:
            item += ", the largest count tested, and still improving."
        else:
            rise = s["wall"][top] - s["wall"][best]
            startup_rise = s["startup"][top] - s["startup"][best]
            item += (f". Past that it slowed down: from {best} to {top} workers its wall time rose "
                     f"{rise:.1f} s")
            item += (f", {startup_rise:.1f} s of it from a longer startup (starting more workers)."
                     if startup_rise > 0 else ".")
            item += (f" With {scenes[n]} scenes there are only {scenes[n]} batches of work to share out, "
                     "so beyond a few workers there's little left for them to do.")
        item += (f" Leaving out startup, its time after startup was {s['work_speedup'][s['best_work']]:.2f}× faster "
                 f"at {s['best_work']} workers.")
        speed_items.append(item)
    speed_items.append("The gap between each dataset's solid and dotted lines is its fixed startup "
                       "cost: more workers shrink the time after startup, but not the startup.")
    speedup = (
        '<h2 id="scaling-speedup">Speedup vs. worker count</h2>\n'
        + note("Two speedups per dataset, each relative to its own "
               f"{base}-worker runs, where \\(T_N\\) and \\(S_N\\) are the mean wall time and mean "
               "startup with \\(N\\) workers (see "
               "<a href=\"#scaling-timing\">Wall time and time after startup</a>): "
               "\\[\\text{wall clock} = \\frac{T_1}{T_N} \\qquad "
               "\\text{after startup} = \\frac{T_1 - S_1}{T_N - S_N}\\] "
               "The dashed line is ideal linear scaling.")
        + figure_html(fig) + observations(speed_items))

    # Where the time goes: wall time stacked as startup + work, one chart per dataset.
    split_charts, split_items = "", []
    for n in names:
        s = stats[n]
        labels = [str(wc) for wc in s["worker_counts"]]
        fig = go.Figure()
        fig.add_bar(x=labels, y=[s["startup"][wc] for wc in s["worker_counts"]], name="Startup",
                    marker=dict(color=STARTUP_COLOR))
        fig.add_bar(x=labels, y=[s["work"][wc] for wc in s["worker_counts"]], name="After startup",
                    marker=dict(color=color[n]))
        fig.update_layout(barmode="stack", bargap=0.3,
                          title=dict(text=f"Mean wall time, split: {version[n]}"),
                          xaxis=dict(title=dict(text="Worker count"), type="category"),
                          yaxis=dict(title=dict(text="Seconds")))
        split_charts += figure_html(fig)
        share = {wc: s["startup"][wc] / s["wall"][wc] for wc in s["worker_counts"]}
        split_items.append(f"<strong>{version[n]}:</strong> startup is {share[s['base']]:.0%} of the run "
                           f"at {s['base']} worker{'s' if s['base'] > 1 else ''} and "
                           f"{share[s['max']]:.0%} at {s['max']} workers.")
    split_items.append(f"Startup grows with the dataset, because it includes loading the devkit's "
                       f"tables: {stats[small]['startup'][stats[small]['base']]:.0f} s for "
                       f"{version[small]} against {stats[large]['startup'][stats[large]['base']]:.0f} s "
                       f"for {version[large]} at {base} worker{'s' if base > 1 else ''}. No number of "
                       "workers removes it, so it is what caps the wall-clock speedup above.")
    ratio = stats[large]["wall"][base] / stats[small]["wall"][base]
    split = (
        "<h2>Where the time goes</h2>\n"
        + note("Each bar is one worker count's mean wall time, split into startup and the time after "
               "it; the two happen one after the other, so they add up to the whole run. The time after "
               "startup is mostly the tasks running across the workers, but it also holds the driver "
               "building each later batch's input one at a time, collecting results and shutting down. The datasets get separate charts because "
               f"their times differ about {ratio:.0f} times at {base} worker{'s' if base > 1 else ''}.")
        + split_charts + observations(split_items))

    # Per-task time: wait, compute and write, one panel each, a line per dataset.
    phases = [("mean_wait", "Wait"), ("mean_compute", "Compute"), ("mean_write", "Disk write")]
    fig = make_subplots(rows=1, cols=3, subplot_titles=[label for _, label in phases],
                        horizontal_spacing=0.08)
    for n in names:
        wcs = stats[n]["worker_counts"]
        for col, (key, _) in enumerate(phases, start=1):
            fig.add_scatter(x=wcs, y=[summaries[n][wc][key] for wc in wcs], mode="lines+markers",
                            name=version[n], legendgroup=n, showlegend=col == 1,
                            line=dict(color=color[n], width=2), marker=dict(size=8), row=1, col=col)
    for col in range(1, 4):
        fig.update_xaxes(title=dict(text="Worker count"), type="log", tickvals=all_worker_counts,
                         ticktext=[str(wc) for wc in all_worker_counts], row=1, col=col)
        fig.update_yaxes(rangemode="tozero", row=1, col=col)
    fig.update_yaxes(title=dict(text="Seconds per task (mean)"), row=1, col=1)
    fig.update_layout(title=dict(text="Average time per task, by phase"), height=420)

    def per_task(n: str, key: str, wc: int) -> float:
        return summaries[n][wc][key]

    def first_and_rest(n: str, wc: int) -> tuple[float, float | None]:
        """Mean compute of each run's first wc tasks to start (one per worker, each also loading
        the pipeline's code on that worker) and of the tasks after them (None if there are none)."""
        first, rest = [], []
        for run in raw[n]:
            if run["success"] and run["worker_count"] == wc:
                ordered = sorted(run["task_timings"], key=lambda t: t["received_at"])
                first += [t["compute_time"] for t in ordered[:wc]]
                rest += [t["compute_time"] for t in ordered[wc:]]
        return statistics.mean(first), statistics.mean(rest) if rest else None

    def cpu_after_startup(n: str, wc: int) -> float:
        """Mean CPU use over the representative run's time after startup."""
        run = next(r for r in raw[n] if r["success"] and r["worker_count"] == wc)
        return statistics.mean(v for t, v in run["cpu_samples"] if t > run["startup_overhead"])

    task_items = []
    for n in names:
        b, top = stats[n]["base"], stats[n]["max"]
        task_items.append(
            f"<strong>{version[n]}:</strong> wait falls from {per_task(n, 'mean_wait', b):.2f} s to "
            f"{per_task(n, 'mean_wait', top):.2f} s between {b} and {top} workers, while compute rises "
            f"from {per_task(n, 'mean_compute', b):.3f} s to {per_task(n, 'mean_compute', top):.3f} s "
            f"({per_task(n, 'mean_compute', top) / per_task(n, 'mean_compute', b):.1f}×).")
    lb, ltop = stats[large]["base"], stats[large]["max"]
    first_b, rest_b = first_and_rest(large, lb)
    first_top, rest_top = first_and_rest(large, ltop)
    small_tasks = len(next(r for r in raw[small] if r["success"])["task_timings"])
    task_items.append(
        f"<strong>Part of the rise is a one-off cost.</strong> A worker's first task is slower, most "
        f"likely because it also loads the pipeline's code on that worker: on {version[large]} at {lb} worker{'s' if lb > 1 else ''} it took "
        f"{first_b:.2f} s against {rest_b:.3f} s for later tasks, and {first_top:.2f} s at {ltop} "
        f"workers, where {ltop} workers load at once. More workers means more first tasks. "
        f"{version[small]} has only {small_tasks} tasks per run, so at {stats[small]['max']} workers "
        + ("nearly every task is a worker's first, which accounts for most of its rise."
           if small_tasks <= stats[small]["max"] else
           "a large share of its tasks are a worker's first, which accounts for much of its rise."))
    task_items.append(
        f"<strong>The rest is tasks slowing each other down.</strong> Leaving out first tasks, "
        f"{version[large]}'s compute still rises from {rest_b:.3f} s to {rest_top:.3f} s "
        f"({rest_top / rest_b:.1f}×) between {lb} and {ltop} workers, though each task does the same "
        f"work. The pod averaged {cpu_after_startup(large, ltop):.0f}% busy over the time after "
        f"startup at {ltop} workers, so tasks aren't short of CPUs; they are probably competing for "
        "something shared, such as memory bandwidth, which these runs don't measure. This is part of "
        "why the speedup after startup stays below ideal.")
    write_max = max(per_task(n, "mean_write", wc) for n in names for wc in stats[n]["worker_counts"])
    task_items.append(f"<strong>Disk isn't a bottleneck.</strong> Writes average at most "
                      f"{write_max * 1000:.0f} ms per task, for either dataset at any worker count.")
    tasks = (
        "<h2>Per-task time breakdown</h2>\n"
        + note('Each task\'s time, averaged over every task in every repeat:<ul style="margin:4px 0 8px 20px; padding:0;">'
               "<li><strong>Wait:</strong> from the driver submitting the task to it starting on a "
               "worker, i.e. queueing for a free worker.</li>"
               "<li><strong>Compute:</strong> building and projecting the batch's rows.</li>"
               "<li><strong>Disk write:</strong> time inside pq.write_table.</li></ul>"
               "Each batch has two tasks, a light one writing the core tables and a heavy one "
               "projecting boxes, and the averages cover both. Tasks run at the same time on "
               "different workers, so these don't add up to a run's wall time; the chart above "
               "does that. Each panel has its own scale.")
        + figure_html(fig) + observations(task_items))

    # CPU use over time, one chart per dataset.
    cpu_charts, cpu_items = "", [f"One fully busy CPU shows as {100 / cpus:.0f}%."]
    for n in names:
        fig = build_cpu_utilization_chart(summaries[n])
        fig.update_layout(title=dict(text=f"CPU use over one run per worker count: {version[n]}"),
                          yaxis=dict(title=dict(text=f"Share of the pod's {cpus} CPUs busy (%)")))
        cpu_charts += figure_html(fig)
        top = stats[n]["max"]
        values = [v for _, v in summaries[n][top]["cpu_samples"]]
        cpu_items.append(f"<strong>{version[n]}</strong> at {top} workers: mean "
                         f"{statistics.mean(values):.0f}%, peak {max(values):.0f}%.")
    cpu_items.append("The low stretch at the start of every run is startup, which happens mostly in "
                     "a single process before any task begins.")
    cpu = (
        "<h2>CPU use over time</h2>\n"
        + note("The share of the pod's CPUs busy, sampled every "
               f"{BENCHMARK_CPU_SAMPLE_INTERVAL:g} s over one run per worker count.")
        + cpu_charts + observations(cpu_items))

    section_html = ('<section id="scaling">\n' f"{KATEX_CDN_HTML}\n"
                    + glance + method + timing + speedup + split + tasks + cpu + "</section>\n")
    output_path = os.path.join(BENCHMARK_REPORT_DIR, fragment_name)
    with open(output_path, "w") as fragment_file:
        fragment_file.write(section_html)

    print(f"Wrote {output_path} comparing {len(names)} datasets: {', '.join(names)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--report-only", action="store_true",
                        help="rebuild the fragment from saved results without running a sweep")
    args = parser.parse_args()

    if not args.report_only:
        run_sweep()

    # Every saved series, not just this sweep's, so sweeping one dataset or
    # machine keeps the others' pages. Series are grouped by machine: each
    # machine gets its own fragment, and comparisons stay within one machine.
    by_machine: dict[str, dict[str, list[dict]]] = {}
    for name, results in load_all_series(RESULTS_PREFIX).items():
        if aggregate_by_worker_count(results):
            by_machine.setdefault(name.rpartition("_")[2], {})[name] = results
    if not by_machine:
        print("No successful runs in any series -- nothing to report.")
        sys.exit(1)

    for machine, group in by_machine.items():
        fragment_name = (LAPTOP_FRAGMENT if machine == LAPTOP_MACHINE
                         else f"_performance_section_{machine}.html")
        if len(group) > 1 and machine != LAPTOP_MACHINE:
            build_comparison_report(group, fragment_name)
        else:
            build_report({name: aggregate_by_worker_count(results) for name, results in group.items()},
                         fragment_name)
