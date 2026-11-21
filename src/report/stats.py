"""Read every validation/benchmark script's reports/ output into one flat dict of numbers.

Used by report.validation (the Validation page's check table) and by
report.markup's {{stat:...}} directive (headline numbers quoted in the
hand-written pages), so a number quoted in prose and the same number in the
check table always come from one parse of the same file -- never typed in
by hand, so they can't drift from the latest run.

Each reader returns None if its report file doesn't exist yet (the check
hasn't been run on this machine); consumers render that as "not run" rather
than failing the whole build.
"""

import json
import os
import re
import statistics
from functools import cache

REPORTS_DIR = "reports"

# "Verified: 10/10", "SAMPLE_TOKEN: 40/40", "sample: 40/40" -- every tally
# line the verify_*/check_* scripts write has this shape.
TALLY_RE = re.compile(r"^\s*([^:\n]+):\s*(\d+)/(\d+)\s*$", re.MULTILINE)


def _read(name: str) -> str | None:
    """A reports/ file's text, or None if that report hasn't been generated yet."""
    path = os.path.join(REPORTS_DIR, name)
    if not os.path.exists(path):
        return None
    with open(path) as fp:
        return fp.read()


def _read_json(name: str):
    """A reports/ JSON file, parsed, or None if it hasn't been generated yet."""
    text = _read(name)
    return None if text is None else json.loads(text)


def _tally(text: str, section: str | None = None) -> tuple[int, int]:
    """Sum every 'label: verified/total' line, optionally only within one '--- section ---' block."""
    if section is not None:
        match = re.search(rf"--- {re.escape(section)}[^\n]*---\n(.*?)(?=\n--- |\Z)", text, re.DOTALL)
        text = match.group(1) if match else ""
    pairs = [(int(v), int(t)) for _, v, t in TALLY_RE.findall(text)]
    return sum(v for v, _ in pairs), sum(t for _, t in pairs)


def _float(text: str, label: str) -> float:
    """The number after the first "label:" in text (label is a regex)."""
    return float(re.search(rf"{label}:\s*([-\d.]+)", text).group(1))


def _int(text: str, label: str) -> int:
    """The integer after the first "label:" in text (label is a regex)."""
    return int(re.search(rf"{label}:\s*(\d+)", text).group(1))


def sample_tokens() -> dict | None:
    """verify_sample_tokens.py: scenes whose linked-list walk matches field2token."""
    text = _read("verify_sample_tokens.txt")
    if text is None:
        return None
    verified, total = _tally(text)
    return {"verified": verified, "total": total}


def ingestion_keys() -> dict | None:
    """verify_ingestion_keys.py: field values matching the raw devkit records, and how many fields were checked."""
    text = _read("verify_ingestion_keys.txt")
    if text is None:
        return None
    verified, total = _tally(text)
    return {"verified": verified, "total": total, "fields": len(TALLY_RE.findall(text))}


def table_rows() -> dict | None:
    """verify_table_rows.py: overall tally, plus the column-set, per-value and smoke-test sections separately."""
    text = _read("verify_table_rows.txt")
    if text is None:
        return None
    verified, total = _tally(text)
    return {
        "verified": verified,
        "total": total,
        "columns": _tally(text, "Column-set vs. schema"),
        "values": _tally(text, "Per-column values"),
        "smoke": _tally(text, "Orchestration-layer smoke test"),
    }


def table_counts() -> dict | None:
    """verify_table_counts.py: overall tally across every table check, and the rows read."""
    text = _read("verify_table_counts.txt")
    if text is None:
        return None
    verified, total = _tally(text)
    rows = sum(int(n) for n in re.findall(r"^Rows read: (\d+)$", text, re.MULTILINE))
    tables = len(re.findall(r"^--- ", text, re.MULTILINE))
    return {"verified": verified, "total": total, "rows": rows, "tables": tables}


def projection() -> dict | None:
    """evaluate_projections.py: per-stage corner tallies, final-box IoU stats, and missing/extra box counts."""
    text = _read("evaluate_projections.txt")
    if text is None:
        return None
    stages = {name: _tally(text, name) for name in
              ("ann -> global", "global -> camera", "camera -> sensor (pre-clip)", "clip -> canvas (bbox)")}
    return {
        "stages": stages,
        "corner_sets": stages["global -> camera"][1],
        "iou_count": _int(text, "Count"),
        "iou_min": _float(text, "Min"),
        "iou_mean": _float(text, "Mean"),
        "agreed_none": _int(text, r"Agreed no bbox \(both None\)"),
        "missing": _int(text, r"Missing in ours[^:]*"),
        "extra": _int(text, r"Extra in ours[^:]*"),
        "skipped_no_lidar": _int(text, "Skipped annotations"),
    }


def box_containment() -> dict | None:
    """check_box_containment.py: boxes whose point count matches num_lidar_pts, plus the separation diagnostic."""
    text = _read("box_containment_check.txt")
    if text is None:
        return None
    verified, total = _tally(text, "Count check")
    computed = re.search(r"Computed:\s*(\d+)/(\d+)", text)
    return {
        "verified": verified,
        "total": total,
        "separation_computed": int(computed.group(1)),
        "separation_median": _float(text, "Median"),
        "separation_min": _float(text, "Min"),
    }


def orientation() -> dict | None:
    """check_orientation.py: how well the ego +x axis tracks the vehicle's real motion, across moving scenes."""
    scenes = _read_json("orientation_check.json")
    if scenes is None:
        return None
    # A scene where the ego never moves past check_orientation's motion
    # threshold (parked the whole time) has no heading to measure -- its
    # cosine/backward fields are None, so it's counted but not averaged.
    moving = [s for s in scenes if s["pairs_with_motion"]]
    pairs = sum(s["pairs_with_motion"] for s in moving)
    return {
        "scenes": len(scenes),
        "stationary_scenes": len(scenes) - len(moving),
        "pairs": pairs,
        # Weighted by each scene's own pair count, so a short scene doesn't
        # count as much as a long one.
        "mean_cos_heading": sum(s["mean_cos_heading"] * s["pairs_with_motion"] for s in moving) / pairs,
        "worst_scene_cos_heading": min(s["mean_cos_heading"] for s in moving),
        "backward_fraction_max": max(s["backward_fraction"] for s in moving),
    }


def remap() -> dict | None:
    """check_remap.py: whether colour sampling matches hand-computed bilinear interpolation, and the largest difference."""
    data = _read_json("remap_check.json")
    if data is None:
        return None
    return {
        "passed": data["status"] == "PASS" and data["indexing_status"] == "PASS",
        "max_abs_diff": data["max_abs_diff_overall"],
        "threshold": data["pass_threshold"],
    }


def intensity() -> dict | None:
    """check_intensity.py: LiDAR points whose columns match the devkit's own loader."""
    text = _read("intensity_check.txt")
    if text is None:
        return None
    verified, total = _tally(text)
    return {"verified": verified, "total": total}


def raster() -> dict | None:
    """check_raster.py: samples passing the consistency rules, and matches against the naive implementation."""
    text = _read("raster_check.txt")
    if text is None:
        return None
    results = re.findall(r"^Sample \w+: (\w+)$", text, re.MULTILINE)
    return {
        "passed": results.count("PASS"),
        "samples": len(results),
        "naive_matches": text.count(": MATCH"),
        "naive_compared": len(re.findall(r"vs\. naive reference implementation", text)),
    }


def benchmark() -> dict | None:
    """benchmark.py: mean wall time per worker count, reduced to the headline speedup and startup share.

    Reads the v1.0-mini sweep on the laptop, the series the site's headline numbers quote;
    other series (another dataset or machine) are drawn on the Ray scaling page alongside it.
    """
    text = _read("benchmark_results_v1.0-mini_laptop.jsonl")
    if text is None:
        return None
    by_wc: dict[int, list[float]] = {}
    startup_fraction: dict[int, list[float]] = {}
    for line in text.splitlines():
        run = json.loads(line)
        if run["success"]:
            by_wc.setdefault(run["worker_count"], []).append(run["wall_time"])
            startup_fraction.setdefault(run["worker_count"], []).append(
                run["startup_overhead"] / run["wall_time"])
    means = {wc: statistics.mean(times) for wc, times in by_wc.items()}
    baseline = min(means)
    best = min(means, key=means.get)
    return {
        "worker_counts": "/".join(str(wc) for wc in sorted(means)),
        # Fixed per-run cost (interpreter, imports, ray.init(), devkit table
        # load) as a share of the whole run, at the best worker count --
        # the part of the runtime more workers can't shrink.
        "best_startup_fraction": statistics.mean(startup_fraction[best]),
        "baseline_workers": baseline,
        "best_workers": best,
        "best_speedup": means[baseline] / means[best],
        "max_workers": max(means),
        "max_workers_speedup": means[baseline] / means[max(means)],
    }


@cache
def camera_matching() -> dict | None:
    """check_camera_matching.py: sweep-camera pairs matching the brute-force nearest frame, plus its self-test."""
    text = _read("camera_matching_check.txt")
    if text is None:
        return None
    verified, total = _tally(text, "Nearest-frame check")
    caught, tests = _tally(text, "Self-test")
    return {"verified": verified, "total": total, "self_test_caught": caught, "self_tests": tests}


def multisweep() -> dict | None:
    """check_multisweep.py: alignment vs. the devkit's accumulator, own-ego distance, merge rule, self-tests.

    Alignment's tally is points only (its first line); the "Sweeps fully
    aligned" line restates the same result per sweep, so it isn't summed in.
    """
    text = _read("multisweep_check.txt")
    if text is None:
        return None
    alignment = re.search(r"Points within [^:]*, same intensity:\s*(\d+)/(\d+)", text)
    sweeps = re.search(r"Sweeps fully aligned:\s*(\d+)/(\d+)", text)
    distance = _tally(text, "Own-ego distance")
    merge = _tally(text, "Merge rule")
    self_tests = _tally(text, "Self-tests")
    return {
        "aligned_points": int(alignment.group(1)), "points": int(alignment.group(2)),
        "aligned_sweeps": int(sweeps.group(1)), "sweeps": int(sweeps.group(2)),
        "max_error_mm": float(re.search(r"Max alignment error \(mm\) = ([\d.]+)", text).group(1)),
        "distance_ok": distance[0], "distance_total": distance[1],
        "merge_ok": merge[0], "merge_total": merge[1],
        "self_tests_caught": self_tests[0], "self_tests": self_tests[1],
    }


def task_layout() -> dict | None:
    """benchmark_task_layout.py: 2 vs. 4 Ray tasks per batch, after startup, plus where task compute goes.

    "Within noise" uses the same rule as benchmark_task_layout.py's verdict
    column: the difference is smaller than the two layouts' stdevs added.
    """
    text = _read("task_layout_results_v1.0-mini_laptop.jsonl")  # the published sweep
    if text is None:
        return None
    work: dict[tuple[str, int], list[float]] = {}
    bbox: dict[tuple[str, int], list[float]] = {}
    core: dict[tuple[str, int], list[float]] = {}
    for line in text.splitlines():
        run = json.loads(line)
        if not run["success"] or run["startup_overhead"] is None:
            continue
        key = (run["task_layout"], run["worker_count"])
        work.setdefault(key, []).append(run["wall_time"] - run["startup_overhead"])
        bbox.setdefault(key, []).append(sum(t["compute_time"] for t in run["task_timings"]
                                            if t["label"].endswith("write_bboxes")))
        core.setdefault(key, []).append(sum(t["compute_time"] for t in run["task_timings"]
                                            if not t["label"].endswith("write_bboxes")))

    worker_counts = sorted(wc for layout, wc in work if layout == "consolidated" and ("split", wc) in work)
    diff_pct, faster, noise = {}, [], []
    for wc in worker_counts:
        cons, split = work[("consolidated", wc)], work[("split", wc)]
        diff = statistics.mean(split) - statistics.mean(cons)
        diff_pct[wc] = diff / statistics.mean(cons)
        (noise if abs(diff) < statistics.stdev(cons) + statistics.stdev(split) else faster).append(wc)
    shares = [statistics.mean(bbox[key]) / (statistics.mean(bbox[key]) + statistics.mean(core[key]))
              for key in work]
    core_means = [statistics.mean(core[key]) for key in work]

    def joined(counts: list[int]) -> str:
        """Worker counts as prose: "4", "4 and 8", "1, 2 and 4"."""
        return " and ".join([", ".join(map(str, counts[:-1])), str(counts[-1])]) if len(counts) > 1 \
            else "".join(map(str, counts))

    return {
        "repeats": min(len(runs) for runs in work.values()),
        "worker_counts": "/".join(map(str, worker_counts)),
        "diff_pct_max": max(diff_pct.values()),
        "faster_workers": joined(faster),
        "noise_workers": joined(noise),
        "bbox_share_min": min(shares),
        "bbox_share_max": max(shares),
        "core_compute_min": min(core_means),
        "core_compute_max": max(core_means),
        "bbox_compute_fewest_workers": statistics.mean(bbox[("consolidated", worker_counts[0])]),
        "bbox_compute_most_workers": statistics.mean(bbox[("consolidated", worker_counts[-1])]),
        "fewest_workers": worker_counts[0],
        "most_workers": worker_counts[-1],
    }


def all_stats() -> dict:
    """Every reader's result, keyed by reader name (None for a report not yet generated)."""
    readers = [sample_tokens, ingestion_keys, table_rows, table_counts, projection, box_containment,
               orientation, remap, intensity, raster, camera_matching, multisweep,
               benchmark, task_layout]
    return {reader.__name__: reader() for reader in readers}
