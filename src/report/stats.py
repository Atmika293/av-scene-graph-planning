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
    path = os.path.join(REPORTS_DIR, name)
    if not os.path.exists(path):
        return None
    with open(path) as fp:
        return fp.read()


def _read_json(name: str):
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
    return float(re.search(rf"{label}:\s*([-\d.]+)", text).group(1))


def _int(text: str, label: str) -> int:
    return int(re.search(rf"{label}:\s*(\d+)", text).group(1))


def sample_tokens() -> dict | None:
    text = _read("verify_sample_tokens.txt")
    if text is None:
        return None
    verified, total = _tally(text)
    return {"verified": verified, "total": total}


def ingestion_keys() -> dict | None:
    text = _read("verify_ingestion_keys.txt")
    if text is None:
        return None
    verified, total = _tally(text)
    return {"verified": verified, "total": total, "fields": len(TALLY_RE.findall(text))}


def table_rows() -> dict | None:
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


def projection() -> dict | None:
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
    data = _read_json("remap_check.json")
    if data is None:
        return None
    return {
        "passed": data["status"] == "PASS" and data["indexing_status"] == "PASS",
        "max_abs_diff": data["max_abs_diff_overall"],
        "threshold": data["pass_threshold"],
    }


def intensity() -> dict | None:
    text = _read("intensity_check.txt")
    if text is None:
        return None
    verified, total = _tally(text)
    return {"verified": verified, "total": total}


def raster() -> dict | None:
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
    text = _read("benchmark_results.jsonl")
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
def all_stats() -> dict:
    """Every reader's result, keyed by reader name (None for a report not yet generated)."""
    readers = [sample_tokens, ingestion_keys, table_rows, projection, box_containment,
               orientation, remap, intensity, raster, benchmark]
    return {reader.__name__: reader() for reader in readers}
