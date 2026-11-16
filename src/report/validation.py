"""Render the Validation page's chain-of-trust overview and per-stage check tables.

Each pipeline stage is checked against a reference independent of the code
under test before the next stage is allowed to build on it. CHECKS lists
every check in chain order: which stage it guards, the script, what it
compares against, the class of bug it catches, and how to read its result
out of report.stats -- so every number in the table comes from the
script's own latest reports/ output, never from hand-typed prose.

A check's status is "pass", "diagnostic" (a stat for flagging outliers,
with no pass/fail threshold of its own), or "not run" (its report doesn't
exist on this machine yet).
"""

from html import escape
from typing import Callable, NamedTuple

from report.html import REPO_URL
from report.stats import all_stats

SCRIPT_URL = f"{REPO_URL}/blob/main/scripts/validation/"

# (anchor, name, one-line summary of what the stage produces)
STAGES = [
    ("ingestion", "Ingestion", "nuScenes tables &rarr; synced per-sample records &rarr; Parquet rows"),
    ("projection", "Projection", "3D boxes &rarr; global &rarr; ego &rarr; camera &rarr; pixel"),
    ("bev", "BEV fusion", "LiDAR points + camera colour &rarr; top-down raster"),
    ("multisweep", "Multisweep", "Earlier sweeps + their own cameras &rarr; reference frame &rarr; merged raster"),
]


class Check(NamedTuple):
    """One row of the Validation page's check tables."""
    stage: str  # STAGES anchor this check belongs to
    name: str
    script: str
    reference: str
    catches: str
    stat: str  # report.stats reader name
    # stats dict -> (status, result text); only called when the report exists
    result: Callable[[dict], tuple[str, str]]


def _tally(s: dict, unit: str) -> tuple[str, str]:
    """Pass when every item verified; result text "verified / total unit"."""
    status = "pass" if s["verified"] == s["total"] else "fail"
    return status, f"{s['verified']:,} / {s['total']:,} {unit}"


def _projection_stages(s: dict) -> tuple[str, str]:
    """Pass when every projection stage matched the devkit fork on every corner set."""
    ok = all(v == t for v, t in s["stages"].values())
    return ("pass" if ok else "fail",
            f"{s['corner_sets']:,} / {s['corner_sets']:,} corner sets match at every stage")


def _projection_iou(s: dict) -> tuple[str, str]:
    """Pass when no final 2D box is missing or extra; result text leads with the lowest IoU."""
    ok = s["missing"] == 0 and s["extra"] == 0
    return ("pass" if ok else "fail",
            f"min IoU {s['iou_min']:.4f} over {s['iou_count']:,} boxes; "
            f"{s['missing']} missing, {s['extra']} extra")


def _orientation(s: dict) -> tuple[str, str]:
    """Pass when no moving scene has the vehicle appearing to drive backward."""
    ok = s["backward_fraction_max"] == 0
    stationary = (f"; {s['stationary_scenes']} parked scene skipped" if s["stationary_scenes"] == 1
                  else f"; {s['stationary_scenes']} parked scenes skipped" if s["stationary_scenes"] else "")
    return ("pass" if ok else "fail",
            f"mean cos(heading) {s['mean_cos_heading']:.4f} over {s['pairs']} sample pairs, "
            f"0 backward{stationary}")


def _separation(s: dict) -> tuple[str, str]:
    """Always "diagnostic": box separation flags outliers but has no pass/fail threshold."""
    return ("diagnostic",
            f"median gap ratio {s['separation_median']:.2f} over {s['separation_computed']} boxes "
            f"(lowest {s['separation_min']:.2f})")


def _remap(s: dict) -> tuple[str, str]:
    """Pass when colour sampling matched the hand-computed interpolation within threshold."""
    return ("pass" if s["passed"] else "fail",
            f"max difference {s['max_abs_diff']:.2f} (threshold {s['threshold']:.1f})")


def _raster(s: dict) -> tuple[str, str]:
    """Pass when every sample passes and every compared sample matches the naive implementation."""
    ok = s["passed"] == s["samples"] and s["naive_matches"] == s["naive_compared"]
    return ("pass" if ok else "fail",
            f"{s['passed']} / {s['samples']} samples pass; "
            f"matches naive implementation on {s['naive_matches']} / {s['naive_compared']} compared")


def _self_tested(passed: bool, text: str, caught: int, tests: int) -> tuple[str, str]:
    """Pass only when the comparison passed and every self-test's wrong input was caught."""
    ok = passed and caught == tests
    return "pass" if ok else "fail", f"{text}; self-test caught {caught} / {tests} wrong inputs"


def _camera_matching(s: dict) -> tuple[str, str]:
    """Pass when every sweep/camera pair matches brute force and the self-test catches a wrong frame."""
    return _self_tested(s["verified"] == s["total"],
                        f"{s['verified']:,} / {s['total']:,} sweep/camera pairs",
                        s["self_test_caught"], s["self_tests"])


def _sweep_alignment(s: dict) -> tuple[str, str]:
    """Pass when every point matches the devkit's accumulator and a sweep left unaligned is caught."""
    return _self_tested(s["aligned_points"] == s["points"],
                        f"{s['aligned_points']:,} / {s['points']:,} points across {s['sweeps']} sweeps "
                        f"(max error {s['max_error_mm']:.3f} mm)",
                        s["self_tests_caught"], s["self_tests"])


def _own_distance(s: dict) -> tuple[str, str]:
    """Pass when every point's own-ego distance matches the devkit's single-sweep loader."""
    ok = s["distance_ok"] == s["distance_total"]
    return "pass" if ok else "fail", f"{s['distance_ok']:,} / {s['distance_total']:,} points"


def _merge(s: dict) -> tuple[str, str]:
    """Pass when every merged cell matches the naive newest-first loop."""
    ok = s["merge_ok"] == s["merge_total"]
    return "pass" if ok else "fail", f"{s['merge_ok']:,} / {s['merge_total']:,} cells"


CHECKS = [
    Check("ingestion", "Scene &rarr; sample traversal", "verify_sample_tokens.py",
          "nuScenes' own <code>field2token</code> table scan",
          "Samples skipped or duplicated by walking each scene's linked list",
          "sample_tokens", lambda s: _tally(s, "scenes")),
    Check("ingestion", "Ingestion field mapping", "verify_ingestion_keys.py",
          "Raw devkit records, re-fetched independently and compared field by field",
          "Reading the wrong record or the wrong field (wiring bugs)",
          "ingestion_keys", lambda s: _tally(s, f"values across {s['fields']} fields")),
    Check("ingestion", "Table row construction", "verify_table_rows.py",
          "Target schema's column names + an independent re-indexer (not the code under test)",
          "Columns drifting from the schema; a row built by the wrong builder",
          "table_rows", lambda s: _tally(s, "columns + values + smoke tests")),
    Check("projection", "Per-stage transform", "evaluate_projections.py",
          "Intermediate corners from a fork of the devkit's 2D exporter "
          "(<code>export_2d_annotations_debug.py</code>)",
          "Frame-order, sign, or intrinsics errors, localised to the stage that introduced them",
          "projection", _projection_stages),
    Check("projection", "Final 2D boxes", "evaluate_projections.py",
          "nuScenes' <code>image_annotations.json</code>",
          "Clipping and in-front-of-camera errors; boxes missing or invented",
          "projection", _projection_iou),
    Check("bev", "Ego-frame orientation", "check_orientation.py",
          "The vehicle's real motion between consecutive samples (it drives forward, not sideways)",
          "A swapped or flipped axis in the ego-pose rotation handling",
          "orientation", _orientation),
    Check("bev", "LiDAR file columns", "check_intensity.py",
          "The devkit's own loader, <code>LidarPointCloud.from_file</code>",
          "Off-by-one column slicing (reading ring index as intensity)",
          "intensity", lambda s: _tally(s, "points")),
    Check("bev", "Points inside each box", "check_box_containment.py",
          "nuScenes' annotated <code>num_lidar_pts</code> for every box",
          "LiDAR points and boxes misaligned between frames",
          "box_containment", lambda s: _tally(s, "boxes")),
    Check("bev", "Box separation", "check_box_containment.py",
          "None: a nearest-neighbour gap ratio used to flag outliers",
          "Boxes sitting off their object (flagged for inspection, not failed)",
          "box_containment", _separation),
    Check("bev", "Camera colour sampling", "check_remap.py",
          "Bilinear interpolation computed by hand from the 4 neighbouring pixels",
          "Pixel-centre convention mismatch; transposed image indexing",
          "remap", _remap),
    Check("bev", "Raster channels", "check_raster.py",
          "A naive per-point loop implementation + sentinel/mask consistency rules",
          "Empty-cell sentinels leaking into real cells; mask mismatches",
          "raster", _raster),
    Check("multisweep", "Camera frame per sweep", "check_camera_matching.py",
          "The nearest frame by brute force, over an independent scan of nuScenes' "
          "<code>sample_data</code> table",
          "A frame chain walked wrong (skipped, repeated, or from another scene); "
          "an off-by-one in the binary search; matching a frame past the max time gap",
          "camera_matching", _camera_matching),
    Check("multisweep", "Sweep alignment", "check_multisweep.py",
          "The devkit's own sweep accumulator, <code>LidarPointCloud.from_file_multisweep</code>",
          "An older sweep left in its own ego frame or moved with the wrong pose; "
          "a sweep paired with the wrong file",
          "multisweep", _sweep_alignment),
    Check("multisweep", "Distance used for nearest point", "check_multisweep.py",
          "Each sweep's points from the devkit's single-sweep loader, in that sweep's own ego frame",
          "Picking each cell's nearest point by distance from the reference vehicle position "
          "instead of the sweep's own",
          "multisweep", _own_distance),
    Check("multisweep", "Merge rule", "check_multisweep.py",
          "A naive loop: per cell, every channel from the newest sweep with a point there",
          "An older sweep overwriting a newer one; one cell mixing channels from different sweeps",
          "multisweep", _merge),
]

STATUS_LABEL = {"pass": "Pass", "fail": "Fail", "diagnostic": "Diagnostic", "notrun": "Not run"}


def _evaluate(check: Check) -> tuple[str, str]:
    """(status, result text) for one check, or "notrun" if its report doesn't exist yet."""
    stats = all_stats()[check.stat]
    if stats is None:
        return "notrun", "Report not generated yet"
    return check.result(stats)


def render() -> str:
    """The chain overview plus one check table per stage, as one <section>."""
    results = {check: _evaluate(check) for check in CHECKS}

    links = []
    for i, (anchor, name, summary) in enumerate(STAGES, start=1):
        stage_results = [status for check, (status, _) in results.items() if check.stage == anchor]
        # Diagnostics have no pass/fail, so they're counted separately rather
        # than shown as a shortfall in "N / M pass".
        graded = [s for s in stage_results if s != "diagnostic"]
        diagnostics = len(stage_results) - len(graded)
        tally = f"<strong>{graded.count('pass')} / {len(graded)}</strong> checks pass"
        if diagnostics:
            tally += f" &middot; {diagnostics} diagnostic"
        if i > 1:
            links.append('<span class="arrow" aria-hidden="true">&rarr;</span>')
        links.append(
            f'<a class="link" href="#{anchor}"><div class="n">Stage {i}</div>'
            f'<div class="name">{name}</div><div class="sub">{summary}</div>'
            f'<div class="sub">{tally}</div></a>'
        )

    tables = []
    for anchor, name, summary in STAGES:
        rows = []
        for check, (status, text) in results.items():
            if check.stage != anchor:
                continue
            badge_class = "fail" if status == "fail" else status
            rows.append(
                f"<tr><td><strong>{check.name}</strong><br>"
                f'<a href="{SCRIPT_URL}{check.script}"><code>{escape(check.script)}</code></a></td>'
                f"<td>{check.reference}</td><td>{check.catches}</td>"
                f'<td class="result"><span class="badge {badge_class}">{STATUS_LABEL[status]}</span><br>{text}</td></tr>'
            )
        tables.append(
            f'<h3 id="{anchor}">{name}</h3>\n<p class="chart-note">{summary}</p>\n'
            '<div class="table-scroll"><table class="checks">\n'
            "<tr><th>Check</th><th>Compared against</th><th>Catches</th><th>Latest result</th></tr>\n"
            + "\n".join(rows) + "\n</table></div>"
        )

    return ('<section id="checks">\n<h2>Checks by stage</h2>\n<div class="chain">' + "".join(links) + "</div>\n"
            + "\n".join(tables) + "\n</section>")
