"""Validate the multisweep window: sweep alignment, per-point distance, and the merge rule.

For every scene's window (bev.multisweep.assemble_window +
project_window_to_bev, the code the BEV page uses), three comparisons, each
against something independent of the code under test:

  1. Sweep alignment. Every sweep's points, in the reference sweep's ego
     frame, are compared with the devkit's own accumulator,
     LidarPointCloud.from_file_multisweep, which reads the same sweeps and
     chains the same poses with its own (pyquaternion, 4x4 matrix) code.
     The devkit returns points in the reference *sensor* frame, so they're
     moved into the reference ego frame with that sensor's calibration. Both
     lists are newest first, sweep by sweep, so points are compared one to
     one: position within ALIGNMENT_TOLERANCE_M, and the same intensity (so a
     sweep paired with the wrong file can't pass on position alone).
     min_distance=0 turns off the devkit's near-point filter, which this
     pipeline doesn't apply.
  2. Own-ego distance. build_raster keeps the point nearest the sensor in
     each cell, so each sweep must report each point's distance from *its
     own* ego origin, not the reference's. Compared with the devkit's
     single-sweep loader moved into that sweep's own ego frame.
  3. Merge rule. bev.raster.merge_rasters is compared cell by cell with a
     naive loop: walk the sweeps newest first and take every channel from
     the first sweep that has a point in that cell; a cell no sweep reached
     stays empty, with source -1.

Two self-tests confirm the comparisons can fail: one sweep left in its own
ego frame (the classic missing-transform bug) must break alignment, and an
oldest-first merge must break the merge comparison.

Running this script as __main__ writes reports/multisweep_check.txt
(overwritten each run). Nothing is printed to stdout.

Run with: python scripts/validation/check_multisweep.py (from the repo root).
"""

import os

import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from pyquaternion import Quaternion

from av_data_pipeline.config import NUSCENES_VERSION, NUSCENES_DATAROOT
from av_data_pipeline.ingestion import extract_sensor_data_from_sample_data
from bev.lidar import LidarProjection
from bev.multisweep import WINDOW_LENGTH, assemble_window, project_window_to_bev
from bev.raster import build_raster, merge_rasters

REPORT_DIR = "reports"
REPORT_FILE = os.path.join(REPORT_DIR, "multisweep_check.txt")
LIDAR_CHANNEL = "LIDAR_TOP"
# Same grid as check_raster.py and the BEV page.
GRID_HALF_EXTENT_METERS = 25.0
CELL_SIZE_METERS = 0.25
# The devkit transforms float32 points through the global frame (hundreds of
# metres from the origin), so its own rounding is around 0.1 mm; 1 mm leaves
# room for that while still catching any real frame error, which is metres.
ALIGNMENT_TOLERANCE_M = 1e-3
MERGE_CHANNELS = ("height", "intensity", "color", "colored_mask", "occupied_mask", "source")


def sensor_to_ego(points: np.array, calibrated_sensor: dict) -> np.array:
    """Move (N, 3) sensor-frame points into the ego frame with a calibrated_sensor record."""
    rotation = Quaternion(calibrated_sensor["rotation"]).rotation_matrix
    return points @ rotation.T + np.array(calibrated_sensor["translation"])


def devkit_window(nusc: NuScenes, reference_token: str, sweep_count: int) -> tuple[np.array, np.array, list[int]]:
    """The devkit's accumulated window, in the reference ego frame.

    Args:
        nusc: NuScenes devkit instance.
        reference_token: The reference (keyframe) LIDAR_TOP sample_data token.
        sweep_count: How many sweeps to accumulate (the window's length).

    Returns:
        (points (N, 3) in the reference ego frame, intensity (N,), points per
        sweep, newest first).
    """
    reference = nusc.get("sample_data", reference_token)
    sample = nusc.get("sample", reference["sample_token"])
    cloud, times = LidarPointCloud.from_file_multisweep(
        nusc, sample, LIDAR_CHANNEL, LIDAR_CHANNEL, nsweeps=sweep_count, min_distance=0.0)
    calibration = nusc.get("calibrated_sensor", reference["calibrated_sensor_token"])
    points = sensor_to_ego(cloud.points[:3].T, calibration)
    # Each sweep's points share one time lag and were appended in order, so
    # sweep boundaries are where the lag changes.
    lags = times[0]
    boundaries = np.concatenate([[0], np.flatnonzero(np.diff(lags)) + 1, [len(lags)]])
    return points, cloud.points[3], np.diff(boundaries).tolist()


def devkit_own_ego_distance(nusc: NuScenes, sweep_token: str) -> np.array:
    """Each point's distance from its own sweep's ego origin, via the devkit's loader."""
    record = nusc.get("sample_data", sweep_token)
    cloud = LidarPointCloud.from_file(os.path.join(nusc.dataroot, record["filename"]))
    calibration = nusc.get("calibrated_sensor", record["calibrated_sensor_token"])
    return np.linalg.norm(sensor_to_ego(cloud.points[:3].T, calibration), axis=1)


def naive_merge(rasters: list[dict]) -> dict:
    """Newest-first merge, one cell at a time: every channel from the first sweep with a point there."""
    shape = rasters[0]["occupied_mask"].shape
    merged = {
        "height": np.full(shape, np.nan), "intensity": np.full(shape, np.nan),
        "color": np.zeros((*shape, 3), dtype=rasters[0]["color"].dtype),
        "colored_mask": np.zeros(shape, dtype=bool), "occupied_mask": np.zeros(shape, dtype=bool),
        "source": np.full(shape, -1),
    }
    for row in range(shape[0]):
        for col in range(shape[1]):
            for index, raster in enumerate(rasters):
                if raster["occupied_mask"][row, col]:
                    for channel in ("height", "intensity", "color", "colored_mask", "occupied_mask"):
                        merged[channel][row, col] = raster[channel][row, col]
                    merged["source"][row, col] = index
                    break
    return merged


def merge_mismatches(merged: dict, reference: dict) -> np.array:
    """Boolean (H, W) mask of cells where any channel differs (NaN equal to NaN)."""
    bad = np.zeros(reference["occupied_mask"].shape, dtype=bool)
    for channel in MERGE_CHANNELS:
        ours, theirs = merged[channel], reference[channel]
        if ours.dtype.kind == "f" or theirs.dtype.kind == "f":
            differs = ~np.isclose(ours, theirs, equal_nan=True)
        else:
            differs = ours != theirs
        bad |= differs.any(axis=-1) if differs.ndim == 3 else differs
    # Empty cells' colour is a placeholder in both; only compare where occupied.
    return bad & (merged["occupied_mask"] | reference["occupied_mask"])


if __name__ == "__main__":
    nusc = NuScenes(version=NUSCENES_VERSION, dataroot=NUSCENES_DATAROOT, verbose=False)

    alignment = {"points": [0, 0], "sweeps": [0, 0]}
    distance = [0, 0]
    merge = [0, 0]
    max_error_m = 0.0
    problems: list[str] = []
    self_test_alignment = None
    self_test_merge = None

    for scene in nusc.scene:
        sweep_tokens, camera_groups = assemble_window(nusc, scene["token"])
        if not sweep_tokens:
            continue
        sweeps = project_window_to_bev(nusc, sweep_tokens, camera_groups)

        # 1. Alignment, sweep by sweep, against the devkit's accumulator.
        reference_points, reference_intensity, per_sweep = devkit_window(nusc, sweep_tokens[0], len(sweep_tokens))
        offsets = np.concatenate([[0], np.cumsum(per_sweep)])
        if len(per_sweep) != len(sweeps):
            problems.append(f"{scene['name']}: devkit window has {len(per_sweep)} sweeps, ours {len(sweeps)}")
        for index, sweep in enumerate(sweeps[:len(per_sweep)]):
            theirs = reference_points[offsets[index]:offsets[index + 1]]
            alignment["sweeps"][1] += 1
            if len(theirs) != len(sweep["points"]):
                alignment["points"][1] += len(theirs)
                problems.append(f"{scene['name']} sweep {index}: {len(sweep['points'])} points, devkit {len(theirs)}")
                continue
            error = np.linalg.norm(sweep["points"] - theirs, axis=1)
            same_intensity = sweep["intensity"] == reference_intensity[offsets[index]:offsets[index + 1]]
            ok = (error <= ALIGNMENT_TOLERANCE_M) & same_intensity
            alignment["points"][0] += int(ok.sum())
            alignment["points"][1] += len(ok)
            alignment["sweeps"][0] += int(ok.all())
            max_error_m = max(max_error_m, float(error.max()))
            if not ok.all():
                problems.append(f"{scene['name']} sweep {index}: {int((~ok).sum())} points off "
                                f"(max error {error.max():.4f} m)")

            # 2. Distance from the sweep's own ego origin.
            own = devkit_own_ego_distance(nusc, sweep_tokens[index])
            close = np.abs(sweep["distance"] - own) <= ALIGNMENT_TOLERANCE_M
            distance[0] += int(close.sum())
            distance[1] += len(close)

            # Self-test material: an older sweep left in its own ego frame.
            if self_test_alignment is None and index > 0:
                unaligned = LidarProjection(extract_sensor_data_from_sample_data(nusc, sweep_tokens[index])).lidar_to_ego()
                self_test_alignment = bool((np.linalg.norm(unaligned - theirs, axis=1) > ALIGNMENT_TOLERANCE_M).any())

        # 3. Merge rule, cell by cell.
        rasters = [build_raster(sweep["points"], sweep["colors"], sweep["intensity"],
                                GRID_HALF_EXTENT_METERS, CELL_SIZE_METERS, distance=sweep["distance"])
                   for sweep in sweeps]
        merged = merge_rasters(rasters)
        expected = naive_merge(rasters)
        bad = merge_mismatches(merged, expected)
        merge[0] += int(bad.size - bad.sum())
        merge[1] += int(bad.size)
        if bad.any():
            problems.append(f"{scene['name']}: merge differs from the naive loop in {int(bad.sum())} cells")
        if self_test_merge is None and len(rasters) > 1:
            self_test_merge = bool(merge_mismatches(merge_rasters(rasters[::-1]), expected).any())

    report = [
        "--- Sweep alignment: project_window_to_bev vs. devkit LidarPointCloud.from_file_multisweep ---",
        f"Points within {ALIGNMENT_TOLERANCE_M * 1000:.0f} mm, same intensity: {alignment['points'][0]}/{alignment['points'][1]}",
        f"Sweeps fully aligned: {alignment['sweeps'][0]}/{alignment['sweeps'][1]}",
        f"Max alignment error (mm) = {max_error_m * 1000:.3f}",
        f"Window length (bev.multisweep.WINDOW_LENGTH) = {WINDOW_LENGTH}",
        "",
        "--- Own-ego distance: per-point distance vs. devkit single-sweep loader ---",
        f"Points within {ALIGNMENT_TOLERANCE_M * 1000:.0f} mm: {distance[0]}/{distance[1]}",
        "",
        "--- Merge rule: merge_rasters vs. naive newest-first per-cell loop ---",
        f"Cells matching: {merge[0]}/{merge[1]}",
        "",
        "--- Self-tests: deliberately wrong input must be reported ---",
        f"Sweep left in its own ego frame detected: {int(bool(self_test_alignment))}/1",
        f"Oldest-first merge detected: {int(bool(self_test_merge))}/1",
    ]
    if problems:
        report += ["", f"--- Problems ({len(problems)}) ---"] + [f"  {problem}" for problem in problems[:20]]

    os.makedirs(REPORT_DIR, exist_ok=True)
    with open(REPORT_FILE, "w") as outfile:
        outfile.write("\n".join(report) + "\n")
