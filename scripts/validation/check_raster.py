"""Validates bev.raster.build_raster (#5, the raster-height check): each
occupied cell's values should be plausible, with no sentinel leakage into
real cells or vice versa -- plus an independent per-point-loop
cross-check (build_raster_naive) of build_raster's own vectorized
nearest-to-ego averaging.

build_raster itself (the binning/averaging logic) now lives in
src/bev/raster.py, since scripts/report/build_bev_report.py also needs it
for the published page -- both import the exact same validated code rather
than risking two copies drifting apart.

Run with: python scripts/validation/check_raster.py (from the repo root, after
ingesting/having local access to the configured nuScenes dataset).
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from nuscenes.nuscenes import NuScenes

from av_data_pipeline.config import NUSCENES_VERSION, NUSCENES_DATAROOT
from av_data_pipeline.ingestion import build_scene_data
from av_data_pipeline.ingestion_keys import SAMPLE_INGESTION_KEYS
from bev.fusion import GeometricBEVPipeline
from bev.raster import to_plot_xy, build_raster

REPORT_DIR = "reports"
IMAGE_DIR = "reports/raster_images"
REPORT_FILE = os.path.join(REPORT_DIR, "raster_check.txt")
DATA_FILE = os.path.join(REPORT_DIR, "raster_check_data.json")
NUM_SAMPLES = 3
# Grid bounds/resolution, both configurable. Measured planar range across
# this scene's samples: median ~6m, p90 ~29m, p95 ~40m, max ~100m+ (a thin
# tail of far reflective returns) -- density isn't remotely uniform across a
# 100m-wide grid, so most of a wide grid's area sits empty regardless of
# cell size. 25m keeps most of the meaningfully dense data while focusing
# the grid on the ego's actual neighborhood instead of its mostly-empty
# full sensor range.
GRID_HALF_EXTENT_METERS = 25.0
CELL_SIZE_METERS = 0.25
# build_raster_naive is a per-point Python loop, not vectorized -- runs
# against only this many of NUM_SAMPLES' samples (the first N), not the
# full sweep.
NUM_NAIVE_CHECK_SAMPLES = 1


def build_raster_naive(ego_points: np.array, colors: np.array, intensity: np.array,
                        half_extent: float, cell_size: float) -> dict:
    """Reference implementation of build_raster's binning rule, for cross-checking it.

    build_raster's nearest-to-ego group selection relies on a non-obvious
    vectorized trick (np.minimum.at/np.add.at scatter-reduce). This
    reimplements the same rule the plain way instead -- a per-point loop
    collecting every candidate per cell, then an explicit min-distance
    filter and average -- so the two can be compared as independent
    implementations, the same way points_in_box and the remap check are
    each verified against an independent source. A Python-level loop over
    every point doesn't scale, which is why this is only ever run against a
    single sample (see NUM_NAIVE_CHECK_SAMPLES), not the full validation
    sweep.

    Args:
        Same as build_raster.

    Returns:
        Same shape as build_raster's result, for direct comparison.
    """
    plot_x, plot_y = to_plot_xy(ego_points)
    grid_size = int(round(2 * half_extent / cell_size))

    candidates = {}  # (row, col) -> list of (distance, colored, color, height, intensity)
    for i in range(ego_points.shape[0]):
        col = int(np.floor((plot_x[i] + half_extent) / cell_size))
        row = int(np.floor((plot_y[i] + half_extent) / cell_size))
        if not (0 <= col < grid_size and 0 <= row < grid_size):
            continue

        dist = float(np.linalg.norm(ego_points[i]))
        colored = bool(np.all(colors[i] != -1))
        candidates.setdefault((row, col), []).append(
            (dist, colored, colors[i].astype(float), float(ego_points[i, 2]), float(intensity[i])))

    occupied_mask = np.zeros((grid_size, grid_size), dtype=bool)
    colored_mask = np.zeros((grid_size, grid_size), dtype=bool)
    color = np.zeros((grid_size, grid_size, 3), dtype=np.uint8)
    height = np.full((grid_size, grid_size), np.nan)
    intensity_channel = np.full((grid_size, grid_size), np.nan)

    for (row, col), points in candidates.items():
        min_dist = min(p[0] for p in points)
        # Average every point tied for nearest in this cell -- repeated
        # measurements of one physical spot (a real lidar multi-return
        # artifact, not floating-point noise), not different surfaces.
        nearest = [p for p in points if p[0] == min_dist]

        occupied_mask[row, col] = True
        height[row, col] = np.mean([p[3] for p in nearest])
        intensity_channel[row, col] = np.mean([p[4] for p in nearest])

        colored_members = [p for p in nearest if p[1]]
        if colored_members:
            colored_mask[row, col] = True
            color[row, col] = np.round(np.mean([p[2] for p in colored_members], axis=0)).astype(np.uint8)

    return {
        "occupied_mask": occupied_mask,
        "colored_mask": colored_mask,
        "color": color,
        "height": height,
        "intensity": intensity_channel,
        "extent": (-half_extent, half_extent, -half_extent, half_extent),
    }


def compare_rasters(a: dict, b: dict) -> dict:
    """Cell-by-cell comparison of two build_raster-shaped results.

    Args:
        a: A build_raster() result.
        b: A build_raster_naive() result (or another build_raster() result).

    Returns:
        A dict of exact-match booleans per channel.
    """
    return {
        "occupied_mask_matches": bool(np.array_equal(a["occupied_mask"], b["occupied_mask"])),
        "colored_mask_matches": bool(np.array_equal(a["colored_mask"], b["colored_mask"])),
        "color_matches": bool(np.array_equal(a["color"], b["color"])),
        "height_matches": bool(np.array_equal(a["height"], b["height"], equal_nan=True)),
        "intensity_matches": bool(np.array_equal(a["intensity"], b["intensity"], equal_nan=True)),
    }


def validate_raster(raster: dict) -> dict:
    """Raster-height check (#5): sanity-check each channel's sentinel/value consistency.

    Args:
        raster: A build_raster() result.

    Returns:
        A dict of pass/fail booleans plus the observed height/intensity
        ranges, for both printing and persisting.
    """
    occupied = raster["occupied_mask"]
    colored = raster["colored_mask"]
    height = raster["height"]
    intensity = raster["intensity"]
    color = raster["color"]

    no_nan_leak_into_occupied = bool(not np.isnan(height[occupied]).any() and
                                      not np.isnan(intensity[occupied]).any())
    no_real_value_outside_occupied = bool(np.isnan(height[~occupied]).all() and
                                           np.isnan(intensity[~occupied]).all())
    colored_implies_occupied = bool(np.all(occupied[colored]))
    # color has no embedded sentinel (uint8, validity lives in colored_mask
    # alone) -- this instead confirms the zero-placeholder substitution in
    # build_raster actually ran for every uncolored cell, not leftover
    # garbage from a future refactor that skips it.
    uncolored_cells_are_zeroed = bool(np.all(color[~colored] == 0))

    return {
        "no_nan_leak_into_occupied_cells": no_nan_leak_into_occupied,
        "no_real_value_outside_occupied_cells": no_real_value_outside_occupied,
        "colored_implies_occupied": colored_implies_occupied,
        "uncolored_cells_are_zeroed": uncolored_cells_are_zeroed,
        "num_occupied_cells": int(occupied.sum()),
        "num_colored_cells": int(colored.sum()),
        "fill_fraction": float(occupied.mean()),
        "height_min": float(np.nanmin(height)) if occupied.any() else None,
        "height_max": float(np.nanmax(height)) if occupied.any() else None,
        "height_mean": float(np.nanmean(height)) if occupied.any() else None,
        "intensity_min": float(np.nanmin(intensity)) if occupied.any() else None,
        "intensity_max": float(np.nanmax(intensity)) if occupied.any() else None,
    }


def render_raster(raster: dict, title: str) -> "plt.Figure":
    """Small-multiples display of a raster's RGB, height, and intensity channels.

    Args:
        raster: A build_raster() result.
        title: Figure title.

    Returns:
        The rendered matplotlib Figure.
    """
    occupied = raster["occupied_mask"]
    extent = raster["extent"]

    # imshow's row 0 is the array's first row; origin="lower" puts it at
    # the bottom of the display, so row index increasing upward matches
    # build_raster's row = f(plot_y) (plot_y increasing = more forward),
    # keeping forward "up" consistent with the scatter views.
    rgba = np.zeros((*occupied.shape, 4))
    rgba[..., :3] = raster["color"] / 255.0
    rgba[..., 3] = raster["colored_mask"].astype(float)

    height_masked = np.ma.masked_invalid(raster["height"])
    intensity_masked = np.ma.masked_invalid(raster["intensity"])
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="#dddddd")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(rgba, origin="lower", extent=extent)
    axes[0].set_title("color")

    im1 = axes[1].imshow(height_masked, origin="lower", extent=extent, cmap=cmap)
    axes[1].set_title("height (z, m)")
    fig.colorbar(im1, ax=axes[1], fraction=0.046)

    im2 = axes[2].imshow(intensity_masked, origin="lower", extent=extent, cmap=cmap)
    axes[2].set_title("intensity")
    fig.colorbar(im2, ax=axes[2], fraction=0.046)

    for ax in axes:
        ax.set_xlabel("left <- ego y -> right (flipped)")
        ax.set_ylabel("ego x (forward)")

    fig.suptitle(title)
    return fig


if __name__ == "__main__":
    nusc = NuScenes(version=NUSCENES_VERSION, dataroot=NUSCENES_DATAROOT, verbose=True)

    scene_token = nusc.scene[0]["token"]
    scene_data = build_scene_data(nusc, scene_token)[:NUM_SAMPLES]

    os.makedirs(IMAGE_DIR, exist_ok=True)

    sample_results = []
    for sample_index, sample_data in enumerate(scene_data):
        sample_token = sample_data[SAMPLE_INGESTION_KEYS['token']]

        pipeline = GeometricBEVPipeline(sample_data)
        ego_points, colors = pipeline.project_to_bev()
        intensity = pipeline.lidar_projection.intensity

        raster = build_raster(ego_points, colors, intensity, GRID_HALF_EXTENT_METERS, CELL_SIZE_METERS)
        validation = validate_raster(raster)

        checks = {k: v for k, v in validation.items() if isinstance(v, bool)}
        all_passed = all(checks.values())
        print(f"Sample {sample_token}: fill={validation['fill_fraction']:.1%}, "
              f"height=[{validation['height_min']:.2f}, {validation['height_max']:.2f}]m, "
              f"checks={'PASS' if all_passed else 'FAIL'} ({checks})")

        if sample_index < NUM_NAIVE_CHECK_SAMPLES:
            naive_raster = build_raster_naive(ego_points, colors, intensity,
                                               GRID_HALF_EXTENT_METERS, CELL_SIZE_METERS)
            naive_comparison = compare_rasters(raster, naive_raster)
            naive_all_match = all(naive_comparison.values())
            print(f"  vs. naive reference implementation: "
                  f"{'MATCH' if naive_all_match else 'MISMATCH'} ({naive_comparison})")
            validation["naive_comparison"] = naive_comparison

        fig = render_raster(raster, title=f"BEV raster -- sample {sample_token} "
                                           f"(cell_size={CELL_SIZE_METERS}m)")
        out_path = os.path.join(IMAGE_DIR, f"{sample_token}.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)

        sample_results.append({
            "sample_token": sample_token,
            "image_path": out_path,
            **validation,
        })

    os.makedirs(REPORT_DIR, exist_ok=True)

    with open(DATA_FILE, "w") as data_outfile:
        json.dump({
            "grid_half_extent_meters": GRID_HALF_EXTENT_METERS,
            "cell_size_meters": CELL_SIZE_METERS,
            "samples": sample_results,
        }, data_outfile, indent=2)
    print(f"Wrote {DATA_FILE}")

    with open(REPORT_FILE, "w") as outfile:
        outfile.write(f"--- Raster-height check (grid_half_extent={GRID_HALF_EXTENT_METERS}m, "
                       f"cell_size={CELL_SIZE_METERS}m) ---\n")
        bool_keys = [k for k, v in sample_results[0].items() if isinstance(v, bool)]
        for sample in sample_results:
            all_passed = all(sample[k] for k in bool_keys)
            outfile.write(f"\nSample {sample['sample_token']}: {'PASS' if all_passed else 'FAIL'}\n")
            outfile.write(f"  Fill fraction: {sample['fill_fraction']:.1%} "
                           f"({sample['num_occupied_cells']} occupied, {sample['num_colored_cells']} colored)\n")
            outfile.write(f"  Height range: [{sample['height_min']:.2f}, {sample['height_max']:.2f}]m, "
                           f"mean={sample['height_mean']:.2f}m\n")
            outfile.write(f"  Intensity range: [{sample['intensity_min']:.1f}, {sample['intensity_max']:.1f}]\n")
            for k in bool_keys:
                if not sample[k]:
                    outfile.write(f"  FAILED: {k}\n")
            if "naive_comparison" in sample:
                naive_status = "MATCH" if all(sample["naive_comparison"].values()) else "MISMATCH"
                outfile.write(f"  vs. naive reference implementation: {naive_status} ({sample['naive_comparison']})\n")
        outfile.write(f"\nImages written to {IMAGE_DIR}/\n")
    print(f"Wrote {REPORT_FILE}")
