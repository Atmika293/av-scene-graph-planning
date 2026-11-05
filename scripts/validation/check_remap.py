"""Validates GeometricBEVPipeline.sample_colors' cv2.remap call against an
independently hand-computed bilinear interpolation, for a handful of real
lidar points projected onto one camera.

Two tiers, per the validation plan:
- Indexing check: the sampled color is in the right neighborhood of the
  source image at a point's (rounded) projected coordinate -- catches a
  gross flip/transpose bug in canvas_coords, but says nothing about whether
  the sub-pixel interpolation itself is correct.
- Interpolation check (the real test): hand-computes standard bilinear
  interpolation from the camera image's 4 neighboring pixels around a
  point's exact (fractional) projected coordinate, and compares it against
  cv2.remap(interpolation=cv2.INTER_LINEAR)'s actual output for that same
  coordinate -- validates that remap's pixel-center convention matches the
  one assumed by canvas_coords/clip_to_canvas, not just that the point
  landed in the right neighborhood.

Run with: python scripts/validation/check_remap.py (from the repo root, after
ingesting/having local access to the configured nuScenes dataset).
"""

import json
import os

import cv2
import numpy as np
from nuscenes.nuscenes import NuScenes

from av_data_pipeline.config import NUSCENES_VERSION, NUSCENES_DATAROOT
from av_data_pipeline.ingestion import build_scene_data
from av_data_pipeline.ingestion_keys import SAMPLE_INGESTION_KEYS
from bev.fusion import GeometricBEVPipeline

REPORT_DIR = "reports"
REPORT_FILE = os.path.join(REPORT_DIR, "remap_check.json")
NUM_POINTS_TO_CHECK = 8
# Points too close to an integer coordinate barely exercise bilinear
# weighting (they're ~equal to a single pixel's value either way); points
# too close to the canvas edge risk BORDER_CONSTANT and manual interpolation
# disagreeing on out-of-bounds neighbors for reasons unrelated to the thing
# being tested. Both are filtered out below.
MIN_FRACTIONAL_OFFSET = 0.15
EDGE_MARGIN = 2
PASS_THRESHOLD = 1.0  # color units (0-255 scale); cv2.remap rounds internally
# Loose on purpose: the indexing check only confirms the point landed in the
# right neighborhood, not that interpolation is correct -- a real edge in the
# image can easily make the nearest pixel differ from the true bilinear
# value by much more than rounding noise, without indicating a bug.
INDEXING_PASS_THRESHOLD = 60.0


def nearest_pixel(image: np.array, x: float, y: float) -> np.array:
    """Read image's color at the pixel nearest to (x, y), no interpolation.

    Args:
        image: Source image (H, W, 3), as float.
        x: Horizontal (column) coordinate, fractional.
        y: Vertical (row) coordinate, fractional.

    Returns:
        That nearest pixel's RGB color (3,).
    """
    return image[int(round(y)), int(round(x))]


def bilinear_interpolate(image: np.array, x: float, y: float) -> np.array:
    """Hand-compute standard bilinear interpolation of image at (x, y).

    Assumes integer coordinates are pixel centers (OpenCV's convention),
    matching cv2.remap's own assumption -- confirming that match is exactly
    the point of this check.

    Args:
        image: Source image (H, W, 3), as float.
        x: Horizontal (column) coordinate, fractional.
        y: Vertical (row) coordinate, fractional.

    Returns:
        The interpolated RGB color (3,).
    """
    x0, y0 = int(np.floor(x)), int(np.floor(y))
    x1, y1 = x0 + 1, y0 + 1
    dx, dy = x - x0, y - y0

    return ((1 - dx) * (1 - dy) * image[y0, x0] +
            dx * (1 - dy) * image[y0, x1] +
            (1 - dx) * dy * image[y1, x0] +
            dx * dy * image[y1, x1])


if __name__ == "__main__":
    nusc = NuScenes(version=NUSCENES_VERSION, dataroot=NUSCENES_DATAROOT, verbose=True)

    scene_token = nusc.scene[0]["token"]
    sample_data = build_scene_data(nusc, scene_token)[0]

    pipeline = GeometricBEVPipeline(sample_data)
    global_points = pipeline.lidar_projection.lidar_to_global()

    sd_token, camera_pipe = next(iter(pipeline.camera_pipelines.items()))
    camera_image = cv2.imread(pipeline.sample_camera_images[sd_token])[:, :, ::-1]  # BGR -> RGB
    camera_image = np.ascontiguousarray(camera_image)

    canvas_coords, in_front_mask = pipeline.map_lidar_points_to_camera_canvas(camera_pipe, global_points)
    canvas_filter = pipeline.clip_to_canvas(canvas_coords, camera_pipe.sensor_width, camera_pipe.sensor_height)
    clipped_coords = canvas_coords[canvas_filter]

    fractional = clipped_coords - np.floor(clipped_coords)
    well_offset = np.all((fractional > MIN_FRACTIONAL_OFFSET) & (fractional < 1 - MIN_FRACTIONAL_OFFSET), axis=1)
    in_bounds = (
        (clipped_coords[:, 0] > EDGE_MARGIN) & (clipped_coords[:, 0] < camera_pipe.sensor_width - EDGE_MARGIN) &
        (clipped_coords[:, 1] > EDGE_MARGIN) & (clipped_coords[:, 1] < camera_pipe.sensor_height - EDGE_MARGIN)
    )
    candidates = clipped_coords[well_offset & in_bounds]

    rng = np.random.default_rng(seed=0)
    sample_idx = rng.choice(len(candidates), size=min(NUM_POINTS_TO_CHECK, len(candidates)), replace=False)
    test_points = candidates[sample_idx]

    remap_colors = cv2.remap(src=camera_image,
                              map1=test_points[:, 0].astype(np.float32).reshape(-1, 1),
                              map2=test_points[:, 1].astype(np.float32).reshape(-1, 1),
                              interpolation=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=(0, 0, 0)).reshape(-1, 3)

    camera_image_float = camera_image.astype(float)

    print("Indexing check (nearest pixel vs. remap -- coarse neighborhood sanity check):")
    print(f"{'x':>8} {'y':>8} {'nearest (R,G,B)':>26} {'cv2.remap (R,G,B)':>26} {'max abs diff':>14}")
    indexing_max_diff_overall = 0.0
    nearest_colors = []
    for (x, y), remap_color in zip(test_points, remap_colors):
        nearest_color = nearest_pixel(camera_image_float, x, y)
        remap_color_float = remap_color.astype(float)
        diff = np.abs(nearest_color - remap_color_float)
        indexing_max_diff_overall = max(indexing_max_diff_overall, diff.max())
        nearest_colors.append((nearest_color, diff))
        print(f"{x:8.2f} {y:8.2f} {np.array2string(nearest_color, precision=2):>26} "
              f"{np.array2string(remap_color_float, precision=2):>26} {diff.max():14.4f}")
    indexing_status = "PASS" if indexing_max_diff_overall < INDEXING_PASS_THRESHOLD else "FAIL"
    print(f"Max abs difference: {indexing_max_diff_overall:.4f} "
          f"({indexing_status} -- within {INDEXING_PASS_THRESHOLD} color units)\n")

    print("Interpolation check (hand-computed bilinear vs. remap -- the real test):")
    print(f"{'x':>8} {'y':>8} {'manual (R,G,B)':>26} {'cv2.remap (R,G,B)':>26} {'max abs diff':>14}")
    points = []
    max_diff_overall = 0.0
    for (x, y), remap_color, (nearest_color, indexing_diff) in zip(test_points, remap_colors, nearest_colors):
        manual_color = bilinear_interpolate(camera_image_float, x, y)
        remap_color_float = remap_color.astype(float)
        diff = np.abs(manual_color - remap_color_float)
        max_diff_overall = max(max_diff_overall, diff.max())
        print(f"{x:8.2f} {y:8.2f} {np.array2string(manual_color, precision=2):>26} "
              f"{np.array2string(remap_color_float, precision=2):>26} {diff.max():14.4f}")

        points.append({
            "x": float(x),
            "y": float(y),
            "nearest_rgb": nearest_color.tolist(),
            "indexing_max_abs_diff": float(indexing_diff.max()),
            "manual_rgb": manual_color.tolist(),
            "remap_rgb": remap_color_float.tolist(),
            "max_abs_diff": float(diff.max()),
        })

    status = "PASS" if max_diff_overall < PASS_THRESHOLD else "FAIL"
    print(f"\nMax abs difference across all test points: {max_diff_overall:.4f} "
          f"({status} -- within {PASS_THRESHOLD} color unit)")

    os.makedirs(REPORT_DIR, exist_ok=True)
    with open(REPORT_FILE, "w") as outfile:
        json.dump({
            "sample_token": sample_data[SAMPLE_INGESTION_KEYS['token']],
            "camera_sample_data_token": sd_token,
            "indexing_pass_threshold": INDEXING_PASS_THRESHOLD,
            "indexing_max_abs_diff_overall": indexing_max_diff_overall,
            "indexing_status": indexing_status,
            "pass_threshold": PASS_THRESHOLD,
            "max_abs_diff_overall": max_diff_overall,
            "status": status,
            "points": points,
        }, outfile, indent=2)
    print(f"Wrote {REPORT_FILE}")
