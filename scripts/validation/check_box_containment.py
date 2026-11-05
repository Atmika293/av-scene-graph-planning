"""Validates each sample's annotation boxes (SampleBoxProjection) against its
LiDAR points (LidarProjection), both already expressed in the ego frame.

Two checks, per the validation plan:
- Count check (#4, the strongest check): compares this pipeline's own
  points-inside-box count against nuScenes' own ground-truth N_LIDAR_PTS for
  that annotation -- independent of anything this pipeline computes.
- Separation check (#3): a nearest-neighbor gap ratio -- mean distance from
  a box's interior points to a dilated "shell" just outside it, divided by
  mean spacing between the interior points themselves. High means the box
  sits in a real density gap; low can mean a misaligned box, or two
  genuinely adjacent real objects (parked bumper-to-bumper, etc.), so this
  is an aggregate/outlier-flagging stat, not a strict per-box threshold.

Containment uses a from-scratch reimplementation of
nuscenes.utils.geometry_utils.points_in_box's exact algorithm (reference
corner + 3 adjacent-corner axis vectors, project each point onto each axis
and bound-check it), applied directly to this pipeline's own ego-frame box
corners (SampleBoxProjection) instead of reconstructing a nuScenes Box
object -- same method nuScenes used to compute N_LIDAR_PTS in the first
place, without a pyquaternion round-trip, since the corners are already
computed and already visually validated (see render_bev.py's box overlay).

Also renders one zoomed, two-color (inside vs. shell) plot per category per
sample -- for only the highest-N_LIDAR_PTS annotation in each category, not
every box, to keep the output legible.

Run with: python scripts/validation/check_box_containment.py (from the repo root,
after ingesting/having local access to the configured nuScenes dataset).
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from nuscenes.nuscenes import NuScenes
from scipy.spatial import cKDTree

from av_data_pipeline.config import NUSCENES_VERSION, NUSCENES_DATAROOT
from av_data_pipeline.ingestion import build_scene_data
from av_data_pipeline.ingestion_keys import SAMPLE_INGESTION_KEYS, SAMPLE_ANNOTATION_INGESTION_KEYS, SENSOR_MODALITY_KEYS
from bev.lidar import LidarProjection
from bev.sample import SampleBoxProjection

REPORT_DIR = "reports"
IMAGE_DIR = "reports/box_check_images"
REPORT_FILE = os.path.join(REPORT_DIR, "box_containment_check.txt")
DATA_FILE = os.path.join(REPORT_DIR, "box_containment_check_data.json")
NUM_SAMPLES = 3
# How far the "shell" boundary used by the separation check extends past the
# real box, as a multiple of the box's own size (nuScenes' own
# points_in_box dilation convention) -- scales with object size rather than
# a fixed meters margin, so a truck's shell isn't the same width as a
# pedestrian's.
SHELL_WLH_FACTOR = 1.5
PLOT_MARGIN_METERS = 1.0


def points_in_box(corners: np.array, points: np.array, wlh_factor: float = 1.0) -> np.array:
    """Check which points fall inside a box, given its 8 corners.

    Reimplements nuscenes.utils.geometry_utils.points_in_box's exact
    algorithm (reference corner + 3 adjacent-axis vectors, project each
    point onto each axis and bound-check it) against this pipeline's own
    corner array instead of a nuScenes Box object.

    Args:
        corners: The box's 8 corners (8, 3), in SampleAnnotation's own
            corner ordering (see av_data_pipeline.transform's
            __extract_corners).
        points: Points to test (N, 3), in the same frame as corners.
        wlh_factor: Scales the box uniformly outward from its own center
            before testing -- 1.0 for the real box, >1.0 for a dilated
            "shell" boundary (matches nuScenes' own Box.corners(wlh_factor),
            which dilates the same way: every corner scaled away from the
            center, not just the half-extents).

    Returns:
        Boolean mask (N,): True where a point falls inside the (possibly
        dilated) box.
    """
    center = corners.mean(axis=0)
    corners = center + (corners - center) * wlh_factor

    # corners[3] is adjacent to corners[5]/[1]/[2] by exactly one sign flip
    # each (length/width/height respectively) -- see __extract_corners'
    # ordering: [3] = (+l/2,-w/2,-h/2), [5] = (-l/2,-w/2,-h/2),
    # [1] = (+l/2,+w/2,-h/2), [2] = (+l/2,-w/2,+h/2). These four corners are
    # all points_in_box's algorithm needs.
    p1 = corners[3]
    axis_length = corners[5] - p1
    axis_width = corners[1] - p1
    axis_height = corners[2] - p1

    v = points - p1
    proj_length, proj_width, proj_height = v @ axis_length, v @ axis_width, v @ axis_height

    in_length = (0 <= proj_length) & (proj_length <= axis_length @ axis_length)
    in_width = (0 <= proj_width) & (proj_width <= axis_width @ axis_width)
    in_height = (0 <= proj_height) & (proj_height <= axis_height @ axis_height)
    return in_length & in_width & in_height


def separation_ratio(inside_points: np.array, shell_points: np.array) -> float:
    """Nearest-neighbor gap ratio: boundary-crossing distance vs. intra-object spacing.

    Args:
        inside_points: Points found inside the real (wlh_factor=1.0) box.
        shell_points: Points found inside the dilated box but not the real
            one -- the "background" just outside the box.

    Returns:
        Mean distance from each inside point to its nearest shell point,
        divided by mean distance from each inside point to its nearest
        OTHER inside point. None if there isn't enough data (fewer than 2
        inside points, or no shell points) to compute either side.
    """
    if len(inside_points) < 2 or len(shell_points) == 0:
        return None

    intra_tree = cKDTree(inside_points)
    intra_dist, _ = intra_tree.query(inside_points, k=2)
    mean_intra_spacing = intra_dist[:, 1].mean()
    if mean_intra_spacing == 0:
        return None

    shell_tree = cKDTree(shell_points)
    boundary_dist, _ = shell_tree.query(inside_points, k=1)
    mean_boundary_gap = boundary_dist.mean()

    return mean_boundary_gap / mean_intra_spacing


def to_plot_xy(ego_xy: np.array) -> tuple[np.array, np.array]:
    """Map ego-frame (x, y) to plot (horizontal, vertical) so forward is up.

    Same convention as render_bev.py's to_plot_xy: ego x = forward, y =
    left, so plotting -y horizontally and x vertically puts forward at the
    top of the image.
    """
    return -ego_xy[:, 1], ego_xy[:, 0]


def render_box_check(ego_points: np.array, inside_mask: np.array, shell_mask: np.array,
                      corners: np.array, title: str) -> "plt.Figure":
    """Zoomed plot of one box's interior vs. shell points, for visual inspection.

    Args:
        ego_points: The sample's full lidar point cloud (N, 3), ego frame.
        inside_mask: Boolean mask (N,) of points inside the real box.
        shell_mask: Boolean mask (N,) of points inside the dilated box but
            not the real one.
        corners: The box's own 8 corners (8, 3), ego frame.
        title: Plot title.

    Returns:
        The rendered matplotlib Figure.
    """
    footprint_x, footprint_y = to_plot_xy(corners[[0, 2, 4, 6, 0]])

    fig, ax = plt.subplots(figsize=(6, 6))

    inside_x, inside_y = to_plot_xy(ego_points[inside_mask])
    shell_x, shell_y = to_plot_xy(ego_points[shell_mask])
    ax.scatter(shell_x, shell_y, s=20, c="#ff7f0e", label="shell (outside box)")
    ax.scatter(inside_x, inside_y, s=20, c="#2ca02c", label="inside box")
    ax.plot(footprint_x, footprint_y, c="#333333", linewidth=1.5, label="box footprint")

    ax.set_aspect("equal")
    ax.set_xlim(footprint_x.min() - PLOT_MARGIN_METERS, footprint_x.max() + PLOT_MARGIN_METERS)
    ax.set_ylim(footprint_y.min() - PLOT_MARGIN_METERS, footprint_y.max() + PLOT_MARGIN_METERS)
    ax.set_xlabel("left <- ego y -> right (flipped)")
    ax.set_ylabel("ego x (forward)")
    ax.legend(markerscale=1.5, loc="upper right", fontsize=8)
    ax.set_title(title, fontsize=9)

    return fig


if __name__ == "__main__":
    nusc = NuScenes(version=NUSCENES_VERSION, dataroot=NUSCENES_DATAROOT, verbose=True)

    scene_token = nusc.scene[0]["token"]
    scene_data = build_scene_data(nusc, scene_token)[:NUM_SAMPLES]

    os.makedirs(IMAGE_DIR, exist_ok=True)

    sample_results = []
    for sample_data in scene_data:
        sample_token = sample_data[SAMPLE_INGESTION_KEYS['token']]

        for _, lidar_sensor_data in sample_data[SENSOR_MODALITY_KEYS['lidar']].items():
            lidar_projection = LidarProjection(lidar_sensor_data)
        ego_points = lidar_projection.lidar_to_ego()

        ego_boxes = SampleBoxProjection(sample_data).project_to_ego()
        annotations = sample_data[SAMPLE_INGESTION_KEYS['annotations']]

        # Picks the highest-N_LIDAR_PTS annotation per category in this
        # sample for the zoomed visual -- not every box, to keep the output
        # legible (see the validation plan).
        best_per_category = {}

        annotation_results = []
        for ann_token, corners in ego_boxes.items():
            ann_data = annotations[ann_token]
            category = ann_data[SAMPLE_ANNOTATION_INGESTION_KEYS['category']]
            n_lidar_pts = ann_data[SAMPLE_ANNOTATION_INGESTION_KEYS['n_lidar']]

            inside_mask = points_in_box(corners, ego_points, wlh_factor=1.0)
            dilated_mask = points_in_box(corners, ego_points, wlh_factor=SHELL_WLH_FACTOR)
            shell_mask = dilated_mask & ~inside_mask

            count_inside = int(inside_mask.sum())
            ratio = separation_ratio(ego_points[inside_mask], ego_points[shell_mask])

            annotation_results.append({
                "annotation_token": ann_token,
                "category": category,
                "n_lidar_pts": int(n_lidar_pts),
                "count_inside": count_inside,
                "count_matches_n_lidar_pts": count_inside == int(n_lidar_pts),
                "num_shell_points": int(shell_mask.sum()),
                "separation_ratio": ratio,
            })

            if category not in best_per_category or n_lidar_pts > best_per_category[category][1]:
                best_per_category[category] = (ann_token, n_lidar_pts, corners, inside_mask, shell_mask)

        mismatches = sum(1 for r in annotation_results if not r["count_matches_n_lidar_pts"])
        print(f"Sample {sample_token}: {len(annotation_results)} annotations, "
              f"{mismatches} with a count/N_LIDAR_PTS mismatch")

        image_paths = {}
        for category, (ann_token, n_lidar_pts, corners, inside_mask, shell_mask) in best_per_category.items():
            fig = render_box_check(ego_points, inside_mask, shell_mask, corners,
                                    title=f"{category} ({n_lidar_pts} lidar pts) -- {sample_token}")
            safe_category = category.replace(".", "_")
            out_path = os.path.join(IMAGE_DIR, f"{sample_token}_{safe_category}.png")
            fig.savefig(out_path, dpi=150)
            plt.close(fig)
            image_paths[category] = out_path

        sample_results.append({
            "sample_token": sample_token,
            "annotations": annotation_results,
            "category_images": image_paths,
        })

    os.makedirs(REPORT_DIR, exist_ok=True)

    with open(DATA_FILE, "w") as data_outfile:
        json.dump({
            "shell_wlh_factor": SHELL_WLH_FACTOR,
            "samples": sample_results,
        }, data_outfile, indent=2)
    print(f"Wrote {DATA_FILE}")

    all_annotations = [a for s in sample_results for a in s["annotations"]]
    mismatched = [a for a in all_annotations if not a["count_matches_n_lidar_pts"]]
    ratios = [(s["sample_token"], a["category"], a["annotation_token"], a["separation_ratio"])
              for s in sample_results for a in s["annotations"] if a["separation_ratio"] is not None]
    ratios.sort(key=lambda r: r[3])

    with open(REPORT_FILE, "w") as outfile:
        outfile.write("--- Count check: points-in-box vs. N_LIDAR_PTS ---\n")
        outfile.write(f"Verified: {len(all_annotations) - len(mismatched)}/{len(all_annotations)}\n")
        if mismatched:
            outfile.write("Mismatches:\n")
            for a in mismatched:
                outfile.write(f"  ({a['annotation_token']}, {a['category']}): "
                               f"count_inside={a['count_inside']} vs n_lidar_pts={a['n_lidar_pts']}\n")
        outfile.write("\n--- Separation check: nearest-neighbor gap ratio (shell_wlh_factor="
                       f"{SHELL_WLH_FACTOR}) ---\n")
        outfile.write(f"Computed: {len(ratios)}/{len(all_annotations)} "
                       f"({len(all_annotations) - len(ratios)} skipped: fewer than 2 interior points)\n")
        if ratios:
            values = [r[3] for r in ratios]
            outfile.write(f"Min: {min(values):.4f}  Max: {max(values):.4f}  "
                           f"Mean: {np.mean(values):.4f}  Median: {np.median(values):.4f}\n")
            outfile.write("Lowest 10 (weakest separation -- possible misalignment, or genuinely adjacent objects):\n")
            for sample_token, category, ann_token, ratio in ratios[:10]:
                outfile.write(f"  ({ann_token}, {category}, {sample_token}): {ratio:.4f}\n")
        outfile.write(f"\n--- Zoomed category images (one per category per sample, highest N_LIDAR_PTS) ---\n")
        outfile.write(f"Written to {IMAGE_DIR}/\n")
    print(f"Wrote {REPORT_FILE}")
