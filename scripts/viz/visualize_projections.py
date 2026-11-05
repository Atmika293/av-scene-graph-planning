"""Visualize a sample of projections from evaluate_projections.py as annotated images.

Loads reports/evaluate_projections_data.jsonl (one record per annotation/
camera pair, see evaluate_projections.py) and selects the 5 best-IOU and 5
worst-IOU records (see select_best_n_diverse/select_worst_n_diverse), each
group deduplicated for variety: the same physical object (instance_token)
is never shown twice, and distinct categories are preferred while enough
remain to fill every slot. The quantitative IOU-vs-size/category
relationship is build_correctness_report.py's job (Phase 5) -- on real
(non-degenerate) boxes this pipeline agrees with the reference to well
under a pixel even at its "worst", so there's rarely a visible gap to show
for the worst cases alone. These 10 images are instead a representative,
diverse sample of what localizations actually look like across the quality
spectrum, best and worst side by side.

For each selected record, loads the camera image via the nuScenes devkit
and draws both boxes directly on it -- our computed box and the reference
box, in distinguishable colors -- then saves it as a WebP under
docs/correctness_images/ (lossy, WEBP_QUALITY; these are published on the
site, and the PNGs were ~1.5-2.5MB each).

Also writes reports/selected_projections.json: the selected records (plus
each one's saved image path and which group -- "best"/"worst" -- it came
from), so build_correctness_report.py (Phase 5) can build captions without
redoing this selection.

Run with: python scripts/viz/visualize_projections.py (from the repo root,
after evaluate_projections.py has produced its data file).
"""

import json
import os

import cv2

from av_data_pipeline.config import NUSCENES_VERSION, NUSCENES_DATAROOT, CORRECTNESS_MIN_BBOX_AREA

DATA_FILE = "reports/evaluate_projections_data.jsonl"
SELECTED_FILE = "reports/selected_projections.json"
IMAGE_DIR = "docs/correctness_images"
NUM_IMAGES_PER_GROUP = 5
WEBP_QUALITY = 85

# IOU is extremely sensitive to tiny absolute coordinate differences once a
# box's area is near zero (a sub-pixel float32-vs-float64 rounding gap can
# swing IOU sharply), so ranking by raw IOU ascending with no area floor
# systematically surfaces degenerate, edge-clipped slivers rather than
# genuinely informative near-misses. The default, 500px^2, trims only that
# degenerate tail (areas 1-100px^2 in practice) -- it's well below the
# 5th-percentile box area (~750px^2 on the mini dataset), so it doesn't
# exclude real small or distant objects, just sub-pixel artifacts.
# Configurable via config.toml's [correctness] min_bbox_area (or the
# CORRECTNESS_MIN_BBOX_AREA env var) since the right floor depends on the
# dataset's own area distribution -- see av_data_pipeline.config.
MIN_BBOX_AREA = CORRECTNESS_MIN_BBOX_AREA

OUR_BOX_COLOR = (0, 0, 255)  # red, BGR (cv2 convention)
REF_BOX_COLOR = (0, 255, 0)  # green, BGR


def load_projection_records(data_file: str) -> list[dict]:
    """Load every record written by evaluate_projections.py.

    Args:
        data_file: Path to the JSONL file (see evaluate_projections.py).

    Returns:
        One dict per (annotation, camera) pair.
    """
    records = []
    with open(data_file) as fp:
        for line in fp:
            records.append(json.loads(line))

    return records


def _select_n_diverse(sorted_records: list[dict], n: int) -> list[dict]:
    """Select the first n records, deduplicated by instance and spread across categories.

    Two passes over records already sorted into the desired priority order:
      1. Strict: skip any record whose instance_token or category has
         already been selected, so the first n picks (if the data has
         enough variety) cover n distinct objects across n distinct
         categories.
      2. Relaxed: if pass 1 couldn't fill every slot (too few distinct
         categories among the leading records), fill the remainder from the
         same sorted order, only still requiring a distinct instance_token
         -- instance-uniqueness is never relaxed, since showing the same
         physical object twice is never useful, but requiring a never-seen
         category might be unsatisfiable.

    Args:
        sorted_records: Records already filtered and sorted into priority order.
        n: Number of records to select.

    Returns:
        Up to n records, in priority order, deduplicated as described.
    """
    selected = []
    seen_instances = set()
    seen_categories = set()
    for r in sorted_records:
        if len(selected) >= n:
            break
        if r["instance_token"] in seen_instances or r["category"] in seen_categories:
            continue
        selected.append(r)
        seen_instances.add(r["instance_token"])
        seen_categories.add(r["category"])

    if len(selected) < n:
        for r in sorted_records:
            if len(selected) >= n:
                break
            if r["instance_token"] in seen_instances:
                continue
            selected.append(r)
            seen_instances.add(r["instance_token"])

    return selected


def select_worst_n_diverse(records: list[dict], n: int = NUM_IMAGES_PER_GROUP,
                           min_area: float = MIN_BBOX_AREA) -> list[dict]:
    """Select n low-IOU records, deduplicated by instance and spread across categories.

    Records below min_area are excluded before ranking -- see MIN_BBOX_AREA:
    without this floor, "worst IOU" is dominated by sub-pixel, edge-clipped
    sliver boxes (an artifact of IOU's sensitivity to tiny absolute
    coordinate differences at near-zero area) rather than genuine,
    visually-inspectable projection mismatches.

    Args:
        records: Every (annotation, camera) record (see load_projection_records).
        n: Number of records to select.
        min_area: Minimum bbox area (our computed box) a record must have
            to be eligible.

    Returns:
        Up to n records, worst-IOU-first among eligible records, deduplicated per _select_n_diverse.
    """
    eligible = (r for r in records if r["area"] >= min_area)
    return _select_n_diverse(sorted(eligible, key=lambda r: r["iou"]), n)


def select_best_n_diverse(records: list[dict], n: int = NUM_IMAGES_PER_GROUP,
                          min_area: float = MIN_BBOX_AREA) -> list[dict]:
    """Select n high-IOU records, deduplicated by instance and spread across categories.

    See select_worst_n_diverse -- same min_area rationale and dedup logic,
    just sorted in the opposite direction.

    Args:
        records: Every (annotation, camera) record (see load_projection_records).
        n: Number of records to select.
        min_area: Minimum bbox area (our computed box) a record must have
            to be eligible.

    Returns:
        Up to n records, best-IOU-first among eligible records, deduplicated per _select_n_diverse.
    """
    eligible = (r for r in records if r["area"] >= min_area)
    return _select_n_diverse(sorted(eligible, key=lambda r: r["iou"], reverse=True), n)


def draw_boxes_on_image(image: "cv2.Mat", bbox: list[float], ref_bbox: list[float]) -> "cv2.Mat":
    """Draw our computed box and the reference box on a copy of the camera image.

    Args:
        image: Camera image (BGR, as loaded by cv2.imread).
        bbox: Our computed box as [xmin, ymin, xmax, ymax].
        ref_bbox: The reference box as [xmin, ymin, xmax, ymax].

    Returns:
        A copy of image with both boxes drawn.
    """
    annotated = image.copy()

    x1, y1, x2, y2 = (round(v) for v in ref_bbox)
    cv2.rectangle(annotated, (x1, y1), (x2, y2), REF_BOX_COLOR, thickness=2)

    x1, y1, x2, y2 = (round(v) for v in bbox)
    cv2.rectangle(annotated, (x1, y1), (x2, y2), OUR_BOX_COLOR, thickness=2)

    return annotated


if __name__ == "__main__":
    from nuscenes.nuscenes import NuScenes

    nusc = NuScenes(version=NUSCENES_VERSION, dataroot=NUSCENES_DATAROOT, verbose=True)

    records = load_projection_records(DATA_FILE)

    worst = select_worst_n_diverse(records, n=NUM_IMAGES_PER_GROUP)
    for r in worst:
        r["group"] = "worst"
    best = select_best_n_diverse(records, n=NUM_IMAGES_PER_GROUP)
    for r in best:
        r["group"] = "best"
    selected = worst + best

    os.makedirs(IMAGE_DIR, exist_ok=True)

    for r in selected:
        image_path = nusc.get_sample_data_path(r["data_token"])
        image = cv2.imread(image_path)

        annotated = draw_boxes_on_image(image, r["bbox"], r["ref_bbox"])

        full_filename = f"{r['group']}_{r['annotation_token']}_{r['data_token']}.webp"
        cv2.imwrite(os.path.join(IMAGE_DIR, full_filename), annotated,
                    [cv2.IMWRITE_WEBP_QUALITY, WEBP_QUALITY])
        r["image"] = os.path.join(IMAGE_DIR, full_filename)

    with open(SELECTED_FILE, "w") as outfile:
        json.dump(selected, outfile, indent=2)

    print(f"Wrote {len(selected)} annotated images ({len(worst)} worst, {len(best)} best) "
         f"to {IMAGE_DIR}/ and selection metadata to {SELECTED_FILE}.")
