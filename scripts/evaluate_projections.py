"""Validate transform.py's projection pipeline against nuScenes' reference re-projections.

Loads the per-stage reference values produced by export_2d_annotations_debug.py
(v1.0-mini/image_annotations_debug.json) and re-runs this project's own
CameraTransformPipeline / SampleAnnotation pipeline (transform.py) on every
sample of every scene in the dataset, comparing outputs stage by stage:

    SampleAnnotation.globalcoordinates()      <-> corners_global_frame
    CameraTransformPipeline.global_to_camera  <-> corners_camera_frame
    CameraTransformPipeline.camera_to_sensor  <-> pre_clip_corners
    CameraTransformPipeline.clip_to_canvas    <-> bbox_corners

Corner-array comparisons use compare_corners (exact, up to corner ordering).
The final clipped bbox is also compared via IOU (compute_iou) as a more
forgiving check, since clip_to_canvas crosses OpenCV's float32 boundary and
so won't exactly match the float64 Shapely-based reference.

Also flags reference annotations that SampleTransformPipeline silently skips
because it only builds a SampleAnnotation for N_LIDAR_PTS > 0 — those never
enter the comparison loop at all, so they'd otherwise go unnoticed.

Running this script as __main__ writes a structured report — per-stage
verified counts and mismatch details, IOU statistics, canvas-intersection
agreement, and the skipped-annotation count — to reports/evaluate_projections.txt
(overwritten each run). Nothing is printed to stdout; every result goes into
the report file.

It also writes reports/evaluate_projections_data.jsonl: one line per
(annotation, camera) pair that had both a computed and a reference bbox,
with its IOU, category, instance token, camera channel, projected bbox
area, and both boxes' own coordinates. This is the raw data behind
visualize_projections.py's worst-N image selection (and the boxes it
draws) and build_correctness_report.py's IOU-vs-category/size/camera
charts (Phases 4-5).
"""

from collections import defaultdict

import numpy as np

from av_data_pipeline.config import NUSCENES_VERSION, NUSCENES_DATAROOT
from av_data_pipeline.ingestion import extract_sample_data_from_sample, get_sample_tokens_from_scene
from av_data_pipeline.transform import *


def compute_area(x1: float, y1: float, x2: float, y2: float) -> float:
    """Compute the area of an axis-aligned box, clamped to zero if degenerate.

    Args:
        x1: Minimum x-coordinate.
        y1: Minimum y-coordinate.
        x2: Maximum x-coordinate.
        y2: Maximum y-coordinate.

    Returns:
        The box's area, or 0 if x2 < x1 or y2 < y1.
    """
    return max((x2 - x1) * (y2 - y1), 0)


def compute_iou(bbox1: list[float], bbox2: list[float]) -> float:
    """Compute the intersection-over-union of two axis-aligned boxes.

    Args:
        bbox1: First box as [xmin, ymin, xmax, ymax].
        bbox2: Second box as [xmin, ymin, xmax, ymax].

    Returns:
        The IOU in [0, 1], or -1 if the two boxes have zero combined area.
    """
    area1 = compute_area(*bbox1)
    area2 = compute_area(*bbox2)

    intersection_x1 = max(bbox1[0], bbox2[0])
    intersection_y1 = max(bbox1[1], bbox2[1])
    intersection_x2 = min(bbox1[2], bbox2[2])
    intersection_y2 = min(bbox1[3], bbox2[3])

    intersection_area = compute_area(intersection_x1, intersection_y1,
                                    intersection_x2, intersection_y2)

    union_area = area1 + area2 - intersection_area

    if union_area > 0:
        return intersection_area / union_area

    return -1


def _sort_rows(corners: np.array) -> np.array:
    """Canonicalize a 2D point array's row order for order-independent comparison.

    Sorts lexicographically by every column (x, then y, then z, ...), not just
    the first, so that rows sharing the same x (common for axis-aligned boxes,
    e.g. unrotated objects) still land in a consistent, unique order instead
    of an arbitrary one determined by each array's original row order.

    Args:
        corners: 2D array of points, shape (n, d).

    Returns:
        The same rows, reordered into a canonical order.
    """
    # np.lexsort's primary key is its LAST argument, so reverse the columns.
    return corners[np.lexsort(corners.T[::-1])]


def compare_corners(corners1: np.array, corners2: np.array, rtol: float = 1e-05, atol: float = 1e-08) -> bool:
    """Check whether two corner arrays match, up to corner ordering.

    Canonicalizes row order via _sort_rows before comparing, since two
    implementations may enumerate a box's corners in different orders. Flat
    (1D) inputs, such as an [xmin, ymin, xmax, ymax] bbox, are compared as-is
    since their element order is meaningful rather than an interchangeable
    point set.
    """
    if corners1.ndim > 1:
        corners1 = _sort_rows(corners1)
        corners2 = _sort_rows(corners2)

    return np.allclose(corners1, corners2, rtol=rtol, atol=atol)


class StageTracker:
    """Tallies verified/mismatched corner comparisons for one pipeline stage."""

    name: str
    verified: int
    mismatches: list[tuple[str, str, str]]

    def __init__(self, name: str) -> None:
        """Initialize an empty tracker for the named pipeline stage.

        Args:
            name: Human-readable stage name, used as the report section heading.
        """
        self.name = name
        self.verified = 0
        self.mismatches = []  # (annotation_token, data_token, detail)

    def check(self, annotation_token: str, data_token: str, computed: np.array, reference: np.array,
             rtol: float = 1e-05, atol: float = 1e-08) -> None:
        """Compare one computed/reference corner-array pair and tally the result.

        Args:
            annotation_token: Annotation token this comparison belongs to.
            data_token: Camera sample_data token this comparison belongs to.
            computed: This project's own transform.py output for this stage.
            reference: The matching reference value from image_annotations_debug.json.
            rtol: Relative tolerance forwarded to compare_corners.
            atol: Absolute tolerance forwarded to compare_corners.
        """
        if computed.shape != reference.shape:
            self.mismatches.append((annotation_token, data_token,
                                    f"shape mismatch: mine={computed.shape} ref={reference.shape}"))
            return

        if compare_corners(computed, reference, rtol=rtol, atol=atol):
            self.verified += 1
        else:
            if computed.ndim > 1:
                sorted1 = _sort_rows(computed)
                sorted2 = _sort_rows(reference)
            else:
                sorted1, sorted2 = computed, reference
            max_diff = np.max(np.abs(sorted1 - sorted2))
            self.mismatches.append((annotation_token, data_token, f"max abs diff={max_diff:.6g}"))

    @property
    def total(self) -> int:
        """Total comparisons recorded so far (verified + mismatched)."""
        return self.verified + len(self.mismatches)

    def report_lines(self) -> list[str]:
        """Render this stage's verified count and mismatch details as report lines.

        Returns:
            Lines to append to the overall report, ending with a blank separator.
        """
        lines = [f"--- {self.name} ---", f"Verified: {self.verified}/{self.total}"]
        for ann_token, data_token, detail in self.mismatches:
            lines.append(f"  MISMATCH ({ann_token}, {data_token}): {detail}")
        lines.append("")
        return lines


if __name__ == "__main__":
    from nuscenes.nuscenes import NuScenes
    import json, os

    IMAGE_ANN_FILE = os.path.join(NUSCENES_VERSION, "image_annotations_debug.json")
    REPORT_DIR = "reports"
    REPORT_FILE = os.path.join(REPORT_DIR, "evaluate_projections.txt")
    DATA_FILE = os.path.join(REPORT_DIR, "evaluate_projections_data.jsonl")

    nusc = NuScenes(version=NUSCENES_VERSION, dataroot=NUSCENES_DATAROOT, verbose=True)

    with open(os.path.join(NUSCENES_DATAROOT, IMAGE_ANN_FILE)) as fp:
        image_annotations = json.load(fp)

    # Index reference records by annotation token for fast per-sample lookup,
    # instead of rescanning the whole reference file for every sample.
    annotations_by_token: dict[str, list[dict]] = defaultdict(list)
    for img_ann in image_annotations:
        annotations_by_token[img_ann["sample_annotation_token"]].append(img_ann)

    global_stage = StageTracker("ann -> global")
    camera_stage = StageTracker("global -> camera")
    sensor_stage = StageTracker("camera -> sensor (pre-clip)")
    canvas_stage = StageTracker("clip -> canvas (bbox)")

    # One entry per (annotation, camera) pair with both a computed and a
    # reference bbox -- the IOU comparison, plus the metadata
    # visualize_projections.py/build_correctness_report.py need (Phases 4-5).
    projection_records: list[dict] = []
    missing_bbox: list[tuple[str, str]] = []  # reference has a clipped bbox, ours doesn't
    extra_bbox: list[tuple[str, str]] = []  # ours has a clipped bbox, reference doesn't
    agreed_no_bbox = 0  # neither pipeline finds a canvas intersection

    # SampleTransformPipeline only builds a SampleAnnotation for annotations with
    # N_LIDAR_PTS > 0, so any reference annotation with zero lidar points is never
    # visited by the comparison loop below. Track those separately.
    skipped_zero_lidar_tokens: set[str] = set()
    skipped_zero_lidar_rows = 0

    for scene in nusc.scene:
        for sample_token in get_sample_tokens_from_scene(nusc, scene['token']):
            sample_data = extract_sample_data_from_sample(nusc, sample_token)
            sample_transform_pipeline = SampleTransformPipeline(sample_data)

            # Keyed here once per sample so the comparison loop below can look
            # up each annotation's CATEGORY/INSTANCE_TOKEN without re-walking
            # sample_data["ANNOTATIONS"] (which SampleTransformPipeline.annotations
            # -- built from N_LIDAR_PTS > 0 entries only -- doesn't carry).
            ann_meta_by_token: dict[str, dict] = {}
            for annotation_token, ann_data in sample_data["ANNOTATIONS"].items():
                ann_meta_by_token[annotation_token] = ann_data
                if ann_data["N_LIDAR_PTS"] == 0 and annotation_token in annotations_by_token:
                    skipped_zero_lidar_tokens.add(annotation_token)
                    skipped_zero_lidar_rows += len(annotations_by_token[annotation_token])

            for annotation_token, annotation in sample_transform_pipeline.annotations.items():
                global_coords = annotation.globalcoordinates()

                # An annotation can appear in more than one camera's reference
                # records (if visible from multiple cameras in this sample).
                for img_ann in annotations_by_token.get(annotation_token, []):
                    data_token = img_ann["sample_data_token"]

                    global_stage.check(annotation_token, data_token, global_coords,
                                       np.array(img_ann["corners_global_frame"], dtype=np.float32))

                    if data_token not in sample_transform_pipeline.camera_pipelines:
                        continue

                    camera_pipe = sample_transform_pipeline.camera_pipelines[data_token]
                    camera_coords = camera_pipe.global_to_camera(global_coords)
                    camera_stage.check(annotation_token, data_token, camera_coords,
                                       np.array(img_ann["corners_camera_frame"], dtype=np.float32))

                    sensor_coords = camera_pipe.camera_to_sensor(camera_coords)
                    sensor_stage.check(annotation_token, data_token, sensor_coords,
                                       np.array(img_ann["pre_clip_corners"], dtype=np.float32))

                    bbox = camera_pipe.clip_to_canvas(sensor_coords)
                    ref_bbox = img_ann["bbox_corners"]

                    if bbox is not None and ref_bbox is not None:
                        # Canvas clipping crosses OpenCV's float32 boundary, so tolerate sub-pixel diffs.
                        canvas_stage.check(annotation_token, data_token, np.array(bbox, dtype=np.float32),
                                           np.array(ref_bbox, dtype=np.float32), atol=1e-3)

                        # Original IOU-based comparison strategy.
                        iou = compute_iou(bbox, ref_bbox)
                        ann_meta = ann_meta_by_token[annotation_token]
                        projection_records.append({
                            "annotation_token": annotation_token,
                            "data_token": data_token,
                            "instance_token": ann_meta["INSTANCE_TOKEN"],
                            "category": ann_meta["CATEGORY"],
                            "camera_channel": nusc.get("sample_data", data_token)["channel"],
                            "iou": float(iou),
                            "area": float(compute_area(*bbox)),
                            "bbox": [float(v) for v in bbox],
                            "ref_bbox": [float(v) for v in ref_bbox],
                        })
                    elif bbox is not None and ref_bbox is None:
                        extra_bbox.append((annotation_token, data_token))
                    elif bbox is None and ref_bbox is not None:
                        missing_bbox.append((annotation_token, data_token))
                    else:
                        agreed_no_bbox += 1

    report = []
    report.extend(global_stage.report_lines())
    report.extend(camera_stage.report_lines())
    report.extend(sensor_stage.report_lines())
    report.extend(canvas_stage.report_lines())

    report.append("--- IOU (clipped bbox vs reference) ---")
    if projection_records:
        values = np.array([r["iou"] for r in projection_records])
        report.append(f"Count: {len(values)}")
        report.append(f"Min: {values.min():.6f}  Max: {values.max():.6f}  "
                      f"Mean: {values.mean():.6f}  Median: {np.median(values):.6f}")
        report.append("Worst 10:")
        for r in sorted(projection_records, key=lambda r: r["iou"])[:10]:
            report.append(f"  ({r['annotation_token']}, {r['data_token']}): {r['iou']:.6f}")
    else:
        report.append("No IOU pairs computed.")
    report.append("")

    report.append("--- Canvas-intersection disagreements ---")
    report.append(f"Agreed no bbox (both None): {agreed_no_bbox}")
    report.append(f"Missing in ours (reference has a bbox, ours is None): {len(missing_bbox)}")
    for ann_token, data_token in missing_bbox:
        report.append(f"  ({ann_token}, {data_token})")
    report.append(f"Extra in ours (ours has a bbox, reference is None): {len(extra_bbox)}")
    for ann_token, data_token in extra_bbox:
        report.append(f"  ({ann_token}, {data_token})")
    report.append("")

    report.append("--- Reference annotations skipped (N_LIDAR_PTS == 0) ---")
    report.append(f"Skipped annotations: {len(skipped_zero_lidar_tokens)} "
                  f"({skipped_zero_lidar_rows} reference rows across cameras) "
                  f"never entered the comparison loop, since SampleTransformPipeline "
                  f"only builds a SampleAnnotation for N_LIDAR_PTS > 0.")

    os.makedirs(REPORT_DIR, exist_ok=True)
    with open(REPORT_FILE, "w") as outfile:
        outfile.write("\n".join(report) + "\n")

    with open(DATA_FILE, "w") as data_outfile:
        for r in projection_records:
            data_outfile.write(json.dumps(r) + "\n")
