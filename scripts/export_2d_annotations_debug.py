"""Export 2D re-projected annotations with every intermediate transform stage.

This is a fork of nuscenes.scripts.export_2d_annotations_as_json that exposes the
box corners after each stage of the global -> ego -> camera -> sensor pipeline,
instead of only the final clipped bbox_corners, so they can be diffed against
this project's own transform.py pipeline stage by stage:

    corners_global_frame      <-> SampleAnnotation.globalcoordinates
    corners_ego_frame         <-> (no direct equivalent; transform.py composes
                                  global->ego and ego->camera into one step)
    corners_camera_frame     <-> CameraTransformPipeline.global_to_camera
    corners_after_calibration<-> calibration_matrix @ camera_coords.T, before
                                  the perspective divide, inside camera_to_sensor
    pre_clip_corners         <-> CameraTransformPipeline.camera_to_sensor (full,
                                  incl. perspective divide)
    bbox_corners             <-> CameraTransformPipeline.clip_to_canvas

The projection logic (global -> ego -> camera -> sensor, in-front filtering,
convex-hull canvas clipping) is copied verbatim from the upstream script to
guarantee the same reference values as v1.0-mini/image_annotations.json.
"""

import argparse
import json
import os
from collections import OrderedDict
from typing import List, Optional, Tuple

import numpy as np
from pyquaternion import Quaternion
from shapely.geometry import MultiPoint, box

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points


def post_process_coords(corner_coords: List,
                        imsize: Tuple[int, int] = (1600, 900)) -> Optional[Tuple[float, float, float, float]]:
    """Clip the convex hull of the re-projected corners to the image canvas.

    Copied from nuscenes.scripts.export_2d_annotations_as_json.post_process_coords.

    Args:
        corner_coords: 2D re-projected corner coordinates (pixel space), one
            [x, y] pair per point that was in front of the camera.
        imsize: Canvas size as (width, height) in pixels.

    Returns:
        (min_x, min_y, max_x, max_y) of the hull/canvas intersection, or None
        if the hull doesn't intersect the canvas at all.
    """
    # Convex hull of the re-projected points (only some of a box's 8 corners
    # may be in front of the camera, so this isn't always all 8 points).
    polygon_from_2d_box = MultiPoint(corner_coords).convex_hull
    img_canvas = box(0, 0, imsize[0], imsize[1])

    if polygon_from_2d_box.intersects(img_canvas):
        img_intersection = polygon_from_2d_box.intersection(img_canvas)
        # Vertices of the intersection polygon; their axis-aligned min/max is
        # the clipped bounding box.
        intersection_coords = np.array([coord for coord in img_intersection.exterior.coords])

        min_x = min(intersection_coords[:, 0])
        min_y = min(intersection_coords[:, 1])
        max_x = max(intersection_coords[:, 0])
        max_y = max(intersection_coords[:, 1])

        return min_x, min_y, max_x, max_y
    else:
        # Hull and canvas don't overlap at all; nothing to clip.
        return None


def generate_record(ann_rec: dict,
                    corners_global_frame: List[List[float]],
                    corners_ego_frame: List[List[float]],
                    corners_camera_frame: List[List[float]],
                    corners_after_calibration: List[List[float]],
                    pre_clip_corners: List[List[float]],
                    post_clip_bbox: Optional[Tuple[float, float, float, float]],
                    sample_data_token: str,
                    filename: str) -> OrderedDict:
    """Build one output record carrying every intermediate transform stage.

    Args:
        ann_rec: Source sample_annotation record, plus the 'sample_annotation_token'
            and 'sample_data_token' keys injected by the caller; only the keys
            listed in `relevant_keys` are copied into the output record.
        corners_global_frame: All 8 box corners (3D) in the global frame.
        corners_ego_frame: All 8 box corners (3D) in the ego-pose frame.
        corners_camera_frame: All 8 box corners (3D) in the camera frame.
        corners_after_calibration: In-front-of-camera corners after the intrinsic
            matrix, homogeneous [x, y, w], before the perspective divide.
        pre_clip_corners: Same corners, perspective-divided into 2D pixel coordinates.
        post_clip_bbox: (min_x, min_y, max_x, max_y) after canvas clipping, or
            None if the box doesn't intersect the canvas.
        sample_data_token: Camera sample_data token this projection is for.
        filename: Image filename for that sample_data record.

    Returns:
        An OrderedDict with the relevant annotation fields plus every
        intermediate-stage corner array and the final bbox_corners.
    """
    repro_rec = OrderedDict()
    repro_rec['sample_data_token'] = sample_data_token

    # Only these annotation fields are worth carrying into the debug record;
    # the rest of the sample_annotation record is noise for this purpose.
    relevant_keys = [
        'attribute_tokens',
        'category_name',
        'instance_token',
        'next',
        'num_lidar_pts',
        'num_radar_pts',
        'prev',
        'sample_annotation_token',
        'sample_data_token',
        'visibility_token',
    ]

    for key, value in ann_rec.items():
        if key in relevant_keys:
            repro_rec[key] = value

    # All 8 box corners (3D), converted from the annotation's local box frame into global
    # coordinates (translation + rotation applied, no further transform).
    repro_rec['corners_global_frame'] = corners_global_frame
    # All 8 box corners (3D) after moving from the global frame into the ego-pose frame.
    repro_rec['corners_ego_frame'] = corners_ego_frame
    # All 8 box corners (3D) after further moving from the ego frame into the camera frame.
    repro_rec['corners_camera_frame'] = corners_camera_frame
    # In-front-of-camera corners after multiplying by the intrinsic calibration matrix,
    # homogeneous [x, y, w] — before the perspective divide.
    repro_rec['corners_after_calibration'] = corners_after_calibration
    # Pre-clip: the same corners perspective-divided into raw 2D pixel coordinates.
    repro_rec['pre_clip_corners'] = pre_clip_corners
    # Post-clip: axis-aligned box of the corners' convex hull intersected with the canvas.
    repro_rec['bbox_corners'] = list(post_clip_bbox) if post_clip_bbox is not None else None
    repro_rec['filename'] = filename

    return repro_rec


def get_2d_boxes_with_intermediates(nusc: NuScenes, sample_data_token: str,
                                    visibilities: List[str]) -> List[OrderedDict]:
    """Project every annotation in a camera keyframe, keeping every intermediate stage.

    Same projection as nuscenes.scripts.export_2d_annotations_as_json.get_2d_boxes
    (global -> ego -> camera -> sensor, in-front filtering, canvas clipping), but
    returns each intermediate stage instead of only the final clipped bbox.

    Args:
        nusc: NuScenes devkit instance used to look up records.
        sample_data_token: Token of a camera sample_data record (must be a keyframe).
        visibilities: Visibility-token bins to include (e.g. ['', '1', '2', '3', '4']);
            annotations outside these bins are skipped.

    Returns:
        One record (see generate_record) per annotation that has at least one
        corner in front of the camera.

    Raises:
        AssertionError: If sample_data_token isn't a camera sensor.
        ValueError: If the sample_data record isn't a keyframe (2D re-projections
            are only available for keyframes).
    """
    sd_rec = nusc.get('sample_data', sample_data_token)

    assert sd_rec['sensor_modality'] == 'camera', 'Error: only works for camera sample_data!'
    if not sd_rec['is_key_frame']:
        raise ValueError('The 2D re-projections are available only for keyframes.')

    s_rec = nusc.get('sample', sd_rec['sample_token'])

    cs_rec = nusc.get('calibrated_sensor', sd_rec['calibrated_sensor_token'])
    pose_rec = nusc.get('ego_pose', sd_rec['ego_pose_token'])
    camera_intrinsic = np.array(cs_rec['camera_intrinsic'])

    ann_recs = [nusc.get('sample_annotation', token) for token in s_rec['anns']]
    # Keep only annotations within the requested visibility bins.
    ann_recs = [ann_rec for ann_rec in ann_recs if ann_rec['visibility_token'] in visibilities]

    repro_recs = []

    for ann_rec in ann_recs:
        # Stash these tokens on the record so generate_record's relevant_keys
        # filter can pick them up below.
        ann_rec['sample_annotation_token'] = ann_rec['token']
        ann_rec['sample_data_token'] = sample_data_token

        # Get the box in global coordinates (annotation's local corners already
        # converted to global via its own translation/rotation).
        annotation_box = nusc.get_box(ann_rec['token'])
        corners_global_frame = annotation_box.corners()  # <np.float32: 3, 8>, all 8 corners

        # Move to the ego-pose frame (global -> ego).
        annotation_box.translate(-np.array(pose_rec['translation']))
        annotation_box.rotate(Quaternion(pose_rec['rotation']).inverse)
        corners_ego_frame = annotation_box.corners()  # <np.float32: 3, 8>, all 8 corners

        # Move to the calibrated sensor (camera) frame (ego -> camera).
        annotation_box.translate(-np.array(cs_rec['translation']))
        annotation_box.rotate(Quaternion(cs_rec['rotation']).inverse)
        corners_camera_frame = annotation_box.corners()  # <np.float32: 3, 8>, all 8 corners

        # Filter out corners that are behind the camera.
        corners_3d = corners_camera_frame
        in_front = np.argwhere(corners_3d[2, :] > 0).flatten()
        corners_3d = corners_3d[:, in_front]

        if corners_3d.shape[1] == 0:
            # Every corner is behind the camera; nothing to project for this annotation.
            continue

        # Apply the intrinsic calibration matrix (homogeneous, before perspective divide).
        corners_after_calibration = camera_intrinsic @ corners_3d  # <np.float32: 3, n>

        # Pre-clip: perspective-divide to get raw 2D pixel coordinates.
        corner_coords = view_points(corners_3d, camera_intrinsic, True).T[:, :2].tolist()

        # Post-clip: intersect the corners' convex hull with the image canvas.
        final_coords = post_process_coords(corner_coords)

        repro_rec = generate_record(ann_rec, corners_global_frame.T.tolist(), corners_ego_frame.T.tolist(),
                                    corners_camera_frame.T.tolist(), corners_after_calibration.T.tolist(),
                                    corner_coords, final_coords, sample_data_token, sd_rec['filename'])
        repro_recs.append(repro_rec)

    return repro_recs


def main(args: argparse.Namespace) -> None:
    """Generate 2D re-projections with every intermediate transform stage for each camera keyframe.

    Iterates every camera keyframe sample_data token in the dataset (optionally
    limited by args.image_limit), collects get_2d_boxes_with_intermediates for
    each, and writes the combined list to <dataroot>/<version>/<filename> as JSON.

    Args:
        args: Parsed CLI arguments — see the argparse setup below for the
            expected fields (dataroot, version, filename, visibilities, image_limit).
    """
    print("Generating 2D reprojections (with intermediate transform stages) of the nuScenes dataset")

    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=True)

    # Only camera sample_data at keyframes have 2D re-projections available.
    sample_data_camera_tokens = [s['token'] for s in nusc.sample_data if (s['sensor_modality'] == 'camera') and
                                 s['is_key_frame']]

    if args.image_limit != -1:
        # Debugging/smoke-test aid: process only the first N camera images.
        sample_data_camera_tokens = sample_data_camera_tokens[:args.image_limit]

    reprojections = []
    for token in sample_data_camera_tokens:
        reprojections.extend(get_2d_boxes_with_intermediates(nusc, token, args.visibilities))

    dest_path = os.path.join(args.dataroot, args.version)
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)
    out_file = os.path.join(dest_path, args.filename)
    with open(out_file, 'w') as fh:
        json.dump(reprojections, fh, sort_keys=True, indent=4)

    print(f"Saved {len(reprojections)} records with intermediate transform stages under {out_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export 2D annotations with intermediate transform-stage values.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataroot', type=str, default='data/nuscenes', help="Path where nuScenes is saved.")
    parser.add_argument('--version', type=str, default='v1.0-mini', help='Dataset version.')
    parser.add_argument('--filename', type=str, default='image_annotations_debug.json', help='Output filename.')
    # '' is nuScenes' "unknown visibility" bin; the rest are the 4 visibility-percent buckets.
    parser.add_argument('--visibilities', type=str, default=['', '1', '2', '3', '4'],
                        help='Visibility bins, the higher the number the higher the visibility.', nargs='+')
    parser.add_argument('--image_limit', type=int, default=-1, help='Number of images to process or -1 to process all.')
    args = parser.parse_args()

    main(args)
