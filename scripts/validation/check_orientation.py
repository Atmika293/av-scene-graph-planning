"""Validates the ego frame's forward-axis convention (ego +x = forward) against
real inter-sample motion, as an independent check on this pipeline's own
ego-pose rotation handling (see LidarProjection/SampleBoxProjection's
__build_transform_matrix).

For each pair of consecutive samples in a scene, expresses sample i+1's
global ego position in sample i's own ego frame (via sample i's own
ego<->global transform, inverse-applied) and measures how much of that
motion lies along ego +x vs +y. This doesn't assume the convention is
correct to test it: the check only relies on the physical fact that a
vehicle mostly drives forward rather than sideways between two ~0.5s-apart
samples, so a systematic swapped/flipped axis in the rotation handling
would show up as motion that isn't mostly forward.

Run with: python scripts/validation/check_orientation.py (from the repo root, after
ingesting/having local access to the configured nuScenes dataset).
"""

import json
import os

import numpy as np
from nuscenes.nuscenes import NuScenes
from scipy.spatial.transform import Rotation, RigidTransform

from av_data_pipeline.config import NUSCENES_VERSION, NUSCENES_DATAROOT
from av_data_pipeline.ingestion import build_scene_data
from av_data_pipeline.ingestion_keys import SENSOR_MODALITY_KEYS, TRANSFORM_KEYS

REPORT_DIR = "reports"
REPORT_FILE = os.path.join(REPORT_DIR, "orientation_check.json")
NUM_SCENES = 10
# Ego-pose GPS/IMU noise alone produces ~1e-5m of apparent motion between
# consecutive samples even when the vehicle is parked -- well below any real
# driving motion (samples are ~0.5s apart), so pairs below this are excluded
# rather than having their heading dominated by noise direction.
MIN_MOTION_METERS = 0.1


def ego_to_global_transform(sample_data: dict) -> RigidTransform:
    """Build a sample's ego<->global transform from its LIDAR_TOP ego pose.

    Args:
        sample_data: One sample's sample_data dict (as built by
            av_data_pipeline.ingestion.build_scene_data), containing a
            "LIDAR" mapping with exactly one entry (LIDAR_TOP).

    Returns:
        The sample's ego<->global RigidTransform.
    """
    for _, lidar_sensor_data in sample_data[SENSOR_MODALITY_KEYS['lidar']].items():
        return RigidTransform.from_components(
            translation=lidar_sensor_data[TRANSFORM_KEYS['ego_pose_translation']],
            rotation=Rotation.from_quat(lidar_sensor_data[TRANSFORM_KEYS['ego_pose_rotation']], scalar_first=True),
        )


def heading_cosine(transform_i: RigidTransform, transform_i1: RigidTransform) -> float:
    """Measure how much of sample i -> i+1's ego motion lies along ego +x (forward).

    Expresses sample i+1's global position in sample i's own ego frame via
    transform_i's inverse -- since sample i's own position maps to the
    origin under its own transform, this directly gives the motion vector
    in sample i's ego frame, no separate subtraction needed. Only the
    planar (x, y) components are used; z motion here is road-surface noise,
    not heading.

    Args:
        transform_i: Sample i's ego<->global transform.
        transform_i1: Sample i+1's ego<->global transform.

    Returns:
        cos(heading) = motion_x / ||motion_xy||: 1.0 for pure forward
        motion, 0.0 for pure sideways, -1.0 for pure backward. NaN if the
        pair's planar motion is below MIN_MOTION_METERS, i.e. too close to
        stationary for heading to be a meaningful direction rather than
        ego-pose noise.
    """
    motion_in_ego_i = transform_i.apply(transform_i1.translation, inverse=True)
    planar_motion = motion_in_ego_i[:2]
    planar_distance = np.linalg.norm(planar_motion)
    if planar_distance < MIN_MOTION_METERS:
        return np.nan
    return planar_motion[0] / planar_distance


if __name__ == "__main__":
    nusc = NuScenes(version=NUSCENES_VERSION, dataroot=NUSCENES_DATAROOT, verbose=True)

    results = []
    for scene in nusc.scene[:NUM_SCENES]:
        scene_data = build_scene_data(nusc, scene["token"])
        transforms = [ego_to_global_transform(sample_data) for sample_data in scene_data]

        cosines = np.array([
            heading_cosine(transforms[i], transforms[i + 1])
            for i in range(len(transforms) - 1)
        ])
        valid = ~np.isnan(cosines)
        pairs_with_motion = int(valid.sum())

        # NaN isn't valid JSON; a scene with zero moving pairs (e.g. parked
        # for its whole duration) reports null for both stats instead.
        mean_cosine = float(np.nanmean(cosines)) if pairs_with_motion else None
        backward_fraction = float(np.mean(cosines[valid] < 0)) if pairs_with_motion else None

        mean_cosine_str = f"{mean_cosine:.3f}" if mean_cosine is not None else "n/a"
        backward_fraction_str = f"{backward_fraction:.1%}" if backward_fraction is not None else "n/a"
        print(f"Scene {scene['name']}: {pairs_with_motion}/{len(cosines)} sample pairs with "
              f"real motion (>= {MIN_MOTION_METERS}m), "
              f"mean cos(heading)={mean_cosine_str}, "
              f"backward fraction={backward_fraction_str}")

        results.append({
            "scene_name": scene["name"],
            "scene_token": scene["token"],
            "total_pairs": len(cosines),
            "pairs_with_motion": pairs_with_motion,
            "min_motion_meters": MIN_MOTION_METERS,
            "mean_cos_heading": mean_cosine,
            "backward_fraction": backward_fraction,
        })

    os.makedirs(REPORT_DIR, exist_ok=True)
    with open(REPORT_FILE, "w") as outfile:
        json.dump(results, outfile, indent=2)
    print(f"\nWrote {REPORT_FILE}")
