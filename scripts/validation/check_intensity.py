"""Validates LidarProjection's column-slicing of the raw LIDAR_TOP .pcd file
against nuScenes' own independent loader (LidarPointCloud.from_file).

Unlike x/y/z, intensity never passes through a RigidTransform -- it's a raw
per-point value read straight off the file (LidarProjection's
raw_points[:, 3]), so there's no geometric correctness to check the way the
other validation scripts do. The actual risk here is indexing: nuScenes'
LIDAR_TOP format is (x, y, z, intensity, ring_index), and an off-by-one
column error would silently hand us ring_index instead of intensity without
looking obviously wrong. This exists because the design plan's 3D
scatter-by-intensity view is the first consumer of this channel, and nothing
else so far has verified it.

Cross-checks both xyz and intensity (not just intensity) against the
devkit's own LidarPointCloud.from_file, loaded via nusc.get_sample_data_path
so it reads the identical file LidarProjection does, for an exact
(not approximate) match.

Run with: python scripts/validation/check_intensity.py (from the repo root,
after ingesting/having local access to the configured nuScenes dataset).
"""

import json
import os

import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud

from av_data_pipeline.config import NUSCENES_VERSION, NUSCENES_DATAROOT
from av_data_pipeline.ingestion import build_scene_data
from av_data_pipeline.ingestion_keys import SAMPLE_INGESTION_KEYS, SENSOR_MODALITY_KEYS
from bev.lidar import LidarProjection

REPORT_DIR = "reports"
REPORT_FILE = os.path.join(REPORT_DIR, "intensity_check.txt")
DATA_FILE = os.path.join(REPORT_DIR, "intensity_check_data.json")
NUM_SCENES = 3
NUM_SAMPLES_PER_SCENE = 5


if __name__ == "__main__":
    nusc = NuScenes(version=NUSCENES_VERSION, dataroot=NUSCENES_DATAROOT, verbose=True)

    results = []
    for scene in nusc.scene[:NUM_SCENES]:
        scene_data = build_scene_data(nusc, scene["token"])[:NUM_SAMPLES_PER_SCENE]

        for sample_data in scene_data:
            sample_token = sample_data[SAMPLE_INGESTION_KEYS['token']]

            for _, lidar_sensor_data in sample_data[SENSOR_MODALITY_KEYS['lidar']].items():
                lidar_projection = LidarProjection(lidar_sensor_data)

            sample = nusc.get("sample", sample_token)
            lidar_sd_token = sample["data"]["LIDAR_TOP"]
            devkit_path = nusc.get_sample_data_path(lidar_sd_token)
            devkit_pc = LidarPointCloud.from_file(devkit_path)

            xyz_match = bool(np.array_equal(lidar_projection.lidar_points, devkit_pc.points[:3, :].T))
            intensity_match = bool(np.array_equal(lidar_projection.intensity, devkit_pc.points[3, :]))

            results.append({
                "sample_token": sample_token,
                "scene_name": scene["name"],
                "num_points": int(lidar_projection.lidar_points.shape[0]),
                "xyz_match": xyz_match,
                "intensity_match": intensity_match,
            })

    all_match = all(r["xyz_match"] and r["intensity_match"] for r in results)
    print(f"Checked {len(results)} samples across {NUM_SCENES} scenes: "
          f"{'PASS' if all_match else 'FAIL'} "
          f"({sum(r['xyz_match'] and r['intensity_match'] for r in results)}/{len(results)} exact matches)")

    os.makedirs(REPORT_DIR, exist_ok=True)

    with open(DATA_FILE, "w") as data_outfile:
        json.dump({"samples": results}, data_outfile, indent=2)
    print(f"Wrote {DATA_FILE}")

    with open(REPORT_FILE, "w") as outfile:
        outfile.write("--- Intensity/xyz column check: LidarProjection vs. devkit's LidarPointCloud.from_file ---\n")
        outfile.write(f"Verified: {sum(r['xyz_match'] and r['intensity_match'] for r in results)}/{len(results)}\n")
        mismatches = [r for r in results if not (r["xyz_match"] and r["intensity_match"])]
        if mismatches:
            outfile.write("Mismatches:\n")
            for r in mismatches:
                outfile.write(f"  ({r['sample_token']}, {r['scene_name']}): "
                               f"xyz_match={r['xyz_match']}, intensity_match={r['intensity_match']}\n")
    print(f"Wrote {REPORT_FILE}")
