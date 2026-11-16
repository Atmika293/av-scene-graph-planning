"""Loads one LIDAR_TOP sweep's point cloud and its lidar<->ego<->global transforms.

LidarProjection reads the raw .pcd.bin file (points in the lidar sensor's own
frame) plus the lidar extrinsics and ego pose needed to express those points
in the ego or global frame. Coloring the points from camera images -- the
next step toward a BEV render -- lives in bev.fusion.GeometricBEVPipeline,
which builds on this class.
"""

import os

import numpy as np
from scipy.spatial.transform import Rotation, RigidTransform

from av_data_pipeline.config import NUSCENES_DATAROOT
from av_data_pipeline.ingestion_keys import *



class LidarProjection:
    """One LIDAR_TOP sweep's point cloud (a keyframe sample's, or any sweep in
    between), plus the rigid transforms needed to express those points in the
    ego (vehicle) or global (world) frame.

    Intensity is loaded alongside the xyz positions (self.intensity, indexed
    the same as the points) but never passed through a transform -- it's a
    per-point sensor reading, not a coordinate, so it's unaffected by frame
    changes.
    """

    def __init__(self, sensor_data: dict):
        """Load the point cloud and build the lidar<->ego and ego<->global transforms.

        Args:
            sensor_data: This sweep's LIDAR_TOP sensor record (see
                ingestion.extract_sensor_data_from_sample_data), containing
                ego-pose rotation/translation (EP_ROTATION, EP_TRANSLATION),
                lidar extrinsics (SC_EXT_ROTATION, SC_EXT_TRANSLATION), and
                the .pcd.bin file path (SD_FILEPATH).
        """
        # Rigid transform taking points from the ego (vehicle) frame to the global frame.
        self.T_ego_to_global = self.__build_transform_matrix(rotation=sensor_data[TRANSFORM_KEYS['ego_pose_rotation']],
                                                                translation=sensor_data[TRANSFORM_KEYS['ego_pose_translation']])
        # Rigid transform taking points from the lidar sensor frame to the ego (vehicle) frame.
        self.T_lidar_to_ego = self.__build_transform_matrix(rotation=sensor_data[TRANSFORM_KEYS['sensor_rotation']],
                                                                translation=sensor_data[TRANSFORM_KEYS['sensor_translation']])

        # SD_FILEPATH is dataroot-relative (nuScenes' own convention), so it
        # has to be joined with NUSCENES_DATAROOT before opening -- same as
        # nusc.get_sample_data_path does for the devkit's own callers.
        pcd_filepath = os.path.join(NUSCENES_DATAROOT, sensor_data[SAMPLE_DATA_INGESTION_KEYS['filepath']])
        raw_points = self.__extract_lidar_points_from_file(pcd_filepath)
        # Keep xyz and intensity separate so every transform method below only
        # ever sees (N, 3) coordinates -- intensity rides along by position,
        # not through RigidTransform.apply.
        self.lidar_points = raw_points[:, :3]
        self.intensity = raw_points[:, 3]

    def __extract_lidar_points_from_file(self, pcd_filepath: str) -> np.array:
        """Load a LIDAR_TOP .pcd.bin file's raw per-point rows, in the lidar sensor's own frame.

        nuScenes' LIDAR_TOP sweeps are stored as flat float32 binary files, one
        row of 5 values per point: x, y, z, intensity, ring index. Ring index
        isn't used anywhere in this pipeline, but intensity and xyz both are.

        Args:
            pcd_filepath: Path to the .pcd.bin file.

        Returns:
            Array of shape (N, 5): each point's raw (x, y, z, intensity, ring).
        """
        return np.fromfile(pcd_filepath, dtype=np.float32).reshape(-1, 5)


    def __build_transform_matrix(self, rotation: np.array, translation: np.array) -> RigidTransform:
        """Construct a RigidTransform from a quaternion rotation and translation vector.

        Args:
            rotation: Quaternion components, scalar-first (w, x, y, z).
            translation: 3D translation vector.

        Returns:
            The corresponding RigidTransform.
        """
        return RigidTransform.from_components(
            translation=translation,
            rotation=Rotation.from_quat(rotation, scalar_first=True),
        )


    def get_lidar_intensity(self) -> np.array:
        """Each point's return intensity (N,), indexed the same as the points."""
        return self.intensity


    def get_ego_to_global_transform(self) -> RigidTransform:
        """The ego-to-global rigid transform: the vehicle's pose when this sweep was captured."""
        return self.T_ego_to_global


    def lidar_to_ego(self) -> np.array:
        """Transform this sample's point cloud from the lidar sensor frame into the ego frame.

        Returns:
            The point cloud expressed in the ego frame.
        """
        return self.T_lidar_to_ego.apply(self.lidar_points)


    def lidar_to_global(self) -> np.array:
        """Transform this sample's point cloud from the lidar sensor frame into the global frame.

        Returns:
            The point cloud expressed in the global frame.
        """
        # Compose global<-ego and ego<-lidar to go lidar -> global.
        transform = self.T_ego_to_global * self.T_lidar_to_ego
        return transform.apply(self.lidar_points)


    def global_to_ego(self, global_coordinates: np.array) -> np.array:
        """Transform global-frame points into this sweep's ego frame.

        Unlike the methods above, this takes points as an argument rather than
        using this sweep's own: bev.multisweep uses it to bring other sweeps'
        points into one reference sweep's ego frame.

        Args:
            global_coordinates: Points (N, 3) in the global frame.

        Returns:
            The same points (N, 3), expressed in the ego frame at the time
            this sweep was captured.
        """
        # Compose global<-ego, then invert to go global -> ego.
        return self.T_ego_to_global.apply(global_coordinates, inverse=True)