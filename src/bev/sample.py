"""Projects a sample's annotated 3D boxes from the global frame into the ego frame.

For each annotation with LiDAR support in a sample, SampleTransformPipeline
builds a SampleAnnotation (av_data_pipeline.transform) for its global-frame
box corners, then uses the sample's own ego pose (from its LIDAR_TOP sensor
record) to express those corners in the ego frame -- the same frame
bev.lidar.LidarProjection's points are in, so the two can be compared or
overlaid for a BEV footprint view.
"""

import numpy as np
from scipy.spatial.transform import Rotation, RigidTransform

from av_data_pipeline.ingestion_keys import *
from av_data_pipeline.transform import SampleAnnotation

class SampleBoxProjection:
    """One sample's LiDAR-supported annotations, as ego-frame 3D box corners.

    Builds the ego<->global rigid transform from the sample's LIDAR_TOP ego
    pose, and one SampleAnnotation per annotation with LiDAR support
    (N_LIDAR_PTS > 0) in the sample.
    """

    def __init__(self, sample_data: dict):
        """Build the ego<->global transform and per-annotation 3D boxes for a sample.

        Args:
            sample_data: Sample record containing a "LIDAR" mapping (expected
                to hold exactly one entry, LIDAR_TOP) keyed by sensor
                sample-data token, and an "ANNOTATIONS" mapping of annotation
                records keyed by annotation token. Annotations with zero
                LiDAR points (N_LIDAR_PTS) are skipped.
        """
        # Rigid transform taking points from the ego (vehicle) frame to the global frame.
        for _, lidar_sensor_data in sample_data[SENSOR_MODALITY_KEYS['lidar']].items():
            self.T_ego_to_global = self.__build_transform_matrix(rotation=lidar_sensor_data[TRANSFORM_KEYS['ego_pose_rotation']],
                                                                translation=lidar_sensor_data[TRANSFORM_KEYS['ego_pose_translation']])

        self.annotations = {}
        annotations_key = SAMPLE_INGESTION_KEYS['annotations']
        for ann_token, ann_data in sample_data[annotations_key].items():
            if ann_data[SAMPLE_ANNOTATION_INGESTION_KEYS['n_lidar']] > 0:
                self.annotations[ann_token] = SampleAnnotation(ann_data)


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


    def project_to_ego(self) -> dict:
        """Transform every LiDAR-supported annotation's box corners into the ego frame.

        Returns:
            Mapping from annotation token to that annotation's 3D box corners
            (8, 3), transformed from the global frame into the ego frame.
        """
        ego_coordinates = {}

        for ann_token, annotation in self.annotations.items():
            corners = annotation.globalcoordinates()
            ego_coords = self.T_ego_to_global.apply(corners, inverse=True)

            ego_coordinates[ann_token] = ego_coords

        return ego_coordinates