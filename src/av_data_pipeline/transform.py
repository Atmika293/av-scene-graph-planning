"""Geometry pipeline for projecting 3D scene-graph annotations into camera image space.

Given per-sample sensor calibration (ego pose + camera extrinsics/intrinsics) and
3D bounding-box annotations in the global (world) frame, this module chains the
global -> ego -> camera -> sensor(pixel) transforms and clips the resulting
projection to the camera's canvas to produce 2D bounding boxes per annotation
per camera.
"""

from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation, RigidTransform
import cv2

from av_data_pipeline.ingestion_keys import *


class CameraTransformPipeline:
    """Projects 3D points in the global frame into a single camera's 2D canvas.

    Wraps the rigid-body transforms and intrinsic calibration for one camera
    sensor within a sample, and exposes the full global-coordinates -> canvas
    bounding-box pipeline via `apply`.
    """

    def __init__(self, sensor_data: dict):
        """Build the ego and camera rigid transforms and store intrinsics/canvas size.

        Args:
            sensor_data: Per-camera sensor record containing ego-pose rotation/
                translation (EP_ROTATION, EP_TRANSLATION), camera extrinsics
                (SC_EXT_ROTATION, SC_EXT_TRANSLATION), intrinsic calibration
                matrix (SC_INTRINSIC), and sensor resolution (SENSOR_WD, SENSOR_HT).
        """
        # Rigid transform taking points from the ego (vehicle) frame to the global frame.
        self.T_ego_to_global = self.__build_transform_matrix(rotation=sensor_data[TRANSFORM_KEYS['ego_pose_rotation']],
                                                             translation=sensor_data[TRANSFORM_KEYS['ego_pose_translation']])
        # Rigid transform taking points from the camera frame to the ego (vehicle) frame.
        self.T_camera_to_ego = self.__build_transform_matrix(rotation=sensor_data[TRANSFORM_KEYS['sensor_rotation']],
                                                             translation=sensor_data[TRANSFORM_KEYS['sensor_translation']])
        self.calibration_matrix = np.array(sensor_data[TRANSFORM_KEYS['intrinsic_matrix']])
        self.sensor_width = sensor_data[TRANSFORM_KEYS['canvas_width']]
        self.sensor_height = sensor_data[TRANSFORM_KEYS['canvas_height']]


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


    def global_to_camera(self, global_coords: np.array) -> np.array:
        """Transform points from the global frame into this camera's frame.

        Args:
            global_coords: Array of 3D points expressed in the global frame.

        Returns:
            The input points expressed in the camera frame.
        """
        # Compose global<-ego and ego<-camera, then invert to go global -> camera.
        transform = self.T_ego_to_global * self.T_camera_to_ego
        return transform.apply(global_coords, inverse=True)


    def camera_to_sensor(self, camera_coords: np.array, get_mask=False) -> np.array:
        """Project camera-frame 3D points onto the 2D sensor plane via the intrinsics.

        Points behind the camera (non-positive depth) are discarded before projection.

        Args:
            camera_coords: Array of 3D points expressed in the camera frame.

        Returns:
            Array of 2D points in sensor (pixel) coordinates for the points in front
            of the camera.
        """
        # Keep only points in front of the camera (positive depth along the optical axis).
        in_front = np.argwhere(camera_coords[:, 2] > 0).flatten()
        camera_coords = camera_coords[in_front, :]
        sensor_coords = self.calibration_matrix @ camera_coords.T
        # Perspective divide: normalize by depth to get pixel coordinates.
        normalized_pts = sensor_coords[:2, :] / sensor_coords[2, :]

        if get_mask:
            return normalized_pts.T, in_front
        else:
            return normalized_pts.T


    def clip_to_canvas(self, sensor_projection: np.array) -> Optional[list]:
        """Clip the projected point cloud's convex hull to the sensor canvas bounds.

        Args:
            sensor_projection: Array of 2D points in sensor (pixel) coordinates.

        Returns:
            The axis-aligned [xmin, ymin, xmax, ymax] bounding box of the overlap
            between the projected hull and the canvas, or None if there is no
            overlap.
        """
        # Compute convex_hull of projected points
        if sensor_projection.shape[0] > 0:
            convex_hull = cv2.convexHull(sensor_projection.astype(np.float32))

            canvas_hull = np.array([[0, 0], [self.sensor_width, 0],
                                    [self.sensor_width, self.sensor_height],
                                    [0, self.sensor_height]], dtype=np.float32)

            area, intersection_points = cv2.intersectConvexConvex(convex_hull, canvas_hull, handleNested=True)

            if area > 0:
                pts = intersection_points.reshape(-1, 2)   # cv2 often returns shape (N,1,2); flatten to (N,2)
                xmin, ymin = pts[:, 0].min(), pts[:, 1].min()
                xmax, ymax = pts[:, 0].max(), pts[:, 1].max()

                return [xmin, ymin, xmax, ymax]

        return None

    def apply(self, global_coordinates: np.array) -> Optional[list]:
        """Run the full global-frame -> canvas bounding-box pipeline for this camera.

        Args:
            global_coordinates: Array of 3D points (e.g. box corners) in the global frame.

        Returns:
            The clipped 2D bounding box on this camera's canvas, or None if the
            points don't project onto the canvas.
        """
        return self.clip_to_canvas(self.camera_to_sensor(self.global_to_camera(global_coordinates)))


class SampleAnnotation:
    """A single 3D bounding-box annotation, expressed as corners in its own local frame
    plus the rigid transform from that local (object) frame into the global frame.
    """

    def __init__(self, sample_annotation: dict):
        """Build the annotation's box corners and its local-to-global transform.

        Args:
            sample_annotation: Annotation record containing box dimensions (SIZE),
                and pose rotation/translation (ANN_ROTATION, ANN_TRANSLATION).
        """
        self.corners = self.__extract_corners(sample_annotation[TRANSFORM_KEYS['annotation_size']])
        self.transform = self.__build_transform_matrix(rotation=sample_annotation[TRANSFORM_KEYS['annotation_rotation']],
                                                       translation=sample_annotation[TRANSFORM_KEYS['annotation_translation']])


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


    def __extract_corners(self, dimensions_3d: list) -> np.array:
        """Compute the 8 corners of the box in its local (object-centered) frame.

        Args:
            dimensions_3d: Box dimensions as [width, length, height].

        Returns:
            Array of shape (8, 3) with the box's corners centered on the origin.
        """
        # dimensions: [width, length, height]
        w_by_2, l_by_2, h_by_2 = 0.5 * dimensions_3d[0], 0.5 * dimensions_3d[1], 0.5 * dimensions_3d[2]
        corners = [
            [l_by_2, w_by_2, h_by_2],
            [l_by_2, w_by_2, -h_by_2],
            [l_by_2, -w_by_2, h_by_2],
            [l_by_2, -w_by_2, -h_by_2],
            [-l_by_2, -w_by_2, h_by_2],
            [-l_by_2, -w_by_2, -h_by_2],
            [-l_by_2, w_by_2, h_by_2],
            [-l_by_2, w_by_2, -h_by_2]
        ]

        return np.array(corners, dtype=float)


    def globalcoordinates(self) -> np.array:
        """Transform this annotation's local-frame corners into the global frame.

        Returns:
            Array of shape (8, 3) with the box's corners in the global frame.
        """
        return self.transform.apply(self.corners)


class SampleTransformPipeline:
    """Projects every annotation in a sample onto every camera's canvas.

    Builds one CameraTransformPipeline per camera sensor and one SampleAnnotation
    per annotation with LiDAR support in the sample, then computes per-camera 2D
    bounding boxes for each annotation.
    """

    def __init__(self, sample_data: dict):
        """Build per-camera transform pipelines and per-annotation 3D boxes for a sample.

        Args:
            sample_data: Sample record containing a "CAMERA" mapping of sensor
                records keyed by camera name, and an "ANNOTATIONS" mapping of
                annotation records keyed by annotation token. Annotations with
                zero LiDAR points (N_LIDAR_PTS) are skipped.
        """
        self.camera_pipelines = {}
        for _, sensor_data in sample_data[SENSOR_MODALITY_KEYS['camera']].items():
            sample_data_token_key = SAMPLE_DATA_INGESTION_KEYS['token']
            self.camera_pipelines[sensor_data[sample_data_token_key]] = CameraTransformPipeline(sensor_data)

        self.annotations = {}
        annotations_key = SAMPLE_INGESTION_KEYS['annotations']
        for ann_token, ann_data in sample_data[annotations_key].items():
            if ann_data[SAMPLE_ANNOTATION_INGESTION_KEYS['n_lidar']] > 0:
                self.annotations[ann_token] = SampleAnnotation(ann_data)


    def project_to_canvas(self) -> dict:
        """Project every annotation onto every camera's canvas.

        Returns:
            Mapping from annotation token to a mapping from camera sample-data
            token to that camera's clipped [xmin, ymin, xmax, ymax] bounding box.
            Cameras where the annotation doesn't project onto the canvas are
            omitted from the inner mapping.
        """
        canvas_bboxes = {}

        for ann_token, annotation in self.annotations.items():
            corners = annotation.globalcoordinates()
            canvas_bboxes[ann_token] = {}

            for sd_token, camera_pipe in self.camera_pipelines.items():
                bbox = camera_pipe.apply(corners)
                if bbox is not None:
                    canvas_bboxes[ann_token][sd_token] = bbox

        return canvas_bboxes



def project_scene_annotations_to_canvas(scene_data: list[dict]) -> list[dict]:
    canvas_projections = []

    for sample_data in scene_data:
        canvas_projections.append(SampleTransformPipeline(sample_data).project_to_canvas())

    return canvas_projections