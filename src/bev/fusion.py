
import os

import numpy as np
import cv2

from av_data_pipeline.config import NUSCENES_DATAROOT
from av_data_pipeline.ingestion_keys import *
from av_data_pipeline.transform import CameraTransformPipeline

from bev.lidar import LidarProjection

class GeometricBEVPipeline:
    """Builds a colored, top-down point cloud for one sample by fusing its LiDAR
    sweep with every camera it's visible in.

    Builds one CameraTransformPipeline per camera sensor (reused from
    av_data_pipeline.transform for the global<->camera<->pixel projection) and
    one LidarProjection for the sample's LIDAR_TOP sweep, then (via
    project_to_bev) looks up each lidar point's color from whichever camera
    sees it.
    """

    def __init__(self, sample_data: dict):
        """Build the per-camera transform pipelines and the sample's lidar projection.

        Args:
            sample_data: Sample record containing a "CAMERA" mapping of sensor
                records keyed by camera sample-data token, and a "LIDAR"
                mapping (expected to hold exactly one entry, LIDAR_TOP --
                nuScenes' only lidar channel) keyed the same way.
        """
        self.camera_pipelines = {}
        self.sample_camera_images = {}
        for _, sensor_data in sample_data[SENSOR_MODALITY_KEYS['camera']].items():
            sample_data_token_key = SAMPLE_DATA_INGESTION_KEYS['token']
            self.camera_pipelines[sensor_data[sample_data_token_key]] = CameraTransformPipeline(sensor_data)
            # SD_FILEPATH is dataroot-relative; see LidarProjection.__init__'s comment.
            self.sample_camera_images[sensor_data[sample_data_token_key]] = os.path.join(
                NUSCENES_DATAROOT, sensor_data[SAMPLE_DATA_INGESTION_KEYS['filepath']])
        self.n_cameras = len(sample_data[SENSOR_MODALITY_KEYS['camera']])

        # nuScenes samples carry exactly one lidar channel (LIDAR_TOP), so this
        # loop always runs once; it just avoids hardcoding that channel name.
        for _, lidar_sensor_data in sample_data[SENSOR_MODALITY_KEYS['lidar']].items():
            self.lidar_projection = LidarProjection(lidar_sensor_data)


    def map_lidar_points_to_camera_canvas(self, camera_pipe: CameraTransformPipeline,
                                         lidar_points: np.array) -> tuple[np.array, np.array]:
        """Project global-frame lidar points onto one camera's pixel canvas.

        Points behind that camera are dropped by camera_to_sensor, so the
        returned array can have fewer rows than lidar_points.

        Args:
            camera_pipe: Transform pipeline for the camera to project onto.
            lidar_points: Lidar points (N, 3), in the global frame.

        Returns:
            Array of 2D pixel coordinates for the points in front of the camera.
        """
        return camera_pipe.camera_to_sensor(camera_pipe.global_to_camera(lidar_points), get_mask=True)


    def clip_to_canvas(self, canvas_coords: np.array, canvas_width: int, canvas_height: int):
        """Mask of which projected points land within the camera's pixel bounds.

        Args:
            canvas_coords: Projected pixel coordinates (N, 2), as returned by
                map_lidar_points_to_camera_canvas.
            canvas_width: Camera canvas width in pixels.
            canvas_height: Camera canvas height in pixels.

        Returns:
            Boolean mask (N,), intended to be True where a point's x is in
            [0, canvas_width) AND its y is in [0, canvas_height).
        """
        return ((canvas_coords[:, 0] >= 0) & (canvas_coords[:, 0] < canvas_width)) & \
                    ((canvas_coords[:, 1] >= 0) & (canvas_coords[:, 1] < canvas_height))


    def sample_colors(self, lidar_points: np.array, camera_pipe: CameraTransformPipeline,
                      camera_image: np.array) -> np.array:
        """Look up one camera's pixel color for each lidar point that projects onto it.

        Args:
            lidar_points: Lidar points (N, 3), in the global frame.
            camera_pipe: Transform pipeline for the camera to sample from.
            camera_image: That camera's image, as loaded by cv2 (BGR).

        Returns:
            Array of shape (N, 3): camera_image's pixel color at each lidar
            point's projected canvas location, or [-1, -1, -1] for points
            that don't land on this camera's canvas -- matching
            project_to_bev's own "uncolored" sentinel, since that's the
            caller that decides whether to try the next camera for a point.
        """
        sampled_colors = np.full((lidar_points.shape[0], 3), -1, dtype=int)

        # canvas_coords/in_front_mask are already restricted to the points in
        # front of this camera, so they're shorter than lidar_points;
        # in_front_mask[i] is lidar_points' original row index for
        # canvas_coords[i]. canvas_filter is relative to that same shorter
        # array, so it has to be composed with in_front_mask (not used on its
        # own) to know which rows of the full-length sampled_colors to write.
        canvas_coords, in_front_mask = self.map_lidar_points_to_camera_canvas(camera_pipe, lidar_points)
        canvas_filter = self.clip_to_canvas(canvas_coords, camera_pipe.sensor_width, camera_pipe.sensor_height)
        original_indices = in_front_mask[canvas_filter]
        clipped_canvas_coords = canvas_coords[canvas_filter]

        # remap requires float32 maps shaped like the output (one column here,
        # one row per point) and a contiguous src -- camera_image's BGR->RGB
        # conversion ([:, :, ::-1]) leaves it as a reversed-stride view.
        pixel_colors = cv2.remap(src=np.ascontiguousarray(camera_image),
                                     map1=clipped_canvas_coords[:, 0].astype(np.float32).reshape(-1, 1),
                                     map2=clipped_canvas_coords[:, 1].astype(np.float32).reshape(-1, 1),
                                     interpolation=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT,  # How to handle out-of-bounds coords
                                     borderValue=(0, 0, 0)            # Color used if coord is out-of-bounds
                                    )
        # remap's output mirrors the (M, 1) map shape -- squeeze back to (M, 3)
        # to match sampled_colors[original_indices, :] below.
        pixel_colors = pixel_colors.reshape(-1, 3)
        sampled_colors[original_indices, :] = pixel_colors

        return sampled_colors

    def project_to_bev(self) -> tuple[np.array, np.array]:
        """Color this sample's point cloud and express it in the ego frame for BEV plotting.

        Projects the lidar sweep (global frame) into every camera in turn,
        coloring each point (RGB) from the first camera that sees it -- points
        still uncolored (all -1) after a camera are tried against the next
        one, so later cameras only fill in points earlier ones missed.

        Returns:
            (ego_lidar_points, bev_colors): the sweep's points in the ego
            frame (N, 3), and their RGB colors (N, 3), still [-1, -1, -1]
            where no camera saw the point -- distinguishable from a genuinely
            black pixel, which would be [0, 0, 0].
        """
        global_lidar_points = self.lidar_projection.lidar_to_global()

        bev_colors = np.ones((global_lidar_points.shape[0], 3), dtype=int) * -1
        bev_filter = np.ones((global_lidar_points.shape[0], ), dtype=bool)

        for sd_token, camera_pipe in self.camera_pipelines.items():
            camera_image = cv2.imread(self.sample_camera_images[sd_token])[:, :, ::-1]  # BGR -> RGB

            # Only write still-uncolored points, then recompute which points
            # are still uncolored for the next camera.
            bev_colors[bev_filter] = self.sample_colors(global_lidar_points, camera_pipe, camera_image)[bev_filter]
            bev_filter = np.all(bev_colors == -1, axis=-1)

        ego_lidar_points = self.lidar_projection.lidar_to_ego()

        return ego_lidar_points, bev_colors


def project_scene_to_bev(scene_data: list[dict]) -> list[tuple[np.array, np.array]]:
    """Color every sample's LiDAR sweep in a scene, for BEV rendering.

    Args:
        scene_data: One sample_data dict per sample in the scene (as built by
            av_data_pipeline.ingestion.build_scene_data), each containing a
            "CAMERA" and a "LIDAR" mapping.

    Returns:
        One (ego_lidar_points, bev_colors) pair per sample, as returned by
        GeometricBEVPipeline.project_to_bev, in scene order.
    """
    return [GeometricBEVPipeline(sample_data).project_to_bev() for sample_data in scene_data]