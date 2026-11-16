"""Assembles a window of consecutive LIDAR_TOP sweeps, each coloured by its own cameras.

A single keyframe sweep is sparse at range. This module walks back through a
scene's LiDAR sweeps (keyframes at 2 Hz and the in-between sweeps at 20 Hz),
pairs each sweep with every camera's frame nearest to it in time, colours
each sweep with GeometricBEVPipeline exactly as for a single sample, and
brings every sweep's points into one reference sweep's ego frame so they can
be rasterized on a shared grid (bev.raster.build_raster per sweep, then
merged newest first).

The reference sweep is the newest one in the window, and a keyframe, so its
annotations are available. Static scenery from older sweeps lines up with it
because each sweep goes through its own ego pose to the global frame; moving
objects from older sweeps do not, and appear where they were at the time.
"""

import numpy as np

from nuscenes.nuscenes import NuScenes

from av_data_pipeline.nuscenes import *
from av_data_pipeline.ingestion import extract_sensor_data_from_sample_data
from av_data_pipeline.ingestion_keys import (SAMPLE_DATA_INGESTION_KEYS, KNOWN_SENSOR_CHANNELS, 
                                             KNOWN_SENSOR_MODALITIES, SENSOR_MODALITY_KEYS)
from bev.fusion import GeometricBEVPipeline, LidarProjection

# Number of LiDAR sweeps in the window, reference sweep included (~0.5 s at 20 Hz).
WINDOW_LENGTH = 10
# Largest accepted time gap between a lidar sweep and a camera frame matched to
# it, in microseconds; a camera with no frame this close leaves the sweep uncoloured.
MAX_TIME_GAP_US = 50000


def nearest_by_time(camera_ts: np.array, lidar_ts: np.array, max_gap_us: int = MAX_TIME_GAP_US) -> np.array:
    """Index of each lidar timestamp's nearest camera frame, or -1 if none within max_gap_us.

    Binary search (np.searchsorted) rather than computing an index from the
    nominal frame rate: camera frames aren't evenly spaced (some are dropped),
    so a rate-based index can silently land on the wrong frame.

    Args:
        camera_ts: One camera's frame timestamps, sorted ascending, in
            microseconds (int64: nuScenes timestamps overflow int32).
        lidar_ts: Lidar sweep timestamps to match, in microseconds.
        max_gap_us: Largest accepted time gap, in microseconds; a sweep with
            no frame that close gets -1 rather than a distant frame.

    Returns:
        One index into camera_ts per lidar timestamp, or -1 where no frame is
        within max_gap_us.
    """
    # searchsorted gives the frame just after each lidar timestamp; the nearest
    # may be that one or the one before it. Clipping keeps both neighbours
    # valid for timestamps before the first frame or after the last.
    right = np.clip(np.searchsorted(camera_ts, lidar_ts), 1, len(camera_ts) - 1)
    left = right - 1
    nearest = np.where(lidar_ts - camera_ts[left] <= camera_ts[right] - lidar_ts, left, right)
    return np.where(np.abs(camera_ts[nearest] - lidar_ts) <= max_gap_us, nearest, -1)


def build_lidar_window(nusc: NuScenes, last_sample_data_token: str, window_length: int) -> tuple[list[str], list[int]]:
    """Walk back to the keyframe before last_sample_data_token, then collect a window of sweeps from there.

    Args:
        nusc: NuScenes devkit instance used to look up records.
        last_sample_data_token: A LIDAR_TOP sample_data token, normally a
            keyframe's. The window starts at the keyframe before it.
        window_length: Maximum number of sweeps to collect.

    Returns:
        (sweep_tokens, sweep_timestamps), newest first: the earlier keyframe
        (the reference sweep), then the sweeps before it, up to window_length
        in total. Shorter if the scene's first sweep is reached first, and
        both empty if there's no earlier keyframe.
    """
    sweep_tokens = []
    sweep_timestamps = []

    # Step past the given sweep, then keep walking back until the previous keyframe.
    current_sd = nusc.get(SAMPLE_DATA_TABLE, last_sample_data_token)
    current_data_token = current_sd[PREVIOUS_SAMPLE_DATA]
    while True:
        if current_data_token == '':
            break

        current_sd = nusc.get(SAMPLE_DATA_TABLE, current_data_token)

        if current_sd['is_key_frame']:
            break

        current_data_token = current_sd[PREVIOUS_SAMPLE_DATA]

    # Collect the window, from that keyframe backwards in time.
    for _ in range(window_length):
        if current_data_token == '':
            break

        current_sd = nusc.get(SAMPLE_DATA_TABLE, current_data_token)
        sweep_timestamps.append(current_sd[SAMPLE_DATA_TIMESTAMP])
        sweep_tokens.append(current_data_token)
        current_data_token = current_sd[PREVIOUS_SAMPLE_DATA]

    return sweep_tokens, sweep_timestamps


def match_camera_frames_to_sweeps(nusc: NuScenes, first_camera_data_token: str, 
                                  lidar_sweep_tokens: list[str], 
                                  lidar_sweep_timestamps: list[int]) -> dict[str, str]:
    """Match each lidar sweep to one camera's nearest frame in time.

    Args:
        nusc: NuScenes devkit instance used to look up records.
        first_camera_data_token: Any sample_data token from the camera's
            chain (e.g. the reference sample's frame for that camera); the
            whole chain is walked in both directions from it.
        lidar_sweep_tokens: The lidar sweeps to match.
        lidar_sweep_timestamps: Their timestamps, in microseconds, in the
            same order.

    Returns:
        Lidar sweep token -> this camera's matched sample_data token. Sweeps
        with no frame within MAX_TIME_GAP_US are left out.
    """
    camera_frame_timestamps = []
    camera_frame_tokens = []

    # Walk back to the camera's first frame, then reverse so the list is in
    # ascending time order, as nearest_by_time requires.
    current_data_token = first_camera_data_token
    while current_data_token != '':
        current_sd = nusc.get(SAMPLE_DATA_TABLE, current_data_token)
        camera_frame_timestamps.append(current_sd[SAMPLE_DATA_TIMESTAMP])
        camera_frame_tokens.append(current_data_token)
        current_data_token = current_sd[PREVIOUS_SAMPLE_DATA]

    camera_frame_timestamps = camera_frame_timestamps[::-1]
    camera_frame_tokens = camera_frame_tokens[::-1]

    # Then walk forward to its last frame. The starting frame is appended a
    # second time here; the duplicate is harmless (still sorted, same token).
    current_data_token = first_camera_data_token
    while current_data_token != '':
        current_sd = nusc.get(SAMPLE_DATA_TABLE, current_data_token)
        camera_frame_timestamps.append(current_sd[SAMPLE_DATA_TIMESTAMP])
        camera_frame_tokens.append(current_data_token)
        current_data_token = current_sd[NEXT_SAMPLE_DATA]

    camera_token_indices = nearest_by_time(np.array(camera_frame_timestamps, dtype=np.int64),
                                           np.array(lidar_sweep_timestamps, dtype=np.int64), 
                                           max_gap_us=MAX_TIME_GAP_US).tolist()

    lidar_camera_map = {}
    for lidar_token, camera_token_index in zip(lidar_sweep_tokens, camera_token_indices):
        if camera_token_index > -1:
            lidar_camera_map[lidar_token] = camera_frame_tokens[camera_token_index]

    return lidar_camera_map


def assemble_window(nusc: NuScenes, scene_token: str) -> tuple[list[str], dict[str, dict[str, str]]]:
    """Build a scene's sweep window and pair every sweep with its cameras.

    Starts from the scene's last sample: its LIDAR_TOP token anchors the
    window (see build_lidar_window), and its camera tokens give each camera's
    frame chain to match against.

    Args:
        nusc: NuScenes devkit instance used to look up records.
        scene_token: Token of the scene.

    Returns:
        (lidar_sweep_tokens, lidar_camera_groups): the window's lidar sweep
        tokens, newest (the reference) first; and lidar sweep token ->
        {camera channel: matched camera sample_data token}, with a channel
        left out when it had no frame close enough in time.
    """
    scene = nusc.get(SCENE_TABLE, scene_token)

    last_sample_token = scene[LAST_SAMPLE_TOKEN]
    sample = nusc.get(SAMPLE_TABLE, last_sample_token)
    sample_data_tokens = sample[SAMPLE_DATA_TOKENS]

    sample_data = {}
    for channel, data_token in sample_data_tokens.items():
        # Extract this channel's sensor data (ego pose, file metadata, calibration)
        sensor_data = extract_sensor_data_from_sample_data(nusc, data_token)

        # Uppercased to match KNOWN_SENSOR_MODALITIES/KNOWN_SENSOR_CHANNELS and
        # the "CAMERA"/"LIDAR" keys this function files sensor_data under below.
        sensor_modality = sensor_data[SAMPLE_DATA_INGESTION_KEYS['modality']].upper()
        channel = sensor_data[SAMPLE_DATA_INGESTION_KEYS['channel']].upper()

        # Skip unknown modalities/channels (as ingestion.py does), and radar,
        # which the BEV doesn't use.
        if sensor_modality not in KNOWN_SENSOR_MODALITIES:
            continue

        if sensor_modality == SENSOR_MODALITY_KEYS['radar']:
            continue

        if channel not in KNOWN_SENSOR_CHANNELS[sensor_modality]:
            continue


        # File the sensor data under its modality (e.g. CAMERA) and channel name,
        # creating the modality's dict the first time it's encountered
        if sensor_modality in sample_data:
            sample_data[sensor_modality][channel] = sensor_data
        else:
            sample_data[sensor_modality] = { channel : sensor_data }

    # LIDAR_TOP is nuScenes' only lidar channel.
    lidar_sensor_channel = KNOWN_SENSOR_CHANNELS[SENSOR_MODALITY_KEYS['lidar']][0]
    lidar_data_token = sample_data[SENSOR_MODALITY_KEYS['lidar']][lidar_sensor_channel][SAMPLE_DATA_INGESTION_KEYS['token']]
    lidar_sweep_tokens, lidar_sweep_timestamps = build_lidar_window(nusc, lidar_data_token, 
                                                                    window_length=WINDOW_LENGTH)

    # Match each camera separately: each one's frames have their own timing and gaps.
    lidar_camera_groups = { lidar_token: {} for lidar_token in lidar_sweep_tokens }
    for channel in KNOWN_SENSOR_CHANNELS[SENSOR_MODALITY_KEYS['camera']]:
        camera_data_token = sample_data[SENSOR_MODALITY_KEYS['camera']][channel][SAMPLE_DATA_INGESTION_KEYS['token']]
        lidar_camera_map = match_camera_frames_to_sweeps(nusc, camera_data_token,
                                                         lidar_sweep_tokens, 
                                                         lidar_sweep_timestamps)

        for lidar_token in lidar_camera_groups:
            if lidar_token in lidar_camera_map:
                lidar_camera_groups[lidar_token][channel] = lidar_camera_map[lidar_token]

    return lidar_sweep_tokens, lidar_camera_groups


def project_window_to_bev(nusc: NuScenes, lidar_sweep_tokens: list[str], 
                         lidar_camera_groups: dict[str, dict[str, str]]) -> list[dict]:
    """Colour every sweep in the window and bring its points into the reference ego frame.

    Each sweep is run through GeometricBEVPipeline as if it were one sample,
    with its own matched camera frames, so it's coloured by images taken at
    nearly the same moment.

    Args:
        nusc: NuScenes devkit instance used to look up records.
        lidar_sweep_tokens: The window's sweeps, newest first; the first is
            the reference sweep (see assemble_window).
        lidar_camera_groups: Lidar sweep token -> {camera channel: camera
            sample_data token} (see assemble_window).

    Returns:
        One dict per sweep, in the same newest-first order, each indexed
        per point the same way:
          points (N, 3) -- the sweep's points in the reference sweep's ego frame.
          colors (N, 3) -- RGB, or [-1, -1, -1] where none of this sweep's
              matched cameras saw the point.
          distance (N,) -- each point's distance from its own sweep's ego
              origin, for build_raster's distance argument.
          intensity (N,) -- each point's lidar return intensity.
    """
    reference_lidar_projection = None
    sweep_projections = []

    for count, lidar_token in enumerate(lidar_sweep_tokens, 0):
        lidar_aligned_sample_data = {}
        lidar_aligned_sample_data[SENSOR_MODALITY_KEYS['lidar']] = {}

        for channel in KNOWN_SENSOR_CHANNELS[SENSOR_MODALITY_KEYS['lidar']]:
            lidar_sensor_data = extract_sensor_data_from_sample_data(nusc, lidar_token)
            lidar_aligned_sample_data[SENSOR_MODALITY_KEYS['lidar']][channel] = lidar_sensor_data

            # The first sweep is the reference: every sweep's points go into its ego frame.
            if count == 0:
                reference_lidar_projection = LidarProjection(lidar_sensor_data)

        # Only the cameras matched to this sweep; GeometricBEVPipeline handles fewer than six.
        lidar_aligned_sample_data[SENSOR_MODALITY_KEYS['camera']] = {}
        for channel in lidar_camera_groups[lidar_token]:
            lidar_aligned_sample_data[SENSOR_MODALITY_KEYS['camera']][channel] = extract_sensor_data_from_sample_data(nusc, lidar_camera_groups[lidar_token][channel])

        bev_pipeline = GeometricBEVPipeline(lidar_aligned_sample_data)
        global_lidar_points, ego_lidar_points, bev_colors = bev_pipeline.project_to_bev(return_global_points=True)
        # Through the global frame into the reference sweep's ego frame, so
        # every sweep's points share one grid.
        ref_ego_lidar_points = reference_lidar_projection.global_to_ego(global_lidar_points)

        sweep_projections.append({
            "points": ref_ego_lidar_points,
            "colors": bev_colors,
            "distance": np.linalg.norm(ego_lidar_points, axis=1),
            "intensity": bev_pipeline.get_lidar_projection_pipeline().get_lidar_intensity()
        })

    return sweep_projections
    

    







    

            







    
    

    


    