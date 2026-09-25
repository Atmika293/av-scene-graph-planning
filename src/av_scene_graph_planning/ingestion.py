"""Ingestion helpers for pulling nuScenes records into flat, column-mapped dicts.

Walks scenes/samples via the NuScenes devkit and extracts sensor data, ego
poses, calibration, and annotations into the aggregate key names defined in
columns.py, ready for downstream transform/projection code.
"""

from nuscenes.nuscenes import NuScenes
from av_scene_graph_planning.utils import calculate_timestamp_skew_ms, visibility_range, extract_key_to_column
from av_scene_graph_planning.columns import *


def get_sample_tokens_from_scene(nusc: NuScenes, scene_token: str) -> list[str]:
    """Return all sample tokens belonging to a scene, in temporal order.

    Walks the sample linked list from the scene's first to last sample
    token via each sample's 'next' field.

    Args:
        nusc: NuScenes devkit instance used to look up records.
        scene_token: Token of the scene to walk.

    Returns:
        list[str]: Sample tokens for the scene, ordered from first to last.
    """
    # Look up the scene record for this token
    scene = nusc.get('scene', scene_token)

    # Extract the first and last sample token for the scene
    sample_token = scene['first_sample_token']
    last_sample_token = scene['last_sample_token']

    # Loop over the sample tokens belonging to a scene by using the
    # 'next' attribute in the sample dict
    sample_tokens = [sample_token]
    while sample_token != last_sample_token:
        sample = nusc.get('sample', sample_token)
        next_sample_token = sample['next']
        sample_tokens.append(next_sample_token)
        sample_token = next_sample_token

    sample_tokens.append(last_sample_token)

    return sample_tokens


def extract_sensor_data_from_sample_data(nusc: NuScenes, sample_data_token: str) -> tuple[str, str, dict]:
    """Extract sensor readings, ego pose, and calibration for one sample_data record.

    Returns a (sensor_modality, channel, sensor_data) tuple, where sensor_data
    contains the ego pose, file metadata, and intrinsic/extrinsic calibration.

    Args:
        nusc: NuScenes devkit instance used to look up records.
        sample_data_token: Token of the sample_data record to extract.

    Returns:
        tuple[str, str, dict]: The sensor modality, channel name, and a dict
        with keys 'ego_pose', 'timestamp', 'filepath', 'fileformat',
        'image_height', 'image_width', 'sensor_intrinsic', and
        'sensor_extrinsic'.
    """
    # Look up the sample_data record for this sensor reading
    sd_record = nusc.get('sample_data', sample_data_token)

    sensor_data = {"SAMPLE_DATA_TOKEN": sample_data_token}

    # sensor_modality distinguishes camera/lidar/radar; channel is the specific sensor name
    sensor_modality = sd_record['sensor_modality']
    channel = sd_record['channel']

    # Resolve and extract this reading's ego pose (vehicle position/orientation at capture time)
    ego_pose_token = sd_record['ego_pose_token']
    ego_pose_record = nusc.get('ego_pose', ego_pose_token)

    sensor_data = extract_key_to_column(sensor_data, ego_pose_record, EGO_POSE_COLUMNS)

    # Extract sample_data file metadata (timestamp, filepath/format, image dimensions)
    sensor_data = extract_key_to_column(sensor_data, sd_record, SENSOR_DATA_COLUMNS)

    # Resolve this sensor's calibration record (holds intrinsics + extrinsics, not yet extracted)
    calibrated_sensor_token = sd_record['calibrated_sensor_token']
    calibrated_sensor_record = nusc.get('calibrated_sensor', calibrated_sensor_token)


    # Extract intrinsic calibration matrix and extrinsic (camera-to-ego) rotation/translation
    sensor_data = extract_key_to_column(sensor_data, calibrated_sensor_record, SENSOR_CALIBRATION_COLUMNS)

    return sensor_modality.upper(), channel.upper(), sensor_data


def extract_annotation_from_sample_data(nusc: NuScenes, sample_ann_token: str) -> tuple[str, str, dict]:
    """Extract one 3D bounding-box annotation's fields into a column-mapped dict.

    Args:
        nusc: NuScenes devkit instance used to look up records.
        sample_ann_token: Token of the sample_annotation record to extract.

    Returns:
        dict: Annotation fields per SAMPLE_ANNOTATION_COLUMNS (instance/visibility/
        prev/next tokens, pose translation/rotation, size, category, lidar/radar
        point counts), plus a derived 'VISIBILITY_RANGE' (min%, max%) pair.
    """
    # Look up the sample_annotation record for this annotation
    sa_record = nusc.get('sample_annotation', sample_ann_token)

    ann_data = {}

    ann_data = extract_key_to_column(ann_data, sa_record, SAMPLE_ANNOTATION_COLUMNS)

    # Map the coarse visibility_token to its (min%, max%) visible-fraction range
    ann_data["VISIBILITY_RANGE"] = visibility_range(sa_record['visibility_token'])

    return ann_data


def extract_sample_data_from_sample(nusc: NuScenes, sample_token: str) -> dict:
    """Aggregate sensor data and annotations from a sample into one nested dict.

    For each sensor channel in the sample, extracts its sensor data (see
    extract_sensor_data_from_sample_data) and files it under its modality and
    channel name. Also extracts every annotation attached to the sample.

    Note: timestamp-skew tracking (max_camera_skew_ms / calculate_timestamp_skew_ms)
    is scaffolded below but not currently computed or included in the output.

    Args:
        nusc: NuScenes devkit instance used to look up records.
        sample_token: Token of the sample to aggregate.

    Returns:
        dict: Aggregate with keys 'SAMPLE_TOKEN', 'SCENE_TOKEN',
        'SAMPLE_TIMESTAMP', 'ANNOTATIONS' (annotation token -> annotation dict,
        see extract_annotation_from_sample_data), and one key per sensor
        modality encountered (e.g. 'CAMERA'), each mapping channel name to
        that channel's sensor_data dict.
    """
    # Sample record holds per-channel sample_data tokens plus the reference timestamp and annotations
    sample = nusc.get('sample', sample_token)
    sample_data_tokens = sample['data']
    sample_ts = sample['timestamp']
    sample_ann_tokens = sample['anns']

    # Initialize the sample aggregate dict
    sample_data = {"SAMPLE_TOKEN": sample_token, "SCENE_TOKEN": sample['scene_token'],
                   "SAMPLE_TIMESTAMP": sample_ts}

    # sample_data_tokens maps channel name -> sample_data token; loop over every channel/sensor
    for channel, data_token in sample_data_tokens.items():
        # Extract this channel's sensor data (ego pose, file metadata, calibration)
        sensor_modality, channel, sensor_data = extract_sensor_data_from_sample_data(nusc, data_token)

        # File the sensor data under its modality (e.g. CAMERA) and channel name,
        # logging the first time a new modality is encountered
        if sensor_modality in sample_data:
            sample_data[sensor_modality][channel] = sensor_data
        else:
            sample_data[sensor_modality] = { channel : sensor_data }

    sample_data["ANNOTATIONS"] = {}
    # Extract every annotation attached to this sample, keyed by annotation token
    for ann_token in sample_ann_tokens:
        annotation_data = extract_annotation_from_sample_data(nusc, ann_token)

    return sample_data

