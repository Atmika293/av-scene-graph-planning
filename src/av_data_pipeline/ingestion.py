"""Ingestion helpers for pulling nuScenes records into flat, column-mapped dicts.

Walks scenes/samples via the NuScenes devkit and extracts sensor data, ego
poses, calibration, and annotations into the aggregate key names defined in
columns.py, ready for downstream transform/projection code.
"""

from nuscenes.nuscenes import NuScenes
from av_data_pipeline.utils import visibility_range, extract_key_values
from av_data_pipeline.nuscenes import *
from av_data_pipeline.ingestion_keys import *



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
    scene = nusc.get(SCENE_TABLE, scene_token)

    # Extract the first and last sample token for the scene
    sample_token = scene[FIRST_SAMPLE_TOKEN]
    last_sample_token = scene[LAST_SAMPLE_TOKEN]

    # Loop over the sample tokens belonging to a scene by using the
    # 'next' attribute in the sample dict
    sample_tokens = [sample_token]
    while sample_token != last_sample_token:
        sample = nusc.get(SAMPLE_TABLE, sample_token)
        next_sample_token = sample[NEXT_SAMPLE_TOKEN]
        sample_tokens.append(next_sample_token)
        sample_token = next_sample_token

    sample_tokens.append(last_sample_token)

    return sample_tokens


def extract_sensor_data_from_sample_data(nusc: NuScenes, sample_data_token: str) -> dict:
    """Extract sensor readings, ego pose, and calibration for one sample_data record.

    Returns a (sensor_modality, channel, sensor_data) tuple, where sensor_data
    contains the ego pose, file metadata, and intrinsic/extrinsic calibration.

    Args:
        nusc: NuScenes devkit instance used to look up records.
        sample_data_token: Token of the sample_data record to extract.

    Returns:
        dict: A dict
        with keys 'ego_pose', 'timestamp', 'filepath', 'fileformat',
        'image_height', 'image_width', 'sensor_intrinsic', and
        'sensor_extrinsic'.
    """
    # Look up the sample_data record for this sensor reading
    sd_record = nusc.get(SAMPLE_DATA_TABLE, sample_data_token)


    sensor_data = {SAMPLE_DATA_INGESTION_KEYS['token']: sample_data_token}

    # Resolve and extract this reading's ego pose (vehicle position/orientation at capture time)
    ego_pose_token = sd_record[EGO_POSE_TOKEN]
    ego_pose_record = nusc.get(EGO_POSE_TABLE, ego_pose_token)

    sensor_data = extract_key_values(sensor_data, ego_pose_record, EGO_POSE_KEYS_MAP)

    # Extract sample_data file metadata (timestamp, filepath/format, image dimensions)
    sensor_data = extract_key_values(sensor_data, sd_record, SENSOR_DATA_KEYS_MAP)

    # Resolve this sensor's calibration record (holds intrinsics + extrinsics, not yet extracted)
    calibrated_sensor_token = sd_record[CALIBRATED_SENSOR_TOKEN]
    calibrated_sensor_record = nusc.get(CALIBRATED_SENSOR_TABLE, calibrated_sensor_token)


    # Extract intrinsic calibration matrix and extrinsic (camera-to-ego) rotation/translation
    sensor_data = extract_key_values(sensor_data, calibrated_sensor_record, SENSOR_CALIBRATION_KEYS_MAP)

    return sensor_data


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
    sa_record = nusc.get(SAMPLE_ANNOTATION_TABLE, sample_ann_token)

    ann_data = {SAMPLE_ANNOTATION_INGESTION_KEYS['token']: sample_ann_token}

    ann_data = extract_key_values(ann_data, sa_record, SAMPLE_ANNOTATION_KEYS_MAP)

    # Map the coarse visibility_token to its (min%, max%) visible-fraction range
    ann_data[SAMPLE_ANNOTATION_INGESTION_KEYS['visibility_range']] = visibility_range(sa_record['visibility_token'])

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
    sample_data = {
                    SAMPLE_INGESTION_KEYS['token']: sample_token, 
                    SAMPLE_INGESTION_KEYS['scene_token']: sample['scene_token'],
                    SAMPLE_INGESTION_KEYS['sample_ts']: sample_ts
                   }

    # sample_data_tokens maps channel name -> sample_data token; loop over every channel/sensor
    for channel, data_token in sample_data_tokens.items():
        # Extract this channel's sensor data (ego pose, file metadata, calibration)
        sensor_data = extract_sensor_data_from_sample_data(nusc, data_token)

        # Modality/channel now come from the returned column-mapped dict itself
        # (SENSOR_DATA_KEYS), not a separate return value. Uppercased here to
        # match KNOWN_SENSOR_MODALITIES/KNOWN_SENSOR_CHANNELS and the "CAMERA"/
        # "LIDAR"/"RADAR" keys this function files sensor_data under below.
        sensor_modality = sensor_data[SAMPLE_DATA_INGESTION_KEYS['modality']].upper()
        channel = sensor_data[SAMPLE_DATA_INGESTION_KEYS['channel']].upper()

        # Guard against sensor_modality/channel values not in columns.py's known
        # sets, in case nuScenes ever reports one this ingestion code doesn't expect.
        if sensor_modality not in KNOWN_SENSOR_MODALITIES:
            break

        if channel not in KNOWN_SENSOR_CHANNELS[sensor_modality]:
            break

        # File the sensor data under its modality (e.g. CAMERA) and channel name,
        # logging the first time a new modality is encountered
        if sensor_modality in sample_data:
            sample_data[sensor_modality][channel] = sensor_data
        else:
            sample_data[sensor_modality] = { channel : sensor_data }

    sample_data[SAMPLE_INGESTION_KEYS['annotations']] = {}
    # Extract every annotation attached to this sample, keyed by annotation token
    for ann_token in sample_ann_tokens:
        annotation_data = extract_annotation_from_sample_data(nusc, ann_token)
        sample_data[SAMPLE_INGESTION_KEYS['annotations']][ann_token] = annotation_data

    return sample_data


def build_scene_data(nusc: NuScenes, scene_token: str) -> list[dict]:
    """Aggregate sensor data and annotations for every sample in a scene.

    Args:
        nusc: NuScenes devkit instance used to look up records.
        scene_token: Token of the scene to aggregate.

    Returns:
        One aggregate dict per sample in the scene, in temporal order (see
        extract_sample_data_from_sample), as produced by walking the scene's
        sample tokens via get_sample_tokens_from_scene.
    """
    scene_data = []
    for sample_token in get_sample_tokens_from_scene(nusc, scene_token):
        scene_data.append(extract_sample_data_from_sample(nusc, sample_token))

    return scene_data
