from nuscenes.nuscenes import NuScenes
from utils import calculate_timestamp_skew_ms

from pprint import pprint

NUSCENES_DIR = "data/nuscenes"


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
    # get the scene dict using the scene_token
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
    # get sample data record using sample data token
    sd_record = nusc.get('sample_data', sample_data_token)

    sensor_data = {}

    # Get sensor modality and channel
    sensor_modality = sd_record['sensor_modality']
    channel = sd_record['channel']

    print(f"Sample data token: {sample_data_token}")
    print(f"Sensor modality: {sensor_modality}")
    print(f"Channel: {channel}")

    # Get ego pose token
    ego_pose_token = sd_record['ego_pose_token']
    ego_pose = nusc.get('ego_pose', ego_pose_token)
    sensor_data["ego_pose"] = {"translation": ego_pose['translation'], "rotation": ego_pose['rotation'], "timestamp": ego_pose['timestamp']}

    # Extract sample metadata
    sensor_data["timestamp"] = sd_record['timestamp']
    sensor_data["filepath"] = sd_record['filename']
    sensor_data["fileformat"] = sd_record['fileformat']
    sensor_data["image_height"] = sd_record['height']
    sensor_data["image_width"] = sd_record['width']

    # Retrieve calibration matrix
    calibrated_sensor_token = sd_record['calibrated_sensor_token']
    calibrated_sensor_record = nusc.get('calibrated_sensor', calibrated_sensor_token)
    sensor_data["sensor_intrinsic"] = calibrated_sensor_record['camera_intrinsic']

    # Retrive extrinsic parameters
    sensor_data["sensor_extrinsic"] = {"translation": calibrated_sensor_record['translation'], "rotation": calibrated_sensor_record['rotation']}

    return sensor_modality, channel, sensor_data


def extract_sample_data_from_sample(nusc: NuScenes, sample_token: str) -> dict:
    """Aggregate sensor data from every channel of a sample into one dict.

    For each sensor channel in the sample, extracts its sensor data and
    computes its timestamp skew relative to the sample's reference timestamp,
    tracking the maximum camera skew across channels.

    Args:
        nusc: NuScenes devkit instance used to look up records.
        sample_token: Token of the sample to aggregate.

    Returns:
        dict: Aggregate with keys 'sample_token', 'scene_token',
        'reference_timestamp', 'max_camera_skew_ms', and one key per
        sensor modality encountered, each mapping channel name to that
        channel's sensor_data dict (see
        extract_sensor_data_from_sample_data) with an added 'ts_skew_ms'.
    """
    # get sample data tokens per channel
    sample = nusc.get('sample', sample_token)
    sample_data_tokens = sample['data']
    sample_ts = sample['timestamp']

    # Initialize the sample aggregate dict
    sample_data = {"sample_token": sample_token, "scene_token": sample['scene_token'],
                   "reference_timestamp": sample_ts}
    # Variable to track the maximum skew of a smaple data from the reference timestamp
    max_camera_skew_ms = 0

    for channel, data_token in sample_data_tokens.items():
        # extract sensor data
        sensor_modality, channel, sensor_data = extract_sensor_data_from_sample_data(nusc, data_token)

        # get timestamp skew
        sample_data_ts = sensor_data["timestamp"]

        # Calculate the skew (absolute difference) between the sample timestamp and the sample data timestamp
        # The UNIX timestamps are recorded in microseconds and the skew is calculated in milliseconds
        timestamp_skew_ms = calculate_timestamp_skew_ms(sample_ts, sample_data_ts)
        sensor_data["ts_skew_ms"] = timestamp_skew_ms
        max_camera_skew_ms = max(max_camera_skew_ms, timestamp_skew_ms)

        # Add sensor data to sample aggregate dict
        # Keyed by sensor modality and channel
        if sensor_modality in sample_data:
            sample_data[sensor_modality][channel] = sensor_data
        else:
            sample_data[sensor_modality] = { channel : sensor_data } 
            print(f"UNIQUE Sensor modality: {sensor_modality}, Channel: {channel}")

    # Add max skew to the aggregate dict
    sample_data["max_camera_skew_ms"] = max_camera_skew_ms

    return sample_data


if __name__ == "__main__":
    nusc = NuScenes(version='v1.0-mini', dataroot=NUSCENES_DIR, verbose=True)
    # scene_token = 'cc8c0bf57f984915a77078b10eb33198'
    # st1 = nusc.field2token('sample', 'scene_token', scene_token)
    # st2 = get_sample_tokens_from_scene(nusc, scene_token)
    # print("Sample tokens from get_sample_tokens_from_scene:" + str(st2))
    # print("Sample tokens from field2token:" + str(st1))
    # print("Are they the same? " + str(set(st1) == set(st2)))

    my_scene = nusc.scene[0]
    print("---SCENE---")
    print(my_scene)
    print("---SCENE---")
    print("---SAMPLE---")
    first_sample_token = my_scene['first_sample_token']
    my_sample = nusc.get('sample', first_sample_token)
    print("Ground Truth:")
    pprint(my_sample)
    print("Extracted:")
    pprint(extract_sample_data_from_sample(nusc, first_sample_token))
    print("---SAMPLE---")
