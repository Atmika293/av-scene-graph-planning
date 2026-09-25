# Mapping from aggregate key names to NuScenes tabe columns/keys

EGO_POSE_COLUMNS = {
    "EP_TRANSLATION": 'translation',
    "EP_ROTATION": 'rotation',
    "EGO_TIMESTAMP": 'timestamp'
}


SENSOR_DATA_COLUMNS = {
    "SD_TIMESTAMP": 'timestamp',
    "SD_FILEPATH": 'filename',
    "SD_FILE_FORMAT": 'fileformat',
    "SENSOR_HT": 'height',
    "SENSOR_WD": 'width'
}


SENSOR_CALIBRATION_COLUMNS = {
    "SC_INTRINSIC": 'camera_intrinsic',
    "SC_EXT_TRANSLATION": 'translation',
    "SC_EXT_ROTATION": 'rotation',
}


SAMPLE_ANNOTATION_COLUMNS = {
    "INSTANCE_TOKEN": 'instance_token',
    "VISIBILITY_TOKEN": 'visibility_token',
    "NEXT_ANNOTATION": 'next',
    "PREVIOUS_ANNOTATION": 'prev',
    "ANN_TRANSLATION": 'translation',
    "ANN_ROTATION": 'rotation',
    "SIZE": 'size',
    "CATEGORY": 'category_name',
    "N_LIDAR_PTS": 'num_lidar_pts',
    "N_RADAR_PTS": 'num_radar_pts',
    "VISIBILITY_RANGE": ''
}


SAMPLE_COLUMNS = {
    "SAMPLE_TOKEN", "SCENE_TOKEN", "SAMPLE_TIMESTAMP", "ANNOTATIONS"
}


SENSOR_TYPE = {
    "CAMERA", "RADAR", "LIDAR"
}


