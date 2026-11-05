"""Key-name vocabulary shared across the ingestion pipeline.

Two layers of mapping live here:

1. The *_INGESTION_KEYS dicts (SAMPLE_INGESTION_KEYS, SAMPLE_DATA_INGESTION_KEYS,
   SAMPLE_ANNOTATION_INGESTION_KEYS) map a generic field name (e.g. "token") to
   the aggregate key constant that holds it for a given record type (e.g.
   "SAMPLE_TOKEN" vs "ANNOTATION_TOKEN"). This lets downstream code (table.py,
   schema.py) reference "the token field" generically instead of hardcoding a
   different constant per record type.

2. The *_KEY_MAPS dicts (EGO_POSE_KEY_MAPS, SENSOR_DATA_KEY_MAPS, SENSOR_CALIBRATION_KEY_MAPS,
   SAMPLE_ANNOTATION_KEY_MAPS) map those aggregate key constants to the raw
   nuScenes record field names. These are consumed by
   utils.extract_key_values to build the column-mapped dicts ingestion.py
   produces.

KNOWN_SENSOR_MODALITIES/KNOWN_SENSOR_CHANNELS enumerate the valid values
ingestion.py checks extracted sensor readings against.
"""

# Aggregate key constant that holds each record type's own token/timestamp/etc,
# keyed by a generic field name so callers don't need to hardcode a different
# constant per record type.
SAMPLE_INGESTION_KEYS = {
    "token": "SAMPLE_TOKEN",
    "scene_token": "SCENE_TOKEN",
    "timestamp": "SAMPLE_TIMESTAMP",
    "annotations": "ANNOTATIONS"
}

SAMPLE_DATA_INGESTION_KEYS = {
    "token": "SAMPLE_DATA_TOKEN",
    "modality": "SENSOR_MODALITY",
    "channel": "SENSOR_CHANNEL",
    "timestamp": "SD_TIMESTAMP",
    "filepath": "SD_FILEPATH",
    "file_format": "SD_FILE_FORMAT"
}

SAMPLE_ANNOTATION_INGESTION_KEYS = {
    "token": "ANNOTATION_TOKEN",
    "visibility_range": "VISIBILITY_RANGE",
    "n_lidar": "N_LIDAR_PTS",
    "n_radar": "N_RADAR_PTS",
    "instance_token": "INSTANCE_TOKEN",
    "visibility_token": "VISIBILITY_TOKEN",
    "next_annotation": "NEXT_ANNOTATION",
    "previous_annotation": "PREVIOUS_ANNOTATION",
    "category": "CATEGORY"

}

TRANSFORM_KEYS = {
    "ego_pose_translation": "EP_TRANSLATION",
    "ego_pose_rotation": "EP_ROTATION",
    "sensor_translation": "SC_EXT_TRANSLATION",
    "sensor_rotation": "SC_EXT_ROTATION",
    "canvas_width": "SENSOR_WD",
    "canvas_height": "SENSOR_HT",
    "intrinsic_matrix": "SC_INTRINSIC",
    "annotation_translation": "ANN_TRANSLATION",
    "annotation_rotation": "ANN_ROTATION",
    "annotation_size": "SIZE"
}

SENSOR_MODALITY_KEYS = {
    "camera": "CAMERA",
    "radar": "RADAR",
    "lidar": "LIDAR"
}

# Aggregate key constant -> raw nuScenes ego_pose field name.
EGO_POSE_KEYS_MAP = {
    TRANSFORM_KEYS['ego_pose_translation']: 'translation',
    TRANSFORM_KEYS['ego_pose_rotation']: 'rotation',
    "EGO_TIMESTAMP": 'timestamp'
}


# Aggregate key constant -> raw nuScenes sample_data field name. The token
# itself has no source field (it's the sample_data record's own primary key,
# already known to the caller), hence the empty string.
SENSOR_DATA_KEYS_MAP = {
	SAMPLE_DATA_INGESTION_KEYS['token']: '',
    SAMPLE_DATA_INGESTION_KEYS['timestamp']: 'timestamp',
    SAMPLE_DATA_INGESTION_KEYS['filepath']: 'filename',
    SAMPLE_DATA_INGESTION_KEYS['file_format']: 'fileformat',
    TRANSFORM_KEYS['canvas_height']: 'height',
    TRANSFORM_KEYS['canvas_width']: 'width',
    SAMPLE_DATA_INGESTION_KEYS['modality']: 'sensor_modality',
    SAMPLE_DATA_INGESTION_KEYS['channel']: 'channel'
}


# Aggregate key constant -> raw nuScenes calibrated_sensor field name.
SENSOR_CALIBRATION_KEYS_MAP = {
    TRANSFORM_KEYS['intrinsic_matrix']: 'camera_intrinsic',
    TRANSFORM_KEYS['sensor_translation']: 'translation',
    TRANSFORM_KEYS['sensor_rotation']: 'rotation',
}


# Aggregate key constant -> raw nuScenes sample_annotation field name.
SAMPLE_ANNOTATION_KEYS_MAP = {
    SAMPLE_ANNOTATION_INGESTION_KEYS['token']: '',
    SAMPLE_ANNOTATION_INGESTION_KEYS['instance_token']: 'instance_token',
    SAMPLE_ANNOTATION_INGESTION_KEYS['visibility_token']: 'visibility_token',
    SAMPLE_ANNOTATION_INGESTION_KEYS['next_annotation']: 'next',
    SAMPLE_ANNOTATION_INGESTION_KEYS['previous_annotation']: 'prev',
    TRANSFORM_KEYS['annotation_translation']: 'translation',
    TRANSFORM_KEYS['annotation_rotation']: 'rotation',
    TRANSFORM_KEYS['annotation_size']: 'size',
    SAMPLE_ANNOTATION_INGESTION_KEYS['category']: 'category_name',
    SAMPLE_ANNOTATION_INGESTION_KEYS['n_lidar']: 'num_lidar_pts',
    SAMPLE_ANNOTATION_INGESTION_KEYS['n_radar']: 'num_radar_pts',
    SAMPLE_ANNOTATION_INGESTION_KEYS['visibility_range']: ''
}


# Modalities ingestion.py accepts; anything else trips its validation guard.
KNOWN_SENSOR_MODALITIES = {
    "CAMERA", "RADAR", "LIDAR"
}

# Valid channel names per modality, for the same validation guard.
KNOWN_SENSOR_CHANNELS = {
    "RADAR":   [
        "RADAR_FRONT", 
        "RADAR_FRONT_LEFT",
        "RADAR_FRONT_RIGHT", 
        "RADAR_BACK_LEFT",
        "RADAR_BACK_RIGHT"
        ], 

    "LIDAR": ["LIDAR_TOP"],
    
    "CAMERA": [
        "CAM_FRONT", 
        "CAM_FRONT_RIGHT",
        "CAM_BACK_RIGHT", 
        "CAM_BACK",
        "CAM_BACK_LEFT", 
        "CAM_FRONT_LEFT"
        ]
}

