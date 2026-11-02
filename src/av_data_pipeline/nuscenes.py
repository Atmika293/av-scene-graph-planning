"""Raw nuScenes devkit table and field name constants.

Names the tables passed to nusc.get(...)/nusc.field2token(...) and the raw
field names read off the records they return, so ingestion code can reference
them by constant instead of repeating the nuScenes schema's magic strings.
"""

# Table names, as passed to nusc.get(table_name, token).
SCENE_TABLE = 'scene'
SAMPLE_TABLE = 'sample'
SAMPLE_DATA_TABLE = 'sample_data'
SAMPLE_ANNOTATION_TABLE = 'sample_annotation'
EGO_POSE_TABLE = 'ego_pose'
CALIBRATED_SENSOR_TABLE = 'calibrated_sensor'

# Scene Table: fields used to walk a scene's linked list of samples.
SCENE_TOKEN = 'token'
FIRST_SAMPLE_TOKEN = 'first_sample_token'
LAST_SAMPLE_TOKEN = 'last_sample_token'
NEXT_SAMPLE_TOKEN = 'next'

# Sample Table: per-channel sample_data tokens, reference timestamp, and
# the sample's attached annotation tokens.
SAMPLE_DATA_TOKENS = 'data'
SAMPLE_TIMESTAMP = 'timestamp'
SAMPLE_ANNOTATION_TOKENS = 'anns'

# Sample Data Table: foreign keys into the ego_pose and calibrated_sensor tables.
EGO_POSE_TOKEN = 'ego_pose_token'
CALIBRATED_SENSOR_TOKEN = 'calibrated_sensor_token'



