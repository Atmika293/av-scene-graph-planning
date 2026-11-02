"""Flat table column mappings and pyarrow schemas for the Parquet output tables.

Two related but distinct things live here:

1. The *_COLUMNS dicts (SAMPLE_COLUMNS, SAMPLE_DATA_COLUMNS,
   SAMPLE_ANNOTATION_COLUMNS, CANVAS_BOXES_COLUMNS) map a flat output column
   name to where its value lives in an aggregate/sensor/annotation dict —
   either an aggregate key constant directly, or an (aggregate key, index)
   tuple for a single component of a vector/matrix field (e.g. one element of
   a quaternion or camera-intrinsic matrix). table.py's construct_table_row
   uses these to flatten a nested dict into one table row.

2. The *_schema pyarrow schemas describe those same flat tables' column names
   and types for table.py's write_batch_table_to_file to validate against and
   write out as Parquet.
"""

import pyarrow as pa
import pyarrow.parquet as pq

from av_data_pipeline.ingestion_keys import *


# Output column name -> aggregate key constant (or (key, index) for a single
# vector/matrix component), for the sample table.
SAMPLE_COLUMNS = {
    "sample_token": SAMPLE_INGESTION_KEYS['token'],
    "scene_token": SAMPLE_INGESTION_KEYS['scene_token'],
    "timestamp": SAMPLE_INGESTION_KEYS['timestamp']
}

# Same mapping for the sample_data table (one row per sensor reading).
SAMPLE_DATA_COLUMNS = {
    "sample_token": '',
    "sample_data_token": SAMPLE_DATA_INGESTION_KEYS['token'],
    "modality": SAMPLE_DATA_INGESTION_KEYS['modality'],
    "channel": SAMPLE_DATA_INGESTION_KEYS['channel'],
    "filepath": SAMPLE_DATA_INGESTION_KEYS['filepath'],
    "timestamp": SAMPLE_DATA_INGESTION_KEYS['timestamp'],
    "width": TRANSFORM_KEYS['canvas_width'],
    "height": TRANSFORM_KEYS['canvas_height'],
    "ego_tx": (TRANSFORM_KEYS['ego_pose_translation'], 0),
    "ego_ty": (TRANSFORM_KEYS['ego_pose_translation'], 1),
    "ego_tz": (TRANSFORM_KEYS['ego_pose_translation'], 2),
    "ego_qw": (TRANSFORM_KEYS['ego_pose_rotation'], 0),
    "ego_qx": (TRANSFORM_KEYS['ego_pose_rotation'], 1),
    "ego_qy": (TRANSFORM_KEYS['ego_pose_rotation'], 2),
    "ego_qz": (TRANSFORM_KEYS['ego_pose_rotation'], 3),
    "sensor_tx": (TRANSFORM_KEYS['sensor_translation'], 0),
    "sensor_ty": (TRANSFORM_KEYS['sensor_translation'], 1),
    "sensor_tz": (TRANSFORM_KEYS['sensor_translation'], 2),
    "sensor_qw": (TRANSFORM_KEYS['sensor_rotation'], 0),
    "sensor_qx": (TRANSFORM_KEYS['sensor_rotation'], 1),
    "sensor_qy": (TRANSFORM_KEYS['sensor_rotation'], 2),
    "sensor_qz": (TRANSFORM_KEYS['sensor_rotation'], 3),
    "cam_fx": (TRANSFORM_KEYS['intrinsic_matrix'], 0, 0),
    "cam_cx": (TRANSFORM_KEYS['intrinsic_matrix'], 0, 2),
    "cam_fy": (TRANSFORM_KEYS['intrinsic_matrix'], 1, 1),
    "cam_cy": (TRANSFORM_KEYS['intrinsic_matrix'], 1, 2),
}


# Same mapping for the sample_annotation table (one row per 3D annotation).
SAMPLE_ANNOTATION_COLUMNS = {
    "sample_token": '',
    "annotation_token": SAMPLE_ANNOTATION_INGESTION_KEYS['token'],
    "instance_token": SAMPLE_ANNOTATION_INGESTION_KEYS['instance_token'],
    "prev_annotation": SAMPLE_ANNOTATION_INGESTION_KEYS['previous_annotation'],
    "next_annotation": SAMPLE_ANNOTATION_INGESTION_KEYS['next_annotation'],
    "category_name": SAMPLE_ANNOTATION_INGESTION_KEYS['category'],
    "visibility_token": SAMPLE_ANNOTATION_INGESTION_KEYS['visibility_token'],
    "tx": (TRANSFORM_KEYS['annotation_translation'], 0),
    "ty": (TRANSFORM_KEYS['annotation_translation'], 1),
    "tz": (TRANSFORM_KEYS['annotation_translation'], 2),
    "qw": (TRANSFORM_KEYS['annotation_rotation'], 0),
    "qx": (TRANSFORM_KEYS['annotation_rotation'], 1),
    "qy": (TRANSFORM_KEYS['annotation_rotation'], 2),
    "qz": (TRANSFORM_KEYS['annotation_rotation'], 3),
    "wd": (TRANSFORM_KEYS['annotation_size'], 0),
    "ln": (TRANSFORM_KEYS['annotation_size'], 1),
    "ht": (TRANSFORM_KEYS['annotation_size'], 2),
    "num_lidar_pts": SAMPLE_ANNOTATION_INGESTION_KEYS['n_lidar'],
    "num_radar_pts": SAMPLE_ANNOTATION_INGESTION_KEYS['n_radar'],
}

# Same mapping for the canvas_bboxes table (one row per clipped 2D projection);
# built directly by table.build_canvas_bbox_row rather than construct_table_row,
# so every value here is blank.
CANVAS_BOXES_COLUMNS = {
    "annotation_token": '',
	"sample_data_token": '',
	"xmin": '',
	"ymin": '',
	"xmax": '',
	"ymax": ''
}

# pyarrow schema for the sample table.
sample_schema = pa.schema([
    ('sample_token', pa.string()),
    ('scene_token', pa.string()),
    ('timestamp', pa.int64()),
])

# pyarrow schema for the sample_data table.
sample_data_schema = pa.schema([
    ('sample_token', pa.string()),
    ('sample_data_token', pa.string()),
    ('modality', pa.dictionary(pa.int8(), pa.string())), # categorical
    ('channel', pa.dictionary(pa.int8(), pa.string())),   # categorical
    ('filepath', pa.string()),
    ('timestamp', pa.int64()),
    ('width', pa.int32()),
    ('height', pa.int32()),
    ('ego_tx', pa.float64()), ('ego_ty', pa.float64()), ('ego_tz', pa.float64()),
    ('ego_qw', pa.float64()), ('ego_qx', pa.float64()), ('ego_qy', pa.float64()), ('ego_qz', pa.float64()),
    ('sensor_tx', pa.float64()), ('sensor_ty', pa.float64()), ('sensor_tz', pa.float64()),
    ('sensor_qw', pa.float64()), ('sensor_qx', pa.float64()), ('sensor_qy', pa.float64()), ('sensor_qz', pa.float64()),
    ('cam_fx', pa.float64()), ('cam_fy', pa.float64()), ('cam_cx', pa.float64()), ('cam_cy', pa.float64()),
])

# pyarrow schema for the sample_annotation table.
sample_annotation_schema = pa.schema([
    ('annotation_token', pa.string()),
    ('sample_token', pa.string()),
    ('instance_token', pa.string()),
    ('prev_annotation', pa.string()), ('next_annotation', pa.string()),
    ('category_name', pa.dictionary(pa.int8(), pa.string())), # categorical
    ('visibility_token', pa.dictionary(pa.int8(), pa.string())),   # categorical
    ('tx', pa.float64()), ('ty', pa.float64()), ('tz', pa.float64()),
    ('qw', pa.float64()), ('qx', pa.float64()), ('qy', pa.float64()), ('qz', pa.float64()),
    ('wd', pa.float64()), ('ln', pa.float64()), ('ht', pa.float64()),
    ('num_lidar_pts', pa.int32()), ('num_radar_pts', pa.int32())
])

# pyarrow schema for the canvas_bboxes table.
canvas_bboxes_schema = pa.schema([
    ('annotation_token', pa.string()),
    ('sample_data_token', pa.string()),
    ('xmin', pa.float64()), ('ymin', pa.float64()), 
    ('xmax', pa.float64()), ('ymax', pa.float64()), 
])
