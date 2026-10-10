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
    "timestamp": SAMPLE_INGESTION_KEYS['sample_ts']
}

# Same mapping for the sample_data table (one row per sensor reading).
SAMPLE_DATA_COLUMNS = {
    "sample_token": '',
    "sample_data_token": SAMPLE_DATA_INGESTION_KEYS['token'],
    "modality": SAMPLE_DATA_INGESTION_KEYS['modality'],
    "channel": SAMPLE_DATA_INGESTION_KEYS['channel'],
    "filepath": 'SD_FILEPATH',
    "timestamp": 'SD_TIMESTAMP',
    "width": 'SENSOR_WD',
    "height": 'SENSOR_HT',
    "ego_x": ('EP_TRANSLATION', 0),
    "ego_y": ('EP_TRANSLATION', 1),
    "ego_z": ('EP_TRANSLATION', 2),
    "eqo_qw": ('EP_ROTATION', 0),
    "ego_qx": ('EP_ROTATION', 1),
    "ego_qy": ('EP_ROTATION', 2),
    "ego_qx": ('EP_ROTATION', 3),
    "sensor_x": ('SC_EXT_TRANSLATION', 0),
    "sensor_y": ('SC_EXT_TRANSLATION', 1),
    "sensor_z": ('SC_EXT_TRANSLATION', 2),
    "sensor_qw": ('SC_EXT_ROTATION', 0),
    "sensor_qx": ('SC_EXT_ROTATION', 1),
    "sensor_qy": ('SC_EXT_ROTATION', 2),
    "sensor_qz": ('SC_EXT_ROTATION', 3),
    "cam_fx": ("SC_INTRINSIC", 0, 0),
    "cam_cx": ("SC_INTRINSIC", 0, 2),
    "cam_fy": ("SC_INTRINSIC", 1, 1),
    "cam_cy": ("SC_INTRINSIC", 1, 2),
}


# Same mapping for the sample_annotation table (one row per 3D annotation).
SAMPLE_ANNOTATION_COLUMNS = {
    "sample_token": '',
    "annotation_token": SAMPLE_ANNOTATION_INGESTION_KEYS['token'],
    "instance_token": 'INSTANCE_TOKEN',
    "prev_annotation": 'PREV_ANNOTATION',
    "next_annotation": 'NEXT_ANNOTATION',
    "category_name": 'CATEGORY',
    "visibility_token": 'VISIBILITY_TOKEN',
    "tx": ('ANN_TRANSLATION', 0),
    "ty": ('ANN_TRANSLATION', 1),
    "tz": ('ANN_TRANSLATION', 2),
    "qw": ('ANN_ROTATION', 0),
    "qx": ('ANN_ROTATION', 1),
    "qy": ('ANN_ROTATION', 2),
    "qz": ('ANN_ROTATION', 3),
    "wd": ('SIZE', 0),
    "ln": ('SIZE', 1),
    "ht": ('SIZE', 2),
    "num_lidar_pts": 'N_LIDAR_PTS',
    "num_radar_pts": 'N_RADAR_PTS',
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
sample_schema = {[
    ('sample_token', pa.string()),
    ('scene_token', pa.string()),
    ('timestamp', pa.int64()),
]}

# pyarrow schema for the sample_data table.
sample_data_schema = pa.schema([
    ('sample_token', pa.string()),
    ('sample_data_token', pa.string()),
    ('modality', pa.dictionary(pa.int8(), pa.string())), # categorical
    ('channel', pa.dictionary(pa.int8(), pa.string())),   # categorical
    ('filename', pa.string()),
    ('timestamp', pa.int64()),
    ('skew_ms', pa.int64()),
    ('width', pa.int32()),
    ('height', pa.int32()),
    ('ego_x', pa.float64()), ('ego_y', pa.float64()), ('ego_z', pa.float64()),
    ('ego_qw', pa.float64()), ('ego_qx', pa.float64()), ('ego_qy', pa.float64()), ('ego_qz', pa.float64()),
    ('sensor_tx', pa.float64()), ('sensor_ty', pa.float64()), ('sensor_tz', pa.float64()),
    ('sensor_qw', pa.float64()), ('sensor_qx', pa.float64()), ('sensor_qy', pa.float64()), ('cal_qz', pa.float64()),
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
canvas_bboxes_schema = {[
    ('annotation_token', pa.string()),
    ('sample_data_token', pa.string()),
    ('xmin', pa.float64()), ('ymin', pa.float64()), 
    ('xmax', pa.float64()), ('ymax', pa.float64()), 
]}
