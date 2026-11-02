"""Flatten aggregate per-sample dicts into per-table row dicts for Parquet export.

ingestion.py produces one nested aggregate dict per sample (sensor data keyed
by modality/channel, annotations keyed by token). This module converts that
nested shape into the flat row-per-record shape of the sample/sample_data/
sample_annotation/canvas_bbox tables (via schema.py's *_COLUMNS mappings and
utils.construct_table_row), and batches rows out to Parquet files matching
schema.py's pyarrow schemas.
"""
import os
import time

from av_data_pipeline.schema import *
from av_data_pipeline.utils import *
from av_data_pipeline.ingestion_keys import (KNOWN_SENSOR_MODALITIES, KNOWN_SENSOR_CHANNELS,
                                             SAMPLE_INGESTION_KEYS)


SAMPLES_TABLE_NAME = 'samples'
SAMPLE_DATA_TABLE_NAME = 'sample_data'
SAMPLE_ANNOTATIONS_TABLE_NAME = 'annotations'
CANVAS_BBOXES_TABLE_NAME = 'canvas_bboxes'


def build_sample_row(sample_data: dict) -> dict:
    """Build one sample-table row from a sample's aggregate dict.

    Args:
        sample_data: Aggregate dict for one sample, as produced by
            ingestion.extract_sample_data_from_sample.

    Returns:
        A flat row dict per schema.SAMPLE_COLUMNS.
    """
    return construct_table_row(sample_data, SAMPLE_COLUMNS)

def build_sample_data_row(sample_token: str, sensor_data: dict) -> dict:
    """Build one sample_data-table row for a single sensor reading.

    Args:
        sample_token: Token of the sample this sensor reading belongs to.
        sensor_data: One sensor's column-mapped dict (see
            ingestion.extract_sensor_data_from_sample_data).

    Returns:
        A flat row dict per schema.SAMPLE_DATA_COLUMNS, with 'sample_token' set.
    """
    row = construct_table_row(sensor_data, SAMPLE_DATA_COLUMNS)
    row['sample_token'] = sample_token

    return row

def build_sample_annotation_row(sample_token: str, annotation_data: dict) -> dict:
    """Build one sample_annotation-table row for a single 3D annotation.

    Args:
        sample_token: Token of the sample this annotation belongs to.
        annotation_data: One annotation's column-mapped dict (see
            ingestion.extract_annotation_from_sample_data).

    Returns:
        A flat row dict per schema.SAMPLE_DATA_COLUMNS, with 'sample_token' set.
    """
    row = construct_table_row(annotation_data, SAMPLE_ANNOTATION_COLUMNS)
    row['sample_token'] = sample_token

    return row

def build_canvas_bbox_row(annotation_token: str, sample_data_token: str, bbox: list[float]) -> dict:
    """Build one canvas_bboxes-table row from a clipped 2D bounding box.

    Args:
        annotation_token: Token of the annotation this box belongs to.
        sample_data_token: Token of the camera sample_data this box was
            projected onto.
        bbox: Clipped box as [xmin, ymin, xmax, ymax].

    Returns:
        A flat row dict per schema.CANVAS_BOXES_COLUMNS.
    """
    return {
        "annotation_token": annotation_token,
        "sample_data_token": sample_data_token,
        "xmin": bbox[0],
        "ymin": bbox[1],
        "xmax": bbox[2],
        "ymax": bbox[3]
    }

def write_batch_table_to_file(batch_id: str, table_name: str, table_rows: list[dict],
                        table_schema: pa.schema, output_root: str) -> float:
    """Write one batch of table rows to a Parquet file.

    Creates the <output_root>/<table_name> output folder first if it doesn't
    already exist.

    Args:
        batch_id: Identifier for this batch, used in the output filename.
        table_name: Table this batch belongs to (e.g. 'sample'); used as the
            output subdirectory name.
        table_rows: Row dicts to write, matching table_schema's columns.
        table_schema: pyarrow schema the rows are validated/written against.
        output_root: Root directory rows are written under, as
            <output_root>/<table_name>/batch_<batch_id>.parquet.

    Returns:
        Seconds spent inside pq.write_table -- the benchmark harness' disk-write
        component of a task's total duration.
    """
    os.makedirs(f'{output_root}/{table_name}', exist_ok=True)
    table = pa.Table.from_pylist(table_rows, table_schema)
    output_path = f'{output_root}/{table_name}/batch_{batch_id}.parquet'

    write_start = time.perf_counter()
    pq.write_table(table, output_path)
    return time.perf_counter() - write_start


def extract_sample_rows(scene_data: list[dict]) -> list[dict]:
    """Build sample-table rows for every sample in a scene.

    Args:
        scene_data: One aggregate dict per sample in the scene.

    Returns:
        One sample-table row per sample.
    """
    rows = []

    for sample_data in scene_data:
        rows.append(build_sample_row(sample_data))

    return rows


def extract_sample_data_rows(scene_data: list[dict]) -> list[dict]:
    """Build sample_data-table rows for every sensor reading in a scene.

    Iterates every known modality/channel combination for each sample, so
    only sensors present under sample_data[modality][channel] are picked up.

    Args:
        scene_data: One aggregate dict per sample in the scene.

    Returns:
        One sample_data-table row per sensor reading across all samples.
    """
    rows = []

    for sample_data in scene_data:
        sample_token = sample_data[SAMPLE_INGESTION_KEYS['token']]

        for modality in KNOWN_SENSOR_MODALITIES:
            for channel in KNOWN_SENSOR_CHANNELS[modality]:
                rows.append(build_sample_data_row(sample_token, sample_data[modality][channel]))

    return rows


def extract_sample_annotations_rows(scene_data: list[dict]) -> list[dict]:
    """Build sample_annotation-table rows for every annotation in a scene.

    Args:
        scene_data: One aggregate dict per sample in the scene.

    Returns:
        One sample_annotation-table row per annotation across all samples.
    """
    rows = []

    for sample_data in scene_data:
        sample_token = sample_data[SAMPLE_INGESTION_KEYS['token']]

        for _, annotation_data in sample_data[SAMPLE_INGESTION_KEYS['annotations']].items():
            rows.append(build_sample_annotation_row(sample_token, annotation_data))

    return rows

    
def collate_canvas_bboxes(canvas_projections: list[dict]) -> list[dict]:
    rows = []

    for canvas_bboxes in canvas_projections:
        for annotation_token in canvas_bboxes:
            for sample_data_token in canvas_bboxes[annotation_token]:
                rows.append(build_canvas_bbox_row(annotation_token,
                                                  sample_data_token,
                                                  canvas_bboxes[annotation_token][sample_data_token]))

    return rows



