import uuid

import ray
from itertools import batched
from nuscenes.nuscenes import NuScenes

from av_data_pipeline.ingestion import build_scene_data
from av_data_pipeline.schema import sample_schema, sample_data_schema, sample_annotation_schema
from av_data_pipeline.table import *
from av_data_pipeline.transform import project_scene_annotations_to_canvas

def generate_batch_id() -> str:
    """Generate an 8-character random identifier for one batch's output files."""
    return uuid.uuid4().hex[:8]


def build_manifest_for_batch(nusc, scene_batch):
    manifest = {}
    for scene_token in scene_batch:
        manifest[scene_token] = build_scene_data(nusc, scene_token)

    return manifest


@ray.remote
def write_samples_table(batch_id: str, batch: dict[str, list], output_root: str):
    # build rows for samples / sample_data / annotations, write 3 parquet files
    sample_rows = []
    for _, scene_data in batch.items():
        sample_rows.extend(extract_sample_rows(scene_data))

    write_batch_table_to_file(batch_id, table_name=SAMPLES_TABLE_NAME, 
                              table_rows=sample_rows,
                              table_schema=sample_schema,
                              output_root=output_root)


@ray.remote
def write_sample_data_table(batch_id: str, batch: dict[str, list], output_root: str):
    # build rows for samples / sample_data / annotations, write 3 parquet files
    sample_data_rows = []
    for _, scene_data in batch.items():
        sample_data_rows.extend(extract_sample_data_rows(scene_data))

    write_batch_table_to_file(batch_id, table_name=SAMPLE_DATA_TABLE_NAME, 
                              table_rows=sample_data_rows,
                              table_schema=sample_data_schema,
                              output_root=output_root)


@ray.remote
def write_sample_annotations_table(batch_id: str, batch: dict[str, list], output_root: str):
    # build rows for samples / sample_data / annotations, write 3 parquet files
    sample_annotations_rows = []
    for _, scene_data in batch.items():
        sample_annotations_rows.extend(extract_sample_annotations_rows(scene_data))

    write_batch_table_to_file(batch_id, table_name=SAMPLE_ANNOTATIONS_TABLE_NAME, 
                              table_rows=sample_annotations_rows,
                              table_schema=sample_annotation_schema,
                              output_root=output_root)
        

@ray.remote
def write_bboxes(batch_id: str, batch: dict[str, list], output_root: str):
    # projection + clip per (annotation, camera), write boxes_2d parquet file
    canvas_bboxes_rows = []
    for _, scene_data in batch.items():
        canvas_projections = project_scene_annotations_to_canvas(scene_data)
        canvas_bboxes_rows.extend(collate_canvas_bboxes(canvas_projections))

    write_batch_table_to_file(batch_id, table_name=CANVAS_BBOXES_TABLE_NAME, 
                              table_rows=canvas_bboxes_rows,
                              table_schema=canvas_bboxes_schema,
                              output_root=output_root)
    

MAX_IN_FLIGHT = 8
all_refs, pending = [], []

for scene_batch in batched(nusc.scene, 5):
    batch_ref = ray.put(build_manifest_for_batch(nusc, scene_batch))  # once per batch

    refs = [
        write_core_tables.remote(batch_ref, batch_id, output_root),
        write_bboxes.remote(batch_ref, batch_id, output_root),
    ]
    all_refs += refs
    pending += refs

    while len(pending) >= MAX_IN_FLIGHT:            # backpressure
        _, pending = ray.wait(pending, num_returns=1)

ray.get(all_refs)  # surfaces any task exception; ray.wait alone won't raise

