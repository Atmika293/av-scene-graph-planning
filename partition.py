"""Ray-parallel driver exporting nuScenes scenes to batched Parquet tables.

Batches nusc.scene (BATCH_SIZE scenes per batch), builds each batch's
aggregate scene data via av_data_pipeline.ingestion, then farms out two
independent writing tasks per batch to Ray workers: the core
sample/sample_data/sample_annotation tables (write_core_tables) and the
projected 2D canvas bboxes (write_bboxes). TASK_LAYOUT=split instead gives
each core table its own task (write_core_table), for 4 tasks per batch; see
benchmark_task_layout.py, which compares the two. Output lands under
OUTPUT_ROOT/<table_name>/batch_<batch_id>.parquet.

Run with: python partition.py (from the repo root, so config.toml and the
relative NUSCENES_DATAROOT/OUTPUT_ROOT defaults resolve correctly).

This driver's execution-strategy knobs -- BATCH_SIZE, MAX_IN_FLIGHT,
NUM_CPUS, TASK_LAYOUT -- live in config.toml's [partition] table alongside every other
setting (see av_data_pipeline.config), overridable the same way:
    export BATCH_SIZE=1        # scenes aggregated into one manifest/batch
    export MAX_IN_FLIGHT=32    # cap on simultaneously in-flight Ray tasks
    export NUM_CPUS=4          # passed to ray.init(); unset = every CPU available (machine.py)
    export TASK_LAYOUT=split   # 4 tasks per batch instead of 2 (consolidated)
"""

import json
import os
import sys
import time
import uuid

import ray
from itertools import batched
from nuscenes.nuscenes import NuScenes

from av_data_pipeline.config import (
    NUSCENES_VERSION, NUSCENES_DATAROOT, OUTPUT_ROOT,
    BATCH_SIZE, MAX_IN_FLIGHT, NUM_CPUS, TASK_LAYOUT,
)
from av_data_pipeline.ingestion import build_scene_data_from_scene_dict
from av_data_pipeline.machine import available_cpus
from av_data_pipeline.nuscenes import SCENE_TOKEN
from av_data_pipeline.schema import sample_schema, sample_data_schema, sample_annotation_schema
from av_data_pipeline.table import *
from av_data_pipeline.transform import project_scene_annotations_to_canvas

def generate_batch_id() -> str:
    """Generate an 8-character random identifier for one batch's output files."""
    return uuid.uuid4().hex[:8]


def build_manifest_for_batch(nusc: NuScenes, scene_batch: tuple[dict, ...]) -> dict[str, list[dict]]:
    """Aggregate every scene in a batch into one manifest, keyed by scene token.

    Runs on the driver (not a Ray task) so the resulting manifest can be
    ray.put() once per batch and shared read-only by every remote task for
    that batch, instead of each task re-aggregating the same scenes itself.

    Args:
        nusc: NuScenes devkit instance used to look up records.
        scene_batch: One batch's worth of scene records (e.g. one group from
            itertools.batched(nusc.scene, 5)).

    Returns:
        Scene token -> that scene's aggregate per-sample data (see
        ingestion.build_scene_data_from_scene_dict).
    """
    manifest = {}
    for scene_dict in scene_batch:
        scene_token = scene_dict[SCENE_TOKEN]
        manifest[scene_token] = build_scene_data_from_scene_dict(nusc, scene_dict)

    return manifest


def write_samples_table(batch_id: str, batch: dict[str, list], output_root: str) -> float:
    """Build and write one batch's sample-table rows to a Parquet file.

    Args:
        batch_id: Identifier for this batch, used in the output filename.
        batch: This batch's manifest (see build_manifest_for_batch): scene
            token -> that scene's list of per-sample aggregate dicts.
        output_root: Root directory to write under.

    Returns:
        Seconds spent writing this table's Parquet file.
    """
    sample_rows = []
    for _, scene_data in batch.items():
        sample_rows.extend(extract_sample_rows(scene_data))

    return write_batch_table_to_file(batch_id, table_name=SAMPLES_TABLE_NAME,
                                     table_rows=sample_rows,
                                     table_schema=sample_schema,
                                     output_root=output_root)


def write_sample_data_table(batch_id: str, batch: dict[str, list], output_root: str) -> float:
    """Build and write one batch's sample_data-table rows to a Parquet file.

    Args:
        batch_id: Identifier for this batch, used in the output filename.
        batch: This batch's manifest (see build_manifest_for_batch).
        output_root: Root directory to write under.

    Returns:
        Seconds spent writing this table's Parquet file.
    """
    sample_data_rows = []
    for _, scene_data in batch.items():
        sample_data_rows.extend(extract_sample_data_rows(scene_data))

    return write_batch_table_to_file(batch_id, table_name=SAMPLE_DATA_TABLE_NAME,
                                     table_rows=sample_data_rows,
                                     table_schema=sample_data_schema,
                                     output_root=output_root)


def write_sample_annotations_table(batch_id: str, batch: dict[str, list], output_root: str) -> float:
    """Build and write one batch's sample_annotation-table rows to a Parquet file.

    Args:
        batch_id: Identifier for this batch, used in the output filename.
        batch: This batch's manifest (see build_manifest_for_batch).
        output_root: Root directory to write under.

    Returns:
        Seconds spent writing this table's Parquet file.
    """
    sample_annotations_rows = []
    for _, scene_data in batch.items():
        sample_annotations_rows.extend(extract_sample_annotations_rows(scene_data))

    return write_batch_table_to_file(batch_id, table_name=SAMPLE_ANNOTATIONS_TABLE_NAME,
                                     table_rows=sample_annotations_rows,
                                     table_schema=sample_annotation_schema,
                                     output_root=output_root)


@ray.remote
def write_core_tables(batch_id: str, batch: dict[str, list], output_root: str) -> dict:
    """Write one batch's sample, sample_data, and sample_annotation tables.

    Ray task: runs on a worker, writing 3 Parquet files for this batch.

    Args:
        batch_id: Identifier for this batch, used in each output filename.
        batch: This batch's manifest (see build_manifest_for_batch).
        output_root: Root directory to write under.

    Returns:
        Timing info for the benchmark harness: task name, the wall-clock time
        (time.time(), comparable across the driver and this worker process)
        this task actually started running, total seconds spent writing
        Parquet files (summed across all 3 tables), and when it finished.
    """
    received_at = time.time()

    write_time = write_samples_table(batch_id, batch, output_root)
    write_time += write_sample_data_table(batch_id, batch, output_root)
    write_time += write_sample_annotations_table(batch_id, batch, output_root)

    return {"task": "write_core_tables", "received_at": received_at,
           "write_time": write_time, "ended_at": time.time()}


CORE_TABLE_WRITERS = {
    "samples": write_samples_table,
    "sample_data": write_sample_data_table,
    "sample_annotations": write_sample_annotations_table,
}


@ray.remote
def write_core_table(table: str, batch_id: str, batch: dict[str, list], output_root: str) -> dict:
    """Write one of a batch's core tables (TASK_LAYOUT=split).

    Ray task: the split-layout counterpart of write_core_tables, writing just
    1 of its 3 Parquet files, so each core table costs its own task.

    Args:
        table: Which core table to write (a key of CORE_TABLE_WRITERS).
        batch_id: Identifier for this batch, used in the output filename.
        batch: This batch's manifest (see build_manifest_for_batch).
        output_root: Root directory to write under.

    Returns:
        Timing info for the benchmark harness -- see write_core_tables.
    """
    received_at = time.time()

    write_time = CORE_TABLE_WRITERS[table](batch_id, batch, output_root)

    return {"task": f"write_core_table[{table}]", "received_at": received_at,
           "write_time": write_time, "ended_at": time.time()}


@ray.remote
def write_bboxes(batch_id: str, batch: dict[str, list], output_root: str) -> dict:
    """Project every annotation onto its cameras and write the clipped 2D boxes.

    Ray task: runs on a worker, writing 1 Parquet file (one row per
    annotation/camera projection that lands on the canvas) for this batch.

    Args:
        batch_id: Identifier for this batch, used in the output filename.
        batch: This batch's manifest (see build_manifest_for_batch).
        output_root: Root directory to write under.

    Returns:
        Timing info for the benchmark harness -- see write_core_tables.
    """
    received_at = time.time()

    canvas_bboxes_rows = []
    for _, scene_data in batch.items():
        canvas_projections = project_scene_annotations_to_canvas(scene_data)
        canvas_bboxes_rows.extend(collate_canvas_bboxes(canvas_projections))

    write_time = write_batch_table_to_file(batch_id, table_name=CANVAS_BBOXES_TABLE_NAME,
                                           table_rows=canvas_bboxes_rows,
                                           table_schema=canvas_bboxes_schema,
                                           output_root=output_root)

    return {"task": "write_bboxes", "received_at": received_at,
           "write_time": write_time, "ended_at": time.time()}



if __name__ == "__main__":
    # Explicit rather than relying on .remote()'s implicit auto-init on first
    # call; ignore_reinit_error so re-running in the same process (e.g. a
    # REPL/notebook) doesn't crash on an already-initialized Ray instance.
    # NUM_CPUS unset means every CPU this process may use: machine.available_cpus()
    # rather than Ray's own detection, which in a container can count the host's
    # cores (it misses a cgroup v2 CPU set). Set NUM_CPUS to compare worker counts.
    if TASK_LAYOUT not in ("consolidated", "split"):
        sys.exit(f"TASK_LAYOUT must be 'consolidated' or 'split', not {TASK_LAYOUT!r}")
    ray.init(ignore_reinit_error=True, num_cpus=NUM_CPUS or available_cpus())

    all_refs, pending = [], []  # every ref submitted; refs not yet known-complete
    ref_label: dict[ray.ObjectRef, str] = {}  # ref -> human-readable description, for failure reporting
    ref_submitted_at: dict[ray.ObjectRef, float] = {}  # ref -> time.time() right before .remote()
    nusc = NuScenes(version=NUSCENES_VERSION, dataroot=NUSCENES_DATAROOT, verbose=True)
    output_root = OUTPUT_ROOT

    for scene_batch in batched(nusc.scene, BATCH_SIZE):
        batch_id = generate_batch_id()
        batch_ref = ray.put(build_manifest_for_batch(nusc, scene_batch))  # once per batch

        # Independent write tasks per batch -- core tables and bboxes don't
        # depend on each other's output, so they run as separate tasks: two
        # (consolidated), or four with each core table split out (split).
        # time.time() (not perf_counter) since it's compared against each
        # task's own received_at, recorded in a different (worker) process.
        submitted_at = time.time()
        if TASK_LAYOUT == "split":
            refs = {
                write_core_table.remote(table, batch_id, batch_ref, output_root):
                    f"batch {batch_id}: write_core_table[{table}]"
                for table in CORE_TABLE_WRITERS
            }
        else:
            refs = {
                write_core_tables.remote(batch_id, batch_ref, output_root): f"batch {batch_id}: write_core_tables",
            }
        refs[write_bboxes.remote(batch_id, batch_ref, output_root)] = f"batch {batch_id}: write_bboxes"
        ref_label.update(refs)
        ref_submitted_at.update({ref: submitted_at for ref in refs})
        all_refs += refs.keys()
        pending += refs.keys()

        while len(pending) >= MAX_IN_FLIGHT:            # backpressure
            _, pending = ray.wait(pending, num_returns=1)

    # Collect every task's outcome individually, rather than one ray.get(all_refs)
    # call that raises (and stops looking) on the first failure -- a bad scene in
    # one batch shouldn't hide whether every other batch succeeded.
    failures: list[tuple[str, Exception]] = []
    task_timings: list[dict] = []  # per-task timing, for the benchmark harness
    for ref in all_refs:
        try:
            result = ray.get(ref)
            submitted_at = ref_submitted_at[ref]
            task_timings.append({
                "label": ref_label[ref],
                "received_at": result["received_at"],  # for the benchmark harness' startup-overhead calc
                "wait_time": result["received_at"] - submitted_at,
                "write_time": result["write_time"],
                "compute_time": result["ended_at"] - result["received_at"] - result["write_time"],
            })
        except Exception as exc:
            failures.append((ref_label[ref], exc))

    # Written even if some tasks failed, so the benchmark harness still has
    # timing for whichever tasks did succeed.
    os.makedirs("reports", exist_ok=True)
    with open("reports/partition_timing.json", "w") as timing_file:
        json.dump(task_timings, timing_file, indent=2)

    if failures:
        print(f"{len(failures)}/{len(all_refs)} batch tasks failed:")
        for label, exc in failures:
            print(f"  {label}: {exc}")
        sys.exit(1)

    print(f"All {len(all_refs)} batch tasks completed successfully.")

