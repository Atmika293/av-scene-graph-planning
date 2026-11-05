"""Validate table.py's row construction against the already-ingested aggregate dict.

build_scene_data(nusc, scene_token) produces one trusted aggregate dict per
sample -- trusted because verify_ingestion_keys.py already proved it matches
nuscenes-devkit field-for-field. This script doesn't re-derive from the
devkit again; it only checks the next transformation stage: does table.py
correctly flatten that aggregate dict into each table's row shape?

Two kinds of check, run independently so one failure mode doesn't hide another:

1. Column-set check: does each constructed row's key set match the target
   pyarrow schema's column names (schema.py's *_schema)? Catches shape bugs --
   a row builder reading the wrong *_COLUMNS dict, or *_COLUMNS/*_schema
   having drifted apart (misspelled or renamed columns on one side only).
2. Value check: for every column in a *_COLUMNS dict, does the constructed
   row's value match resolve(aggregate, columns_map[col])? resolve() is a
   small, independent re-indexer written here, not a reuse of
   utils.construct_table_row/multiindex -- those are exactly what's under
   test, so trusting them as the ground truth would hide real regressions.

Also smoke-tests the orchestration layer (extract_sample_rows,
extract_sample_data_rows, extract_sample_annotations_rows,
collate_canvas_bboxes) end-to-end over the scene, catching exceptions per
function -- this is what catches a wrapper calling the wrong row-builder
with the wrong arguments, as distinct from a single row being built wrong.

Running this script as __main__ writes a structured report to
reports/verify_table_rows.txt (overwritten each run). Nothing is printed to
stdout; every result goes into the report file.
"""

from av_data_pipeline.ingestion import build_scene_data
from av_data_pipeline.ingestion_keys import (
    SAMPLE_INGESTION_KEYS, SAMPLE_DATA_INGESTION_KEYS,
    KNOWN_SENSOR_MODALITIES, KNOWN_SENSOR_CHANNELS,
)
from av_data_pipeline.schema import (
    SAMPLE_COLUMNS, SAMPLE_DATA_COLUMNS, SAMPLE_ANNOTATION_COLUMNS, CANVAS_BOXES_COLUMNS,
    sample_schema, sample_data_schema, sample_annotation_schema, canvas_bboxes_schema,
)
from av_data_pipeline.table import (
    build_sample_row, build_sample_data_row, build_sample_annotation_row, build_canvas_bbox_row,
    extract_sample_rows, extract_sample_data_rows, extract_sample_annotations_rows,
    collate_canvas_bboxes,
)
from av_data_pipeline.transform import project_scene_annotations_to_canvas


def resolve(aggregate: dict, key_or_tuple):
    """Independently resolve a *_COLUMNS value spec against an aggregate dict.

    Mirrors what construct_table_row/multiindex are supposed to do, written
    as simple, obviously-correct indexing rather than calling those
    functions -- they're exactly what this script is verifying.

    Args:
        aggregate: The source dict a *_COLUMNS entry points into (an
            aggregate sample dict, sensor_data dict, or annotation_data dict).
        key_or_tuple: A *_COLUMNS value: '' (no source field), a plain
            aggregate key string, or an (aggregate key, *indices) tuple for
            one component of a vector/matrix field.

    Returns:
        The resolved value, or None if key_or_tuple is '' or any intermediate
        value along the way is None/empty (matching multiindex's short-circuit,
        e.g. a non-camera sensor's empty SC_INTRINSIC list).
    """
    if key_or_tuple == '':
        return None
    if isinstance(key_or_tuple, tuple):
        agg_key, *indices = key_or_tuple
        value = aggregate[agg_key]
        for idx in indices:
            if value is None or len(value) == 0:
                return None
            value = value[idx]
        return value
    return aggregate[key_or_tuple]


class KeyTracker:
    """Tallies verified/mismatched values per check name (e.g. a table, or a table.column)."""

    def __init__(self, name: str) -> None:
        """Initialize an empty tracker.

        Args:
            name: Human-readable name for this group of checks, used as the
                report section heading.
        """
        self.name = name
        self.verified: dict[str, int] = {}
        self.mismatches: dict[str, list[tuple[str, str]]] = {}

    def check(self, check_key: str, record_token: str, actual, expected) -> None:
        """Compare one actual/expected pair and tally the result.

        Args:
            check_key: What's being checked (e.g. 'SAMPLE_DATA_COLUMNS.ego_x'
                or 'sample_data:__columns__').
            record_token: Token of the record this check belongs to, for
                mismatch reporting.
            actual: Value table.py actually produced.
            expected: Independently resolved expected value.
        """
        if actual == expected:
            self.verified[check_key] = self.verified.get(check_key, 0) + 1
        else:
            self.mismatches.setdefault(check_key, []).append(
                (record_token, f"actual={actual!r} expected={expected!r}"))

    def check_set(self, check_key: str, record_token: str, actual: set, expected: set) -> None:
        """Compare two sets (e.g. a row's columns vs. a schema's), reporting only the diff.

        Unlike check(), which would dump both full sets via !r, this reports
        just what's missing and/or extra -- the useful part when most of a
        large set matches and only a column or two actually differs.

        Args:
            check_key: What's being checked (e.g. 'sample_data').
            record_token: Token of the record this check belongs to, for
                mismatch reporting.
            actual: The set table.py actually produced.
            expected: The independently derived expected set.
        """
        if actual == expected:
            self.verified[check_key] = self.verified.get(check_key, 0) + 1
            return

        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        parts = []
        if missing:
            parts.append(f"missing={missing}")
        if extra:
            parts.append(f"extra={extra}")
        self.mismatches.setdefault(check_key, []).append((record_token, "; ".join(parts)))

    def record_exception(self, check_key: str, record_token: str, exc: Exception) -> None:
        """Record a caught exception as a mismatch instead of letting it propagate.

        Args:
            check_key: What was being checked when the exception was raised.
            record_token: Token of the record this check belongs to.
            exc: The caught exception.
        """
        self.mismatches.setdefault(check_key, []).append(
            (record_token, f"raised {type(exc).__name__}: {exc}"))

    def report_lines(self) -> list[str]:
        """Render this tracker's per-check verified/total counts and mismatch details.

        Returns:
            Lines to append to the overall report, ending with a blank separator.
        """
        lines = [f"--- {self.name} ---"]
        all_keys = sorted(set(self.verified) | set(self.mismatches))
        for check_key in all_keys:
            verified = self.verified.get(check_key, 0)
            mismatches = self.mismatches.get(check_key, [])
            total = verified + len(mismatches)
            lines.append(f"{check_key}: {verified}/{total}")

            # Group identical mismatch details -- a static structural bug (e.g.
            # a wrong columns_map) produces the same detail for every affected
            # record, so show it once with a few examples instead of once per record.
            by_detail: dict[str, list[str]] = {}
            for token, detail in mismatches:
                by_detail.setdefault(detail, []).append(token)

            for detail, tokens in by_detail.items():
                examples = ", ".join(tokens[:3])
                suffix = f" (+{len(tokens) - 3} more)" if len(tokens) > 3 else ""
                lines.append(f"  MISMATCH ({examples}{suffix}): {detail}")
        lines.append("")
        return lines


def check_row(columns_tracker: KeyTracker, values_tracker: KeyTracker, table_name: str,
             columns_map_name: str, row_token: str, row: dict, columns_map: dict,
             schema, source: dict, injected: dict | None = None) -> None:
    """Check one constructed row's column set and per-column values.

    Args:
        columns_tracker: Tracker for column-set-vs-schema checks.
        values_tracker: Tracker for per-column value checks.
        table_name: Name of the table this row belongs to (e.g. 'sample_data'),
            used as the column-set check's key.
        columns_map_name: Name of the *_COLUMNS dict used to build this row
            (e.g. 'SAMPLE_DATA_COLUMNS'), used to prefix per-column check keys.
        row_token: Token of the record this row was built from, for mismatch reporting.
        row: The constructed row dict.
        columns_map: The *_COLUMNS dict that was used to build it.
        schema: The target pyarrow schema this row should match.
        source: The aggregate/sensor/annotation dict the row was built from --
            resolve()'s ground truth for every column mapped to a real key.
        injected: Expected values for columns the row builder sets directly
            rather than through columns_map/source (columns_map maps these to
            '', e.g. 'sample_token' -- construct_table_row can't resolve a ''
            key, so resolve() would wrongly expect None for them).
    """
    columns_tracker.check_set(table_name, row_token, set(row.keys()), set(schema.names))
    injected = injected or {}

    for col, key_spec in columns_map.items():
        expected = injected[col] if col in injected else resolve(source, key_spec)
        values_tracker.check(f"{columns_map_name}.{col}", row_token, row.get(col), expected)


def build_and_check(values_tracker: KeyTracker, builder_name: str, record_token: str,
                    build_fn, build_args: tuple, check_fn) -> None:
    """Build one row, then verify it, labeling build vs. check failures separately.

    A row-builder crash is a real table.py bug; a crash while checking an
    already-built row is a bug in this script's own verification code (e.g.
    resolve()) -- these must not share one label, or a bug in this script
    could get misattributed to table.py.

    Args:
        values_tracker: Tracker to record a build-time exception into.
        builder_name: Name of the row-builder function, used as the
            build-failure exception label.
        record_token: Token of the record being built, for mismatch reporting.
        build_fn: The table.py row-builder function to call.
        build_args: Positional arguments to pass to build_fn.
        check_fn: Called with the built row if construction succeeded; any
            exception it raises is labeled as a check-time failure instead.
    """
    try:
        row = build_fn(*build_args)
    except Exception as exc:
        values_tracker.record_exception(builder_name, record_token, exc)
        return

    try:
        check_fn(row)
    except Exception as exc:
        values_tracker.record_exception(f"check({builder_name})", record_token, exc)


def smoke_test(tracker: KeyTracker, name: str, fn, *args) -> None:
    """Run one orchestration-layer function, catching any exception as a finding.

    Args:
        tracker: Tracker to record the result into.
        name: Name of the function being smoke-tested, used as the check key.
        fn: The function to call.
        *args: Positional arguments to pass to fn.
    """
    try:
        rows = fn(*args)
        tracker.check(f"{name} ({len(rows)} rows)", "(scene)", True, True)
    except Exception as exc:
        tracker.record_exception(name, "(scene)", exc)


if __name__ == "__main__":
    from nuscenes.nuscenes import NuScenes
    import os

    from av_data_pipeline.config import NUSCENES_VERSION, NUSCENES_DATAROOT

    REPORT_DIR = "reports"
    REPORT_FILE = os.path.join(REPORT_DIR, "verify_table_rows.txt")

    nusc = NuScenes(version=NUSCENES_VERSION, dataroot=NUSCENES_DATAROOT, verbose=True)
    scene_token = nusc.scene[0]['token']
    scene_data = build_scene_data(nusc, scene_token)

    columns_tracker = KeyTracker("Column-set vs. schema")
    values_tracker = KeyTracker("Per-column values")
    orchestration_tracker = KeyTracker("Orchestration-layer smoke test")

    for sample_data in scene_data:
        sample_token = sample_data[SAMPLE_INGESTION_KEYS['token']]

        # sample table
        build_and_check(
            values_tracker, "build_sample_row", sample_token,
            build_sample_row, (sample_data,),
            lambda row: check_row(columns_tracker, values_tracker, "sample", "SAMPLE_COLUMNS",
                                  sample_token, row, SAMPLE_COLUMNS, sample_schema, sample_data))

        # sample_data table: one row per sensor reading
        for modality in KNOWN_SENSOR_MODALITIES:
            for channel in KNOWN_SENSOR_CHANNELS[modality]:
                sensor_data = sample_data[modality][channel]
                data_token = sensor_data[SAMPLE_DATA_INGESTION_KEYS['token']]
                build_and_check(
                    values_tracker, "build_sample_data_row", data_token,
                    build_sample_data_row, (sample_token, sensor_data),
                    lambda row, sensor_data=sensor_data, data_token=data_token: check_row(
                        columns_tracker, values_tracker, "sample_data", "SAMPLE_DATA_COLUMNS",
                        data_token, row, SAMPLE_DATA_COLUMNS, sample_data_schema, sensor_data,
                        injected={"sample_token": sample_token}))

        # sample_annotation table: one row per annotation
        for ann_token, annotation_data in sample_data[SAMPLE_INGESTION_KEYS['annotations']].items():
            build_and_check(
                values_tracker, "build_sample_annotation_row", ann_token,
                build_sample_annotation_row, (sample_token, annotation_data),
                lambda row, annotation_data=annotation_data, ann_token=ann_token: check_row(
                    columns_tracker, values_tracker, "sample_annotation", "SAMPLE_ANNOTATION_COLUMNS",
                    ann_token, row, SAMPLE_ANNOTATION_COLUMNS, sample_annotation_schema, annotation_data,
                    injected={"sample_token": sample_token}))

    # canvas_bboxes table: one row per (annotation, camera) projection
    canvas_projections = project_scene_annotations_to_canvas(scene_data)
    for canvas_bboxes in canvas_projections:
        for annotation_token, per_camera in canvas_bboxes.items():
            for sample_data_token, bbox in per_camera.items():
                record_token = f"{annotation_token}/{sample_data_token}"

                def check_bbox_row(row, annotation_token=annotation_token,
                                   sample_data_token=sample_data_token, bbox=bbox, record_token=record_token):
                    columns_tracker.check_set("canvas_bboxes", record_token,
                                             set(row.keys()), set(canvas_bboxes_schema.names))
                    values_tracker.check("CANVAS_BOXES_COLUMNS.annotation_token", record_token,
                                        row.get("annotation_token"), annotation_token)
                    values_tracker.check("CANVAS_BOXES_COLUMNS.sample_data_token", record_token,
                                        row.get("sample_data_token"), sample_data_token)
                    for i, col in enumerate(("xmin", "ymin", "xmax", "ymax")):
                        values_tracker.check(f"CANVAS_BOXES_COLUMNS.{col}", record_token,
                                            row.get(col), bbox[i])

                build_and_check(
                    values_tracker, "build_canvas_bbox_row", record_token,
                    build_canvas_bbox_row, (annotation_token, sample_data_token, bbox),
                    check_bbox_row)

    # Orchestration layer: do the extract_*/collate_* wrappers run end-to-end?
    smoke_test(orchestration_tracker, "extract_sample_rows", extract_sample_rows, scene_data)
    smoke_test(orchestration_tracker, "extract_sample_data_rows", extract_sample_data_rows, scene_data)
    smoke_test(orchestration_tracker, "extract_sample_annotations_rows", extract_sample_annotations_rows, scene_data)
    smoke_test(orchestration_tracker, "collate_canvas_bboxes", collate_canvas_bboxes, canvas_projections)

    report = []
    report.extend(columns_tracker.report_lines())
    report.extend(values_tracker.report_lines())
    report.extend(orchestration_tracker.report_lines())

    os.makedirs(REPORT_DIR, exist_ok=True)
    with open(REPORT_FILE, "w") as outfile:
        outfile.write("\n".join(report) + "\n")
