"""Validate ingestion.py's key-mapping wiring against nuScenes-devkit records directly.

For every sample in nusc.scene[0], independently re-fetches each raw nuScenes
record ingestion.py reads from (sample, sample_data, ego_pose,
calibrated_sensor, sample_annotation) and compares every key in
ingestion_keys.py's mapping dicts against ingestion.py's own extracted values.
This catches wiring bugs (wrong record passed, wrong field read) rather than
re-deriving the same mapping ingestion.py already uses.

Covers:
    SAMPLE_INGESTION_KEYS       <-> sample/scene raw fields ('annotations' is
                                     checked as a key-set, not a scalar value)
    EGO_POSE_KEYS_MAP           <-> ego_pose raw fields
    SENSOR_DATA_KEYS_MAP        <-> sample_data raw fields
    SENSOR_CALIBRATION_KEYS_MAP <-> calibrated_sensor raw fields
    SAMPLE_ANNOTATION_KEYS_MAP  <-> sample_annotation raw fields

Out of scope: SAMPLE_DATA_INGESTION_KEYS, SAMPLE_ANNOTATION_INGESTION_KEYS, and
TRANSFORM_KEYS are aliases into the *_KEYS_MAP dicts above -- every one of
their values is used as a *_KEYS_MAP key, so exercising the *_KEYS_MAP dicts
already covers them (verified directly: see the ingestion_keys.py coverage
check run before this script was written). KNOWN_SENSOR_MODALITIES/
KNOWN_SENSOR_CHANNELS are static whitelists, not per-record fields, so
there's nothing to check them against.

Running this script as __main__ writes a structured report -- per-key
verified/total counts and mismatch details -- to
reports/verify_ingestion_keys.txt (overwritten each run). Nothing is printed
to stdout; every result goes into the report file.
"""

from av_data_pipeline.config import NUSCENES_VERSION, NUSCENES_DATAROOT
from av_data_pipeline.ingestion import (
    get_sample_tokens_from_scene,
    extract_sensor_data_from_sample_data,
    extract_annotation_from_sample_data,
    extract_sample_data_from_sample,
)
from av_data_pipeline.ingestion_keys import *
from av_data_pipeline.nuscenes import *
from av_data_pipeline.utils import visibility_range


class KeyTracker:
    """Tallies verified/mismatched values per aggregate key for one record type."""

    def __init__(self, name: str) -> None:
        """Initialize an empty tracker for one raw-record type's key mappings.

        Args:
            name: Human-readable name for this record type, used as the
                report section heading.
        """
        self.name = name
        self.verified: dict[str, int] = {}
        self.mismatches: dict[str, list[tuple[str, str]]] = {}

    def check(self, agg_key: str, record_token: str, extracted_value, raw_value) -> None:
        """Compare one extracted/raw value pair for an aggregate key and tally the result.

        Args:
            agg_key: Aggregate key constant being checked (e.g. 'EP_ROTATION').
            record_token: Token of the record this value came from, for mismatch reporting.
            extracted_value: Value ingestion.py produced for this key.
            raw_value: Value independently fetched straight from nuscenes-devkit.
        """
        if extracted_value == raw_value:
            self.verified[agg_key] = self.verified.get(agg_key, 0) + 1
        else:
            self.mismatches.setdefault(agg_key, []).append(
                (record_token, f"extracted={extracted_value!r} raw={raw_value!r}"))

    def report_lines(self) -> list[str]:
        """Render this record type's per-key verified/total counts and mismatch details.

        Returns:
            Lines to append to the overall report, ending with a blank separator.
        """
        lines = [f"--- {self.name} ---"]
        all_keys = sorted(set(self.verified) | set(self.mismatches))
        for agg_key in all_keys:
            verified = self.verified.get(agg_key, 0)
            mismatches = self.mismatches.get(agg_key, [])
            total = verified + len(mismatches)
            lines.append(f"{agg_key}: {verified}/{total}")
            for token, detail in mismatches:
                lines.append(f"  MISMATCH ({token}): {detail}")
        lines.append("")
        return lines


def check_key_map(tracker: KeyTracker, key_map: dict[str, str], record_token: str,
                  extracted: dict, raw_record: dict) -> None:
    """Check every non-derived key in a *_KEYS_MAP dict against its raw record.

    Skips keys mapped to '' (the record's own token, or a derived value like
    VISIBILITY_RANGE) -- those are checked separately by the caller.

    Args:
        tracker: Tracker to record results into.
        key_map: One of the *_KEYS_MAP dicts (aggregate key -> raw field name).
        record_token: Token of the raw_record, for mismatch reporting.
        extracted: ingestion.py's extracted dict for this record.
        raw_record: The matching raw nuScenes record, fetched independently.
    """
    for agg_key, raw_field in key_map.items():
        if len(raw_field) == 0:
            continue
        tracker.check(agg_key, record_token, extracted[agg_key], raw_record[raw_field])


if __name__ == "__main__":
    from nuscenes.nuscenes import NuScenes
    import os

    REPORT_DIR = "reports"
    REPORT_FILE = os.path.join(REPORT_DIR, "verify_ingestion_keys.txt")

    nusc = NuScenes(version=NUSCENES_VERSION, dataroot=NUSCENES_DATAROOT, verbose=True)
    scene_token = nusc.scene[0]['token']

    sample_tracker = KeyTracker("SAMPLE_INGESTION_KEYS")
    ego_pose_tracker = KeyTracker("EGO_POSE_KEYS_MAP")
    sensor_data_tracker = KeyTracker("SENSOR_DATA_KEYS_MAP")
    sensor_calibration_tracker = KeyTracker("SENSOR_CALIBRATION_KEYS_MAP")
    annotation_tracker = KeyTracker("SAMPLE_ANNOTATION_KEYS_MAP")

    for sample_token in get_sample_tokens_from_scene(nusc, scene_token):
        sample = nusc.get(SAMPLE_TABLE, sample_token)
        extracted_sample_data = extract_sample_data_from_sample(nusc, sample_token)

        # SAMPLE_INGESTION_KEYS: sample/scene-level fields.
        sample_tracker.check(SAMPLE_INGESTION_KEYS['token'], sample_token,
                             extracted_sample_data[SAMPLE_INGESTION_KEYS['token']], sample_token)
        sample_tracker.check(SAMPLE_INGESTION_KEYS['scene_token'], sample_token,
                             extracted_sample_data[SAMPLE_INGESTION_KEYS['scene_token']],
                             sample['scene_token'])
        sample_tracker.check(SAMPLE_INGESTION_KEYS['sample_ts'], sample_token,
                             extracted_sample_data[SAMPLE_INGESTION_KEYS['sample_ts']],
                             sample['timestamp'])
        # 'annotations' isn't a scalar field -- check its key set instead.
        extracted_ann_tokens = set(extracted_sample_data[SAMPLE_INGESTION_KEYS['annotations']].keys())
        sample_tracker.check(SAMPLE_INGESTION_KEYS['annotations'], sample_token,
                             extracted_ann_tokens, set(sample['anns']))

        # EGO_POSE_KEYS_MAP / SENSOR_DATA_KEYS_MAP / SENSOR_CALIBRATION_KEYS_MAP:
        # per-sensor-reading fields, all merged into one extracted dict.
        for data_token in sample['data'].values():
            extracted = extract_sensor_data_from_sample_data(nusc, data_token)

            sd_record = nusc.get(SAMPLE_DATA_TABLE, data_token)
            ego_pose_record = nusc.get(EGO_POSE_TABLE, sd_record[EGO_POSE_TOKEN])
            calibrated_sensor_record = nusc.get(CALIBRATED_SENSOR_TABLE, sd_record[CALIBRATED_SENSOR_TOKEN])

            check_key_map(ego_pose_tracker, EGO_POSE_KEYS_MAP, data_token, extracted, ego_pose_record)
            check_key_map(sensor_data_tracker, SENSOR_DATA_KEYS_MAP, data_token, extracted, sd_record)
            check_key_map(sensor_calibration_tracker, SENSOR_CALIBRATION_KEYS_MAP, data_token,
                         extracted, calibrated_sensor_record)

            # Token-itself special case (mapped to '' in SENSOR_DATA_KEYS_MAP).
            sensor_data_tracker.check(SAMPLE_DATA_INGESTION_KEYS['token'], data_token,
                                     extracted[SAMPLE_DATA_INGESTION_KEYS['token']], data_token)

        # SAMPLE_ANNOTATION_KEYS_MAP: per-annotation fields.
        for ann_token in sample['anns']:
            extracted = extract_annotation_from_sample_data(nusc, ann_token)
            sa_record = nusc.get(SAMPLE_ANNOTATION_TABLE, ann_token)

            check_key_map(annotation_tracker, SAMPLE_ANNOTATION_KEYS_MAP, ann_token, extracted, sa_record)

            # Token-itself special case (mapped to '' in SAMPLE_ANNOTATION_KEYS_MAP).
            annotation_tracker.check(SAMPLE_ANNOTATION_INGESTION_KEYS['token'], ann_token,
                                    extracted[SAMPLE_ANNOTATION_INGESTION_KEYS['token']], ann_token)
            # Derived value special case: VISIBILITY_RANGE isn't a raw passthrough,
            # so it's checked against re-deriving it from the raw visibility_token.
            annotation_tracker.check(SAMPLE_ANNOTATION_INGESTION_KEYS['visibility_range'], ann_token,
                                    extracted[SAMPLE_ANNOTATION_INGESTION_KEYS['visibility_range']],
                                    visibility_range(sa_record['visibility_token']))

    report = []
    report.extend(sample_tracker.report_lines())
    report.extend(ego_pose_tracker.report_lines())
    report.extend(sensor_data_tracker.report_lines())
    report.extend(sensor_calibration_tracker.report_lines())
    report.extend(annotation_tracker.report_lines())

    os.makedirs(REPORT_DIR, exist_ok=True)
    with open(REPORT_FILE, "w") as outfile:
        outfile.write("\n".join(report) + "\n")
