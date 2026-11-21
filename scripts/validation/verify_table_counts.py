"""Validate partition.py's finished Parquet tables against the nuScenes devkit's own tables.

The other ingestion checks test how rows are built, one scene at a time. This
one reads everything a full partition.py run wrote under OUTPUT_ROOT and asks
whether the export as a whole is complete: every devkit record present exactly
once, nothing extra, and every row attached to the right parent.

    samples        every nusc.sample token present; no repeats; no unknown tokens
    sample_data    every token in a sample's 'data' dict present (all sensors,
                   from the devkit, not the pipeline's own channel whitelist);
                   no repeats; no unknown tokens; each row's sample_token is the
                   devkit's sample for that reading
    annotations    every nusc.sample_annotation token present; no repeats; no
                   unknown tokens; each row's sample_token is the devkit's
    canvas_bboxes  no repeated (annotation, camera reading) pair; every box
                   points at an annotation and a camera reading that are both in
                   the exported tables, from the same sample. Its expected row
                   count isn't known in advance (it depends on which cameras see
                   each box), so it gets reference checks instead of a count.

What this catches: a batch whose task failed or never ran (missing tokens),
a scene written twice or a record repeated within one (repeats, like the
duplicated last sample this check was written after), leftover files from an
earlier run or another dataset version in the same folder (repeats or unknown
tokens), and rows filed under the wrong sample.

Only token columns are read, so this stays quick on trainval. Running it as
__main__ writes reports/verify_table_counts.txt (overwritten each run): one
"label: verified/total" tally per check, then up to MAX_EXAMPLES offending
tokens per failed check. Nothing is printed to stdout.
"""

import os
from collections import Counter

import pyarrow.parquet as pq
from nuscenes.nuscenes import NuScenes

from av_data_pipeline.config import NUSCENES_VERSION, NUSCENES_DATAROOT, OUTPUT_ROOT

REPORT_DIR = "reports"
REPORT_FILE = os.path.join(REPORT_DIR, "verify_table_counts.txt")
MAX_EXAMPLES = 5


class TableReport:
    """Tallies and failure examples for one output table."""

    def __init__(self, name: str, files: int, rows: int) -> None:
        """Start one table's section.

        Args:
            name: Table (output folder) name.
            files: Number of Parquet files read for it.
            rows: Total rows across those files.
        """
        self.name = name
        self.files = files
        self.rows = rows
        self.lines: list[str] = []

    def tally(self, label: str, verified: int, total: int, bad: list[str]) -> None:
        """Record one check as "label: verified/total", plus a few offending tokens if it failed."""
        self.lines.append(f"{label}: {verified}/{total}")
        if bad:
            self.lines.append(f"  e.g. {', '.join(bad[:MAX_EXAMPLES])}")

    def report_lines(self) -> list[str]:
        """This table's section of the report, ending with a blank separator."""
        return [f"--- {self.name} ---", f"Files read: {self.files}", f"Rows read: {self.rows}",
                *self.lines, ""]


def read_columns(table: str, columns: list[str]) -> tuple[int, dict[str, list]]:
    """Read some columns of every Parquet file in one output table's folder.

    Args:
        table: Table folder name under OUTPUT_ROOT.
        columns: Columns to read.

    Returns:
        (number of files, column name -> values across all files). A missing
        folder reads as zero files and empty columns, so every expected token
        shows up as missing rather than crashing the check.
    """
    folder = os.path.join(OUTPUT_ROOT, table)
    files = sorted(f for f in os.listdir(folder) if f.endswith(".parquet")) if os.path.isdir(folder) else []
    values: dict[str, list] = {c: [] for c in columns}
    for name in files:
        part = pq.read_table(os.path.join(folder, name), columns=columns)
        for c in columns:
            values[c].extend(part.column(c).to_pylist())
    return len(files), values


def check_tokens(report: TableReport, found: list[str], expected: set[str]) -> None:
    """The three completeness tallies for a table keyed by one token column.

    Present: expected tokens found at least once. Not repeated: rows whose
    token isn't a repeat of an earlier row. Known: rows whose token is one the
    devkit expects.
    """
    counts = Counter(found)
    missing = sorted(expected - counts.keys())
    repeated = sorted(t for t, n in counts.items() if n > 1)
    unknown = sorted(t for t in counts if t not in expected)

    report.tally("Present", len(expected) - len(missing), len(expected), missing)
    report.tally("Not repeated", len(counts), len(found), repeated)
    report.tally("Known", sum(n for t, n in counts.items() if t in expected), len(found), unknown)


def check_parent(report: TableReport, tokens: list[str], sample_tokens: list[str],
                 devkit_sample: dict[str, str]) -> None:
    """Rows whose sample_token is the sample the devkit files that record under."""
    wrong = [t for t, s in zip(tokens, sample_tokens) if devkit_sample.get(t) != s]
    report.tally("Right sample", len(tokens) - len(wrong), len(tokens), wrong)


if __name__ == "__main__":
    nusc = NuScenes(version=NUSCENES_VERSION, dataroot=NUSCENES_DATAROOT, verbose=True)

    expected_samples = {s["token"] for s in nusc.sample}
    # Every reading a sample points at, whatever the sensor: the devkit's own
    # list, so a sensor the pipeline doesn't know about shows up as missing.
    reading_sample = {tok: s["token"] for s in nusc.sample for tok in s["data"].values()}
    annotation_sample = {a["token"]: a["sample_token"] for a in nusc.sample_annotation}

    reports = []

    files, cols = read_columns("samples", ["sample_token"])
    samples = TableReport("samples", files, len(cols["sample_token"]))
    check_tokens(samples, cols["sample_token"], expected_samples)
    reports.append(samples)

    files, cols = read_columns("sample_data", ["sample_data_token", "sample_token"])
    sample_data = TableReport("sample_data", files, len(cols["sample_data_token"]))
    check_tokens(sample_data, cols["sample_data_token"], set(reading_sample))
    check_parent(sample_data, cols["sample_data_token"], cols["sample_token"], reading_sample)
    reports.append(sample_data)
    # Camera readings by the devkit's own sensor records, not the table's
    # modality column, so a reading filed under the wrong modality can't pass.
    devkit_cameras = {tok for tok in reading_sample
                      if nusc.get("sample_data", tok)["sensor_modality"] == "camera"}
    exported_cameras = devkit_cameras & set(cols["sample_data_token"])

    files, cols = read_columns("annotations", ["annotation_token", "sample_token"])
    annotations = TableReport("annotations", files, len(cols["annotation_token"]))
    check_tokens(annotations, cols["annotation_token"], set(annotation_sample))
    check_parent(annotations, cols["annotation_token"], cols["sample_token"], annotation_sample)
    reports.append(annotations)
    exported_annotations = set(cols["annotation_token"])

    files, cols = read_columns("canvas_bboxes", ["annotation_token", "sample_data_token"])
    pairs = list(zip(cols["annotation_token"], cols["sample_data_token"]))
    bboxes = TableReport("canvas_bboxes", files, len(pairs))
    pair_counts = Counter(pairs)
    bboxes.tally("Not repeated", len(pair_counts), len(pairs),
                 sorted(f"{a}/{s}" for (a, s), n in pair_counts.items() if n > 1))
    no_annotation = [a for a, _ in pairs if a not in exported_annotations]
    bboxes.tally("Annotation exported", len(pairs) - len(no_annotation), len(pairs), no_annotation)
    no_camera = [s for _, s in pairs if s not in exported_cameras]
    bboxes.tally("Camera reading exported", len(pairs) - len(no_camera), len(pairs), no_camera)
    wrong_sample = [f"{a}/{s}" for a, s in pairs
                    if annotation_sample.get(a) is None or annotation_sample[a] != reading_sample.get(s)]
    bboxes.tally("Same sample", len(pairs) - len(wrong_sample), len(pairs), wrong_sample)
    reports.append(bboxes)

    lines = [f"Dataset: {NUSCENES_VERSION}", f"Output root: {OUTPUT_ROOT}", ""]
    for report in reports:
        lines.extend(report.report_lines())

    os.makedirs(REPORT_DIR, exist_ok=True)
    with open(REPORT_FILE, "w") as outfile:
        outfile.write("\n".join(lines))
