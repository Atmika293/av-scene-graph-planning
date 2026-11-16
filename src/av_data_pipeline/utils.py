"""Small helpers shared by the ingestion and table-building code.

Timestamp-skew arithmetic, the nuScenes visibility-token lookup, and the
generic key-mapping functions that copy fields out of nuScenes records
(extract_key_values) and flatten aggregate dicts into table rows
(construct_table_row) using the maps in ingestion_keys.py and schema.py.
"""

import math

# nuScenes visibility_token -> (min%, max%) range of the annotation that's visible across all cameras.
VISIBILITY_RANGES = {
    '1': [0, 40],
    '2': [40, 60],
    '3': [60, 80],
    '4': [80, 100],
}

def calculate_timestamp_skew_ms(sample_ts, sample_data_ts):
    """Absolute skew between a sample's timestamp and one sensor reading's, in milliseconds.

    Args:
        sample_ts: The sample (keyframe) timestamp, in microseconds (UNIX time).
        sample_data_ts: The sensor reading's timestamp, in microseconds.

    Returns:
        |sample_data_ts - sample_ts| / 1000.
    """
    return math.fabs(sample_data_ts - sample_ts) / 1000

def calculate_timestamp_skew_us(sample_ts, sample_data_ts):
    """Same as calculate_timestamp_skew_ms, but left in microseconds."""
    return math.fabs(sample_data_ts - sample_ts)

def visibility_range(token: str) -> list[int, int]:
    """Map a sample_annotation visibility_token to its (min%, max%) range."""
    return VISIBILITY_RANGES[token]

def extract_key_values(extract_dict: dict, record, keys_map: dict[str, str]):
    """Copy fields from a raw nuScenes record into an aggregate dict, renaming them.

    Args:
        extract_dict: The aggregate dict to add fields to (modified in place).
        record: A raw nuScenes record (e.g. from nusc.get).
        keys_map: Aggregate key -> raw record field name (an ingestion_keys
            *_KEYS_MAP). Entries with an empty field name are skipped.

    Returns:
        extract_dict, for chaining.
    """
    for col, key in keys_map.items():
        if len(key) > 0: # Check if the key is not empty
            extract_dict[col] = record[key]

    return extract_dict

def multiindex(obj: object, multi_index: tuple):
    """Index into nested containers one key/index at a time (obj[a][b]...).

    Args:
        obj: The outermost container.
        multi_index: Keys/indices to apply in order, e.g. (SC_INTRINSIC, 0, 2).

    Returns:
        The value at that path, or None if any level along the way is None or empty.
    """
    for idx in multi_index:
        if obj is None or len(obj) == 0:
            return None

        obj = obj[idx]

    return obj

def construct_table_row(record: dict, columns_map: dict):
    """Flatten one aggregate dict into a table row using a schema.py *_COLUMNS map.

    Args:
        record: An aggregate dict (sample, sensor, or annotation).
        columns_map: Output column -> aggregate key, (key, index, ...) tuple
            for one component of a vector/matrix field, or '' for a column
            the caller fills in itself.

    Returns:
        One row dict with every column in columns_map; '' columns are None.
    """
    row = {}
    for col, key in columns_map.items():
        if len(key) > 0:
            if isinstance(key, tuple):
                row[col] = multiindex(record, key)
            else:
                row[col] = record[key]
        else:
            row[col] = None

    return row





