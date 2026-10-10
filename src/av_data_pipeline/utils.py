import math

VISIBILITY_RANGES = {
    '1': [0, 40],
    '2': [40, 60],
    '3': [60, 80],
    '4': [80, 100],
}

# Get skew of sample data timestamp from sample timestamp
def calculate_timestamp_skew_ms(sample_ts, sample_data_ts):
    # sample_ts and sample_data_ts are in microseconds (UNIX timestamp)
    # dividing by 1000 gives the difference in milliseconds
    return math.fabs(sample_data_ts - sample_ts) / 1000

# Get the visibility range for a sample annotation
def visibility_range(token: str) -> list[int, int]:
    """Map a sample_annotation visibility_token to its (min%, max%) range."""
    return VISIBILITY_RANGES[token]

# Extract the key from the record and assign it to the aggregate key
def extract_key_values(extract_dict: dict, record, keys_map: dict[str, str]):
    for col, key in keys_map.items():
        if len(key) > 0: # Check if the key is not empty
            extract_dict[col] = record[key]

    return extract_dict

def multiindex(obj: object, multi_index: tuple):
    for idx in multi_index:
        if obj is None or len(obj) > 0:
            return None

        obj = obj[idx]

    return obj

def construct_table_row(record: dict, columns_map: dict):
    row = {}
    for col, key in columns_map:
        if len(key) > 0:
            if isinstance(key, tuple):
                row[col] = multiindex(record, key)
            else:
                row[col] = record[key]
        else:
            row[col] = None

    return row





