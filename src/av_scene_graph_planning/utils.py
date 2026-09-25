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

# Extract the key from the record and assign it to the aggregate column
def extract_key_to_column(extract_dict: dict, record, col2key: dict[str, str]):
    for col, key in col2key.items():
        if len(key) > 0: # Check if the key is not empty
            extract_dict[col] = record[key]

    return extract_dict