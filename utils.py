import math

# Get skew of sample data timestamp from sample timestamp
def calculate_timestamp_skew_ms(sample_ts, sample_data_ts):
    # sample_ts and sample_data_ts are in microseconds (UNIX timestamp)
    # dividing by 1000 gives the difference in milliseconds
    return math.fabs(sample_data_ts - sample_ts) / 1000