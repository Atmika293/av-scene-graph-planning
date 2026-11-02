"""Single consolidated configuration for the whole data pipeline.

Every setting -- which dataset to read (NUSCENES_VERSION/NUSCENES_DATAROOT/
OUTPUT_ROOT, shared by every entry point), partition.py's own
execution-strategy knobs (BATCH_SIZE/MAX_IN_FLIGHT/NUM_CPUS), and the
benchmark harness's settings (BENCHMARK_*) -- is defined once in config.toml
at the repo root, deliberately outside this package: this module is a
generic loader (TOML + env-var overrides), not itself environment-specific
data, so it's fine for it to live in the installable library; the actual
values aren't.

Precedence per setting: an environment variable of the matching name, if
set, always wins; otherwise the value comes from config.toml.
    export NUSCENES_VERSION=v1.0-trainval
    export BATCH_SIZE=1
    export NUM_CPUS=4
"""

import os
import tomllib

with open("config.toml", "rb") as _fh:
    _config = tomllib.load(_fh)


def _get(section: str, key: str, env_var: str, cast=str, default=None):
    """Resolve one setting: env var override, else config.toml, else default.

    Args:
        section: config.toml table this setting lives under (e.g. 'dataset').
        key: Key within that table.
        env_var: Environment variable name that overrides this setting.
        cast: Type to coerce the env var's (always-string) value into.
        default: Used if the key is absent from config.toml entirely (e.g.
            NUM_CPUS, which is intentionally omitted to mean "unset").

    Returns:
        The resolved value.
    """
    if env_var in os.environ:
        return cast(os.environ[env_var])
    return _config.get(section, {}).get(key, default)


NUSCENES_VERSION = _get("dataset", "version", "NUSCENES_VERSION")
NUSCENES_DATAROOT = _get("dataset", "dataroot", "NUSCENES_DATAROOT")
OUTPUT_ROOT = _get("dataset", "output_root", "OUTPUT_ROOT")

BATCH_SIZE = _get("partition", "batch_size", "BATCH_SIZE", cast=int)
MAX_IN_FLIGHT = _get("partition", "max_in_flight", "MAX_IN_FLIGHT", cast=int)
# None means "let Ray auto-detect and use all available cores" (ray.init()'s own default).
NUM_CPUS = _get("partition", "num_cpus", "NUM_CPUS", cast=int, default=None)

BENCHMARK_WORKER_COUNTS = _get("benchmark", "worker_counts", "BENCHMARK_WORKER_COUNTS",
                               cast=lambda s: [int(x) for x in s.split(",")])
BENCHMARK_REPEATS = _get("benchmark", "repeats", "BENCHMARK_REPEATS", cast=int)
BENCHMARK_CPU_SAMPLE_INTERVAL = _get("benchmark", "cpu_sample_interval",
                                     "BENCHMARK_CPU_SAMPLE_INTERVAL", cast=float)
BENCHMARK_REPORT_DIR = _get("benchmark", "report_dir", "BENCHMARK_REPORT_DIR")
