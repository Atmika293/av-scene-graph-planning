"""How many CPUs this process may actually use, and how busy they are, inside a container too.

In a container (a Runpod pod, say), os.cpu_count() and /proc/stat describe the
whole host, not the share the container is given. Two cgroup mechanisms set
that share:

    CPU set   the container may only run on certain cores. Python 3.13's
              os.process_cpu_count() already respects this (like `nproc`);
              Ray's own detection misses it under cgroup v2.
    quota     the container may use at most N cores' worth of time
              (cgroup v2 cpu.max, or v1 cpu.cfs_quota_us / cpu.cfs_period_us).

available_cpus() applies both. cpu_usage_seconds() reads the CPU time used by
every process in this cgroup (the container), so utilization measured with it
is the container's own, not the host's. Off Linux (e.g. macOS) there are no
cgroup files: available_cpus() falls back to the process's core count, and
cpu_usage_seconds() returns None so callers can fall back to psutil.

The file paths are module constants so they can be pointed at test files.
"""

import math
import os

CGROUP_V2_CPU_MAX = "/sys/fs/cgroup/cpu.max"
CGROUP_V2_CPU_STAT = "/sys/fs/cgroup/cpu.stat"
CGROUP_V1_QUOTA = "/sys/fs/cgroup/cpu/cpu.cfs_quota_us"
CGROUP_V1_PERIOD = "/sys/fs/cgroup/cpu/cpu.cfs_period_us"
CGROUP_V1_USAGE = "/sys/fs/cgroup/cpuacct/cpuacct.usage"


def _read(path: str) -> str | None:
    """A cgroup file's contents, stripped, or None if it doesn't exist or can't be read."""
    try:
        with open(path) as fp:
            return fp.read().strip()
    except OSError:
        return None


def cpu_quota() -> float | None:
    """The cgroup's CPU quota in cores (e.g. 16.0), or None if there's no quota."""
    v2 = _read(CGROUP_V2_CPU_MAX)
    if v2 is not None:
        quota, _, period = v2.partition(" ")
        if quota.isdigit() and period.isdigit() and int(period) > 0:
            return int(quota) / int(period)
        return None  # "max": no quota
    quota, period = _read(CGROUP_V1_QUOTA), _read(CGROUP_V1_PERIOD)
    if quota and period and quota.lstrip("-").isdigit() and period.isdigit():
        if int(quota) > 0 and int(period) > 0:
            return int(quota) / int(period)
    return None  # -1 or missing: no quota


def available_cpus() -> int:
    """The number of CPUs this process can actually use: its CPU set, capped by any quota.

    A fractional quota rounds down (but never below 1): 15.5 cores' worth of time can't keep
    16 workers busy.
    """
    cpus = os.process_cpu_count() or os.cpu_count() or 1
    quota = cpu_quota()
    if quota is not None:
        cpus = min(cpus, max(1, math.floor(quota)))
    return cpus


def cpu_usage_seconds() -> float | None:
    """Total CPU seconds used by every process in this cgroup so far, or None off Linux.

    Sample it twice: (difference in usage) / (difference in wall time x available_cpus())
    is the share of the container's CPUs that were busy in between.
    """
    v2 = _read(CGROUP_V2_CPU_STAT)
    if v2 is not None:
        for line in v2.splitlines():
            key, _, value = line.partition(" ")
            if key == "usage_usec" and value.isdigit():
                return int(value) / 1e6
        return None
    v1 = _read(CGROUP_V1_USAGE)
    if v1 is not None and v1.isdigit():
        return int(v1) / 1e9
    return None
