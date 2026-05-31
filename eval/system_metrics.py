from __future__ import annotations

import sys


def get_current_memory_mb() -> float | None:
    try:
        import psutil
    except ImportError:
        return get_peak_memory_mb_from_resource()
    return psutil.Process().memory_info().rss / (1024 * 1024)


def get_peak_memory_mb_from_resource() -> float | None:
    try:
        import resource
    except ImportError:
        return None

    usage = resource.getrusage(resource.RUSAGE_SELF)
    max_rss = float(usage.ru_maxrss)
    if max_rss <= 0:
        return None
    if sys.platform == "darwin":
        return max_rss / (1024 * 1024)
    return max_rss / 1024
