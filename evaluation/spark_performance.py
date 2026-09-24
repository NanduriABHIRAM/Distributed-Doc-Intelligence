"""Measure distributed processing performance."""

from time import perf_counter


def evaluate_spark_performance(operation, item_count: int) -> dict:
    """Time a processing operation and report basic throughput."""
    started = perf_counter()
    operation()
    duration_seconds = perf_counter() - started
    return {
        "duration_seconds": round(duration_seconds, 4),
        "items_per_second": round(item_count / duration_seconds, 2) if duration_seconds else 0.0,
    }
