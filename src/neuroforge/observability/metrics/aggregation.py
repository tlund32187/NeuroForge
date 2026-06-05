"""Metric aggregation and normalization helpers."""

from __future__ import annotations

from typing import Any

__all__ = ["bytes_to_mb", "merge_metric_payloads"]


def bytes_to_mb(value: Any) -> float | None:
    """Convert a byte count to MiB, returning None for non-numeric values."""
    if value in (None, ""):
        return None
    if isinstance(value, bool):
        return None
    try:
        return float(value) / (1024.0 * 1024.0)
    except (TypeError, ValueError):
        return None


def merge_metric_payloads(*payloads: dict[str, float]) -> dict[str, float]:
    """Merge metric dictionaries, ignoring empty payloads."""
    merged: dict[str, float] = {}
    for payload in payloads:
        merged.update(payload)
    return merged
