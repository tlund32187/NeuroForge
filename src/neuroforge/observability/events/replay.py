"""Replay helpers for persisted event streams."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

__all__ = ["iter_event_records", "load_event_records"]


def iter_event_records(path: Path) -> Iterator[dict[str, Any]]:
    """Yield decoded event records from an NDJSON event file."""
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            if isinstance(record, dict):
                yield record


def load_event_records(path: Path) -> list[dict[str, Any]]:
    """Load all event records from an NDJSON event file."""
    return list(iter_event_records(path))
