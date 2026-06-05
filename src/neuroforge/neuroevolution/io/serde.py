"""Shared JSON (de)serialisation primitives for evolution checkpoints.

One home for the small helpers the engine (writer) and ``checkpoints`` (reader)
both need, so they cannot drift apart.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["cast_json_object", "rng_state_from_json", "rng_state_to_json"]


def cast_json_object(value: Any) -> dict[str, Any]:
    """Validate that *value* decoded to a JSON object and key it by ``str``."""
    if not isinstance(value, dict):
        msg = "expected JSON object"
        raise ValueError(msg)
    # isinstance narrows Any->dict[Unknown] (pyright); naming the key/value types here
    # recovers them for pyright while staying a non-identity cast for mypy.
    mapping = cast("dict[str, Any]", value)
    return {str(key): item for key, item in mapping.items()}


def rng_state_to_json(value: Any) -> Any:
    """Convert Python ``random`` state tuples to JSON-safe nested lists."""
    if isinstance(value, (tuple, list)):
        seq = cast("Sequence[Any]", value)
        return [rng_state_to_json(item) for item in seq]
    return value


def rng_state_from_json(value: Any) -> Any:
    """Convert JSON-loaded lists back to tuples for ``random.setstate``."""
    if isinstance(value, list):
        items = cast("Sequence[Any]", value)
        return tuple(rng_state_from_json(item) for item in items)
    return value
