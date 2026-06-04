# pyright: basic
"""Shared JSON (de)serialisation primitives for evolution checkpoints.

One home for the small helpers the engine (writer) and ``checkpoints`` (reader)
both need, so they cannot drift apart.
"""

from __future__ import annotations

from typing import Any

__all__ = ["cast_json_object", "rng_state_from_json", "rng_state_to_json"]


def cast_json_object(value: Any) -> dict[str, Any]:
    """Validate that *value* decoded to a JSON object and key it by ``str``."""
    if not isinstance(value, dict):
        msg = "expected JSON object"
        raise ValueError(msg)
    return {str(key): item for key, item in value.items()}


def rng_state_to_json(value: Any) -> Any:
    """Convert Python ``random`` state tuples to JSON-safe nested lists."""
    if isinstance(value, (tuple, list)):
        return [rng_state_to_json(item) for item in value]
    return value


def rng_state_from_json(value: Any) -> Any:
    """Convert JSON-loaded lists back to tuples for ``random.setstate``."""
    if isinstance(value, list):
        return tuple(rng_state_from_json(item) for item in value)
    return value
