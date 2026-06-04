"""Environment parsing helpers for SMB3 scripts."""

from __future__ import annotations

import os


def env_bool(name: str, default: bool) -> bool:
    """Read a boolean environment variable with common on/off spellings."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return default


def env_int(name: str, default: int, *, min_value: int | None = None) -> int:
    """Read an integer environment variable, falling back on invalid values."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    if min_value is not None and value < min_value:
        return default
    return value


def env_float(name: str, default: float, *, min_value: float | None = None) -> float:
    """Read a float environment variable, falling back on invalid values."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    if min_value is not None and value < min_value:
        return default
    return value
