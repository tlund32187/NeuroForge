"""Execution policy configuration."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["ExecutionPolicy"]


@dataclass(frozen=True, slots=True)
class ExecutionPolicy:
    """Basic execution policy for work scheduling."""

    parallel: bool = False
    batch_size: int = 1
