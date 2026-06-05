"""Worker pool configuration."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["WorkerPoolConfig"]


@dataclass(frozen=True, slots=True)
class WorkerPoolConfig:
    """Configuration for CPU worker pools."""

    max_workers: int = 1
