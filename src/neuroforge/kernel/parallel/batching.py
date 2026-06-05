"""Batching helpers for parallel execution."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

__all__ = ["batch_iterable"]

T = TypeVar("T")


def batch_iterable(values: Iterable[T], batch_size: int) -> Iterator[list[T]]:
    """Yield fixed-size batches from an iterable."""
    if batch_size <= 0:
        msg = "batch_size must be positive"
        raise ValueError(msg)
    batch: list[T] = []
    for value in values:
        batch.append(value)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch
