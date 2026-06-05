"""Factory contracts."""

from __future__ import annotations

from typing import Protocol, TypeVar, runtime_checkable

__all__ = ["IFactory"]

T_co = TypeVar("T_co", covariant=True)


@runtime_checkable
class IFactory(Protocol[T_co]):
    """Protocol for factories that create configured objects."""

    def create(self, *args: object, **kwargs: object) -> T_co:
        """Create an object."""
        ...
