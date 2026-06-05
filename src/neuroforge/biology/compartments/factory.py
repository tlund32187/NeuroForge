"""Factory for compartment model constructors."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = ["CompartmentFactory"]


class CompartmentFactory:
    """Small registry-backed factory for compartment models."""

    def __init__(self) -> None:
        self._constructors: dict[str, Callable[..., Any]] = {}

    def register(self, name: str, constructor: Callable[..., Any]) -> None:
        """Register a compartment constructor by name."""
        self._constructors[name] = constructor

    def create(self, name: str, **kwargs: Any) -> Any:
        """Create a registered compartment model."""
        try:
            constructor = self._constructors[name]
        except KeyError as exc:
            msg = f"unknown compartment model: {name!r}"
            raise KeyError(msg) from exc
        return constructor(**kwargs)

    def list_keys(self) -> tuple[str, ...]:
        """Return registered constructor keys."""
        return tuple(sorted(self._constructors))
