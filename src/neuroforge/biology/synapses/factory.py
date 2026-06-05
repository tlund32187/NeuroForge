"""Factory for synapse model constructors."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = ["SynapseFactory"]


class SynapseFactory:
    """Small registry-backed factory for synapse models."""

    def __init__(self) -> None:
        self._constructors: dict[str, Callable[..., Any]] = {}

    def register(self, name: str, constructor: Callable[..., Any]) -> None:
        """Register a synapse constructor by name."""
        self._constructors[name] = constructor

    def create(self, name: str, **kwargs: Any) -> Any:
        """Create a registered synapse model."""
        try:
            constructor = self._constructors[name]
        except KeyError as exc:
            msg = f"unknown synapse model: {name!r}"
            raise KeyError(msg) from exc
        return constructor(**kwargs)

    def list_keys(self) -> tuple[str, ...]:
        """Return registered constructor keys."""
        return tuple(sorted(self._constructors))
