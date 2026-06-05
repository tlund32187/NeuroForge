"""Factory for neuron model constructors."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = ["NeuronFactory"]


class NeuronFactory:
    """Small registry-backed factory for neuron models."""

    def __init__(self) -> None:
        self._constructors: dict[str, Callable[..., Any]] = {}

    def register(self, name: str, constructor: Callable[..., Any]) -> None:
        """Register a neuron constructor by name."""
        self._constructors[name] = constructor

    def create(self, name: str, **kwargs: Any) -> Any:
        """Create a registered neuron model."""
        try:
            constructor = self._constructors[name]
        except KeyError as exc:
            msg = f"unknown neuron model: {name!r}"
            raise KeyError(msg) from exc
        return constructor(**kwargs)

    def list_keys(self) -> tuple[str, ...]:
        """Return registered constructor keys."""
        return tuple(sorted(self._constructors))
