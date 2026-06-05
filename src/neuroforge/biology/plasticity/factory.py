"""Factory for plasticity rule constructors."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = ["PlasticityRuleFactory"]


class PlasticityRuleFactory:
    """Small registry-backed factory for plasticity rules."""

    def __init__(self) -> None:
        self._constructors: dict[str, Callable[..., Any]] = {}

    def register(self, name: str, constructor: Callable[..., Any]) -> None:
        """Register a plasticity rule constructor."""
        self._constructors[name] = constructor

    def create(self, name: str, **kwargs: Any) -> Any:
        """Create a registered plasticity rule."""
        try:
            constructor = self._constructors[name]
        except KeyError as exc:
            msg = f"unknown plasticity rule: {name!r}"
            raise KeyError(msg) from exc
        return constructor(**kwargs)
