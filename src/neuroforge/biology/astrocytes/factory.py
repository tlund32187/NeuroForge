"""Factory for astrocyte model constructors."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = ["AstrocyteFactory"]


class AstrocyteFactory:
    """Small registry-backed factory for astrocyte models."""

    def __init__(self) -> None:
        self._constructors: dict[str, Callable[..., Any]] = {}

    def register(self, name: str, constructor: Callable[..., Any]) -> None:
        """Register an astrocyte constructor."""
        self._constructors[name] = constructor

    def create(self, name: str, **kwargs: Any) -> Any:
        """Create a registered astrocyte model."""
        try:
            constructor = self._constructors[name]
        except KeyError as exc:
            msg = f"unknown astrocyte model: {name!r}"
            raise KeyError(msg) from exc
        return constructor(**kwargs)
