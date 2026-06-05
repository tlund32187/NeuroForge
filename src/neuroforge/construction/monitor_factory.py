"""Monitor factory built from construction registrations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from neuroforge.construction.composition_root import DEFAULT_HUB

if TYPE_CHECKING:
    from neuroforge.construction.registry import Registry

__all__ = ["MonitorFactory", "build_monitor_factory"]


class MonitorFactory:
    """Factory for observability monitor constructors."""

    def __init__(self, registry: Registry | None = None) -> None:
        self._registry = registry or DEFAULT_HUB.monitors

    def create(self, key: str, **kwargs: Any) -> Any:
        """Create a registered monitor."""
        return self._registry.create(key, **kwargs)

    def list_keys(self) -> list[str]:
        """Return registered monitor keys."""
        return self._registry.list_keys()


def build_monitor_factory() -> MonitorFactory:
    """Build a monitor factory from the default composition root."""
    return MonitorFactory(DEFAULT_HUB.monitors)
