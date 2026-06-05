"""Controller state container."""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["ControllerState"]


def _empty_buttons() -> dict[str, bool]:
    return {}


@dataclass(frozen=True, slots=True)
class ControllerState:
    """Generic named-button controller state."""

    buttons: dict[str, bool] = field(default_factory=_empty_buttons)
