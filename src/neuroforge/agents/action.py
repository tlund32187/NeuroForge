"""Agent action containers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = ["AgentAction"]


def _empty_action_payload() -> dict[str, Any]:
    return {}


@dataclass(frozen=True, slots=True)
class AgentAction:
    """Action selected by an agent policy or actuator."""

    data: dict[str, Any] = field(default_factory=_empty_action_payload)
