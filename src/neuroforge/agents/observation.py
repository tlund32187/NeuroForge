"""Agent observation containers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = ["AgentObservation"]


def _empty_observation_payload() -> dict[str, Any]:
    return {}


@dataclass(frozen=True, slots=True)
class AgentObservation:
    """Structured observation consumed by an agent brain or policy."""

    data: dict[str, Any] = field(default_factory=_empty_observation_payload)
    step: int = 0
    t: float = 0.0
