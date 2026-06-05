"""Episode runner protocols for agents."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

__all__ = ["IEpisodeRunner"]


@runtime_checkable
class IEpisodeRunner(Protocol):
    """Runs one environment episode for an agent."""

    def run_episode(self) -> object:
        """Run an episode and return implementation-defined results."""
        ...
