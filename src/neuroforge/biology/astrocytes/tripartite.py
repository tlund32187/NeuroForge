"""Tripartite synapse astrocyte model."""

from __future__ import annotations

from typing import TYPE_CHECKING

from neuroforge.biology.astrocytes.base import AstrocyteModelBase

if TYPE_CHECKING:
    from neuroforge.biology.astrocytes.state import AstrocyteState

__all__ = ["TripartiteAstrocyteModel"]


class TripartiteAstrocyteModel(AstrocyteModelBase):
    """Minimal tripartite astrocyte model."""

    def step(self, state: AstrocyteState, dt: float) -> AstrocyteState:
        """Advance astrocyte state one step."""
        _ = dt
        return state
