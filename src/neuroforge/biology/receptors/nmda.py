"""NMDA receptor model."""

from __future__ import annotations

from neuroforge.biology.receptors.base import ReceptorModelBase

__all__ = ["NMDAReceptor"]


class NMDAReceptor(ReceptorModelBase):
    """Identity-current NMDA receptor placeholder."""

    def current(self, activation: object) -> object:
        """Return activation as current."""
        return activation
