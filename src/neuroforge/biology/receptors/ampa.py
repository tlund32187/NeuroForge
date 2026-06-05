"""AMPA receptor model."""

from __future__ import annotations

from neuroforge.biology.receptors.base import ReceptorModelBase

__all__ = ["AMPAReceptor"]


class AMPAReceptor(ReceptorModelBase):
    """Identity-current AMPA receptor placeholder."""

    def current(self, activation: object) -> object:
        """Return activation as current."""
        return activation
