"""Leak ion-channel model."""

from __future__ import annotations

from dataclasses import dataclass

from neuroforge.biology.ion_channels.base import IonChannelBase

__all__ = ["LeakChannel"]


@dataclass(frozen=True, slots=True)
class LeakChannel(IonChannelBase):
    """Linear leak current model."""

    conductance: float = 0.0
    reversal: float = 0.0

    def current(self, voltage: object) -> object:
        """Compute leak current for scalar or tensor voltage."""
        return self.conductance * (self.reversal - voltage)  # type: ignore[operator]
