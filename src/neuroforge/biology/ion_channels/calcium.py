"""Calcium ion-channel model."""

from __future__ import annotations

from neuroforge.biology.ion_channels.base import IonChannelBase

__all__ = ["CalciumChannel"]


class CalciumChannel(IonChannelBase):
    """Zero-current calcium channel placeholder."""

    def current(self, voltage: object) -> float:
        """Return zero current."""
        _ = voltage
        return 0.0
