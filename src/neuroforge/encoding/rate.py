"""Rate-based spike encoder.

Converts scalar values in [0, 1] to current amplitudes that produce
firing rates proportional to the input value.

For the logic gate task, the encoder maps binary inputs (0 or 1) to
a constant current drive over a presentation window:
    drive = value * amplitude

This is the simplest encoding: high value → high drive → high spike rate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = ["RateEncoder", "RateEncoderParams"]


@dataclass(frozen=True, slots=True)
class RateEncoderParams:
    """Parameters for the rate encoder.

    Attributes
    ----------
    amplitude:
        Maximum current amplitude (drive when input=1.0).
    """

    amplitude: float = 50.0


class RateEncoder:
    """Encode scalar values as constant drive currents.

    ``encode(values)`` returns a tensor of drive currents where each
    element is ``value * amplitude``.
    """

    def __init__(self, params: RateEncoderParams | None = None) -> None:
        self.params = params or RateEncoderParams()

    def encode(self, values: Any) -> Any:
        """Encode input values to drive currents.

        Parameters
        ----------
        values:
            Tensor or float in [0, 1].

        Returns
        -------
        Tensor:
            Drive currents (same shape as *values*).
        """
        from neuroforge.core.torch_utils import require_torch

        torch = require_torch()
        if not isinstance(values, torch.Tensor):
            values = torch.tensor(values, dtype=torch.float64)
        return values * self.params.amplitude

    # ── math helper ─────────────────────────────────────────────────

    def predict_drive(self, value: float) -> float:
        """Predict the drive current for a given input value."""
        return value * self.params.amplitude
