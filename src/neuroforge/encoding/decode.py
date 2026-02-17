"""Rate-based spike decoder.

Converts spike counts over a time window into output values. For the
logic gate task, we use a threshold on the spike count to decode
binary output:

    output = 1 if spike_count >= threshold else 0

Also provides a continuous decode: spike_rate = spike_count / window_steps.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = ["RateDecoder", "RateDecoderParams"]


@dataclass(frozen=True, slots=True)
class RateDecoderParams:
    """Parameters for the rate decoder.

    Attributes
    ----------
    window_steps:
        Number of simulation steps in the decoding window.
    threshold:
        Spike count threshold for binary decode.
    """

    window_steps: int = 50
    threshold: int = 5


class RateDecoder:
    """Decode spike trains into output values.

    Accumulates spikes over a window and decodes:
    - ``decode_binary(spike_counts)`` → 0 or 1 per neuron
    - ``decode_rate(spike_counts)`` → firing rate per neuron
    """

    def __init__(self, params: RateDecoderParams | None = None) -> None:
        self.params = params or RateDecoderParams()

    def decode_binary(self, spike_counts: Any) -> Any:
        """Decode spike counts to binary output.

        Parameters
        ----------
        spike_counts:
            Integer tensor of spike counts per neuron.

        Returns
        -------
        Tensor:
            Boolean tensor — True if spike_count >= threshold.
        """
        return spike_counts >= self.params.threshold

    def decode_rate(self, spike_counts: Any) -> Any:
        """Decode spike counts to firing rates.

        Parameters
        ----------
        spike_counts:
            Integer tensor of spike counts per neuron.

        Returns
        -------
        Tensor:
            Float tensor of rates (spikes / window_steps).
        """
        return spike_counts.float() / self.params.window_steps

    # ── math helpers ────────────────────────────────────────────────

    def predict_binary(self, spike_count: int) -> bool:
        """Predict binary output for a given spike count."""
        return spike_count >= self.params.threshold

    def predict_rate(self, spike_count: int) -> float:
        """Predict firing rate for a given spike count."""
        return spike_count / self.params.window_steps
