"""Output decoders and readout heads used by learning workflows."""

from neuroforge.learning.readouts.factory import build_readout_registry
from neuroforge.learning.readouts.rate_decoder import RateDecoder, RateDecoderParams
from neuroforge.learning.readouts.spike_count import ReadoutResult, SpikeCountReadout

__all__ = [
    "RateDecoder",
    "RateDecoderParams",
    "ReadoutResult",
    "SpikeCountReadout",
    "build_readout_registry",
]
