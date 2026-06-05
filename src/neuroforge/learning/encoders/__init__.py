"""Input encoders used by learning workflows."""

from neuroforge.learning.encoders.factory import build_encoder_registry
from neuroforge.learning.encoders.rate import RateEncoder, RateEncoderParams

__all__ = [
    "RateEncoder",
    "RateEncoderParams",
    "build_encoder_registry",
]
