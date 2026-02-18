"""Tests for rate encoder and decoder.

Math-predictive: we compute expected values analytically first, then verify.
"""

from __future__ import annotations

import pytest
import torch

from neuroforge.encoding.decode import RateDecoder, RateDecoderParams
from neuroforge.encoding.rate import RateEncoder, RateEncoderParams

# ── Encoder tests ────────────────────────────────────────────────────


class TestRateEncoder:
    """Verify rate encoder math."""

    def test_encode_one(self) -> None:
        """Input 1.0 → amplitude."""
        enc = RateEncoder(RateEncoderParams(amplitude=50.0))
        expected = enc.predict_drive(1.0)
        assert expected == 50.0

        result = enc.encode(torch.tensor([1.0], dtype=torch.float64))
        assert result.item() == pytest.approx(expected)

    def test_encode_zero(self) -> None:
        """Input 0.0 → 0."""
        enc = RateEncoder(RateEncoderParams(amplitude=50.0))
        expected = enc.predict_drive(0.0)
        assert expected == 0.0

        result = enc.encode(torch.tensor([0.0], dtype=torch.float64))
        assert result.item() == pytest.approx(expected)

    def test_encode_half(self) -> None:
        """Input 0.5 → amplitude / 2."""
        enc = RateEncoder(RateEncoderParams(amplitude=40.0))
        expected = enc.predict_drive(0.5)
        assert expected == 20.0

        result = enc.encode(torch.tensor([0.5], dtype=torch.float64))
        assert result.item() == pytest.approx(expected)

    def test_encode_batch(self) -> None:
        """Batch encoding: [0, 0.5, 1.0] with amplitude 60."""
        enc = RateEncoder(RateEncoderParams(amplitude=60.0))
        values = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)
        result = enc.encode(values)

        expected = [enc.predict_drive(v) for v in [0.0, 0.5, 1.0]]
        for i, exp in enumerate(expected):
            assert result[i].item() == pytest.approx(exp)

    def test_encode_from_float(self) -> None:
        """Encoding a plain float works."""
        enc = RateEncoder(RateEncoderParams(amplitude=50.0))
        result = enc.encode(0.7)
        assert result.item() == pytest.approx(35.0)

    def test_default_params(self) -> None:
        """Default amplitude is 50.0."""
        enc = RateEncoder()
        assert enc.params.amplitude == 50.0


# ── Decoder tests ────────────────────────────────────────────────────


class TestRateDecoder:
    """Verify rate decoder math."""

    def test_binary_above_threshold(self) -> None:
        """Spike count >= threshold → True."""
        dec = RateDecoder(RateDecoderParams(threshold=5))
        assert dec.predict_binary(5) is True
        assert dec.predict_binary(10) is True

        counts = torch.tensor([5, 10])
        result = dec.decode_binary(counts)
        assert result[0].item() is True
        assert result[1].item() is True

    def test_binary_below_threshold(self) -> None:
        """Spike count < threshold → False."""
        dec = RateDecoder(RateDecoderParams(threshold=5))
        assert dec.predict_binary(4) is False
        assert dec.predict_binary(0) is False

        counts = torch.tensor([4, 0])
        result = dec.decode_binary(counts)
        assert result[0].item() is False
        assert result[1].item() is False

    def test_rate_computation(self) -> None:
        """Firing rate = spike_count / window_steps."""
        dec = RateDecoder(RateDecoderParams(window_steps=100))
        assert dec.predict_rate(50) == pytest.approx(0.5)
        assert dec.predict_rate(0) == pytest.approx(0.0)
        assert dec.predict_rate(100) == pytest.approx(1.0)

        counts = torch.tensor([50, 0, 100])
        rates = dec.decode_rate(counts)
        assert rates[0].item() == pytest.approx(0.5)
        assert rates[1].item() == pytest.approx(0.0)
        assert rates[2].item() == pytest.approx(1.0)

    def test_default_params(self) -> None:
        """Default window=50, threshold=5."""
        dec = RateDecoder()
        assert dec.params.window_steps == 50
        assert dec.params.threshold == 5
