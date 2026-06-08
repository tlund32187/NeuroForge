"""Tests for the retinal encoder (Layer A0).

The core claim is *invariance*: the same scene structure must encode almost the
same way across a palette/brightness/contrast change (so "a hole is a hole" in
world 2), while a genuine change in structure must encode differently. These
tests build synthetic scenes and measure that with cosine similarity — no
emulator, no labels in training.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch

from neuroforge.contracts.applications.games import ScreenFrame
from neuroforge.perception.vision.encoding import RetinaEncoder, RetinaEncoderConfig

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from torch import Tensor

pytest.importorskip("torch")

_H, _W = 56, 64


def _scene(hole_col: int, *, block: bool = True) -> NDArray[np.float32]:
    """A luminance map with a floor, a hole in it, and an optional floating block."""
    luma = np.full((_H, _W), 0.20, dtype=np.float32)  # background
    luma[44:, :] = 0.80                                # floor
    luma[44:, hole_col : hole_col + 10] = 0.20         # a hole in the floor
    if block:
        luma[20:28, 30:38] = 0.60                      # a floating block
    return luma


def _frame(
    luma: NDArray[np.float32],
    *,
    contrast: float = 1.0,
    bright: float = 0.0,
    gain: tuple[float, float, float] = (1.0, 1.0, 1.0),
    bias: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> ScreenFrame:
    """Render a luminance map to an RGB frame under a (monotonic) palette."""
    adj = np.clip(luma * contrast + bright, 0.0, 1.0)
    rgb = np.stack(
        [np.clip(adj * gain[c] + bias[c], 0.0, 1.0) for c in range(3)], axis=-1,
    )
    data = (rgb * 255.0).astype(np.uint8)
    return ScreenFrame(width=_W, height=_H, channels=3, data=data.tobytes())


def _cos(a: Tensor, b: Tensor) -> float:
    return float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-9))


@pytest.mark.unit
def test_input_size_and_output_shape() -> None:
    enc = RetinaEncoder(RetinaEncoderConfig(out_h=28, out_w=32))
    assert enc.input_size == 28 * 32 * 2
    drive = enc.to_drive(_frame(_scene(20)))
    assert tuple(drive.shape) == (28 * 32 * 2,)
    assert float(drive.min()) >= 0.0  # ON/OFF channels are non-negative currents


@pytest.mark.unit
def test_same_structure_survives_palette_change() -> None:
    enc = RetinaEncoder()
    luma = _scene(20)
    gray = enc.to_drive(_frame(luma))
    # World-2-like: dimmer, lower contrast, recoloured — same structure.
    recolored = enc.to_drive(
        _frame(luma, contrast=0.6, bright=0.25, gain=(1.0, 0.7, 0.4), bias=(0.0, 0.1, 0.3)),
    )
    assert _cos(gray, recolored) > 0.8


@pytest.mark.unit
def test_palette_change_preserves_more_than_content_change() -> None:
    enc = RetinaEncoder()
    base = enc.to_drive(_frame(_scene(20)))
    palette = enc.to_drive(
        _frame(_scene(20), contrast=0.6, bright=0.25, gain=(1.0, 0.7, 0.4)),
    )
    content = enc.to_drive(_frame(_scene(45, block=False)))  # hole moved, block gone
    assert _cos(base, palette) > _cos(base, content)


@pytest.mark.unit
def test_brightness_shift_barely_changes_code() -> None:
    enc = RetinaEncoder()
    luma = _scene(20)
    a = enc.to_drive(_frame(luma))
    b = enc.to_drive(_frame(luma, bright=0.15))
    assert _cos(a, b) > 0.9


@pytest.mark.unit
def test_contrast_magnitude_is_normalized_away() -> None:
    enc = RetinaEncoder()
    luma = _scene(20)
    full = enc.to_drive(_frame(luma, contrast=1.0))
    half = enc.to_drive(_frame(luma, contrast=0.5))
    assert _cos(full, half) > 0.95  # divisive normalization removes contrast scale


@pytest.mark.unit
def test_blank_frame_gives_near_zero_drive() -> None:
    enc = RetinaEncoder(RetinaEncoderConfig(amplitude=10.0))
    blank = ScreenFrame(width=_W, height=_H, channels=3, data=bytes(_W * _H * 3))
    drive = enc.to_drive(blank)
    assert float(drive.abs().max()) < 1.0  # no contrast => essentially no drive


@pytest.mark.unit
def test_reset_is_noop_and_repeatable() -> None:
    enc = RetinaEncoder()
    luma = _scene(20)
    first = enc.to_drive(_frame(luma))
    enc.reset()
    second = enc.to_drive(_frame(luma))
    assert torch.allclose(first, second)
