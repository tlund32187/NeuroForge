"""Tests for motion figure-ground (Layer A3).

Core claim: with the background scrolling and a sprite moving differently,
scroll-compensated differencing must light up the *sprite* (figure) while the
scrolling background cancels (ground). Built on synthetic frames — no emulator.
"""

from __future__ import annotations

import numpy as np
import pytest

from neuroforge.contracts.applications.games import ScreenFrame

pytest.importorskip("torch")

from neuroforge.perception.vision.encoding import MotionFigureGround  # noqa: E402

_H, _W = 112, 128
_RNG = np.random.default_rng(0)
_COLS = _RNG.uniform(0.1, 0.6, size=_W).astype(np.float32)  # textured background columns


def _frame(bg_shift: int) -> ScreenFrame:
    """A scrolling column-textured background with a screen-fixed bright sprite."""
    img = np.tile(_COLS, (_H, 1))
    img = np.roll(img, bg_shift, axis=1)
    img[40:72, 80:96] = 0.95                      # sprite (fixed in screen space)
    rgb = np.stack([(np.clip(img, 0, 1) * 255).astype(np.uint8)] * 3, axis=-1)
    return ScreenFrame(width=_W, height=_H, channels=3, data=rgb.tobytes())


@pytest.mark.unit
def test_first_frame_has_no_saliency() -> None:
    mfg = MotionFigureGround()
    sal = mfg.saliency(_frame(0))
    assert float(sal.abs().max()) == 0.0


@pytest.mark.unit
def test_moving_sprite_pops_against_scrolling_background() -> None:
    mfg = MotionFigureGround()
    mfg.saliency(_frame(0))           # reference frame
    sal = mfg.saliency(_frame(8))     # background scrolled 8px; sprite stayed put

    # Downsampled (4x): sprite ~rows 10:18, cols 20:24 (plus displaced edges).
    sprite = float(sal[10:18, 18:26].mean())
    background = float(sal[10:18, 0:8].mean())
    assert sprite > 5.0 * background + 1e-3   # the figure stands out from the ground
    assert float(sal[10:18, 18:26].max()) > 0.5


@pytest.mark.unit
def test_reset_drops_the_motion_reference() -> None:
    mfg = MotionFigureGround()
    mfg.saliency(_frame(0))
    mfg.saliency(_frame(8))
    mfg.reset()
    sal = mfg.saliency(_frame(0))     # first frame again after reset
    assert float(sal.abs().max()) == 0.0
