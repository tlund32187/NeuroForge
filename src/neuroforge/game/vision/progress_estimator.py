# pyright: basic
"""Estimate horizontal level progress from screen scrolling (no calibration).

SMB3 has no on-screen position meter, so forward progress is inferred from how
far the playfield scrolls. Each frame we reduce a band of the playfield to a 1-D
horizontal brightness profile and find the pixel shift that best aligns it to
the previous frame. Rightward camera motion (the world sliding left) accumulates
as forward progress; the reported ``x_progress`` tracks the furthest point
reached so backtracking neither adds nor removes credit.

Pure numpy on a small strip — cheap enough for the per-frame hot path.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from neuroforge.game.vision.digit_reader import to_grayscale

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = ["ScrollProgressEstimator"]


class ScrollProgressEstimator:
    """Accumulate rightward scroll into a normalised ``x_progress`` in [0, 1]."""

    def __init__(
        self,
        *,
        playfield: tuple[int, int, int, int] | None = None,
        hud_height: int = 40,
        band_rows: int = 48,
        search_px: int = 24,
        deadband_px: int = 1,
        level_length_px: float = 4000.0,
    ) -> None:
        self._playfield = playfield
        self._hud_height = int(hud_height)
        self._band_rows = int(band_rows)
        self._search = max(1, int(search_px))
        self._deadband = max(0, int(deadband_px))
        self._level_length = max(1.0, float(level_length_px))
        self._prev_profile: NDArray[np.float32] | None = None
        self._net_px = 0.0
        self._max_px = 0.0

    def reset(self) -> None:
        """Forget scroll history (call at the start of each episode/level)."""
        self._prev_profile = None
        self._net_px = 0.0
        self._max_px = 0.0

    @property
    def forward_px(self) -> float:
        """Furthest rightward scroll reached so far, in pixels."""
        return self._max_px

    def update(self, frame: NDArray[np.uint8]) -> float:
        """Ingest a frame and return the current ``x_progress`` in [0, 1]."""
        profile = self._profile(to_grayscale(frame))
        if self._prev_profile is None or profile.shape != self._prev_profile.shape:
            self._prev_profile = profile
            return self._x_progress()

        shift = self._best_shift(self._prev_profile, profile)
        self._prev_profile = profile
        if abs(shift) > self._deadband:
            self._net_px += float(shift)
            self._max_px = max(self._max_px, self._net_px)
        return self._x_progress()

    def _x_progress(self) -> float:
        return float(np.clip(self._max_px / self._level_length, 0.0, 1.0))

    def _profile(self, gray: NDArray[np.uint8]) -> NDArray[np.float32]:
        h, w = gray.shape
        if self._playfield is not None:
            px, py, pw, ph = self._playfield
            region = gray[py : py + ph, px : px + pw]
        else:
            # Whole width, vertically centred band of the non-HUD playfield.
            usable = max(1, h - self._hud_height)
            mid = usable // 2
            half = min(self._band_rows, usable) // 2
            top = max(0, mid - half)
            region = gray[top : top + max(1, 2 * half), 0:w]
        profile = region.astype(np.float32).mean(axis=0)
        return np.asarray(profile, dtype=np.float32)

    def _best_shift(
        self, prev: NDArray[np.float32], curr: NDArray[np.float32],
    ) -> int:
        n = prev.shape[0]
        best_d = 0
        best_cost = np.inf
        min_overlap = max(1, n // 2)
        for d in range(-self._search, self._search + 1):
            if d >= 0:
                p = prev[d:]
                c = curr[: n - d]
            else:
                p = prev[: n + d]
                c = curr[-d:]
            if p.shape[0] < min_overlap:
                continue
            cost = float(np.abs(p - c).mean())
            # Prefer smaller |shift| on ties to resist jitter.
            if cost < best_cost - 1e-6 or (
                abs(cost - best_cost) <= 1e-6 and abs(d) < abs(best_d)
            ):
                best_cost = cost
                best_d = d
        return best_d
