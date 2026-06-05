"""Motion figure-ground (Layer A3): separate moving sprites from the background.

The organic object-discovery cue. In a side-scroller the background scrolls
*coherently* with the camera, while sprites (enemies, items, the player) move
*differently*. So if we estimate the global scroll and **cancel it** â€” warp the
previous frame by that shift and difference it with the current frame â€” the
background subtracts to ~zero and only the things that moved differently survive
as a **figure-ground / object-saliency map**.

This mirrors the dorsal/magnocellular motion pathway (direction-selective cells
that respond to motion relative to self-motion). It needs no labels and no
prepopulated shapes, and it feeds attention / object-attributed credit later
(Layer D). The global shift is estimated the same way :class:`ScrollProgressEstimator`
does it (1-D horizontal profile alignment), so the two stay consistent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from neuroforge.kernel.torch_utils import require_torch, resolve_device_dtype

__all__ = ["MotionFigureGround", "MotionFigureGroundConfig"]


@dataclass(frozen=True, slots=True)
class MotionFigureGroundConfig:
    """Configuration for :class:`MotionFigureGround`."""

    out_h: int = 28
    out_w: int = 32
    search_px: int = 6      # max global horizontal scroll to search for (downsampled px)
    device: str = "cpu"
    dtype: str = "float32"

    def __post_init__(self) -> None:
        if self.out_h <= 0 or self.out_w <= 0:
            msg = "MotionFigureGroundConfig out_h/out_w must be > 0"
            raise ValueError(msg)
        if self.search_px < 1:
            msg = "MotionFigureGroundConfig.search_px must be >= 1"
            raise ValueError(msg)


class MotionFigureGround:
    """Scroll-compensated frame differencing â†’ a per-pixel object-saliency map."""

    def __init__(self, config: MotionFigureGroundConfig | None = None) -> None:
        self._cfg = config or MotionFigureGroundConfig()
        self._torch = require_torch()
        self._dev, self._dtype = resolve_device_dtype(self._cfg.device, self._cfg.dtype)
        self._gray_w = self._torch.tensor([0.299, 0.587, 0.114], dtype=self._dtype).view(3, 1, 1)
        self._prev: Any = None

    def reset(self) -> None:
        """Forget the previous frame (call at episode start)."""
        self._prev = None

    def saliency(self, frame: Any) -> Any:
        """Return an object-saliency map ``[out_h, out_w]`` in [0, 1].

        Zeros on the first frame after a reset (no motion reference yet).
        """
        torch = self._torch
        gray = self._luminance(frame)
        if self._prev is None or self._prev.shape != gray.shape:
            self._prev = gray
            return torch.zeros_like(gray)

        shift = -self._best_shift(self._prev.mean(dim=0), gray.mean(dim=0))
        aligned_prev = torch.roll(self._prev, shifts=shift, dims=1)
        residual = (gray - aligned_prev).abs()
        # Columns wrapped by the roll are invalid â€” they aren't a real comparison.
        if shift > 0:
            residual[:, :shift] = 0.0
        elif shift < 0:
            residual[:, shift:] = 0.0

        self._prev = gray
        peak = residual.max()
        return residual / (peak + 1e-6)

    #

    def _luminance(self, frame: Any) -> Any:
        torch = self._torch
        raw = torch.frombuffer(bytearray(frame.data), dtype=torch.uint8)
        img = raw.reshape(frame.height, frame.width, frame.channels).to(self._dtype) / 255.0
        chw = img.permute(2, 0, 1)
        if frame.channels >= 3:
            gray = (chw[:3] * self._gray_w).sum(dim=0, keepdim=True)
        else:
            gray = chw[:1]
        resized = torch.nn.functional.interpolate(
            gray[None], size=(self._cfg.out_h, self._cfg.out_w), mode="area",
        )
        return resized[0, 0].to(self._dev)

    def _best_shift(self, prev_profile: Any, curr_profile: Any) -> int:
        n = prev_profile.shape[0]
        min_overlap = max(1, n // 2)
        best_d, best_cost = 0, float("inf")
        for d in range(-self._cfg.search_px, self._cfg.search_px + 1):
            if d >= 0:
                p, c = prev_profile[d:], curr_profile[: n - d]
            else:
                p, c = prev_profile[:n + d], curr_profile[-d:]
            if p.shape[0] < min_overlap:
                continue
            cost = float((p - c).abs().mean())
            if cost < best_cost - 1e-9 or (abs(cost - best_cost) <= 1e-9 and abs(d) < abs(best_d)):
                best_cost, best_d = cost, d
        return best_d
