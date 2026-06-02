# pyright: basic
"""Retinal encoding: palette-invariant contrast channels from pixels (Layer A0).

A real retina does not forward raw color to the brain. Ganglion cells report
*local contrast* through ON/OFF center-surround receptive fields, with gain
control (divisive normalization) that adapts to overall contrast. This encoder
mirrors that, and the payoff is invariance to exactly the things that differ
between SMB3 worlds while structure is preserved:

* **grayscale** drops hue;
* **center-surround (difference-of-Gaussians)** drops local mean brightness —
  it responds to edges/holes/outlines, not absolute luminance;
* **divisive normalization** drops contrast *magnitude* — a low-contrast and a
  high-contrast rendering of the same shape encode almost identically.

So "a hole is a hole" and "that outline is a sprite" survive a palette change
(world 1 black → world 2 blue). There are **no labels and no prepopulated sprite
shapes** — the invariance is a pure property of the encoding. The output is a
1-D current drive in the same format as
:class:`~neuroforge.game.policies.preprocess.FramePreprocessor`, so this is a
drop-in front-end for the spiking policy and the later STDP feature layers (A1+).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from neuroforge.core.torch_utils import require_torch, resolve_device_dtype

if TYPE_CHECKING:
    from neuroforge.contracts.game import ScreenFrame

__all__ = ["RetinaEncoderConfig", "RetinaEncoder"]


@dataclass(frozen=True, slots=True)
class RetinaEncoderConfig:
    """Configuration for :class:`RetinaEncoder`."""

    out_h: int = 28
    out_w: int = 32
    # Difference-of-Gaussians: a sharp center minus a blurred surround. center_sigma
    # 0 => a delta (sharpest) center; surround_sigma sets the contrast scale.
    center_sigma: float = 0.0
    surround_sigma: float = 1.5
    amplitude: float = 10.0     # peak drive current after normalization
    # Divisive-normalization semi-saturation: the contrast gain control. Larger =>
    # gentler normalization (near-blank frames stay near-zero instead of amplifying
    # noise); small enough that real contrast is normalized away (magnitude
    # invariance). Tuned on luminance in [0, 1].
    norm_eps: float = 0.02
    device: str = "cpu"
    dtype: str = "float32"

    def __post_init__(self) -> None:
        if self.out_h <= 0 or self.out_w <= 0:
            msg = "RetinaEncoderConfig out_h/out_w must be > 0"
            raise ValueError(msg)
        if self.surround_sigma <= 0.0:
            msg = "RetinaEncoderConfig.surround_sigma must be > 0"
            raise ValueError(msg)
        if self.center_sigma < 0.0:
            msg = "RetinaEncoderConfig.center_sigma must be >= 0"
            raise ValueError(msg)
        if self.amplitude <= 0.0:
            msg = "RetinaEncoderConfig.amplitude must be > 0"
            raise ValueError(msg)


class RetinaEncoder:
    """Encode frames into palette-invariant ON/OFF contrast drive currents."""

    def __init__(self, config: RetinaEncoderConfig | None = None) -> None:
        self._cfg = config or RetinaEncoderConfig()
        self._torch = require_torch()
        self._dev, self._dtype = resolve_device_dtype(self._cfg.device, self._cfg.dtype)
        self._gray_w = self._torch.tensor(
            [0.299, 0.587, 0.114], dtype=self._dtype,
        ).view(3, 1, 1)
        self._surround_k = self._gaussian_kernel(self._cfg.surround_sigma)
        self._center_k = (
            self._gaussian_kernel(self._cfg.center_sigma)
            if self._cfg.center_sigma > 0.0
            else None
        )

    @property
    def input_size(self) -> int:
        """Number of input neurons (ON and OFF channels over the grid)."""
        return self._cfg.out_h * self._cfg.out_w * 2

    def reset(self) -> None:
        """No-op: A0 is stateless. Present for front-end interface parity."""

    def to_drive(self, frame: ScreenFrame) -> Any:
        """Return a 1-D ON/OFF contrast drive ``[2*out_h*out_w]``."""
        torch = self._torch
        luminance = self._to_luminance(frame)                 # [1, 1, H, W]
        center = (
            self._blur(luminance, self._center_k)
            if self._center_k is not None
            else luminance
        )
        surround = self._blur(luminance, self._surround_k)
        dog = center - surround                                # signed contrast

        rms = torch.sqrt(torch.mean(dog * dog))               # global gain control
        scale = self._cfg.amplitude / (rms + self._cfg.norm_eps)
        on = torch.relu(dog) * scale
        off = torch.relu(-dog) * scale

        return torch.cat([on.reshape(-1), off.reshape(-1)]).to(self._dev)

    # ── internals ─────────────────────────────────────────────────────

    def _to_luminance(self, frame: ScreenFrame) -> Any:
        torch = self._torch
        raw = torch.frombuffer(bytearray(frame.data), dtype=torch.uint8)
        img = raw.reshape(frame.height, frame.width, frame.channels).to(self._dtype) / 255.0
        chw = img.permute(2, 0, 1)
        if frame.channels >= 3:
            gray = (chw[:3] * self._gray_w).sum(dim=0, keepdim=True)
        else:
            gray = chw[:1]
        return torch.nn.functional.interpolate(
            gray[None], size=(self._cfg.out_h, self._cfg.out_w), mode="area",
        )

    def _gaussian_kernel(self, sigma: float) -> Any:
        torch = self._torch
        radius = max(1, math.ceil(3.0 * sigma))
        coords = torch.arange(-radius, radius + 1, dtype=self._dtype)
        kernel_1d = torch.exp(-(coords * coords) / (2.0 * sigma * sigma))
        kernel_1d = kernel_1d / kernel_1d.sum()
        kernel_2d = torch.outer(kernel_1d, kernel_1d)
        return kernel_2d.view(1, 1, kernel_2d.shape[0], kernel_2d.shape[1])

    def _blur(self, img: Any, kernel: Any) -> Any:
        torch = self._torch
        pad = kernel.shape[-1] // 2
        padded = torch.nn.functional.pad(img, (pad, pad, pad, pad), mode="reflect")
        return torch.nn.functional.conv2d(padded, kernel)
