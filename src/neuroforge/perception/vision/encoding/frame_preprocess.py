"""Frame preprocessing: ``ScreenFrame`` -> input-population drive current.

Turns a raw emulator frame into a 1-D drive tensor for the policy network's
input population: grayscale, downsample, normalise, and scale to a current
amplitude. Optionally appends a frame-difference (motion) channel — cheap and
informative for a side-scroller. Downsampling here is the SNN's business; the
HUD extractor reads the native-resolution frame separately.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from neuroforge.kernel.torch_utils import require_torch, resolve_device_dtype

if TYPE_CHECKING:
    from neuroforge.contracts.applications.games import ScreenFrame

__all__ = ["FramePreprocessConfig", "FramePreprocessor"]


@dataclass(frozen=True, slots=True)
class FramePreprocessConfig:
    """Configuration for :class:`FramePreprocessor`."""

    out_h: int = 28
    out_w: int = 32
    grayscale: bool = True
    # Peak drive current for a full-brightness pixel — a rate code (brighter
    # pixel -> faster firing). Tuned so a bright pixel fires within a tick or
    # two for the policy network's fast (tau~5ms) input neurons.
    amplitude: float = 10.0
    motion: bool = False        # append |frame - prev_frame| as a second channel
    device: str = "cpu"
    dtype: str = "float32"

    def __post_init__(self) -> None:
        if self.out_h <= 0 or self.out_w <= 0:
            msg = "FramePreprocessConfig out_h/out_w must be > 0"
            raise ValueError(msg)
        if self.amplitude <= 0:
            msg = "FramePreprocessConfig.amplitude must be > 0"
            raise ValueError(msg)


class FramePreprocessor:
    """Convert frames to input drive, optionally tracking motion across frames."""

    def __init__(self, config: FramePreprocessConfig | None = None) -> None:
        self._cfg = config or FramePreprocessConfig()
        self._torch = require_torch()
        self._dev, self._dtype = resolve_device_dtype(self._cfg.device, self._cfg.dtype)
        channels = 1 if self._cfg.grayscale else 3
        self._base_size = self._cfg.out_h * self._cfg.out_w * channels
        self._prev: Any = None

    @property
    def input_size(self) -> int:
        """Number of input neurons this preprocessor drives."""
        return self._base_size * (2 if self._cfg.motion else 1)

    def reset(self) -> None:
        """Forget the previous frame (call at episode start)."""
        self._prev = None

    def to_drive(self, frame: ScreenFrame) -> Any:
        """Return a 1-D drive tensor ``[input_size]`` on the configured device."""
        torch = self._torch
        raw = torch.frombuffer(bytearray(frame.data), dtype=torch.uint8)
        img = raw.reshape(frame.height, frame.width, frame.channels).to(self._dtype) / 255.0
        chw = img.permute(2, 0, 1)  # [C, H, W]

        if self._cfg.grayscale and frame.channels >= 3:
            weights = torch.tensor([0.299, 0.587, 0.114], dtype=self._dtype).view(3, 1, 1)
            chw = (chw[:3] * weights).sum(dim=0, keepdim=True)  # [1, H, W]
        elif self._cfg.grayscale:
            chw = chw[:1]

        resized = torch.nn.functional.interpolate(
            chw[None], size=(self._cfg.out_h, self._cfg.out_w), mode="area",
        )[0]
        flat = resized.reshape(-1).to(self._dev)
        base = flat * self._cfg.amplitude

        if not self._cfg.motion:
            return base

        if self._prev is None or self._prev.shape != flat.shape:
            diff = torch.zeros_like(base)
        else:
            diff = (flat - self._prev).abs() * self._cfg.amplitude
        self._prev = flat
        return torch.cat([base, diff])
