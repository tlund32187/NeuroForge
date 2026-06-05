"""SMB3 vision-only metric extractor.

Implements :class:`~neuroforge.contracts.applications.games.IFrameMetricExtractor`: turns a
:class:`ScreenFrame` into :class:`VisionGameMetrics` with per-field confidence,
reading only pixels. Digit fields (score/lives/world) come from a calibrated
glyph atlas; the always-available ``x_progress`` comes from scroll tracking.

Graceful degradation is deliberate: with no calibrated atlas the digit fields
read as ``None`` (confidence 0) while ``x_progress`` still works â€” so the
extractor is safe to wire into the loop before calibration is finished.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from neuroforge.contracts.applications.games import ScreenFrame, VisionGameMetrics

if TYPE_CHECKING:
    from numpy.typing import NDArray
from neuroforge.environments.games.smb3.hud.digit_reader import (
    DigitGlyphAtlas,
    read_field,
    to_grayscale,
)
from neuroforge.environments.games.smb3.hud.progress_estimator import ScrollProgressEstimator
from neuroforge.environments.games.smb3.hud.rois import HudLayout, smb3_level_layout

__all__ = ["SMB3HudConfig", "SMB3HudExtractor"]

_DEFAULT_ATLAS = Path(__file__).parent / "assets" / "smb3" / "atlas.npz"


@dataclass(frozen=True, slots=True)
class SMB3HudConfig:
    """Configuration for :class:`SMB3HudExtractor`."""

    atlas_path: str | None = None       # None â†’ packaged default atlas
    min_confidence: float = 0.82        # digit fields below this read as None
    track_progress: bool = True
    level_length_px: float = 4000.0


class SMB3HudExtractor:
    """Read SMB3 HUD metrics + scroll progress from frame pixels."""

    def __init__(
        self,
        config: SMB3HudConfig | None = None,
        *,
        layout: HudLayout | None = None,
    ) -> None:
        self._cfg = config or SMB3HudConfig()
        self._layout = layout or smb3_level_layout()
        atlas_path = self._cfg.atlas_path or str(_DEFAULT_ATLAS)
        self._atlas = DigitGlyphAtlas.load(atlas_path)
        self._progress = ScrollProgressEstimator(level_length_px=self._cfg.level_length_px)

    @property
    def is_calibrated(self) -> bool:
        """True if a glyph atlas was loaded (digit reading is available)."""
        return self._atlas is not None

    def reset(self) -> None:
        """Reset per-episode state (scroll history). Call at episode start."""
        self._progress.reset()

    def extract(self, frame: ScreenFrame) -> VisionGameMetrics:
        """Infer metrics from a single frame (vision only)."""
        arr = np.frombuffer(frame.as_memoryview(), dtype=np.uint8).reshape(
            frame.height, frame.width, frame.channels,
        )

        x_progress: float | None = None
        if self._cfg.track_progress:
            x_progress = self._progress.update(arr)

        confidence: dict[str, float] = {}
        score = lives = time_left = None
        world = None

        if self._atlas is not None:
            gray = to_grayscale(arr)
            layout = self._layout.scaled_to(frame.width, frame.height)
            score = self._read_int(gray, layout, "score", confidence)
            lives = self._read_int(gray, layout, "lives", confidence)
            time_left = self._read_int(gray, layout, "time", confidence)
            world = self._read_digits(gray, layout, "world", confidence)

        if x_progress is not None:
            confidence["x_progress"] = 1.0

        return VisionGameMetrics(
            score=score,
            lives=lives,
            world=world,
            time_left=time_left,
            x_progress=x_progress,
            confidence=confidence,
        )

    #

    def _read_int(
        self,
        gray: NDArray[np.uint8],
        layout: HudLayout,
        name: str,
        confidence: dict[str, float],
    ) -> int | None:
        field = layout.field(name)
        if field is None or self._atlas is None:
            return None
        reading = read_field(gray, field, self._atlas)
        confidence[name] = reading.confidence
        if reading.value is None or reading.confidence < self._cfg.min_confidence:
            return None
        return reading.value

    def _read_digits(
        self,
        gray: NDArray[np.uint8],
        layout: HudLayout,
        name: str,
        confidence: dict[str, float],
    ) -> str | None:
        field = layout.field(name)
        if field is None or self._atlas is None:
            return None
        reading = read_field(gray, field, self._atlas)
        confidence[name] = reading.confidence
        if not reading.digits or reading.confidence < self._cfg.min_confidence:
            return None
        return reading.digits
