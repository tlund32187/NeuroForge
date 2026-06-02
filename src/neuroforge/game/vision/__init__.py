"""Vision-only metric extraction for game frames (SMB3 HUD + scroll progress)."""

from neuroforge.game.vision.digit_reader import DigitGlyphAtlas, FieldReading, read_field
from neuroforge.game.vision.hud_extractor import SMB3HudConfig, SMB3HudExtractor
from neuroforge.game.vision.progress_estimator import ScrollProgressEstimator
from neuroforge.game.vision.rois import (
    HudField,
    HudLayout,
    smb3_level_layout,
    smb3_map_layout,
)

__all__ = [
    "DigitGlyphAtlas",
    "FieldReading",
    "HudField",
    "HudLayout",
    "SMB3HudConfig",
    "SMB3HudExtractor",
    "ScrollProgressEstimator",
    "read_field",
    "smb3_level_layout",
    "smb3_map_layout",
]
