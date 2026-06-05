"""Vision-only metric extraction for game frames (SMB3 HUD + scroll progress)."""

from neuroforge.environments.games.smb3.hud.digit_reader import (
    DigitGlyphAtlas,
    FieldReading,
    read_field,
)
from neuroforge.environments.games.smb3.hud.extractor import SMB3HudConfig, SMB3HudExtractor
from neuroforge.environments.games.smb3.hud.progress_estimator import ScrollProgressEstimator
from neuroforge.environments.games.smb3.hud.rois import (
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
