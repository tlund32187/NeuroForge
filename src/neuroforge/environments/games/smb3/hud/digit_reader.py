"""Read HUD digit fields by matching glyph cells against a template atlas.

The atlas is a set of binarised glyph templates (one per character) captured
from real frames by ``scripts/calibrate_smb3_hud.py`` and stored as a single
``.npz``. Matching is deliberately simple and fast (per-frame, on the hot path):
binarise each cell, score it against every template by fraction of agreeing
pixels, and take the best. A near-empty cell reads as blank.

No torch — pure numpy on small ROIs, so this stays cheap and host-side.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from neuroforge.environments.games.smb3.hud.rois import HudField

__all__ = ["DigitGlyphAtlas", "FieldReading", "read_field", "to_grayscale"]

# A cell with fewer than this fraction of bright pixels is treated as blank.
_BLANK_FRACTION = 0.04
# Brightness (0-255) above which a pixel counts as "lit". Set to isolate the
# white digit body (~254) from both a black HUD bar (~0, world map) and the
# light-cyan in-level HUD panel (~220) — measured from real SMB3 frames.
_LIT_THRESHOLD = 235


def to_grayscale(frame: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Convert an ``[H, W, C]`` (or ``[H, W]``) uint8 image to ``[H, W]`` gray."""
    if frame.ndim == 2:
        return frame
    if frame.shape[2] == 1:
        return frame[:, :, 0]
    rgb = frame[:, :, :3].astype(np.float32)
    gray = rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    return gray.round().clip(0, 255).astype(np.uint8)


class DigitGlyphAtlas:
    """Binarised glyph templates keyed by the character they represent."""

    def __init__(self, glyphs: dict[str, NDArray[np.bool_]], cell_w: int, cell_h: int) -> None:
        if not glyphs:
            msg = "DigitGlyphAtlas requires at least one glyph"
            raise ValueError(msg)
        self._glyphs = glyphs
        self.cell_w = int(cell_w)
        self.cell_h = int(cell_h)

    @property
    def labels(self) -> list[str]:
        return sorted(self._glyphs)

    @classmethod
    def from_arrays(
        cls, raw: dict[str, NDArray[np.uint8]], *, lit_threshold: int = _LIT_THRESHOLD,
    ) -> DigitGlyphAtlas:
        """Build an atlas from grayscale glyph arrays (binarised on load)."""
        glyphs: dict[str, NDArray[np.bool_]] = {}
        cell_h = cell_w = 0
        for label, arr in raw.items():
            binary = arr >= lit_threshold
            glyphs[label] = binary
            cell_h, cell_w = binary.shape
        return cls(glyphs, cell_w, cell_h)

    @classmethod
    def load(cls, path: str | Path) -> DigitGlyphAtlas | None:
        """Load an atlas from a ``.npz`` produced by the calibration script.

        Returns ``None`` if the file is absent, so an uncalibrated extractor can
        degrade gracefully instead of failing.
        """
        npz_path = Path(path)
        if not npz_path.is_file():
            return None
        with np.load(npz_path) as data:
            raw = {key: np.asarray(data[key], dtype=np.uint8) for key in data.files}
        if not raw:
            return None
        return cls.from_arrays(raw)

    def match(self, cell: NDArray[np.uint8]) -> tuple[str, float]:
        """Return the best ``(label, confidence)`` for a grayscale *cell*.

        A near-empty cell returns ``("", confidence)`` to signal "blank".
        """
        binary = cell >= _LIT_THRESHOLD
        if binary.mean() < _BLANK_FRACTION:
            return "", 1.0  # confidently blank
        best_label = ""
        best_score = -1.0
        for label, glyph in self._glyphs.items():
            resized = _fit(binary, glyph.shape)
            score = float((resized == glyph).mean())
            if score > best_score:
                best_score = score
                best_label = label
        return best_label, best_score


class FieldReading:
    """The parsed result of reading one HUD field."""

    __slots__ = ("confidence", "digits", "value")

    def __init__(self, value: int | None, digits: str, confidence: float) -> None:
        self.value = value
        self.digits = digits
        self.confidence = confidence


def _fit(cell: NDArray[np.bool_], shape: tuple[int, ...]) -> NDArray[np.bool_]:
    """Crop/pad *cell* to *shape* (templates and ROIs may differ by a pixel)."""
    if cell.shape == shape:
        return cell
    h, w = int(shape[0]), int(shape[1])
    out = np.zeros((h, w), dtype=bool)
    ch, cw = min(h, cell.shape[0]), min(w, cell.shape[1])
    out[:ch, :cw] = cell[:ch, :cw]
    return out


def read_field(
    gray: NDArray[np.uint8], field: HudField, atlas: DigitGlyphAtlas,
) -> FieldReading:
    """Read a HUD *field* from a grayscale frame using *atlas*.

    Confidence is the lowest per-cell match score across the non-blank digits
    (the weakest link), or 0.0 when nothing legible was found.
    """
    labels: list[str] = []
    scores: list[float] = []
    for i in range(field.n_cells):
        cx, cy = field.cell_origin(i)
        cell = gray[cy : cy + field.cell_h, cx : cx + field.cell_w]
        if cell.shape[0] == 0 or cell.shape[1] == 0:
            continue
        label, score = atlas.match(cell)
        if label == "":
            continue  # blank cell — skip (leading/trailing padding)
        labels.append(label)
        scores.append(score)

    digits = "".join(labels)
    confidence = min(scores) if scores else 0.0
    value: int | None = None
    if field.kind == "int" and digits.isdigit():
        value = int(digits)
    return FieldReading(value=value, digits=digits, confidence=confidence)
