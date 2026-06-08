"""Generate the SMB3 HUD glyph atlas from the pixel-verified font.

The 8x8 digit glyphs below were read directly from real SMB3 frames (white
digit bodies isolated at the calibrated brightness threshold). This regenerates
``src/neuroforge/game/vision/assets/smb3/atlas.npz`` — the template atlas the
:class:`SMB3HudExtractor` loads. ``#`` is a lit (white) pixel, ``.`` is unlit.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

# SMB3 status-bar digit font (8x8), transcribed from real frames.
# Each glyph is 8 rows of 8 chars: '#' = lit (white), '.' = unlit.
_GLYPHS: dict[str, tuple[str, ...]] = {
    "0": ("........", "........", "..####..", ".##..##.",
          ".##..##.", ".##..##.", "..####..", "........"),
    "1": ("........", "........", "..###...", "...##...",
          "...##...", "...##...", "..####..", "........"),
    "2": ("........", "........", ".#####..", ".....##.",
          "..####..", ".##.....", ".######.", "........"),
    "3": ("........", "........", ".#####..", ".....##.",
          "..####..", ".....##.", ".#####..", "........"),
    "4": ("........", "........", ".##..#..", ".##..#..",
          ".##..#..", ".######.", ".....#..", "........"),
    "5": ("........", "........", ".#####..", ".##.....",
          ".#####..", ".....##.", ".#####..", "........"),
    "6": ("........", "........", "..####..", ".##.....",
          ".#####..", ".##..##.", "..####..", "........"),
    "7": ("........", "........", ".######.", ".....##.",
          "....##..", "...##...", "..##....", "........"),
    "8": ("........", "........", "..####..", ".##..##.",
          "..####..", ".##..##.", "..####..", "........"),
    "9": ("........", "........", "..####..", ".##..##.",
          "..#####.", ".....##.", "..####..", "........"),
}

_ATLAS_OUT = (
    Path(__file__).resolve().parents[1]
    / "src" / "neuroforge" / "game" / "vision" / "assets" / "smb3" / "atlas.npz"
)


def _to_array(rows: tuple[str, ...]) -> NDArray[np.uint8]:
    arr: NDArray[np.uint8] = np.array(
        [[255 if ch == "#" else 0 for ch in row] for row in rows], dtype=np.uint8,
    )
    return arr


def main() -> int:
    glyphs = {label: _to_array(rows) for label, rows in _GLYPHS.items()}
    _ATLAS_OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(_ATLAS_OUT, **glyphs)
    print(f"Wrote {len(glyphs)} glyphs {sorted(glyphs)} -> {_ATLAS_OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
