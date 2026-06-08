"""Calibrate the SMB3 HUD glyph atlas from a real captured frame.

Template matching needs pixel-exact glyphs, which can only come from real
frames. This tool slices the digit cells out of a known frame - where you've
read the on-screen value for each field - and saves them as the atlas the
extractor loads.

Workflow:
  1. Capture a frame (the smoke test saves them to artifacts/smoke/).
  2. Fill in the SPEC below: the frame path, and for each field its known digit
     string plus the cell ROI (x, y, cell size, stride).
  3. Run this file (Run/play on "NeuroForge: calibrate HUD", or from a terminal).
  4. It writes src/neuroforge/game/vision/assets/smb3/atlas.npz and saves
     per-field crops to artifacts/calibration/ so you can verify the ROIs lined
     up with the digits.

The ROIs here should match neuroforge.environments.games.smb3.hud.rois - adjust both together.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

#
FRAME_PATH = r"artifacts/smoke/frame_600.png"
# field name -> (known digits, x, y, n_cells, cell_w, cell_h, step_x)
FIELDS: dict[str, tuple[str, int, int, int, int, int, int]] = {
    # Example (placeholder - replace with values read off your frame):
    # "score": ("0000000", 96, 208, 7, 8, 8, 8),
    # "lives": ("4",       40, 208, 1, 8, 8, 8),
    # "world": ("1",       56, 200, 1, 8, 8, 8),
}
#

_REPO = Path(__file__).resolve().parents[1]
_ATLAS_OUT = _REPO / "src" / "neuroforge" / "game" / "vision" / "assets" / "smb3" / "atlas.npz"
_CROP_OUT = _REPO / "artifacts" / "calibration"


def _load_gray(path: Path) -> NDArray[np.uint8]:
    from torchvision.io import read_image

    image = read_image(str(path))  # [C, H, W] uint8
    hwc = np.asarray(image.permute(1, 2, 0).contiguous().numpy(), dtype=np.uint8)
    if hwc.shape[2] == 1:
        return hwc[:, :, 0]
    rgb = hwc[:, :, :3].astype(np.float32)
    gray = (rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)).round().clip(0, 255)
    return gray.astype(np.uint8)


def _save_png(path: Path, gray: NDArray[np.uint8]) -> None:
    import torch
    from torchvision.io import write_png

    path.parent.mkdir(parents=True, exist_ok=True)
    tensor = torch.from_numpy(np.ascontiguousarray(gray))[None, :, :]  # [1, H, W]
    write_png(tensor, str(path))


def main() -> int:
    frame_path = Path(FRAME_PATH)
    if not frame_path.is_absolute():
        frame_path = _REPO / frame_path
    if not frame_path.is_file():
        print(f"ERROR: frame not found: {frame_path}")
        return 1
    if not FIELDS:
        print("Nothing to calibrate: fill in the FIELDS spec at the top of this file.")
        print("Tip: capture an in-level frame, read each HUD value, and add a row per field.")
        return 1

    gray = _load_gray(frame_path)
    print(f"Loaded {frame_path.name}: {gray.shape[1]}x{gray.shape[0]} grayscale")

    glyphs: dict[str, NDArray[np.uint8]] = {}
    for name, (digits, x, y, n_cells, cw, ch, step) in FIELDS.items():
        if len(digits) > n_cells:
            print(f"  [{name}] WARNING: {len(digits)} digits > {n_cells} cells")
        region = gray[y : y + ch, x : x + step * max(1, n_cells)]
        _save_png(_CROP_OUT / f"{name}.png", region)
        for i, ch_char in enumerate(digits):
            cx = x + i * step
            cell = gray[y : y + ch, cx : cx + cw]
            if cell.shape != (ch, cw):
                print(f"  [{name}] cell {i} out of bounds - skipped")
                continue
            glyphs.setdefault(ch_char, cell.copy())
        print(f"  [{name}] '{digits}' -> {len(digits)} cells (crop saved)")

    if not glyphs:
        print("No glyphs extracted - check the ROIs.")
        return 1

    _ATLAS_OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(_ATLAS_OUT, **glyphs)
    print(f"\nWrote {len(glyphs)} glyph(s) {sorted(glyphs)} -> {_ATLAS_OUT}")
    print(f"Verify the crops in {_CROP_OUT} line up with the digits, then re-run training.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
