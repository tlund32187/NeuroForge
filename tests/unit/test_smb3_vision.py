"""Unit tests for SMB3 vision metric extraction (Phase 1).

Uses synthetic glyphs/frames so the engine is verified without real ROM frames:
digit reading, scroll-based progress, graceful degradation when uncalibrated,
and a calibrated end-to-end extract.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from neuroforge.contracts.game import ScreenFrame
from neuroforge.game.vision import (
    DigitGlyphAtlas,
    HudField,
    HudLayout,
    ScrollProgressEstimator,
    SMB3HudConfig,
    SMB3HudExtractor,
    read_field,
    smb3_map_layout,
)

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray


def _glyph(digit: int) -> NDArray[np.uint8]:
    """A distinct 8x8 grayscale glyph per digit (lit-pixel count = (d+1)*4)."""
    arr = np.zeros((8, 8), dtype=np.uint8)
    flat = arr.reshape(-1)
    flat[: (digit + 1) * 4] = 255
    return arr


def _atlas() -> DigitGlyphAtlas:
    return DigitGlyphAtlas.from_arrays({str(d): _glyph(d) for d in range(10)})


def _render_digits(gray: NDArray[np.uint8], field: HudField, digits: str) -> None:
    for i, ch in enumerate(digits):
        cx, cy = field.cell_origin(i)
        gray[cy : cy + field.cell_h, cx : cx + field.cell_w] = _glyph(int(ch))


# ── digit reading ──────────────────────────────────────────────────────


@pytest.mark.unit
def test_read_field_reads_multi_digit_int() -> None:
    atlas = _atlas()
    field = HudField(name="score", x=10, y=5, n_cells=3, kind="int")
    gray = np.zeros((40, 80), dtype=np.uint8)
    _render_digits(gray, field, "123")
    reading = read_field(gray, field, atlas)
    assert reading.digits == "123"
    assert reading.value == 123
    assert reading.confidence == pytest.approx(1.0)


@pytest.mark.unit
def test_read_field_skips_blank_cells() -> None:
    atlas = _atlas()
    field = HudField(name="score", x=4, y=4, n_cells=4, kind="int")
    gray = np.zeros((40, 80), dtype=np.uint8)
    _render_digits(gray, field, "42")  # only first 2 of 4 cells lit
    reading = read_field(gray, field, atlas)
    assert reading.value == 42


@pytest.mark.unit
def test_blank_cell_reads_as_blank() -> None:
    atlas = _atlas()
    label, conf = atlas.match(np.zeros((8, 8), dtype=np.uint8))
    assert label == ""
    assert conf == pytest.approx(1.0)


# ── scroll progress ──────────────────────────────────────────────────────


def _striped(width: int, height: int = 120, phase: int = 0) -> NDArray[np.uint8]:
    cols = (np.sin((np.arange(width) + phase) / 5.0) * 100 + 128).astype(np.uint8)
    return np.tile(cols, (height, 1))


@pytest.mark.unit
def test_progress_detects_rightward_scroll() -> None:
    est = ScrollProgressEstimator(level_length_px=1000.0, search_px=24, deadband_px=0)
    base = _striped(width=200, phase=0)
    assert est.update(base) == pytest.approx(0.0)
    # Content shifted left by 10 px == camera moved right by 10 px.
    est.update(_striped(width=200, phase=10))
    assert est.forward_px == pytest.approx(10.0, abs=2.0)


@pytest.mark.unit
def test_progress_tracks_max_and_resets() -> None:
    est = ScrollProgressEstimator(level_length_px=1000.0, search_px=24, deadband_px=0)
    est.update(_striped(200, phase=0))
    est.update(_striped(200, phase=10))
    peak = est.forward_px
    est.update(_striped(200, phase=5))  # backtrack
    assert est.forward_px == pytest.approx(peak)  # max is not reduced
    est.reset()
    assert est.forward_px == 0.0


# ── layout scaling ───────────────────────────────────────────────────────


@pytest.mark.unit
def test_layout_scaled_to_rescales_coordinates() -> None:
    layout = HudLayout(
        name="t", width=256, height=224,
        fields=(HudField(name="score", x=128, y=112, n_cells=2),),
    )
    scaled = layout.scaled_to(512, 448)
    field = scaled.field("score")
    assert field is not None
    assert (field.x, field.y) == (256, 224)
    # Identity when geometry matches.
    assert layout.scaled_to(256, 224) is layout


# ── extractor: graceful degradation + calibrated read ────────────────────


def _frame_from_gray(gray: NDArray[np.uint8]) -> ScreenFrame:
    rgb = np.stack([gray, gray, gray], axis=-1).astype(np.uint8)
    h, w = gray.shape
    return ScreenFrame(width=w, height=h, channels=3, data=rgb.tobytes())


@pytest.mark.unit
def test_extractor_degrades_gracefully_without_atlas() -> None:
    ext = SMB3HudExtractor(SMB3HudConfig(atlas_path="does_not_exist.npz"))
    assert ext.is_calibrated is False
    frame = ScreenFrame(width=256, height=224, channels=3, data=bytes(256 * 224 * 3))
    metrics = ext.extract(frame)
    assert metrics.score is None
    assert metrics.lives is None
    assert metrics.world is None
    # Progress is tracked even without a calibrated atlas.
    assert metrics.x_progress is not None
    assert "x_progress" in metrics.confidence


@pytest.mark.unit
def test_extractor_reads_score_when_calibrated(tmp_path: Path) -> None:
    # Write a temp atlas the extractor will load.
    atlas_path = tmp_path / "atlas.npz"
    np.savez(atlas_path, **{str(d): _glyph(d) for d in range(10)})

    layout = smb3_map_layout()
    score_field = layout.field("score")
    assert score_field is not None

    gray = np.zeros((224, 256), dtype=np.uint8)
    _render_digits(gray, score_field, "0001234")

    ext = SMB3HudExtractor(SMB3HudConfig(atlas_path=str(atlas_path)), layout=layout)
    assert ext.is_calibrated is True
    metrics = ext.extract(_frame_from_gray(gray))
    assert metrics.score == 1234
    assert metrics.confidence["score"] == pytest.approx(1.0)


@pytest.mark.unit
def test_extractor_satisfies_protocol() -> None:
    from neuroforge.contracts.game import IFrameMetricExtractor

    extractor = SMB3HudExtractor(SMB3HudConfig(atlas_path="none.npz"))
    assert isinstance(extractor, IFrameMetricExtractor)


@pytest.mark.unit
def test_packaged_atlas_reads_inlevel_hud() -> None:
    # Locks in the real (packaged) atlas + verified in-level ROIs + threshold,
    # using a synthetic cyan-background frame rendered from the atlas glyphs
    # (no copyrighted ROM image committed).
    from neuroforge.game.vision.hud_extractor import _DEFAULT_ATLAS
    from neuroforge.game.vision.rois import smb3_level_layout

    ext = SMB3HudExtractor(SMB3HudConfig(track_progress=False))
    if not ext.is_calibrated:
        pytest.skip("packaged SMB3 atlas not present")

    with np.load(_DEFAULT_ATLAS) as data:
        glyphs = {k: np.asarray(data[k], dtype=np.uint8) for k in data.files}

    layout = smb3_level_layout()
    gray = np.full((224, 256), 220, dtype=np.uint8)  # cyan-ish HUD background

    def _render(field_name: str, digits: str) -> None:
        field = layout.field(field_name)
        assert field is not None
        for i, ch in enumerate(digits):
            cx, cy = field.cell_origin(i)
            gray[cy : cy + 8, cx : cx + 8] = glyphs[ch]

    _render("world", "1")
    _render("lives", "4")
    _render("score", "0001400")
    _render("time", "280")

    rgb = np.stack([gray, gray, gray], axis=-1).astype(np.uint8)
    metrics = ext.extract(ScreenFrame(width=256, height=224, channels=3, data=rgb.tobytes()))
    assert metrics.score == 1400
    assert metrics.lives == 4
    assert metrics.time_left == 280
    assert metrics.world == "1"
