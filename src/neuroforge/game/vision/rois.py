"""HUD region-of-interest layout (data, not code).

SMB3's HUD digits live at fixed pixel positions, so where to read each field is
pure configuration: a :class:`HudLayout` is a list of :class:`HudField`
rectangles. Keeping it as data means re-tuning for a different resolution or
core crop is a calibration step (see ``scripts/calibrate_smb3_hud.py``), never a
code change.

Coordinates are in the native frame's pixel space (origin top-left). A field is
a horizontal run of ``n_cells`` glyph cells, each ``cell_w × cell_h``, advancing
by ``step_x`` between cells.

The SMB3 defaults below are seeded from the world-map status bar at 256×224 and
are intended to be refined by calibration against real frames.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["HudField", "HudLayout", "smb3_level_layout", "smb3_map_layout"]


@dataclass(frozen=True, slots=True)
class HudField:
    """A horizontal run of glyph cells to read as one value."""

    name: str
    x: int
    y: int
    n_cells: int
    cell_w: int = 8
    cell_h: int = 8
    step_x: int = 8
    kind: str = "int"  # "int" parses digits to an int; "digits" keeps the string

    def cell_origin(self, index: int) -> tuple[int, int]:
        """Return the top-left ``(x, y)`` of the *index*-th cell."""
        if not 0 <= index < self.n_cells:
            msg = f"cell index {index} out of range for field {self.name!r}"
            raise IndexError(msg)
        return self.x + index * self.step_x, self.y


@dataclass(frozen=True, slots=True)
class HudLayout:
    """The set of HUD fields for a given frame geometry."""

    name: str
    width: int
    height: int
    fields: tuple[HudField, ...]

    def field(self, name: str) -> HudField | None:
        """Return the field named *name*, or ``None``."""
        for f in self.fields:
            if f.name == name:
                return f
        return None

    def scaled_to(self, width: int, height: int) -> HudLayout:
        """Return this layout rescaled to a different frame geometry.

        Lets a layout calibrated at one resolution apply to another (e.g. a core
        that crops 256×224 vs 256×240) without re-authoring coordinates.
        """
        if width == self.width and height == self.height:
            return self
        sx = width / self.width
        sy = height / self.height
        scaled = tuple(
            HudField(
                name=f.name,
                x=round(f.x * sx),
                y=round(f.y * sy),
                n_cells=f.n_cells,
                cell_w=max(1, round(f.cell_w * sx)),
                cell_h=max(1, round(f.cell_h * sy)),
                step_x=max(1, round(f.step_x * sx)),
                kind=f.kind,
            )
            for f in self.fields
        )
        return HudLayout(name=self.name, width=width, height=height, fields=scaled)


def smb3_map_layout() -> HudLayout:
    """Best-effort SMB3 world-map status-bar layout at 256×224.

    Seeded from observed frames; refine with ``scripts/calibrate_smb3_hud.py``.
    The status bar occupies roughly the bottom 32 px. These positions are a
    starting point for calibration, not yet pixel-verified.
    """
    return HudLayout(
        name="smb3_map",
        width=256,
        height=224,
        fields=(
            # "WORLD n" — single world digit after the label.
            HudField(name="world", x=56, y=200, n_cells=1, kind="digits"),
            # "M x n" lives count.
            HudField(name="lives", x=40, y=208, n_cells=1, kind="int"),
            # 7-digit score.
            HudField(name="score", x=96, y=208, n_cells=7, kind="int"),
            # 2-digit coin count after the "$".
            HudField(name="coins", x=176, y=208, n_cells=2, kind="int"),
        ),
    )


def smb3_level_layout() -> HudLayout:
    """SMB3 in-level status-bar layout at 256x224 (cyan panel, white digits).

    Pixel-verified against real frames: the bar has two text rows — the top
    (y=193) carries "WORLD n"; the bottom (y=201) carries lives, the 7-digit
    score, and the 3-digit TIME counter. Each glyph is an 8x8 cell on an 8 px
    stride.
    """
    return HudLayout(
        name="smb3_level",
        width=256,
        height=224,
        fields=(
            HudField(name="world", x=48, y=193, n_cells=1, kind="digits"),
            HudField(name="lives", x=48, y=201, n_cells=1, kind="int"),
            HudField(name="score", x=64, y=201, n_cells=7, kind="int"),
            HudField(name="time", x=136, y=201, n_cells=3, kind="int"),
        ),
    )
