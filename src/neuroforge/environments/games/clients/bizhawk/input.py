"""BizHawk controller input codec helpers."""

from __future__ import annotations

import struct

from neuroforge.contracts.applications.games import NINTENDO_BUTTONS, ControllerAction

__all__ = ["decode_controller_action", "encode_controller_action"]

_ACTION_STRUCT: str = "<BB"

_BUTTON_TO_FIELD: dict[str, str] = {
    "Up": "up",
    "Down": "down",
    "Left": "left",
    "Right": "right",
    "A": "a",
    "B": "b",
    "Start": "start",
    "Select": "select",
}


def encode_controller_action(action: ControllerAction, *, flags: int = 0) -> bytes:
    """Encode a controller action as a button bitmask + flags byte."""
    mask = 0
    for i, pressed in enumerate(action.as_dense_tuple()):
        if pressed:
            mask |= 1 << i
    return struct.pack(_ACTION_STRUCT, mask & 0xFF, flags & 0xFF)


def decode_controller_action(payload: bytes) -> ControllerAction:
    """Decode a button bitmask payload back into a controller action."""
    if len(payload) < struct.calcsize(_ACTION_STRUCT):
        msg = "ACTION payload too short"
        raise ValueError(msg)
    mask, _flags = struct.unpack_from(_ACTION_STRUCT, payload, 0)
    kwargs: dict[str, bool] = {}
    for i, name in enumerate(NINTENDO_BUTTONS):
        if mask & (1 << i):
            kwargs[_BUTTON_TO_FIELD[name]] = True
    return ControllerAction(**kwargs)
