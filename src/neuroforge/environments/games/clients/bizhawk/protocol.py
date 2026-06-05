"""NeuroForge bridge wire protocol codec.

A tiny, explicit, little-endian framing used by both the Python client and the
BizHawk Lua side-car. Keeping the codec pure makes every message unit-testable.

Frame on the wire::

    [ magic "NFB1" (4B) ][ type (1B) ][ payload length (4B) ][ payload ]

Message types:

* Lua to Python: ``HELLO``, ``FRAME``, ``BYE``.
* Python to Lua: ``WELCOME``, ``ACTION``, ``RESET``, ``CLOSE``.

The action bitmask packs the 8 buttons in ``NINTENDO_BUTTONS`` order.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING, cast

from neuroforge.environments.games.clients.bizhawk.input import (
    decode_controller_action,
    encode_controller_action,
)

if TYPE_CHECKING:
    from neuroforge.contracts.applications.games import ControllerAction

__all__ = [
    "MAGIC",
    "PROTOCOL_VERSION",
    "ACTION_FLAG_PASSIVE",
    "PIXEL_FORMAT_RAW",
    "PIXEL_FORMAT_PNG",
    "HEADER_STRUCT",
    "HEADER_SIZE",
    "MsgType",
    "BridgeHello",
    "encode_message",
    "decode_header",
    "encode_hello",
    "decode_hello",
    "encode_welcome",
    "decode_welcome",
    "encode_frame_payload",
    "decode_frame_payload",
    "encode_action",
    "decode_action",
    "encode_reset",
    "decode_reset",
]

MAGIC: bytes = b"NFB1"
PROTOCOL_VERSION: int = 1

#: ACTION flag bit: do NOT inject the buttons â€” let the human's controller drive
#: (used for passive frame capture / recording human play). The frame is still
#: advanced and captured; only ``joypad.set`` is skipped on the Lua side.
ACTION_FLAG_PASSIVE: int = 0x01

#: Pixel encodings a FRAME payload may carry (after its fixed prefix).
PIXEL_FORMAT_RAW: str = "raw"  # raw row-major width*height*channels bytes
PIXEL_FORMAT_PNG: str = "png"  # a complete PNG file (decoded Python-side)
_VALID_PIXEL_FORMATS: frozenset[str] = frozenset({PIXEL_FORMAT_RAW, PIXEL_FORMAT_PNG})

# Header: magic(4s) + type(B) + payload length(I, uint32). Little-endian.
HEADER_STRUCT: str = "<4sBI"
HEADER_SIZE: int = struct.calcsize(HEADER_STRUCT)  # 9 bytes

# Frame payload prefix: frame_id(Q) + emu_time_us(Q), then raw pixel bytes.
_FRAME_PREFIX_STRUCT: str = "<QQ"
_FRAME_PREFIX_SIZE: int = struct.calcsize(_FRAME_PREFIX_STRUCT)  # 16 bytes

class MsgType(IntEnum):
    """Wire message type tags."""

    # Lua â†’ Python
    HELLO = 1
    FRAME = 2
    BYE = 3
    # Python â†’ Lua
    WELCOME = 16
    ACTION = 17
    RESET = 18
    CLOSE = 19


#


def encode_message(msg_type: MsgType, payload: bytes = b"") -> bytes:
    """Frame *payload* with the protocol header."""
    return struct.pack(HEADER_STRUCT, MAGIC, int(msg_type), len(payload)) + payload


def decode_header(header: bytes) -> tuple[MsgType, int]:
    """Decode a 9-byte header into ``(msg_type, payload_length)``.

    Raises
    ------
    ValueError:
        If the header is the wrong size, the magic is wrong, or the type tag
        is unknown.
    """
    if len(header) != HEADER_SIZE:
        msg = f"header must be {HEADER_SIZE} bytes; got {len(header)}"
        raise ValueError(msg)
    magic, type_tag, length = struct.unpack(HEADER_STRUCT, header)
    if magic != MAGIC:
        msg = f"bad magic: {magic!r}"
        raise ValueError(msg)
    return MsgType(type_tag), int(length)


#


@dataclass(frozen=True, slots=True)
class BridgeHello:
    """Geometry/format the Lua side announces in its HELLO message."""

    version: int
    width: int
    height: int
    channels: int
    pixel_format: str


def _json_object(payload: bytes, *, what: str) -> dict[str, object]:
    """Parse *payload* as a JSON object, strictly typed."""
    try:
        parsed: object = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        msg = f"invalid {what} payload: {exc}"
        raise ValueError(msg) from exc
    if not isinstance(parsed, dict):
        msg = f"{what} payload must be a JSON object"
        raise ValueError(msg)
    # JSON object keys are always str; values are coerced at use sites.
    return cast("dict[str, object]", parsed)


def _as_int(value: object, *, default: int = 0) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return default
    return default


def _as_str(value: object, *, default: str) -> str:
    return value if isinstance(value, str) else default


def encode_hello(
    *,
    width: int,
    height: int,
    channels: int,
    pixel_format: str = PIXEL_FORMAT_RAW,
    version: int = PROTOCOL_VERSION,
) -> bytes:
    """Encode a HELLO payload announcing frame geometry and pixel format."""
    body = {
        "version": int(version),
        "width": int(width),
        "height": int(height),
        "channels": int(channels),
        "format": str(pixel_format),
    }
    return json.dumps(body).encode("utf-8")


def decode_hello(payload: bytes) -> BridgeHello:
    """Decode a HELLO payload to a :class:`BridgeHello`."""
    body = _json_object(payload, what="HELLO")
    pixel_format = _as_str(body.get("format"), default=PIXEL_FORMAT_RAW)
    if pixel_format not in _VALID_PIXEL_FORMATS:
        msg = f"unsupported pixel format: {pixel_format!r}"
        raise ValueError(msg)
    return BridgeHello(
        version=_as_int(body.get("version")),
        width=_as_int(body.get("width")),
        height=_as_int(body.get("height")),
        channels=_as_int(body.get("channels")),
        pixel_format=pixel_format,
    )


def encode_welcome(
    *, width: int, height: int, channels: int, version: int = PROTOCOL_VERSION,
) -> bytes:
    """Encode a WELCOME ack confirming the negotiated geometry."""
    body = {
        "version": int(version),
        "accepted": True,
        "width": int(width),
        "height": int(height),
        "channels": int(channels),
    }
    return json.dumps(body).encode("utf-8")


def decode_welcome(payload: bytes) -> dict[str, object]:
    """Decode a WELCOME payload to a plain object map."""
    return _json_object(payload, what="WELCOME")


#


def encode_frame_payload(frame_id: int, emu_time_us: int, pixels: bytes) -> bytes:
    """Encode a FRAME payload: ``frame_id``, ``emu_time_us``, then raw pixels."""
    return struct.pack(_FRAME_PREFIX_STRUCT, int(frame_id), int(emu_time_us)) + bytes(pixels)


def decode_frame_payload(payload: bytes) -> tuple[int, int, bytes]:
    """Decode a FRAME payload into ``(frame_id, emu_time_us, pixels)``.

    Raises
    ------
    ValueError:
        If the payload is shorter than the fixed prefix.
    """
    if len(payload) < _FRAME_PREFIX_SIZE:
        msg = f"FRAME payload too short: {len(payload)} < {_FRAME_PREFIX_SIZE}"
        raise ValueError(msg)
    frame_id, emu_time_us = struct.unpack_from(_FRAME_PREFIX_STRUCT, payload, 0)
    pixels = bytes(payload[_FRAME_PREFIX_SIZE:])
    return int(frame_id), int(emu_time_us), pixels


#


def encode_action(action: ControllerAction, *, flags: int = 0) -> bytes:
    """Encode a controller action as a button bitmask + flags byte."""
    return encode_controller_action(action, flags=flags)


def decode_action(payload: bytes) -> ControllerAction:
    """Decode a button bitmask payload back into a :class:`ControllerAction`.

    Used by the loopback test's "fake Lua" side. Masks are produced only from
    valid actions, so the d-pad exclusivity invariant always holds.
    """
    return decode_controller_action(payload)


#


def encode_reset(savestate: str = "") -> bytes:
    """Encode a RESET payload.

    An empty payload means "reboot the core" (a fresh power-on). A non-empty
    payload is a raw UTF-8 savestate **file path**: the Lua side loads it as an
    *environment reset* (like resetting a Gym env to a start state) so the brain
    can begin inside a level. This is NOT memory reading â€” the policy still only
    ever sees pixels. A raw path (not JSON) avoids backslash-escaping pitfalls
    with Windows paths.
    """
    return savestate.encode("utf-8") if savestate else b""


def decode_reset(payload: bytes) -> str | None:
    """Decode a RESET payload into a savestate path, or ``None`` to reboot."""
    if not payload:
        return None
    try:
        path = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        msg = f"invalid RESET payload: {exc}"
        raise ValueError(msg) from exc
    return path or None
