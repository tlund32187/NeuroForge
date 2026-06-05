"""Vision-only BizHawk client over the bridge protocol.

The client speaks the bridge wire format to a BizHawk Lua side-car over an
injected transport. It reads only screen pixels and returns frame observations.
It never touches emulator RAM, preserving the vision-only invariant.

Lifecycle::

    client = BizHawkClient(BizHawkClientConfig(...))
    obs = client.reset()
    step = client.step(ControllerAction(right=True))
    client.close()

"""

from __future__ import annotations

import contextlib
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from neuroforge.contracts.applications.games import GameClientStep, GameObservation, ScreenFrame
from neuroforge.environments.games.clients.bizhawk import protocol as proto
from neuroforge.environments.games.clients.bizhawk.errors import (
    BizHawkConnectionError,
    BizHawkProtocolError,
    BizHawkStateError,
)
from neuroforge.environments.games.clients.bizhawk.screenshot_socket import (
    ScreenshotSocketReceiver,
)
from neuroforge.environments.games.clients.bizhawk.transport import SocketTransport

if TYPE_CHECKING:
    from neuroforge.contracts.applications.games import ControllerAction
    from neuroforge.environments.games.clients.bizhawk.launcher import EmuHawkLauncher
    from neuroforge.environments.games.clients.bizhawk.transport import ITransport

__all__ = ["BizHawkClient", "BizHawkClientConfig"]


@dataclass(frozen=True, slots=True)
class BizHawkClientConfig:
    """Configuration for :class:`BizHawkClient`.

    Geometry (``width``/``height``/``channels``) is the native frame size the
    Lua side announces; it is validated against the HELLO handshake so a
    mismatched script fails fast and clearly. Defaults are NES native (256x240
    RGB).

    ``transport`` selects how messages reach the emulator:

    * ``"file"`` (default) - atomically-renamed files in ``comm_dir``. Works on
      NLua EmuHawk builds that lack LuaSocket.
    * ``"socket"`` - a localhost TCP server. The Lua side tries LuaSocket,
      including BizHawk's ``Lua/socket/core.dll`` path, then LuaCOM/MSWinsock.
    """

    host: str = "127.0.0.1"
    port: int = 8650
    width: int = 256
    height: int = 240
    channels: int = 3
    frameskip: int = 1
    connect_timeout_s: float = 30.0
    step_timeout_s: float = 10.0
    max_episode_steps: int = 0  # 0 = unlimited
    launch: bool = False
    transport: str = "file"
    comm_dir: str | None = None  # None -> a default scratch dir under TEMP
    bridge_error_path: str | None = None
    passive_input: bool = False  # True -> don't inject buttons; the human drives

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            msg = "BizHawkClientConfig width/height must be > 0"
            raise ValueError(msg)
        if self.channels not in {1, 3, 4}:
            msg = "BizHawkClientConfig.channels must be one of {1, 3, 4}"
            raise ValueError(msg)
        if self.frameskip < 1:
            msg = "BizHawkClientConfig.frameskip must be >= 1"
            raise ValueError(msg)
        if self.connect_timeout_s <= 0 or self.step_timeout_s <= 0:
            msg = "BizHawkClientConfig timeouts must be > 0"
            raise ValueError(msg)
        if self.max_episode_steps < 0:
            msg = "BizHawkClientConfig.max_episode_steps must be >= 0"
            raise ValueError(msg)
        if self.transport not in {"file", "socket"}:
            msg = "BizHawkClientConfig.transport must be 'file' or 'socket'"
            raise ValueError(msg)


class BizHawkClient:
    """Frame-only emulator client (implements :class:`IGameClient` structurally)."""

    def __init__(
        self,
        config: BizHawkClientConfig | None = None,
        *,
        transport: ITransport | None = None,
        launcher: EmuHawkLauncher | None = None,
    ) -> None:
        self._cfg = config or BizHawkClientConfig()
        self._transport = transport
        self._owns_transport = transport is None
        self._launcher = launcher
        self._screenshot_receiver: ScreenshotSocketReceiver | None = None
        self._bridge_error_path = self._cfg.bridge_error_path or str(
            Path(tempfile.gettempdir()) / "neuroforge_bridge_error.log"
        )
        self._connected = False
        self._step_count = 0
        self._next_savestate: str | None = None
        self._pixel_format = proto.PIXEL_FORMAT_RAW
        self._comm_dir = self._cfg.comm_dir or str(
            Path(tempfile.gettempdir()) / "neuroforge_bridge"
        )
        # Active frame geometry. Seeded from config, then replaced with whatever
        # the emulator announces in HELLO (it is authoritative for its buffer).
        self._frame_w = self._cfg.width
        self._frame_h = self._cfg.height
        self._frame_c = self._cfg.channels

    #

    def queue_savestate(self, path: str | None) -> None:
        """Load *path* on the next :meth:`reset` (an environment reset).

        Loading a savestate restores the emulator to a chosen start state - for
        SMB3 this is how training begins *inside a level* (a curriculum), where
        the vision reward is dense, instead of in the unrewarded title/map
        screens. It is an environment reset, not memory reading: the policy
        still only ever sees pixels. ``None`` (the default) reboots the core.
        """
        self._next_savestate = path or None

    def reset(self) -> GameObservation:
        """Connect (first call) or reboot/load-savestate, returning the frame."""
        if not self._connected:
            if self._transport is None:
                self._transport = self._create_owned_transport()
            self._handshake()
            self._connected = True
            self._step_count = 0
            first = self._recv_frame()  # auto FRAME the Lua sends after WELCOME
            if self._next_savestate is None:
                return first
            return self._reset_to_savestate()  # episode 0 starts from a savestate
        return self._reset_to_savestate()

    def _reset_to_savestate(self) -> GameObservation:
        """Send RESET (optionally with a savestate path) and read the frame."""
        savestate = self._next_savestate or ""
        self._next_savestate = None
        self._send(proto.MsgType.RESET, proto.encode_reset(savestate))
        self._step_count = 0
        return self._recv_frame()

    def _create_owned_transport(self) -> ITransport:
        """Build (and, when launching, start the emulator for) the transport."""
        if self._cfg.transport == "file":
            from neuroforge.environments.games.clients.bizhawk.file_transport import FileTransport

            transport = FileTransport.create(
                self._comm_dir, timeout=self._cfg.connect_timeout_s,
            )
            if self._launcher is not None:
                self._launcher.launch(comm_dir=self._comm_dir)
            return transport
        # socket: bind first, then launch via on_bound, then accept.
        return SocketTransport.serve(
            self._cfg.host,
            self._cfg.port,
            accept_timeout=self._cfg.connect_timeout_s,
            on_bound=self._on_bound if self._launcher is not None else None,
        )

    def step(self, action: ControllerAction) -> GameClientStep:
        """Apply *action* and return the next frame-derived observation."""
        if not self._connected or self._transport is None:
            msg = "BizHawkClient.step() called before reset()"
            raise BizHawkStateError(msg)
        self._set_timeout(self._cfg.step_timeout_s)
        flags = proto.ACTION_FLAG_PASSIVE if self._cfg.passive_input else 0
        self._send(proto.MsgType.ACTION, proto.encode_action(action, flags=flags))
        self._step_count += 1
        observation = self._recv_frame()
        truncated = (
            self._cfg.max_episode_steps > 0
            and self._step_count >= self._cfg.max_episode_steps
        )
        return GameClientStep(observation, terminated=False, truncated=truncated)

    def close(self) -> None:
        """Send CLOSE (best-effort) and release the transport (idempotent)."""
        transport = self._transport
        if transport is not None:
            if self._connected:
                with contextlib.suppress(BizHawkConnectionError):
                    transport.send(proto.encode_message(proto.MsgType.CLOSE))
            if self._owns_transport:
                transport.close()
        if self._launcher is not None:
            self._launcher.close()
        if self._screenshot_receiver is not None:
            self._screenshot_receiver.close()
            self._screenshot_receiver = None
        self._connected = False

    def __enter__(self) -> BizHawkClient:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    #

    def _on_bound(self, port: int) -> None:
        if self._launcher is not None:
            with contextlib.suppress(OSError):
                Path(self._bridge_error_path).unlink()
            screenshot_receiver = ScreenshotSocketReceiver.serve(self._cfg.host, 0)
            self._screenshot_receiver = screenshot_receiver
            self._launcher.launch(
                port=port,
                host=self._cfg.host,
                screenshot_port=screenshot_receiver.port,
                screenshot_host=self._cfg.host,
                error_path=self._bridge_error_path,
            )

    def _handshake(self) -> None:
        self._set_timeout(self._cfg.connect_timeout_s)
        msg_type, payload = self._recv_message()
        if msg_type is not proto.MsgType.HELLO:
            detail = f"expected HELLO, got {msg_type.name}"
            raise BizHawkProtocolError(detail)
        try:
            hello = proto.decode_hello(payload)
        except ValueError as exc:
            raise BizHawkProtocolError(str(exc)) from exc
        self._adopt_geometry(hello)
        self._pixel_format = hello.pixel_format
        self._send(
            proto.MsgType.WELCOME,
            proto.encode_welcome(
                width=self._frame_w,
                height=self._frame_h,
                channels=self._frame_c,
            ),
        )

    def _adopt_geometry(self, hello: proto.BridgeHello) -> None:
        """Adopt the emulator-announced frame geometry (it owns its buffer)."""
        if hello.version != proto.PROTOCOL_VERSION:
            detail = (
                f"protocol version mismatch: client {proto.PROTOCOL_VERSION}, "
                f"emulator {hello.version}"
            )
            raise BizHawkProtocolError(detail)
        if hello.width <= 0 or hello.height <= 0 or hello.channels not in {1, 3, 4}:
            detail = (
                "emulator announced invalid geometry "
                f"{hello.width}x{hello.height}x{hello.channels}"
            )
            raise BizHawkProtocolError(detail)
        self._frame_w = hello.width
        self._frame_h = hello.height
        self._frame_c = hello.channels

    def _recv_message(self) -> tuple[proto.MsgType, bytes]:
        header = self._recv_exactly(proto.HEADER_SIZE)
        try:
            msg_type, length = proto.decode_header(header)
        except ValueError as exc:
            raise BizHawkProtocolError(str(exc)) from exc
        payload = self._recv_exactly(length) if length > 0 else b""
        return msg_type, payload

    def _recv_frame(self) -> GameObservation:
        self._set_timeout(self._cfg.step_timeout_s)
        msg_type, payload = self._recv_message()
        if msg_type is proto.MsgType.BYE:
            msg = "emulator sent BYE (session ended)"
            raise BizHawkConnectionError(msg)
        if msg_type is not proto.MsgType.FRAME:
            detail = f"expected FRAME, got {msg_type.name}"
            raise BizHawkProtocolError(detail)
        try:
            frame_id, emu_time_us, pixels = proto.decode_frame_payload(payload)
            if self._pixel_format == proto.PIXEL_FORMAT_PNG:
                if not pixels and self._screenshot_receiver is not None:
                    pixels = self._screenshot_receiver.recv_screenshot(
                        timeout=self._cfg.step_timeout_s,
                    )
                from neuroforge.environments.games.clients.bizhawk.frame_codec import (
                    decode_png_to_raw,
                )

                pixels = decode_png_to_raw(
                    pixels,
                    width=self._frame_w,
                    height=self._frame_h,
                    channels=self._frame_c,
                )
            frame = ScreenFrame(
                width=self._frame_w,
                height=self._frame_h,
                channels=self._frame_c,
                data=pixels,
                frame_id=frame_id,
                t=emu_time_us / 1_000_000.0,
            )
        except ValueError as exc:
            raise BizHawkProtocolError(str(exc)) from exc
        return GameObservation(step=self._step_count, t=frame.t, frame=frame)

    def _send(self, msg_type: proto.MsgType, payload: bytes = b"") -> None:
        assert self._transport is not None  # noqa: S101 - guarded by callers
        self._transport.send(proto.encode_message(msg_type, payload))

    def _recv_exactly(self, n: int) -> bytes:
        assert self._transport is not None  # noqa: S101 - guarded by callers
        try:
            return self._transport.recv_exactly(n)
        except BizHawkConnectionError as exc:
            detail = self._read_bridge_error()
            if detail:
                msg = f"{exc}; Lua bridge error: {detail}"
                raise BizHawkConnectionError(msg) from exc
            raise

    def _set_timeout(self, seconds: float) -> None:
        setter = getattr(self._transport, "set_timeout", None)
        if callable(setter):
            setter(seconds)

    def _read_bridge_error(self) -> str:
        with contextlib.suppress(OSError, UnicodeDecodeError):
            text = Path(self._bridge_error_path).read_text(encoding="utf-8").strip()
            return text[-1000:]
        return ""
