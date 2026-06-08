"""Unit tests for the BizHawk bridge (Phase 0).

Covers the wire codec, the client state machine over an in-memory fake, the
client over a real loopback socket (framing + recv buffer growth), the PNG
frame path, the scripted clients through the loop, and the vision-only
"no RAM reads" invariant of the Lua side-car. No emulator is required.
"""

from __future__ import annotations

import socket
import threading
from pathlib import Path

import pytest

from neuroforge.contracts.applications.games import NINTENDO_BUTTONS, ControllerAction, IGameClient
from neuroforge.environments.games.clients.bizhawk import protocol as proto
from neuroforge.environments.games.clients.bizhawk.client import BizHawkClient, BizHawkClientConfig
from neuroforge.environments.games.clients.bizhawk.errors import (
    BizHawkConnectionError,
    BizHawkProtocolError,
    BizHawkStateError,
)
from neuroforge.environments.games.clients.bizhawk.transport import SocketTransport
from neuroforge.environments.games.clients.scripted import (
    ActionProgressGameClient,
    ReplayGameClient,
    ScriptedGameClient,
)
from neuroforge.environments.games.smb3.environment import VisionOnlyGameLoop

_REPO_ROOT = Path(__file__).resolve().parents[2]
_LUA_BRIDGE = _REPO_ROOT / "scripts" / "bizhawk" / "neuroforge_bridge.lua"


#


def _hello_msg(w: int, h: int, c: int, *, fmt: str = "raw", version: int = 1) -> bytes:
    return proto.encode_message(
        proto.MsgType.HELLO,
        proto.encode_hello(width=w, height=h, channels=c, pixel_format=fmt, version=version),
    )


def _raw_frame_msg(frame_id: int, w: int, h: int, c: int, *, fill: int) -> bytes:
    pixels = bytes([fill % 256]) * (w * h * c)
    payload = proto.encode_frame_payload(frame_id, frame_id * 1000, pixels)
    return proto.encode_message(proto.MsgType.FRAME, payload)


def _read_msg(transport: SocketTransport) -> tuple[proto.MsgType, bytes]:
    header = transport.recv_exactly(proto.HEADER_SIZE)
    msg_type, length = proto.decode_header(header)
    payload = transport.recv_exactly(length) if length else b""
    return msg_type, payload


def _parse_sent(buf: bytes) -> list[tuple[proto.MsgType, bytes]]:
    out: list[tuple[proto.MsgType, bytes]] = []
    i = 0
    while i < len(buf):
        msg_type, length = proto.decode_header(buf[i : i + proto.HEADER_SIZE])
        i += proto.HEADER_SIZE
        out.append((msg_type, buf[i : i + length]))
        i += length
    return out


class _FakeTransport:
    """In-memory ITransport: scripted recv stream + captured sent bytes."""

    def __init__(self) -> None:
        self.incoming = bytearray()
        self.sent = bytearray()
        self.closed = False

    def feed(self, data: bytes) -> None:
        self.incoming += data

    def send(self, data: bytes) -> None:
        self.sent += data

    def recv_exactly(self, n: int) -> bytes:
        if n <= 0:
            return b""
        if len(self.incoming) < n:
            msg = "fake transport exhausted"
            raise BizHawkConnectionError(msg)
        out = bytes(self.incoming[:n])
        del self.incoming[:n]
        return out

    def set_timeout(self, _seconds: float) -> None:
        pass

    def close(self) -> None:
        self.closed = True


#


@pytest.mark.unit
def test_header_roundtrip() -> None:
    framed = proto.encode_message(proto.MsgType.FRAME, b"hello")
    msg_type, length = proto.decode_header(framed[: proto.HEADER_SIZE])
    assert msg_type is proto.MsgType.FRAME
    assert length == len(b"hello")
    assert framed[proto.HEADER_SIZE :] == b"hello"


@pytest.mark.unit
def test_decode_header_rejects_bad_magic() -> None:
    bad = b"XXXX" + bytes([int(proto.MsgType.FRAME)]) + (0).to_bytes(4, "little")
    with pytest.raises(ValueError, match="magic"):
        proto.decode_header(bad)


@pytest.mark.unit
@pytest.mark.parametrize("field", ["up", "down", "left", "right", "a", "b", "start", "select"])
def test_action_bitmask_roundtrip_each_button(field: str) -> None:
    action = ControllerAction(**{field: True})
    decoded = proto.decode_action(proto.encode_action(action))
    assert decoded == action


@pytest.mark.unit
def test_action_bitmask_roundtrip_combo() -> None:
    action = ControllerAction(right=True, b=True, a=True)
    assert proto.decode_action(proto.encode_action(action)) == action


@pytest.mark.unit
def test_action_bit_order_matches_nintendo_buttons() -> None:
    # Bit i must correspond to NINTENDO_BUTTONS[i].
    for i, _name in enumerate(NINTENDO_BUTTONS):
        mask = proto.encode_action(_action_from_bit(i))[0]
        assert mask == (1 << i)


def _action_from_bit(bit: int) -> ControllerAction:
    field = {
        0: "up", 1: "down", 2: "left", 3: "right",
        4: "a", 5: "b", 6: "start", 7: "select",
    }[bit]
    return ControllerAction(**{field: True})


@pytest.mark.unit
def test_hello_roundtrip_with_format() -> None:
    payload = proto.encode_hello(width=256, height=240, channels=3, pixel_format="png")
    hello = proto.decode_hello(payload)
    assert (hello.width, hello.height, hello.channels) == (256, 240, 3)
    assert hello.pixel_format == "png"
    assert hello.version == proto.PROTOCOL_VERSION


@pytest.mark.unit
def test_hello_rejects_unknown_format() -> None:
    payload = proto.encode_hello(width=2, height=2, channels=1, pixel_format="jpeg")
    with pytest.raises(ValueError, match="pixel format"):
        proto.decode_hello(payload)


@pytest.mark.unit
def test_frame_payload_roundtrip() -> None:
    pixels = bytes(range(12))
    frame_id, emu_time_us, decoded = proto.decode_frame_payload(
        proto.encode_frame_payload(42, 7, pixels)
    )
    assert (frame_id, emu_time_us) == (42, 7)
    assert decoded == pixels


#


@pytest.mark.unit
def test_client_reset_then_step_raw() -> None:
    cfg = BizHawkClientConfig(width=2, height=2, channels=1, max_episode_steps=2)
    ft = _FakeTransport()
    ft.feed(_hello_msg(2, 2, 1))
    ft.feed(_raw_frame_msg(0, 2, 2, 1, fill=10))
    ft.feed(_raw_frame_msg(1, 2, 2, 1, fill=11))
    client = BizHawkClient(cfg, transport=ft)

    obs0 = client.reset()
    assert obs0.frame.data == bytes([10]) * 4
    assert obs0.frame.frame_id == 0

    step1 = client.step(ControllerAction(right=True))
    assert step1.observation.frame.data == bytes([11]) * 4
    assert step1.terminated is False
    assert step1.truncated is False  # 1 < max_episode_steps(2)

    sent = _parse_sent(bytes(ft.sent))
    assert sent[0][0] is proto.MsgType.WELCOME
    assert sent[1][0] is proto.MsgType.ACTION
    assert proto.decode_action(sent[1][1]) == ControllerAction(right=True)


@pytest.mark.unit
def test_passive_input_sets_action_flag() -> None:
    cfg = BizHawkClientConfig(width=1, height=1, channels=1, passive_input=True)
    ft = _FakeTransport()
    ft.feed(_hello_msg(1, 1, 1))
    ft.feed(_raw_frame_msg(0, 1, 1, 1, fill=0))
    ft.feed(_raw_frame_msg(1, 1, 1, 1, fill=1))
    client = BizHawkClient(cfg, transport=ft)
    client.reset()
    client.step(ControllerAction(right=True))
    action = next(m for m in _parse_sent(bytes(ft.sent)) if m[0] is proto.MsgType.ACTION)
    assert action[1][1] & proto.ACTION_FLAG_PASSIVE  # flags byte carries the passive bit


@pytest.mark.unit
def test_default_input_has_no_passive_flag() -> None:
    cfg = BizHawkClientConfig(width=1, height=1, channels=1)
    ft = _FakeTransport()
    ft.feed(_hello_msg(1, 1, 1))
    ft.feed(_raw_frame_msg(0, 1, 1, 1, fill=0))
    ft.feed(_raw_frame_msg(1, 1, 1, 1, fill=1))
    client = BizHawkClient(cfg, transport=ft)
    client.reset()
    client.step(ControllerAction(right=True))
    action = next(m for m in _parse_sent(bytes(ft.sent)) if m[0] is proto.MsgType.ACTION)
    assert action[1][1] == 0  # no flags


@pytest.mark.unit
def test_client_truncates_at_max_episode_steps() -> None:
    cfg = BizHawkClientConfig(width=1, height=1, channels=1, max_episode_steps=1)
    ft = _FakeTransport()
    ft.feed(_hello_msg(1, 1, 1))
    ft.feed(_raw_frame_msg(0, 1, 1, 1, fill=0))
    ft.feed(_raw_frame_msg(1, 1, 1, 1, fill=1))
    client = BizHawkClient(cfg, transport=ft)
    client.reset()
    step = client.step(ControllerAction())
    assert step.truncated is True


@pytest.mark.unit
def test_second_reset_sends_reset_message() -> None:
    cfg = BizHawkClientConfig(width=1, height=1, channels=1)
    ft = _FakeTransport()
    ft.feed(_hello_msg(1, 1, 1))
    ft.feed(_raw_frame_msg(0, 1, 1, 1, fill=0))
    ft.feed(_raw_frame_msg(0, 1, 1, 1, fill=5))  # frame after RESET
    client = BizHawkClient(cfg, transport=ft)
    client.reset()
    obs = client.reset()
    assert obs.frame.data == bytes([5])
    sent = _parse_sent(bytes(ft.sent))
    assert [m[0] for m in sent] == [proto.MsgType.WELCOME, proto.MsgType.RESET]


@pytest.mark.unit
def test_step_before_reset_raises_state_error() -> None:
    client = BizHawkClient(BizHawkClientConfig(), transport=_FakeTransport())
    with pytest.raises(BizHawkStateError):
        client.step(ControllerAction())


@pytest.mark.unit
def test_client_adopts_emulator_geometry() -> None:
    # The emulator owns its buffer size; the client adopts whatever HELLO says.
    cfg = BizHawkClientConfig(width=2, height=2, channels=1)  # hint only
    ft = _FakeTransport()
    ft.feed(_hello_msg(8, 7, 3))  # emulator announces 8x7x3
    ft.feed(_raw_frame_msg(0, 8, 7, 3, fill=9))
    client = BizHawkClient(cfg, transport=ft)
    obs = client.reset()
    assert (obs.frame.width, obs.frame.height, obs.frame.channels) == (8, 7, 3)
    assert obs.frame.data == bytes([9]) * (8 * 7 * 3)
    # WELCOME echoes the adopted geometry back to the emulator.
    welcome = _parse_sent(bytes(ft.sent))[0]
    body = proto.decode_welcome(welcome[1])
    assert (body["width"], body["height"], body["channels"]) == (8, 7, 3)


@pytest.mark.unit
def test_invalid_announced_geometry_raises() -> None:
    ft = _FakeTransport()
    ft.feed(_hello_msg(8, 7, 2))  # channels=2 is invalid
    client = BizHawkClient(BizHawkClientConfig(), transport=ft)
    with pytest.raises(BizHawkProtocolError, match="invalid geometry"):
        client.reset()


@pytest.mark.unit
def test_version_mismatch_raises_protocol_error() -> None:
    cfg = BizHawkClientConfig(width=1, height=1, channels=1)
    ft = _FakeTransport()
    ft.feed(_hello_msg(1, 1, 1, version=999))
    client = BizHawkClient(cfg, transport=ft)
    with pytest.raises(BizHawkProtocolError, match="version"):
        client.reset()


@pytest.mark.unit
def test_unexpected_message_where_frame_expected() -> None:
    cfg = BizHawkClientConfig(width=1, height=1, channels=1)
    ft = _FakeTransport()
    ft.feed(_hello_msg(1, 1, 1))
    ft.feed(proto.encode_message(proto.MsgType.WELCOME, b"{}"))  # not a FRAME
    client = BizHawkClient(cfg, transport=ft)
    with pytest.raises(BizHawkProtocolError, match="FRAME"):
        client.reset()


@pytest.mark.unit
def test_bye_during_frame_raises_connection_error() -> None:
    cfg = BizHawkClientConfig(width=1, height=1, channels=1)
    ft = _FakeTransport()
    ft.feed(_hello_msg(1, 1, 1))
    ft.feed(proto.encode_message(proto.MsgType.BYE))
    client = BizHawkClient(cfg, transport=ft)
    with pytest.raises(BizHawkConnectionError, match="BYE"):
        client.reset()


@pytest.mark.unit
def test_bizhawk_client_is_game_client() -> None:
    assert isinstance(BizHawkClient(BizHawkClientConfig()), IGameClient)


#


@pytest.mark.unit
def test_client_over_real_socket_loopback() -> None:
    a, b = socket.socketpair()
    w, h, c = 300, 300, 1  # 90_000 bytes > 64 KiB -> forces recv-buffer growth
    captured: dict[str, object] = {}
    errors: list[BaseException] = []

    def fake_lua() -> None:
        try:
            t = SocketTransport(b)
            t.send(_hello_msg(w, h, c))
            assert _read_msg(t)[0] is proto.MsgType.WELCOME
            t.send(_raw_frame_msg(0, w, h, c, fill=7))
            msg_type, payload = _read_msg(t)
            assert msg_type is proto.MsgType.ACTION
            captured["action"] = proto.decode_action(payload)
            t.send(_raw_frame_msg(1, w, h, c, fill=8))
            captured["final"] = _read_msg(t)[0]  # CLOSE
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=fake_lua)
    thread.start()
    try:
        client = BizHawkClient(
            BizHawkClientConfig(width=w, height=h, channels=c),
            transport=SocketTransport(a),
        )
        obs = client.reset()
        assert obs.frame.data == bytes([7]) * (w * h * c)
        step = client.step(ControllerAction(right=True, b=True))
        assert step.observation.frame.data == bytes([8]) * (w * h * c)
        client.close()
    finally:
        thread.join(timeout=5.0)
        a.close()
        b.close()

    assert not errors, errors
    assert captured["action"] == ControllerAction(right=True, b=True)
    assert captured["final"] is proto.MsgType.CLOSE


#


@pytest.mark.unit
def test_client_over_file_transport(tmp_path: Path) -> None:
    from neuroforge.environments.games.clients.bizhawk.file_transport import FileTransport

    comm = tmp_path / "bridge"
    transport = FileTransport.create(str(comm), timeout=5.0)

    # Stage the messages the emulator would write (l2p/), in order.
    l2p = comm / "l2p"
    (l2p / "00000000.msg").write_bytes(_hello_msg(2, 2, 1))
    (l2p / "00000001.msg").write_bytes(_raw_frame_msg(0, 2, 2, 1, fill=10))
    (l2p / "00000002.msg").write_bytes(_raw_frame_msg(1, 2, 2, 1, fill=11))

    cfg = BizHawkClientConfig(width=2, height=2, channels=1, transport="file")
    client = BizHawkClient(cfg, transport=transport)

    obs0 = client.reset()
    assert obs0.frame.data == bytes([10]) * 4
    step = client.step(ControllerAction(left=True))
    assert step.observation.frame.data == bytes([11]) * 4

    # Inspect what the client wrote back (p2l/): WELCOME then ACTION.
    p2l = comm / "p2l"
    welcome_type, _ = proto.decode_header((p2l / "00000000.msg").read_bytes()[: proto.HEADER_SIZE])
    assert welcome_type is proto.MsgType.WELCOME
    action_bytes = (p2l / "00000001.msg").read_bytes()
    a_type, a_len = proto.decode_header(action_bytes[: proto.HEADER_SIZE])
    assert a_type is proto.MsgType.ACTION
    assert proto.decode_action(action_bytes[proto.HEADER_SIZE : proto.HEADER_SIZE + a_len]) == (
        ControllerAction(left=True)
    )


@pytest.mark.unit
def test_file_transport_times_out_when_no_message(tmp_path: Path) -> None:
    from neuroforge.environments.games.clients.bizhawk.file_transport import FileTransport

    transport = FileTransport.create(str(tmp_path / "bridge"), timeout=0.2)
    with pytest.raises(BizHawkConnectionError, match="timed out"):
        transport.recv_exactly(9)


#


@pytest.mark.unit
def test_screenshot_socket_receiver_reads_length_prefixed_payload() -> None:
    from neuroforge.environments.games.clients.bizhawk.screenshot_socket import (
        ScreenshotSocketReceiver,
    )

    receiver = ScreenshotSocketReceiver.serve("127.0.0.1", 0)
    errors: list[BaseException] = []

    def fake_bizhawk() -> None:
        try:
            with socket.create_connection((receiver.host, receiver.port), timeout=5.0) as sock:
                sock.sendall(b"5 hello")
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=fake_bizhawk)
    thread.start()
    try:
        assert receiver.recv_screenshot(timeout=5.0) == b"hello"
    finally:
        receiver.close()
        thread.join(timeout=5.0)
    assert not errors, errors


@pytest.mark.unit
def test_png_frame_decoded_by_client() -> None:
    torch = pytest.importorskip("torch")
    torchvision_io = pytest.importorskip("torchvision.io")
    from neuroforge.environments.games.clients.bizhawk.frame_codec import decode_png_to_raw

    w, h, c = 4, 6, 3
    image = torch.arange(c * h * w, dtype=torch.uint8).reshape(c, h, w)
    png_bytes = bytes(torchvision_io.encode_png(image).numpy().tobytes())

    # Direct codec round-trip: PNG -> row-major HWC bytes.
    raw = decode_png_to_raw(png_bytes, width=w, height=h, channels=c)
    expected = bytes(image.permute(1, 2, 0).contiguous().flatten().tolist())
    assert raw == expected

    # Through the client with format negotiated to PNG.
    ft = _FakeTransport()
    ft.feed(_hello_msg(w, h, c, fmt="png"))
    ft.feed(proto.encode_message(proto.MsgType.FRAME, proto.encode_frame_payload(0, 0, png_bytes)))
    client = BizHawkClient(BizHawkClientConfig(width=w, height=h, channels=c), transport=ft)
    obs = client.reset()
    assert obs.frame.data == expected


#


class _HoldRight:
    def act(self, observation: object) -> ControllerAction:
        return ControllerAction(right=True)


@pytest.mark.unit
def test_scripted_client_drives_loop() -> None:
    client = ScriptedGameClient(width=2, height=2, channels=1, max_steps=3)
    assert isinstance(client, IGameClient)
    loop = VisionOnlyGameLoop(client=client, policy=_HoldRight())
    transitions = list(loop.run(max_steps=10))
    loop.close()
    assert len(transitions) == 3
    assert transitions[-1].truncated is True
    assert client.closed is True
    assert all(a == ControllerAction(right=True) for a in client.actions)


@pytest.mark.unit
def test_replay_client_replays_recorded_frames() -> None:
    base = ScriptedGameClient(width=1, height=1, channels=1)
    frames = [base.reset().frame, base.step(ControllerAction()).observation.frame]
    client = ReplayGameClient(frames)
    assert isinstance(client, IGameClient)
    first = client.reset()
    assert first.frame is frames[0]
    step = client.step(ControllerAction())
    assert step.observation.frame is frames[1]
    assert step.truncated is True


@pytest.mark.unit
def test_action_progress_client_uses_full_dpad_surface() -> None:
    baseline = ActionProgressGameClient(width=8, height=8, channels=1, max_steps=4)
    baseline.reset()
    right_x = baseline.step(ControllerAction(right=True)).observation.metrics.x_progress

    climbing = ActionProgressGameClient(width=8, height=8, channels=1, max_steps=4)
    climbing.reset()
    up_right_x = climbing.step(
        ControllerAction(up=True, right=True),
    ).observation.metrics.x_progress

    crouching = ActionProgressGameClient(width=8, height=8, channels=1, max_steps=4)
    crouching.reset()
    down_right_x = crouching.step(
        ControllerAction(down=True, right=True),
    ).observation.metrics.x_progress

    assert up_right_x is not None
    assert right_x is not None
    assert down_right_x is not None
    assert up_right_x > right_x
    assert down_right_x < right_x


@pytest.mark.unit
def test_replay_client_requires_frames() -> None:
    with pytest.raises(ValueError, match="at least one frame"):
        ReplayGameClient([])


#


@pytest.mark.unit
def test_lua_bridge_has_no_memory_reads() -> None:
    lua_path = _LUA_BRIDGE
    assert lua_path.is_file(), f"missing Lua side-car at {lua_path}"
    # Strip full-line comments, then assert no RAM-access API survives in code.
    code_lines = [
        line for line in lua_path.read_text(encoding="utf-8").splitlines()
        if not line.lstrip().startswith("--")
    ]
    code = "\n".join(code_lines).lower()
    assert "memory" not in code, "Lua bridge must not access emulator RAM (vision-only)"
    # Sanity: it really is the bridge (captures screen, sets buttons).
    assert "client.screenshot" in code
    assert "joypad.set" in code


@pytest.mark.unit
def test_lua_bridge_reads_frameskip_env() -> None:
    lua_path = _LUA_BRIDGE
    assert "NEUROFORGE_BRIDGE_FRAMESKIP" in lua_path.read_text(encoding="utf-8")


@pytest.mark.unit
def test_lua_bridge_reads_speed_percent_env() -> None:
    lua_path = _LUA_BRIDGE
    code = lua_path.read_text(encoding="utf-8")
    assert "NEUROFORGE_BRIDGE_SPEED_PERCENT" in code
    assert "client.speedmode" in code
    assert "emu.limitframerate" in code


@pytest.mark.unit
def test_lua_bridge_supports_socket_backends() -> None:
    lua_path = _LUA_BRIDGE
    code = lua_path.read_text(encoding="utf-8")
    assert "NEUROFORGE_BRIDGE_TRANSPORT" in code
    assert "NEUROFORGE_BRIDGE_SOCKET_BACKEND" in code
    assert "NEUROFORGE_BRIDGE_FRAME_CAPTURE" in code
    assert "NEUROFORGE_BRIDGE_SCREENSHOT_PORT" in code
    assert "socket.core" in code  # BizHawk may only ship Lua/socket/core.dll
    assert "MSWinsock.Winsock" in code
    assert "comm.socketServerScreenShot" in code
    assert "SendData" in code
    assert "GetData" in code
    assert "socket_recv_message" in code


@pytest.mark.unit
def test_launcher_passes_frameskip_and_speed_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from neuroforge.environments.games.clients.bizhawk import launcher as lmod

    (tmp_path / "EmuHawk.exe").write_text("")
    (tmp_path / "bridge.lua").write_text("")
    captured: dict[str, object] = {}

    def _fake_popen(_cmd: object, env: object = None, **_kw: object) -> object:
        captured["env"] = env

        class _Proc:
            def poll(self) -> None:
                return None

        return _Proc()

    monkeypatch.setattr(
        "neuroforge.environments.games.clients.bizhawk.launcher.subprocess.Popen",
        _fake_popen,
    )
    lmod.EmuHawkLauncher(
        emuhawk_path=str(tmp_path / "EmuHawk.exe"),
        lua_script=str(tmp_path / "bridge.lua"),
        frameskip=4,
        speed_percent=400,
    ).launch(comm_dir=str(tmp_path))
    env = captured["env"]
    assert isinstance(env, dict)
    assert env[lmod.BRIDGE_FRAMESKIP_ENV] == "4"
    assert env[lmod.BRIDGE_SPEED_PERCENT_ENV] == "400"
    assert env[lmod.BRIDGE_TRANSPORT_ENV] == "file"


@pytest.mark.unit
def test_launcher_passes_socket_transport_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from neuroforge.environments.games.clients.bizhawk import launcher as lmod

    (tmp_path / "EmuHawk.exe").write_text("")
    (tmp_path / "bridge.lua").write_text("")
    rom_path = tmp_path / "game.nes"
    rom_path.write_text("")
    captured: dict[str, object] = {}

    def _fake_popen(_cmd: object, env: object = None, **_kw: object) -> object:
        captured["cmd"] = _cmd
        captured["env"] = env

        class _Proc:
            def poll(self) -> None:
                return None

        return _Proc()

    monkeypatch.setattr(
        "neuroforge.environments.games.clients.bizhawk.launcher.subprocess.Popen",
        _fake_popen,
    )
    lmod.EmuHawkLauncher(
        emuhawk_path=str(tmp_path / "EmuHawk.exe"),
        lua_script=str(tmp_path / "bridge.lua"),
        rom_path=str(rom_path),
        socket_backend="auto",
    ).launch(
        port=9999,
        host="127.0.0.1",
        screenshot_port=10001,
        screenshot_host="127.0.0.1",
    )
    env = captured["env"]
    assert isinstance(env, dict)
    assert env[lmod.BRIDGE_TRANSPORT_ENV] == "socket"
    assert env[lmod.BRIDGE_HOST_ENV] == "127.0.0.1"
    assert env[lmod.BRIDGE_PORT_ENV] == "9999"
    assert env[lmod.BRIDGE_SOCKET_BACKEND_ENV] == "auto"
    assert env[lmod.BRIDGE_FRAME_CAPTURE_ENV] == "socket"
    assert env[lmod.BRIDGE_SCREENSHOT_HOST_ENV] == "127.0.0.1"
    assert env[lmod.BRIDGE_SCREENSHOT_PORT_ENV] == "10001"
    cmd = captured["cmd"]
    assert isinstance(cmd, list)
    assert "--socket_ip=127.0.0.1" in cmd
    assert "--socket_port=10001" in cmd
    assert cmd.index("--socket_port=10001") < cmd.index(str(rom_path))
    assert cmd.index(f"--lua={tmp_path / 'bridge.lua'}") < cmd.index(str(rom_path))


@pytest.mark.unit
def test_lua_bridge_supports_savestate_reset() -> None:
    # Phase 4: RESET may load a savestate (an environment reset, not RAM reading)
    # so training can begin inside a level. savestate.load is not a memory API.
    lua_path = _LUA_BRIDGE
    code = lua_path.read_text(encoding="utf-8")
    assert "savestate.load" in code
    assert "reboot_core" in code  # still the fallback when no savestate is given


#


@pytest.mark.unit
def test_reset_payload_roundtrip() -> None:
    path = r"C:\BizHawk\States\smb3_level1.State"
    assert proto.decode_reset(proto.encode_reset(path)) == path


@pytest.mark.unit
def test_reset_empty_payload_means_reboot() -> None:
    assert proto.encode_reset("") == b""
    assert proto.decode_reset(b"") is None


@pytest.mark.unit
def test_first_reset_loads_queued_savestate() -> None:
    cfg = BizHawkClientConfig(width=1, height=1, channels=1)
    ft = _FakeTransport()
    ft.feed(_hello_msg(1, 1, 1))
    ft.feed(_raw_frame_msg(0, 1, 1, 1, fill=1))  # auto boot frame after WELCOME
    ft.feed(_raw_frame_msg(0, 1, 1, 1, fill=2))  # frame after savestate load
    client = BizHawkClient(cfg, transport=ft)

    client.queue_savestate(r"C:\states\one.State")
    obs = client.reset()

    assert obs.frame.data == bytes([2])  # the post-savestate frame, not the boot frame
    sent = _parse_sent(bytes(ft.sent))
    assert [m[0] for m in sent] == [proto.MsgType.WELCOME, proto.MsgType.RESET]
    assert proto.decode_reset(sent[1][1]) == r"C:\states\one.State"


@pytest.mark.unit
def test_first_reset_without_savestate_sends_no_reset() -> None:
    cfg = BizHawkClientConfig(width=1, height=1, channels=1)
    ft = _FakeTransport()
    ft.feed(_hello_msg(1, 1, 1))
    ft.feed(_raw_frame_msg(0, 1, 1, 1, fill=1))
    client = BizHawkClient(cfg, transport=ft)

    obs = client.reset()

    assert obs.frame.data == bytes([1])  # the boot frame is returned directly
    assert [m[0] for m in _parse_sent(bytes(ft.sent))] == [proto.MsgType.WELCOME]


@pytest.mark.unit
def test_second_reset_uses_queued_savestate() -> None:
    cfg = BizHawkClientConfig(width=1, height=1, channels=1)
    ft = _FakeTransport()
    ft.feed(_hello_msg(1, 1, 1))
    ft.feed(_raw_frame_msg(0, 1, 1, 1, fill=1))  # boot frame
    ft.feed(_raw_frame_msg(0, 1, 1, 1, fill=3))  # frame after savestate load
    client = BizHawkClient(cfg, transport=ft)

    client.reset()                                  # episode 0: boot
    client.queue_savestate(r"D:\s\level2.State")
    obs = client.reset()                            # episode 1: from savestate

    assert obs.frame.data == bytes([3])
    sent = _parse_sent(bytes(ft.sent))
    assert sent[-1][0] is proto.MsgType.RESET
    assert proto.decode_reset(sent[-1][1]) == r"D:\s\level2.State"
