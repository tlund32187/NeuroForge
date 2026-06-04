"""Optional EmuHawk process launcher.

Launching the emulator is a *separate concern* from the bridge protocol
(single responsibility), so it lives behind this small, opt-in object. The
default workflow is ``launch=False`` — the user starts EmuHawk and loads the
Lua side-car manually, because BIOS/firmware/ROM provisioning and per-machine
paths are environment-specific and not something CI can reproduce.

When used, the launcher starts EmuHawk only *after* the bridge server is bound
(via the ``on_bound`` seam) and passes launch-time bridge settings through
``NEUROFORGE_BRIDGE_*`` environment variables, which the Lua script reads.
"""

from __future__ import annotations

import os
import subprocess  # noqa: S404 — launching a user-specified emulator is the whole point
from dataclasses import dataclass, field
from pathlib import Path

from neuroforge.game.clients.errors import BizHawkConnectionError

__all__ = ["EmuHawkLauncher"]

#: Environment variable the Lua side-car reads to find the bridge port (socket).
BRIDGE_PORT_ENV: str = "NEUROFORGE_BRIDGE_PORT"
#: Environment variable the Lua side-car reads to find the bridge host (socket).
BRIDGE_HOST_ENV: str = "NEUROFORGE_BRIDGE_HOST"
#: Environment variable selecting the bridge transport: file or socket.
BRIDGE_TRANSPORT_ENV: str = "NEUROFORGE_BRIDGE_TRANSPORT"
#: Environment variable selecting the socket implementation: auto, luasocket, or luacom.
BRIDGE_SOCKET_BACKEND_ENV: str = "NEUROFORGE_BRIDGE_SOCKET_BACKEND"
#: Environment variable selecting frame capture mode: socket or file.
BRIDGE_FRAME_CAPTURE_ENV: str = "NEUROFORGE_BRIDGE_FRAME_CAPTURE"
#: Environment variable the Lua side-car reads for BizHawk screenshot socket host.
BRIDGE_SCREENSHOT_HOST_ENV: str = "NEUROFORGE_BRIDGE_SCREENSHOT_HOST"
#: Environment variable the Lua side-car reads for BizHawk screenshot socket port.
BRIDGE_SCREENSHOT_PORT_ENV: str = "NEUROFORGE_BRIDGE_SCREENSHOT_PORT"
#: Optional Lua-side crash log path for diagnosing bridge startup failures.
BRIDGE_ERROR_PATH_ENV: str = "NEUROFORGE_BRIDGE_ERROR_PATH"
#: Environment variable the Lua side-car reads to find the comm dir (file transport).
BRIDGE_DIR_ENV: str = "NEUROFORGE_BRIDGE_DIR"
#: Environment variable: emulator frames advanced per captured frame (action-repeat).
#: >1 holds each action for that many emulator frames and only captures one — the
#: big wall-clock lever (fewer IPC round-trips + SNN steps per second of gameplay)
#: and it doubles as action-repeat, which helps a side-scroller build run momentum.
BRIDGE_FRAMESKIP_ENV: str = "NEUROFORGE_BRIDGE_FRAMESKIP"
#: Environment variable: BizHawk client speed percent (e.g. 400 = 400% / 4x).
#: 0 means leave the user's current EmuHawk speed setting untouched.
BRIDGE_SPEED_PERCENT_ENV: str = "NEUROFORGE_BRIDGE_SPEED_PERCENT"


@dataclass
class EmuHawkLauncher:
    """Launch (and later terminate) an EmuHawk process running the Lua bridge.

    Attributes
    ----------
    emuhawk_path:
        Path to ``EmuHawk.exe``.
    lua_script:
        Path to ``neuroforge_bridge.lua``.
    rom_path:
        Optional ROM to open on launch.
    extra_args:
        Extra command-line arguments appended verbatim.
    """

    emuhawk_path: str
    lua_script: str
    rom_path: str | None = None
    extra_args: tuple[str, ...] = ()
    frameskip: int = 1  # emulator frames per captured frame (action-repeat / speed)
    speed_percent: int = 0  # BizHawk throttle target percent; 0 leaves current setting
    socket_backend: str = "auto"  # auto -> LuaSocket first, then LuaCOM/MSWinsock
    _process: subprocess.Popen[bytes] | None = field(default=None, init=False, repr=False)

    def launch(
        self,
        *,
        port: int | None = None,
        comm_dir: str | None = None,
        host: str | None = None,
        screenshot_port: int | None = None,
        screenshot_host: str | None = None,
        error_path: str | None = None,
    ) -> None:
        """Start EmuHawk with the Lua side-car.

        The bridge endpoint is advertised to the Lua script via environment
        variables: ``port`` (socket transport) and/or ``comm_dir`` (file
        transport).
        """
        exe = Path(self.emuhawk_path)
        script = Path(self.lua_script)
        if not exe.is_file():
            msg = f"EmuHawk executable not found: {exe}"
            raise BizHawkConnectionError(msg)
        if not script.is_file():
            msg = f"Lua bridge script not found: {script}"
            raise BizHawkConnectionError(msg)

        cmd: list[str] = [str(exe)]

        env = dict(os.environ)
        if port is not None:
            env[BRIDGE_PORT_ENV] = str(int(port))
            env[BRIDGE_HOST_ENV] = host or "127.0.0.1"
            env[BRIDGE_TRANSPORT_ENV] = "socket"
            env[BRIDGE_SOCKET_BACKEND_ENV] = self.socket_backend
        if screenshot_port is not None:
            shot_host = screenshot_host or host or "127.0.0.1"
            env[BRIDGE_FRAME_CAPTURE_ENV] = "socket"
            env[BRIDGE_SCREENSHOT_HOST_ENV] = shot_host
            env[BRIDGE_SCREENSHOT_PORT_ENV] = str(int(screenshot_port))
            cmd.extend([f"--socket_ip={shot_host}", f"--socket_port={int(screenshot_port)}"])
        if error_path is not None:
            env[BRIDGE_ERROR_PATH_ENV] = str(error_path)
        if comm_dir is not None:
            env[BRIDGE_DIR_ENV] = str(comm_dir)
            env.setdefault(BRIDGE_TRANSPORT_ENV, "file")
            env.setdefault(BRIDGE_FRAME_CAPTURE_ENV, "file")
        env[BRIDGE_FRAMESKIP_ENV] = str(max(1, int(self.frameskip)))
        speed_percent = max(0, int(self.speed_percent))
        if speed_percent > 0:
            env[BRIDGE_SPEED_PERCENT_ENV] = str(speed_percent)
        cmd.extend(self.extra_args)
        cmd.append(f"--lua={script}")
        if self.rom_path:
            cmd.append(str(self.rom_path))
        try:
            self._process = subprocess.Popen(cmd, env=env)  # noqa: S603 — user-supplied paths
        except OSError as exc:
            msg = f"failed to launch EmuHawk: {exc}"
            raise BizHawkConnectionError(msg) from exc

    def close(self) -> None:
        """Terminate the launched process if it is still running (idempotent)."""
        process = self._process
        if process is None:
            return
        self._process = None
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                process.kill()
