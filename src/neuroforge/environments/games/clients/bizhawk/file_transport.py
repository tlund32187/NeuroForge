"""File-based :class:`ITransport` for BizHawk builds without LuaSocket.

Newer EmuHawk uses the NLua engine, which does **not** bundle LuaSocket â€” so
``require("socket")`` fails in the Lua side-car. This transport carries the
exact same framed protocol messages over atomically-renamed files in a scratch
directory, using only Lua's standard ``io``/``os`` (always available in NLua).

It is slower than a socket (Phase 6 optimises transport) but dependency-free
and reliable for bring-up. Because each side writes whole framed messages as
sequentially-numbered files and renames them into place atomically, a reader
never observes a half-written message.

Directory layout under ``comm_dir``::

    l2p/NNNNNNNN.msg   emulator -> Python   (Python reads, then deletes)
    p2l/NNNNNNNN.msg   Python -> emulator   (Python writes atomically)
"""

from __future__ import annotations

import contextlib
import os
import time
from pathlib import Path

from neuroforge.environments.games.clients.bizhawk.errors import BizHawkConnectionError

__all__ = ["FileTransport"]

_LARGE_TIMEOUT = 1e9


class FileTransport:
    """Move framed protocol messages over files (no sockets, no LuaSocket)."""

    def __init__(
        self,
        comm_dir: str | os.PathLike[str],
        *,
        poll_interval: float = 0.0005,
        timeout: float = 30.0,
    ) -> None:
        self._dir = Path(comm_dir)
        self._l2p = self._dir / "l2p"  # emulator -> Python (we read)
        self._p2l = self._dir / "p2l"  # Python -> emulator (we write)
        self._recv_buf = bytearray()
        self._recv_seq = 0
        self._send_seq = 0
        self._poll = float(poll_interval)
        self._timeout = float(timeout)
        self._closed = False

    @classmethod
    def create(
        cls, comm_dir: str | os.PathLike[str], *, timeout: float = 30.0,
    ) -> FileTransport:
        """Create the transport and prepare clean comm directories."""
        transport = cls(comm_dir, timeout=timeout)
        transport._prepare_dirs()
        return transport

    @property
    def comm_dir(self) -> str:
        return str(self._dir)

    def _prepare_dirs(self) -> None:
        for directory in (self._l2p, self._p2l):
            directory.mkdir(parents=True, exist_ok=True)
            for stale in (*directory.glob("*.msg"), *directory.glob("*.tmp")):
                with contextlib.suppress(OSError):
                    stale.unlink()

    #

    def set_timeout(self, seconds: float | None) -> None:
        self._timeout = _LARGE_TIMEOUT if seconds is None else float(seconds)

    def send(self, data: bytes) -> None:
        final = self._p2l / f"{self._send_seq:08d}.msg"
        tmp = self._p2l / f"{self._send_seq:08d}.tmp"
        deadline = time.monotonic() + self._timeout
        # Retry transient Windows sharing violations (AV/indexer) on write/rename.
        while True:
            try:
                tmp.write_bytes(data)
                os.replace(tmp, final)  # atomic publish
                break
            except OSError as exc:
                if time.monotonic() > deadline:
                    msg = f"file send failed: {exc}"
                    raise BizHawkConnectionError(msg) from exc
                time.sleep(self._poll)
        self._send_seq += 1

    def recv_exactly(self, n: int) -> bytes:
        if n <= 0:
            return b""
        while len(self._recv_buf) < n:
            self._read_next_message()
        out = bytes(self._recv_buf[:n])
        del self._recv_buf[:n]
        return out

    def _read_next_message(self) -> None:
        path = self._l2p / f"{self._recv_seq:08d}.msg"
        deadline = time.monotonic() + self._timeout
        data = b""
        # Poll until the message file appears AND is readable. On Windows a
        # just-created file can briefly raise PermissionError (AV/indexer holds
        # a scan lock) â€” that is transient, so we retry rather than fail.
        while True:
            try:
                data = path.read_bytes()
            except OSError:
                data = b""  # not present yet, or a transient sharing violation
            if data:
                break
            if time.monotonic() > deadline:
                msg = (
                    f"timed out waiting for emulator message {self._recv_seq} "
                    f"in {self._l2p}"
                )
                raise BizHawkConnectionError(msg)
            time.sleep(self._poll)
        self._recv_buf += data
        with contextlib.suppress(OSError):
            path.unlink()
        self._recv_seq += 1

    def close(self) -> None:
        self._closed = True
