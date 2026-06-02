"""Byte-transport for the emulator bridge.

The transport owns *bytes*, not protocol semantics (separation of concerns):
it sends framed messages and reads an exact number of bytes back. The
:class:`BizHawkClient` layers the wire protocol on top of this thin seam, so
the same client logic runs over a real socket or an in-memory fake.

:class:`SocketTransport` reuses a single receive buffer (``recv_into``) so the
steady-state per-frame path performs no new heap allocation for the read; the
only unavoidable copy is the immutable ``bytes`` handed to ``ScreenFrame``.
"""

from __future__ import annotations

import contextlib
import socket
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from neuroforge.game.clients.errors import BizHawkConnectionError

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = ["ITransport", "SocketTransport"]


@runtime_checkable
class ITransport(Protocol):
    """A bidirectional byte channel."""

    def send(self, data: bytes) -> None:
        """Send all of *data*."""
        ...

    def recv_exactly(self, n: int) -> bytes:
        """Block until exactly *n* bytes are read, or raise on disconnect."""
        ...

    def close(self) -> None:
        """Release the channel (idempotent)."""
        ...


class SocketTransport:
    """A :class:`ITransport` over a connected stream socket."""

    def __init__(self, sock: socket.socket) -> None:
        self._sock = sock
        # Not a TCP socket (e.g. socketpair on some platforms) — ignore.
        with contextlib.suppress(OSError):
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._buf = bytearray(1 << 16)
        self._closed = False

    # ── construction helpers ─────────────────────────────────────────

    @classmethod
    def serve(
        cls,
        host: str,
        port: int,
        *,
        accept_timeout: float,
        on_bound: Callable[[int], None] | None = None,
    ) -> SocketTransport:
        """Bind/listen on ``host:port``, accept one peer, and wrap it.

        ``on_bound(port)`` (if given) is invoked once the server socket is
        bound but before ``accept`` blocks — the seam an emulator launcher uses
        to start EmuHawk only after the port is known and listening.
        """
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((host, int(port)))
            server.listen(1)
            bound_port = int(server.getsockname()[1])
            if on_bound is not None:
                on_bound(bound_port)
            server.settimeout(float(accept_timeout))
            try:
                conn, _ = server.accept()
            except TimeoutError as exc:
                msg = f"timed out waiting for emulator to connect on {host}:{bound_port}"
                raise BizHawkConnectionError(msg) from exc
            except OSError as exc:
                msg = f"accept failed on {host}:{bound_port}: {exc}"
                raise BizHawkConnectionError(msg) from exc
        finally:
            server.close()
        return cls(conn)

    @classmethod
    def connect(cls, host: str, port: int, *, timeout: float = 10.0) -> SocketTransport:
        """Connect to a listening peer (used by the loopback/fake-Lua side)."""
        try:
            sock = socket.create_connection((host, int(port)), timeout=float(timeout))
        except OSError as exc:
            msg = f"could not connect to {host}:{port}: {exc}"
            raise BizHawkConnectionError(msg) from exc
        return cls(sock)

    # ── ITransport ────────────────────────────────────────────────────

    def set_timeout(self, seconds: float | None) -> None:
        """Set the per-operation socket timeout (``None`` blocks indefinitely)."""
        self._sock.settimeout(seconds)

    def send(self, data: bytes) -> None:
        try:
            self._sock.sendall(data)
        except OSError as exc:
            msg = f"send failed: {exc}"
            raise BizHawkConnectionError(msg) from exc

    def recv_exactly(self, n: int) -> bytes:
        if n <= 0:
            return b""
        if len(self._buf) < n:
            self._buf = bytearray(n)
        view = memoryview(self._buf)
        got = 0
        while got < n:
            try:
                read = self._sock.recv_into(view[got:n])
            except TimeoutError as exc:
                msg = f"timed out reading {n} bytes (got {got})"
                raise BizHawkConnectionError(msg) from exc
            except OSError as exc:
                msg = f"recv failed: {exc}"
                raise BizHawkConnectionError(msg) from exc
            if read == 0:
                msg = f"connection closed mid-message ({got}/{n} bytes)"
                raise BizHawkConnectionError(msg)
            got += read
        return bytes(view[:n])

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        with contextlib.suppress(OSError):
            self._sock.close()
