"""Socket receiver for BizHawk's ``comm.socketServerScreenShot`` payloads."""

from __future__ import annotations

import contextlib
import socket
import time

from neuroforge.game.clients.errors import BizHawkConnectionError

__all__ = ["ScreenshotSocketReceiver"]


class ScreenshotSocketReceiver:
    """Accept BizHawk screenshot payloads on a dedicated localhost socket.

    BizHawk's ``comm.socketServerScreenShot()`` sends a length-prefixed PNG
    payload. Keeping screenshot bytes on this side channel lets the main bridge
    socket continue carrying the NeuroForge control protocol.
    """

    def __init__(self, server: socket.socket, *, host: str, port: int) -> None:
        self._server = server
        self.host = host
        self.port = int(port)
        self._conn: socket.socket | None = None
        self._closed = False

    @classmethod
    def serve(cls, host: str, port: int = 0) -> ScreenshotSocketReceiver:
        """Bind/listen and return a receiver ready for the emulator."""
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((host, int(port)))
            server.listen(1)
            bound_host, bound_port = server.getsockname()[:2]
        except OSError as exc:
            server.close()
            msg = f"could not bind screenshot socket on {host}:{port}: {exc}"
            raise BizHawkConnectionError(msg) from exc
        return cls(server, host=str(bound_host), port=int(bound_port))

    def recv_screenshot(self, *, timeout: float) -> bytes:
        """Read one length-prefixed screenshot payload."""
        deadline = time.monotonic() + float(timeout)
        while True:
            conn = self._ensure_conn(deadline)
            try:
                return self._read_length_prefixed(conn, deadline)
            except BizHawkConnectionError:
                self._drop_conn()
                if time.monotonic() >= deadline:
                    raise

    def close(self) -> None:
        """Release sockets (idempotent)."""
        if self._closed:
            return
        self._closed = True
        self._drop_conn()
        with contextlib.suppress(OSError):
            self._server.close()

    def _ensure_conn(self, deadline: float) -> socket.socket:
        if self._conn is not None:
            return self._conn
        remaining = max(0.001, deadline - time.monotonic())
        self._server.settimeout(remaining)
        try:
            conn, _ = self._server.accept()
        except TimeoutError as exc:
            msg = f"timed out waiting for BizHawk screenshot socket on {self.host}:{self.port}"
            raise BizHawkConnectionError(msg) from exc
        except OSError as exc:
            msg = f"screenshot socket accept failed on {self.host}:{self.port}: {exc}"
            raise BizHawkConnectionError(msg) from exc
        with contextlib.suppress(OSError):
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._conn = conn
        return conn

    def _drop_conn(self) -> None:
        conn = self._conn
        self._conn = None
        if conn is not None:
            with contextlib.suppress(OSError):
                conn.close()

    def _read_length_prefixed(self, conn: socket.socket, deadline: float) -> bytes:
        prefix = bytearray()
        first_byte_deadline = min(deadline, time.monotonic() + 2.0)
        while True:
            b = self._recv_exactly(conn, 1, first_byte_deadline if not prefix else deadline)
            if b == b" ":
                break
            if not b.isdigit():
                msg = f"invalid screenshot length prefix byte: {b!r}"
                raise BizHawkConnectionError(msg)
            prefix.extend(b)
            if len(prefix) > 12:
                msg = "screenshot length prefix is too long"
                raise BizHawkConnectionError(msg)
        if not prefix:
            msg = "empty screenshot length prefix"
            raise BizHawkConnectionError(msg)
        length = int(prefix.decode("ascii"))
        if length <= 0:
            msg = f"invalid screenshot payload length: {length}"
            raise BizHawkConnectionError(msg)
        return self._recv_exactly(conn, length, deadline)

    def _recv_exactly(self, conn: socket.socket, n: int, deadline: float) -> bytes:
        chunks: list[bytes] = []
        remaining = n
        while remaining:
            conn.settimeout(max(0.001, deadline - time.monotonic()))
            try:
                chunk = conn.recv(remaining)
            except TimeoutError as exc:
                msg = f"timed out reading screenshot payload ({n - remaining}/{n} bytes)"
                raise BizHawkConnectionError(msg) from exc
            except OSError as exc:
                msg = f"screenshot socket recv failed: {exc}"
                raise BizHawkConnectionError(msg) from exc
            if not chunk:
                msg = f"screenshot socket closed mid-payload ({n - remaining}/{n} bytes)"
                raise BizHawkConnectionError(msg)
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)
