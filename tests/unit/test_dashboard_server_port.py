"""Dashboard server bind-port helpers."""

from __future__ import annotations

import socket


def test_resolve_dashboard_port_skips_busy_port() -> None:
    import neuroforge.dashboard.server as srv

    host = "127.0.0.1"
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        sock.listen(1)
        busy_port = int(sock.getsockname()[1])

        resolved = srv._resolve_dashboard_port(host, busy_port, attempts=3)

    assert resolved != busy_port
    assert busy_port < resolved <= busy_port + 3
