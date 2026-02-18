"""NeuroForge dashboard — web-based visualization of SNN training."""

from __future__ import annotations

__all__ = ["run_dashboard"]


def run_dashboard(
    *,
    host: str = "127.0.0.1",
    port: int = 8050,
) -> None:
    """Launch the dashboard web server.

    Parameters
    ----------
    host:
        Bind address.
    port:
        Bind port.
    """
    from neuroforge.dashboard.server import start_server

    start_server(host=host, port=port)
