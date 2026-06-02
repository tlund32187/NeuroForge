"""Error hierarchy for the emulator bridge.

A small, explicit hierarchy keeps failure modes distinguishable at call sites
without leaking transport details (Law of Demeter): connection/transport
faults, protocol/framing faults, and client-state misuse are separate types.
"""

from __future__ import annotations

__all__ = [
    "BridgeError",
    "BizHawkConnectionError",
    "BizHawkProtocolError",
    "BizHawkStateError",
]


class BridgeError(Exception):
    """Base class for all emulator-bridge errors."""


class BizHawkConnectionError(BridgeError):
    """Transport could not connect, timed out, or was closed by the peer."""


class BizHawkProtocolError(BridgeError):
    """A received message violated the wire protocol (bad magic, type, size)."""


class BizHawkStateError(BridgeError):
    """The client was used in an invalid order (e.g. ``step`` before ``reset``)."""
