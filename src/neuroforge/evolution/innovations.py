"""Innovation registry for structural neuroevolution (NEAT historical markings).

When two genomes independently make the *same* structural change (add the same
connection, or split the same connection into a node), they must receive the
*same* innovation number — that is what lets crossover align genes that share an
origin and lets speciation measure real structural distance. This registry hands
out stable innovation numbers keyed by the change's identity.

A single shared registry per evolution run keeps numbering consistent across the
whole population; it is deterministic given the order changes are first seen.
"""

from __future__ import annotations

__all__ = ["InnovationRegistry"]


class InnovationRegistry:
    """Allocate stable innovation numbers for connections and node-splits."""

    def __init__(self, *, start: int = 0) -> None:
        self._next = int(start)
        self._connections: dict[tuple[str, str], int] = {}
        self._splits: dict[int, tuple[int, int, int]] = {}

    def connection(self, src: str, dst: str) -> int:
        """Return the innovation number for a ``src -> dst`` connection."""
        key = (src, dst)
        existing = self._connections.get(key)
        if existing is not None:
            return existing
        number = self._take()
        self._connections[key] = number
        return number

    def node_split(self, connection_innovation: int) -> tuple[int, int, int]:
        """Return ``(node_id, in_innovation, out_innovation)`` for splitting a connection.

        Splitting the same connection always yields the same new node id and the
        same two replacement-connection innovations, so the structure is shared.
        """
        existing = self._splits.get(connection_innovation)
        if existing is not None:
            return existing
        triple = (self._take(), self._take(), self._take())
        self._splits[connection_innovation] = triple
        return triple

    @property
    def peek_next(self) -> int:
        """The next innovation number that would be allocated (for inspection)."""
        return self._next

    def _take(self) -> int:
        number = self._next
        self._next += 1
        return number
