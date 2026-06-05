"""CPPN graph used by HyperNEAT genomes."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any

__all__ = ["ACTIVATIONS", "CPPN", "CPPNConn", "CPPNNode"]

ACTIVATIONS: tuple[str, ...] = (
    "identity",
    "linear",
    "tanh",
    "sigmoid",
    "relu",
    "sin",
    "gauss",
)


@dataclass(frozen=True, slots=True)
class CPPNNode:
    """One CPPN node."""

    id: str
    kind: str
    activation: str = "tanh"


@dataclass(frozen=True, slots=True)
class CPPNConn:
    """One directed CPPN connection."""

    innovation: int
    src: str
    dst: str
    weight: float = 1.0
    enabled: bool = True


@dataclass(frozen=True, slots=True)
class CPPN:
    """Small feed-forward graph queried over substrate coordinates."""

    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    nodes: tuple[CPPNNode, ...]
    connections: tuple[CPPNConn, ...]

    def hidden_nodes(self) -> tuple[CPPNNode, ...]:
        """Hidden CPPN nodes only."""
        return tuple(node for node in self.nodes if node.kind == "hidden")

    def content_key(self) -> str:
        """Stable key over graph structure."""
        node_part = ";".join(
            f"{n.id}:{n.kind}:{n.activation}" for n in sorted(self.nodes, key=lambda n: n.id)
        )
        conn_part = ";".join(
            f"{c.innovation}:{c.src}->{c.dst}:{c.weight:.6f}:{int(c.enabled)}"
            for c in sorted(self.connections, key=lambda c: c.innovation)
        )
        return f"N[{node_part}]|C[{conn_part}]"

    def query(self, coords: Any, *, torch: Any) -> Any:
        """Evaluate the CPPN for a batch of coordinate features."""
        if int(coords.shape[1]) != len(self.inputs):
            msg = f"expected {len(self.inputs)} query features, got {int(coords.shape[1])}"
            raise ValueError(msg)

        values: dict[str, Any] = {
            name: coords[:, idx] for idx, name in enumerate(self.inputs)
        }
        incoming: dict[str, list[CPPNConn]] = {}
        for conn in self.connections:
            if conn.enabled:
                incoming.setdefault(conn.dst, []).append(conn)

        for node in self.nodes:
            if node.kind == "input":
                continue
            total = torch.zeros(coords.shape[0], device=coords.device, dtype=coords.dtype)
            for conn in incoming.get(node.id, []):
                if conn.src in values:
                    total = total + values[conn.src] * float(conn.weight)
            values[node.id] = _activate(total, node.activation, torch=torch)

        return torch.stack([values[name] for name in self.outputs], dim=1)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation."""
        return {
            "inputs": list(self.inputs),
            "outputs": list(self.outputs),
            "nodes": [dataclasses.asdict(node) for node in self.nodes],
            "connections": [dataclasses.asdict(conn) for conn in self.connections],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> CPPN:
        """Decode a CPPN from :meth:`to_dict` output."""
        return cls(
            inputs=tuple(str(item) for item in payload.get("inputs", [])),
            outputs=tuple(str(item) for item in payload.get("outputs", [])),
            nodes=tuple(
                CPPNNode(
                    id=str(item.get("id", item.get("name", ""))),
                    kind=str(item["kind"]),
                    activation=str(item.get("activation", "tanh")),
                )
                for item in payload.get("nodes", [])
            ),
            connections=tuple(
                CPPNConn(
                    innovation=int(item["innovation"]),
                    src=str(item["src"]),
                    dst=str(item["dst"]),
                    weight=float(item.get("weight", 1.0)),
                    enabled=bool(item.get("enabled", True)),
                )
                for item in payload.get("connections", [])
            ),
        )


def _activate(value: Any, activation: str, *, torch: Any) -> Any:
    if activation in {"identity", "linear"}:
        return value
    if activation == "tanh":
        return torch.tanh(value)
    if activation == "sigmoid":
        return torch.sigmoid(value) * 2.0 - 1.0
    if activation == "relu":
        return torch.clamp(value, min=0.0, max=1.0)
    if activation == "sin":
        return torch.sin(value)
    if activation == "gauss":
        return torch.exp(-(value * value))
    msg = f"unknown CPPN activation {activation!r}; expected one of {ACTIVATIONS}"
    raise ValueError(msg)


def activation_for_index(index: int) -> str:
    """Deterministically choose a hidden activation."""
    return ACTIVATIONS[2 + (index % (len(ACTIVATIONS) - 2))]
