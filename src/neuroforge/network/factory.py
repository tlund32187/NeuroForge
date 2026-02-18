"""NetworkFactory — config-driven network construction.

Builds a :class:`CoreEngine` from a declarative :class:`NetworkSpec`
using injected registries.  No file I/O, no global singletons beyond
what the caller provides.

Usage::

    factory = NetworkFactory(neuron_registry, synapse_registry)
    engine  = factory.build(spec, device="cpu", dtype="float32", seed=42)
"""

from __future__ import annotations

import random as _random
from typing import TYPE_CHECKING, Any

from neuroforge.contracts.simulation import SimulationConfig
from neuroforge.contracts.synapses import SynapseTopology
from neuroforge.engine.core_engine import CoreEngine, Population, Projection

if TYPE_CHECKING:
    from neuroforge.contracts.factories import IRegistry
    from neuroforge.network.specs import NetworkSpec

__all__ = ["NetworkFactory", "to_topology_json"]


class NetworkFactory:
    """Builds a :class:`CoreEngine` from a :class:`NetworkSpec`.

    Parameters
    ----------
    neuron_registry:
        Registry mapping neuron model keys to constructors.
    synapse_registry:
        Registry mapping synapse model keys to constructors.
    learning_registry:
        Optional registry for learning rules (reserved for future use).
    """

    def __init__(
        self,
        neuron_registry: IRegistry[Any],
        synapse_registry: IRegistry[Any],
        learning_registry: IRegistry[Any] | None = None,
    ) -> None:
        self._neuron_reg = neuron_registry
        self._synapse_reg = synapse_registry
        self._learning_reg = learning_registry

    # ── public API ──────────────────────────────────────────────────

    def build(
        self,
        spec: NetworkSpec,
        *,
        device: str,
        dtype: str,
        seed: int | None = None,
    ) -> CoreEngine:
        """Build and return a fully initialised :class:`CoreEngine`.

        All population state tensors are allocated on *device*/*dtype*.
        Weight matrices and optional biases are stored in each
        projection's ``state`` dict under the keys ``"weight_matrix"``
        and ``"bias"``.

        Parameters
        ----------
        spec:
            Declarative network specification.
        device:
            Target torch device string (``"cpu"`` or ``"cuda"``).
        dtype:
            Torch dtype string (``"float32"``, ``"float64"``).
        seed:
            Optional random seed for reproducible weight initialisation.

        Returns
        -------
        CoreEngine:
            Fully built engine (``engine._built is True``).
        """
        from neuroforge.core.torch_utils import require_torch, resolve_device_dtype

        torch = require_torch()

        if seed is not None:
            _random.seed(seed)
            torch.manual_seed(seed)

        dev, tdt = resolve_device_dtype(device, dtype)

        config = SimulationConfig(
            device=device,
            dtype=dtype,
            seed=seed if seed is not None else 42,
        )
        engine = CoreEngine(config)

        # ── populations ─────────────────────────────────────────────
        pop_sizes: dict[str, int] = {}
        for ps in spec.populations:
            neuron_kw: dict[str, object] = dict(ps.neuron_params)
            neuron_model = self._neuron_reg.create(ps.neuron_model, **neuron_kw)
            pop = Population(name=ps.name, model=neuron_model, n=ps.n)
            pop.state = neuron_model.init_state(ps.n, device, dtype)
            engine.add_population(pop)
            pop_sizes[ps.name] = ps.n

        # ── projections ─────────────────────────────────────────────
        for prj in spec.projections:
            if prj.source not in pop_sizes:
                msg = f"Projection {prj.name!r}: source {prj.source!r} not found"
                raise ValueError(msg)
            if prj.target not in pop_sizes:
                msg = f"Projection {prj.name!r}: target {prj.target!r} not found"
                raise ValueError(msg)

            synapse_kw: dict[str, object] = dict(prj.synapse_params)
            synapse_model = self._synapse_reg.create(prj.synapse_model, **synapse_kw)

            n_pre = pop_sizes[prj.source]
            n_post = pop_sizes[prj.target]

            topo_cfg = dict(prj.topology) if prj.topology else {}
            topo_type = str(topo_cfg.get("type", "dense"))

            if topo_type == "dense":
                weight_matrix, bias = _make_dense_weights(
                    n_pre, n_post, topo_cfg, dev, tdt, torch,
                )
            else:
                msg = f"Unsupported topology type: {topo_type!r}"
                raise ValueError(msg)

            # Build an edge-list SynapseTopology for synapse-model compat.
            edge_topo = _dense_edge_topology(
                weight_matrix, n_pre, n_post, dev, torch,
            )

            proj = Projection(
                name=prj.name,
                model=synapse_model,
                source=prj.source,
                target=prj.target,
                topology=edge_topo,
            )
            # Init synapse-model state (e.g. empty for StaticSynapseModel).
            proj.state = synapse_model.init_state(edge_topo, device, dtype)

            # Store trainable weight matrix and optional bias.
            proj.state["weight_matrix"] = weight_matrix
            if bias is not None:
                proj.state["bias"] = bias

            engine.add_projection(proj)

        # Mark engine as built (states are already initialised).
        engine._clock.reset()  # pyright: ignore[reportPrivateUsage]
        engine._built = True  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
        return engine


# ── topology helpers ────────────────────────────────────────────────


def to_topology_json(engine: CoreEngine) -> dict[str, Any]:
    """Return a JSON-friendly topology dict from *engine*.

    The format is compatible with the existing ``TOPOLOGY`` event::

        {
            "layers": ["input(8)", "hidden(24)", "output(1)"],
            "edges": [
                {"src": "input", "dst": "hidden", "weights": <Tensor>},
                ...
            ],
        }
    """
    layers: list[str] = [
        f"{name}({pop.n})" for name, pop in engine.populations.items()
    ]

    edges: list[dict[str, Any]] = []
    for _name, proj in engine.projections.items():
        edge: dict[str, Any] = {
            "src": proj.source,
            "dst": proj.target,
        }
        wm = proj.state.get("weight_matrix")
        if wm is not None:
            edge["weights"] = wm.detach()
        edges.append(edge)

    return {"layers": layers, "edges": edges}


# ── private helpers ─────────────────────────────────────────────────


def _make_dense_weights(
    n_pre: int,
    n_post: int,
    cfg: dict[str, Any],
    dev: Any,
    tdt: Any,
    torch: Any,
) -> tuple[Any, Any | None]:
    """Create a dense weight matrix ``[n_pre, n_post]`` and optional bias.

    When *n_post* is 1 the weight matrix is stored as a 1-D tensor
    ``[n_pre]`` so that downstream code can use it without squeezing.
    """
    init = str(cfg.get("init", "uniform"))
    squeeze = n_post == 1

    if init == "uniform":
        low = float(cfg.get("low", -0.3))
        high = float(cfg.get("high", 0.3))
        if squeeze:
            weight_matrix = torch.empty(n_pre, dtype=tdt, device=dev).uniform_(low, high)
        else:
            weight_matrix = torch.empty(n_pre, n_post, dtype=tdt, device=dev).uniform_(
                low, high,
            )
    elif init == "zeros":
        if squeeze:
            weight_matrix = torch.zeros(n_pre, dtype=tdt, device=dev)
        else:
            weight_matrix = torch.zeros(n_pre, n_post, dtype=tdt, device=dev)
    else:
        msg = f"Unsupported weight init: {init!r}"
        raise ValueError(msg)

    bias: Any = None
    if cfg.get("bias", False):
        bias = torch.zeros(n_post, dtype=tdt, device=dev)

    return weight_matrix, bias


def _dense_edge_topology(
    weight_matrix: Any,
    n_pre: int,
    n_post: int,
    dev: Any,
    torch: Any,
) -> SynapseTopology:
    """Create a :class:`SynapseTopology` edge-list from a dense matrix."""
    pre_idx = torch.arange(n_pre, device=dev).repeat_interleave(n_post)
    post_idx = torch.arange(n_post, device=dev).repeat(n_pre)
    flat_w = weight_matrix.detach().reshape(-1)
    delays = torch.zeros(n_pre * n_post, dtype=torch.int64, device=dev)

    return SynapseTopology(
        pre_idx=pre_idx,
        post_idx=post_idx,
        weights=flat_w,
        delays=delays,
        n_pre=n_pre,
        n_post=n_post,
    )
