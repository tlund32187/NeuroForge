"""Gate-network builder — high-level helper for logic-gate SNNs.

Accepts a :class:`GateNetworkSpec` and returns a ready-to-train
:class:`GateNetwork` with the engine, trainable parameters, and
metadata.

The builder delegates to focused helper functions that can be tested
and reused independently:

- :func:`build_dale_signs` — Dale's-Law excitatory/inhibitory masks.
- :func:`init_projection_weights` — trainable weight & bias allocation.
- :func:`build_projection` — full projection assembly (weights +
  topology + synapse model).

Usage::

    gn = build_gate_network(GateNetworkSpec(hidden_size=6))
    loss.backward(); optimizer.step()  # gn.trainables has the tensors
"""

from __future__ import annotations

import random as _random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from neuroforge.contracts.simulation import SimulationConfig
from neuroforge.contracts.synapses import SynapseTopology
from neuroforge.engine.core_engine import CoreEngine, Population, Projection

if TYPE_CHECKING:
    from neuroforge.network.specs import GateNetworkSpec

__all__ = [
    "GateNetwork",
    "build_dale_signs",
    "build_gate_network",
    "build_projection",
    "init_projection_weights",
]


# ── GateNetwork result object ──────────────────────────────────────


@dataclass
class GateNetwork:
    """A trainable gate network produced by :func:`build_gate_network`.

    Attributes
    ----------
    engine:
        Fully built (but not yet run) :class:`CoreEngine`.
    trainables:
        Mapping of parameter names to raw weight / bias tensors.
        Every tensor has ``requires_grad=True``.
    meta:
        Extra information (layer sizes, Dale sign masks, etc.).
    """

    engine: CoreEngine
    trainables: dict[str, Any] = field(default_factory=lambda: {})
    meta: dict[str, Any] = field(default_factory=lambda: {})


# ── topology helpers ────────────────────────────────────────────────


def _sparse_edge_topology(
    weight_tensor: Any,
    n_pre: int,
    n_post: int,
    p_connect: float,
    dev: Any,
    torch_mod: Any,
    rng: Any,
) -> tuple[Any, Any]:
    """Create a sparse edge-list topology by sampling edges.

    Returns ``(SynapseTopology, keep_idx)``.
    """
    squeeze = n_post == 1
    n_edges_full = n_pre * n_post
    keep = torch_mod.rand(n_edges_full, generator=rng) < p_connect
    keep_idx = keep.nonzero(as_tuple=False).squeeze(1).to(dev)

    pre_full = torch_mod.arange(n_pre, device=dev).repeat_interleave(n_post)
    post_full = torch_mod.arange(n_post, device=dev).repeat(n_pre)

    pre_idx = pre_full[keep_idx]
    post_idx = post_full[keep_idx]

    flat_w = weight_tensor.detach().reshape(-1)
    selected_w = flat_w[keep_idx]
    delays = torch_mod.zeros(keep_idx.shape[0], dtype=torch_mod.int64, device=dev)

    topo = SynapseTopology(
        pre_idx=pre_idx,
        post_idx=post_idx,
        weights=selected_w,
        delays=delays,
        n_pre=n_pre,
        n_post=n_post,
    )
    _ = squeeze  # reserved for future squeeze handling
    return topo, keep_idx


def _dense_edge_topology(
    weight_matrix: Any,
    n_pre: int,
    n_post: int,
    dev: Any,
    torch_mod: Any,
) -> SynapseTopology:
    """Create a :class:`SynapseTopology` edge-list from a dense matrix."""
    pre_idx = torch_mod.arange(n_pre, device=dev).repeat_interleave(n_post)
    post_idx = torch_mod.arange(n_post, device=dev).repeat(n_pre)
    flat_w = weight_matrix.detach().reshape(-1)
    delays = torch_mod.zeros(n_pre * n_post, dtype=torch_mod.int64, device=dev)

    return SynapseTopology(
        pre_idx=pre_idx,
        post_idx=post_idx,
        weights=flat_w,
        delays=delays,
        n_pre=n_pre,
        n_post=n_post,
    )


# ── extracted helpers (SRP) ─────────────────────────────────────────


def build_dale_signs(
    n_input: int,
    n_hidden: int,
    n_inhibitory: int,
    dev: Any,
    tdt: Any,
    torch_mod: Any,
) -> dict[str, Any]:
    """Build Dale's-Law excitatory / inhibitory sign masks.

    Parameters
    ----------
    n_input:
        Number of input neurons (always excitatory).
    n_hidden:
        Number of hidden neurons.  If 0 no ``sign_hidden`` key is
        produced.
    n_inhibitory:
        How many of the hidden neurons are inhibitory.
    dev:
        Resolved torch device.
    tdt:
        Resolved torch dtype.
    torch_mod:
        The ``torch`` module.

    Returns
    -------
    dict[str, Any]:
        Always contains ``sign_input``; contains ``sign_hidden`` when
        *n_hidden* > 0.
    """
    signs: dict[str, Any] = {
        "sign_input": torch_mod.ones(n_input, device=dev, dtype=tdt),
    }
    if n_hidden > 0:
        n_exc = max(0, n_hidden - n_inhibitory)
        sign_hidden = torch_mod.ones(n_hidden, device=dev, dtype=tdt)
        sign_hidden[n_exc:] = -1.0
        signs["sign_hidden"] = sign_hidden
    return signs


def init_projection_weights(
    n_pre: int,
    n_post: int,
    scale: float,
    tdt: Any,
    dev: Any,
    torch_mod: Any,
) -> tuple[Any, Any]:
    """Allocate trainable weight and bias tensors.

    Weights are drawn from ``Uniform(−scale, +scale)`` on CPU then
    moved to *dev*.  Both tensors have ``requires_grad=True``.

    Parameters
    ----------
    n_pre, n_post:
        Pre / post population sizes.
    scale:
        Symmetric uniform init range.
    tdt:
        Resolved torch dtype.
    dev:
        Resolved torch device.
    torch_mod:
        The ``torch`` module.

    Returns
    -------
    tuple[Tensor, Tensor]:
        ``(raw_w, bias)`` — weight matrix and bias vector.
    """
    if n_post == 1:
        raw_w = torch_mod.empty(n_pre, dtype=tdt).uniform_(-scale, scale).to(dev)
    else:
        raw_w = (
            torch_mod.empty(n_pre, n_post, dtype=tdt)
            .uniform_(-scale, scale)
            .to(dev)
        )
    raw_w.requires_grad_(True)

    bias = torch_mod.zeros(n_post, dtype=tdt, device=dev)
    bias.requires_grad_(True)

    return raw_w, bias


def build_projection(
    name: str,
    src_name: str,
    tgt_name: str,
    n_pre: int,
    n_post: int,
    sign_pre: Any,
    *,
    synapse_key: str,
    p_connect: float,
    init_scale: float,
    device: str,
    dtype: str,
    hub: Any,
    dev: Any,
    tdt: Any,
    rng: Any,
    torch_mod: Any,
) -> tuple[Projection, dict[str, Any]]:
    """Create a fully initialised projection with trainable parameters.

    Allocates weights via :func:`init_projection_weights`, builds
    topology, creates the synapse model from the hub, and returns
    the :class:`Projection` plus its trainable tensors.

    Parameters
    ----------
    name:
        Projection identifier (e.g. ``"in_to_hidden"``).
    src_name, tgt_name:
        Source / target population names.
    n_pre, n_post:
        Source / target population sizes.
    sign_pre:
        Dale's-Law sign mask for pre-synaptic neurons.
    synapse_key:
        Registry key for the synapse model.
    p_connect:
        Edge-keep probability (``1.0`` = dense).
    init_scale:
        Symmetric uniform init range.
    device, dtype:
        String-form device / dtype for ``init_state``.
    hub:
        :class:`FactoryHub` used to create the synapse model.
    dev, tdt:
        Resolved torch device / dtype for tensor ops.
    rng:
        ``torch.Generator`` for reproducible topology sampling.
    torch_mod:
        The ``torch`` module.

    Returns
    -------
    tuple[Projection, dict[str, Any]]:
        ``(projection, trainables)`` where *trainables* maps
        ``raw_w_{name}`` and ``bias_{name}`` to their tensors.
    """
    raw_w, bias = init_projection_weights(
        n_pre, n_post, init_scale, tdt, dev, torch_mod,
    )

    trainables: dict[str, Any] = {
        f"raw_w_{name}": raw_w,
        f"bias_{name}": bias,
    }

    syn_model: Any = hub.synapses.create(synapse_key, sign_pre=sign_pre)

    if p_connect < 1.0:
        topo, _keep = _sparse_edge_topology(
            raw_w, n_pre, n_post, p_connect, dev, torch_mod, rng,
        )
    else:
        topo = _dense_edge_topology(raw_w, n_pre, n_post, dev, torch_mod)

    proj = Projection(
        name=name, model=syn_model,
        source=src_name, target=tgt_name, topology=topo,
    )
    proj.state = syn_model.init_state(topo, device, dtype)
    proj.state["weight_matrix"] = raw_w
    proj.state["bias"] = bias

    return proj, trainables


# ── population helper ───────────────────────────────────────────────


def _add_population(
    engine: CoreEngine,
    name: str,
    hub: Any,
    neuron_key: str,
    n: int,
    device: str,
    dtype: str,
) -> None:
    """Create and register a single neuron population on *engine*."""
    model: Any = hub.neurons.create(neuron_key)
    pop = Population(name, model, n)
    pop.state = model.init_state(n, device, dtype)
    engine.add_population(pop)


# ── gate-network builder (orchestrator) ─────────────────────────────


def build_gate_network(
    spec: GateNetworkSpec,
    hub: Any | None = None,
) -> GateNetwork:
    """Build a trainable gate network from a :class:`GateNetworkSpec`.

    This is a thin orchestrator that delegates to
    :func:`build_dale_signs`, :func:`init_projection_weights`, and
    :func:`build_projection` for each separable concern, then wires
    everything into a :class:`CoreEngine`.

    Parameters
    ----------
    spec:
        High-level description of the gate network topology.
    hub:
        Optional :class:`FactoryHub`.  Defaults to
        :data:`~neuroforge.factories.hub.DEFAULT_HUB`.

    Returns
    -------
    GateNetwork:
        Contains the ready-to-run :class:`CoreEngine`, a dict of
        trainable weight/bias tensors, and metadata.
    """
    from neuroforge.core.torch_utils import require_torch, resolve_device_dtype
    from neuroforge.factories.hub import DEFAULT_HUB

    if hub is None:
        hub = DEFAULT_HUB

    torch = require_torch()

    rng = torch.Generator()
    rng.manual_seed(spec.seed)
    _random.seed(spec.seed)
    torch.manual_seed(spec.seed)

    dev, tdt = resolve_device_dtype(spec.device, spec.dtype)

    config = SimulationConfig(
        device=spec.device, dtype=spec.dtype, seed=spec.seed,
    )
    engine = CoreEngine(config)

    # ── populations ─────────────────────────────────────────────
    _add_population(engine, "input", hub, spec.neuron_model,
                    spec.input_size, spec.device, spec.dtype)
    if spec.hidden_size > 0:
        _add_population(engine, "hidden", hub, spec.neuron_model,
                        spec.hidden_size, spec.device, spec.dtype)
    _add_population(engine, "output", hub, spec.neuron_model,
                    spec.output_size, spec.device, spec.dtype)

    # ── Dale sign masks ─────────────────────────────────────────
    signs = build_dale_signs(
        spec.input_size, spec.hidden_size, spec.n_inhibitory_hidden,
        dev, tdt, torch,
    )

    meta: dict[str, Any] = {
        "input_size": spec.input_size,
        "hidden_size": spec.hidden_size,
        "output_size": spec.output_size,
        **signs,
    }

    # ── projections ─────────────────────────────────────────────
    proj_kw: dict[str, Any] = {
        "synapse_key": spec.synapse_model,
        "p_connect": spec.p_connect,
        "init_scale": spec.init_scale,
        "device": spec.device,
        "dtype": spec.dtype,
        "hub": hub,
        "dev": dev,
        "tdt": tdt,
        "rng": rng,
        "torch_mod": torch,
    }

    trainables: dict[str, Any] = {}

    if spec.hidden_size > 0:
        proj1, t1 = build_projection(
            "in_to_hidden", "input", "hidden",
            spec.input_size, spec.hidden_size, signs["sign_input"],
            **proj_kw,
        )
        engine.add_projection(proj1)
        trainables.update(t1)

        proj2, t2 = build_projection(
            "hidden_to_out", "hidden", "output",
            spec.hidden_size, spec.output_size, signs["sign_hidden"],
            **proj_kw,
        )
        engine.add_projection(proj2)
        trainables.update(t2)
    else:
        proj0, t0 = build_projection(
            "in_to_out", "input", "output",
            spec.input_size, spec.output_size, signs["sign_input"],
            **proj_kw,
        )
        engine.add_projection(proj0)
        trainables.update(t0)

    # Mark engine as built.
    engine._clock.reset()  # pyright: ignore[reportPrivateUsage]
    engine._built = True  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]

    return GateNetwork(engine=engine, trainables=trainables, meta=meta)
