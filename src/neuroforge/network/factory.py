"""NetworkFactory - config-driven network construction.

Builds a :class:`CoreEngine` from declarative :class:`NetworkSpec`
using injected registries. No file I/O and no global singleton access.
"""

from __future__ import annotations

import random as _random
from typing import TYPE_CHECKING, Any, cast

from neuroforge.contracts.simulation import SimulationConfig
from neuroforge.engine.core_engine import CoreEngine, Population, Projection
from neuroforge.network.topology_builders import (
    build_block_sparse_topology,
    build_dense_topology,
    build_sparse_fanin_topology,
    build_sparse_fanout_topology,
    build_sparse_random_topology,
)

if TYPE_CHECKING:
    from neuroforge.contracts.factories import IRegistry
    from neuroforge.network.specs import NetworkSpec

__all__ = ["NetworkFactory", "to_topology_json"]


class NetworkFactory:
    """Build a :class:`CoreEngine` from a :class:`NetworkSpec`."""

    def __init__(
        self,
        neuron_registry: IRegistry[Any],
        synapse_registry: IRegistry[Any],
        learning_registry: IRegistry[Any] | None = None,
    ) -> None:
        self._neuron_reg = neuron_registry
        self._synapse_reg = synapse_registry
        self._learning_reg = learning_registry

    def build(
        self,
        spec: NetworkSpec,
        *,
        device: str,
        dtype: str,
        seed: int | None = None,
    ) -> CoreEngine:
        """Build and return a fully initialised :class:`CoreEngine`."""
        from neuroforge.core.torch_utils import require_torch, resolve_device_dtype

        torch = require_torch()

        if seed is not None:
            _random.seed(seed)
            torch.manual_seed(seed)

        base_seed = int(seed if seed is not None else 42)
        topo_rng = torch.Generator(device="cpu")
        topo_rng.manual_seed(base_seed)

        dev, tdt = resolve_device_dtype(device, dtype)

        config = SimulationConfig(
            device=device,
            dtype=dtype,
            seed=base_seed,
        )
        engine = CoreEngine(config)

        pop_sizes: dict[str, int] = {}
        for ps in spec.populations:
            neuron_kw: dict[str, object] = dict(ps.neuron_params)
            neuron_model = self._neuron_reg.create(ps.neuron_model, **neuron_kw)
            pop = Population(name=ps.name, model=neuron_model, n=ps.n)
            pop.state = neuron_model.init_state(ps.n, device, dtype)
            engine.add_population(pop)
            pop_sizes[ps.name] = ps.n

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
            init_cfg = {
                "init": topo_cfg.get("init", "uniform"),
                "low": topo_cfg.get("low", -0.3),
                "high": topo_cfg.get("high", 0.3),
            }
            delay_cfg_raw: object = topo_cfg.get("delays", {})
            delay_cfg: dict[str, Any]
            if isinstance(delay_cfg_raw, dict):
                typed_delay_cfg = cast("dict[object, Any]", delay_cfg_raw)
                delay_cfg = {str(key): value for key, value in typed_delay_cfg.items()}
            else:
                delay_cfg = {}
            sort_edges = bool(topo_cfg.get("sort", True))
            rng = _projection_rng(topo_cfg.get("seed"), fallback=topo_rng, torch=torch)

            proj = Projection(
                name=prj.name,
                model=synapse_model,
                source=prj.source,
                target=prj.target,
                topology=None,
            )

            if topo_type == "dense":
                weight_matrix, bias = _make_dense_weights(
                    n_pre,
                    n_post,
                    topo_cfg,
                    dev,
                    tdt,
                    torch,
                )
                edge_topo = build_dense_topology(weight_matrix, n_pre, n_post, dev, torch)

                proj.topology = edge_topo
                proj.state = synapse_model.init_state(edge_topo, device, dtype)
                proj.state["weight_matrix"] = weight_matrix
                if bias is not None:
                    proj.state["bias"] = bias
                engine.add_projection(proj)
                continue

            bias = _maybe_make_bias(n_post, topo_cfg, dev, tdt, torch)
            if topo_type == "sparse_random":
                p_connect = float(topo_cfg.get("p_connect", 1.0))
                edge_topo, weights_vec = build_sparse_random_topology(
                    n_pre,
                    n_post,
                    p_connect,
                    init_cfg=init_cfg,
                    delay_cfg=delay_cfg,
                    dev=dev,
                    tdt=tdt,
                    torch=torch,
                    rng=rng,
                    sort=sort_edges,
                )
            elif topo_type == "sparse_fanout":
                fanout = int(topo_cfg.get("fanout", n_post))
                edge_topo, weights_vec = build_sparse_fanout_topology(
                    n_pre,
                    n_post,
                    fanout,
                    init_cfg=init_cfg,
                    delay_cfg=delay_cfg,
                    dev=dev,
                    tdt=tdt,
                    torch=torch,
                    rng=rng,
                    sort=sort_edges,
                )
            elif topo_type == "sparse_fanin":
                fanin = int(topo_cfg.get("fanin", n_pre))
                edge_topo, weights_vec = build_sparse_fanin_topology(
                    n_pre,
                    n_post,
                    fanin,
                    init_cfg=init_cfg,
                    delay_cfg=delay_cfg,
                    dev=dev,
                    tdt=tdt,
                    torch=torch,
                    rng=rng,
                    sort=sort_edges,
                )
            elif topo_type == "block_sparse":
                block_pre = int(topo_cfg.get("block_pre", n_pre))
                block_post = int(topo_cfg.get("block_post", n_post))
                p_block = float(topo_cfg.get("p_block", 1.0))
                edge_topo, weights_vec = build_block_sparse_topology(
                    n_pre,
                    n_post,
                    block_pre,
                    block_post,
                    p_block,
                    init_cfg=init_cfg,
                    delay_cfg=delay_cfg,
                    dev=dev,
                    tdt=tdt,
                    torch=torch,
                    rng=rng,
                    sort=sort_edges,
                )
            else:
                msg = f"Unsupported topology type: {topo_type!r}"
                raise ValueError(msg)

            proj.topology = edge_topo
            proj.state = synapse_model.init_state(edge_topo, device, dtype)
            proj.state["weights"] = weights_vec
            if bias is not None:
                proj.state["bias"] = bias
            engine.add_projection(proj)

        engine._clock.reset()  # pyright: ignore[reportPrivateUsage]
        engine._built = True  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
        return engine


def to_topology_json(engine: CoreEngine) -> dict[str, Any]:
    """Return a JSON-serialisable topology dict from ``engine``.

    For large projections this emits summaries instead of full matrices to
    avoid huge payloads in live/replay artifacts.
    """
    max_full_edges = 4096
    sample_limit = 256

    layers: list[str] = [f"{name}({pop.n})" for name, pop in engine.populations.items()]

    edges: list[dict[str, Any]] = []
    projection_meta: list[dict[str, Any]] = []
    for proj_name, proj in engine.projections.items():
        topo = proj.topology
        flat_weights = topo.weights.detach().reshape(-1)
        n_edges = int(flat_weights.numel())
        dense = "weight_matrix" in proj.state
        dtype_name = str(topo.weights.dtype).replace("torch.", "")

        if n_edges > 0:
            w_mean = float(flat_weights.mean().item())
            w_std = float(flat_weights.std(unbiased=False).item())
            w_min = float(flat_weights.min().item())
            w_max = float(flat_weights.max().item())
        else:
            w_mean = 0.0
            w_std = 0.0
            w_min = 0.0
            w_max = 0.0

        edge: dict[str, Any] = {
            "name": proj_name,
            "src": proj.source,
            "dst": proj.target,
            "n_pre": int(topo.n_pre),
            "n_post": int(topo.n_post),
            "n_edges": n_edges,
            "dense": dense,
            "dtype": dtype_name,
            "topology_type": "dense" if dense else "sparse",
            "weight_stats": {
                "mean": w_mean,
                "std": w_std,
                "min": w_min,
                "max": w_max,
            },
        }

        projection_meta.append(
            {
                "name": proj_name,
                "src": proj.source,
                "dst": proj.target,
                "n_pre": int(topo.n_pre),
                "n_post": int(topo.n_post),
                "n_edges": int(n_edges),
                "dense": dense,
                "dtype": dtype_name,
                "topology_type": "dense" if dense else "sparse",
            }
        )

        if dense and n_edges <= max_full_edges:
            dense_weights = proj.state.get("weight_matrix")
            if dense_weights is not None:
                edge["weights"] = dense_weights.detach().cpu().tolist()
        elif n_edges > 0:
            sample_n = min(sample_limit, n_edges)
            edge["weights_sample"] = flat_weights[:sample_n].cpu().tolist()
            edge["sample_size"] = sample_n
            if "weight_matrix" not in proj.state:
                edge["sparse_sample"] = {
                    "pre_idx": topo.pre_idx.detach().reshape(-1)[:sample_n].cpu().tolist(),
                    "post_idx": topo.post_idx.detach().reshape(-1)[:sample_n].cpu().tolist(),
                    "delays": topo.delays.detach().reshape(-1)[:sample_n].cpu().tolist(),
                }

        edges.append(edge)

    return {"layers": layers, "edges": edges, "projection_meta": projection_meta}


def _projection_rng(
    seed_override: Any,
    *,
    fallback: Any,
    torch: Any,
) -> Any:
    if seed_override is None:
        return fallback
    seed_value = int(seed_override)
    rng = torch.Generator(device="cpu")
    rng.manual_seed(seed_value)
    return rng


def _make_dense_weights(
    n_pre: int,
    n_post: int,
    cfg: dict[str, Any],
    dev: Any,
    tdt: Any,
    torch: Any,
) -> tuple[Any, Any | None]:
    """Create a dense weight matrix and optional bias vector."""
    init = str(cfg.get("init", "uniform"))
    squeeze = n_post == 1

    if init == "uniform":
        low = float(cfg.get("low", -0.3))
        high = float(cfg.get("high", 0.3))
        if squeeze:
            weight_matrix = torch.empty(n_pre, dtype=tdt).uniform_(low, high).to(dev)
        else:
            weight_matrix = torch.empty(n_pre, n_post, dtype=tdt).uniform_(low, high).to(dev)
    elif init == "zeros":
        if squeeze:
            weight_matrix = torch.zeros(n_pre, dtype=tdt, device=dev)
        else:
            weight_matrix = torch.zeros(n_pre, n_post, dtype=tdt, device=dev)
    else:
        msg = f"Unsupported weight init: {init!r}"
        raise ValueError(msg)

    return weight_matrix, _maybe_make_bias(n_post, cfg, dev, tdt, torch)


def _maybe_make_bias(
    n_post: int,
    cfg: dict[str, Any],
    dev: Any,
    tdt: Any,
    torch: Any,
) -> Any | None:
    if not cfg.get("bias", False):
        return None
    return torch.zeros(n_post, dtype=tdt, device=dev)
