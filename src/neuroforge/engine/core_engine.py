"""Core simulation engine — orchestrates populations and projections.

The engine manages a graph of neuron *populations* (nodes) connected
by synapse *projections* (edges).  Each ``step()`` call:

1. Collects post-synaptic currents from all projections.
2. Feeds currents as drive to the target populations.
3. Steps every population's neuron model.
4. Records which neurons spiked.

The engine is the ``ISimulationEngine`` implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from neuroforge.contracts.neurons import NeuronInputs, StepContext
from neuroforge.contracts.simulation import SimulationConfig, StepResult
from neuroforge.contracts.synapses import SynapseInputs
from neuroforge.core.time.clock import SimulationClock

if TYPE_CHECKING:
    from neuroforge.contracts.types import Compartment

__all__ = ["CoreEngine", "Population", "Projection"]


@dataclass
class Population:
    """A group of neurons with a shared model.

    Attributes
    ----------
    name:
        Unique identifier for this population.
    model:
        Neuron model instance (implements INeuronModel).
    n:
        Number of neurons in the population.
    state:
        Mutable state dict (allocated by ``model.init_state``).
    """

    name: str
    model: Any  # INeuronModel
    n: int
    state: dict[str, Any] = field(default_factory=lambda: {})


@dataclass
class Projection:
    """A synaptic connection between two populations.

    Attributes
    ----------
    name:
        Unique identifier for this projection.
    model:
        Synapse model instance (implements ISynapseModel).
    source:
        Name of the pre-synaptic population.
    target:
        Name of the post-synaptic population.
    topology:
        Wiring specification (pre_idx, post_idx, weights, delays).
    state:
        Mutable state dict (allocated by ``model.init_state``).
    """

    name: str
    model: Any  # ISynapseModel
    source: str
    target: str
    topology: Any  # SynapseTopology
    state: dict[str, Any] = field(default_factory=lambda: {})


class CoreEngine:
    """Core simulation engine.

    Manages a flat graph of populations and projections.
    Call ``build()`` after adding all components to initialise state,
    then call ``step()`` or ``run()`` to advance the simulation.
    """

    def __init__(self, config: SimulationConfig | None = None) -> None:
        self.config = config or SimulationConfig()
        self._clock = SimulationClock(dt=self.config.dt)
        self._populations: dict[str, Population] = {}
        self._projections: dict[str, Projection] = {}
        self._built = False

    # ── builder API ─────────────────────────────────────────────────

    def add_population(self, pop: Population) -> CoreEngine:
        """Add a population. Returns self for chaining."""
        if pop.name in self._populations:
            msg = f"Duplicate population name: {pop.name!r}"
            raise ValueError(msg)
        self._populations[pop.name] = pop
        return self

    def add_projection(self, proj: Projection) -> CoreEngine:
        """Add a projection. Returns self for chaining."""
        if proj.name in self._projections:
            msg = f"Duplicate projection name: {proj.name!r}"
            raise ValueError(msg)
        self._projections[proj.name] = proj
        return self

    @property
    def populations(self) -> dict[str, Population]:
        """Read-only access to populations."""
        return dict(self._populations)

    @property
    def projections(self) -> dict[str, Projection]:
        """Read-only access to projections."""
        return dict(self._projections)

    # ── lifecycle ───────────────────────────────────────────────────

    def build(self) -> CoreEngine:
        """Initialise all state tensors. Must be called before step/run."""
        from neuroforge.core.determinism.seeding import set_global_seed

        set_global_seed(self.config.seed)

        device = self.config.device
        dtype = self.config.dtype

        # Validate projection sources/targets
        for proj in self._projections.values():
            if proj.source not in self._populations:
                msg = f"Projection {proj.name!r}: source {proj.source!r} not found"
                raise ValueError(msg)
            if proj.target not in self._populations:
                msg = f"Projection {proj.name!r}: target {proj.target!r} not found"
                raise ValueError(msg)

        # Init population states
        for pop in self._populations.values():
            pop.state = pop.model.init_state(pop.n, device, dtype)

        # Init projection states
        for proj in self._projections.values():
            proj.state = proj.model.init_state(proj.topology, device, dtype)

        self._clock.reset()
        self._built = True
        return self

    def reset(self) -> None:
        """Reset all states and the clock."""
        for pop in self._populations.values():
            pop.model.reset_state(pop.state)
        self._clock.reset()

    def step(
        self, external_drive: dict[str, dict[Compartment, Any]] | None = None
    ) -> StepResult:
        """Advance the simulation by one time step.

        Parameters
        ----------
        external_drive:
            Optional external drive per population per compartment.
            ``{pop_name: {Compartment.SOMA: tensor}}``.

        Returns
        -------
        StepResult:
            Step index, time, and spike tensors for each population.
        """
        if not self._built:
            msg = "Engine not built. Call build() first."
            raise RuntimeError(msg)

        from neuroforge.core.torch_utils import require_torch

        torch = require_torch()

        dt = self._clock.dt
        step_idx = self._clock.step
        t = self._clock.t
        ctx = StepContext(dt=dt, step=step_idx, t=t)

        # 1) Accumulate drive for each population
        drive: dict[str, dict[Compartment, Any]] = {}

        # Start with external drive (if any)
        if external_drive:
            for pop_name, compartment_drives in external_drive.items():
                drive[pop_name] = dict(compartment_drives)

        # 2) Compute synaptic currents from projections
        # We need pre-spikes from the previous step. On step 0, all are zero.
        for proj in self._projections.values():
            source_pop = self._populations[proj.source]
            target_pop = self._populations[proj.target]

            # Get pre-synaptic spikes (from state, not from "this step" which hasn't happened)
            # On step 0, use zeros — allocated on the correct device.
            # The dtype mirrors last_spikes when available; otherwise
            # defaults to bool (the standard LIF model returns bools).
            if "last_spikes" in source_pop.state:
                pre_spikes = source_pop.state["last_spikes"]
            else:
                _src_dev = source_pop.state["v"].device
                pre_spikes = torch.zeros(source_pop.n, dtype=torch.bool, device=_src_dev)

            if "last_spikes" in target_pop.state:
                post_spikes = target_pop.state["last_spikes"]
            else:
                _tgt_dev = target_pop.state["v"].device
                post_spikes = torch.zeros(target_pop.n, dtype=torch.bool, device=_tgt_dev)

            syn_inputs = SynapseInputs(pre_spikes=pre_spikes, post_spikes=post_spikes)
            syn_result = proj.model.step(proj.state, proj.topology, syn_inputs, ctx)

            # Add synaptic currents to target population's drive
            if proj.target not in drive:
                drive[proj.target] = {}

            for compartment, current in syn_result.post_current.items():
                if compartment in drive[proj.target]:
                    drive[proj.target][compartment] = (
                        drive[proj.target][compartment] + current
                    )
                else:
                    drive[proj.target][compartment] = current

        # 3) Step each population
        spikes: dict[str, Any] = {}
        for pop_name, pop in self._populations.items():
            pop_drive = drive.get(pop_name, {})
            neuron_inputs = NeuronInputs(drive=pop_drive)
            result = pop.model.step(pop.state, neuron_inputs, ctx)
            spikes[pop_name] = result.spikes

            # Save spikes for next step's synapse computation
            pop.state["last_spikes"] = result.spikes

        # 4) Advance clock
        self._clock.advance()

        return StepResult(
            step=step_idx,
            t=t,
            spikes=spikes,
        )

    def run(
        self,
        steps: int,
        external_drive_fn: Any | None = None,
    ) -> list[StepResult]:
        """Run multiple steps.

        Parameters
        ----------
        steps:
            Number of time steps to run.
        external_drive_fn:
            Optional callable ``(step: int) -> dict[str, dict[Compartment, Tensor]]``
            that provides external drive for each step.

        Returns
        -------
        list[StepResult]:
            Results for each step.
        """
        results: list[StepResult] = []
        for i in range(steps):
            ext = external_drive_fn(i) if external_drive_fn else None
            results.append(self.step(external_drive=ext))
        return results
