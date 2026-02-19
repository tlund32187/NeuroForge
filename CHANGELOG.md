# Changelog

## [Unreleased]

### Phase 6 - Stability and instrumentation

- Determinism controls wired into CLI startup:
  - `--seed`
  - `--deterministic` / `--no-deterministic`
  - `--benchmark`
  - `--warn-only` / `--strict-determinism`
- Added `TrialStatsMonitor` (opt-in) to enrich trial/scalar events with:
  - spike rates (`rate_in_*`, `rate_hid_*`, `rate_out_hz`)
  - sparsity (`sparsity_in`, `sparsity_hid`)
  - convergence counters (`conv_streak`, `conv_per_pattern_*`)
- Added `StabilityMonitor` (opt-in) with flags:
  - `stab_nan_inf`
  - `stab_weight_explode`
  - `stab_rate_saturation`
  - `stab_oscillation`
  - `stab_stagnation`
- Added optional fail-fast for critical instability via `--fail-fast`.
- Added Phase 6 scalar schema/header support in artifact writing so new fields
  are persisted safely while remaining backward-compatible with older runs.
- Added multi-seed stability harness and CLI command:
  - `neuroforge stability --task multi_gate --seeds ...`
  - JSON summary includes pass rate, convergence stats, and stability flag counts.
- Dashboard live and replay paths now surface Phase 6 stability signals so users
  can interpret flags and trends from both websocket events and `scalars.csv`.

## [0.1.0] — 2026-02-16

### Phase 8 — Logic gate task (milestone)

- `tasks/logic_gates.py` — `LogicGateTask` with `LogicGateConfig`, `LogicGateResult`
- Trains SNN on all 6 logic gates (AND, OR, NAND, NOR, XOR, XNOR)
- Simple gates use 2→1 architecture; XOR/XNOR use 2→N_hidden→1
- Spike-count error perceptron learning with trainable bias terms
- Hidden-layer error propagation via `sign(w_ho)` for XOR/XNOR
- 7 truth-table tests + 6 training-convergence tests
- All 6 gates converge within budget (5 000 trials simple, 20 000 XOR/XNOR)

### Phase 7 — R-STDP learning rule

- `learning/rstdp.py` — `RSTDPRule` with `RSTDPParams`
- Eligibility trace: `e_new = e * decay + pre * a_plus − post * a_minus`
- Weight update: `dw = lr * reward * e_new`
- Math helpers: `eligibility_decay`, `predict_eligibility_update`, `predict_dw`,
  `predict_clamped_weight`
- `learning/registry.py` — learning-rule registry
- 16 math-predictive tests

### Phase 6 — Rate encoding / decoding

- `encoding/rate.py` — `RateEncoder` with `RateEncoderParams(amplitude)`
- `encoding/decode.py` — `RateDecoder` with `RateDecoderParams(window_steps, threshold)`
- Encoding: `drive = value × amplitude`; decoding: binary threshold on spike count
- 10 math-predictive tests

### Phase 5 — Simulation engine

- `engine/core_engine.py` — `CoreEngine` with `Population` / `Projection` dataclasses
- `add_population()`, `add_projection()`, `build()`, `step()`, `run()` API
- 13 tests

### Phase 4 — Static synapse model

- `synapses/static.py` — `StaticSynapseModel`
- Uses `scatter_add` for post-synaptic current accumulation
- 12 math-predictive tests

### Phase 3 — LIF neuron model

- `neurons/lif/model.py` — `LIFModel` with `LIFParams`
- LIF equation: `v = v_rest + (v − v_rest) × exp(−dt/τ_mem) + drive × dt/τ_mem`
- Spike at `v ≥ v_thresh`, reset to `v_reset`
- 20 math-predictive tests

### Phase 2 — Core utilities

- `core/registry.py` — `Registry[T]` with register / get / list
- `core/determinism/seeding.py` — `seed_all()` for reproducibility
- `core/time/clock.py` — `SimClock` (frozen dataclass with `tick()` → new clock)
- `core/torch_utils.py` — `require_torch()` lazy-import helper

### Phase 1 — Contracts (7 protocol interfaces)

- `contracts/tensor.py` — `ITensor`
- `contracts/types.py` — `Compartment` enum, `SpikeEvent`, `NeuronID`, etc.
- `contracts/factories.py` — `ITensorFactory`
- `contracts/neurons.py` — `INeuronModel`, `NeuronState`, `NeuronResult`, `NeuronInputs`,
  `StepContext`
- `contracts/synapses.py` — `ISynapseModel`, `SynapseState`, `SynapseResult`
- `contracts/learning.py` — `ILearningRule`, `LearningResult`, `LearningBatch`
- `contracts/simulation.py` — `ISimulationEngine`

### Phase 0 — Project skeleton

- Project structure, build config, test infrastructure
- `pyproject.toml` (Hatchling), `pytest.ini`, `pyrightconfig.json`
- `src/neuroforge` package layout with `py.typed` marker

