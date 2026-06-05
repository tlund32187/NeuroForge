# Agent and Game Architecture

This document defines the target game-playing architecture for NeuroForge. It
separates the world, the player, the decision model, reusable evolution
infrastructure, and user-facing workflows so new games and brains can be added
without turning SMB3 into a catch-all package.

## Core Model

| Concept | Meaning | Canonical home |
|---|---|---|
| Environment | The world, game, or emulator-backed state transition surface. | `neuroforge.environments` |
| BizHawk client | A reusable emulator bridge for screenshots, input, Lua protocol, launch, and health. | `neuroforge.environments.games.clients.bizhawk` |
| SMB3 environment | SMB3-specific observations, actions, rewards, HUD parsing, scoring, curriculum, and termination. | `neuroforge.environments.games.smb3` |
| Agent | The entity that senses, decides, remembers, and acts. | `neuroforge.agents` |
| Brain | The model inside the agent that turns observations into decision signals. | `neuroforge.agents.brains` |
| Sensor | Converts environment output into agent observations. | `neuroforge.agents.sensors` |
| Actuator | Converts decisions into valid environment actions. | `neuroforge.agents.actuators` |
| Policy | Selects actions from decisions, state, randomness, or scripts. | `neuroforge.agents.policies` |
| Neuroevolution | Reusable population search, genomes, CPPNs, NEAT, HyperNEAT, and operators. | `neuroforge.neuroevolution` |
| Application | A complete workflow such as train/evolve/evaluate SMB3. | `neuroforge.applications.smb3` |
| Interface | CLI or dashboard entrypoint that delegates to applications. | `neuroforge.interfaces` |

The intended game-playing flow is:

```text
BizHawk client
    -> SMB3 environment
    -> observation builder / HUD extractor / visual sensor
    -> agent
    -> brain / policy
    -> action decoder / button mapper
    -> BizHawk input bridge
```

## Package Boundaries

All importable Python code lives under `src/neuroforge/`. Configs, examples,
scripts, docs, tests, and generated artifacts live outside `src`.

```text
src/neuroforge/
  api/
  contracts/
  kernel/
  messaging/
  biology/
  simulation/
  construction/
  learning/
  neuroevolution/
  agents/
  perception/
  environments/
  applications/
  observability/
  interfaces/
```

Repository-level locations:

```text
configs/          YAML/JSON configs
examples/         runnable examples and sample projects
scripts/          helper scripts, launchers, Lua scripts, one-off tools
scripts/bizhawk/  BizHawk Lua bridge scripts
docs/             documentation
tests/            tests
artifacts/        generated runs, logs, checkpoints, plots
```

## Dependency Rules

- `agents` may depend on contracts, perception interfaces, environment
  interfaces, simulation/biology abstractions, and neuroevolution adapters.
- `agents` must not depend on CLI, dashboard, or concrete BizHawk internals.
- `brains` expose a stable decision API and do not know how BizHawk works.
- `neuroevolution` must not depend on SMB3, BizHawk, CLI, dashboard, or
  observability.
- `environments.games.clients.bizhawk` must not depend on SMB3, agents, or
  neuroevolution.
- `environments.games.smb3` may depend on BizHawk only through an explicit
  adapter.
- `perception.vision` is generic and must not depend on SMB3 or BizHawk.
- `applications.smb3` wires the pieces together for complete SMB3 workflows.
- `interfaces` call applications; they do not own business logic.
- Optional dependencies are imported lazily behind concrete factories or
  runtime methods.
- No forwarding import wrappers are used. Old paths are updated to canonical
  paths and removed in the same phase.

## Migration Phases

1. Document these boundaries and add cheap import/boundary tests.
2. Add side-effect-free package skeletons for agents, neuroevolution, BizHawk,
   and SMB3 applications.
3. Move reusable BizHawk bridge code into
   `environments/games/clients/bizhawk`.
4. Normalize SMB3 world logic under `environments/games/smb3`.
5. Extract generic agent, brain, policy, sensor, and actuator abstractions.
6. Promote reusable NEAT/HyperNEAT/CPPN infrastructure to `neuroevolution`.
7. Move SMB3 train/evolve/evaluate orchestration to `applications/smb3` and
   make CLI commands delegate there.
8. Move configs, examples, and Lua scripts outside `src`.

## Extension Points

- Add a new game environment under `environments/games/<game>/`.
- Add a new emulator bridge under `environments/games/clients/<client>/`.
- Add a new brain under `agents/brains/`.
- Add a new policy under `agents/policies/`.
- Add a new NEAT genome or operator under `neuroevolution/`.
- Add a new HyperNEAT substrate or decoder under `neuroevolution/hyperneat/`.
- Add a new SMB3 reward function under `environments/games/smb3/rewards.py`.
- Add a new visual sensor under `agents/sensors/`.
- Add a new action decoder under `agents/actuators/`.
