# Simulation And Construction Boundary

Phase 3 separates simulation runtime, topology description, and network
construction. Phase 4 makes `neuroforge.construction` the canonical owner of
the registry hub.

## Canonical Paths

- Engine runtime: `neuroforge.simulation.engine`
- Run metadata/runtime context: `neuroforge.simulation.runtime`
- Topology specs and builders: `neuroforge.simulation.topology`
- Network and gate construction: `neuroforge.construction`
- Component registry hub: `neuroforge.construction.hub`

## Dependency Rules

- `simulation.engine` owns tensor stepping and may depend on contracts and
  low-level core utilities.
- `simulation.topology` owns declarative DTOs and topology builder functions.
  It must not import tasks, games, runners, dashboards, or evolution.
- `construction` can compose registries, biology implementations, topology
  builders, and simulation engines into runnable networks.
- `construction.hub` is the only built-in component registry surface. Code
  should use `DEFAULT_HUB.<registry>` or inject a `FactoryHub`.
- Application layers should import these canonical paths directly.

Legacy engine, network, and run-context wrapper modules are not retained.
Legacy factory and per-domain registry wrapper modules are not retained.
