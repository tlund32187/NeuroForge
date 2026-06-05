# Evolution Package Boundaries

Neuroevolution lives under `neuroforge.applications.evolution`:

- `neuroforge.applications.evolution.search` owns population search (`EvolutionEngine`,
  progress callbacks, state, speciation, and reproduction defaults).
- `neuroforge.applications.evolution.genomes` owns genome representations and structural
  mechanics (`PolicyGenome`, `GraphGenome`, `HyperNEATGenome`, `CPPN`,
  `Substrate`, `InnovationRegistry`, and `neat_ops`).
- `neuroforge.applications.evolution.fitness` owns evaluators and reusable objectives.
- `neuroforge.applications.evolution.io` owns checkpoint reading, JSON helpers,
  and the genome type-tag codec.

The old top-level `neuroforge.evolution` package and its flat module paths are
intentionally removed. Import from `neuroforge.applications.evolution.*`.
