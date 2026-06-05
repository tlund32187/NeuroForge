# Biological Modeling Boundary

Phase 2 starts moving biological concepts out of technical buckets and into
`neuroforge.biology`.

## Current Canonical Paths

- Compartments: `neuroforge.contracts.biology.compartments`
- Neuron contracts: `neuroforge.contracts.biology.neurons`
- Synapse contracts: `neuroforge.contracts.biology.synapses`
- Neuron implementations: `neuroforge.biology.neurons`
- Synapse implementations: `neuroforge.biology.synapses`
- Plasticity rules: `neuroforge.biology.plasticity.rules`

The current migrated implementations are:

- `LIFModel`, `LIFParams`, and `SurrogateLIFModel`
- `StaticSynapseModel`, `DelayedStaticSynapseModel`, and
  `DalesStaticSynapseModel`
- `RSTDPRule`, `RSTDPParams`, and the online R-STDP trainer

## Legacy Paths

Legacy wrappers are not retained. Do not add forwarding modules for the old
technical neuron, synapse, or R-STDP locations.

## Hot Path Rule

Biological package structure is for clarity and construction-time composition.
Simulation stepping remains tensorized inside the model implementations. Avoid
adding per-neuron or per-edge Python object loops in `step()` methods.

## Adding a Biological Component

Put new implementations near their domain:

- Neuron model: `neuroforge.biology.neurons.models.<family>`
- Synapse model: `neuroforge.biology.synapses.models`
- Plasticity rule: `neuroforge.biology.plasticity.rules`

Register built-ins through `neuroforge.construction.hub.FactoryHub`.
