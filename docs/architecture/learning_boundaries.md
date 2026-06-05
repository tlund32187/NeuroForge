# Learning Boundaries

Training-facing utilities live under `neuroforge.learning`:

- `learning.encoders` owns input encoders, currently `rate`.
- `learning.readouts` owns output decoding and readout heads, currently
  `rate_decoder` and `spike_count`.
- `learning.losses` owns supervised loss functions, currently `mse` and `bce`.
- `learning.stats` owns gradient and tensor summary helpers.

Biological plasticity rules stay under `neuroforge.biology.plasticity`; they are
part of biological modeling, not supervised training orchestration.

The old top-level `neuroforge.encoding`, `neuroforge.readout`, and
`neuroforge.losses` packages are intentionally removed. Import from
`neuroforge.learning.*`.
