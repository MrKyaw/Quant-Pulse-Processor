# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — 2026-10-20

Initial public release: a dependency-light quantum computing toolkit built on
NumPy.

### Added

- **Statevector engine**
  - 14 named gates plus `RX`, `RY`, `RZ`, phase and controlled-phase.
  - Arbitrary `k`-qubit unitaries and full-register operators.
  - Circuit inversion, depth, gate counts and ASCII drawing.
  - Seeded shot sampling, reduced density matrices, von Neumann entropy and
    Pauli expectations.

- **Algorithms**
  - Quantum Fourier transform and inverse, verified against the dense DFT.
  - Grover search with a phase oracle, diffusion operator and rounded optimal
    iteration count.
  - Deutsch–Jozsa constant-versus-balanced decision with a permutation oracle.
  - QAOA for MaxCut with CNOT–RZ–CNOT cost layers and an `RX(2β)` mixer.
  - VQE for the transverse-field Ising ring with a hardware-efficient ansatz and
    exact parameter-shift gradients.
  - Adam optimiser implemented locally (no SciPy dependency).

- **Pulse-level control**
  - Gaussian, square and DRAG envelopes with numerical pulse-area
    normalisation.
  - Driven two-level system with analytic step propagators.
  - Lindblad master equation with amplitude damping (`T1`) and pure dephasing
    (`T2`).
  - Three-level transmon with `√2`-enhanced `1–2` coupling and anharmonicity.
  - DRAG leakage suppression, benchmarked across `β`.

- **Quality and interfaces**
  - 15-case acceptance suite (`quantpulse eval`) built on analytic and exact
    references.
  - Benchmark harness (`quantpulse bench`) with reproducible Markdown output.
  - CLI (`demo`, `eval`, `bench`, `info`) with JSON output for automation.
  - 250 tests, 96% statement coverage, Ruff lint/format, and CI on Python 3.10,
    3.11 and 3.12.

[0.1.0]: https://github.com/MrKyaw/Quant-Pulse-Processor/releases/tag/v0.1.0
