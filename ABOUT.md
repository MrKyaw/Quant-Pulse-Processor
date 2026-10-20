# About Quant-Pulse-Processor

**Quant-Pulse-Processor** is a compact quantum computing toolkit written in pure
Python on top of NumPy. It combines a dense statevector engine, five canonical
quantum algorithms and a pulse-level transmon simulator in one auditable
codebase.

The project answers two questions that are usually studied separately:

1. **What does a quantum algorithm compute?** QFT, Grover, Deutsch–Jozsa, QAOA
   and VQE are implemented gate by gate and scored against exact mathematical
   references — the DFT matrix, amplitude-amplification probabilities, and
   brute-force MaxCut or dense diagonalisation.
2. **How is a superconducting qubit actually driven?** The pulse layer models a
   transmon with Gaussian and DRAG envelopes, Rabi oscillations, `T1`/`T2`
   relaxation through a Lindblad master equation, and leakage to the second
   excited state.

No vendor SDK sits between the reader and the physics. Every unitary, gradient
and master-equation step is visible in about 1,300 statements of library code.

## Who it is for

- **Recruiters and hiring managers** evaluating quantum-software, research
  engineering or scientific-computing work: the repository shows algorithm
  depth, numerical care, testing discipline and documentation in one place.
- **Academics and students** who want a small reference implementation whose
  conventions are stated explicitly and whose results can be checked against
  analytic formulas.
- **GitHub users** looking for a runnable project: clone it, run
  `quantpulse demo`, and see every layer produce a number in under a minute.

## What it is not

- It is not a hardware control stack: there is no device backend, calibration
  pipeline or compiler.
- It is not a large-scale simulator: dense statevectors scale as `2^n` and the
  practical ceiling is roughly 15 qubits on a laptop.
- It is not a replacement for Qiskit, Cirq or PennyLane. It is the readable
  reference you reach for when you want to see how those systems work.

## Topics / tags

`quantum-computing` · `quantum-algorithms` · `statevector-simulator` ·
`quantum-circuit` · `qft` · `grover-algorithm` · `deutsch-jozsa` · `qaoa` ·
`vqe` · `pulse-control` · `drag-pulse` · `transmon` · `lindblad` ·
`quantum-simulation` · `quantum-software` · `quantum-education` · `numpy` ·
`python` · `scientific-computing` · `physics-simulation`

## At a glance

| | |
| --- | --- |
| Language | Python 3.10 – 3.12 |
| Runtime dependency | NumPy only |
| Library size | ~1,300 statements |
| Tests | 250 passing, 96% statement coverage |
| Quality gates | `ruff check`, `ruff format --check`, pytest, CI on three Python versions |
| Algorithms | QFT, Grover, Deutsch–Jozsa, QAOA, VQE |
| Physics | Rabi dynamics, Lindblad `T1`/`T2`, three-level leakage, DRAG |
| Interfaces | Python API + `quantpulse demo / eval / bench / info` |

## Links

- [README](README.md) — quickstart, architecture and measured results
- [Architecture](docs/ARCHITECTURE.md) — layers, conventions and design decisions
- [Method](docs/METHOD.md) — the math behind every algorithm and simulation
- [Benchmarks](docs/BENCHMARKS.md) — reproducible tables generated on this machine
- [Contributing](CONTRIBUTING.md) — development setup and review expectations
- [Issues](https://github.com/MrKyaw/Quant-Pulse-Processor/issues) — bug reports,
  feature requests and questions

## License

MIT © 2026 Kyaw Kyaw Woo.
