# Architecture

This document describes how the package is put together, the conventions that
hold across modules, and the trade-offs behind the current design.

## Goals

1. **Auditability.** A reader should be able to follow a basis state from
   preparation through gates to measurement without leaving the package.
2. **One dependency.** NumPy only. Optimisation, matrix exponentials for small
   systems, and sampling are implemented locally.
3. **Exactness where it is possible.** Algorithm tests compare against closed
   forms or dense reference calculations, not against previous outputs.
4. **Two levels of abstraction.** Gate-level circuits and pulse-level control
   share the same measurement and metric vocabulary.

## Layering

```
                    ┌──────────────────────────────┐
                    │  cli.py  demo | eval | bench │
                    └──────────────┬───────────────┘
                                   │
              ┌────────────────────┴────────────────────┐
              │                                         │
      ┌───────▼────────┐                       ┌────────▼────────┐
      │ acceptance.py  │                       │    bench.py     │
      │ 15 invariants  │                       │ Markdown report │
      └───────┬────────┘                       └────────┬────────┘
              │                                         │
    ┌─────────┴──────────┬───────────────┬──────────────┴──────┐
    │                    │               │                     │
┌───▼─────────┐  ┌───────▼──────┐  ┌─────▼──────┐   ┌──────────▼─────┐
│ algorithms/ │  │  circuit.py  │  │  pulse.py  │   │   metrics.py   │
│ QFT Grover  │  │  operations  │  │ Lindblad   │   │ fidelity, TVD  │
│ DJ QAOA VQE │  │  execution   │  │ DRAG       │   │ cut ratio      │
└───┬─────────┘  └───────┬──────┘  └─────┬──────┘   └────────────────┘
    │                    │               │
    │            ┌───────▼──────┐        │
    │            │ statevector  │        │
    │            │ sampling     │        │
    │            │ partial trace│        │
    │            └───────┬──────┘        │
    │                    │               │
    │            ┌───────▼──────┐        │
    └────────────►   gates.py   ◄────────┘
                 │ dense unitaries │
                 └─────────────────┘
```

## Core conventions

### Little-endian state indices

A basis state is an integer whose bit `q` is the value of qubit `q`:

```
|q2 q1 q0>  ↔  index = q0 + 2 q1 + 4 q2
```

Bitstrings printed for humans are big-endian (`format(index, "0nb")`), matching
the usual `q_{n-1} … q_0` notation.

### Gate matrix ordering

For a gate acting on wires `(q0, q1, …, qk-1)`, its matrix index treats `q0` as
the most significant bit of the gate-local index. This is the standard textbook
ordering and the one used by the named gates:

- `CNOT` is the usual control-first matrix, so `cnot(c, t)` maps
  `|c=1, t=0⟩ → |c=1, t=1⟩`.
- `TOFFOLI` is the usual 8×8 controlled-controlled-X matrix.

Implementation: selected tensor axes are moved to the front in the listed order,
the matrix multiplies the reshaped rows, and the axes are restored
(`Circuit._apply_operator`).

### Global operators

A matrix expressed directly in the state basis is applied without any wire
permutation through `Circuit.add_global_unitary`. Grover's phase oracle and
diffusion operator, and the Deutsch–Jozsa oracle, use this path because they are
defined over the full register.

This distinction is deliberate: `add_unitary` follows the per-gate convention,
while `add_global_unitary` follows the state-vector convention. Mixing them up
silently bit-reverses an operator, which is exactly the class of bug the QFT
matrix test is designed to catch.

## Module responsibilities

| Module | Responsibility | Notable details |
| --- | --- | --- |
| `gates.py` | dense matrices and Pauli products | unitarity checked on insertion |
| `statevector.py` | probabilities, sampling, reduced states | seeded `Generator`, clipped probabilities |
| `circuit.py` | operation list, execution, inspection | inversion, depth, gate counts, drawing |
| `metrics.py` | fidelity, trace distance, TVD, cut values | shared by algorithms and benchmarks |
| `optim.py` | Adam and parameter-shift gradients | no SciPy dependency |
| `algorithms/qft.py` | QFT circuit and dense DFT reference | circuit equals DFT to `1e-15` |
| `algorithms/grover.py` | phase oracle, diffusion, iteration formula | global unitaries over the register |
| `algorithms/deutsch_jozsa.py` | permutation oracle + phase kickback | one query, deterministic decision |
| `algorithms/qaoa.py` | MaxCut cost and mixer layers, Adam loop | brute-force optimum as reference |
| `algorithms/vqe.py` | TFIM Hamiltonian, RY/CNOT ansatz | exact diagonalisation as reference |
| `pulse.py` | envelopes, propagators, Lindblad, transmon | analytic step propagators for 2 levels |
| `acceptance.py` | 15 end-to-end invariants | measurable detail strings |
| `bench.py` | six benchmark sections | Markdown and JSON rendering |
| `cli.py` | four subcommands | JSON contracts for automation |

## Execution model

Circuits are immutable descriptions. Building a circuit appends dense matrices;
`run()` walks the operations and applies them to a fresh statevector (or a
caller-supplied initial state). There is no intermediate IR, no transpiler and
no gate fusion — the operation list is the program.

This costs performance on deep circuits but keeps the mapping from source to
execution exact, which is the point of the project. The largest optimisations
that do exist are local and visible: analytic 2×2 propagators, cached
per-step propagators in the pulse simulator, and vectorised cut-value
evaluation.

## Error handling

- Invalid qubit indices, duplicate wires, non-unitary matrices and shape
  mismatches raise `ValueError` at construction time.
- Numerical routines validate physical constraints, for example `T2 ≤ 2 T1`.
- Acceptance cases report the measured quantity and its reference in prose, so
  a failure says *what* drifted, not just that a boolean flipped.

## Testing strategy

| Layer | How it is tested |
| --- | --- |
| gates | unitarity, truth tables, algebraic identities (`S² = Z`) |
| statevector | normalisation, sampling determinism, entropy of Bell pairs |
| circuit | known states, inversion, wire-order regression |
| algorithms | exact references (DFT, analytic success probability, brute force) |
| physics | analytic Rabi/decay solutions, leakage monotonicity |
| interfaces | CLI JSON schema, exit codes, Markdown output |

The one slow test (`collect_benchmarks`) is marked `slow` and deselected by
default; `pytest -m slow` runs the complete benchmark sweep.

## Extension points

- **New gate:** add a matrix in `gates.py` and a method in `Circuit`; the
  insertion check verifies unitarity.
- **New algorithm:** add a module under `algorithms/`, expose it in
  `algorithms/__init__.py`, and add an acceptance case with an exact reference.
- **New pulse shape:** return an array from an envelope function; normalisation
  to a target area is handled by `pi_pulse`'s area calculation.
- **New benchmark:** append a `Section` builder to `collect_benchmarks`.

## Known limits

- Dense statevectors: `O(2^n)` memory and `O(4^n)` time for full unitaries.
- Sampling uses NumPy's generator; statistical error is reported alongside
  exact probabilities in the acceptance suite.
- The pulse model is a single anharmonic oscillator. Coupled-qubit dynamics,
  crosstalk and two-qubit gates are out of scope for v0.1.0.
