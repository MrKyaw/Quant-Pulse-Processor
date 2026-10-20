# Method

This document states the conventions, equations and numerical choices behind
each module. It is the reference to read if you want to check that the code is
solving the problem it claims to solve.

All frequencies are angular frequencies and `ħ = 1`.

---

## 1. Statevectors and gate application

A register of `n` qubits is a complex vector of length `2^n`:

```
|ψ> = Σ_x c_x |x>,    x = Σ_q b_q 2^q,   b_q ∈ {0, 1}
```

Qubit `q` is bit `q` of the index. A `k`-qubit gate with matrix `U` acting on
wires `(q_0, …, q_{k-1})` is applied by

1. reshaping the state to a rank-`n` tensor with qubit `q` on axis `n-1-q`,
2. transposing the selected axes to the front in the listed order,
3. multiplying the `2^k × 2^k` matrix into the flattened rows,
4. inverting the transpose and flattening back.

For a full-register operator the same steps reduce to `U @ ψ`; the package
exposes that case separately as `add_global_unitary` so the distinction is
explicit.

Measurement probabilities are `|c_x|²`, renormalised against floating-point
drift. Sampling uses `numpy.random.default_rng(seed).choice` over the
probability vector.

The reduced state on a set of wires is obtained by moving those axes to the
front, reshaping to `(2^k, 2^{n-k})`, and computing `M M†`.

---

## 2. Quantum Fourier transform

The reference matrix is

```
U[x, y] = exp(2πi x y / N) / √N,    N = 2^n.
```

The circuit is the textbook construction with Hadamards and controlled phases

```
for q = n-1 … 0:
    H(q)
    for c = q-1 … 0:
        CP(π / 2^(q-c)) on (control c, target q)
swap(i, n-1-i) for i < n/2
```

laid out for the little-endian wire convention. Consequently the circuit
unitary equals the DFT matrix exactly, not up to a bit reversal. The inverse
circuit is the adjoint of the forward circuit. The acceptance suite checks:

- `max |U_circuit - U_DFT| < 1e-12` for 2–4 qubits (observed `1.4e-15`),
- `QFT† QFT |ψ> = |ψ>` for a random 5-qubit state (observed `3.9e-16`),
- `QFT |0…0>` is the uniform superposition (observed `8.3e-17`).

---

## 3. Grover search

With `N = 2^n` items and `M` marked states, define

```
θ = arcsin(√(M/N)),    r* = round(π / (4θ) - 1/2).
```

After `r` iterations the marked amplitude is `sin((2r+1)θ)`, so the success
probability is `sin²((2r+1)θ)`. The implementation clamps `r*` to at least 1;
passing `iterations=0` is allowed for experiments and returns the uniform
distribution.

The circuit is `H^⊗n`, then `r` repetitions of:

- a **phase oracle** `I - 2 Σ_{m ∈ marked} |m><m|`,
- the **diffusion** operator `2|s><s| - I`, with `|s>` the uniform state.

Both are applied as global unitaries because they are defined over the full
register. The oracle and diffusion are mathematically exact; no ancilla
tricks are needed at this scale.

Checked invariants: optimal iteration counts for `n = 3…7`, the analytic
success probability, the two-marked-item case, and a zero-iteration control.

---

## 4. Deutsch–Jozsa

The oracle is a permutation on `n+1` qubits implementing

```
|x>|y> → |x>|y ⊕ f(x)>,
```

with the ancilla on qubit 0 (so the basis index is `(x << 1) | y`). The
circuit prepares `|0…0>|1>`, applies `H^{⊗(n+1)}`, the oracle, then `H^{⊗n}` on
the input register. The input register is in `|0…0>` with probability 1 for a
constant oracle and with probability 0 for a balanced one, so a single query
decides the promise with certainty.

The decision uses the exact probability (`P(input = 0) > 0.5`); sampled counts
are returned for inspection. The test suite covers constant 0/1, parity,
half-split and top-bit oracles for `n = 1…4`.

---

## 5. QAOA for MaxCut

For a graph `G = (V, E)` the cost Hamiltonian is

```
H_C = Σ_{(u,v) ∈ E} (1 - Z_u Z_v) / 2,
```

and the mixer is `H_B = Σ_q X_q`. One layer applies

```
U_C(γ) = exp(-i γ H_C),    U_B(β) = exp(-i β H_B).
```

Implementation per edge:

```
CNOT(u, v) · RZ(-γ) · CNOT(u, v) = exp(+i γ Z_u Z_v / 2)
```

which equals `exp(-iγ H_C)` up to a global phase (the phase is irrelevant for
probabilities). The mixer is `RX(2β)` on every wire.

The objective is the expected cut

```
⟨C⟩ = Σ_x p_x · #{ (u,v) : x_u ≠ x_v },
```

computed vectorised over the `2^n` basis states. The reference optimum is a
brute-force search over all bitstrings.

**Optimisation.** Layer angles are shared by all wires, so the single-gate
parameter-shift identity does not apply: shifting a shared angle moves many
gates at once and the objective contains higher harmonics. The implementation
therefore uses a central difference with step `1e-3`, Adam (`lr = 0.15`) and
three random restarts in the benchmark. VQE, whose parameters are independent
gates, uses the exact parameter-shift rule instead.

---

## 6. Variational quantum eigensolver

The reference Hamiltonian is a transverse-field Ising ring

```
H = -J Σ_{<i,j>} Z_i Z_j - h Σ_i X_i,    J = 1, h = 0.5.
```

The ansatz is hardware-efficient:

```
for layer in 1…L:
    RY(θ) on every qubit
    CNOT(q, q+1) ladder
RY(θ) on every qubit
```

Parameter count is `n(L+1)`. The energy is `⟨ψ(θ)|H|ψ(θ)⟩`, evaluated by
building the dense Hamiltonian once (it is at most `2^n × 2^n` for the small
registers used here).

**Gradients.** Each parameter is the angle of an independent `RY` gate, whose
generator has eigenvalues ±1/2, so

```
∂E/∂θ = [E(θ + π/2) - E(θ - π/2)] / 2.
```

Adam then minimises the energy. The exact ground energy comes from
`numpy.linalg.eigvalsh`. For the 2-qubit model the analytic value is `-√2`
(observed error `1.9e-10`); the 3-qubit model needs three layers to reach the
exact energy at this optimisation budget.

---

## 7. Pulse-level control

### Two-level system

In the rotating frame and with the rotating-wave approximation,

```
H(t) = (Ω(t) X + δ Z) / 2.
```

For a piecewise-constant drive the step propagator is analytic. With
`r = √(Ω² + δ²)`,

```
U(Δt) = cos(r Δt / 2) I - i sin(r Δt / 2) (Ω X + δ Z) / r.
```

For a resonant drive (`δ = 0`) starting in `|0>` this gives the Rabi formula

```
P_1(t) = sin²(Ω t / 2).
```

The test suite checks this against the analytic curve to `1e-9`, and checks
that a Gaussian pulse with area `π` inverts the qubit (`P_1 > 0.999999`).

### Lindblad noise

When `T1` and `T2` are supplied, the density matrix evolves by

```
dρ/dt = -i[H, ρ]
        + Γ1 (σ⁻ ρ σ⁺ - ½ {σ⁺σ⁻, ρ})
        + γφ (σz ρ σz - ρ),
```

with `Γ1 = 1/T1` and

```
γφ = ½ (1/T2 - 1/(2T1)).
```

The off-diagonal coherence then decays at `Γ1/2 + 2γφ = 1/T2`, while the
excited population decays at `1/T1`. The physical constraint `T2 ≤ 2 T1` is
enforced. The integrator is classical RK4 with a fixed step; the acceptance
suite compares `P_1(T1)` with `e⁻¹` and `|ρ₀₁(T2)|` with `e⁻¹/2` (observed
errors `1.1e-16` and `5.6e-17`).

### Three-level transmon and DRAG

The transmon is truncated to `{|0>, |1>, |2>}` with anharmonicity `α`:

```
H(t) = α |2><2| + ½ [Ω_I(t) (X₀₁ + √2 X₁₂) + Ω_Q(t) (Y₀₁ + √2 Y₁₂)].
```

The `√2` enhancement of the `1–2` coupling is what makes a finite-bandwidth
pulse leak into `|2>`. The first-order DRAG correction adds a quadrature

```
Ω_Q(t) = -(β / α) dΩ_I/dt,
```

which cancels the leading leakage path. The `pi_pulse` helper normalises a
Gaussian (or DRAG) envelope to area `π` by numerical integration, so changing
`β` does not change the rotation angle.

A 12 ns Gaussian with `σ = 2 ns` and `α = -1.2 rad/ns` leaks `1.28e-2` of the
population without DRAG. At `β = 0.5` the leakage is `2.6e-3` while
`P(|1>) = 0.996`; at `β = 1.0` the leakage is `6.1e-5` but the AC Stark shift
costs population (`P(|1>) = 0.947`). The benchmark reports the full sweep so
the trade-off is visible rather than hidden behind a single "best" number.

For three levels the step propagator is computed by eigendecomposition of the
instantaneous Hermitian Hamiltonian, `U = V e^{-iΛΔt} V†`.

---

## 8. Numerical tolerances

| Quantity | Method | Observed error |
| --- | --- | --- |
| QFT vs DFT | dense matrix comparison | `1.4e-15` |
| QFT round trip | random 5-qubit state | `3.9e-16` |
| Bell / GHZ probabilities | statevector | `1.1e-16` |
| Rabi oscillation | analytic `sin²` | `< 1e-9` |
| `T1` decay | analytic `e^{-t/T1}` | `1.1e-16` |
| `T2` coherence | analytic `e^{-t/T2}/2` | `5.6e-17` |
| VQE (2 qubits) | dense diagonalisation | `1.9e-10` |
| QAOA | brute-force max cut | reported ratio |

Tolerances in the tests are set one to two orders of magnitude above the
observed errors so that ordinary BLAS variation across platforms does not
produce false failures, while genuine convention or ordering bugs still fail.

---

## 9. Acceptance thresholds

| Case | Threshold |
| --- | --- |
| Bell / GHZ | max probability error `< 1e-12` |
| QFT | circuit vs DFT `< 1e-12`; round trip `< 1e-12` |
| Grover (4 and 5 qubits) | success probability `≥ 0.94` |
| Deutsch–Jozsa | exact decision, `P(0) > 0.5` vs `< 1e-12` |
| QAOA, depth 2 | approximation ratio `≥ 0.80` |
| VQE, 2 qubits | energy error `≤ 1e-2` |
| π-pulse | `P(|1>) ≥ 0.999` |
| `T1` | `|P(T1) - e⁻¹| ≤ 2e-3` |
| `T2` | `||ρ₀₁(T2)| - e⁻¹/2| ≤ 2e-3` |
| DRAG | leakage ratio `≤ 0.1` |

---

## References

1. Nielsen & Chuang, *Quantum Computation and Quantum Information*, 10th
   anniversary edition — QFT circuit, Grover, Deutsch–Jozsa.
2. Farhi, Goldstone & Gutmann, *A Quantum Approximate Optimization Algorithm*
   (2014).
3. Peruzzo et al., *A variational eigenvalue solver on a photonic quantum
   processor* (2014).
4. Motzoi, Gambetta, Rebentrost & Wilhelm, *Simple pulses for elimination of
   leakage in weakly nonlinear qubits* (2009) — DRAG.
5. Breuer & Petruccione, *The Theory of Open Quantum Systems* — Lindblad form.
