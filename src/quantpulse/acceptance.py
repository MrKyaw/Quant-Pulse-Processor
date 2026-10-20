"""Physics and algorithm acceptance suite.

Each case checks a measurable property against an analytic or exact reference
rather than a stored golden file, so the suite stays meaningful when the
implementation is refactored.  ``quantpulse eval`` runs the same cases from
the command line.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .algorithms.deutsch_jozsa import deutsch_jozsa
from .algorithms.grover import run_grover
from .algorithms.qaoa import RING6_EDGES, optimize_qaoa
from .algorithms.qft import qft_circuit, qft_unitary
from .algorithms.vqe import optimize_vqe
from .circuit import Circuit
from .pulse import pi_pulse, simulate_three_level, simulate_two_level

__all__ = ["ACCEPTANCE_CASES", "CaseResult", "run_acceptance"]


@dataclass(frozen=True)
class CaseResult:
    """Outcome of one acceptance check."""

    name: str
    category: str
    passed: bool
    measured: float
    reference: str
    detail: str
    duration_ms: float


def _bell() -> tuple[bool, float, str]:
    state = Circuit(2).h(0).cnot(0, 1).run()
    probs = np.abs(state) ** 2
    expected = np.array([0.5, 0.0, 0.0, 0.5])
    error = float(np.max(np.abs(probs - expected)))
    return error < 1e-12, error, "max |P - target| = {:.2e}"


def _ghz() -> tuple[bool, float, str]:
    state = Circuit(3).h(0).cnot(0, 1).cnot(1, 2).run()
    probs = np.abs(state) ** 2
    expected = np.zeros(8)
    expected[0] = 0.5
    expected[7] = 0.5
    error = float(np.max(np.abs(probs - expected)))
    return error < 1e-12, error, "max |P - GHZ| = {:.2e}"


def _qft_uniform() -> tuple[bool, float, str]:
    state = qft_circuit(4).run()
    target = np.full(16, 1 / np.sqrt(16), dtype=complex)
    error = float(np.max(np.abs(state - target)))
    return error < 1e-12, error, "max |amp - 1/sqrt(N)| = {:.2e}"


def _qft_roundtrip() -> tuple[bool, float, str]:
    rng = np.random.default_rng(7)
    vector = rng.normal(size=32) + 1j * rng.normal(size=32)
    vector /= np.linalg.norm(vector)
    n_qubits = 5
    forward = qft_circuit(n_qubits).run(vector)
    back = qft_circuit(n_qubits, inverse=True).run(forward)
    error = float(np.max(np.abs(back - vector)))
    return error < 1e-12, error, "max |QFT^-1 QFT psi - psi| = {:.2e}"


def _qft_matrix() -> tuple[bool, float, str]:
    n_qubits = 3
    circuit = qft_circuit(n_qubits)
    columns = [circuit.run(np.eye(2**n_qubits)[:, i]) for i in range(2**n_qubits)]
    error = float(np.max(np.abs(np.column_stack(columns) - qft_unitary(n_qubits))))
    return error < 1e-12, error, "max |circuit - DFT| = {:.2e}"


def _grover(n_qubits: int) -> tuple[bool, float, str]:
    marked = 2**n_qubits - 2
    result = run_grover(n_qubits, [marked], shots=200, seed=0)
    return (
        result.success_probability >= 0.94,
        result.success_probability,
        "success probability = {:.4f}",
    )


def _dj_constant() -> tuple[bool, float, str]:
    result = deutsch_jozsa(lambda _: 1, 4)
    return result.is_constant, result.zero_probability, "P(input = 0) = {:.4f}"


def _dj_balanced() -> tuple[bool, float, str]:
    result = deutsch_jozsa(lambda x: bin(x).count("1") % 2, 4)
    return not result.is_constant, result.zero_probability, "P(input = 0) = {:.2e}"


def _qaoa() -> tuple[bool, float, str]:
    result = optimize_qaoa(6, RING6_EDGES, depth=2, steps=60, restarts=2, seed=3)
    return result.ratio >= 0.80, result.ratio, "approximation ratio = {:.4f}"


def _vqe() -> tuple[bool, float, str]:
    result = optimize_vqe(2, layers=2, steps=200, seed=0)
    return result.error <= 1e-2, result.error, "|E - E_exact| = {:.2e}"


def _rabi_pi_pulse() -> tuple[bool, float, str]:
    _, in_phase, _ = pi_pulse(24.0, 0.05)
    result = simulate_two_level(in_phase, 0.05)
    return (
        result.final_population >= 0.999,
        result.final_population,
        "P(|1>) after pi-pulse = {:.6f}",
    )


def _t1_relaxation() -> tuple[bool, float, str]:
    t1 = 20.0
    dt = 0.01
    drive = np.zeros(int(t1 / dt))
    result = simulate_two_level(drive, dt, t1=t1, t2=t1, initial="excited")
    target = float(np.exp(-1.0))
    error = abs(result.final_population - target)
    return error <= 2e-3, error, "|P(T1) - 1/e| = {:.2e}"


def _t2_dephasing() -> tuple[bool, float, str]:
    t1, t2 = 40.0, 20.0
    dt = 0.01
    drive = np.zeros(int(t2 / dt))
    result = simulate_two_level(drive, dt, t1=t1, t2=t2, initial="plus")
    target = 0.5 * float(np.exp(-1.0))
    error = abs(result.coherence[-1] - target)
    return error <= 2e-3, error, "||rho01(T2)| - e^-1/2| = {:.2e}"


def _drag_leakage() -> tuple[bool, float, str]:
    _, bare_i, bare_q = pi_pulse(12.0, 0.02, sigma=2.0)
    bare = simulate_three_level(bare_i, bare_q, 0.02, anharmonicity=-1.2)
    _, drag_i, drag_q = pi_pulse(12.0, 0.02, sigma=2.0, beta=0.75)
    corrected = simulate_three_level(drag_i, drag_q, 0.02, anharmonicity=-1.2)
    ratio = corrected.final_leakage / max(bare.final_leakage, 1e-12)
    return ratio <= 0.1, ratio, "leakage(DRAG) / leakage(bare) = {:.4f}"


@dataclass(frozen=True)
class _Case:
    name: str
    category: str
    check: Callable[[], tuple[bool, float, str]]


ACCEPTANCE_CASES: tuple[_Case, ...] = (
    _Case("Bell state correlations", "circuits", _bell),
    _Case("GHZ-3 state", "circuits", _ghz),
    _Case("QFT |0...0> is uniform", "algorithms", _qft_uniform),
    _Case("QFT circuit equals DFT matrix", "algorithms", _qft_matrix),
    _Case("Inverse QFT round trip", "algorithms", _qft_roundtrip),
    _Case("Grover, 4 qubits", "algorithms", lambda: _grover(4)),
    _Case("Grover, 5 qubits", "algorithms", lambda: _grover(5)),
    _Case("Deutsch-Jozsa, constant", "algorithms", _dj_constant),
    _Case("Deutsch-Jozsa, balanced", "algorithms", _dj_balanced),
    _Case("QAOA MaxCut approximation", "algorithms", _qaoa),
    _Case("VQE ground-state energy", "algorithms", _vqe),
    _Case("Rabi pi-pulse population", "pulse", _rabi_pi_pulse),
    _Case("T1 amplitude damping", "pulse", _t1_relaxation),
    _Case("T2 coherence decay", "pulse", _t2_dephasing),
    _Case("DRAG leakage suppression", "pulse", _drag_leakage),
)


def run_acceptance() -> list[CaseResult]:
    """Run every acceptance case and return their measurements."""
    results: list[CaseResult] = []
    for case in ACCEPTANCE_CASES:
        start = time.perf_counter()
        passed, measured, template = case.check()
        elapsed = (time.perf_counter() - start) * 1000.0
        results.append(
            CaseResult(
                name=case.name,
                category=case.category,
                passed=bool(passed),
                measured=float(measured),
                reference="pass",
                detail=template.format(measured),
                duration_ms=elapsed,
            )
        )
    return results
