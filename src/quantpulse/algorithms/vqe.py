"""Variational quantum eigensolver for small spin Hamiltonians.

The reference problem is a transverse-field Ising model

    H = -J sum_<i,j> Z_i Z_j - h sum_i X_i

on a ring, with the exact ground energy obtained by dense diagonalisation so
every optimisation run can be scored against the true answer.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ..circuit import Circuit
from ..gates import pauli
from ..optim import adam_minimize, parameter_shift_gradient

__all__ = [
    "VQEResult",
    "ansatz_circuit",
    "ansatz_energy",
    "exact_ground_energy",
    "hamiltonian_matrix",
    "optimize_vqe",
    "tfim_terms",
]


@dataclass(frozen=True)
class VQEResult:
    """Optimisation outcome compared with the exact ground state."""

    n_qubits: int
    layers: int
    energy: float
    exact_energy: float
    error: float
    parameters: tuple[float, ...]
    history: tuple[float, ...]


def tfim_terms(
    n_qubits: int,
    *,
    coupling: float = 1.0,
    field: float = 0.5,
) -> list[tuple[float, str]]:
    """Pauli decomposition of the ring TFIM Hamiltonian.

    Each label is big-endian: character ``i`` acts on qubit ``n-1-i``.
    """
    if n_qubits < 2:
        raise ValueError("TFIM needs at least two qubits")
    terms: list[tuple[float, str]] = []
    for q in range(n_qubits):
        label = ["I"] * n_qubits
        label[n_qubits - 1 - q] = "X"
        terms.append((-field, "".join(label)))
    for q in range(n_qubits):
        u, v = q, (q + 1) % n_qubits
        if n_qubits == 2 and q == 1:
            break  # avoid counting the single edge twice
        label = ["I"] * n_qubits
        label[n_qubits - 1 - u] = "Z"
        label[n_qubits - 1 - v] = "Z"
        terms.append((-coupling, "".join(label)))
    return terms


def hamiltonian_matrix(terms: Sequence[tuple[float, str]]) -> np.ndarray:
    """Dense matrix for a weighted Pauli sum."""
    if not terms:
        raise ValueError("terms must not be empty")
    width = len(terms[0][1])
    matrix = np.zeros((2**width, 2**width), dtype=complex)
    for coefficient, label in terms:
        if len(label) != width:
            raise ValueError("All Pauli labels must have the same length")
        matrix += coefficient * pauli(label)
    return matrix


def exact_ground_energy(terms: Sequence[tuple[float, str]]) -> float:
    """Lowest eigenvalue of the Hamiltonian, by dense diagonalisation."""
    return float(np.linalg.eigvalsh(hamiltonian_matrix(terms))[0])


def ansatz_circuit(n_qubits: int, layers: int, parameters: Sequence[float]) -> Circuit:
    """Hardware-efficient ansatz: RY layers separated by CNOT ladders."""
    if layers < 1:
        raise ValueError("layers must be at least 1")
    params = np.asarray(parameters, dtype=float).reshape(-1)
    expected = n_qubits * (layers + 1)
    if params.size != expected:
        raise ValueError(f"Expected {expected} parameters, received {params.size}")
    circuit = Circuit(n_qubits)
    cursor = 0
    for _ in range(layers):
        for q in range(n_qubits):
            circuit.ry(q, float(params[cursor]))
            cursor += 1
        for q in range(n_qubits - 1):
            circuit.cnot(q, q + 1)
    for q in range(n_qubits):
        circuit.ry(q, float(params[cursor]))
        cursor += 1
    return circuit


def ansatz_energy(
    n_qubits: int,
    layers: int,
    parameters: Sequence[float],
    terms: Sequence[tuple[float, str]],
) -> float:
    """Expectation value of the Hamiltonian on the ansatz state."""
    state = ansatz_circuit(n_qubits, layers, parameters).run()
    matrix = hamiltonian_matrix(terms)
    return float(np.vdot(state, matrix @ state).real)


def optimize_vqe(
    n_qubits: int = 2,
    layers: int = 2,
    *,
    coupling: float = 1.0,
    field: float = 0.5,
    steps: int = 200,
    learning_rate: float = 0.1,
    seed: int = 0,
) -> VQEResult:
    """Optimise the ansatz with parameter-shift gradients and Adam."""
    terms = tfim_terms(n_qubits, coupling=coupling, field=field)
    exact = exact_ground_energy(terms)
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(0.0, 2 * np.pi, size=n_qubits * (layers + 1))

    def objective(params: np.ndarray) -> float:
        return ansatz_energy(n_qubits, layers, params, terms)

    shifts = [np.pi / 2] * x0.size

    def gradient(params: np.ndarray) -> np.ndarray:
        return parameter_shift_gradient(objective, params, shifts)

    best, history = adam_minimize(
        objective,
        gradient,
        x0,
        steps=steps,
        learning_rate=learning_rate,
    )
    energy = float(objective(best))
    return VQEResult(
        n_qubits=n_qubits,
        layers=layers,
        energy=energy,
        exact_energy=exact,
        error=abs(energy - exact),
        parameters=tuple(float(x) for x in best),
        history=tuple(history),
    )
