"""Amplitude-amplification (Grover) search."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from ..circuit import Circuit
from ..metrics import success_probability
from ..statevector import sample_counts

__all__ = [
    "GroverResult",
    "diffusion_operator",
    "grover_circuit",
    "optimal_iterations",
    "phase_oracle",
    "run_grover",
]


@dataclass(frozen=True)
class GroverResult:
    """Outcome of a Grover run."""

    n_qubits: int
    marked: tuple[int, ...]
    iterations: int
    success_probability: float
    top_bitstring: str
    counts: dict[str, int]

    @property
    def found(self) -> bool:
        """True when the most likely bitstring is a marked item."""
        return int(self.top_bitstring, 2) in self.marked


def optimal_iterations(n_qubits: int, n_marked: int = 1) -> int:
    """Iteration count maximising the marked amplitude (rounded optimum)."""
    size = 2**n_qubits
    if not 1 <= n_marked < size:
        raise ValueError("n_marked must satisfy 1 <= n_marked < 2**n_qubits")
    theta = np.arcsin(np.sqrt(n_marked / size))
    return max(1, round(np.pi / (4 * theta) - 0.5))


def phase_oracle(n_qubits: int, marked: Iterable[int]) -> np.ndarray:
    """Diagonal unitary that flips the sign of every marked basis state."""
    targets = sorted(set(int(m) for m in marked))
    size = 2**n_qubits
    if not targets:
        raise ValueError("marked must not be empty")
    if any(t < 0 or t >= size for t in targets):
        raise ValueError("marked index out of range")
    diagonal = np.ones(size, dtype=complex)
    diagonal[targets] = -1.0
    return np.diag(diagonal)


def diffusion_operator(n_qubits: int) -> np.ndarray:
    """Grover diffusion ``2|s><s| - I`` about the uniform superposition."""
    size = 2**n_qubits
    uniform = np.ones(size, dtype=complex) / np.sqrt(size)
    return 2.0 * np.outer(uniform, uniform.conj()) - np.eye(size, dtype=complex)


def grover_circuit(
    n_qubits: int,
    marked: Iterable[int],
    iterations: int | None = None,
) -> Circuit:
    """Prepare ``H^n`` then alternate the phase oracle and diffusion."""
    targets = tuple(sorted(set(int(m) for m in marked)))
    if iterations is None:
        iterations = optimal_iterations(n_qubits, len(targets))
    if iterations < 0:
        raise ValueError("iterations must be non-negative")
    circuit = Circuit(n_qubits)
    for q in range(n_qubits):
        circuit.h(q)
    oracle = phase_oracle(n_qubits, targets)
    diffusion = diffusion_operator(n_qubits)
    for _ in range(iterations):
        circuit.add_global_unitary(oracle, name="oracle")
        circuit.add_global_unitary(diffusion, name="diffusion")
    return circuit


def run_grover(
    n_qubits: int,
    marked: Iterable[int],
    iterations: int | None = None,
    *,
    shots: int = 4096,
    seed: int = 0,
) -> GroverResult:
    """Execute Grover search and report the success probability."""
    targets = tuple(sorted(set(int(m) for m in marked)))
    circuit = grover_circuit(n_qubits, targets, iterations)
    state = circuit.run()
    probability = success_probability(state, targets)
    counts = sample_counts(state, shots, seed)
    top = next(iter(counts))
    return GroverResult(
        n_qubits=n_qubits,
        marked=targets,
        iterations=(len(circuit) - n_qubits) // 2,
        success_probability=probability,
        top_bitstring=top,
        counts=counts,
    )
