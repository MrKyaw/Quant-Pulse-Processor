"""Statevector utilities: construction, measurement, sampling and entanglement.

The engine stores a pure state as a complex vector of length ``2**n`` with
qubit 0 as the least significant bit.  Bitstring labels are printed
big-endian (``q_{n-1} ... q_0``), matching the usual quantum-computing
convention.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np

__all__ = [
    "basis_state",
    "bitstring",
    "density_matrix",
    "expectation_pauli",
    "fidelity",
    "most_likely",
    "normalize",
    "partial_trace",
    "probabilities",
    "sample_counts",
    "von_neumann_entropy",
]


def _check_dim(state: np.ndarray, n_qubits: int | None = None) -> np.ndarray:
    vec = np.asarray(state, dtype=complex).reshape(-1)
    size = vec.size
    if size == 0 or size & (size - 1):
        raise ValueError("Statevector length must be a positive power of two")
    if n_qubits is not None and size != 2**n_qubits:
        raise ValueError(f"Expected {2**n_qubits} amplitudes, received {size}")
    return vec


def basis_state(n_qubits: int, index: int = 0) -> np.ndarray:
    """Return the computational basis state ``|index>``."""
    if n_qubits < 1:
        raise ValueError("n_qubits must be at least 1")
    if not 0 <= index < 2**n_qubits:
        raise ValueError(f"index must be in [0, {2**n_qubits})")
    state = np.zeros(2**n_qubits, dtype=complex)
    state[index] = 1.0
    return state


def bitstring(index: int, n_qubits: int) -> str:
    """Format a basis index as a big-endian bitstring."""
    return format(index, f"0{n_qubits}b")


def normalize(state: np.ndarray) -> np.ndarray:
    """Return a unit-norm copy of ``state``."""
    vec = _check_dim(state)
    norm = np.linalg.norm(vec)
    if norm == 0:
        raise ValueError("Cannot normalize a zero vector")
    return vec / norm


def probabilities(state: np.ndarray) -> np.ndarray:
    """Measurement probabilities ``|amplitude|^2`` in the computational basis."""
    vec = _check_dim(state)
    probs = np.abs(vec) ** 2
    total = probs.sum()
    if total == 0:
        raise ValueError("Cannot compute probabilities of a zero vector")
    # Clip the tiny negative values that BLAS rounding can produce.
    return np.clip(probs / total, 0.0, None)


def density_matrix(state: np.ndarray) -> np.ndarray:
    """Pure-state density matrix ``|psi><psi|``."""
    vec = _check_dim(state)
    return np.outer(vec, vec.conj())


def sample_counts(
    state: np.ndarray,
    shots: int,
    seed: int | None = None,
    *,
    qubits: Sequence[int] | None = None,
) -> dict[str, int]:
    """Sample ``shots`` bitstrings from ``|psi|^2`` with a deterministic RNG.

    ``qubits`` selects a subset of wires, ordered from most to least
    significant in the resulting labels.
    """
    if shots < 1:
        raise ValueError("shots must be positive")
    vec = _check_dim(state)
    probs = probabilities(vec)
    n_qubits = int(np.log2(vec.size))
    rng = np.random.default_rng(seed)
    draws = rng.choice(vec.size, size=shots, p=probs)
    if qubits is None:
        labels = [bitstring(int(i), n_qubits) for i in draws]
    else:
        wires = list(qubits)
        if not wires or len(set(wires)) != len(wires):
            raise ValueError("qubits must be a non-empty set of distinct wires")
        if any(q < 0 or q >= n_qubits for q in wires):
            raise ValueError("qubit index out of range")
        labels = ["".join(str((int(i) >> q) & 1) for q in wires) for i in draws]
    counts: dict[str, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: -item[1]))


def fidelity(left: np.ndarray, right: np.ndarray) -> float:
    """State fidelity ``|<left|right>|^2`` for pure states."""
    a = normalize(left)
    b = normalize(right)
    if a.size != b.size:
        raise ValueError("States must have the same dimension")
    return float(np.abs(np.vdot(a, b)) ** 2)


def partial_trace(state: np.ndarray, keep: Iterable[int]) -> np.ndarray:
    """Reduced density matrix over the ``keep`` wires."""
    vec = _check_dim(state)
    n_qubits = int(np.log2(vec.size))
    keep_list = list(keep)
    if len(set(keep_list)) != len(keep_list):
        raise ValueError("keep must contain distinct wires")
    if any(q < 0 or q >= n_qubits for q in keep_list):
        raise ValueError("qubit index out of range")
    trace_out = [q for q in range(n_qubits) if q not in keep_list]
    tensor = vec.reshape([2] * n_qubits)
    # C-order reshape puts qubit q on tensor axis ``n_qubits - 1 - q``.
    keep_axes = [n_qubits - 1 - q for q in keep_list]
    trace_axes = [n_qubits - 1 - q for q in trace_out]
    tensor = np.transpose(tensor, keep_axes + trace_axes)
    dim_keep = 2 ** len(keep_list)
    matrix = tensor.reshape(dim_keep, -1)
    return matrix @ matrix.conj().T


def von_neumann_entropy(rho: np.ndarray) -> float:
    """Von Neumann entropy in bits of a density matrix."""
    matrix = np.asarray(rho, dtype=complex)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Density matrix must be square")
    eigenvalues = np.linalg.eigvalsh(matrix).real
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    eigenvalues = eigenvalues[eigenvalues > 1e-12]
    return float(-np.sum(eigenvalues * np.log2(eigenvalues)))


def expectation_pauli(state: np.ndarray, label: str) -> float:
    """Expectation value of a Pauli string, e.g. ``"XZ"``."""
    vec = _check_dim(state)
    n_qubits = int(np.log2(vec.size))
    if len(label) != n_qubits:
        raise ValueError(f"Pauli label must have {n_qubits} characters")
    from .gates import pauli

    return float(np.vdot(vec, pauli(label) @ vec).real)


def most_likely(state: np.ndarray) -> str:
    """Most probable computational-basis bitstring."""
    vec = _check_dim(state)
    probs = probabilities(vec)
    return bitstring(int(np.argmax(probs)), int(np.log2(vec.size)))
