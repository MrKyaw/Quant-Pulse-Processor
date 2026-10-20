"""Quantum Fourier transform and its inverse."""

from __future__ import annotations

import numpy as np

from ..circuit import Circuit

__all__ = ["apply_qft", "qft_circuit", "qft_unitary"]


def qft_circuit(n_qubits: int, *, inverse: bool = False) -> Circuit:
    """Textbook QFT circuit (Hadamards, controlled phases, final swaps).

    The circuit is laid out for the package's little-endian convention, where
    qubit 0 is the least significant bit, so that the resulting unitary equals
    :func:`qft_unitary` exactly.
    """
    if n_qubits < 1:
        raise ValueError("n_qubits must be at least 1")
    circuit = Circuit(n_qubits)
    for q in reversed(range(n_qubits)):
        circuit.h(q)
        for control in range(q - 1, -1, -1):
            circuit.cphase(control, q, np.pi / 2 ** (q - control))
    for i in range(n_qubits // 2):
        circuit.swap(i, n_qubits - 1 - i)
    return circuit.inverse() if inverse else circuit


def qft_unitary(n_qubits: int) -> np.ndarray:
    """Dense DFT matrix ``U[x, y] = exp(2 pi i x y / N) / sqrt(N)``."""
    if n_qubits < 1:
        raise ValueError("n_qubits must be at least 1")
    size = 2**n_qubits
    indices = np.arange(size)
    omega = np.exp(2j * np.pi * np.outer(indices, indices) / size)
    return omega / np.sqrt(size)


def apply_qft(state: np.ndarray, *, inverse: bool = False) -> np.ndarray:
    """Apply the QFT (or its inverse) directly to a statevector."""
    vector = np.asarray(state, dtype=complex).reshape(-1)
    n_qubits = int(np.log2(vector.size))
    if 2**n_qubits != vector.size:
        raise ValueError("Statevector length must be a power of two")
    unitary = qft_unitary(n_qubits)
    if inverse:
        unitary = unitary.conj().T
    return unitary @ vector
