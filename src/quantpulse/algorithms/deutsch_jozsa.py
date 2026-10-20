"""Deutsch-Jozsa constant-versus-balanced oracle discrimination."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from ..circuit import Circuit

__all__ = ["DeutschJozsaResult", "deutsch_jozsa", "oracle_unitary"]


@dataclass(frozen=True)
class DeutschJozsaResult:
    """Decision and diagnostic data for one oracle."""

    n_qubits: int
    is_constant: bool
    zero_probability: float
    counts: dict[str, int]


def oracle_unitary(oracle: Callable[[int], int], n_qubits: int) -> np.ndarray:
    """Permutation unitary ``|x>|y> -> |x>|y xor f(x)>``.

    Qubit 0 is the ancilla so a basis index is ``(x << 1) | y``.
    """
    if n_qubits < 1:
        raise ValueError("n_qubits must be at least 1")
    size = 2 ** (n_qubits + 1)
    unitary = np.zeros((size, size), dtype=complex)
    for x in range(2**n_qubits):
        value = int(oracle(x))
        if value not in (0, 1):
            raise ValueError("oracle must return 0 or 1")
        for y in (0, 1):
            column = (x << 1) | y
            row = (x << 1) | (y ^ value)
            unitary[row, column] = 1.0
    return unitary


def deutsch_jozsa(
    oracle: Callable[[int], int],
    n_qubits: int,
    *,
    shots: int = 2048,
    seed: int = 0,
) -> DeutschJozsaResult:
    """Decide whether ``oracle`` is constant or balanced in one query."""
    unitary = oracle_unitary(oracle, n_qubits)
    circuit = Circuit(n_qubits + 1)
    circuit.x(0)
    for q in range(n_qubits + 1):
        circuit.h(q)
    circuit.add_global_unitary(unitary, name="oracle")
    for q in range(1, n_qubits + 1):
        circuit.h(q)
    state = circuit.run()
    probs = np.abs(state) ** 2
    # Input register holds qubits 1..n; all-zero iff the state index < 2.
    zero_probability = float(probs[0] + probs[1])
    counts = circuit.sample(shots, seed, qubits=list(range(n_qubits, 0, -1)))
    return DeutschJozsaResult(
        n_qubits=n_qubits,
        is_constant=zero_probability > 0.5,
        zero_probability=zero_probability,
        counts=counts,
    )
