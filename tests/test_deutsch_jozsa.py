"""Deutsch-Jozsa discrimination of constant and balanced oracles."""

from __future__ import annotations

import numpy as np
import pytest

from quantpulse.algorithms.deutsch_jozsa import deutsch_jozsa, oracle_unitary


@pytest.mark.parametrize("n_qubits", [1, 2, 3, 4])
@pytest.mark.parametrize("constant", [0, 1])
def test_constant_oracles(n_qubits: int, constant: int) -> None:
    result = deutsch_jozsa(lambda _: constant, n_qubits, shots=64, seed=0)
    assert result.is_constant
    assert np.isclose(result.zero_probability, 1.0)


@pytest.mark.parametrize("n_qubits", [1, 2, 3, 4])
def test_parity_oracles_are_balanced(n_qubits: int) -> None:
    result = deutsch_jozsa(lambda x: bin(x).count("1") % 2, n_qubits, shots=64, seed=0)
    assert not result.is_constant
    assert result.zero_probability < 1e-12


@pytest.mark.parametrize("n_qubits", [2, 3, 4])
def test_half_split_oracles_are_balanced(n_qubits: int) -> None:
    half = 2 ** (n_qubits - 1)
    result = deutsch_jozsa(lambda x: int(x >= half), n_qubits, shots=64, seed=0)
    assert not result.is_constant


def test_top_bit_oracle_is_balanced() -> None:
    n_qubits = 4
    result = deutsch_jozsa(lambda x: (x >> (n_qubits - 1)) & 1, n_qubits, shots=64, seed=0)
    assert not result.is_constant


def test_counts_are_restricted_to_the_zero_string_for_constant() -> None:
    result = deutsch_jozsa(lambda _: 1, 3, shots=128, seed=1)
    assert set(result.counts) == {"000"}
    assert sum(result.counts.values()) == 128


def test_oracle_unitary_is_a_permutation() -> None:
    unitary = oracle_unitary(lambda x: x % 2, 3)
    assert np.allclose(unitary.sum(axis=0), 1)
    assert np.allclose(unitary.sum(axis=1), 1)
    assert np.allclose(unitary @ unitary.conj().T, np.eye(16))


def test_oracle_unitary_flips_the_ancilla_when_required() -> None:
    unitary = oracle_unitary(lambda x: 1, 1)
    # |x=0, y=0> -> |x=0, y=1>
    assert unitary[1, 0] == 1
    # |x=1, y=1> -> |x=1, y=0>
    assert unitary[2, 3] == 1


def test_invalid_oracle_output() -> None:
    with pytest.raises(ValueError):
        oracle_unitary(lambda _: 2, 2)


def test_invalid_qubit_count() -> None:
    with pytest.raises(ValueError):
        oracle_unitary(lambda _: 0, 0)
