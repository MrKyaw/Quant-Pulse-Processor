"""Quantum Fourier transform against the exact DFT matrix."""

from __future__ import annotations

import numpy as np
import pytest

from quantpulse.algorithms.qft import apply_qft, qft_circuit, qft_unitary


@pytest.mark.parametrize("n_qubits", [1, 2, 3, 4])
def test_circuit_matches_dft(n_qubits: int) -> None:
    circuit = qft_circuit(n_qubits)
    size = 2**n_qubits
    columns = [circuit.run(np.eye(size)[:, i]) for i in range(size)]
    assert np.allclose(np.column_stack(columns), qft_unitary(n_qubits))


@pytest.mark.parametrize("n_qubits", [2, 3, 4])
def test_inverse_is_adjoint(n_qubits: int) -> None:
    forward = qft_circuit(n_qubits)
    inverse = qft_circuit(n_qubits, inverse=True)
    assert np.allclose(inverse.run(forward.run()), np.eye(2**n_qubits)[0])


def test_inverse_round_trip_on_random_state() -> None:
    rng = np.random.default_rng(0)
    state = rng.normal(size=16) + 1j * rng.normal(size=16)
    state /= np.linalg.norm(state)
    restored = qft_circuit(4, inverse=True).run(qft_circuit(4).run(state))
    assert np.allclose(restored, state)


def test_qft_of_zero_is_uniform() -> None:
    state = qft_circuit(5).run()
    assert np.allclose(state, np.full(32, 1 / np.sqrt(32), dtype=complex))


def test_qft_of_basis_state_has_expected_phases() -> None:
    n_qubits = 3
    state = qft_circuit(n_qubits).run(np.eye(8)[1])
    expected = np.exp(2j * np.pi * np.arange(8) / 8) / np.sqrt(8)
    assert np.allclose(state, expected)


def test_apply_qft_matches_circuit() -> None:
    state = np.eye(4)[2].astype(complex)
    assert np.allclose(apply_qft(state), qft_circuit(2).run(state))
    assert np.allclose(apply_qft(state, inverse=True), qft_circuit(2, inverse=True).run(state))


def test_apply_qft_validation() -> None:
    with pytest.raises(ValueError):
        apply_qft(np.ones(3))
    with pytest.raises(ValueError):
        qft_circuit(0)


def test_qft_unitary_is_unitary() -> None:
    matrix = qft_unitary(4)
    assert np.allclose(matrix @ matrix.conj().T, np.eye(16))
    assert np.isclose(np.max(np.abs(np.abs(matrix) - 1 / 4)), 0.0)
