"""VQE ansatz, Hamiltonians and convergence against exact diagonalisation."""

from __future__ import annotations

import numpy as np
import pytest

from quantpulse.algorithms.vqe import (
    ansatz_circuit,
    ansatz_energy,
    exact_ground_energy,
    hamiltonian_matrix,
    optimize_vqe,
    tfim_terms,
)


def test_tfim_terms_two_qubits() -> None:
    terms = tfim_terms(2, coupling=1.0, field=0.5)
    assert len(terms) == 3
    assert sum(coeff for coeff, _ in terms) == pytest.approx(-2.0)


def test_tfim_terms_ring_structure() -> None:
    terms = tfim_terms(3)
    labels = [label for _, label in terms]
    assert labels.count("XII") + labels.count("IXI") + labels.count("IIX") == 3
    zz_labels = [label for label in labels if label.count("Z") == 2]
    assert len(zz_labels) == 3


def test_tfim_terms_validation() -> None:
    with pytest.raises(ValueError):
        tfim_terms(1)


def test_hamiltonian_is_hermitian() -> None:
    matrix = hamiltonian_matrix(tfim_terms(3))
    assert np.allclose(matrix, matrix.conj().T)
    assert matrix.shape == (8, 8)


def test_hamiltonian_validation() -> None:
    with pytest.raises(ValueError):
        hamiltonian_matrix([])
    with pytest.raises(ValueError):
        hamiltonian_matrix([(1.0, "X"), (1.0, "ZZ")])


def test_exact_energy_of_two_qubit_model() -> None:
    terms = tfim_terms(2, coupling=1.0, field=0.5)
    energy = exact_ground_energy(terms)
    # H = -ZZ - 0.5 (X0 + X1); the ground energy is -sqrt(2).
    assert energy == pytest.approx(-np.sqrt(2), abs=1e-12)


def test_ansatz_parameter_count() -> None:
    circuit = ansatz_circuit(3, 2, np.zeros(9))
    assert len(circuit) == 3 * 3 + 2 * 2
    assert circuit.gate_counts()["cnot"] == 4


def test_ansatz_validation() -> None:
    with pytest.raises(ValueError):
        ansatz_circuit(2, 0, np.zeros(4))
    with pytest.raises(ValueError):
        ansatz_circuit(2, 1, np.zeros(3))


def test_ansatz_energy_of_zero_state() -> None:
    terms = tfim_terms(2)
    energy = ansatz_energy(2, 1, np.zeros(4), terms)
    matrix = hamiltonian_matrix(terms)
    expected = np.real(np.eye(4)[0] @ matrix @ np.eye(4)[0])
    assert energy == pytest.approx(expected)


def test_vqe_reaches_the_ground_state() -> None:
    result = optimize_vqe(2, layers=2, steps=150, seed=0)
    assert result.error < 1e-3
    assert result.energy <= result.exact_energy + 1e-6


def test_vqe_three_qubits() -> None:
    result = optimize_vqe(3, layers=3, steps=200, seed=1)
    assert result.error < 1e-6


def test_vqe_history_length() -> None:
    result = optimize_vqe(2, layers=1, steps=30, seed=0)
    assert len(result.history) == 30
    assert len(result.parameters) == 4
