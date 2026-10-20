"""Statevector math: sampling, fidelity, reduced states and entropy."""

from __future__ import annotations

import numpy as np
import pytest

from quantpulse import statevector as sv


def bell() -> np.ndarray:
    return np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)


def test_basis_state_and_bitstring() -> None:
    state = sv.basis_state(3, 5)
    assert np.allclose(state, np.eye(8)[5])
    assert sv.bitstring(5, 3) == "101"


@pytest.mark.parametrize("n_qubits,index", [(0, 0), (2, -1), (2, 4)])
def test_basis_state_rejects_bad_arguments(n_qubits: int, index: int) -> None:
    with pytest.raises(ValueError):
        sv.basis_state(n_qubits, index)


def test_normalize_and_zero_vector() -> None:
    vector = np.array([3, 4], dtype=complex)
    assert np.isclose(np.linalg.norm(sv.normalize(vector)), 1.0)
    with pytest.raises(ValueError):
        sv.normalize(np.zeros(2))


@pytest.mark.parametrize("size", [0, 3, 6])
def test_probabilities_reject_non_power_of_two(size: int) -> None:
    with pytest.raises(ValueError):
        sv.probabilities(np.ones(size))


def test_probabilities_sum_to_one() -> None:
    probs = sv.probabilities(bell())
    assert np.isclose(probs.sum(), 1.0)
    assert np.allclose(probs, [0.5, 0, 0, 0.5])


def test_density_matrix_is_projector() -> None:
    rho = sv.density_matrix(bell())
    assert np.allclose(rho, rho.conj().T)
    assert np.isclose(np.trace(rho).real, 1.0)
    assert np.allclose(rho @ rho, rho)


def test_sampling_is_deterministic_with_seed() -> None:
    counts_a = sv.sample_counts(bell(), 400, seed=11)
    counts_b = sv.sample_counts(bell(), 400, seed=11)
    assert counts_a == counts_b
    assert sum(counts_a.values()) == 400
    assert set(counts_a) <= {"00", "11"}


def test_sampling_subset_of_qubits() -> None:
    counts = sv.sample_counts(sv.basis_state(3, 0b101), 10, seed=2, qubits=[2, 0])
    assert counts == {"11": 10}


def test_sampling_validation() -> None:
    with pytest.raises(ValueError):
        sv.sample_counts(bell(), 0)
    with pytest.raises(ValueError):
        sv.sample_counts(bell(), 10, qubits=[])
    with pytest.raises(ValueError):
        sv.sample_counts(bell(), 10, qubits=[0, 0])
    with pytest.raises(ValueError):
        sv.sample_counts(bell(), 10, qubits=[2])


def test_fidelity_limits() -> None:
    assert np.isclose(sv.fidelity(bell(), bell()), 1.0)
    assert np.isclose(sv.fidelity(sv.basis_state(1, 0), sv.basis_state(1, 1)), 0.0)
    with pytest.raises(ValueError):
        sv.fidelity(sv.basis_state(1, 0), bell())


def test_partial_trace_of_bell_state_is_maximally_mixed() -> None:
    rho = sv.partial_trace(bell(), [0])
    assert np.allclose(rho, 0.5 * np.eye(2))
    entropy = sv.von_neumann_entropy(rho)
    assert np.isclose(entropy, 1.0)


def test_partial_trace_of_product_state_is_pure() -> None:
    state = np.kron(np.array([1, 0]), np.array([1, 1]) / np.sqrt(2))
    rho = sv.partial_trace(state, [0])
    assert np.allclose(rho, 0.5 * np.ones((2, 2)))
    assert np.isclose(sv.von_neumann_entropy(rho), 0.0)


def test_partial_trace_of_all_wires_matches_density_matrix() -> None:
    state = bell()
    assert np.allclose(sv.partial_trace(state, [0, 1]), sv.density_matrix(state))


def test_partial_trace_validation() -> None:
    with pytest.raises(ValueError):
        sv.partial_trace(bell(), [0, 0])
    with pytest.raises(ValueError):
        sv.partial_trace(bell(), [2])


def test_entropy_rejects_non_square() -> None:
    with pytest.raises(ValueError):
        sv.von_neumann_entropy(np.zeros((2, 3)))


@pytest.mark.parametrize(
    "state,label,expected",
    [
        (sv.basis_state(1, 0), "Z", 1.0),
        (sv.basis_state(1, 1), "Z", -1.0),
        (sv.basis_state(1, 0), "X", 0.0),
        (np.array([1, 1], dtype=complex) / np.sqrt(2), "X", 1.0),
        (bell(), "ZZ", 1.0),
        (bell(), "XX", 1.0),
        (bell(), "ZI", 0.0),
    ],
)
def test_expectation_pauli(state: np.ndarray, label: str, expected: float) -> None:
    assert np.isclose(sv.expectation_pauli(state, label), expected)


def test_expectation_pauli_length_validation() -> None:
    with pytest.raises(ValueError):
        sv.expectation_pauli(bell(), "Z")


def test_most_likely() -> None:
    assert sv.most_likely(sv.basis_state(2, 2)) == "10"
    state = np.array([0, 3, 0, 0], dtype=complex)
    assert sv.most_likely(state) == "01"
