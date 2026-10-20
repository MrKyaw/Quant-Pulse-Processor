"""Gate matrices: unitarity, known actions and parametrisations."""

from __future__ import annotations

import numpy as np
import pytest

from quantpulse import gates

CONSTANTS = ["I", "X", "Y", "Z", "H", "S", "SDG", "T", "TDG", "SX", "CNOT", "CZ", "SWAP", "TOFFOLI"]


@pytest.mark.parametrize("name", CONSTANTS)
def test_named_gates_are_unitary(name: str) -> None:
    assert gates.is_unitary(getattr(gates, name))


@pytest.mark.parametrize("name", CONSTANTS)
def test_named_gates_have_power_of_two_dimension(name: str) -> None:
    matrix = getattr(gates, name)
    assert matrix.shape[0] == matrix.shape[1]
    assert matrix.shape[0] & (matrix.shape[0] - 1) == 0


@pytest.mark.parametrize("theta", [0.0, 0.3, np.pi / 2, np.pi, -1.7, 4.2])
def test_rotations_are_unitary(theta: float) -> None:
    for factory in (gates.rx, gates.ry, gates.rz):
        assert gates.is_unitary(factory(theta))


def test_pauli_actions() -> None:
    zero, one = np.array([1, 0], dtype=complex), np.array([0, 1], dtype=complex)
    assert np.allclose(gates.X @ zero, one)
    assert np.allclose(gates.Z @ zero, zero)
    assert np.allclose(gates.Z @ one, -one)
    assert np.allclose(gates.H @ zero, np.array([1, 1]) / np.sqrt(2))


def test_pauli_y_is_hermitian_and_squares_to_identity() -> None:
    assert np.allclose(gates.Y, gates.Y.conj().T)
    assert np.allclose(gates.Y @ gates.Y, np.eye(2))


def test_rx_pi_is_minus_i_x() -> None:
    assert np.allclose(gates.rx(np.pi), -1j * gates.X)


def test_ry_pi_swaps_basis_states() -> None:
    zero = np.array([1, 0], dtype=complex)
    assert np.allclose(gates.ry(np.pi) @ zero, np.array([0, 1], dtype=complex))


def test_rz_pi_is_minus_i_z() -> None:
    assert np.allclose(gates.rz(np.pi), -1j * gates.Z)


def test_s_and_t_are_fourth_and_eighth_roots() -> None:
    assert np.allclose(gates.S @ gates.S, gates.Z)
    assert np.allclose(gates.T @ gates.T, gates.S)
    assert np.allclose(gates.SDG @ gates.S, np.eye(2))


def test_sx_squares_to_x() -> None:
    assert np.allclose(gates.SX @ gates.SX, gates.X)


def test_phase_gate_diagonal() -> None:
    assert np.allclose(gates.phase(np.pi / 3), np.diag([1, np.exp(1j * np.pi / 3)]))


def test_cphase_flips_only_eleven() -> None:
    matrix = gates.cphase(np.pi / 2)
    assert np.allclose(np.diag(matrix), [1, 1, 1, 1j])
    assert np.allclose(matrix - np.diag(np.diag(matrix)), 0)


def test_cnot_truth_table() -> None:
    basis = np.eye(4, dtype=complex)
    mapping = {0: 0, 1: 1, 2: 3, 3: 2}
    for column, row in mapping.items():
        assert np.allclose(gates.CNOT @ basis[:, column], basis[:, row])


def test_toffoli_truth_table() -> None:
    basis = np.eye(8, dtype=complex)
    mapping = {6: 7, 7: 6}
    for column in range(8):
        expected = mapping.get(column, column)
        assert np.allclose(gates.TOFFOLI @ basis[:, column], basis[:, expected])


def test_swap_exchanges_wires() -> None:
    state = np.array([0, 1, 0, 0], dtype=complex)
    assert np.allclose(gates.SWAP @ state, np.array([0, 0, 1, 0], dtype=complex))


def test_pauli_kron_product() -> None:
    assert np.allclose(gates.pauli("XZ"), np.kron(gates.X, gates.Z))
    assert np.allclose(gates.pauli("i"), np.eye(2))


@pytest.mark.parametrize("label", ["", "Q", "XQ"])
def test_pauli_rejects_invalid_labels(label: str) -> None:
    with pytest.raises(ValueError):
        gates.pauli(label)


def test_u3_unitarity_and_identity() -> None:
    assert gates.is_unitary(gates.u3(0.3, 0.4, 0.5))
    assert np.allclose(gates.u3(0.0, 0.0, 0.0), np.eye(2))


def test_is_unitary_rejects_bad_input() -> None:
    assert not gates.is_unitary(np.zeros((2, 2)))
    assert not gates.is_unitary(np.zeros((2, 3)))
    assert not gates.is_unitary(np.eye(2)[0])
