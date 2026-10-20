"""Circuit construction, execution and inspection."""

from __future__ import annotations

import numpy as np
import pytest

from quantpulse import Circuit, gates


def test_bell_state() -> None:
    circuit = Circuit(2).h(0).cnot(0, 1)
    probs = circuit.probabilities()
    assert np.allclose(probs, [0.5, 0, 0, 0.5])
    assert circuit.depth == 2
    assert circuit.gate_counts() == {"cnot": 1, "h": 1}


def test_ghz_state() -> None:
    probs = Circuit(3).h(0).cnot(0, 1).cnot(1, 2).probabilities()
    expected = np.zeros(8)
    expected[0] = expected[7] = 0.5
    assert np.allclose(probs, expected)


def test_x_flips_qubit_zero() -> None:
    assert np.allclose(Circuit(1).x(0).run(), [0, 1])


def test_x_on_qubit_one_flips_second_wire() -> None:
    assert np.allclose(Circuit(2).x(1).run(), [0, 0, 1, 0])


def test_cnot_control_and_target_order() -> None:
    state = Circuit(2).x(0).cnot(0, 1).run()
    assert np.allclose(state, [0, 0, 0, 1])
    state = Circuit(2).x(1).cnot(0, 1).run()
    assert np.allclose(state, [0, 0, 1, 0])


def test_cz_only_changes_eleven() -> None:
    state = Circuit(2).h(0).h(1).cz(0, 1).run()
    expected = np.array([1, 1, 1, -1], dtype=complex) / 2
    assert np.allclose(state, expected)


def test_swap() -> None:
    state = Circuit(2).x(0).swap(0, 1).run()
    assert np.allclose(state, [0, 0, 1, 0])


def test_toffoli() -> None:
    state = Circuit(3).x(0).x(1).toffoli(0, 1, 2).run()
    assert np.allclose(state, np.eye(8)[7])


def test_rotation_gates_match_matrices() -> None:
    state = Circuit(1).ry(0, 0.7).run()
    assert np.allclose(state, gates.ry(0.7) @ np.array([1, 0]))


def test_gate_parameters_are_recorded() -> None:
    circuit = Circuit(1).rx(0, 0.25).phase(0, -0.5)
    assert circuit.operations[0].params == (0.25,)
    assert circuit.operations[1].params == (-0.5,)


def test_inverse_restores_state() -> None:
    rng = np.random.default_rng(4)
    state = rng.normal(size=8) + 1j * rng.normal(size=8)
    state /= np.linalg.norm(state)
    circuit = Circuit(3).h(0).cnot(0, 1).ry(1, 0.4).toffoli(0, 1, 2).rz(2, -0.9)
    restored = circuit.inverse().run(circuit.run(state))
    assert np.allclose(restored, state)


def test_global_unitary_matches_matrix_product() -> None:
    rng = np.random.default_rng(2)
    matrix, _ = np.linalg.qr(rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4)))
    state = rng.normal(size=4) + 1j * rng.normal(size=4)
    state /= np.linalg.norm(state)
    result = Circuit(2).add_global_unitary(matrix).run(state)
    assert np.allclose(result, matrix @ state)


def test_add_unitary_uses_listed_qubit_order() -> None:
    rng = np.random.default_rng(3)
    matrix, _ = np.linalg.qr(rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2)))
    state = Circuit(2).x(0).add_unitary(matrix, [0]).run()
    expected = np.kron(np.eye(2), matrix) @ Circuit(2).x(0).run()
    assert np.allclose(state, expected)


@pytest.mark.parametrize(
    "operation",
    [
        lambda: Circuit(2).add_unitary(np.zeros((4, 4)), [0, 1]),
        lambda: Circuit(2).add_unitary(np.eye(4), [0]),
        lambda: Circuit(2).cnot(0, 0),
        lambda: Circuit(2).h(5),
        lambda: Circuit(2).h(-1),
    ],
)
def test_invalid_operations_raise(operation) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ValueError):
        operation()


def test_global_unitary_validation() -> None:
    with pytest.raises(ValueError):
        Circuit(2).add_global_unitary(np.eye(2))
    with pytest.raises(ValueError):
        Circuit(2).add_global_unitary(np.zeros((4, 4)))


def test_invalid_initial_state() -> None:
    with pytest.raises(ValueError):
        Circuit(2).run(np.ones(2))


def test_invalid_qubit_count() -> None:
    with pytest.raises(ValueError):
        Circuit(0)


def test_sampling_with_seed() -> None:
    circuit = Circuit(2).h(0).cnot(0, 1)
    assert circuit.sample(200, seed=7) == circuit.sample(200, seed=7)


def test_depth_respects_wire_overlap() -> None:
    circuit = Circuit(3).h(0).h(1).cnot(0, 2)
    assert circuit.depth == 2
    circuit = Circuit(3).h(0).h(1).h(2)
    assert circuit.depth == 1


def test_gate_counts_and_draw() -> None:
    circuit = Circuit(2).h(0).cnot(0, 1).rx(1, 0.5)
    assert circuit.gate_counts() == {"cnot": 1, "h": 1, "rx": 1}
    drawing = circuit.draw()
    assert "cnot" in drawing
    assert "rx(0.500)" in drawing


def test_len_and_operations() -> None:
    circuit = Circuit(1).h(0).x(0)
    assert len(circuit) == 2
    assert circuit.operations[0].name == "h"
