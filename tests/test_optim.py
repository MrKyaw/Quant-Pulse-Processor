"""Adam optimiser and the parameter-shift gradient rule."""

from __future__ import annotations

import numpy as np
import pytest

from quantpulse.gates import rz
from quantpulse.optim import adam_minimize, parameter_shift_gradient


def test_adam_minimises_a_quadratic() -> None:
    target = np.array([3.0, -1.0])

    def objective(x: np.ndarray) -> float:
        return float(np.sum((x - target) ** 2))

    def gradient(x: np.ndarray) -> np.ndarray:
        return 2.0 * (x - target)

    best, history = adam_minimize(objective, gradient, [0.0, 0.0], steps=200, learning_rate=0.1)
    assert np.allclose(best, target, atol=1e-2)
    assert len(history) == 200
    assert history[-1] < history[0]


def test_adam_validates_input() -> None:
    with pytest.raises(ValueError):
        adam_minimize(lambda x: 0.0, lambda x: x, [], steps=5)
    with pytest.raises(ValueError):
        adam_minimize(
            lambda x: 0.0,
            lambda x: np.zeros(2),
            [0.0],
            steps=5,
        )


def test_parameter_shift_matches_analytic_derivative() -> None:
    # f(theta) = <0| RZ(theta)^dagger Z RZ(theta) |0> = 1 (constant), so use RY.
    def objective(theta: np.ndarray) -> float:
        from quantpulse.gates import ry

        state = ry(theta[0]) @ np.array([1, 0], dtype=complex)
        return float(np.real(state.conj() @ np.diag([1, -1]) @ state))

    for theta in (0.2, 1.0, 2.4):
        grad = parameter_shift_gradient(objective, np.array([theta]), [np.pi / 2])[0]
        assert np.isclose(grad, -np.sin(theta), atol=1e-10)


def test_parameter_shift_multi_parameter() -> None:
    def objective(params: np.ndarray) -> float:
        return float(np.cos(params[0]) + 0.5 * np.sin(params[1]))

    params = np.array([0.3, -0.7])
    grad = parameter_shift_gradient(objective, params, [np.pi / 2, np.pi / 2])
    assert np.allclose(grad, [-np.sin(0.3), 0.5 * np.cos(-0.7)])


def test_parameter_shift_validation() -> None:
    with pytest.raises(ValueError):
        parameter_shift_gradient(lambda p: 0.0, np.zeros(2), [np.pi / 2])


def test_rz_is_unitary_for_shifted_angles() -> None:
    for angle in (0.0, np.pi / 2, np.pi):
        matrix = rz(angle)
        assert np.allclose(matrix @ matrix.conj().T, np.eye(2))
