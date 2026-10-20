"""Small optimisers used by the variational algorithms.

The package deliberately depends only on NumPy, so Adam and the
parameter-shift gradient rule are implemented here rather than imported from
SciPy or a quantum SDK.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

__all__ = ["adam_minimize", "parameter_shift_gradient"]


def adam_minimize(
    objective: Callable[[np.ndarray], float],
    gradient: Callable[[np.ndarray], np.ndarray],
    x0: Sequence[float],
    *,
    steps: int = 120,
    learning_rate: float = 0.15,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1e-8,
    tol: float = 1e-10,
) -> tuple[np.ndarray, list[float]]:
    """Minimise ``objective`` with Adam and return ``(best_x, history)``."""
    params = np.asarray(x0, dtype=float).copy()
    if params.ndim != 1 or params.size == 0:
        raise ValueError("x0 must be a non-empty vector")
    first = np.zeros_like(params)
    second = np.zeros_like(params)
    history: list[float] = []
    best_x = params.copy()
    best_value = float(objective(params))
    for step in range(1, steps + 1):
        grad = np.asarray(gradient(params), dtype=float)
        if grad.shape != params.shape:
            raise ValueError("gradient must match the parameter shape")
        first = beta1 * first + (1 - beta1) * grad
        second = beta2 * second + (1 - beta2) * grad**2
        m_hat = first / (1 - beta1**step)
        v_hat = second / (1 - beta2**step)
        params = params - learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
        value = float(objective(params))
        history.append(value)
        if value < best_value - tol:
            best_value = value
            best_x = params.copy()
    return best_x, history


def parameter_shift_gradient(
    objective: Callable[[np.ndarray], float],
    params: np.ndarray,
    shifts: Sequence[float],
) -> np.ndarray:
    """Central parameter-shift gradient for gate-angle parameters."""
    if len(shifts) != params.size:
        raise ValueError("shifts must match the parameter count")
    grad = np.zeros_like(params)
    for i, shift in enumerate(shifts):
        plus = params.copy()
        minus = params.copy()
        plus[i] += shift
        minus[i] -= shift
        grad[i] = 0.5 * (objective(plus) - objective(minus))
    return grad
