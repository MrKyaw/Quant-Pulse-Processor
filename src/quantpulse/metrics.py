"""Metrics used by the algorithm modules, benchmarks and acceptance checks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from .statevector import fidelity, probabilities

__all__ = [
    "approximation_ratio",
    "cut_value",
    "relative_error",
    "state_fidelity",
    "success_probability",
    "summarize",
    "total_variation_distance",
    "trace_distance",
]


def state_fidelity(left: np.ndarray, right: np.ndarray) -> float:
    """Alias of :func:`quantpulse.statevector.fidelity`."""
    return fidelity(left, right)


def trace_distance(left: np.ndarray, right: np.ndarray) -> float:
    """Trace distance between two density matrices or pure states."""
    a = _as_density(left)
    b = _as_density(right)
    if a.shape != b.shape:
        raise ValueError("Operators must have the same shape")
    return float(0.5 * np.sum(np.abs(np.linalg.eigvalsh(a - b))))


def total_variation_distance(
    left: Mapping[str, float] | Sequence[float],
    right: Mapping[str, float] | Sequence[float],
) -> float:
    """Total variation distance between two probability distributions."""
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        keys = set(left) | set(right)
        total = sum(abs(float(left.get(key, 0.0)) - float(right.get(key, 0.0))) for key in keys)
        return float(0.5 * total)
    a = np.asarray(list(left.values()) if isinstance(left, Mapping) else left, dtype=float)
    b = np.asarray(list(right.values()) if isinstance(right, Mapping) else right, dtype=float)
    if a.shape != b.shape:
        raise ValueError("Sequence distributions must have the same length")
    return float(0.5 * np.sum(np.abs(a - b)))


def success_probability(state: np.ndarray, targets: Sequence[int]) -> float:
    """Probability mass on the marked basis indices."""
    probs = probabilities(state)
    if not targets:
        raise ValueError("targets must not be empty")
    if any(t < 0 or t >= probs.size for t in targets):
        raise ValueError("target index out of range")
    return float(np.sum(probs[list(targets)]))


def cut_value(index: int, edges: Sequence[tuple[int, int]]) -> int:
    """Number of cut edges for the bitstring encoded by ``index``."""
    return int(sum(((index >> u) & 1) != ((index >> v) & 1) for u, v in edges))


def approximation_ratio(expected: float, optimum: float) -> float:
    """``expected / optimum`` clipped to ``[0, 1]`` for maximisation problems."""
    if optimum <= 0:
        raise ValueError("optimum must be positive")
    return float(np.clip(expected / optimum, 0.0, 1.0))


def relative_error(value: float, reference: float) -> float:
    """Absolute error normalised by the magnitude of ``reference``."""
    if reference == 0:
        return float(abs(value))
    return float(abs(value - reference) / abs(reference))


def summarize(values: Sequence[float]) -> dict[str, float]:
    """Mean/min/max/std summary used by the benchmark renderer."""
    data = np.asarray(list(values), dtype=float)
    if data.size == 0:
        raise ValueError("values must not be empty")
    return {
        "mean": float(data.mean()),
        "min": float(data.min()),
        "max": float(data.max()),
        "std": float(data.std()),
    }


def _as_density(state: np.ndarray) -> np.ndarray:
    matrix = np.asarray(state, dtype=complex)
    if matrix.ndim == 1:
        return np.outer(matrix, matrix.conj())
    if matrix.ndim == 2 and matrix.shape[0] == matrix.shape[1]:
        return matrix
    raise ValueError("Expected a statevector or a square density matrix")
