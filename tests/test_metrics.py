"""Metrics used by the algorithms and the benchmark report."""

from __future__ import annotations

import numpy as np
import pytest

from quantpulse import metrics
from quantpulse.statevector import basis_state


def test_state_fidelity_matches_inner_product() -> None:
    a = np.array([1, 0], dtype=complex)
    b = np.array([1, 1], dtype=complex) / np.sqrt(2)
    assert np.isclose(metrics.state_fidelity(a, b), 0.5)


def test_trace_distance_orthogonal_states() -> None:
    assert np.isclose(metrics.trace_distance(basis_state(1, 0), basis_state(1, 1)), 1.0)


def test_trace_distance_identical_states() -> None:
    state = basis_state(2, 3)
    assert np.isclose(metrics.trace_distance(state, state), 0.0)


def test_trace_distance_accepts_density_matrices() -> None:
    left = np.diag([1.0, 0.0])
    right = np.diag([0.0, 1.0])
    assert np.isclose(metrics.trace_distance(left, right), 1.0)


def test_trace_distance_shape_validation() -> None:
    with pytest.raises(ValueError):
        metrics.trace_distance(np.eye(2), np.eye(4))
    with pytest.raises(ValueError):
        metrics.trace_distance(np.zeros((2, 3)), np.eye(2))


def test_total_variation_distance_dicts() -> None:
    left = {"00": 0.5, "11": 0.5}
    right = {"00": 0.25, "11": 0.75}
    assert np.isclose(metrics.total_variation_distance(left, right), 0.25)


def test_total_variation_distance_sequences() -> None:
    assert np.isclose(
        metrics.total_variation_distance([1.0, 0.0], [0.0, 1.0]),
        1.0,
    )


def test_total_variation_distance_length_validation() -> None:
    with pytest.raises(ValueError):
        metrics.total_variation_distance([1.0, 0.0], [1.0])


def test_success_probability() -> None:
    state = np.array([0, np.sqrt(0.3), 0, np.sqrt(0.7)], dtype=complex)
    assert np.isclose(metrics.success_probability(state, [1, 3]), 1.0)
    assert np.isclose(metrics.success_probability(state, [1]), 0.3)


def test_success_probability_validation() -> None:
    state = basis_state(2, 0)
    with pytest.raises(ValueError):
        metrics.success_probability(state, [])
    with pytest.raises(ValueError):
        metrics.success_probability(state, [4])


def test_cut_value() -> None:
    edges = [(0, 1), (1, 2)]
    assert metrics.cut_value(0b01, edges) == 1
    assert metrics.cut_value(0b10, edges) == 2
    assert metrics.cut_value(0b00, edges) == 0
    assert metrics.cut_value(0b11, edges) == 1


def test_approximation_ratio_clips() -> None:
    assert np.isclose(metrics.approximation_ratio(6.0, 8), 0.75)
    assert np.isclose(metrics.approximation_ratio(9.0, 8), 1.0)
    assert np.isclose(metrics.approximation_ratio(-1.0, 8), 0.0)
    with pytest.raises(ValueError):
        metrics.approximation_ratio(1.0, 0)


def test_relative_error() -> None:
    assert np.isclose(metrics.relative_error(1.1, 1.0), 0.1)
    assert np.isclose(metrics.relative_error(0.3, 0.0), 0.3)


def test_summarize() -> None:
    summary = metrics.summarize([1.0, 2.0, 3.0])
    assert summary["mean"] == pytest.approx(2.0)
    assert summary["min"] == 1.0
    assert summary["max"] == 3.0
    assert summary["std"] == pytest.approx(np.std([1, 2, 3]))
    with pytest.raises(ValueError):
        metrics.summarize([])
