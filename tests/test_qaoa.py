"""QAOA circuits, MaxCut references and optimisation quality."""

from __future__ import annotations

import numpy as np
import pytest

from quantpulse.algorithms.qaoa import (
    RING6_EDGES,
    exact_maxcut,
    expected_cut,
    optimize_qaoa,
    qaoa_circuit,
    ring_edges,
)


def test_ring_edges() -> None:
    assert ring_edges(4) == ((0, 1), (1, 2), (2, 3), (3, 0))


def test_ring_edges_with_chords_deduplicates() -> None:
    edges = ring_edges(4, chords=((0, 2), (0, 1)))
    assert (0, 2) in edges
    assert edges.count((0, 1)) == 1
    assert len(edges) == 5


def test_ring_edges_validation() -> None:
    with pytest.raises(ValueError):
        ring_edges(2)
    with pytest.raises(ValueError):
        ring_edges(4, chords=((1, 1),))
    with pytest.raises(ValueError):
        ring_edges(4, chords=((0, 9),))


def test_qaoa_circuit_structure() -> None:
    circuit = qaoa_circuit(3, [(0, 1), (1, 2)], [0.5], [0.25])
    # Three Hadamards, two CNOT-RZ-CNOT sandwiches, three mixers.
    assert len(circuit) == 3 + 2 * 3 + 3
    assert circuit.gate_counts()["cnot"] == 4
    assert circuit.gate_counts()["rz"] == 2
    assert circuit.gate_counts()["rx"] == 3


def test_qaoa_circuit_validation() -> None:
    with pytest.raises(ValueError):
        qaoa_circuit(3, [(0, 1)], [0.5], [])
    with pytest.raises(ValueError):
        qaoa_circuit(3, [(0, 1)], [], [])
    with pytest.raises(ValueError):
        qaoa_circuit(3, [(0, 9)], [0.5], [0.5])
    with pytest.raises(ValueError):
        qaoa_circuit(3, [(0, 0)], [0.5], [0.5])


def test_uniform_distribution_cuts_half_the_edges() -> None:
    edges = [(0, 1), (1, 2), (2, 3)]
    assert np.isclose(expected_cut(np.full(16, 1 / 16), edges), 1.5)


def test_expected_cut_of_known_distribution() -> None:
    probs = np.zeros(4)
    probs[1] = 1.0  # bitstring 01: the single edge is cut
    assert np.isclose(expected_cut(probs, [(0, 1)]), 1.0)
    probs = np.zeros(4)
    probs[0] = 1.0
    assert np.isclose(expected_cut(probs, [(0, 1)]), 0.0)


def test_exact_maxcut_triangle() -> None:
    edges = [(0, 1), (1, 2), (2, 0)]
    assert exact_maxcut(3, edges) == 2


def test_exact_maxcut_bipartite_ring() -> None:
    assert exact_maxcut(6, ring_edges(6)) == 6


def test_ring6_optimum_is_not_all_edges() -> None:
    assert len(RING6_EDGES) == 8
    assert exact_maxcut(6, RING6_EDGES) == 8  # the chords respect the bipartition


def test_qaoa_state_is_normalized() -> None:
    circuit = qaoa_circuit(4, ring_edges(4), [0.3, 0.7], [0.9, 0.2])
    assert np.isclose(np.linalg.norm(circuit.run()), 1.0)


def test_optimization_improves_with_depth() -> None:
    shallow = optimize_qaoa(6, RING6_EDGES, depth=1, steps=60, restarts=2, seed=3)
    deeper = optimize_qaoa(6, RING6_EDGES, depth=2, steps=60, restarts=2, seed=3)
    assert shallow.ratio >= 0.65
    assert deeper.ratio >= 0.80
    assert deeper.ratio >= shallow.ratio


def test_optimization_reports_consistent_parameters() -> None:
    result = optimize_qaoa(5, ring_edges(5), depth=1, steps=40, restarts=1, seed=0)
    assert len(result.gammas) == len(result.betas) == 1
    assert result.exact_cut == exact_maxcut(5, ring_edges(5))
    assert 0.0 <= result.ratio <= 1.0
    assert len(result.history) == 40


def test_optimization_validation() -> None:
    with pytest.raises(ValueError):
        optimize_qaoa(4, ring_edges(4), depth=0)
    with pytest.raises(ValueError):
        optimize_qaoa(4, ring_edges(4), restarts=0)
