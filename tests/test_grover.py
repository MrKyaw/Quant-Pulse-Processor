"""Grover search: iteration count, operator algebra and search success."""

from __future__ import annotations

import numpy as np
import pytest

from quantpulse.algorithms.grover import (
    diffusion_operator,
    grover_circuit,
    optimal_iterations,
    phase_oracle,
    run_grover,
)


@pytest.mark.parametrize(
    "n_qubits,expected",
    [(3, 2), (4, 3), (5, 4), (6, 6), (7, 8)],
)
def test_optimal_iterations(n_qubits: int, expected: int) -> None:
    assert optimal_iterations(n_qubits) == expected


def test_optimal_iterations_two_marked_items() -> None:
    assert optimal_iterations(4, 2) == 2


def test_optimal_iterations_validation() -> None:
    with pytest.raises(ValueError):
        optimal_iterations(3, 0)
    with pytest.raises(ValueError):
        optimal_iterations(3, 8)


def test_phase_oracle_is_diagonal_and_flips_marked() -> None:
    oracle = phase_oracle(3, [2, 5])
    diagonal = np.diag(oracle)
    assert np.allclose(diagonal[[2, 5]], [-1, -1])
    assert np.allclose(np.delete(diagonal, [2, 5]), 1)
    assert np.allclose(oracle.conj().T @ oracle, np.eye(8))


def test_phase_oracle_validation() -> None:
    with pytest.raises(ValueError):
        phase_oracle(2, [])
    with pytest.raises(ValueError):
        phase_oracle(2, [4])


def test_diffusion_is_a_reflection_about_the_uniform_state() -> None:
    n_qubits = 3
    diffusion = diffusion_operator(n_qubits)
    uniform = np.ones(2**n_qubits) / np.sqrt(2**n_qubits)
    assert np.allclose(diffusion @ uniform, uniform)
    assert np.allclose(diffusion.conj().T, diffusion)
    assert np.allclose(diffusion @ diffusion, np.eye(2**n_qubits))


def test_circuit_structure() -> None:
    circuit = grover_circuit(4, [9], iterations=2)
    assert len(circuit) == 4 + 4
    assert circuit.gate_counts() == {"diffusion": 2, "h": 4, "oracle": 2}


def test_circuit_rejects_negative_iterations() -> None:
    with pytest.raises(ValueError):
        grover_circuit(3, [1], iterations=-1)


@pytest.mark.parametrize("n_qubits", [3, 4, 5])
def test_grover_finds_the_marked_state(n_qubits: int) -> None:
    marked = 2**n_qubits - 2
    result = run_grover(n_qubits, [marked], shots=1024, seed=0)
    assert result.found
    assert result.success_probability > 0.94
    assert result.top_bitstring == format(marked, f"0{n_qubits}b")


def test_grover_with_two_marked_items() -> None:
    result = run_grover(4, [3, 12], shots=2048, seed=1)
    assert result.iterations == 2
    assert result.success_probability > 0.9
    assert int(result.top_bitstring, 2) in {3, 12}


def test_zero_iterations_returns_uniform_distribution() -> None:
    result = run_grover(4, [7], iterations=0)
    assert np.isclose(result.success_probability, 1 / 16)


def test_counts_sum_to_shots() -> None:
    result = run_grover(3, [5], shots=300, seed=2)
    assert sum(result.counts.values()) == 300
