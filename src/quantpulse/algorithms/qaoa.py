"""QAOA for MaxCut on small graphs.

The cost layer implements ``exp(-i gamma H_C)`` with
``H_C = sum_(u,v) (1 - Z_u Z_v) / 2`` as a CNOT-RZ-CNOT sandwich per edge, and
the mixer is ``exp(-i beta sum_q X_q) = prod_q RX(2 beta)``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ..circuit import Circuit
from ..metrics import approximation_ratio, cut_value
from ..optim import adam_minimize

__all__ = [
    "RING6_EDGES",
    "QAOAResult",
    "exact_maxcut",
    "expected_cut",
    "optimize_qaoa",
    "qaoa_circuit",
    "ring_edges",
]


@dataclass(frozen=True)
class QAOAResult:
    """Best parameters and cut quality found for one graph and depth."""

    n_qubits: int
    edges: tuple[tuple[int, int], ...]
    depth: int
    gammas: tuple[float, ...]
    betas: tuple[float, ...]
    expected_cut: float
    exact_cut: int
    ratio: float
    history: tuple[float, ...]


def ring_edges(
    n_qubits: int,
    chords: Sequence[tuple[int, int]] = (),
) -> tuple[tuple[int, int], ...]:
    """Ring graph plus optional chord edges."""
    if n_qubits < 3:
        raise ValueError("A ring graph needs at least three nodes")
    edges = [(q, (q + 1) % n_qubits) for q in range(n_qubits)]
    for u, v in chords:
        if u == v:
            raise ValueError("Self-loops are not allowed")
        if not (0 <= u < n_qubits and 0 <= v < n_qubits):
            raise ValueError("Chord endpoint out of range")
        edge = (min(u, v), max(u, v))
        if edge not in edges:
            edges.append(edge)
    return tuple(edges)


#: Six-node ring with two chords: eight edges and odd cycles, so the optimal
#: cut is strictly smaller than the edge count.
RING6_EDGES = ring_edges(6, chords=((0, 3), (1, 4)))


def _validate_edges(edges: Sequence[tuple[int, int]], n_qubits: int) -> tuple[tuple[int, int], ...]:
    if not edges:
        raise ValueError("edges must not be empty")
    out = []
    for u, v in edges:
        if u == v:
            raise ValueError("Self-loops are not allowed")
        if not (0 <= u < n_qubits and 0 <= v < n_qubits):
            raise ValueError("Edge endpoint out of range")
        out.append((int(u), int(v)))
    return tuple(out)


def qaoa_circuit(
    n_qubits: int,
    edges: Sequence[tuple[int, int]],
    gammas: Sequence[float],
    betas: Sequence[float],
) -> Circuit:
    """Build a depth-``p`` QAOA circuit for the MaxCut cost Hamiltonian."""
    if len(gammas) != len(betas):
        raise ValueError("gammas and betas must have the same length")
    if len(gammas) == 0:
        raise ValueError("At least one QAOA layer is required")
    edge_list = _validate_edges(edges, n_qubits)
    circuit = Circuit(n_qubits)
    for q in range(n_qubits):
        circuit.h(q)
    for gamma, beta in zip(gammas, betas):
        for u, v in edge_list:
            circuit.cnot(u, v)
            circuit.rz(v, -float(gamma))
            circuit.cnot(u, v)
        for q in range(n_qubits):
            circuit.rx(q, 2.0 * float(beta))
    return circuit


def expected_cut(probabilities: Sequence[float], edges: Sequence[tuple[int, int]]) -> float:
    """Expected cut value under a computational-basis distribution."""
    probs = np.asarray(probabilities, dtype=float)
    size = probs.size
    indices = np.arange(size)
    values = np.zeros(size, dtype=float)
    for u, v in edges:
        values += ((indices >> u) & 1) != ((indices >> v) & 1)
    return float(probs @ values)


def exact_maxcut(n_qubits: int, edges: Sequence[tuple[int, int]]) -> int:
    """Brute-force maximum cut for reference."""
    edge_list = _validate_edges(edges, n_qubits)
    return max(cut_value(index, edge_list) for index in range(2**n_qubits))


def optimize_qaoa(
    n_qubits: int,
    edges: Sequence[tuple[int, int]] = RING6_EDGES,
    depth: int = 2,
    *,
    steps: int = 100,
    learning_rate: float = 0.15,
    restarts: int = 3,
    seed: int = 0,
) -> QAOAResult:
    """Optimise QAOA angles with central-difference gradients and Adam.

    Layer angles are shared by every wire, so the single-gate parameter-shift
    identity does not apply directly; a central difference with ``1e-3`` is
    used instead.  Several random restarts guard against local minima.
    """
    if depth < 1:
        raise ValueError("depth must be at least 1")
    if restarts < 1:
        raise ValueError("restarts must be at least 1")
    edge_list = _validate_edges(edges, n_qubits)
    optimum = exact_maxcut(n_qubits, edge_list)
    rng = np.random.default_rng(seed)

    def objective(params: np.ndarray) -> float:
        gammas = params[:depth]
        betas = params[depth:]
        circuit = qaoa_circuit(n_qubits, edge_list, gammas, betas)
        return -expected_cut(circuit.probabilities(), edge_list)

    def gradient(params: np.ndarray, epsilon: float = 1e-3) -> np.ndarray:
        grad = np.zeros_like(params)
        for i in range(params.size):
            plus = params.copy()
            minus = params.copy()
            plus[i] += epsilon
            minus[i] -= epsilon
            grad[i] = (objective(plus) - objective(minus)) / (2.0 * epsilon)
        return grad

    best_params = np.zeros(2 * depth)
    best_value = float("inf")
    best_history: list[float] = []
    for _ in range(restarts):
        x0 = rng.uniform(0.0, np.pi, size=2 * depth)
        params, history = adam_minimize(
            objective,
            gradient,
            x0,
            steps=steps,
            learning_rate=learning_rate,
        )
        value = objective(params)
        if value < best_value:
            best_value = value
            best_params = params
            best_history = history
    gammas = tuple(float(x) for x in best_params[:depth])
    betas = tuple(float(x) for x in best_params[depth:])
    circuit = qaoa_circuit(n_qubits, edge_list, gammas, betas)
    value = expected_cut(circuit.probabilities(), edge_list)
    return QAOAResult(
        n_qubits=n_qubits,
        edges=edge_list,
        depth=depth,
        gammas=gammas,
        betas=betas,
        expected_cut=value,
        exact_cut=optimum,
        ratio=approximation_ratio(value, optimum),
        history=tuple(-v for v in best_history),
    )
