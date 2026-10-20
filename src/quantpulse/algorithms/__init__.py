"""Canonical quantum algorithms built on the statevector engine."""

from __future__ import annotations

from .deutsch_jozsa import DeutschJozsaResult, deutsch_jozsa
from .grover import GroverResult, grover_circuit, optimal_iterations, run_grover
from .qaoa import QAOAResult, exact_maxcut, optimize_qaoa, qaoa_circuit, ring_edges
from .qft import apply_qft, qft_circuit, qft_unitary
from .vqe import VQEResult, optimize_vqe

__all__ = [
    "DeutschJozsaResult",
    "GroverResult",
    "QAOAResult",
    "VQEResult",
    "apply_qft",
    "deutsch_jozsa",
    "exact_maxcut",
    "grover_circuit",
    "optimal_iterations",
    "optimize_qaoa",
    "optimize_vqe",
    "qaoa_circuit",
    "qft_circuit",
    "qft_unitary",
    "ring_edges",
    "run_grover",
]
