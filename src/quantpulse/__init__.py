"""Quant-Pulse-Processor: a dependency-light quantum computing toolkit.

The package has three layers:

``quantpulse.circuit`` / ``quantpulse.gates``
    A NumPy statevector engine with dense, inspectable unitaries.
``quantpulse.algorithms``
    Canonical algorithms (QFT, Grover, Deutsch-Jozsa, QAOA, VQE) built on the
    circuit layer and scored against exact references.
``quantpulse.pulse``
    Pulse-level control of a transmon: Rabi dynamics, Lindblad ``T1``/``T2``
    relaxation, three-level leakage and DRAG suppression.

Everything is pure Python on top of NumPy so results are deterministic and
the full stack can be audited in one sitting.
"""

from __future__ import annotations

from .circuit import Circuit
from .gates import CNOT, CZ, SWAP, TOFFOLI, H, X, Y, Z
from .metrics import (
    approximation_ratio,
    cut_value,
    relative_error,
    state_fidelity,
    success_probability,
    total_variation_distance,
    trace_distance,
)
from .pulse import (
    PulseResult,
    drag_envelopes,
    gaussian_envelope,
    pi_pulse,
    simulate_three_level,
    simulate_two_level,
)
from .statevector import (
    basis_state,
    bitstring,
    expectation_pauli,
    fidelity,
    partial_trace,
    probabilities,
    sample_counts,
    von_neumann_entropy,
)

__version__ = "0.1.0"

__all__ = [
    "CNOT",
    "CZ",
    "SWAP",
    "TOFFOLI",
    "Circuit",
    "H",
    "PulseResult",
    "X",
    "Y",
    "Z",
    "__version__",
    "approximation_ratio",
    "basis_state",
    "bitstring",
    "cut_value",
    "drag_envelopes",
    "expectation_pauli",
    "fidelity",
    "gaussian_envelope",
    "partial_trace",
    "pi_pulse",
    "probabilities",
    "relative_error",
    "sample_counts",
    "simulate_three_level",
    "simulate_two_level",
    "state_fidelity",
    "success_probability",
    "total_variation_distance",
    "trace_distance",
    "von_neumann_entropy",
]
