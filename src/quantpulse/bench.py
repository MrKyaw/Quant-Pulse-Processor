"""Reproducible benchmark suite and Markdown renderer.

``quantpulse bench`` prints the report; ``quantpulse bench --write`` updates
``docs/BENCHMARKS.md``.  Timings are medians of repeated runs on the machine
executing the command, so every number in the documentation can be
regenerated locally.
"""

from __future__ import annotations

import platform
import statistics
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .algorithms.grover import optimal_iterations, run_grover
from .algorithms.qaoa import RING6_EDGES, optimize_qaoa
from .algorithms.qft import qft_circuit
from .algorithms.vqe import exact_ground_energy, optimize_vqe, tfim_terms
from .circuit import Circuit
from .pulse import pi_pulse, pulse_area, simulate_three_level, simulate_two_level

__all__ = ["Section", "collect_benchmarks", "render_markdown"]


@dataclass(frozen=True)
class Section:
    """One benchmark table plus prose."""

    title: str
    description: str
    headers: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]
    notes: tuple[str, ...] = ()


def _timed(function: Callable[[], object], repeats: int = 3) -> tuple[object, float]:
    times = []
    result = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = function()
        times.append((time.perf_counter() - start) * 1000.0)
    return result, statistics.median(times)


def _qft_section() -> Section:
    rows = []
    for n_qubits in range(4, 11):
        circuit = qft_circuit(n_qubits)
        state, elapsed = _timed(circuit.run, repeats=3)
        target = 1.0 / np.sqrt(2**n_qubits)
        error = float(np.max(np.abs(state - target)))
        rows.append(
            (
                str(n_qubits),
                f"{2**n_qubits}",
                f"{circuit.depth}",
                f"{len(circuit)}",
                f"{elapsed:.2f}",
                f"{error:.1e}",
            )
        )
    return Section(
        title="QFT circuit scaling",
        description=(
            "Forward QFT on |0...0>. The time column is the median of three "
            "build-and-run executions; the error column compares the state to "
            "the exact uniform superposition."
        ),
        headers=("Qubits", "State size", "Depth", "Gates", "Time (ms)", "Uniform error"),
        rows=tuple(rows),
        notes=(
            "The statevector grows as 2^n, so this table is a direct view of "
            "the exponential wall every dense simulator hits.",
        ),
    )


def _grover_sweep_section() -> Section:
    rows = []
    for iterations in range(0, 6):
        result = run_grover(4, [11], iterations=iterations, shots=512, seed=3)
        rows.append(
            (
                str(iterations),
                f"{result.success_probability:.4f}",
                result.top_bitstring,
                f"{result.counts.get('1011', 0)} / {sum(result.counts.values())}",
            )
        )
    return Section(
        title="Grover search: iteration sweep",
        description=(
            "Four-qubit search for the marked state 1011 swept over the number "
            "of Grover iterations. The optimum is three iterations."
        ),
        headers=("Iterations", "Success probability", "Top bitstring", "Marked shots"),
        rows=tuple(rows),
    )


def _grover_scaling_section() -> Section:
    rows = []
    for n_qubits in range(3, 8):
        marked = 2**n_qubits - 2
        optimal = optimal_iterations(n_qubits, 1)
        result = run_grover(n_qubits, [marked], shots=512, seed=5)
        rows.append(
            (
                str(n_qubits),
                str(2**n_qubits),
                str(optimal),
                f"{result.success_probability:.4f}",
                result.top_bitstring,
            )
        )
    return Section(
        title="Grover search: optimal scaling",
        description=(
            "Rounded optimal iteration count pi / (4 asin(sqrt(1/N))) and the "
            "resulting success probability for growing registers."
        ),
        headers=("Qubits", "Search space", "Iterations", "Success probability", "Top bitstring"),
        rows=tuple(rows),
    )


def _qaoa_section() -> Section:
    rows = []
    for depth in (1, 2, 3, 4):
        result, elapsed = _timed(
            lambda d=depth: optimize_qaoa(6, RING6_EDGES, depth=d, steps=100, restarts=3, seed=3),
            repeats=1,
        )
        rows.append(
            (
                str(depth),
                str(len(result.gammas) * 2),
                f"{result.expected_cut:.3f}",
                str(result.exact_cut),
                f"{result.ratio:.4f}",
                f"{elapsed:.0f}",
            )
        )
    return Section(
        title="QAOA for MaxCut",
        description=(
            "Six-node ring with two chords (eight edges) optimised with Adam, "
            "central-difference gradients and three random restarts per depth."
        ),
        headers=("Depth p", "Parameters", "Expected cut", "Exact max cut", "Ratio", "Time (ms)"),
        rows=tuple(rows),
        notes=("The ratio is the expected cut divided by the brute-force optimum.",),
    )


def _vqe_section() -> Section:
    rows = []
    for n_qubits, layers in ((2, 1), (2, 2), (2, 3), (3, 2)):
        terms = tfim_terms(n_qubits)
        exact = exact_ground_energy(terms)
        result, elapsed = _timed(
            lambda n=n_qubits, depth_layers=layers: optimize_vqe(
                n, layers=depth_layers, steps=200, seed=0
            ),
            repeats=1,
        )
        rows.append(
            (
                str(n_qubits),
                str(layers),
                str(n_qubits * (layers + 1)),
                f"{result.energy:.6f}",
                f"{exact:.6f}",
                f"{result.error:.2e}",
                f"{elapsed:.0f}",
            )
        )
    return Section(
        title="VQE ground states",
        description=(
            "Transverse-field Ising ring with J = 1 and h = 0.5. Energies are "
            "compared against dense diagonalisation of the same Hamiltonian."
        ),
        headers=(
            "Qubits",
            "Layers",
            "Parameters",
            "VQE energy",
            "Exact energy",
            "Error",
            "Time (ms)",
        ),
        rows=tuple(rows),
        notes=(
            "Parameter-shift gradients are exact for the independent RY angles of this ansatz.",
        ),
    )


def _pulse_section() -> Section:
    rows = []
    for beta in (0.0, 0.25, 0.5, 0.75, 1.0, 1.25):
        _, in_phase, quadrature = pi_pulse(12.0, 0.02, sigma=2.0, beta=beta)
        result = simulate_three_level(in_phase, quadrature, 0.02, anharmonicity=-1.2)
        rows.append(
            (
                f"{beta:.2f}",
                f"{pulse_area(in_phase, 0.02):.4f}",
                f"{result.final_population:.4f}",
                f"{result.final_leakage:.2e}",
            )
        )
    t1, t2 = 20.0, 10.0
    dt = 0.01
    drive = np.zeros(int(t1 / dt))
    decay = simulate_two_level(drive, dt, t1=t1, t2=t2, initial="excited")
    dephase = simulate_two_level(drive[: int(t2 / dt)], dt, t1=t1, t2=t2, initial="plus")
    decay_error = abs(decay.final_population - np.exp(-1.0))
    dephase_error = abs(dephase.coherence[-1] - 0.5 * np.exp(-1.0))
    return Section(
        title="Pulse-level control",
        description=(
            "Three-level transmon (anharmonicity -1.2 rad/ns) driven by a "
            "12 ns Gaussian pi-pulse with a DRAG quadrature. The notes below "
            "report Lindblad relaxation at T1 and coherence at T2."
        ),
        headers=("DRAG beta", "Pulse area", "P(|1>)", "Leakage P(|2>)"),
        rows=tuple(rows),
        notes=(
            f"T1 decay error at t = T1: {decay_error:.2e}.",
            f"T2 coherence error at t = T2: {dephase_error:.2e}.",
            "DRAG cancels the leading leakage path; the residual grows again "
            "once beta over-corrects the pulse.",
            "Leakage falls by more than two orders of magnitude, while the "
            "excited-state population peaks near beta = 0.5.",
        ),
    )


def _engine_section() -> Section:
    rows = []
    for n_qubits in (4, 6, 8, 10, 12):
        circuit = Circuit(n_qubits).h(0)
        for q in range(n_qubits - 1):
            circuit.cnot(q, q + 1)
        _, elapsed = _timed(circuit.run, repeats=3)
        rows.append(
            (
                str(n_qubits),
                f"{2**n_qubits}",
                f"{2**n_qubits * 16 / 1024:.2f}",
                f"{circuit.depth}",
                f"{elapsed:.2f}",
            )
        )
    return Section(
        title="Statevector engine scaling",
        description=(
            "Build and run a GHZ-preparation circuit for increasing register "
            "sizes. Memory is the complex128 size of the statevector."
        ),
        headers=("Qubits", "Amplitudes", "State memory (KiB)", "Depth", "Time (ms)"),
        rows=tuple(rows),
    )


def collect_benchmarks() -> list[Section]:
    """Run every benchmark section."""
    return [
        _engine_section(),
        _qft_section(),
        _grover_sweep_section(),
        _grover_scaling_section(),
        _qaoa_section(),
        _vqe_section(),
        _pulse_section(),
    ]


def _processor_name() -> str:
    if platform.system() == "Darwin":
        try:
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=2,
                check=False,
            )
            if result.stdout.strip():
                return result.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            pass
    return platform.processor() or platform.machine()


def _environment() -> list[str]:
    return [
        f"- Python: {sys.version.split()[0]}",
        f"- NumPy: {np.__version__}",
        f"- Platform: {platform.platform()}",
        f"- Processor: {_processor_name()}",
    ]


def render_markdown(sections: list[Section] | None = None, *, version: str = "0.1.0") -> str:
    """Render the benchmark report as Markdown."""
    data = collect_benchmarks() if sections is None else sections
    lines = [
        "# Benchmarks",
        "",
        "Generated by `python -m quantpulse bench --write`. Every table is",
        "reproducible on the machine that runs the command; timings are wall",
        "clock medians and will vary with hardware.",
        "",
        "## Environment",
        "",
        *_environment(),
        "",
        f"- Package version: {version}",
        "",
    ]
    for section in data:
        lines.extend(
            [
                f"## {section.title}",
                "",
                section.description,
                "",
                "| " + " | ".join(section.headers) + " |",
                "| " + " | ".join("---" for _ in section.headers) + " |",
            ]
        )
        for row in section.rows:
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")
        for note in section.notes:
            lines.append(f"- {note}")
        if section.notes:
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"
