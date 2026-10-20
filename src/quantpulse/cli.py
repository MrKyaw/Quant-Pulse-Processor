"""Command-line interface: ``quantpulse demo | eval | bench | info``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from . import __version__
from .acceptance import run_acceptance
from .algorithms.deutsch_jozsa import deutsch_jozsa
from .algorithms.grover import run_grover
from .algorithms.qaoa import RING6_EDGES, optimize_qaoa
from .algorithms.qft import qft_circuit
from .algorithms.vqe import optimize_vqe
from .bench import collect_benchmarks, render_markdown
from .circuit import Circuit
from .pulse import pi_pulse, simulate_three_level, simulate_two_level

__all__ = ["build_parser", "main"]

_REPO_ROOT = Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="quantpulse",
        description="Quantum algorithms and pulse-level control, in pure NumPy.",
    )
    parser.add_argument("--version", action="version", version=f"quantpulse {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo = subparsers.add_parser("demo", help="Run a short end-to-end demonstration.")
    demo.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")

    evaluate = subparsers.add_parser("eval", help="Run the acceptance suite.")
    evaluate.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")

    bench = subparsers.add_parser("bench", help="Run benchmarks and render Markdown.")
    bench.add_argument(
        "--write",
        nargs="?",
        const="docs/BENCHMARKS.md",
        default=None,
        metavar="PATH",
        help="Write the report to PATH (default: docs/BENCHMARKS.md).",
    )
    bench.add_argument("--json", action="store_true", help="Emit raw section data as JSON.")

    info = subparsers.add_parser("info", help="Show version and environment details.")
    info.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    return parser


def _table(headers: list[str], rows: list[list[str]], indent: str = "  ") -> str:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    lines = [indent + "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers))]
    lines.append(indent + "  ".join("-" * width for width in widths))
    for row in rows:
        lines.append(indent + "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)))
    return "\n".join(lines)


def _demo() -> dict[str, object]:
    bell = Circuit(2).h(0).cnot(0, 1)
    bell_probs = bell.probabilities()
    ghz = Circuit(3).h(0).cnot(0, 1).cnot(1, 2)
    ghz_probs = ghz.probabilities()
    qft_state = qft_circuit(4).run()
    qft_error = float(np.max(np.abs(qft_state - 1 / np.sqrt(16))))

    grover = run_grover(4, [11], shots=512, seed=1)
    constant = deutsch_jozsa(lambda _: 1, 4)
    balanced = deutsch_jozsa(lambda x: bin(x).count("1") % 2, 4)
    qaoa = optimize_qaoa(6, RING6_EDGES, depth=2, steps=60, restarts=2, seed=3)
    vqe = optimize_vqe(2, layers=2, steps=150, seed=0)

    _, bare_i, bare_q = pi_pulse(12.0, 0.02, sigma=2.0)
    bare = simulate_three_level(bare_i, bare_q, 0.02, anharmonicity=-1.2)
    _, drag_i, drag_q = pi_pulse(12.0, 0.02, sigma=2.0, beta=0.75)
    drag = simulate_three_level(drag_i, drag_q, 0.02, anharmonicity=-1.2)
    _, pi_in_phase, _ = pi_pulse(24.0, 0.05)
    pi_result = simulate_two_level(pi_in_phase, 0.05)

    return {
        "circuits": {
            "bell": {
                "p00": float(bell_probs[0]),
                "p11": float(bell_probs[3]),
                "depth": bell.depth,
            },
            "ghz": {
                "p000": float(ghz_probs[0]),
                "p111": float(ghz_probs[7]),
                "depth": ghz.depth,
            },
            "qft4_uniform_error": qft_error,
        },
        "algorithms": {
            "grover": {
                "qubits": grover.n_qubits,
                "iterations": grover.iterations,
                "success_probability": grover.success_probability,
                "top_bitstring": grover.top_bitstring,
            },
            "deutsch_jozsa": {
                "constant_oracle": constant.is_constant,
                "balanced_oracle": balanced.is_constant,
            },
            "qaoa": {
                "depth": qaoa.depth,
                "expected_cut": qaoa.expected_cut,
                "exact_cut": qaoa.exact_cut,
                "ratio": qaoa.ratio,
            },
            "vqe": {
                "energy": vqe.energy,
                "exact_energy": vqe.exact_energy,
                "error": vqe.error,
            },
        },
        "pulse": {
            "pi_pulse_population": pi_result.final_population,
            "bare_leakage": bare.final_leakage,
            "drag_leakage": drag.final_leakage,
            "drag_reduction": bare.final_leakage / max(drag.final_leakage, 1e-15),
        },
    }


def _print_demo(data: dict[str, object]) -> None:
    circuits = data["circuits"]  # type: ignore[assignment]
    algorithms = data["algorithms"]  # type: ignore[assignment]
    pulse = data["pulse"]  # type: ignore[assignment]
    print(f"Quant-Pulse-Processor {__version__}")
    print()
    print("Circuit engine")
    print(
        f"  Bell state     P(00) = {circuits['bell']['p00']:.4f}   "
        f"P(11) = {circuits['bell']['p11']:.4f}   depth = {circuits['bell']['depth']}"
    )
    print(
        f"  GHZ-3 state    P(000) = {circuits['ghz']['p000']:.4f}   "
        f"P(111) = {circuits['ghz']['p111']:.4f}   depth = {circuits['ghz']['depth']}"
    )
    print(f"  QFT-4 |0>      uniform-amplitude error = {circuits['qft4_uniform_error']:.1e}")
    print()
    print("Algorithms")
    grover = algorithms["grover"]
    print(
        f"  Grover         n = {grover['qubits']}, iterations = {grover['iterations']}, "
        f"P(marked) = {grover['success_probability']:.4f}, top = {grover['top_bitstring']}"
    )
    dj = algorithms["deutsch_jozsa"]
    print(
        f"  Deutsch-Jozsa  constant -> constant ({dj['constant_oracle']}), "
        f"balanced -> constant ({dj['balanced_oracle']})"
    )
    qaoa = algorithms["qaoa"]
    print(
        f"  QAOA           p = {qaoa['depth']}, cut = {qaoa['expected_cut']:.3f} / "
        f"{qaoa['exact_cut']}, ratio = {qaoa['ratio']:.4f}"
    )
    vqe = algorithms["vqe"]
    print(
        f"  VQE            E = {vqe['energy']:.6f}, exact = {vqe['exact_energy']:.6f}, "
        f"error = {vqe['error']:.1e}"
    )
    print()
    print("Pulse control")
    print(f"  Rabi pi-pulse  P(|1>) = {pulse['pi_pulse_population']:.6f}")
    print(
        f"  DRAG           leakage {pulse['bare_leakage']:.2e} -> "
        f"{pulse['drag_leakage']:.2e} ({pulse['drag_reduction']:.0f}x lower)"
    )
    print()
    print("Run `quantpulse eval` for the full acceptance suite.")


def _evaluate(json_output: bool) -> int:
    results = run_acceptance()
    passed = sum(1 for r in results if r.passed)
    total = len(results)
    payload = {
        "passed": passed,
        "total": total,
        "ok": passed == total,
        "cases": [
            {
                "name": r.name,
                "category": r.category,
                "passed": r.passed,
                "measured": r.measured,
                "detail": r.detail,
                "duration_ms": r.duration_ms,
            }
            for r in results
        ],
    }
    if json_output:
        print(json.dumps(payload, indent=2))
        return 0 if payload["ok"] else 1
    print(f"Quant-Pulse-Processor {__version__} acceptance suite")
    print()
    rows = [
        [
            r.category,
            r.name,
            "PASS" if r.passed else "FAIL",
            r.detail,
            f"{r.duration_ms:.1f} ms",
        ]
        for r in results
    ]
    print(_table(["Scope", "Case", "Status", "Measurement", "Time"], rows))
    print()
    print(f"{passed}/{total} cases passed")
    return 0 if passed == total else 1


def _bench(output: str | None, json_output: bool) -> int:
    sections = collect_benchmarks()
    if json_output:
        payload = [
            {
                "title": section.title,
                "description": section.description,
                "headers": list(section.headers),
                "rows": [list(row) for row in section.rows],
                "notes": list(section.notes),
            }
            for section in sections
        ]
        print(json.dumps(payload, indent=2))
        return 0
    report = render_markdown(sections, version=__version__)
    if output is not None:
        path = Path(output)
        if not path.is_absolute():
            path = _REPO_ROOT / path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(report, encoding="utf-8")
        print(f"Wrote {path}")
        return 0
    print(report)
    return 0


def _info(json_output: bool) -> int:
    import platform

    payload = {
        "version": __version__,
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
    }
    if json_output:
        print(json.dumps(payload, indent=2))
        return 0
    for key, value in payload.items():
        print(f"{key:>10}: {value}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "demo":
        data = _demo()
        if args.json:
            print(json.dumps(data, indent=2))
        else:
            _print_demo(data)
        return 0
    if args.command == "eval":
        return _evaluate(args.json)
    if args.command == "bench":
        return _bench(args.write, args.json)
    if args.command == "info":
        return _info(args.json)
    parser.error(f"Unknown command {args.command!r}")
    return 2


if __name__ == "__main__":  # pragma: no cover - module entry point
    raise SystemExit(main())
