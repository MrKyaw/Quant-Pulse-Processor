"""A small, explicit circuit model on top of the statevector engine.

Circuits are immutable descriptions (a list of operations) that are executed
when ``run`` is called.  Every operation carries the dense unitary that acts
on its wires, so already-built circuits can be inverted, inspected and
re-executed without a compiler in the loop.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from . import gates
from .statevector import basis_state, probabilities, sample_counts

__all__ = ["Circuit", "Operation"]


@dataclass(frozen=True)
class Operation:
    """A single gate applied to ``qubits`` (in matrix-axis order)."""

    name: str
    qubits: tuple[int, ...]
    matrix: np.ndarray
    params: tuple[float, ...] = ()
    global_op: bool = False

    @property
    def width(self) -> int:
        return len(self.qubits)


class Circuit:
    """Ordered sequence of unitary operations on ``n_qubits`` wires."""

    def __init__(self, n_qubits: int) -> None:
        if n_qubits < 1:
            raise ValueError("n_qubits must be at least 1")
        self.n_qubits = n_qubits
        self._ops: list[Operation] = []

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    def _append(
        self,
        name: str,
        matrix: np.ndarray,
        qubits: Sequence[int],
        params: tuple[float, ...] = (),
    ) -> Circuit:
        wires = tuple(int(q) for q in qubits)
        if len(set(wires)) != len(wires):
            raise ValueError("A gate cannot act twice on the same qubit")
        if any(q < 0 or q >= self.n_qubits for q in wires):
            raise ValueError(f"Qubit index out of range for {self.n_qubits} qubits")
        matrix = np.asarray(matrix, dtype=complex)
        expected = 2 ** len(wires)
        if matrix.shape != (expected, expected):
            raise ValueError(f"{name} must be a {expected}x{expected} matrix")
        if not gates.is_unitary(matrix):
            raise ValueError(f"{name} matrix is not unitary")
        self._ops.append(Operation(name, wires, matrix, params))
        return self

    def add_unitary(
        self,
        matrix: np.ndarray,
        qubits: Sequence[int],
        name: str = "unitary",
    ) -> Circuit:
        """Append a unitary acting on ``qubits``.

        The matrix is ordered so that ``qubits[0]`` is the most significant
        bit of its index and ``qubits[-1]`` the least significant one.
        """
        return self._append(name, matrix, qubits)

    def add_global_unitary(self, matrix: np.ndarray, name: str = "unitary") -> Circuit:
        """Append a ``2**n x 2**n`` unitary expressed in the state basis.

        Unlike :meth:`add_unitary`, the matrix is applied directly to the
        little-endian statevector, so ``run`` returns ``matrix @ state``.
        """
        matrix = np.asarray(matrix, dtype=complex)
        size = 2**self.n_qubits
        if matrix.shape != (size, size):
            raise ValueError(f"Expected a {size}x{size} matrix")
        if not gates.is_unitary(matrix):
            raise ValueError("Matrix is not unitary")
        self._ops.append(Operation(name, tuple(range(self.n_qubits)), matrix, (), global_op=True))
        return self

    # Single-qubit gates -------------------------------------------------
    def _one(
        self,
        name: str,
        matrix: np.ndarray,
        q: int,
        params: tuple[float, ...] = (),
    ) -> Circuit:
        return self._append(name, matrix, (q,), params)

    def i(self, q: int) -> Circuit:
        return self._one("i", gates.I, q)

    def x(self, q: int) -> Circuit:
        return self._one("x", gates.X, q)

    def y(self, q: int) -> Circuit:
        return self._one("y", gates.Y, q)

    def z(self, q: int) -> Circuit:
        return self._one("z", gates.Z, q)

    def h(self, q: int) -> Circuit:
        return self._one("h", gates.H, q)

    def s(self, q: int) -> Circuit:
        return self._one("s", gates.S, q)

    def sdg(self, q: int) -> Circuit:
        return self._one("sdg", gates.SDG, q)

    def t(self, q: int) -> Circuit:
        return self._one("t", gates.T, q)

    def tdg(self, q: int) -> Circuit:
        return self._one("tdg", gates.TDG, q)

    def sx(self, q: int) -> Circuit:
        return self._one("sx", gates.SX, q)

    def rx(self, q: int, theta: float) -> Circuit:
        return self._one("rx", gates.rx(theta), q, (float(theta),))

    def ry(self, q: int, theta: float) -> Circuit:
        return self._one("ry", gates.ry(theta), q, (float(theta),))

    def rz(self, q: int, theta: float) -> Circuit:
        return self._one("rz", gates.rz(theta), q, (float(theta),))

    def phase(self, q: int, lam: float) -> Circuit:
        return self._one("phase", gates.phase(lam), q, (float(lam),))

    # Two- and three-qubit gates ----------------------------------------
    def cnot(self, control: int, target: int) -> Circuit:
        return self._append("cnot", gates.CNOT, (control, target))

    def cx(self, control: int, target: int) -> Circuit:
        return self.cnot(control, target)

    def cz(self, control: int, target: int) -> Circuit:
        return self._append("cz", gates.CZ, (control, target))

    def swap(self, first: int, second: int) -> Circuit:
        return self._append("swap", gates.SWAP, (first, second))

    def cphase(self, control: int, target: int, lam: float) -> Circuit:
        return self._append("cphase", gates.cphase(lam), (control, target), (float(lam),))

    def toffoli(self, first: int, second: int, target: int) -> Circuit:
        return self._append("toffoli", gates.TOFFOLI, (first, second, target))

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------
    def run(self, initial: np.ndarray | None = None) -> np.ndarray:
        """Execute the circuit and return the final statevector."""
        state = (
            basis_state(self.n_qubits)
            if initial is None
            else np.asarray(initial, dtype=complex).reshape(-1)
        )
        if state.size != 2**self.n_qubits:
            raise ValueError(f"Initial state must have {2**self.n_qubits} amplitudes")
        for op in self._ops:
            if op.global_op:
                state = op.matrix @ state
            else:
                state = _apply_operator(state, op.matrix, op.qubits, self.n_qubits)
        return state

    def probabilities(self, initial: np.ndarray | None = None) -> np.ndarray:
        return probabilities(self.run(initial))

    def sample(
        self,
        shots: int = 1024,
        seed: int | None = None,
        *,
        qubits: Sequence[int] | None = None,
    ) -> dict[str, int]:
        return sample_counts(self.run(), shots, seed, qubits=qubits)

    def inverse(self) -> Circuit:
        """Return the inverse circuit (operations reversed and adjointed)."""
        out = Circuit(self.n_qubits)
        for op in reversed(self._ops):
            out._ops.append(
                Operation(f"{op.name}†", op.qubits, op.matrix.conj().T, op.params, op.global_op)
            )
        return out

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._ops)

    @property
    def operations(self) -> tuple[Operation, ...]:
        return tuple(self._ops)

    @property
    def depth(self) -> int:
        """Circuit depth counting each operation as one layer on its wires."""
        per_qubit = [0] * self.n_qubits
        for op in self._ops:
            layer = max(per_qubit[q] for q in op.qubits) + 1
            for q in op.qubits:
                per_qubit[q] = layer
        return max(per_qubit, default=0)

    def gate_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for op in self._ops:
            counts[op.name] = counts.get(op.name, 0) + 1
        return dict(sorted(counts.items()))

    def draw(self) -> str:
        """Compact ASCII diagram, one operation per line."""
        lines = []
        for step, op in enumerate(self._ops, start=1):
            wires = ",".join(str(q) for q in op.qubits)
            params = "".join(f"{p:.3f} " for p in op.params).strip()
            label = f"{op.name}({params})" if params else op.name
            lines.append(f"{step:>3}. q[{wires}] {label}")
        return "\n".join(lines)


def _apply_operator(
    state: np.ndarray,
    matrix: np.ndarray,
    qubits: tuple[int, ...],
    n_qubits: int,
) -> np.ndarray:
    """Apply a dense unitary to selected wires via tensor contraction."""
    k = len(qubits)
    tensor = state.reshape([2] * n_qubits)
    # C-order reshape puts qubit q on tensor axis ``n_qubits - 1 - q``.
    selected = [n_qubits - 1 - q for q in qubits]
    rest = [axis for axis in range(n_qubits) if axis not in selected]
    order = selected + rest
    front = np.transpose(tensor, order)
    rows = front.reshape(2**k, -1)
    applied = np.asarray(matrix, dtype=complex) @ rows
    inverse = [0] * n_qubits
    for position, axis in enumerate(order):
        inverse[axis] = position
    return np.transpose(applied.reshape(front.shape), inverse).reshape(-1)
