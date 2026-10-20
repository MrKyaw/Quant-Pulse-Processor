"""Gate library for the statevector engine.

Every gate is a dense unitary matrix acting on a fixed number of qubits.
Parametrised gates return a fresh matrix, while the named Clifford gates are
shared module-level constants.  Matrices follow the little-endian convention
used throughout the package: qubit 0 is the least significant bit of a basis
state index.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "CNOT",
    "CZ",
    "SDG",
    "SWAP",
    "SX",
    "TDG",
    "TOFFOLI",
    "H",
    "I",
    "S",
    "T",
    "X",
    "Y",
    "Z",
    "cphase",
    "is_unitary",
    "pauli",
    "phase",
    "rx",
    "ry",
    "rz",
    "u3",
]

_SQRT2 = np.sqrt(2.0)

I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / _SQRT2
S = np.array([[1, 0], [0, 1j]], dtype=complex)
SDG = np.array([[1, 0], [0, -1j]], dtype=complex)
T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)
TDG = np.array([[1, 0], [0, np.exp(-1j * np.pi / 4)]], dtype=complex)
SX = 0.5 * np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]], dtype=complex)

CNOT = np.array(
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
    dtype=complex,
)
CZ = np.diag([1, 1, 1, -1]).astype(complex)
SWAP = np.array(
    [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
    dtype=complex,
)
TOFFOLI = np.eye(8, dtype=complex)
TOFFOLI[6, 6] = 0
TOFFOLI[7, 7] = 0
TOFFOLI[6, 7] = 1
TOFFOLI[7, 6] = 1


def rx(theta: float) -> np.ndarray:
    """Rotation around the x axis: ``exp(-i theta X / 2)``."""
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)


def ry(theta: float) -> np.ndarray:
    """Rotation around the y axis: ``exp(-i theta Y / 2)``."""
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def rz(theta: float) -> np.ndarray:
    """Rotation around the z axis: ``diag(exp(-i theta/2), exp(i theta/2))``."""
    return np.array(
        [[np.exp(-0.5j * theta), 0], [0, np.exp(0.5j * theta)]],
        dtype=complex,
    )


def phase(lam: float) -> np.ndarray:
    """Global phase gate ``diag(1, exp(i lam))``."""
    return np.array([[1, 0], [0, np.exp(1j * lam)]], dtype=complex)


def cphase(lam: float) -> np.ndarray:
    """Controlled phase gate ``diag(1, 1, 1, exp(i lam))``."""
    return np.diag([1, 1, 1, np.exp(1j * lam)]).astype(complex)


def u3(theta: float, phi: float, lam: float) -> np.ndarray:
    """General single-qubit unitary in the U3 parametrisation."""
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array(
        [
            [c, -np.exp(1j * lam) * s],
            [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c],
        ],
        dtype=complex,
    )


_PAULIS = {"I": I, "X": X, "Y": Y, "Z": Z}


def pauli(label: str) -> np.ndarray:
    """Kronecker product of Pauli operators, e.g. ``pauli("XZ")``."""
    if not label:
        raise ValueError("Pauli label must not be empty")
    out = np.array([[1]], dtype=complex)
    for char in label.upper():
        if char not in _PAULIS:
            raise ValueError(f"Unknown Pauli operator {char!r}")
        out = np.kron(out, _PAULIS[char])
    return out


def is_unitary(matrix: np.ndarray, atol: float = 1e-10) -> bool:
    """Return True when ``M M^dagger = I`` within ``atol``."""
    m = np.asarray(matrix, dtype=complex)
    if m.ndim != 2 or m.shape[0] != m.shape[1]:
        return False
    return bool(np.allclose(m @ m.conj().T, np.eye(m.shape[0]), atol=atol))
