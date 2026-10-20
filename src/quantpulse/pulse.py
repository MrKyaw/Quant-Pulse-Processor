"""Pulse-level control of a superconducting qubit.

Two models are provided:

* a driven two-level system in the rotating frame,
  ``H(t) = (Omega(t) X + delta Z) / 2``, optionally evolved with a Lindblad
  master equation (amplitude damping ``T1`` and pure dephasing set by ``T2``);
* a three-level transmon truncated to ``{|0>, |1>, |2>}`` with anharmonicity
  ``alpha``, used to quantify leakage and the effect of a DRAG correction.

Frequencies are angular frequencies and time is expressed in the same unit
(nanoseconds in the examples), with ``hbar = 1``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

__all__ = [
    "PulseResult",
    "bloch_vector",
    "drag_envelopes",
    "gaussian_envelope",
    "pi_pulse",
    "pulse_area",
    "simulate_three_level",
    "simulate_two_level",
    "square_envelope",
]

_SX = np.array([[0, 1], [1, 0]], dtype=complex)
_SY = np.array([[0, -1j], [1j, 0]], dtype=complex)
_SZ = np.array([[1, 0], [0, -1]], dtype=complex)
_SM = np.array([[0, 1], [0, 0]], dtype=complex)  # |0><1|


@dataclass(frozen=True)
class PulseResult:
    """Time traces of a simulated pulse."""

    times: np.ndarray
    population: np.ndarray
    coherence: np.ndarray
    bloch: np.ndarray
    leakage: np.ndarray | None = None

    @property
    def final_population(self) -> float:
        """Excited-state population at the end of the pulse."""
        return float(self.population[-1])

    @property
    def final_leakage(self) -> float:
        """Population of ``|2>`` at the end of the pulse (0 for two levels)."""
        return 0.0 if self.leakage is None else float(self.leakage[-1])


# ----------------------------------------------------------------------
# Envelopes
# ----------------------------------------------------------------------
def _time_grid(duration: float, dt: float) -> np.ndarray:
    if duration <= 0:
        raise ValueError("duration must be positive")
    if dt <= 0 or dt > duration:
        raise ValueError("dt must be positive and no larger than duration")
    steps = round(duration / dt)
    return np.linspace(0.0, duration, steps + 1)


def gaussian_envelope(
    times: np.ndarray,
    duration: float,
    *,
    amplitude: float = 1.0,
    sigma: float | None = None,
) -> np.ndarray:
    """Truncated Gaussian envelope centred on ``duration / 2``."""
    t = np.asarray(times, dtype=float)
    if duration <= 0:
        raise ValueError("duration must be positive")
    sigma = duration / 6.0 if sigma is None else float(sigma)
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    return amplitude * np.exp(-0.5 * ((t - duration / 2.0) / sigma) ** 2)


def square_envelope(times: np.ndarray, duration: float, *, amplitude: float = 1.0) -> np.ndarray:
    """Ideal rectangular envelope."""
    t = np.asarray(times, dtype=float)
    if duration <= 0:
        raise ValueError("duration must be positive")
    return np.where((t >= 0.0) & (t <= duration), amplitude, 0.0).astype(float)


def drag_envelopes(
    times: np.ndarray,
    duration: float,
    *,
    amplitude: float = 1.0,
    sigma: float | None = None,
    beta: float = 0.5,
    anharmonicity: float = -1.2,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(in_phase, quadrature)`` for a DRAG-corrected Gaussian.

    The quadrature component is ``-beta / alpha * d(Omega_I)/dt``, the standard
    first-order correction that cancels the ``|1> -> |2>`` leakage path.
    """
    if anharmonicity == 0:
        raise ValueError("anharmonicity must be non-zero")
    sigma = duration / 6.0 if sigma is None else float(sigma)
    in_phase = gaussian_envelope(times, duration, amplitude=amplitude, sigma=sigma)
    derivative = -(np.asarray(times, dtype=float) - duration / 2.0) / sigma**2 * in_phase
    quadrature = -beta / anharmonicity * derivative
    return in_phase, quadrature


def pulse_area(envelope: np.ndarray, dt: float) -> float:
    """Integrated pulse area ``int Omega(t) dt`` (trapezoidal rule)."""
    samples = np.asarray(envelope, dtype=float)
    if samples.size < 2:
        return 0.0
    return float(np.sum(0.5 * (samples[1:] + samples[:-1])) * dt)


def pi_pulse(
    duration: float = 24.0,
    dt: float = 0.05,
    *,
    sigma: float | None = None,
    beta: float | None = None,
    anharmonicity: float = -1.2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalise a Gaussian (optionally DRAG) to area ``pi``.

    Returns ``(times, in_phase, quadrature)``.  With ``beta=None`` the
    quadrature is all zeros and the result is a plain Gaussian pi-pulse.
    """
    times = _time_grid(duration, dt)
    unit = gaussian_envelope(times, duration, amplitude=1.0, sigma=sigma)
    area = pulse_area(unit, dt)
    if area <= 0:
        raise ValueError("Envelope has zero area")
    amplitude = np.pi / area
    if beta is None:
        return times, unit * amplitude, np.zeros_like(unit)
    omega_i, omega_q = drag_envelopes(
        times,
        duration,
        amplitude=amplitude,
        sigma=sigma,
        beta=beta,
        anharmonicity=anharmonicity,
    )
    scale = np.pi / pulse_area(omega_i, dt)
    return times, omega_i * scale, omega_q * scale


# ----------------------------------------------------------------------
# Two-level simulation
# ----------------------------------------------------------------------
def bloch_vector(rho: np.ndarray) -> np.ndarray:
    """Bloch vector ``(<X>, <Y>, <Z>)`` of a two-level density matrix."""
    matrix = np.asarray(rho, dtype=complex)
    if matrix.shape != (2, 2):
        raise ValueError("Expected a 2x2 density matrix")
    return np.array(
        [
            np.trace(matrix @ _SX).real,
            np.trace(matrix @ _SY).real,
            np.trace(matrix @ _SZ).real,
        ]
    )


def _two_level_unitary(omega: float, detuning: float, dt: float) -> np.ndarray:
    rate = np.hypot(omega, detuning)
    if rate < 1e-15:
        return np.eye(2, dtype=complex)
    phase = rate * dt / 2.0
    return np.cos(phase) * np.eye(2) - 1j * np.sin(phase) / rate * (omega * _SX + detuning * _SZ)


def _two_level_rhs(
    rho: np.ndarray,
    hamiltonian: np.ndarray,
    gamma1: float,
    gamma_phi: float,
) -> np.ndarray:
    drift = -1j * (hamiltonian @ rho - rho @ hamiltonian)
    lowering = _SM @ rho @ _SM.conj().T
    anti = _SM.conj().T @ _SM @ rho + rho @ _SM.conj().T @ _SM
    damping = gamma1 * (lowering - 0.5 * anti)
    dephasing = gamma_phi * (_SZ @ rho @ _SZ - rho)
    return drift + damping + dephasing


def simulate_two_level(
    omega: np.ndarray,
    dt: float,
    *,
    detuning: float = 0.0,
    t1: float | None = None,
    t2: float | None = None,
    initial: Literal["ground", "excited", "plus"] = "ground",
) -> PulseResult:
    """Evolve a driven two-level system, with optional Lindblad noise."""
    drive = np.asarray(omega, dtype=float).reshape(-1)
    if drive.size < 2:
        raise ValueError("omega must contain at least two samples")
    if dt <= 0:
        raise ValueError("dt must be positive")
    if (t1 is not None) != (t2 is not None):
        raise ValueError("Provide both t1 and t2, or neither")
    if t1 is not None and t2 is not None and t2 > 2 * t1 + 1e-12:
        raise ValueError("Physical decay requires T2 <= 2 * T1")

    if initial == "ground":
        rho = np.array([[1, 0], [0, 0]], dtype=complex)
    elif initial == "excited":
        rho = np.array([[0, 0], [0, 1]], dtype=complex)
    elif initial == "plus":
        rho = np.full((2, 2), 0.5, dtype=complex)
    else:
        raise ValueError(f"Unknown initial state {initial!r}")

    times = np.arange(drive.size + 1) * dt
    population = np.zeros(times.size)
    coherence = np.zeros(times.size)
    bloch = np.zeros((times.size, 3))

    def record(index: int) -> None:
        population[index] = rho[1, 1].real
        coherence[index] = abs(rho[0, 1])
        bloch[index] = bloch_vector(rho)

    record(0)
    if t1 is None:
        unitary_cache: dict[float, np.ndarray] = {}
        for k, value in enumerate(drive):
            key = float(value)
            step = unitary_cache.get(key)
            if step is None:
                step = _two_level_unitary(key, detuning, dt)
                unitary_cache[key] = step
            rho = step @ rho @ step.conj().T
            record(k + 1)
    else:
        decay_time = float(t1)
        coherence_time = float(t2) if t2 is not None else 2.0 * decay_time
        gamma1 = 1.0 / decay_time
        gamma_phi = 0.5 * (1.0 / coherence_time - 0.5 * gamma1)
        for k, value in enumerate(drive):
            hamiltonian = 0.5 * (value * _SX + detuning * _SZ)
            k1 = _two_level_rhs(rho, hamiltonian, gamma1, gamma_phi)
            k2 = _two_level_rhs(rho + 0.5 * dt * k1, hamiltonian, gamma1, gamma_phi)
            k3 = _two_level_rhs(rho + 0.5 * dt * k2, hamiltonian, gamma1, gamma_phi)
            k4 = _two_level_rhs(rho + dt * k3, hamiltonian, gamma1, gamma_phi)
            rho = rho + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
            record(k + 1)
    return PulseResult(times, population, coherence, bloch)


# ----------------------------------------------------------------------
# Three-level transmon
# ----------------------------------------------------------------------
def _three_level_operators() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x01 = np.zeros((3, 3), dtype=complex)
    x01[0, 1] = x01[1, 0] = 1.0
    x12 = np.zeros((3, 3), dtype=complex)
    x12[1, 2] = x12[2, 1] = np.sqrt(2.0)
    y01 = np.zeros((3, 3), dtype=complex)
    y01[0, 1] = -1j
    y01[1, 0] = 1j
    y12 = np.zeros((3, 3), dtype=complex)
    y12[1, 2] = -1j * np.sqrt(2.0)
    y12[2, 1] = 1j * np.sqrt(2.0)
    return x01 + x12, y01 + y12, np.diag([0.0, 0.0, 1.0]).astype(complex)


def simulate_three_level(
    omega_i: np.ndarray,
    omega_q: np.ndarray | None,
    dt: float,
    *,
    anharmonicity: float = -1.2,
    initial: Literal["ground", "excited"] = "ground",
) -> PulseResult:
    """Evolve ``{|0>, |1>, |2>}`` under a two-quadrature drive."""
    drive_i = np.asarray(omega_i, dtype=float).reshape(-1)
    drive_q = (
        np.zeros_like(drive_i) if omega_q is None else np.asarray(omega_q, dtype=float).reshape(-1)
    )
    if drive_i.size != drive_q.size:
        raise ValueError("Both quadratures must have the same length")
    if drive_i.size < 2:
        raise ValueError("Envelopes must contain at least two samples")
    if dt <= 0:
        raise ValueError("dt must be positive")
    x_ops, y_ops, number = _three_level_operators()
    hamiltonian = anharmonicity * number

    rho = np.zeros((3, 3), dtype=complex)
    if initial == "ground":
        rho[0, 0] = 1.0
    elif initial == "excited":
        rho[1, 1] = 1.0
    else:
        raise ValueError(f"Unknown initial state {initial!r}")

    times = np.arange(drive_i.size + 1) * dt
    population = np.zeros(times.size)
    leakage = np.zeros(times.size)
    coherence = np.zeros(times.size)
    bloch = np.zeros((times.size, 3))

    def record(index: int) -> None:
        population[index] = rho[1, 1].real
        leakage[index] = rho[2, 2].real
        coherence[index] = abs(rho[0, 1])
        bloch[index] = np.array(
            [
                2 * rho[0, 1].real,
                -2 * rho[0, 1].imag,
                (rho[0, 0] - rho[1, 1]).real,
            ]
        )

    record(0)
    for k, (amp_i, amp_q) in enumerate(zip(drive_i, drive_q)):
        step_h = hamiltonian + 0.5 * (amp_i * x_ops + amp_q * y_ops)
        eigenvalues, eigenvectors = np.linalg.eigh(step_h)
        propagator = eigenvectors @ np.diag(np.exp(-1j * eigenvalues * dt)) @ eigenvectors.conj().T
        rho = propagator @ rho @ propagator.conj().T
        record(k + 1)
    return PulseResult(times, population, coherence, bloch, leakage)
