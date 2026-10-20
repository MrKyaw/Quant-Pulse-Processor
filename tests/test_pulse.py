"""Pulse envelopes, Rabi dynamics, Lindblad relaxation and DRAG leakage."""

from __future__ import annotations

import numpy as np
import pytest

from quantpulse.pulse import (
    bloch_vector,
    drag_envelopes,
    gaussian_envelope,
    pi_pulse,
    pulse_area,
    simulate_three_level,
    simulate_two_level,
    square_envelope,
)

DT = 0.01


def test_gaussian_envelope_peak_and_symmetry() -> None:
    duration = 10.0
    times = np.linspace(0.0, duration, 101)
    envelope = gaussian_envelope(times, duration, amplitude=2.0, sigma=2.0)
    assert envelope.max() == pytest.approx(2.0)
    assert np.isclose(envelope[50], 2.0)
    assert np.allclose(envelope, envelope[::-1])


def test_gaussian_default_sigma_is_a_sixth_of_duration() -> None:
    duration = 12.0
    times = np.linspace(0.0, duration, 1001)
    envelope = gaussian_envelope(times, duration)
    sigma = duration / 6
    expected = np.exp(-0.5 * ((times - duration / 2) / sigma) ** 2)
    assert np.allclose(envelope, expected)


def test_envelope_validation() -> None:
    times = np.linspace(0, 1, 5)
    with pytest.raises(ValueError):
        gaussian_envelope(times, 0.0)
    with pytest.raises(ValueError):
        gaussian_envelope(times, 1.0, sigma=0.0)
    with pytest.raises(ValueError):
        square_envelope(times, -1.0)


def test_square_envelope() -> None:
    times = np.linspace(0, 2, 5)
    assert np.allclose(square_envelope(times, 1.0, amplitude=3.0), [3, 3, 3, 0, 0])


def test_drag_quadrature_is_zero_at_the_centre() -> None:
    duration = 12.0
    times = np.linspace(0, duration, 121)
    in_phase, quadrature = drag_envelopes(times, duration, sigma=2.0, beta=1.0)
    assert quadrature[60] == pytest.approx(0.0, abs=1e-12)
    assert np.all(np.sign(quadrature[:60]) >= 0)
    assert np.all(np.sign(quadrature[61:]) <= 0)
    assert np.allclose(in_phase, gaussian_envelope(times, duration, sigma=2.0))


def test_drag_with_zero_beta_has_no_quadrature() -> None:
    times = np.linspace(0, 5, 51)
    _, quadrature = drag_envelopes(times, 5.0, beta=0.0)
    assert np.allclose(quadrature, 0.0)


def test_drag_requires_nonzero_anharmonicity() -> None:
    with pytest.raises(ValueError):
        drag_envelopes(np.linspace(0, 1, 5), 1.0, anharmonicity=0.0)


def test_pulse_area_of_square_pulse() -> None:
    times = np.arange(0, 4.5, 0.5)
    envelope = square_envelope(times, 4.0, amplitude=2.0)
    assert pulse_area(envelope, 0.5) == pytest.approx(8.0)


def test_pi_pulse_area_and_quadrature() -> None:
    _, in_phase, quadrature = pi_pulse(12.0, 0.02, sigma=2.0)
    assert pulse_area(in_phase, 0.02) == pytest.approx(np.pi, abs=1e-6)
    assert np.allclose(quadrature, 0.0)


def test_pi_pulse_with_drag_keeps_area() -> None:
    _, in_phase, quadrature = pi_pulse(12.0, 0.02, sigma=2.0, beta=0.75)
    assert pulse_area(in_phase, 0.02) == pytest.approx(np.pi, abs=1e-6)
    assert np.max(np.abs(quadrature)) > 0


def test_pi_pulse_validation() -> None:
    with pytest.raises(ValueError):
        pi_pulse(0.0, 0.01)
    with pytest.raises(ValueError):
        pi_pulse(10.0, 20.0)


def test_rabi_oscillation_matches_analytic_solution() -> None:
    amplitude = 0.5
    dt = 0.001
    steps = 8000
    drive = np.full(steps, amplitude)
    result = simulate_two_level(drive, dt)
    analytic = np.sin(0.5 * amplitude * result.times) ** 2
    assert np.max(np.abs(result.population - analytic)) < 1e-9


def test_pi_pulse_inverts_the_qubit() -> None:
    _, in_phase, _ = pi_pulse(24.0, 0.05)
    result = simulate_two_level(in_phase, 0.05)
    assert result.final_population > 0.999999
    assert np.isclose(result.bloch[-1][2], -1.0, atol=1e-5)


def test_detuned_drive_does_not_fully_invert() -> None:
    _, in_phase, _ = pi_pulse(24.0, 0.05)
    resonant = simulate_two_level(in_phase, 0.05)
    detuned = simulate_two_level(in_phase, 0.05, detuning=0.2)
    assert detuned.final_population < resonant.final_population


def test_t1_decay_matches_exponential() -> None:
    t1 = 10.0
    steps = int(t1 / DT)
    result = simulate_two_level(np.zeros(steps), DT, t1=t1, t2=t1, initial="excited")
    expected = np.exp(-result.times / t1)
    assert np.max(np.abs(result.population - expected)) < 1e-6


def test_t2_coherence_decay() -> None:
    t1, t2 = 20.0, 10.0
    steps = int(t2 / DT)
    result = simulate_two_level(np.zeros(steps), DT, t1=t1, t2=t2, initial="plus")
    expected = 0.5 * np.exp(-result.times / t2)
    assert np.max(np.abs(result.coherence - expected)) < 1e-6


def test_noise_validation() -> None:
    drive = np.zeros(10)
    with pytest.raises(ValueError):
        simulate_two_level(drive, DT, t1=5.0)
    with pytest.raises(ValueError):
        simulate_two_level(drive, DT, t1=5.0, t2=20.0)
    with pytest.raises(ValueError):
        simulate_two_level(drive, 0.0)
    with pytest.raises(ValueError):
        simulate_two_level(np.zeros(1), DT)


def test_initial_state_validation() -> None:
    with pytest.raises(ValueError):
        simulate_two_level(np.zeros(10), DT, initial="up")  # type: ignore[arg-type]


def test_bloch_vector_reference_states() -> None:
    ground = np.array([[1, 0], [0, 0]], dtype=complex)
    excited = np.array([[0, 0], [0, 1]], dtype=complex)
    plus = np.full((2, 2), 0.5, dtype=complex)
    assert np.allclose(bloch_vector(ground), [0, 0, 1])
    assert np.allclose(bloch_vector(excited), [0, 0, -1])
    assert np.allclose(bloch_vector(plus), [1, 0, 0])
    with pytest.raises(ValueError):
        bloch_vector(np.eye(3))


def test_drag_suppresses_leakage() -> None:
    _, bare_i, bare_q = pi_pulse(12.0, 0.02, sigma=2.0)
    bare = simulate_three_level(bare_i, bare_q, 0.02, anharmonicity=-1.2)
    _, drag_i, drag_q = pi_pulse(12.0, 0.02, sigma=2.0, beta=0.75)
    corrected = simulate_three_level(drag_i, drag_q, 0.02, anharmonicity=-1.2)
    assert corrected.final_leakage < 0.1 * bare.final_leakage
    assert corrected.final_population > bare.final_population


def test_three_level_without_drag_quadrature_matches_zeros() -> None:
    _, in_phase, _ = pi_pulse(12.0, 0.02, sigma=2.0)
    with_none = simulate_three_level(in_phase, None, 0.02, anharmonicity=-1.2)
    with_zeros = simulate_three_level(in_phase, np.zeros_like(in_phase), 0.02, anharmonicity=-1.2)
    assert np.allclose(with_none.population, with_zeros.population)
    assert np.allclose(with_none.leakage, with_zeros.leakage)


def test_three_level_validation() -> None:
    drive = np.zeros(10)
    with pytest.raises(ValueError):
        simulate_three_level(drive, np.zeros(9), DT)
    with pytest.raises(ValueError):
        simulate_three_level(np.zeros(1), None, DT)
    with pytest.raises(ValueError):
        simulate_three_level(drive, None, 0.0)
    with pytest.raises(ValueError):
        simulate_three_level(drive, None, DT, initial="plus")  # type: ignore[arg-type]


def test_pulse_result_properties_for_two_level() -> None:
    _, in_phase, _ = pi_pulse(24.0, 0.05)
    result = simulate_two_level(in_phase, 0.05)
    assert result.leakage is None
    assert result.final_leakage == 0.0
    assert result.final_population == result.population[-1]
    assert result.bloch.shape == (result.times.size, 3)
    assert np.all(np.linalg.norm(result.bloch, axis=1) <= 1.0000001)


def test_zero_drive_preserves_ground_state() -> None:
    result = simulate_two_level(np.zeros(100), DT)
    assert np.allclose(result.population, 0.0)
    assert np.allclose(result.coherence, 0.0)
