"""The acceptance suite itself must pass and report meaningful measurements."""

from __future__ import annotations

from collections import Counter

from quantpulse.acceptance import ACCEPTANCE_CASES, run_acceptance


def test_every_acceptance_case_passes() -> None:
    results = run_acceptance()
    assert len(results) == len(ACCEPTANCE_CASES)
    failures = [result.name for result in results if not result.passed]
    assert not failures, f"Failing acceptance cases: {failures}"


def test_case_names_are_unique() -> None:
    names = [case.name for case in ACCEPTANCE_CASES]
    assert len(names) == len(set(names))


def test_cases_cover_all_layers() -> None:
    results = run_acceptance()
    categories = Counter(result.category for result in results)
    assert categories["circuits"] >= 2
    assert categories["algorithms"] >= 6
    assert categories["pulse"] >= 4


def test_measurements_are_finite_and_timed() -> None:
    for result in run_acceptance():
        assert result.duration_ms >= 0.0
        assert result.detail
        assert result.measured == result.measured  # not NaN


def test_each_case_has_a_reference_measurement() -> None:
    for result in run_acceptance():
        assert "=" in result.detail
        assert result.reference == "pass"
