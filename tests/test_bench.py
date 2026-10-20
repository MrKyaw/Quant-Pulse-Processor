"""Benchmark sections and Markdown rendering."""

from __future__ import annotations

import pytest

from quantpulse.bench import (
    Section,
    _engine_section,
    _grover_sweep_section,
    _pulse_section,
    collect_benchmarks,
    render_markdown,
)


def test_render_markdown_contains_sections_and_environment() -> None:
    section = Section(
        title="Demo section",
        description="A short description.",
        headers=("A", "B"),
        rows=(("1", "2"), ("3", "4")),
        notes=("A note.",),
    )
    report = render_markdown([section], version="9.9.9")
    assert "# Benchmarks" in report
    assert "## Demo section" in report
    assert "| A | B |" in report
    assert "| 1 | 2 |" in report
    assert "A note." in report
    assert "9.9.9" in report
    assert "NumPy" in report


def test_render_markdown_is_stable() -> None:
    section = Section("T", "D", ("X",), (("1",),))
    assert render_markdown([section]) == render_markdown([section])


@pytest.mark.parametrize("builder", [_engine_section, _grover_sweep_section, _pulse_section])
def test_section_rows_match_headers(builder) -> None:  # type: ignore[no-untyped-def]
    section = builder()
    assert section.title
    assert section.description
    assert section.rows
    for row in section.rows:
        assert len(row) == len(section.headers)


def test_engine_section_covers_growing_registers() -> None:
    section = _engine_section()
    assert section.rows[0][0] == "4"
    assert section.rows[-1][0] == "12"
    assert len(section.rows) == 5


def test_pulse_section_includes_drag_sweep() -> None:
    section = _pulse_section()
    betas = [row[0] for row in section.rows]
    assert betas[0] == "0.00"
    assert "0.75" in betas


@pytest.mark.slow
def test_collect_benchmarks_returns_all_sections() -> None:
    sections = collect_benchmarks()
    assert len(sections) == 7
    titles = {section.title for section in sections}
    assert "QFT circuit scaling" in titles
    assert "QAOA for MaxCut" in titles
    assert "VQE ground states" in titles
