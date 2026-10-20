"""Command-line behaviour: JSON contracts, exit codes and file output."""

from __future__ import annotations

import json

import pytest

from quantpulse import __version__, cli
from quantpulse.bench import Section

TINY_SECTION = Section(
    title="Tiny",
    description="Tiny benchmark.",
    headers=("Metric", "Value"),
    rows=(("a", "1"),),
    notes=("A note.",),
)


def test_info_json(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["info", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["version"] == __version__
    assert "numpy" in payload


def test_info_plain(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["info"]) == 0
    out = capsys.readouterr().out
    assert "version" in out
    assert "numpy" in out


def test_eval_json(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["eval", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["passed"] == payload["total"]
    assert payload["total"] >= 10
    case = payload["cases"][0]
    assert {"name", "category", "passed", "measured", "detail", "duration_ms"} <= set(case)


def test_eval_table(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["eval"]) == 0
    out = capsys.readouterr().out
    assert "PASS" in out
    assert "cases passed" in out


def test_demo_json(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["demo", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["circuits"]["bell"]["p00"] == pytest.approx(0.5)
    assert payload["algorithms"]["grover"]["success_probability"] > 0.9
    assert payload["pulse"]["drag_reduction"] > 1.0


def test_demo_table(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["demo"]) == 0
    out = capsys.readouterr().out
    assert "Circuit engine" in out
    assert "Pulse control" in out
    assert "DRAG" in out


def test_bench_json_with_patch(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli, "collect_benchmarks", lambda: [TINY_SECTION])
    assert cli.main(["bench", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload[0]["title"] == "Tiny"
    assert payload[0]["rows"] == [["a", "1"]]


def test_bench_write_with_patch(
    tmp_path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli, "collect_benchmarks", lambda: [TINY_SECTION])
    target = tmp_path / "BENCHMARKS.md"
    assert cli.main(["bench", "--write", str(target)]) == 0
    assert target.exists()
    content = target.read_text(encoding="utf-8")
    assert "# Benchmarks" in content
    assert "## Tiny" in content


def test_bench_prints_markdown(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli, "collect_benchmarks", lambda: [TINY_SECTION])
    assert cli.main(["bench"]) == 0
    assert "## Tiny" in capsys.readouterr().out


def test_missing_command_exits_two() -> None:
    with pytest.raises(SystemExit) as error:
        cli.main([])
    assert error.value.code == 2


def test_unknown_command_exits_two() -> None:
    with pytest.raises(SystemExit) as error:
        cli.main(["nope"])
    assert error.value.code == 2
