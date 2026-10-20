# Contributing

Thanks for taking the time to improve Quant-Pulse-Processor. This project keeps
a deliberately small surface: one runtime dependency, readable math, and tests
that compare against exact references.

## Development setup

```bash
git clone https://github.com/MrKyaw/Quant-Pulse-Processor.git
cd Quant-Pulse-Processor
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Run the same gate as CI before opening a pull request:

```bash
make check
```

## What a good change looks like

- **Correctness is demonstrated, not asserted.** New algorithms need a test
  against a closed form, a brute-force reference, or dense diagonalisation.
- **Conventions are preserved.** Qubit 0 is the least significant bit; gates
  listed as `(q0, q1)` treat `q0` as the most significant bit of the gate-local
  index. If a change touches wire ordering, add a regression test.
- **Physics has units.** Pulse code uses angular frequencies and `ħ = 1`.
  Document the unit in docstrings and tests.
- **Errors are explicit.** Invalid indices, non-unitary matrices and impossible
  physical parameters raise `ValueError` with a useful message.
- **The public surface stays small.** Prefer adding a focused function over a
  new abstraction layer.

## Adding an algorithm

1. Create `src/quantpulse/algorithms/<name>.py`.
2. Export the public functions from `algorithms/__init__.py`.
3. Add an exact reference (dense matrix, analytic probability, or brute force).
4. Add unit tests under `tests/` and, if it belongs in the suite, a case in
   `acceptance.py`.
5. Update `docs/METHOD.md` with the equations and the reference used.
6. Re-run `quantpulse bench --write` if you add a benchmark section.

## Tests

```bash
pytest                      # 250 fast tests
pytest -m slow              # full benchmark sweep
pytest --cov=quantpulse     # coverage report
```

The suite must stay above 90% statement coverage; CI enforces it. A bug fix
should include a test that fails on the old code.

## Style

- Python 3.10+ syntax, type hints on public functions.
- `ruff check` and `ruff format --check` must pass (line length 100).
- Docstrings explain *why* and state the convention; comments explain the
  non-obvious step, not the syntax.
- Keep NumPy operations vectorised where it does not hurt readability.

## Commits and pull requests

- Use short, imperative commit subjects (`add DRAG sweep`, `fix QFT wire order`).
- One logical change per pull request.
- Describe the measured effect: which test was added, which number changed.
- Link the issue the change addresses.

## Reporting bugs and asking questions

Use the issue templates:

- [Bug report](https://github.com/MrKyaw/Quant-Pulse-Processor/issues/new?template=bug_report.yml)
- [Feature request](https://github.com/MrKyaw/Quant-Pulse-Processor/issues/new?template=feature_request.yml)
- [Question](https://github.com/MrKyaw/Quant-Pulse-Processor/issues/new?template=question.yml)

Please include the output of `quantpulse info` and, for numerical issues, the
smallest snippet that reproduces the behaviour.

## License

By contributing you agree that your work is released under the MIT license.
