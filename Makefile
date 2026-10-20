.PHONY: install test coverage lint format bench demo eval check clean

install:
	python -m pip install -e ".[dev]"

test:
	python -m pytest

coverage:
	python -m pytest --cov=quantpulse --cov-report=term-missing

lint:
	ruff check src tests
	ruff format --check src tests

format:
	ruff check --fix src tests
	ruff format src tests

bench:
	python -m quantpulse bench --write docs/BENCHMARKS.md

demo:
	python -m quantpulse demo

eval:
	python -m quantpulse eval

check: lint test

clean:
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov build dist
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type d -name "*.egg-info" -prune -exec rm -rf {} +
