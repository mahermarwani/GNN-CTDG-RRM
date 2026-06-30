# Repository Guidelines

## Project Structure & Module Organization

This repository is organized as a Python reimplementation of “Event-Based Temporal Graph Neural Network for Radio Resource Management.”

- `src/tgnn_rrm/` contains importable package code.
- `src/tgnn_rrm/config.py` defines experiment and radio configuration objects.
- `src/tgnn_rrm/radio.py` implements dependency-free wireless rate and constraint utilities.
- `tests/` contains unit tests based on Python's built-in `unittest`.

Add future simulator, CTDG event, model, training, and evaluation modules under `src/tgnn_rrm/` with focused tests under `tests/`.

## Build, Test, and Development Commands

Current checks are dependency-free:

- `PYTHONPATH=src python -m unittest discover -s tests` runs the unit tests.
- `python -m compileall src tests` checks Python syntax.
- `python -m pip install -e .` installs the package in editable mode when packaging tools are available.

Document any new training or evaluation command in `README.md` as soon as it is added.

## Coding Style & Naming Conventions

Use Python 3.10+ with type hints for public functions. Keep modules small and named by responsibility, for example `events.py`, `channel.py`, or `losses.py`.

Use 4-space indentation, `snake_case` for functions and variables, `PascalCase` for dataclasses/classes, and concise docstrings for public APIs. Prefer deterministic, testable functions for simulator and metric logic.

## Testing Guidelines

Use `unittest` until a pytest dependency is introduced. Add focused tests for every equation, constraint, event transition, and tensor shape convention.

Tests should be named `test_*.py` and should avoid requiring GPU, PyTorch, NumPy, or generated datasets unless those dependencies are explicitly declared.

## Commit & Pull Request Guidelines

Local git history is not available in this checkout, so no repository-specific commit convention can be inferred. Use concise, imperative commit messages, for example `Add radio rate utilities` or `Implement CTDG event builder`.

Pull requests should summarize the implemented paper component, list validation commands, and call out any intentional deviations from the manuscript. Link related issues or experiment notes when available.

## Agent-Specific Instructions

Before editing, check existing modules and tests. Keep changes scoped, avoid committing generated caches or datasets, and preserve reproducibility by documenting seeds, parameters, and commands.
