# Contributing to Tessera

Thank you for your interest in contributing to Tessera. This document explains how to get involved.

## Getting started

1. Fork the repository and clone your fork
2. Create a virtual environment and install in development mode:

```bash
python -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
```

3. Run the tests to make sure everything works:

```bash
pytest
```

## Making changes

1. Create a branch from `main` for your work
2. Write tests for any new functionality
3. Make sure all tests pass before opening a PR
4. Follow the existing code style (we use Black with a 100-character line length)

## Pull requests

- Keep PRs focused — one feature or fix per PR
- Write a clear description of what the change does and why
- Reference any related issues
- Make sure CI passes

## Reporting issues

- Use the GitHub issue templates
- Include a minimal reproducible example where possible
- Specify your Python version, PyTorch version, and OS

## Code style

- We use [Black](https://github.com/psf/black) for formatting (line length 100)
- Type hints are encouraged for all public APIs
- Docstrings follow Google style
- Module-level docstrings should explain the purpose and key concepts

## Testing

- Tests live in `tests/` and use pytest
- Aim for unit tests on individual functions and integration tests on the full pipeline
- Use `set_seed(42)` for reproducibility in any test involving randomness

## Licence

By contributing, you agree that your contributions will be licensed under the Apache License 2.0.
