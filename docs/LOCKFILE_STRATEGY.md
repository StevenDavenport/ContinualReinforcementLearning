# Lockfile Strategy

Current reproducible install strategy:
- `pyproject.toml` defines dependency bounds.
- `environment.yml` pins Python/Pip toolchain for conda-based development.

Policy:
- Any dependency-bound change in `pyproject.toml` requires verifying
  `conda env create -f environment.yml` and `python -m pip install -e ".[dev,yaml]"`.
- CI remains the source of truth for quality gates (`ruff`, `mypy`, `pytest`).
