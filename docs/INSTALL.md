# Installation

## Recommended (Conda)
```bash
conda env create -f environment.yml
conda activate crl_bench
```

Update dependencies after local changes to `pyproject.toml`:
```bash
conda activate crl_bench
python -m pip install -e ".[dev,yaml]"
```

## Alternative (venv)
```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,yaml]"
```

## Verify Toolchain
```bash
python -m ruff check .
python -m mypy src tests
python -m pytest
```
