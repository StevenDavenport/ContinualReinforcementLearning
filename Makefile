PYTHON ?= python

.PHONY: help format lint typecheck test check precommit smoke repro-smoke plots

help:
	@echo "Targets:"
	@echo "  format     - Run formatter"
	@echo "  lint       - Run linter"
	@echo "  typecheck  - Run mypy"
	@echo "  test       - Run tests"
	@echo "  precommit  - Run pre-commit hooks on all files"
	@echo "  smoke      - Run dry-run smoke protocol"
	@echo "  repro-smoke- Run reproducibility smoke protocol"
	@echo "  plots      - Generate canonical plot specs from example trace"
	@echo "  check      - Run lint + typecheck + test"

format:
	$(PYTHON) -m ruff format .

lint:
	$(PYTHON) -m ruff check .

typecheck:
	$(PYTHON) -m mypy src tests

test:
	$(PYTHON) -m pytest

precommit:
	$(PYTHON) -m pre_commit run --all-files

smoke:
	$(PYTHON) -m crlbench smoke-run --config configs/base.json --max-tasks 3

repro-smoke:
	$(PYTHON) -m crlbench repro-smoke --config configs/base.json --max-tasks 3

plots:
	$(PYTHON) -m crlbench compute-stream-metrics \
		--trace examples/stream_trace.json \
		--metadata run_id=demo_run \
		--metadata experiment=exp1 \
		--metadata track=toy \
		--metadata env_family=dm_control \
		--metadata env_option=vision \
		--out /tmp/run_metrics_summary.json
	$(PYTHON) -m crlbench generate-canonical-plots \
		--summary /tmp/run_metrics_summary.json \
		--out-dir /tmp/canonical_plots

check: lint typecheck test
