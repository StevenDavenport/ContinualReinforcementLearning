# CRL Benchmark Suite

Production-grade benchmark infrastructure for continual reinforcement learning (CRL), with a focus on Dreamer-style latent dynamics agents.

## Goals
- Standardized CRL protocols across toy and robotics tracks.
- Reproducible runs with auditable artifacts.
- Publication-ready metrics, plots, and tables.
- External-agent compatibility via adapter contracts.

See `ROADMAP.md` for benchmark design and `PLAN.md` for the execution checklist.

## Current Status
Phase 0-5 core platform work is largely complete:
- repository structure and tooling,
- core interface contracts,
- configuration schema and loader,
- artifact and CLI foundations,
- metrics/reporting/publication pack pipeline,
- CI reliability gates (resume/failure tests, artifact validation, coverage threshold, golden regression).

## Quickstart
```bash
conda env create -f environment.yml
conda activate crl_bench
make check
```

Validate a config:
```bash
python -m crlbench validate-config --config configs/base.json
```

Resolve and export merged config:
```bash
python -m crlbench resolve-config --config configs/base.json --out /tmp/resolved.json
```

Run reproducibility smoke:
```bash
python -m crlbench repro-smoke --config configs/base.json --max-tasks 3
```

Validate generated run artifacts:
```bash
python -m crlbench validate-artifacts --artifacts-dir artifacts
```

Run optional heavy-environment smoke matrix (nightly/manual CI equivalent):
```bash
python -m pytest -m heavy_env -rs
```

Run Experiment 1 quality gate (parity + 5-seed reproducibility + plots/tables):
```bash
python -m crlbench run-experiment1-quality-gate --out-dir /tmp/exp1_quality --seed-count 5
```

Compute metrics from a stream trace:
```bash
python -m crlbench compute-stream-metrics \
  --trace examples/stream_trace.json \
  --metadata run_id=demo_run \
  --metadata experiment=exp1 \
  --metadata track=toy \
  --metadata env_family=dm_control \
  --metadata env_option=vision \
  --out /tmp/run_metrics_summary.json
```

Generate canonical plot specs:
```bash
python -m crlbench generate-canonical-plots \
  --summary /tmp/run_metrics_summary.json \
  --out-dir /tmp/canonical_plots
```

Export one-command publication pack:
```bash
python -m crlbench export-publication-pack \
  --run-dir artifacts/<run_id_1> \
  --run-dir artifacts/<run_id_2> \
  --out-dir /tmp/publication_pack \
  --method agent=dreamer \
  --method backbone=latent_dynamics
```

See `docs/INSTALL.md` for install options, `docs/CONTRIBUTING.md` for development workflow, and
`docs/PUBLICATION_PACK.md` for paper-ready bundle export details.

Experiment 1 protocol details (sequence templates, budget tiers, runtime bounds):
- `docs/EXPERIMENT_1.md`
- base configs in `configs/experiment_1/`
- tier overlays in `configs/tiers/`
