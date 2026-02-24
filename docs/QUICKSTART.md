# Quickstart

## Install
```bash
conda env create -f environment.yml
conda activate crl_bench
```

## Validate Config
```bash
python -m crlbench validate-config --config configs/base.json
```

## Export Resolved Config
```bash
python -m crlbench resolve-config --config configs/base.json --out /tmp/resolved.json
```

## Run Checks
```bash
make check
```

## Reproducibility Smoke
```bash
python -m crlbench repro-smoke --config configs/base.json --max-tasks 3
```

## Metrics From Trace
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

## Aggregate Metric Summaries
```bash
python -m crlbench aggregate-metric-summaries \
  --summary /tmp/run_metrics_summary.json \
  --out /tmp/experiment_metrics_summary.json \
  --csv /tmp/experiment_metrics_summary.csv \
  --latex /tmp/experiment_metrics_summary.tex
```

## Generate Canonical Plots
```bash
python -m crlbench generate-canonical-plots \
  --summary /tmp/run_metrics_summary.json \
  --out-dir /tmp/canonical_plots
```

## Export Publication Pack
```bash
python -m crlbench export-publication-pack \
  --run-dir artifacts/<run_id_1> \
  --run-dir artifacts/<run_id_2> \
  --out-dir /tmp/publication_pack \
  --method agent=dreamer
```
