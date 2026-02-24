# Artifact and Result Schema Contract

All runs write artifacts under:
`artifacts/<run_id>/`

Required files:
- `manifest.json`
- `resolved_config.json`
- `events.jsonl`
- `run_metrics_summary.json` (once metric evaluation is executed)

## `manifest.json`
Required top-level fields:
- `schema_version`
- `run_id`
- `created_at_utc`
- `run_name`
- `experiment`
- `track`
- `env_family`
- `env_option`
- `seed`
- `num_seeds`
- `deterministic_mode`
- `seed_namespace`
- `config_sha256`
- `code`
- `environment`
- `lockfiles`

## `events.jsonl`
Each line is a JSON object with:
- `schema_version`
- `sequence`
- `event`
- `timestamp_utc`
- `payload`

Event ordering is defined by monotonic `sequence`.

## Future Result Files
Per-eval summaries should follow `EvalSummaryRecord`.
Run-level summaries should follow `RunMetricsSummaryRecord`.
Experiment-level summaries should follow `ExperimentMetricsSummaryRecord`.
All are defined in `src/crlbench/runtime/schemas.py`.
