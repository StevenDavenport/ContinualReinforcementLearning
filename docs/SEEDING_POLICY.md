# Seeding Policy

Global run seed:
- `RunConfig.seed` is the root seed for a run.

Sub-seed derivation:
- Use `derive_subseed(global_seed, namespace)` from `crlbench.runtime.seeding`.
- Namespace examples: `env/train`, `env/eval`, `agent/update`, `task_stream`.

Deterministic mode:
- `deterministic_mode=off`: never request deterministic backend behavior.
- `deterministic_mode=auto`: enable only when backend advertises support.
- `deterministic_mode=on`: require deterministic backend behavior; error otherwise.
