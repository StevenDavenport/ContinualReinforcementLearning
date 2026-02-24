# Ownership and Review Policy

## Ownership Split
- `src/crlbench/core/`: API contracts and validation policy.
- `src/crlbench/config/`: config schemas, composition, and override semantics.
- `src/crlbench/runtime/`: artifacts, orchestration, reporting, and reproducibility.
- `src/crlbench/metrics/`: metric definitions and statistical utilities.
- `src/crlbench/experiments/`: benchmark protocol implementations.

## Review Requirements
- Contract or schema changes require at least one core-maintainer review.
- Metric definition changes require explicit benchmark-governance check.
- Any change touching artifacts must include migration/compatibility note.

## Merge Conditions
- CI must pass.
- Tests must cover new behavior.
- `PLAN.md` checkboxes must be updated for completed scoped tasks.
