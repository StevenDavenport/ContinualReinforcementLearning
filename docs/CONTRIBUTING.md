# Contributing

## Development Flow
1. Create and activate `crl_bench` environment.
2. Implement changes with tests.
3. Run local quality gates:
   - `python -m ruff check .`
   - `python -m ruff format --check .`
   - `python -m mypy src tests`
   - `python -m pytest`
4. Open PR with checklist updates in `PLAN.md` when relevant.

## Scope Boundaries
- Core benchmark contracts live under `src/crlbench/core`.
- Experiment-specific implementations must use adapters and shared evaluator contracts.
- Metric definition changes require explicit governance update and benchmark versioning note.

## Pull Request Requirements
- Include tests for new behavior and regressions.
- Preserve deterministic behavior for smoke protocols.
- Add/refresh docs for user-facing or protocol-facing changes.
