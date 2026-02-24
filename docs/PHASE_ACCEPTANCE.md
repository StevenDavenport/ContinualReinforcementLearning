# Phase Acceptance Criteria

This document defines entry/exit gates per phase from `PLAN.md`.

## Phase 0 Exit
- Scope, governance, reporting, and risk documents are merged.
- `ROADMAP.md` and `PLAN.md` are consistent.

## Phase 1 Exit
- Scaffold directories exist and are documented.
- CI runs lint, format check, type check, tests, and smoke gate.
- Dev environment is reproducible via `environment.yml`.

## Phase 2 Exit
- Core contracts are implemented and runtime-checkable.
- Adapter/schema validation utilities exist and are tested.
- Plugin registry paths for environment and experiment factories exist.

## Phase 3 Exit
- Hierarchical config composition (`extends`, layers, overrides) works.
- Resolved config and immutable manifest are emitted per run.
- Repro smoke protocol exists and passes in CI for event-trace stability.

## Phase 4 Exit
- Metric primitives and statistical aggregation are implemented.
- Run and experiment summary artifacts are generated.
- CSV/LaTeX summary exports are supported and tested.

## Phases 5+ Exit
- Follow checklist items in `PLAN.md` with explicit evidence in PRs.
