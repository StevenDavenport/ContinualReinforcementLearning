# Phase 0 Contract (Scope and Governance)

This document operationalizes the Phase 0 checklist from `PLAN.md`.

## Scope Freeze (v1)
- Five experiments.
- Two tracks per experiment: Toy and Robotics.
- One default + one alternative environment option per track.

## Comparability Governance
- Changes to metric definitions require benchmark version bump.
- Changes to environment default task streams require benchmark version bump.
- Additive changes (new optional env options) are minor-version changes.

## Statistical Reporting Minimum
- Report mean and standard deviation across seeds.
- Report confidence intervals for headline comparisons.
- Any omitted seed must include explicit failure reason in manifest.

## Reproducibility Minimum
- Every run must store resolved config and immutable manifest.
- Every table/plot used in publication must map to run ids.
- Determinism expectations must be documented per simulator family.

## Linked Phase 0 Control Docs
- Acceptance criteria: `docs/PHASE_ACCEPTANCE.md`
- Ownership and review policy: `docs/OWNERSHIP.md`
- Versioning and release policy: `docs/VERSIONING.md`
- Risk register: `docs/RISK_REGISTER.md`
