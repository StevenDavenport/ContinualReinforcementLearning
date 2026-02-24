# Risk Register

## R1: Simulator Installation Fragility
- Impact: high
- Likelihood: medium
- Mitigation:
  - maintain environment-specific setup docs,
  - split heavy-dependency CI jobs from core CI.

## R2: Metric Definition Drift
- Impact: high
- Likelihood: medium
- Mitigation:
  - centralize metric implementations under `crlbench.metrics`,
  - require governance review for metric changes,
  - enforce synthetic-trace regression tests.

## R3: Reproducibility Regressions
- Impact: high
- Likelihood: medium
- Mitigation:
  - manifest fingerprints (git/env/lockfiles),
  - repro-smoke gate in CI,
  - deterministic-mode and sub-seed policy.

## R4: Artifact Schema Breakage
- Impact: medium
- Likelihood: medium
- Mitigation:
  - versioned schemas,
  - artifact contract doc,
  - reader/writer tests.

## R5: Compute Budget Explosion
- Impact: medium
- Likelihood: medium
- Mitigation:
  - smoke/dev/full budget tiers,
  - CI limited to smoke tier,
  - scheduled heavy jobs for robotics options.
