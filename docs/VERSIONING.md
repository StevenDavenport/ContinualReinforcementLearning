# Versioning and Release Policy

## Benchmark Versioning
- Use semantic versioning: `MAJOR.MINOR.PATCH`.

## Compatibility Rules
- `PATCH`: bug fixes, docs, and non-behavioral internal improvements.
- `MINOR`: additive functionality that preserves metric/protocol comparability.
- `MAJOR`: changes that alter metric definitions, default protocols, or comparability.

## Artifact Schema Versioning
- `schema_version` field is required in all structured artifact JSON/JSONL records.
- Backward compatibility must hold within a minor release line.

## Release Cadence
- `v0.x`: internal stabilization.
- `v1.0`: first public benchmark release once baseline and experiment coverage gates pass.
