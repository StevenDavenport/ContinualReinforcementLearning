# Error Taxonomy and Handling

Base error type: `crlbench.errors.CRLBenchError`

Specialized categories:
- `ConfigurationError`: invalid configs or override payloads.
- `ContractError`: adapter/schema contract violations.
- `OrchestrationError`: invalid runtime orchestration state.

Guidelines:
- Raise the most specific error available.
- Keep exception messages deterministic and actionable.
- CLI should convert expected errors into concise user-facing messages with non-zero exit codes.
