"""Project-wide error taxonomy."""


class CRLBenchError(Exception):
    """Base error for benchmark runtime and tooling failures."""


class ContractError(CRLBenchError):
    """Raised when integration code violates benchmark contracts."""


class ConfigurationError(CRLBenchError):
    """Raised when configuration payloads are invalid."""


class OrchestrationError(CRLBenchError):
    """Raised when run orchestration preconditions fail."""


class AgentIntegrationError(CRLBenchError):
    """Raised when external agent loading or validation fails."""
