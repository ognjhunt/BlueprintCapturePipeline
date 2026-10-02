"""Compatibility entrypoint for the retired AWS Windows execution adapter."""
from typing import Any
SCHEMA_VERSION = "reconstruction_aws_windows_operation_execution.v1"
PROVIDER = "aws"

class ReconstructionAwsWindowsError(RuntimeError):
    pass


def run_reconstruction_aws_windows_operation(**kwargs: Any) -> dict[str, Any]:
    """Refuse without consulting even an injected provider or admission grant."""
    return {"schema_version": SCHEMA_VERSION, "provider": PROVIDER,
            "status": "blocked", "blockers": ["aws_provider_integration_removed"],
            "provider_mutations_performed": 0, "provider_absence_confirmed": False}
