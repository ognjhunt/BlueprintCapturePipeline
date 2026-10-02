"""Retired direct AWS operation must never use even an injected provider."""
from blueprint_pipeline.reconstruction_aws_windows_operation import run_reconstruction_aws_windows_operation

class ForbiddenProvider:
    def __getattr__(self, key):
        raise AssertionError("AWS provider accessed")


def test_retired_operation_refuses_without_provider_or_authority_access(tmp_path):
    result = run_reconstruction_aws_windows_operation(provider=ForbiddenProvider(), job_dir=tmp_path / "must-not-create")
    assert result["status"] == "blocked"
    assert result["blockers"] == ["aws_provider_integration_removed"]
    assert result["provider_mutations_performed"] == 0
    assert result["provider_absence_confirmed"] is False
    assert not (tmp_path / "must-not-create").exists()
