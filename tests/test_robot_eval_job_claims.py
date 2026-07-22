from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from blueprint_pipeline.robot_eval_job_orchestrator import (
    _claim_robot_eval_job_execution,
    _job_id_from_request,
    build_robot_eval_job,
)
from blueprint_pipeline.core.security_controls import SecurityValidationError
from blueprint_pipeline.security_controls import (
    SecurityValidationError as CompatibilitySecurityValidationError,
)


def test_job_id_traversal_and_cli_request_mismatch_fail_before_writes(
    tmp_path: Path,
) -> None:
    with pytest.raises(SecurityValidationError, match="job_id must match"):
        build_robot_eval_job(
            capture_root=tmp_path / "missing-capture",
            job_request={"job_id": "../../escape"},
            job_id="../../escape",
        )
    with pytest.raises(ValueError, match="job_id_argument_request_mismatch"):
        build_robot_eval_job(
            capture_root=tmp_path / "missing-capture",
            job_request={"job_id": "request-job"},
            job_id="cli-job",
        )
    assert not (tmp_path / "escape").exists()
    with pytest.raises(SecurityValidationError, match="job_id must match"):
        _job_id_from_request(
            tmp_path / "request.json",
            {"job_id": "../crafted"},
        )


def test_concurrent_job_claim_has_one_winner_and_never_reuses_namespace(
    tmp_path: Path,
) -> None:
    job_dir = tmp_path / "robot_eval_jobs" / "job-safe"

    def claim() -> dict:
        return _claim_robot_eval_job_execution(
            job_dir=job_dir,
            job_id="job-safe",
            request_fingerprint="a" * 64,
            generated_at="2026-07-09T12:00:00Z",
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(claim) for _ in range(2)]
    successes = [future.result() for future in futures if future.exception() is None]
    failures = [future.exception() for future in futures if future.exception() is not None]

    assert len(successes) == 1
    assert len(failures) == 1
    assert isinstance(failures[0], ValueError)
    assert str(failures[0]) == "robot_eval_job_id_already_claimed"
    attempt_dir = Path(successes[0]["attempt_dir"])
    assert attempt_dir.is_dir()
    assert attempt_dir.parent == job_dir / "attempts"
    assert (attempt_dir / "attempt_claim.json").is_file()

    with pytest.raises(
        ValueError, match="robot_eval_job_id_request_fingerprint_mismatch"
    ):
        _claim_robot_eval_job_execution(
            job_dir=job_dir,
            job_id="job-safe",
            request_fingerprint="b" * 64,
            generated_at="2026-07-09T12:00:01Z",
        )


def test_security_validation_error_compatibility_import_is_canonical() -> None:
    assert CompatibilitySecurityValidationError is SecurityValidationError
