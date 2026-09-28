"""The source-calibration producer's exact result is visible at the Vast boundary."""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from blueprint_pipeline.decision_evidence_contracts import cross_runtime_canonical_digest
from blueprint_pipeline.source_calibration_render_return import RESULT_SCHEMA, ROLES
from blueprint_pipeline.vast_provider_adapter import _inspect_provider_runtime_output_zip


def _write_output(path: Path, *, status: str = "completed", **changes) -> None:
    result = {
        "schema_version": RESULT_SCHEMA, "status": status, "render_scope": "source_calibration",
        "blueprint_commit": "a" * 40, "preparation_digest": "sha256:" + "b" * 64,
        "candidate_policy_queried": False, "paid_inference_performed": False,
        "provider_mutations_performed": 0, "digest_canonicalization": "rfc8785",
        "render_groups": [{"role": role, "manifest": {
            "relative_path": f"{role}/sealed_camera_render_manifest.v1.json",
            "sha256": "sha256:" + "c" * 64, "size_bytes": 123,
        }} for role in ROLES],
        "blockers": ["camera_render_failed"] if status == "blocked" else [],
        **changes,
    }
    result["result_digest"] = cross_runtime_canonical_digest(result, digest_field="result_digest")
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(RESULT_SCHEMA + ".json", json.dumps(result))


@pytest.mark.parametrize("status", ["completed", "blocked"])
def test_vast_discovers_source_calibration_terminal_result(tmp_path: Path, status: str) -> None:
    output = tmp_path / "source-output.zip"
    _write_output(output, status=status)
    result = _inspect_provider_runtime_output_zip(output, expected_video_count=0)
    assert result["runtime_result_present"] is True
    assert result["runtime_result_member"] == RESULT_SCHEMA + ".json"
    assert result["runtime_result_status"] == status
    assert result["runtime_result_blockers"] == (["camera_render_failed"] if status == "blocked" else [])
    assert result["runtime_result"]["task_success"] is None
    assert result["video_smoke_proven"] is False
    assert result["json_parse_errors"] == []


@pytest.mark.parametrize("changes", [{"schema_version": "foreign.v1"}, {"render_scope": "policy_execution"}])
def test_source_result_filename_does_not_admit_a_foreign_contract(tmp_path: Path, changes: dict) -> None:
    output = tmp_path / "wrong-identity.zip"
    _write_output(output, **changes)
    result = _inspect_provider_runtime_output_zip(output, expected_video_count=0)
    assert result["runtime_result_present"] is False
    assert result["json_parse_errors"] == [RESULT_SCHEMA + ".json:SourceCalibrationResultIdentityMismatch"]
