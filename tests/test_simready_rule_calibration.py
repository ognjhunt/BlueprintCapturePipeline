from __future__ import annotations

import json
from pathlib import Path

import jsonschema

from blueprint_pipeline.simready_rule_calibration import build_simready_rule_calibration


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _result(status: str, findings: list[dict[str, object]]) -> dict[str, object]:
    identity = {"validator_version": "1.0", "profile": "robotics", "profile_version": "1"}
    return {
        "schema_version": "external_simready_validation_result.v1",
        "status": status,
        "repeatability": {"stable_normalized_results": True},
        "requested_identity": identity,
        "reported_identity": identity,
        "normalized_findings": findings,
    }


def test_calibration_requires_expert_review_and_authorizes_only_perfect_rule(
    tmp_path: Path,
) -> None:
    valid = tmp_path / "valid.json"
    invalid = tmp_path / "invalid.json"
    _write(valid, _result("passed_advisory", []))
    _write(invalid, _result("validation_failed", [{"rule_id": "USD.DEFAULT", "severity": "error"}]))
    review = {"status": "approved", "reviewer_id": "expert-1", "reviewed_at": "2026-07-21"}
    manifest = tmp_path / "manifest.json"
    _write(
        manifest,
        {
            "schema_version": "simready_rule_calibration_manifest.v1",
            "frozen": True,
            "cases": [
                {
                    "case_id": "valid",
                    "result_path": valid.name,
                    "expected_validation_status": "passed_advisory",
                    "expert_review": {**review, "expected_error_rule_ids": []},
                },
                {
                    "case_id": "invalid",
                    "result_path": invalid.name,
                    "expected_validation_status": "validation_failed",
                    "expert_review": {**review, "expected_error_rule_ids": ["USD.DEFAULT"]},
                },
            ],
        },
    )
    output = tmp_path / "calibration.json"
    result = build_simready_rule_calibration(
        manifest_path=manifest,
        evidence_root=tmp_path,
        output_path=output,
        authorize_rule_ids=["USD.DEFAULT"],
        human_promotion_approval_id="approval-1",
    )
    assert result["status"] == "completed"
    assert result["authorized_blocking_rule_ids"] == ["USD.DEFAULT"]
    assert result["claim_boundary"]["validator_pass_is_simulator_or_task_proof"] is False
    schema = json.loads(
        (
            Path(__file__).parents[1] / "docs/schemas/simready_rule_calibration.schema.json"
        ).read_text()
    )
    jsonschema.Draft202012Validator(schema).validate(result)


def test_calibration_fails_closed_without_human_promotion_approval(tmp_path: Path) -> None:
    result_path = tmp_path / "result.json"
    _write(result_path, _result("validation_failed", [{"rule_id": "R", "severity": "error"}]))
    manifest = tmp_path / "manifest.json"
    _write(
        manifest,
        {
            "schema_version": "simready_rule_calibration_manifest.v1",
            "frozen": True,
            "cases": [
                {
                    "case_id": "only",
                    "result_path": result_path.name,
                    "expected_validation_status": "validation_failed",
                    "expert_review": {
                        "status": "approved",
                        "reviewer_id": "x",
                        "reviewed_at": "2026-07-21",
                        "expected_error_rule_ids": ["R"],
                    },
                }
            ],
        },
    )
    result = build_simready_rule_calibration(
        manifest_path=manifest,
        evidence_root=tmp_path,
        output_path=tmp_path / "out.json",
        authorize_rule_ids=["R"],
    )
    assert result["status"] == "blocked"
    assert "simready_rule_promotion_human_approval_id_required" in result["blockers"]
