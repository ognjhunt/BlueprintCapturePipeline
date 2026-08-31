from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_evaluation_result_delivery import (
    TaskEvaluationResultDeliveryError,
    materialize_policy_canary_result_delivery,
    resolve_task_evaluation_result_artifact,
)


def _sha(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _closure(path: Path, *, flag: str) -> dict[str, object]:
    path.write_text(json.dumps({"status": "completed"}) + "\n", encoding="utf-8")
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": _sha(path),
        flag: True,
    }


def _result(evidence: Path) -> dict[str, object]:
    telemetry = evidence / "policy_canary_telemetry.jsonl"
    telemetry.write_text('{"episode":"one"}\n', encoding="utf-8")
    value: dict[str, object] = {
        "schema_version": "native_task_arena_policy_canary_session_result.v1",
        "status": "completed_unqualified",
        "run_kind": "internal_policy_canary",
        "claim_ceiling": "diagnostic_policy_execution",
        "episodes": [
            {
                "candidate_id": "pi05_droid",
                "cell_id": "quick-cell-0",
                "seed": 3100,
                "status": "completed",
                "candidate_policy_queried": True,
                "actions_reached_robot": True,
                "arm_moved": True,
                "policy_outcome_interpretable": True,
            }
        ],
        "artifact_inventory": [
            {
                "role": "indexed_episode_telemetry",
                "relative_path": telemetry.name,
                "media_type": "application/x-ndjson",
                "size_bytes": telemetry.stat().st_size,
                "sha256": _sha(telemetry),
            }
        ],
        "result_digest": "",
    }
    value["result_digest"] = canonical_digest(value, digest_field="result_digest")
    return value


def test_canary_delivery_seals_downloads_and_terminal_closure(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    result = _result(evidence)
    closure = {
        "billing": _closure(
            tmp_path / "billing.json", flag="official_billing_sealed"
        ),
        "teardown": _closure(
            tmp_path / "teardown.json", flag="teardown_completed"
        ),
        "provider_zero": _closure(
            tmp_path / "provider-zero.json", flag="provider_zero_verified"
        ),
    }

    delivery = materialize_policy_canary_result_delivery(
        run_root=tmp_path,
        run_id="scene-839873-canary-1",
        result_status="completed_unqualified",
        session_result=result,
        evidence_root=evidence,
        closure_records=closure,
    )

    assert delivery["schema_version"] == "task_evaluation_result_delivery.v2"
    assert delivery["summary"]["learned_policy_rollout_count"] == 20
    assert delivery["closure"]["provider_zero"]["provider_zero_verified"] is True
    report = delivery["report"]["machine_readable_report"]
    path, record = resolve_task_evaluation_result_artifact(
        run_root=tmp_path,
        run_id="scene-839873-canary-1",
        artifact_id=report["artifact_id"],
    )
    assert path.name == "policy_canary_full_report.json"
    assert record["sha256"] == report["digest"]


def test_canary_delivery_refuses_estimated_cost_as_official_billing(
    tmp_path: Path,
) -> None:
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    result = _result(evidence)
    billing = _closure(tmp_path / "billing.json", flag="estimated_cost_only")

    with pytest.raises(
        TaskEvaluationResultDeliveryError,
        match="policy_canary_billing_receipt_missing",
    ):
        materialize_policy_canary_result_delivery(
            run_root=tmp_path,
            run_id="scene-839873-canary-1",
            result_status="completed_unqualified",
            session_result=result,
            evidence_root=evidence,
            closure_records={
                "billing": billing,
                "teardown": _closure(
                    tmp_path / "teardown.json", flag="teardown_completed"
                ),
                "provider_zero": _closure(
                    tmp_path / "provider-zero.json",
                    flag="provider_zero_verified",
                ),
            },
        )
