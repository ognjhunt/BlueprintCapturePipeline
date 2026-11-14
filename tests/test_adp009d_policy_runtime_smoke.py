from __future__ import annotations

import json
from pathlib import Path


from blueprint_pipeline.adp009d_policy_provisioning import (
    build_provisioning_script,
)
from blueprint_pipeline.adp009d_policy_runtime_smoke_worker import (
    seal_policy_runtime_smoke,
)


COMMIT = "1" * 40
MEASUREMENT_PATH = (
    Path(__file__).parents[1]
    / "docs/arm_decision_proof_v1/evidence/"
    "adp009d_scene_840920_policy_runtime_smokes_2026-08-24.json"
)


def test_normal_episode_provisioning_keeps_server_running() -> None:
    assert "--stop-after-handshake" not in build_provisioning_script("pi05_droid")


def _server_receipt(**updates: object) -> dict[str, object]:
    value: dict[str, object] = {
        "candidate_id": "pi05_droid",
        "status": "ready",
        "handshake_completed": True,
        "candidate_policy_queried": False,
        "candidate_inference_performed": False,
        "policy_state_advanced": False,
        "server_stopped_after_handshake": True,
        "server_pid": 999_999_999,
        "transport": "openpi_websocket",
    }
    value.update(updates)
    return value


def test_worker_seals_zero_query_readiness_without_task_claim(tmp_path: Path) -> None:
    server = tmp_path / "server.json"
    server.write_text(json.dumps(_server_receipt()), encoding="utf-8")

    result = seal_policy_runtime_smoke(
        candidate_id="pi05_droid",
        server_receipt_path=server,
        provisioning_exit_code=0,
    )

    assert result["status"] == "completed"
    assert result["synthetic_observation_query_count"] == 0
    assert result["candidate_policy_queried"] is False
    assert result["policy_state_advanced"] is False
    assert result["candidate_outcomes_accessed"] is False
    assert result["claim_boundary"]["actions_executed"] is False
    assert result["claim_boundary"]["task_scene_loaded"] is False
    assert result["claim_boundary"]["task_success_claimed"] is False


def test_worker_refuses_outcome_fields(tmp_path: Path) -> None:
    server = tmp_path / "server.json"
    server.write_text(
        json.dumps(_server_receipt(task_success=False)), encoding="utf-8"
    )

    result = seal_policy_runtime_smoke(
        candidate_id="pi05_droid",
        server_receipt_path=server,
        provisioning_exit_code=0,
    )

    assert result["status"] == "blocked"
    assert "policy_runtime_smoke_outcome_field_forbidden" in result["blockers"]

