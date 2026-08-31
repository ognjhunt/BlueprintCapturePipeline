from __future__ import annotations

import json

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_evaluation_policy_canary_result import (
    validate_policy_canary_result,
)
from blueprint_pipeline.task_evaluation_run_webapp_sync import (
    sync_policy_canary_preprovider_blocked_to_webapp,
    sync_task_evaluation_policy_canary_to_webapp,
)


def _artifact(character: str, artifact_id: str) -> dict[str, object]:
    return {
        "digest": "sha256:" + character * 64,
        "size_bytes": 10,
        "artifact_id": artifact_id,
    }


def _projection() -> tuple[dict[str, object], dict[str, object]]:
    delivery: dict[str, object] = {
        "schema_version": "task_evaluation_result_delivery.v2",
        "run_id": "scene-839873-canary-1",
        "result_status": "blocked",
        "claim_ceiling": "diagnostic_policy_execution",
        "delivery_digest": "",
    }
    delivery["delivery_digest"] = canonical_digest(
        delivery, digest_field="delivery_digest"
    )
    result: dict[str, object] = {
        "schema_version": "task_evaluation_policy_canary_result_projection.v1",
        "run_id": "scene-839873-canary-1",
        "request_digest": "sha256:" + "1" * 64,
        "configuration_digest": "sha256:" + "2" * 64,
        "result_delivery_digest": delivery["delivery_digest"],
        "run_kind": "internal_policy_canary",
        "claim_ceiling": "diagnostic_policy_execution",
        "scene_controls_status": "configured_controls_pending",
        "result_status": "blocked",
        "warning": "Controls pending — results are unqualified.",
        "counts": {
            "policy_count": 2,
            "episodes_per_policy": 10,
            "learned_policy_rollout_count": 20,
            "completed_learned_policy_rollout_count": 0,
            "diagnostic_control_rollout_count": 20,
            "completed_diagnostic_control_rollout_count": 0,
        },
        "candidate_ids": ["pi05_droid", "groot_n17_droid"],
        "candidate_results": [
            {
                "candidate_id": candidate,
                "episodes_completed": 0,
                "interpretable_episode_count": 0,
                "actions_delivered_episode_count": 0,
                "metrics": {},
                "failure_counts": {"pre_provider_blocked": 10},
            }
            for candidate in ("pi05_droid", "groot_n17_droid")
        ],
        "episodes": [],
        "comparison": {
            "matched_cell_count": 0,
            "winner_declared": False,
            "official_ranking_contribution": False,
        },
        "report": {
            "result_digest": "sha256:" + "3" * 64,
            "permanent_result_path": "/internal/task-evaluation-runs/scene-839873-canary-1",
            "machine_readable_report": _artifact("4", "full-report"),
            "evidence_manifest": _artifact("5", "evidence-manifest"),
        },
        "closure": {
            "billing": _artifact("6", "billing"),
            "teardown": _artifact("7", "teardown"),
            "provider_zero": {
                **_artifact("8", "provider-zero"),
                "provider_zero_verified": True,
            },
        },
        "notification_delivery": {
            "terminal_state": "blocked",
            "status": "pending",
            "attempts": 0,
            "provider": "website_terminal_handler",
            "message_id": None,
            "delivered_at": None,
            "run_result_digest": "sha256:" + "3" * 64,
        },
        "blockers": ["provider_capacity_unavailable"],
        "projection_digest": "",
    }
    result["projection_digest"] = canonical_digest(
        result, digest_field="projection_digest"
    )
    return delivery, validate_policy_canary_result(result)


def test_canary_sync_requires_website_notification_receipt(monkeypatch) -> None:
    delivery, result = _projection()

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self) -> bytes:
            return json.dumps(
                {
                    "schema_version": (
                        "capture_task_evaluation_policy_canary_publication_receipt.v1"
                    ),
                    "status": "blocked",
                    "already_exists": False,
                    "capture_session_id": "capture-839873",
                    "intake_id": "intake-839873",
                    "run_id": result["run_id"],
                    "request_digest": result["request_digest"],
                    "configuration_digest": result["configuration_digest"],
                    "result_delivery_digest": delivery["delivery_digest"],
                    "policy_canary_projection_digest": result["projection_digest"],
                    "notification_delivery": {
                        "terminal_state": "blocked",
                        "status": "delivered",
                        "attempts": 1,
                        "provider": "resend",
                        "message_id": "message-1",
                        "delivered_at": "2026-08-31T14:00:00Z",
                        "run_result_digest": result["projection_digest"],
                    },
                }
            ).encode("utf-8")

    monkeypatch.setattr(
        "blueprint_pipeline.task_evaluation_run_webapp_sync.urllib_request.urlopen",
        lambda *_args, **_kwargs: Response(),
    )

    synced = sync_task_evaluation_policy_canary_to_webapp(
        capture_session_id="capture-839873",
        intake_id="intake-839873",
        run_id=str(result["run_id"]),
        request_digest=str(result["request_digest"]),
        configuration_digest=str(result["configuration_digest"]),
        result_status="blocked",
        result_delivery=delivery,
        policy_canary_result=result,
        endpoint_url="https://webapp.example/api/internal/pipeline/task-evaluation-runs",
        token="sync-secret",
        max_attempts=1,
    )

    assert synced["status"] == "succeeded"
    assert synced["notification_delivery"]["status"] == "delivered"
    assert "sync-secret" not in json.dumps(synced)


def test_preprovider_blocked_sync_requires_terminal_email_readback(monkeypatch) -> None:
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self) -> bytes:
            request = captured["request"]
            payload = json.loads(request.data)
            return json.dumps(
                {
                    "schema_version": (
                        "capture_task_evaluation_policy_canary_blocked_receipt.v1"
                    ),
                    "status": "blocked",
                    "activation_id": payload["activation_id"],
                    "request_digest": payload["request_digest"],
                    "payload_digest": payload["payload_digest"],
                    "notification_delivery": {
                        "terminal_state": "blocked",
                        "status": "delivered",
                        "attempts": 1,
                        "provider": "resend",
                        "message_id": "message-blocked-1",
                        "delivered_at": "2026-08-31T14:00:00Z",
                        "run_result_digest": payload["payload_digest"],
                    },
                }
            ).encode("utf-8")

    captured = {}

    def open_response(request, **_kwargs):
        captured["request"] = request
        return Response()

    monkeypatch.setattr(
        "blueprint_pipeline.task_evaluation_run_webapp_sync.urllib_request.urlopen",
        open_response,
    )
    synced = sync_policy_canary_preprovider_blocked_to_webapp(
        activation_id="activation-839873",
        capture_session_id="capture-839873",
        intake_id="intake-839873",
        request_digest="sha256:" + "1" * 64,
        blockers=["policy_canary_setup_invalid"],
        endpoint_url="https://webapp.example/api/internal/pipeline/task-evaluation-runs",
        token="sync-secret",
    )

    assert synced["status"] == "succeeded"
    assert synced["notification_delivery"]["status"] == "delivered"
