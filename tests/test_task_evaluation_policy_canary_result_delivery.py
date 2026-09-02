from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from blueprint_pipeline.decision_evidence_contracts import (
    canonical_digest,
    cross_runtime_canonical_digest,
)
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
    artifacts: dict[str, dict[str, object]] = {}
    inventory = [
        {
            "role": "indexed_episode_telemetry",
            "relative_path": telemetry.name,
            "media_type": "application/x-ndjson",
            "size_bytes": telemetry.stat().st_size,
            "sha256": _sha(telemetry),
        }
    ]
    for role, suffix, payload in (
        ("reset_state", ".json", {"reset": "exact"}),
        ("lossless_frame_manifest", ".json", {"frames": ["external", "wrist"]}),
        ("policy_query_receipt", ".json", {"candidate_policy_queried": True}),
        ("action_sequence", ".json", [{"step_index": 1, "target": [0.1] * 7}]),
        ("action_delivery_readback", ".json", {"actions_reached_robot": True}),
        (
            "state_trace",
            ".json",
            {
                "joint_states": [
                    {"step_index": 0, "joint_positions_rad": [0.0] * 7},
                    {"step_index": 1, "joint_positions_rad": [0.1] * 7},
                ],
                "task_state_samples": [],
            },
        ),
        (
            "contact_force_trace",
            ".json",
            {"samples": [{"step_index": 1, "finger_contact_forces_n": [1.0, 1.1]}]},
        ),
        (
            "task_object_trajectory",
            ".json",
            {"samples": [{"step_index": 1, "task_object_pose_world": [0.1] * 7}]},
        ),
        ("score_receipt", ".json", {"task_succeeded": True}),
    ):
        path = evidence / f"episode-one.{role}{suffix}"
        path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
        record = {
            "role": role,
            "relative_path": path.name,
            "media_type": "application/json",
            "size_bytes": path.stat().st_size,
            "sha256": _sha(path),
        }
        artifacts["frame_manifest" if role == "lossless_frame_manifest" else role] = record
        inventory.append(record)
    review_video = evidence / "episode-one.external.mp4"
    review_video.write_bytes(b"review-video")
    review_record = {
        "role": "review_video",
        "relative_path": review_video.name,
        "media_type": "video/mp4",
        "size_bytes": review_video.stat().st_size,
        "sha256": _sha(review_video),
    }
    inventory.append(review_record)
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
                "checkpoint_digest": "sha256:" + "1" * 64,
                "runtime_identity_digest": "sha256:" + "2" * 64,
                "reset_state_digest": artifacts["reset_state"]["sha256"],
                "scene_revision_digest": "sha256:" + "3" * 64,
                "container_identity_digest": "sha256:" + "4" * 64,
                "scoring_version_digest": "sha256:" + "5" * 64,
                "family": "canonical_anchor",
                "evidence_artifacts": artifacts,
                "telemetry": {
                    "started_at_unix_ns": 1_700_000_000_000_000_000,
                    "completed_at_unix_ns": 1_700_000_001_000_000_000,
                    "wall_time_ns": 1_000_000_000,
                },
                "episode": {
                    "episode_id": "episode-one",
                    "observation_adapter_schema_version": "droid_two_camera_robot_state_v1",
                    "action_space": "droid_absolute_joint_position_v1",
                    "commanded_actions": [
                        {"step_index": 1, "target_joint_positions_rad": [0.1] * 7}
                    ],
                    "state_trace": json.loads(
                        (evidence / "episode-one.state_trace.json").read_text()
                    ),
                    "contact_force_evidence": json.loads(
                        (evidence / "episode-one.contact_force_trace.json").read_text()
                    ),
                    "task_object_trajectory": json.loads(
                        (evidence / "episode-one.task_object_trajectory.json").read_text()
                    ),
                    "score": {
                        "status": "scored",
                        "task_succeeded": True,
                        "outcome": "placed",
                        "outcome_rank": 5,
                        "measurements": {"final_horizontal_distance_to_destination_m": 0.01},
                    },
                    "visual_evidence": {
                        "videos": {
                            "external": {
                                "sha256": review_record["sha256"],
                                "size_bytes": review_record["size_bytes"],
                            }
                        }
                    },
                },
            }
        ],
        "matrix_digest": "sha256:" + "6" * 64,
        "scene_revision_digest": "sha256:" + "7" * 64,
        "runtime_container_digest": "sha256:" + "8" * 64,
        "official_total_usd": 0.379,
        "started_at_iso": "2026-09-02T04:03:45+00:00",
        "completed_at_iso": "2026-09-02T04:49:54+00:00",
        "duration_seconds": 2769.0,
        "provider": "vast",
        "provider_instance_ids": [49_609_705],
        "artifact_inventory": inventory,
        "result_digest": "",
    }
    value["result_digest"] = canonical_digest(value, digest_field="result_digest")
    return value


def test_canary_delivery_seals_downloads_and_terminal_closure(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    result = _result(evidence)
    closure = {
        "billing": _closure(tmp_path / "billing.json", flag="official_billing_sealed"),
        "teardown": _closure(tmp_path / "teardown.json", flag="teardown_completed"),
        "provider_zero": _closure(tmp_path / "provider-zero.json", flag="provider_zero_verified"),
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
    assert delivery["summary"]["successful_episode_count"] == 1
    assert delivery["candidate_results"][0]["success_rate"] == 1.0
    assert delivery["matrix_digest"] == "sha256:" + "6" * 64
    assert delivery["reproducibility"]["scene_revision_digest"] == (
        "sha256:" + "7" * 64
    )
    assert delivery["reproducibility"]["runtime_container_digest"] == (
        "sha256:" + "8" * 64
    )
    assert delivery["reproducibility"]["official_total_usd"] == 0.379
    assert delivery["reproducibility"]["duration_seconds"] == 2769.0
    assert delivery["reproducibility"]["provider_instance_ids"] == [49_609_705]
    assert delivery["closure"]["provider_zero"]["provider_zero_verified"] is True
    assert delivery["delivery_digest"] == cross_runtime_canonical_digest(
        delivery, digest_field="delivery_digest"
    )
    assert delivery["delivery_digest"] != canonical_digest(delivery, digest_field="delivery_digest")
    episode = delivery["episodes"][0]
    assert episode["episode_kind"] == "learned_candidate"
    assert episode["evidence"]["lossless_policy_inputs"]["access_mode"] == ("authenticated_ticket")
    assert episode["evidence"]["videos"]["external"]["content_type"] == "video/mp4"
    assert episode["traces"]["state"]["role"] == "state_trace"
    assert episode["timeline"][-1]["scoring_state"] == "placed"
    assert {artifact["role"] for artifact in delivery["artifacts"]}.issuperset(
        {
            "summary_csv",
            "episode_csv",
            "full_json_report",
            "evidence_manifest",
            "lossless_policy_inputs",
            "review_video",
            "returned_action_sequence",
            "state_trace",
            "contact_force_trace",
            "task_object_trajectory",
        }
    )
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
                "teardown": _closure(tmp_path / "teardown.json", flag="teardown_completed"),
                "provider_zero": _closure(
                    tmp_path / "provider-zero.json",
                    flag="provider_zero_verified",
                ),
            },
        )
