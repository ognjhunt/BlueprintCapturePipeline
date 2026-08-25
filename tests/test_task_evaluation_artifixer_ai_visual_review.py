from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_evaluation_artifixer_ai_visual_review import (
    EXECUTION_SCHEMA_VERSION,
    TaskEvaluationArtifixerAIVisualReviewError,
    materialize_artifixer_ai_visual_review_rights,
    seal_artifixer_ai_visual_review,
    validate_artifixer_ai_visual_review_rights,
)


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _inputs(tmp_path: Path) -> tuple[Path, Path]:
    frames = []
    inventory = []
    for index in range(2):
        frame = tmp_path / f"frame-{index}.png"
        frame.write_bytes(b"png" + bytes([index]))
        record = {
            "path": str(frame),
            "size_bytes": frame.stat().st_size,
            "sha256": _sha256(frame),
        }
        frames.append(
            {
                "frame_index": index,
                "camera_id": f"camera-{index}",
                "final_frame": record,
                "outside_support_changed_pixels": 0,
            }
        )
        inventory.append(
            {
                "frame_index": index,
                "camera_id": f"camera-{index}",
                "sha256": record["sha256"],
                "size_bytes": record["size_bytes"],
            }
        )
    final = {
        "schema_version": "public_scene_artifixer3d_final_composite.v1",
        "status": "final_composite_materialized_pending_human_multiview_review",
        "tasks": [
            {
                "task_id": "remove-instance-104",
                "physical_camera_count": 2,
                "frames": frames,
            }
        ],
        "outside_support_invariance_proven": True,
        "outside_support_changed_pixels_total": 0,
        "semantic_object_absence_review_passed": False,
        "multiview_consistency_review_passed": False,
        "appearance_repair_qualified": False,
        "generated_output_is_capture_or_physical_evidence": False,
        "receipt_digest": "",
    }
    final["receipt_digest"] = canonical_digest(final, digest_field="receipt_digest")
    final_path = tmp_path / "final.json"
    final_path.write_text(json.dumps(final), encoding="utf-8")
    execution = {
        "schema_version": EXECUTION_SCHEMA_VERSION,
        "status": "completed",
        "decision": "accepted",
        "publisher_instance_id": "104",
        "task_id": "remove-instance-104",
        "final_composite_receipt_digest": final["receipt_digest"],
        "review_frame_inventory_digest": canonical_digest({"frames": inventory}),
        "provider_called": True,
        "response_store": False,
        "tracing_disabled": True,
        "raw_secret_values_recorded": False,
        "semantic_object_absence_review_passed": True,
        "multiview_consistency_review_passed": True,
        "rights_attestation_digest": "sha256:" + "a" * 64,
        "reviewer": {
            "kind": "ai",
            "identity": "artifixer-independent-vision-reviewer-v1",
            "runtime": "openai_agents_sdk",
            "model": "gpt-5.6-terra",
        },
        "frames": [
            {
                "task_id": "remove-instance-104",
                "camera_id": row["camera_id"],
                "frame_sha256": row["sha256"],
                "source_object_absent": True,
                "repair_is_locally_plausible": True,
                "preserves_non_target_content": True,
                "decision": "accepted",
                "rationale": "The mug is absent and the table continues naturally.",
            }
            for row in inventory
        ],
        "execution_digest": "",
    }
    execution["execution_digest"] = canonical_digest(
        execution, digest_field="execution_digest"
    )
    execution_path = tmp_path / "execution.json"
    execution_path.write_text(json.dumps(execution), encoding="utf-8")
    return final_path, execution_path


def test_seals_exact_frame_ai_review_without_physics_claim(tmp_path: Path) -> None:
    final, execution = _inputs(tmp_path)
    receipt = seal_artifixer_ai_visual_review(
        final_composite_receipt_path=final,
        review_execution_receipt_path=execution,
        publisher_instance_id="104",
        minimum_review_frames=2,
        output_path=tmp_path / "review.json",
    )

    assert receipt["status"] == "accepted"
    assert receipt["all_review_frames_digest_bound"] is True
    assert receipt["physics_or_collision_authority_granted"] is False


def test_rejects_changed_frame_after_ai_review(tmp_path: Path) -> None:
    final, execution = _inputs(tmp_path)
    (tmp_path / "frame-1.png").write_bytes(b"changed")

    with pytest.raises(
        TaskEvaluationArtifixerAIVisualReviewError,
        match="artifixer_ai_review_frame_bytes_invalid",
    ):
        seal_artifixer_ai_visual_review(
            final_composite_receipt_path=final,
            review_execution_receipt_path=execution,
            publisher_instance_id="104",
            minimum_review_frames=2,
            output_path=tmp_path / "review.json",
        )


def test_rejects_review_missing_one_camera(tmp_path: Path) -> None:
    final, execution = _inputs(tmp_path)
    value = json.loads(execution.read_text(encoding="utf-8"))
    value["frames"].pop()
    value["execution_digest"] = canonical_digest(
        value, digest_field="execution_digest"
    )
    execution.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(
        TaskEvaluationArtifixerAIVisualReviewError,
        match="artifixer_ai_review_execution_not_acceptable",
    ):
        seal_artifixer_ai_visual_review(
            final_composite_receipt_path=final,
            review_execution_receipt_path=execution,
            publisher_instance_id="104",
            minimum_review_frames=2,
            output_path=tmp_path / "review.json",
        )


def test_rights_scope_is_human_issued_before_generated_frames_exist(
    tmp_path: Path,
) -> None:
    path = tmp_path / "rights.json"
    value = materialize_artifixer_ai_visual_review_rights(
        configuration_run_id="configure-scene-839873-v1",
        source_scene_rights_admission_digest="sha256:" + "b" * 64,
        accepted_by="project-owner",
        accepted_on="2026-08-25",
        human_authority_reference="website-configuration-consent-v1",
        output_path=path,
    )

    assert value["raw_interiorgs_bytes_disclosure_authorized"] is False
    assert value["generated_frame_bytes_unknown_until_production_execution"] is True
    assert value["exact_frame_inventory_bound_by_execution_receipt"] is True
    _, reopened = validate_artifixer_ai_visual_review_rights(
        rights_attestation_path=path,
        configuration_run_id="configure-scene-839873-v1",
    )
    assert reopened["attestation_digest"] == value["attestation_digest"]
