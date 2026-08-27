from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import pytest
from PIL import Image

from blueprint_pipeline import task_evaluation_artifixer_ai_visual_review as module
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_evaluation_artifixer_ai_visual_review import (
    DUAL_TARGET_REVIEW_SCHEMA_VERSION,
    EXECUTION_SCHEMA_VERSION,
    TaskEvaluationArtifixerAIVisualReviewError,
    build_artifixer_ai_visual_review_input,
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
        "all_frames_upright": True,
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
                "orientation_is_upright": True,
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
    execution["execution_digest"] = canonical_digest(execution, digest_field="execution_digest")
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
    assert receipt["all_frames_upright"] is True
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
    value["execution_digest"] = canonical_digest(value, digest_field="execution_digest")
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


def test_rejects_review_with_upside_down_frame(tmp_path: Path) -> None:
    final, execution = _inputs(tmp_path)
    value = json.loads(execution.read_text(encoding="utf-8"))
    value["frames"][0]["orientation_is_upright"] = False
    value["execution_digest"] = canonical_digest(value, digest_field="execution_digest")
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


def test_scene_configuration_can_bind_its_visual_review_cost_scope(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The scene lane's exclusive attestation class must reach the cost gate."""

    final, _execution = _inputs(tmp_path)
    rights = tmp_path / "rights.json"
    materialize_artifixer_ai_visual_review_rights(
        configuration_run_id="configure-scene-839873-v1",
        source_scene_rights_admission_digest="sha256:" + "b" * 64,
        accepted_by="project-owner",
        accepted_on="2026-08-25",
        human_authority_reference="website-configuration-consent-v1",
        output_path=rights,
    )
    admin_key = tmp_path / "admin-key"
    admin_key.write_text("not-a-real-key\n", encoding="utf-8")
    observed: dict[str, object] = {}

    class ReachedCostGate(RuntimeError):
        pass

    def build_gate(**kwargs: object) -> object:
        observed.update(kwargs)
        raise ReachedCostGate

    monkeypatch.setattr(module, "build_openai_official_cost_run_gate", build_gate)
    scope = "task_evaluation_scene_configuration_artifixer_visual_review"

    with pytest.raises(ReachedCostGate):
        module.run_artifixer_ai_visual_review(
            final_composite_receipt_path=final,
            rights_attestation_path=rights,
            configuration_run_id="configure-scene-839873-v1",
            publisher_instance_id="104",
            minimum_review_frames=2,
            output_root=tmp_path / "review",
            openai_cost_scope_attestation_path=tmp_path / "scope.json",
            openai_admin_api_key_file=admin_key,
            openai_project_id="project-scene",
            openai_api_key_id="key-scene-review",
            cost_lane_id=scope,
            paid_resource_class=scope,
            require_zero_baseline=False,
        )

    assert observed["lane_id"] == scope
    assert observed["paid_resource_class"] == scope
    assert observed["require_zero_baseline"] is False
    assert (
        inspect.signature(module.run_artifixer_ai_visual_review)
        .parameters["require_zero_baseline"]
        .default
        is True
    )


def test_paired_target_review_binds_source_mask_and_generated_frame(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.png"
    mask = tmp_path / "mask.png"
    generated = tmp_path / "generated.png"
    Image.new("RGB", (8, 8), color=(90, 80, 70)).save(source)
    Image.new("L", (8, 8), color=255).save(mask)
    Image.new("RGB", (8, 8), color=(91, 81, 71)).save(generated)

    def record(path: Path) -> dict[str, object]:
        return {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }

    value = {
        "schema_version": DUAL_TARGET_REVIEW_SCHEMA_VERSION,
        "status": "paired_target_frames_pending_independent_visual_review",
        "publisher_scene_id": "839873",
        "review_scope": ("source_anchor_exact_mask_and_generated_full_frame_comparison"),
        "tasks": [
            {
                "task_id": "remove-source-object-104",
                "physical_camera_count": 1,
                "frames": [
                    {
                        "frame_index": 0,
                        "camera_id": "camera-0",
                        "source_frame": record(source),
                        "exact_repair_mask": record(mask),
                        "final_frame": record(generated),
                    }
                ],
            }
        ],
        "outside_support_invariance_proven": False,
        "outside_support_invariance_claimed": False,
        "semantic_object_absence_review_passed": False,
        "multiview_consistency_review_passed": False,
        "appearance_repair_qualified": False,
        "generated_output_is_capture_or_physical_evidence": False,
        "receipt_digest": "",
    }
    value["receipt_digest"] = canonical_digest(value, digest_field="receipt_digest")
    receipt = tmp_path / "paired-review.json"
    receipt.write_text(json.dumps(value), encoding="utf-8")

    request, task_id, inventory, reopened = build_artifixer_ai_visual_review_input(
        final_composite_receipt_path=receipt
    )

    assert task_id == "remove-source-object-104"
    assert inventory[0]["sha256"] == _sha256(generated)
    assert "right-side-up" in request[0]["content"][0]["text"]
    labels = [row.get("text") for row in request[0]["content"] if row.get("type") == "input_text"]
    assert "source_anchor" in labels
    assert "exact_repair_mask" in labels
    assert "generated_candidate" in labels
    assert reopened["outside_support_invariance_claimed"] is False
