from __future__ import annotations
import hashlib
import json
from pathlib import Path
import pytest
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline import task_evaluation_scene_configuration_appearance_review as human
from blueprint_pipeline.task_evaluation_artifixer_ai_visual_review import (
    build_artifixer_ai_visual_review_input,
)
from tests.test_task_evaluation_artifixer_ai_visual_review import _inputs


def seal(value, field):
    value[field] = canonical_digest(value, digest_field=field)
    return value


def human_case(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    path, execution_path = _inputs(root)
    current = json.loads(path.read_text())
    source = current["tasks"][0]["frames"][0]["final_frame"]
    from PIL import Image

    image_path = Path(source["path"])
    Image.new("RGB", (8, 8), "white").save(image_path)
    source.update(
        size_bytes=image_path.stat().st_size,
        sha256="sha256:" + hashlib.sha256(image_path.read_bytes()).hexdigest(),
    )
    frames = [
        {
            "frame_index": i,
            "camera_id": f"camera-{i:02d}",
            "final_frame": dict(source),
            "source_frame": dict(source),
            "exact_repair_mask": dict(source),
        }
        for i in range(16)
    ]
    current.update(
        schema_version="task_evaluation_artifixer3d_dual_target_review_input.v1",
        status="paired_target_frames_pending_independent_visual_review",
        review_phase="post_training",
        review_scope="source_anchor_exact_mask_and_generated_full_frame_comparison",
        outside_support_invariance_proven=False,
        outside_support_invariance_claimed=False,
    )
    current["tasks"][0].update(frames=frames, physical_camera_count=16)
    seal(current, "receipt_digest")
    path.write_text(json.dumps(current))
    payload, _, inventory, _ = build_artifixer_ai_visual_review_input(
        final_composite_receipt_path=path
    )
    execution = json.loads(execution_path.read_text())
    execution.update(
        decision="rejected",
        review_phase="post_training",
        final_composite_receipt_digest=current["receipt_digest"],
        input_digest=canonical_digest({"input": payload}),
        semantic_object_absence_review_passed=False,
        multiview_consistency_review_passed=False,
        frames=[
            {
                "camera_id": r["camera_id"],
                "frame_sha256": r["sha256"],
                "decision": "rejected",
                "rationale": "Small dark residual mark.",
            }
            for r in inventory
        ],
    )
    seal(execution, "execution_digest")
    approval = seal(
        {
            "schema_version": human.HUMAN_APPROVAL_SCHEMA,
            "status": human.HUMAN_ACCEPTED_STATUS,
            "scope": "appearance_only_continuation",
            "accepted_by": "owner",
            "recorded_at": "2026-09-15T03:40:00Z",
            "statement": "This is good enough; continue.",
            "approval_reference": "task:owner-approval",
            "known_artifacts": ["Small residual marks."],
            "source_launch_id": "source-launch",
            "source_provider_output_sha256": "sha256:" + "a" * 64,
            "source_checkpoint_digest": "sha256:" + "b" * 64,
            "source_post_training_binding_digest": "sha256:" + "c" * 64,
            "source_review_input_digest": current["receipt_digest"],
            "input_digest": execution["input_digest"],
            "frames": [
                {"camera_id": r["camera_id"], "frame_sha256": r["sha256"]} for r in inventory
            ],
            "source_ai_review_execution": execution,
        },
        "approval_digest",
    )
    approval_path = root / "human_appearance_approval.json"
    approval_path.write_text(json.dumps(approval))
    reference = {
        "path": str(approval_path),
        "sha256": "sha256:" + hashlib.sha256(approval_path.read_bytes()).hexdigest(),
        "expected_owner": "owner",
    }
    return path, approval, reference, frames


def test_exact_human_acceptance_reuses_review_without_model_or_grade_rewrite(tmp_path):
    path, approval, reference, frames = human_case(tmp_path)
    before = json.dumps(approval["source_ai_review_execution"], sort_keys=True)
    result = human.reuse_human_approval(
        reference=reference,
        current_input_path=path,
        output_root=tmp_path,
        publisher_instance_id="104",
        minimum_frame_count=16,
    )
    receipt_path = Path(result["review"]["review_receipt"]["path"])
    receipt = json.loads(receipt_path.read_text())
    selected = receipt["task_thumbnail_selection"]
    assert human.human_review_receipt_valid(
        receipt,
        publisher_instance_id="104",
        minimum_frame_count=16,
        thumbnail_digest=selected["frame_sha256"],
        expected_owner="owner",
    )
    assert receipt["ai_visual_review_accepted"] is False
    assert receipt["new_training_executed"] is False
    assert receipt["new_review_provider_call_performed"] is False
    assert (
        json.dumps(receipt["human_approval"]["source_ai_review_execution"], sort_keys=True)
        == before
    )
    from blueprint_pipeline.task_evaluation_scene_configuration_artifixer_artifacts import (
        _materialize_selected_task_thumbnail,
    )

    thumbnail = tmp_path / "thumbnail.png"
    _materialize_selected_task_thumbnail(
        review_receipt=receipt, review_frames=frames, destination=thumbnail
    )
    removal = seal(
        {
            "status": human.HUMAN_REMOVAL_STATUS,
            "publisher_instance_id": "104",
            "visual_review_receipt_digest": receipt["receipt_digest"],
        },
        "result_digest",
    )
    removal_path = tmp_path / "removal.json"
    removal_path.write_text(json.dumps(removal))
    from blueprint_pipeline.task_evaluation_scene_configuration_publication import (
        _thumbnail_selection,
    )

    selection = _thumbnail_selection(
        review_receipt_path=receipt_path,
        thumbnail_path=thumbnail,
        removal_receipt_path=removal_path,
        minimum_frame_count=16,
    )
    assert selection["appearance_review_status"] == human.HUMAN_ACCEPTED_STATUS
    assert selection["reviewer"]["kind"] == "human"
    assert selection["ai_visual_review_status"] == "rejected"


@pytest.mark.parametrize(
    "mutation",
    ["owner", "scope", "ai_grade", "missing_frame", "changed_frame", "missing_statement"],
)
def test_human_acceptance_refuses_changed_authority_or_evidence(tmp_path, mutation):
    _, approval, _, _ = human_case(tmp_path)
    if mutation == "owner":
        approval["accepted_by"] = "other-owner"
    elif mutation == "scope":
        approval["scope"] = "physics_approval"
    elif mutation == "ai_grade":
        approval["source_ai_review_execution"]["decision"] = "accepted"
        seal(approval["source_ai_review_execution"], "execution_digest")
    elif mutation == "missing_frame":
        approval["frames"].pop()
    elif mutation == "changed_frame":
        approval["frames"][0]["frame_sha256"] = "sha256:" + "f" * 64
    else:
        approval["statement"] = ""
    seal(approval, "approval_digest")
    with pytest.raises(human.AppearanceReviewContractError):
        human.validate_human_approval(approval, expected_owner="owner")


def test_human_acceptance_cannot_follow_changed_current_frame_bytes(tmp_path):
    path, _, reference, _ = human_case(tmp_path)
    d = json.loads(path.read_text())
    d["tasks"][0]["frames"][0]["camera_id"] = "changed-camera"
    seal(d, "receipt_digest")
    path.write_text(json.dumps(d))
    with pytest.raises(human.AppearanceReviewContractError, match="review_inputs_changed"):
        human.reuse_human_approval(
            reference=reference,
            current_input_path=path,
            output_root=tmp_path,
            publisher_instance_id="104",
            minimum_frame_count=16,
        )


def test_human_receipt_cannot_claim_ai_acceptance(tmp_path):
    path, _, reference, _ = human_case(tmp_path)
    result = human.reuse_human_approval(
        reference=reference,
        current_input_path=path,
        output_root=tmp_path,
        publisher_instance_id="104",
        minimum_frame_count=16,
    )
    receipt = json.loads(Path(result["review"]["review_receipt"]["path"]).read_text())
    receipt["ai_visual_review_accepted"] = True
    seal(receipt, "receipt_digest")
    assert not human.human_review_receipt_valid(
        receipt,
        publisher_instance_id="104",
        minimum_frame_count=16,
        thumbnail_digest=receipt["task_thumbnail_selection"]["frame_sha256"],
        expected_owner="owner",
    )
