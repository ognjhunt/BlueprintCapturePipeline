"""Truthful review modes for configured-scene appearance publication."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .decision_evidence_contracts import canonical_digest


REQUIRED_MODE = "required"
PAUSED_UNGRADED_MODE = "paused_ungraded"
PAUSED_UNGRADED_WARNING = "Visual review paused - appearance ungraded"
PAUSED_UNGRADED_SCOPE = "artifixer_appearance_only"
PAUSED_RECEIPT_SCHEMA_VERSION = "task_evaluation_artifixer_visual_review_pause_receipt.v1"

_HISTORICAL_PAUSED_OVERRIDE = {
    "mode": PAUSED_UNGRADED_MODE,
    "scope": PAUSED_UNGRADED_SCOPE,
    "ungraded_publication_acknowledged": True,
    "review_provider_call_permitted": False,
    "warning_label": PAUSED_UNGRADED_WARNING,
}


class AppearanceReviewContractError(ValueError):
    """The appearance-review override or receipt is internally inconsistent."""


def appearance_review_mode(
    request: Mapping[str, Any], *, allow_historical_paused: bool = False
) -> str:
    """Require independent grading for every newly admitted configuration.

    Historical ``paused_ungraded`` receipts remain readable so the product can
    display their true claim ceiling. New work may not mint another one.
    """

    override = request.get("appearance_review_override")
    if override is None:
        return REQUIRED_MODE
    if (
        allow_historical_paused
        and isinstance(override, Mapping)
        and dict(override) == _HISTORICAL_PAUSED_OVERRIDE
    ):
        return PAUSED_UNGRADED_MODE
    raise AppearanceReviewContractError("scene_configuration_appearance_review_pause_forbidden")


def paused_review_receipt_valid(
    receipt: Mapping[str, Any],
    *,
    publisher_instance_id: str,
    minimum_frame_count: int,
    thumbnail_digest: str,
) -> bool:
    """Validate a no-grader receipt without upgrading it to an acceptance."""

    selection = receipt.get("task_thumbnail_selection")
    selector = receipt.get("selector")
    frames = receipt.get("frames")
    if not isinstance(selection, Mapping) or set(selection) != {
        "camera_id",
        "frame_sha256",
        "rationale",
    }:
        return False
    if not isinstance(selector, Mapping) or dict(selector) != {
        "kind": "system",
        "identity": "deterministic_ungraded_thumbnail_selector",
        "runtime": "blueprint_pipeline",
        "model": "none",
    }:
        return False
    if not isinstance(frames, list) or len(frames) != minimum_frame_count:
        return False
    frame_pairs = [
        (row.get("camera_id"), row.get("frame_sha256"))
        for row in frames
        if isinstance(row, Mapping)
    ]
    if (
        len(frame_pairs) != minimum_frame_count
        or len(set(frame_pairs)) != minimum_frame_count
        or any(not camera_id or not digest for camera_id, digest in frame_pairs)
        or (
            selection.get("camera_id"),
            selection.get("frame_sha256"),
        )
        not in frame_pairs
    ):
        return False
    return (
        receipt.get("schema_version") == PAUSED_RECEIPT_SCHEMA_VERSION
        and receipt.get("status") == "visual_review_paused_ungraded"
        and receipt.get("decision") == "not_reviewed"
        and receipt.get("visual_review_mode") == PAUSED_UNGRADED_MODE
        and receipt.get("publisher_instance_id") == publisher_instance_id
        and receipt.get("review_frame_count") == minimum_frame_count
        and receipt.get("all_review_frames_digest_bound") is True
        and receipt.get("ai_visual_review_completed") is False
        and receipt.get("human_review_completed") is False
        and receipt.get("semantic_object_absence_review_passed") is False
        and receipt.get("multiview_consistency_review_passed") is False
        and receipt.get("task_thumbnail_is_exact_review_frame") is False
        and receipt.get("task_thumbnail_is_exact_rendered_frame") is True
        and receipt.get("review_provider_call_performed") is False
        and receipt.get("generated_output_is_capture_or_physical_evidence") is False
        and receipt.get("warning_label") == PAUSED_UNGRADED_WARNING
        and selection.get("frame_sha256") == thumbnail_digest
        and bool(str(selection.get("camera_id") or ""))
        and bool(str(selection.get("rationale") or "").strip())
        and receipt.get("receipt_digest")
        == canonical_digest(receipt, digest_field="receipt_digest")
    )


__all__ = [
    "AppearanceReviewContractError",
    "PAUSED_RECEIPT_SCHEMA_VERSION",
    "PAUSED_UNGRADED_MODE",
    "PAUSED_UNGRADED_SCOPE",
    "PAUSED_UNGRADED_WARNING",
    "REQUIRED_MODE",
    "appearance_review_mode",
    "paused_review_receipt_valid",
]

# An owner can accept one observed generated result without rewriting the AI grade.
HUMAN_APPROVAL_SCHEMA = "task_evaluation_artifixer_human_appearance_approval.v1"
HUMAN_REVIEW_SCHEMA = "task_evaluation_artifixer_human_visual_acceptance.v1"
HUMAN_ACCEPTED_STATUS = "human_accepted_with_known_artifacts"
HUMAN_REMOVAL_STATUS = "human_accepted_generated_appearance_edit"
HUMAN_REVIEW_ENV = "BLUEPRINT_ARTIFIXER_HUMAN_APPEARANCE_ACCEPTANCE_ROOT"


def validate_human_approval(value, *, expected_owner=None):
    """Validate explicit owner judgment and preserve the independently rejected grade."""
    import re

    def digest(v):
        return isinstance(v, str) and re.fullmatch(r"sha256:[0-9a-f]{64}", v) is not None

    if not isinstance(value, Mapping):
        raise AppearanceReviewContractError("human_appearance_approval_invalid")
    ai = value.get("source_ai_review_execution")
    frames = value.get("frames")
    valid = (
        value.get("schema_version") == HUMAN_APPROVAL_SCHEMA
        and value.get("status") == HUMAN_ACCEPTED_STATUS
        and value.get("scope") == "appearance_only_continuation"
        and bool(str(value.get("accepted_by") or "").strip())
        and (expected_owner is None or value.get("accepted_by") == expected_owner)
        and bool(str(value.get("recorded_at") or "").strip())
        and bool(str(value.get("statement") or "").strip())
        and bool(str(value.get("approval_reference") or "").strip())
        and isinstance(value.get("known_artifacts"), list)
        and bool(value["known_artifacts"])
        and all(isinstance(x, str) and x.strip() for x in value["known_artifacts"])
        and bool(str(value.get("source_launch_id") or "").strip())
        and all(
            digest(value.get(k))
            for k in (
                "source_provider_output_sha256",
                "source_checkpoint_digest",
                "source_post_training_binding_digest",
                "source_review_input_digest",
                "input_digest",
            )
        )
        and isinstance(frames, list)
        and 8 <= len(frames) <= 64
        and all(
            isinstance(r, Mapping) and bool(r.get("camera_id")) and digest(r.get("frame_sha256"))
            for r in frames
        )
        and len({r["camera_id"] for r in frames}) == len(frames)
        and isinstance(ai, Mapping)
        and ai.get("schema_version") == "task_evaluation_artifixer_ai_visual_review_execution.v1"
        and ai.get("status") == "completed"
        and ai.get("decision") == "rejected"
        and ai.get("review_phase") == "post_training"
        and ai.get("provider_called") is True
        and ai.get("execution_digest") == canonical_digest(ai, digest_field="execution_digest")
        and ai.get("final_composite_receipt_digest") == value.get("source_review_input_digest")
        and ai.get("input_digest") == value.get("input_digest")
        and isinstance(ai.get("frames"), list)
        and all(isinstance(r, Mapping) for r in ai["frames"])
        and len(ai["frames"]) == len(frames)
        and {r["camera_id"]: r["frame_sha256"] for r in frames}
        == {r.get("camera_id"): r.get("frame_sha256") for r in ai.get("frames", [])}
        and value.get("approval_digest") == canonical_digest(value, digest_field="approval_digest")
    )
    if not valid:
        raise AppearanceReviewContractError("human_appearance_approval_invalid")
    return dict(value)


def human_review_receipt_valid(
    receipt, *, publisher_instance_id, minimum_frame_count, thumbnail_digest, expected_owner=None
):
    try:
        approval = validate_human_approval(
            receipt.get("human_approval"), expected_owner=expected_owner
        )
        selection = receipt.get("task_thumbnail_selection", {})
        reviewer = receipt.get("reviewer", {})
        pairs = {(r["camera_id"], r["frame_sha256"]) for r in approval["frames"]}
        return (
            receipt.get("schema_version") == HUMAN_REVIEW_SCHEMA
            and receipt.get("status") == receipt.get("decision") == HUMAN_ACCEPTED_STATUS
            and receipt.get("publisher_instance_id") == publisher_instance_id
            and receipt.get("review_frame_count") == len(pairs) >= minimum_frame_count
            and receipt.get("frames") == approval["frames"]
            and receipt.get("all_review_frames_digest_bound") is True
            and receipt.get("human_review_completed") is True
            and receipt.get("ai_visual_review_completed") is True
            and receipt.get("ai_visual_review_accepted") is False
            and receipt.get("ai_visual_review_status") == "rejected"
            and receipt.get("semantic_object_absence_review_passed") is False
            and receipt.get("multiview_consistency_review_passed") is False
            and receipt.get("generated_output_is_capture_or_physical_evidence") is False
            and receipt.get("physics_or_collision_authority_granted") is False
            and receipt.get("new_training_executed") is False
            and receipt.get("new_review_provider_call_performed") is False
            and receipt.get("second_repair_round_permitted") is False
            and reviewer
            == {
                "kind": "human",
                "identity": approval["accepted_by"],
                "runtime": "owner_approval",
                "model": "none",
            }
            and receipt.get("thumbnail_selector") == "deterministic_first_approved_camera"
            and receipt.get("task_thumbnail_is_exact_review_frame") is True
            and (selection.get("camera_id"), thumbnail_digest) in pairs
            and selection.get("frame_sha256") == thumbnail_digest
            and bool(str(selection.get("rationale") or "").strip())
            and receipt.get("receipt_digest")
            == canonical_digest(receipt, digest_field="receipt_digest")
        )
    except (AppearanceReviewContractError, AttributeError, KeyError, TypeError, ValueError):
        return False


def stage_human_approval(
    *, source_root, source_launch_root, output_root, completed_training, expected_owner
):
    import json
    from pathlib import Path
    from .artifixer_completed_training_reuse import _read, _sha

    if not isinstance(expected_owner, str) or not expected_owner:
        raise AppearanceReviewContractError("human_appearance_owner_required")
    source = Path(source_root) / "human_appearance_approval.json"
    approval = validate_human_approval(_read(source), expected_owner=expected_owner)
    reuse = _read(Path(completed_training["receipt_path"]))
    from .task_evaluation_scene_configuration_artifixer_warm_checkpoint import (
        validate_artifixer_post_training_checkpoint,
    )

    checkpoint = validate_artifixer_post_training_checkpoint(
        checkpoint_root=completed_training["checkpoint_root"]
    )
    if (
        reuse.get("receipt_digest") != completed_training["receipt_digest"]
        or reuse.get("receipt_digest") != canonical_digest(reuse, digest_field="receipt_digest")
        or approval["source_launch_id"] != reuse.get("source_launch_id")
        or approval["source_provider_output_sha256"] != reuse.get("source_provider_output_sha256")
        or approval["source_checkpoint_digest"] != reuse.get("source_checkpoint_digest")
        or approval["source_checkpoint_digest"] != checkpoint.get("checkpoint_digest")
        or approval["source_post_training_binding_digest"] != checkpoint.get("binding_digest")
    ):
        raise AppearanceReviewContractError("human_appearance_training_binding_mismatch")
    import zipfile

    source_launch = Path(source_launch_root)
    source_archive = (
        source_launch
        / "allocator/scene-configuration-job/vast_provider_run/vast_provider_runtime_output.zip"
    )
    if (
        source_launch.name != approval["source_launch_id"]
        or _sha(source_archive) != approval["source_provider_output_sha256"]
    ):
        raise AppearanceReviewContractError("human_appearance_source_archive_changed")
    with zipfile.ZipFile(source_archive) as archive:
        prefix = "stages/stage-1/producer/released_artifixer_runtime/artifixer_candidate_round_0/"
        original_ai = json.loads(
            archive.read(
                prefix
                + "independent_visual_review/task_evaluation_artifixer_ai_visual_review_execution.v1.json"
            )
        )
    runtime = _read(Path(completed_training["checkpoint_root"]) / "runtime/runtime_result.json")
    if original_ai != approval["source_ai_review_execution"] or {
        r["camera_id"]: r["sha256"] for r in runtime["tasks"][0]["artifixer3d_review_frames"]
    } != {r["camera_id"]: r["frame_sha256"] for r in approval["frames"]}:
        raise AppearanceReviewContractError("human_appearance_saved_evidence_changed")
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=False)
    target = root / source.name
    target.write_text(json.dumps(approval, sort_keys=True) + "\n")
    return {"path": str(target), "sha256": _sha(target), "expected_owner": expected_owner}


def reuse_human_approval(
    *, reference, current_input_path, output_root, publisher_instance_id, minimum_frame_count
):
    from pathlib import Path
    from .artifixer_completed_training_reuse import _read, _sha
    from .decision_evidence_contracts import canonical_json
    from .task_evaluation_artifixer_ai_visual_review import build_artifixer_ai_visual_review_input

    source = Path(reference["path"])
    if _sha(source) != reference["sha256"]:
        raise AppearanceReviewContractError("human_appearance_approval_changed")
    approval = validate_human_approval(_read(source), expected_owner=reference["expected_owner"])
    current = _read(current_input_path)
    payload, _, inventory, _ = build_artifixer_ai_visual_review_input(
        final_composite_receipt_path=current_input_path
    )
    if (
        current.get("review_phase") != "post_training"
        or approval["input_digest"] != canonical_digest({"input": payload})
        or {r["camera_id"]: r["sha256"] for r in inventory}
        != {r["camera_id"]: r["frame_sha256"] for r in approval["frames"]}
        or len(inventory) < minimum_frame_count
    ):
        raise AppearanceReviewContractError("human_appearance_review_inputs_changed")
    selected = min(approval["frames"], key=lambda r: r["camera_id"])
    receipt = {
        "schema_version": HUMAN_REVIEW_SCHEMA,
        "status": HUMAN_ACCEPTED_STATUS,
        "decision": HUMAN_ACCEPTED_STATUS,
        "publisher_instance_id": publisher_instance_id,
        "human_approval": approval,
        "frames": approval["frames"],
        "review_frame_count": len(inventory),
        "all_review_frames_digest_bound": True,
        "human_review_completed": True,
        "ai_visual_review_completed": True,
        "ai_visual_review_accepted": False,
        "ai_visual_review_status": "rejected",
        "semantic_object_absence_review_passed": False,
        "multiview_consistency_review_passed": False,
        "generated_output_is_capture_or_physical_evidence": False,
        "physics_or_collision_authority_granted": False,
        "new_training_executed": False,
        "new_review_provider_call_performed": False,
        "second_repair_round_permitted": False,
        "reviewer": {
            "kind": "human",
            "identity": approval["accepted_by"],
            "runtime": "owner_approval",
            "model": "none",
        },
        "thumbnail_selector": "deterministic_first_approved_camera",
        "task_thumbnail_selection": {
            **selected,
            "rationale": "First camera in the owner-accepted frame set; selected automatically.",
        },
        "task_thumbnail_is_exact_review_frame": True,
        "current_review_input_digest": current["receipt_digest"],
        "claim_boundary": "owner_accepted_generated_appearance_with_known_artifacts_only",
    }
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    path = Path(output_root) / "human_appearance_acceptance.json"
    path.write_text(canonical_json(receipt) + "\n")
    return {
        "review_input": current,
        "review_input_path": current_input_path,
        "review": {
            "decision": "accepted",
            "human_acceptance": True,
            "new_model_call_performed": False,
            "review_receipt": {"path": str(path), "receipt_digest": receipt["receipt_digest"]},
        },
    }
