"""Retain digest-bound ArtiFixer appearance artifacts without granting qualification."""
from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest, canonical_json
from .task_evaluation_artifixer_ai_visual_review import (
    EXECUTION_SCHEMA_VERSION as VISUAL_REVIEW_EXECUTION_SCHEMA_VERSION,
)
from .task_evaluation_scene_configuration_appearance_review import (
    PAUSED_RECEIPT_SCHEMA_VERSION, PAUSED_UNGRADED_MODE, PAUSED_UNGRADED_WARNING,
)


_DIAGNOSTIC_ONLY_ENV = "BLUEPRINT_SCENE_CONFIGURATION_DIAGNOSTIC_ONLY"


_DIAGNOSTIC_REJECTED_APPEARANCE_STATUS = (
    "diagnostic_generated_appearance_edit_visual_review_rejected"
)


def _diagnostic_rejection_permitted(
    *, stage_input: Mapping[str, Any], environment: Mapping[str, str]
) -> bool:
    return (
        stage_input.get("execution_mode") == "diagnostic_only"
        and str(environment.get(_DIAGNOSTIC_ONLY_ENV) or "") == "1"
    )


class TaskEvaluationSceneConfigurationArtifixerError(RuntimeError):
    """The released ArtiFixer chain could not satisfy the generic stage."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _read(path: Path, *, code: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TaskEvaluationSceneConfigurationArtifixerError(code) from exc
    if path.is_symlink() or not isinstance(value, Mapping):
        raise TaskEvaluationSceneConfigurationArtifixerError(code)
    return dict(value)


def _record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _component_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "digest": _sha256(path),
    }


def _materialize_selected_task_thumbnail(
    *,
    review_receipt: Mapping[str, Any],
    review_frames: list[Mapping[str, Any]],
    destination: Path,
) -> dict[str, Any]:
    """Copy exactly one AI-selected, already-reviewed frame without alteration."""

    thumbnail_selection = review_receipt.get("task_thumbnail_selection")
    reviewer = review_receipt.get("reviewer")
    if (
        not isinstance(thumbnail_selection, Mapping)
        or not isinstance(reviewer, Mapping)
        or reviewer.get("kind") != "ai"
        or not str(reviewer.get("identity") or "")
        or not str(reviewer.get("model") or "")
        or review_receipt.get("task_thumbnail_is_exact_review_frame") is not True
    ):
        raise TaskEvaluationSceneConfigurationArtifixerError(
            "scene_configuration_artifixer_thumbnail_selection_invalid"
        )
    thumbnail_matches = [
        frame
        for frame in review_frames
        if frame.get("camera_id") == thumbnail_selection.get("camera_id")
        and isinstance(frame.get("final_frame"), Mapping)
        and frame["final_frame"].get("sha256")
        == thumbnail_selection.get("frame_sha256")
    ]
    if len(thumbnail_matches) != 1:
        raise TaskEvaluationSceneConfigurationArtifixerError(
            "scene_configuration_artifixer_thumbnail_selection_invalid"
        )
    selected_frame = Path(str(thumbnail_matches[0]["final_frame"].get("path") or ""))
    if (
        selected_frame.is_symlink()
        or not selected_frame.is_file()
        or _sha256(selected_frame) != thumbnail_selection.get("frame_sha256")
    ):
        raise TaskEvaluationSceneConfigurationArtifixerError(
            "scene_configuration_artifixer_thumbnail_selection_invalid"
        )
    shutil.copyfile(selected_frame, destination)
    if _sha256(destination) != thumbnail_selection["frame_sha256"]:
        raise TaskEvaluationSceneConfigurationArtifixerError(
            "scene_configuration_artifixer_thumbnail_copy_mismatch"
        )
    return {
        "camera_id": thumbnail_selection["camera_id"],
        "frame_sha256": thumbnail_selection["frame_sha256"],
        "rationale": thumbnail_selection["rationale"],
        "reviewer": dict(reviewer),
        "derived_appearance_evidence": True,
        "capture_or_physical_evidence": False,
    }


def _materialize_ungraded_task_thumbnail_and_receipt(
    *,
    review_frames: list[Mapping[str, Any]],
    publisher_instance_id: str,
    minimum_frame_count: int,
    thumbnail_destination: Path,
    receipt_destination: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Select one rendered frame deterministically without claiming review."""

    candidates: list[tuple[str, str, Path]] = []
    for frame in review_frames:
        final_frame = frame.get("final_frame")
        if not isinstance(final_frame, Mapping):
            continue
        camera_id = str(frame.get("camera_id") or "")
        digest = str(final_frame.get("sha256") or "")
        path = Path(str(final_frame.get("path") or ""))
        if (
            not camera_id
            or not digest
            or path.is_symlink()
            or not path.is_file()
            or _sha256(path) != digest
        ):
            continue
        candidates.append((camera_id, digest, path))
    candidates.sort(key=lambda row: (row[0], row[1]))
    if (
        len(candidates) != minimum_frame_count
        or len({(row[0], row[1]) for row in candidates}) != minimum_frame_count
    ):
        raise TaskEvaluationSceneConfigurationArtifixerError(
            "scene_configuration_artifixer_ungraded_thumbnail_inventory_invalid"
        )
    camera_id, digest, selected_path = candidates[0]
    shutil.copyfile(selected_path, thumbnail_destination)
    if _sha256(thumbnail_destination) != digest:
        raise TaskEvaluationSceneConfigurationArtifixerError(
            "scene_configuration_artifixer_thumbnail_copy_mismatch"
        )
    rationale = (
        "Deterministic first camera; visual review was explicitly paused and "
        "this frame remains ungraded."
    )
    selector = {
        "kind": "system",
        "identity": "deterministic_ungraded_thumbnail_selector",
        "runtime": "blueprint_pipeline",
        "model": "none",
    }
    selection = {
        "camera_id": camera_id,
        "frame_sha256": digest,
        "rationale": rationale,
    }
    receipt: dict[str, Any] = {
        "schema_version": PAUSED_RECEIPT_SCHEMA_VERSION,
        "status": "visual_review_paused_ungraded",
        "decision": "not_reviewed",
        "visual_review_mode": PAUSED_UNGRADED_MODE,
        "publisher_instance_id": publisher_instance_id,
        "review_frame_count": minimum_frame_count,
        "frames": [
            {"camera_id": row[0], "frame_sha256": row[1]}
            for row in candidates
        ],
        "all_review_frames_digest_bound": True,
        "ai_visual_review_completed": False,
        "human_review_completed": False,
        "semantic_object_absence_review_passed": False,
        "multiview_consistency_review_passed": False,
        "task_thumbnail_is_exact_review_frame": False,
        "task_thumbnail_is_exact_rendered_frame": True,
        "task_thumbnail_selection": selection,
        "selector": selector,
        "review_provider_call_performed": False,
        "generated_output_is_capture_or_physical_evidence": False,
        "warning_label": PAUSED_UNGRADED_WARNING,
        "receipt_digest": "",
    }
    receipt["receipt_digest"] = canonical_digest(
        receipt, digest_field="receipt_digest"
    )
    receipt_destination.write_text(
        canonical_json(receipt) + "\n", encoding="utf-8"
    )
    return (
        {
            **selection,
            "reviewer": selector,
            "appearance_review_status": PAUSED_UNGRADED_MODE,
            "derived_appearance_evidence": True,
            "capture_or_physical_evidence": False,
        },
        receipt,
    )


def _materialize_diagnostic_rejected_artifixer_artifacts(
    *,
    review: Mapping[str, Any],
    review_frames: list[Mapping[str, Any]],
    native_appearance_source: Path,
    configuration: Mapping[str, Any],
    output_root: Path,
    render_handoff: Mapping[str, Any],
    source_diagnostic_checkpoint_digest: str,
    post_training_binding_digest: str,
) -> list[dict[str, Any]]:
    """Retain one rejected view for downstream diagnostics without qualifying it."""

    execution_record = review.get("execution_receipt")
    if not isinstance(execution_record, Mapping):
        raise TaskEvaluationSceneConfigurationArtifixerError(
            "scene_configuration_artifixer_diagnostic_rejection_invalid"
        )
    execution_path = Path(str(execution_record.get("path") or "")).resolve()
    execution = _read(
        execution_path,
        code="scene_configuration_artifixer_diagnostic_rejection_invalid",
    )
    frame_rows = execution.get("frames")
    reviewer = execution.get("reviewer")
    minimum_views = int((configuration.get("required_views") or {}).get("minimum") or 0)
    if (
        execution.get("schema_version") != VISUAL_REVIEW_EXECUTION_SCHEMA_VERSION
        or execution.get("status") != "completed"
        or execution.get("decision") != "rejected"
        or execution.get("publisher_instance_id")
        != (configuration.get("source_object") or {}).get("publisher_instance_id")
        or execution.get("review_frame_count") != len(review_frames)
        or len(review_frames) != minimum_views
        or minimum_views < 2
        or execution.get("provider_called") is not True
        or execution.get("provider") != "openai"
        or execution.get("response_store") is not False
        or execution.get("tracing_disabled") is not True
        or execution.get("raw_secret_values_recorded") is not False
        or execution.get("generated_output_is_capture_or_physical_evidence") is not False
        or execution.get("physics_or_collision_authority_granted") is not False
        or execution.get("execution_digest")
        != canonical_digest(execution, digest_field="execution_digest")
        or execution_record.get("execution_digest") != execution.get("execution_digest")
        or execution_record.get("sha256") != _sha256(execution_path)
        or execution_record.get("size_bytes") != execution_path.stat().st_size
        or not isinstance(frame_rows, list)
        or not isinstance(reviewer, Mapping)
        or reviewer.get("kind") != "ai"
        or not str(reviewer.get("identity") or "")
        or not str(reviewer.get("runtime") or "")
        or not str(reviewer.get("model") or "")
    ):
        raise TaskEvaluationSceneConfigurationArtifixerError(
            "scene_configuration_artifixer_diagnostic_rejection_invalid"
        )
    expected_frames = {
        (
            str(row.get("camera_id") or ""),
            str((row.get("final_frame") or {}).get("sha256") or ""),
        )
        for row in review_frames
        if isinstance(row, Mapping) and isinstance(row.get("final_frame"), Mapping)
    }
    reviewed_frames = {
        (str(row.get("camera_id") or ""), str(row.get("frame_sha256") or ""))
        for row in frame_rows
        if isinstance(row, Mapping)
    }
    accepted_rows = [
        row
        for row in frame_rows
        if isinstance(row, Mapping) and row.get("decision") == "accepted"
    ]
    rejected_rows = [
        row
        for row in frame_rows
        if isinstance(row, Mapping) and row.get("decision") == "rejected"
    ]
    if (
        expected_frames != reviewed_frames
        or len(expected_frames) != minimum_views
        or len(accepted_rows) != minimum_views - 1
        or len(rejected_rows) != 1
        or any(
            row.get("orientation_is_upright") is not True
            or row.get("source_object_absent") is not True
            or row.get("repair_is_locally_plausible") is not True
            or row.get("preserves_non_target_content") is not True
            for row in accepted_rows
        )
    ):
        raise TaskEvaluationSceneConfigurationArtifixerError(
            "scene_configuration_artifixer_diagnostic_rejection_invalid"
        )
    if not source_diagnostic_checkpoint_digest.startswith("sha256:") or not (
        post_training_binding_digest.startswith("sha256:")
    ):
        raise TaskEvaluationSceneConfigurationArtifixerError(
            "scene_configuration_artifixer_diagnostic_rejection_invalid"
        )
    appearance = output_root / "diagnostic_rejected_appearance_candidate.usdz"
    shutil.copyfile(native_appearance_source, appearance)
    copied_review = output_root / "appearance_visual_review_rejection_execution.v1.json"
    shutil.copyfile(execution_path, copied_review)
    removal: dict[str, Any] = {
        "schema_version": "task_evaluation_artifixer_object_removal_result.v1",
        "status": _DIAGNOSTIC_REJECTED_APPEARANCE_STATUS,
        "publisher_instance_id": (configuration.get("source_object") or {}).get(
            "publisher_instance_id"
        ),
        "raw_interiorgs_bytes_sent_to_external_provider": False,
        "visual_review_execution_digest": execution["execution_digest"],
        "visual_review_execution_sha256": _sha256(copied_review),
        "review_frame_count": minimum_views,
        "accepted_review_frame_count": len(accepted_rows),
        "rejected_review_frame_count": len(rejected_rows),
        "frame_decisions": [
            {
                "camera_id": row["camera_id"],
                "frame_sha256": row["frame_sha256"],
                "decision": row["decision"],
                "rationale": row["rationale"],
            }
            for row in frame_rows
        ],
        "diagnostic_rejected_appearance_candidate_sha256": _sha256(appearance),
        "source_diagnostic_checkpoint_digest": source_diagnostic_checkpoint_digest,
        "post_training_binding_digest": post_training_binding_digest,
        "semantic_object_free_visual_review_passed": False,
        "multiview_consistency_review_passed": False,
        "generated_pixels_labeled": True,
        "diagnostic_only": True,
        "qualification_eligible": False,
        "configured_revision_publication_permitted": False,
        "offering_publication_permitted": False,
        "terminal_e2e_completion_permitted": False,
        "appearance_authority": "rejected_generated_candidate_for_downstream_diagnostics_only",
        "result_digest": "",
    }
    removal["result_digest"] = canonical_digest(
        removal, digest_field="result_digest"
    )
    removal_path = output_root / "appearance_rejection_receipt.v1.json"
    removal_path.write_text(canonical_json(removal) + "\n", encoding="utf-8")
    return [
        {
            "role": "diagnostic_rejected_appearance_candidate",
            **_component_record(appearance),
        },
        {"role": "appearance_rejection_receipt", **_component_record(removal_path)},
        {
            "role": "appearance_visual_review_execution",
            **_component_record(copied_review),
        },
        dict(render_handoff),
    ]
