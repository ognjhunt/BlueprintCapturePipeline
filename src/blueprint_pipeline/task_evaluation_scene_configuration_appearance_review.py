"""Truthful review modes for configured-scene appearance publication."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .decision_evidence_contracts import canonical_digest


REQUIRED_MODE = "required"
PAUSED_UNGRADED_MODE = "paused_ungraded"
PAUSED_UNGRADED_WARNING = "Visual review paused - appearance ungraded"
PAUSED_UNGRADED_SCOPE = "artifixer_appearance_only"
PAUSED_RECEIPT_SCHEMA_VERSION = (
    "task_evaluation_artifixer_visual_review_pause_receipt.v1"
)

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
    raise AppearanceReviewContractError(
        "scene_configuration_appearance_review_pause_forbidden"
    )


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
