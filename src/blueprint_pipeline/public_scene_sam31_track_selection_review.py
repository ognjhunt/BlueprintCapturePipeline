"""Render and seal human or AI review of SAM track selections for 1--5 tasks.

The candidate packet is deterministic visual support.  A separate acceptance
receipt binds the exact task freezes, normalized SAM results, selected track
IDs, and review media.  Downstream mask materialization must reopen that
receipt; a naked digest or unbound reviewer prose is not selection authority.
"""

from __future__ import annotations

import argparse
import base64
from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageChops

from .decision_evidence_contracts import canonical_digest, canonical_json
from .scene_placement.semantic_gaussian_lifting import canonical_json_digest
from .task_evaluation_supervisor.inference_reservations import (
    INFERENCE_COMPLETION_SCHEMA_VERSION,
    INFERENCE_RESERVATION_MANIFEST_SCHEMA_VERSION,
    INFERENCE_RESERVATION_SCHEMA_VERSION,
)


CANDIDATE_SCHEMA_VERSION = "public_scene_sam31_track_selection_review_candidate.v1"
RECEIPT_SCHEMA_VERSION = "public_scene_sam31_track_selection_review.v1"
AI_RECEIPT_SCHEMA_VERSION = "public_scene_sam31_track_selection_ai_visual_review.v1"
AI_EXECUTION_SCHEMA_VERSION = "public_scene_sam31_ai_visual_review_execution.v1"
AI_RIGHTS_SCHEMA_VERSION = "public_scene_sam31_ai_visual_review_rights.v1"
AI_REVIEW_METHOD = "exact_overlay_visual_inspection"
AI_REVIEWER_ID = "blueprint-openai-agents-sdk-sam31-visual-reviewer"
AI_REVIEW_CAPABILITY = "public_scene_sam31_track_selection_visual_review"
AI_REVIEW_MODEL = "gpt-5.6-terra"
AI_REVIEW_FRAME_COUNT = 16
AI_REVIEW_INPUT_TOKEN_CEILING = 250_000
AI_REVIEW_MAX_COST_USD = 1.0
AI_REVIEW_DECLARED_USE = "noncommercial_internal_adp_visual_review"
AI_REVIEW_ACCEPTED_BY = "nijelhunt_1"

_AI_REVIEW_PROMPT = (
    "Review every supplied calibrated overlay. Magenta pixels are the selected SAM mask. "
    "Accept a frame only when those pixels select the named task object and do not materially "
    "select another object or background. A zero-pixel mask is acceptable only when the named "
    "target is genuinely absent or fully occluded in that camera; if the target is visible with "
    "no mask, reject it. Return exactly one decision for every task_id/camera_id pair. Accept "
    "the whole candidate only when every frame is accepted. This is visual selection review, "
    "not geometry, physical, or simulator qualification."
)


class Sam31TrackSelectionReviewError(ValueError):
    """A review candidate or acceptance receipt is invalid."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _record(path: Path, *, root: Path | None = None) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    value: dict[str, Any] = {
        "size_bytes": resolved.stat().st_size,
        "sha256": _sha256(resolved),
    }
    if root is not None:
        try:
            value["relative_path"] = resolved.relative_to(root.resolve()).as_posix()
        except ValueError:
            value["path"] = str(resolved)
    else:
        value["path"] = str(resolved)
    return value


def _read(path: str | Path, *, code: str) -> tuple[Path, dict[str, Any]]:
    unresolved = Path(path).expanduser()
    resolved = unresolved.resolve()
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Sam31TrackSelectionReviewError(code) from exc
    if unresolved.is_symlink() or not resolved.is_file() or not isinstance(value, dict):
        raise Sam31TrackSelectionReviewError(code)
    return resolved, value


def _record_path(record: Mapping[str, Any], *, root: Path) -> Path:
    relative = record.get("relative_path")
    absolute = record.get("path")
    if bool(relative) == bool(absolute):
        raise Sam31TrackSelectionReviewError("sam31_review_media_record_invalid")
    path = (root / str(relative)).resolve() if relative else Path(str(absolute)).resolve()
    if relative:
        try:
            path.relative_to(root.resolve())
        except ValueError as exc:
            raise Sam31TrackSelectionReviewError("sam31_review_media_record_invalid") from exc
    return path


def _verify_record(record: object, *, root: Path) -> Path:
    if not isinstance(record, Mapping):
        raise Sam31TrackSelectionReviewError("sam31_review_media_record_invalid")
    path = _record_path(record, root=root)
    if path.is_symlink() or not path.is_file():
        raise Sam31TrackSelectionReviewError("sam31_review_media_record_invalid")
    expected = _record(path, root=root if record.get("relative_path") else None)
    if any(record.get(key) != value for key, value in expected.items()):
        raise Sam31TrackSelectionReviewError("sam31_review_media_record_invalid")
    return path


def _validate_candidate_file(candidate_path: Path, candidate: Mapping[str, Any]) -> None:
    if (
        candidate.get("schema_version") != CANDIDATE_SCHEMA_VERSION
        or candidate.get("status") != "selected_track_overlays_materialized_pending_visual_review"
        or candidate.get("candidate_digest")
        != canonical_digest(candidate, digest_field="candidate_digest")
    ):
        raise Sam31TrackSelectionReviewError("sam31_review_candidate_invalid")
    root = candidate_path.parent.resolve()
    bindings = candidate.get("selection_bindings")
    review_media = candidate.get("review_media")
    if (
        not isinstance(bindings, list)
        or not isinstance(review_media, list)
        or not 1 <= len(bindings) <= 5
        or candidate.get("task_count") != len(bindings)
        or candidate.get("task_count") != len(review_media)
        or Path(str(candidate.get("candidate_masks_root") or "")).resolve()
        != root / "candidate_masks"
    ):
        raise Sam31TrackSelectionReviewError("sam31_review_candidate_invalid")
    task_ids = [str(row.get("task_id") or "") for row in bindings if isinstance(row, Mapping)]
    if len(task_ids) != len(bindings) or not all(task_ids) or len(set(task_ids)) != len(task_ids):
        raise Sam31TrackSelectionReviewError("sam31_review_candidate_invalid")
    for binding in bindings:
        _verify_record(binding.get("task_freeze"), root=root)
        _verify_record(binding.get("source_track_result"), root=root)
        _verify_record(binding.get("camera_contract"), root=root)
    if [str(row.get("task_id") or "") for row in review_media] != task_ids:
        raise Sam31TrackSelectionReviewError("sam31_review_candidate_invalid")
    for task in review_media:
        frames = task.get("frames")
        if (
            not isinstance(frames, list)
            or not frames
            or task.get("camera_count") != len(frames)
            or len({str(row.get("camera_id") or "") for row in frames}) != len(frames)
        ):
            raise Sam31TrackSelectionReviewError("sam31_review_candidate_invalid")
        for frame in frames:
            source_path = _verify_record(frame.get("source_image"), root=root)
            mask_path = _verify_record(frame.get("selected_mask"), root=root)
            overlay_path = _verify_record(frame.get("overlay"), root=root)
            with (
                Image.open(source_path) as source_file,
                Image.open(mask_path) as mask_file,
                Image.open(overlay_path) as overlay_file,
            ):
                source = source_file.convert("RGB")
                mask = mask_file.convert("L")
                overlay = overlay_file.convert("RGB")
                histogram = mask.histogram()
                foreground = sum(histogram[1:])
                if (
                    source_file.format != "PNG"
                    or mask_file.format != "PNG"
                    or overlay_file.format != "PNG"
                    or source.size != mask.size
                    or source.size != overlay.size
                    or foreground != frame.get("foreground_pixel_count")
                    or foreground != histogram[255]
                ):
                    raise Sam31TrackSelectionReviewError("sam31_review_media_content_invalid")
                color = Image.new("RGB", source.size, (255, 0, 160))
                alpha = mask.point(lambda value: 128 if value else 0)
                expected_overlay = Image.composite(color, source, alpha)
                if ImageChops.difference(expected_overlay, overlay).getbbox() is not None:
                    raise Sam31TrackSelectionReviewError("sam31_review_media_content_invalid")


def load_validated_sam31_track_selection_review_candidate(
    candidate_path: str | Path,
) -> tuple[Path, dict[str, Any]]:
    """Reopen a candidate and verify every digest-bound source/mask/overlay byte."""

    candidate_file, candidate = _read(candidate_path, code="sam31_review_candidate_invalid")
    _validate_candidate_file(candidate_file, candidate)
    return candidate_file, candidate


def resolve_sam31_review_media_path(
    record: Mapping[str, Any],
    *,
    candidate_path: str | Path,
) -> Path:
    """Resolve and rehash one media record under its exact candidate root."""

    candidate_file = Path(candidate_path).expanduser().resolve()
    if candidate_file.is_symlink() or not candidate_file.is_file():
        raise Sam31TrackSelectionReviewError("sam31_review_candidate_invalid")
    return _verify_record(record, root=candidate_file.parent)


def build_sam31_ai_visual_review_input(
    *,
    candidate_path: str | Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build the exact digest-rehashed multimodal SDK input and frame inventory."""

    candidate_file, candidate = load_validated_sam31_track_selection_review_candidate(
        candidate_path
    )
    binding_by_task = {
        str(row["task_id"]): row for row in candidate["selection_bindings"]
    }
    content: list[dict[str, Any]] = [{"type": "input_text", "text": _AI_REVIEW_PROMPT}]
    frame_inventory: list[dict[str, Any]] = []
    for task in candidate["review_media"]:
        task_id = str(task["task_id"])
        binding = binding_by_task[task_id]
        for frame in task["frames"]:
            camera_id = str(frame["camera_id"])
            overlay_path = resolve_sam31_review_media_path(
                frame["overlay"], candidate_path=candidate_file
            )
            overlay_bytes = overlay_path.read_bytes()
            overlay_record = _record(overlay_path)
            if overlay_record["sha256"] != frame["overlay"]["sha256"]:
                raise Sam31TrackSelectionReviewError(
                    "sam31_ai_review_overlay_changed_before_transport"
                )
            metadata = {
                "task_id": task_id,
                "camera_id": camera_id,
                "selected_track_ids": list(binding["selected_track_ids"]),
                "selected_track_labels": list(binding["selected_track_labels"]),
                "foreground_pixel_count": int(frame["foreground_pixel_count"]),
                "overlay_sha256": overlay_record["sha256"],
                "overlay_width_height_bound_by_candidate": True,
            }
            content.append({"type": "input_text", "text": canonical_json(metadata)})
            content.append(
                {
                    "type": "input_image",
                    "image_url": (
                        "data:image/png;base64,"
                        + base64.b64encode(overlay_bytes).decode("ascii")
                    ),
                    "detail": "high",
                }
            )
            frame_inventory.append({**metadata, "overlay": overlay_record})
    return [{"role": "user", "content": content}], frame_inventory


def validate_sam31_ai_visual_review_rights(
    *,
    candidate_path: str | Path,
    rights_attestation_path: str | Path,
) -> tuple[Path, dict[str, Any]]:
    """Require human-issued rights for exact overlay disclosure to OpenAI."""

    candidate_file, candidate = load_validated_sam31_track_selection_review_candidate(
        candidate_path
    )
    _input, frame_inventory = build_sam31_ai_visual_review_input(
        candidate_path=candidate_file
    )
    rights_path, rights = _read(
        rights_attestation_path,
        code="sam31_ai_review_rights_attestation_invalid",
    )
    overlay_sha256 = sorted(str(row["overlay_sha256"]) for row in frame_inventory)
    if (
        rights.get("schema_version") != AI_RIGHTS_SCHEMA_VERSION
        or rights.get("status") != "accepted_for_private_derived_visual_review"
        or rights.get("attestation_digest")
        != canonical_digest(rights, digest_field="attestation_digest")
        or rights.get("source_candidate_digest") != candidate["candidate_digest"]
        or rights.get("review_media_digest")
        != canonical_json_digest(candidate["review_media"])
        or rights.get("review_frame_count") != AI_REVIEW_FRAME_COUNT
        or len(overlay_sha256) != AI_REVIEW_FRAME_COUNT
        or len(set(overlay_sha256)) != AI_REVIEW_FRAME_COUNT
        or rights.get("overlay_sha256") != overlay_sha256
        or rights.get("program_id") != "arm-decision-proof-v1"
        or rights.get("declared_use_scope") != AI_REVIEW_DECLARED_USE
        or rights.get("provider_id") != "openai"
        or rights.get("runtime") != "openai_agents_sdk"
        or rights.get("model") != AI_REVIEW_MODEL
        or rights.get("private_derived_frame_disclosure_authorized") is not True
        or rights.get("api_data_training_policy")
        != "not_used_for_training_by_default_unless_opted_in"
        or rights.get("training_opt_in") is not False
        or rights.get("default_abuse_monitoring_retention_max_days") != 30
        or rights.get("responses_application_state_retention_max_days") != 30
        or rights.get("image_manual_csam_review_retention_possible") is not True
        or rights.get("zero_data_retention_claimed") is not False
        or rights.get("response_store") is not False
        or rights.get("tracing_disabled") is not True
        or rights.get("trace_sensitive_data_included") is not False
        or rights.get("openai_api_data_use_terms_accepted") is not True
        or rights.get("openai_image_safety_review_terms_accepted") is not True
        or rights.get("frame_redistribution_authorized") is not False
        or rights.get("frame_publication_authorized") is not False
        or rights.get("derived_overlay_pngs_only") is not True
        or rights.get("raw_source_splat_or_dataset_bytes_included") is not False
        or rights.get("max_inference_spend_usd") != AI_REVIEW_MAX_COST_USD
        or rights.get("agent_accepted_terms") is not False
        or rights.get("issued_by_agent") is not False
        or rights.get("accepted_by") != AI_REVIEW_ACCEPTED_BY
        or not str(rights.get("accepted_on") or "").strip()
        or not str(rights.get("human_authority_reference") or "").strip()
    ):
        raise Sam31TrackSelectionReviewError("sam31_ai_review_rights_attestation_invalid")
    return rights_path, rights


def materialize_sam31_ai_visual_review_rights(
    *,
    candidate_path: str | Path,
    accepted_by: str,
    accepted_on: str,
    human_authority_reference: str,
    output_path: str | Path,
) -> dict[str, Any]:
    """Retain an exact human authorization after the 16 overlays exist."""

    candidate_file, candidate = load_validated_sam31_track_selection_review_candidate(
        candidate_path
    )
    _input, frame_inventory = build_sam31_ai_visual_review_input(
        candidate_path=candidate_file
    )
    overlay_sha256 = sorted(str(row["overlay_sha256"]) for row in frame_inventory)
    if (
        accepted_by != AI_REVIEW_ACCEPTED_BY
        or not str(accepted_on).strip()
        or not str(human_authority_reference).strip()
        or len(overlay_sha256) != AI_REVIEW_FRAME_COUNT
        or len(set(overlay_sha256)) != AI_REVIEW_FRAME_COUNT
    ):
        raise Sam31TrackSelectionReviewError("sam31_ai_review_rights_authority_invalid")
    rights: dict[str, Any] = {
        "schema_version": AI_RIGHTS_SCHEMA_VERSION,
        "status": "accepted_for_private_derived_visual_review",
        "program_id": "arm-decision-proof-v1",
        "declared_use_scope": AI_REVIEW_DECLARED_USE,
        "source_candidate_digest": candidate["candidate_digest"],
        "review_media_digest": canonical_json_digest(candidate["review_media"]),
        "review_frame_count": AI_REVIEW_FRAME_COUNT,
        "overlay_sha256": overlay_sha256,
        "provider_id": "openai",
        "runtime": "openai_agents_sdk",
        "model": AI_REVIEW_MODEL,
        "private_derived_frame_disclosure_authorized": True,
        "api_data_training_policy": "not_used_for_training_by_default_unless_opted_in",
        "training_opt_in": False,
        "default_abuse_monitoring_retention_max_days": 30,
        "responses_application_state_retention_max_days": 30,
        "image_manual_csam_review_retention_possible": True,
        "zero_data_retention_claimed": False,
        "response_store": False,
        "tracing_disabled": True,
        "trace_sensitive_data_included": False,
        "openai_api_data_use_terms_accepted": True,
        "openai_image_safety_review_terms_accepted": True,
        "frame_redistribution_authorized": False,
        "frame_publication_authorized": False,
        "derived_overlay_pngs_only": True,
        "raw_source_splat_or_dataset_bytes_included": False,
        "max_inference_spend_usd": AI_REVIEW_MAX_COST_USD,
        "agent_accepted_terms": False,
        "issued_by_agent": False,
        "accepted_by": accepted_by,
        "accepted_on": str(accepted_on).strip(),
        "human_authority_reference": str(human_authority_reference).strip(),
        "attestation_digest": "",
    }
    rights["attestation_digest"] = canonical_digest(
        rights, digest_field="attestation_digest"
    )
    destination = Path(output_path).expanduser().resolve()
    if destination.exists() or destination.is_symlink():
        raise Sam31TrackSelectionReviewError("sam31_review_output_exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(canonical_json(rights) + "\n", encoding="utf-8")
    validate_sam31_ai_visual_review_rights(
        candidate_path=candidate_file,
        rights_attestation_path=destination,
    )
    return rights


def validate_sam31_ai_structured_decision(
    *,
    structured_output: Mapping[str, Any],
    frame_inventory: Sequence[Mapping[str, Any]],
) -> tuple[str, list[str]]:
    """Derive the decision from the exact frame set instead of trusting provider prose."""

    expected = {
        (str(row.get("task_id") or ""), str(row.get("camera_id") or "")): row
        for row in frame_inventory
    }
    raw_frames = structured_output.get("frames")
    if (
        not isinstance(raw_frames, list)
        or not expected
        or len(expected) != len(frame_inventory)
    ):
        return "rejected", ["structured_frame_set_invalid"]
    observed: dict[tuple[str, str], Mapping[str, Any]] = {}
    blockers: list[str] = []
    for raw in raw_frames:
        if not isinstance(raw, Mapping):
            blockers.append("structured_frame_decision_invalid")
            continue
        key = (str(raw.get("task_id") or ""), str(raw.get("camera_id") or ""))
        if not all(key):
            blockers.append("structured_frame_identity_missing")
            continue
        if key in observed:
            blockers.append(f"duplicate_frame_decision:{key[0]}:{key[1]}")
        observed[key] = raw
    for task_id, camera_id in sorted(expected.keys() - observed.keys()):
        blockers.append(f"missing_frame_decision:{task_id}:{camera_id}")
    for task_id, camera_id in sorted(observed.keys() - expected.keys()):
        blockers.append(f"unexpected_frame_decision:{task_id}:{camera_id}")
    for key in sorted(expected.keys() & observed.keys()):
        source = expected[key]
        row = observed[key]
        empty = int(source["foreground_pixel_count"]) == 0
        expected_visibility = (
            "absent_or_fully_occluded" if empty else "visible_or_partially_visible"
        )
        if row.get("target_visibility") != expected_visibility:
            blockers.append(
                f"{'empty_mask_target_visible' if empty else 'nonempty_mask_target_not_visible'}:"
                f"{key[0]}:{key[1]}"
            )
        if row.get("selected_mask_matches_target") is not True or row.get("decision") != "accepted":
            blockers.append(f"frame_rejected:{key[0]}:{key[1]}")
    derived = "accepted" if not blockers else "rejected"
    if structured_output.get("decision") != derived:
        blockers.append("global_decision_inconsistent_with_frame_decisions")
    return derived, sorted(set(blockers))


def _contained_evidence_path(*, execution_root: Path, relative_path: object) -> Path:
    path = (execution_root / str(relative_path or "")).resolve()
    try:
        path.relative_to(execution_root)
    except ValueError as exc:
        raise Sam31TrackSelectionReviewError(
            "sam31_review_execution_receipt_invalid"
        ) from exc
    if path.is_symlink() or not path.is_file():
        raise Sam31TrackSelectionReviewError("sam31_review_execution_receipt_invalid")
    return path


def _validate_ai_execution_receipt(
    *,
    execution_path: Path,
    execution: Mapping[str, Any],
    candidate_file: Path,
    candidate: Mapping[str, Any],
) -> None:
    reviewer = execution.get("reviewer")
    execution_candidate = execution.get("candidate")
    structured_output = execution.get("structured_output")
    manifest = execution.get("inference_reservation_manifest")
    manifest_record = execution.get("inference_reservation_manifest_record")
    rights_record = execution.get("rights_attestation")
    if (
        execution.get("schema_version") != AI_EXECUTION_SCHEMA_VERSION
        or execution.get("status") != "ai_visual_review_execution_completed"
        or execution.get("execution_receipt_digest")
        != canonical_digest(execution, digest_field="execution_receipt_digest")
        or not isinstance(reviewer, Mapping)
        or reviewer.get("kind") != "ai"
        or reviewer.get("identity") != AI_REVIEWER_ID
        or reviewer.get("method") != AI_REVIEW_METHOD
        or reviewer.get("runtime") != "openai_agents_sdk"
        or reviewer.get("model") != AI_REVIEW_MODEL
        or reviewer.get("model_version") != AI_REVIEW_MODEL
        or not all(
            str(reviewer.get(field) or "").strip()
            for field in ("model", "model_version", "sdk_version")
        )
        or execution.get("capability") != AI_REVIEW_CAPABILITY
        or not str(execution.get("run_id") or "").strip()
        or not isinstance(execution_candidate, Mapping)
        or _record(candidate_file)
        != {
            key: execution_candidate.get(key)
            for key in ("path", "size_bytes", "sha256")
        }
        or execution_candidate.get("candidate_digest") != candidate["candidate_digest"]
        or execution.get("review_media_digest")
        != canonical_json_digest(candidate["review_media"])
        or execution.get("review_frame_count")
        != sum(len(row["frames"]) for row in candidate["review_media"])
        or execution.get("provider_called") is not True
        or execution.get("provider") != "openai"
        or execution.get("response_store") is not False
        or execution.get("tracing_disabled") is not True
        or execution.get("trace_sensitive_data_included") is not False
        or execution.get("raw_secret_values_recorded") is not False
        or not str(execution.get("reviewed_at") or "").strip()
        or execution.get("decision") not in {"accepted", "rejected"}
        or not isinstance(structured_output, Mapping)
        or execution.get("structured_output_digest")
        != canonical_digest(structured_output)
        or not isinstance(manifest, Mapping)
        or not isinstance(manifest_record, Mapping)
        or not isinstance(rights_record, Mapping)
    ):
        raise Sam31TrackSelectionReviewError("sam31_review_execution_receipt_invalid")

    try:
        rights_path, rights = validate_sam31_ai_visual_review_rights(
            candidate_path=candidate_file,
            rights_attestation_path=str(rights_record.get("path") or ""),
        )
    except Sam31TrackSelectionReviewError as exc:
        raise Sam31TrackSelectionReviewError(
            "sam31_review_execution_receipt_invalid"
        ) from exc
    if (
        _record(rights_path)
        != {key: rights_record.get(key) for key in ("path", "size_bytes", "sha256")}
        or rights.get("attestation_digest") != rights_record.get("attestation_digest")
    ):
        raise Sam31TrackSelectionReviewError("sam31_review_execution_receipt_invalid")

    expected_input, expected_inventory = build_sam31_ai_visual_review_input(
        candidate_path=candidate_file
    )
    if (
        execution.get("frame_inventory") != expected_inventory
        or len(expected_inventory) != AI_REVIEW_FRAME_COUNT
        or execution.get("input_digest") != canonical_digest({"input": expected_input})
        or execution.get("input_transport") != "digest_rehashed_png_data_urls"
    ):
        raise Sam31TrackSelectionReviewError("sam31_review_execution_receipt_invalid")
    derived_decision, derived_blockers = validate_sam31_ai_structured_decision(
        structured_output=structured_output,
        frame_inventory=expected_inventory,
    )
    if (
        execution.get("decision") != derived_decision
        or execution.get("blockers") != derived_blockers
    ):
        raise Sam31TrackSelectionReviewError("sam31_review_execution_receipt_invalid")

    execution_root = execution_path.parent.resolve()
    manifest_path = _contained_evidence_path(
        execution_root=execution_root,
        relative_path="inference_reservations/manifest.json",
    )
    if _record(manifest_path) != dict(manifest_record):
        raise Sam31TrackSelectionReviewError("sam31_review_execution_receipt_invalid")
    _manifest_path, reopened_manifest = _read(
        manifest_path, code="sam31_review_execution_receipt_invalid"
    )
    rows = reopened_manifest.get("reservations")
    if (
        reopened_manifest != manifest
        or manifest.get("schema_version") != INFERENCE_RESERVATION_MANIFEST_SCHEMA_VERSION
        or manifest.get("inference_reservation_manifest_digest")
        != canonical_digest(
            manifest, digest_field="inference_reservation_manifest_digest"
        )
        or manifest.get("run_id") != execution["run_id"]
        or manifest.get("reservation_count") != 1
        or manifest.get("in_flight_unknown_count") != 0
        or not isinstance(manifest.get("reserved_max_cost_usd"), (int, float))
        or not 0 < float(manifest["reserved_max_cost_usd"]) <= AI_REVIEW_MAX_COST_USD
        or not isinstance(rows, list)
        or len(rows) != 1
        or rows[0].get("status") != "completed"
    ):
        raise Sam31TrackSelectionReviewError("sam31_review_execution_receipt_invalid")
    row = rows[0]
    reservation_path = _contained_evidence_path(
        execution_root=execution_root,
        relative_path=row.get("reservation_path"),
    )
    completion_path = _contained_evidence_path(
        execution_root=execution_root,
        relative_path=row.get("completion_path"),
    )
    _reservation_path, reservation = _read(
        reservation_path, code="sam31_review_execution_receipt_invalid"
    )
    _completion_path, completion = _read(
        completion_path, code="sam31_review_execution_receipt_invalid"
    )
    expected_reservation_id = canonical_digest(
        {
            "run_id": execution["run_id"],
            "capability": AI_REVIEW_CAPABILITY,
            "model": reviewer["model"],
            "input_digest": execution["input_digest"],
            "max_turns": reservation.get("max_turns"),
            "max_output_tokens": reservation.get("max_output_tokens"),
        }
    )
    if (
        reservation.get("schema_version") != INFERENCE_RESERVATION_SCHEMA_VERSION
        or reservation.get("reservation_id") != expected_reservation_id
        or reservation.get("run_id") != execution["run_id"]
        or reservation.get("capability") != AI_REVIEW_CAPABILITY
        or reservation.get("model") != reviewer["model"]
        or reservation.get("input_digest") != execution["input_digest"]
        or reservation.get("input_kind") != "multimodal"
        or reservation.get("input_token_ceiling") != AI_REVIEW_INPUT_TOKEN_CEILING
        or not isinstance(reservation.get("projected_max_cost_usd"), (int, float))
        or not 0 < float(reservation["projected_max_cost_usd"]) <= AI_REVIEW_MAX_COST_USD
        or reservation.get("inference_reservation_digest")
        != canonical_digest(reservation, digest_field="inference_reservation_digest")
        or row.get("reservation_id") != expected_reservation_id
        or row.get("reservation_digest")
        != reservation.get("inference_reservation_digest")
        or completion.get("schema_version") != INFERENCE_COMPLETION_SCHEMA_VERSION
        or completion.get("reservation_id") != expected_reservation_id
        or completion.get("run_id") != execution["run_id"]
        or completion.get("capability") != AI_REVIEW_CAPABILITY
        or completion.get("provider") != "openai"
        or completion.get("model") != reviewer["model"]
        or completion.get("agents_sdk_version") != reviewer["sdk_version"]
        or completion.get("structured_output_digest")
        != execution["structured_output_digest"]
        or completion.get("inference_completion_digest")
        != canonical_digest(completion, digest_field="inference_completion_digest")
        or row.get("completion_digest") != completion.get("inference_completion_digest")
    ):
        raise Sam31TrackSelectionReviewError("sam31_review_execution_receipt_invalid")


def _selection_bindings(
    *,
    task_freeze_paths: Sequence[str | Path],
    task_inputs: Mapping[str, Mapping[str, Any]],
    selected_track_ids_by_task: Mapping[str, Sequence[str]],
) -> list[dict[str, Any]]:
    bindings: list[dict[str, Any]] = []
    for freeze_value in task_freeze_paths:
        freeze_path, freeze = _read(freeze_value, code="sam31_review_task_freeze_invalid")
        task_id = str(freeze.get("task_id") or "")
        raw_input = task_inputs.get(task_id)
        selected = sorted(set(str(item) for item in selected_track_ids_by_task.get(task_id, [])))
        if (
            not task_id
            or not isinstance(raw_input, Mapping)
            or not selected
            or freeze.get("task_freeze_digest")
            != canonical_digest(freeze, digest_field="task_freeze_digest")
        ):
            raise Sam31TrackSelectionReviewError("sam31_review_selection_binding_invalid")
        tracks_path, tracks = _read(
            str(raw_input.get("source_track_result_path") or ""),
            code="sam31_review_source_tracks_invalid",
        )
        cameras_path = Path(str(raw_input.get("camera_contract_path") or "")).expanduser().resolve()
        image_root = Path(str(raw_input.get("source_image_root") or "")).expanduser().resolve()
        camera_frame_map = raw_input.get("camera_frame_map")
        if (
            cameras_path.is_symlink()
            or not cameras_path.is_file()
            or image_root.is_symlink()
            or not image_root.is_dir()
            or not isinstance(camera_frame_map, Mapping)
            or not camera_frame_map
        ):
            raise Sam31TrackSelectionReviewError("sam31_review_task_inputs_invalid")
        available = {
            str(row.get("track_id") or "")
            for row in tracks.get("track_registry") or []
            if isinstance(row, Mapping)
        }
        if (
            tracks.get("schema_version") != "semantic_source_track_import_result.v1"
            or tracks.get("status") != "completed"
            or tracks.get("result_digest")
            != canonical_json_digest(
                {key: value for key, value in tracks.items() if key != "result_digest"}
            )
            or any(item not in available for item in selected)
        ):
            raise Sam31TrackSelectionReviewError("sam31_review_source_tracks_invalid")
        bindings.append(
            {
                "task_id": task_id,
                "task_freeze": {
                    **_record(freeze_path),
                    "task_freeze_digest": freeze["task_freeze_digest"],
                },
                "source_track_result": {
                    **_record(tracks_path),
                    "result_digest": tracks["result_digest"],
                },
                "camera_contract": _record(cameras_path),
                "source_image_root": str(image_root),
                "camera_frame_map": {
                    str(camera_id): str(frame_id)
                    for camera_id, frame_id in sorted(camera_frame_map.items())
                },
                "selected_track_ids": selected,
                "selected_track_labels": sorted(
                    str(row.get("label") or "")
                    for row in tracks["track_registry"]
                    if row.get("track_id") in selected
                ),
            }
        )
    if (
        not 1 <= len(bindings) <= 5
        or len({row["task_id"] for row in bindings}) != len(bindings)
        or set(task_inputs) != {row["task_id"] for row in bindings}
        or set(selected_track_ids_by_task) != {row["task_id"] for row in bindings}
    ):
        raise Sam31TrackSelectionReviewError("sam31_review_task_set_invalid")
    return sorted(bindings, key=lambda row: row["task_id"])


def _write_overlay(*, image_path: Path, mask_path: Path, output_path: Path) -> None:
    with Image.open(image_path) as source_image, Image.open(mask_path) as source_mask:
        image = source_image.convert("RGB")
        mask = source_mask.convert("L")
        if image.size != mask.size:
            raise Sam31TrackSelectionReviewError("sam31_review_overlay_dimensions_invalid")
        color = Image.new("RGB", image.size, (255, 0, 160))
        alpha = mask.point(lambda value: 128 if value else 0)
        overlay = Image.composite(color, image, alpha)
        overlay.save(output_path, format="PNG", optimize=False)


def materialize_sam31_track_selection_review_candidate(
    *,
    task_freeze_paths: Sequence[str | Path],
    task_inputs: Mapping[str, Mapping[str, Any]],
    selected_track_ids_by_task: Mapping[str, Sequence[str]],
    output_root: str | Path,
) -> dict[str, Any]:
    """Render exact selected masks over calibrated frames, pending visual review."""

    from .public_scene_calibrated_object_masks import (
        _camera_rows,
        _decode_union,
        _frame_map,
        _track_map,
        _verified_source_tracks,
    )

    output = Path(output_root).expanduser().resolve()
    if output.is_symlink() or (output.exists() and any(output.iterdir())):
        raise Sam31TrackSelectionReviewError("sam31_review_output_not_empty")
    bindings = _selection_bindings(
        task_freeze_paths=task_freeze_paths,
        task_inputs=task_inputs,
        selected_track_ids_by_task=selected_track_ids_by_task,
    )
    masks_root = output / "candidate_masks"
    masks_root.mkdir(parents=True)
    review_rows: list[dict[str, Any]] = []
    for binding in bindings:
        task_id = binding["task_id"]
        task_input = task_inputs[task_id]
        tracks_path = Path(str(task_input["source_track_result_path"])).expanduser().resolve()
        cameras_path = Path(str(task_input["camera_contract_path"])).expanduser().resolve()
        image_root = Path(str(task_input["source_image_root"])).expanduser().resolve()
        source_tracks = _verified_source_tracks(tracks_path)
        tracks = _track_map(source_tracks)
        frames = _frame_map(source_tracks)
        cameras = _camera_rows(cameras_path)
        camera_frame_map = {
            str(camera_id): str(frame_id)
            for camera_id, frame_id in task_input["camera_frame_map"].items()
        }
        selected = set(binding["selected_track_ids"])
        if (
            set(camera_frame_map) != set(cameras)
            or set(camera_frame_map.values()) != set(frames)
            or any(item not in tracks for item in selected)
        ):
            raise Sam31TrackSelectionReviewError("sam31_review_camera_frame_set_invalid")
        media_root = output / "review_media" / task_id
        media_root.mkdir(parents=True)
        mask_task_root = masks_root / task_id
        mask_task_root.mkdir(parents=True)
        frame_rows: list[dict[str, Any]] = []
        for camera_id in sorted(cameras):
            source_frame_id = camera_frame_map[camera_id]
            frame = frames[source_frame_id]
            image_path = image_root / f"{camera_id}.png"
            if (
                not image_path.is_file()
                or image_path.is_symlink()
                or _sha256(image_path) != frame.get("source_frame_digest")
                or canonical_json_digest(cameras[camera_id]) != frame.get("camera_record_digest")
            ):
                raise Sam31TrackSelectionReviewError("sam31_review_source_image_invalid")
            with Image.open(image_path) as image:
                expected_size = (
                    int(cameras[camera_id]["intrinsics"]["width"]),
                    int(cameras[camera_id]["intrinsics"]["height"]),
                )
                if (
                    image.format != "PNG"
                    or image.size != expected_size
                    or image.size != (int(frame["width"]), int(frame["height"]))
                ):
                    raise Sam31TrackSelectionReviewError("sam31_review_source_image_invalid")
            mask = _decode_union(
                frame,
                selected_track_ids=selected,
                code=f"sam31_review_selected_track_missing:{task_id}:{camera_id}",
                allow_empty_selected_tracks=True,
            )
            mask_path = mask_task_root / f"{camera_id}.png"
            Image.fromarray(mask, mode="L").save(mask_path, format="PNG", optimize=False)
            overlay_path = media_root / f"{camera_id}.png"
            _write_overlay(image_path=image_path, mask_path=mask_path, output_path=overlay_path)
            frame_rows.append(
                {
                    "camera_id": camera_id,
                    "source_image": _record(image_path, root=output),
                    "selected_mask": _record(mask_path, root=output),
                    "overlay": _record(overlay_path, root=output),
                    "foreground_pixel_count": int((mask != 0).sum()),
                }
            )
        review_rows.append(
            {"task_id": task_id, "camera_count": len(frame_rows), "frames": frame_rows}
        )
    candidate: dict[str, Any] = {
        "schema_version": CANDIDATE_SCHEMA_VERSION,
        "status": "selected_track_overlays_materialized_pending_visual_review",
        "task_count": len(bindings),
        "selection_bindings": bindings,
        "candidate_masks_root": str(masks_root),
        "review_media": review_rows,
        "overlay_policy": {
            "selected_pixels_rgb": [255, 0, 160],
            "selected_pixels_alpha_255": 128,
            "source_pixels_resampled": False,
        },
        "claim_boundary": {
            "human_review_completed": False,
            "object_identity_qualified": False,
            "gaussian_ownership_qualified": False,
            "physical_evidence": False,
        },
        "candidate_digest": "",
    }
    candidate["candidate_digest"] = canonical_digest(candidate, digest_field="candidate_digest")
    output.mkdir(parents=True, exist_ok=True)
    destination = output / f"{CANDIDATE_SCHEMA_VERSION}.json"
    destination.write_text(canonical_json(candidate) + "\n", encoding="utf-8")
    _validate_candidate_file(destination, candidate)
    return candidate


def seal_sam31_track_selection_review(
    *,
    candidate_path: str | Path,
    reviewed_by: str,
    reviewed_on: str,
    output_path: str | Path,
) -> dict[str, Any]:
    """Accept every selected task track after inspecting the rendered overlays."""

    candidate_file, candidate = _read(candidate_path, code="sam31_review_candidate_invalid")
    _validate_candidate_file(candidate_file, candidate)
    if not str(reviewed_by).strip() or not str(reviewed_on).strip():
        raise Sam31TrackSelectionReviewError("sam31_review_candidate_invalid")
    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "status": "selected_tracks_human_review_accepted",
        "candidate": {
            **_record(candidate_file),
            "candidate_digest": candidate["candidate_digest"],
        },
        "selection_bindings": candidate["selection_bindings"],
        "task_count": candidate["task_count"],
        "reviewed_by": str(reviewed_by).strip(),
        "reviewed_on": str(reviewed_on).strip(),
        "all_selected_tracks_accepted": True,
        "agent_selected_tracks_without_human_review": False,
        "claim_boundary": {
            "track_selection_reviewed": True,
            "object_identity_qualified": False,
            "gaussian_ownership_qualified": False,
            "physical_evidence": False,
        },
        "receipt_digest": "",
    }
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    destination = Path(output_path).expanduser().resolve()
    if destination.exists() or destination.is_symlink():
        raise Sam31TrackSelectionReviewError("sam31_review_output_exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(canonical_json(receipt) + "\n", encoding="utf-8")
    return receipt


def seal_sam31_track_selection_ai_review(
    *,
    candidate_path: str | Path,
    review_execution_receipt_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Seal a production AI review only from a retained Agents SDK execution."""

    candidate_file, candidate = load_validated_sam31_track_selection_review_candidate(
        candidate_path
    )
    execution_path, execution = _read(
        review_execution_receipt_path,
        code="sam31_review_execution_receipt_invalid",
    )
    reviewer = execution.get("reviewer")
    decision = str(execution.get("decision") or "")
    _validate_ai_execution_receipt(
        execution_path=execution_path,
        execution=execution,
        candidate_file=candidate_file,
        candidate=candidate,
    )
    if not isinstance(reviewer, Mapping):  # narrowed by validation above
        raise Sam31TrackSelectionReviewError("sam31_review_execution_receipt_invalid")
    accepted = decision == "accepted"
    review_media = candidate["review_media"]
    receipt: dict[str, Any] = {
        "schema_version": AI_RECEIPT_SCHEMA_VERSION,
        "status": f"selected_tracks_ai_visual_review_{decision}",
        "candidate": {
            **_record(candidate_file),
            "candidate_digest": candidate["candidate_digest"],
        },
        "selection_bindings": candidate["selection_bindings"],
        "task_count": candidate["task_count"],
        "reviewer": dict(reviewer),
        "review_execution_receipt": {
            **_record(execution_path),
            "execution_receipt_digest": execution["execution_receipt_digest"],
        },
        "reviewed_at": execution["reviewed_at"],
        "decision": decision,
        "review_scope": {
            "candidate_digest": candidate["candidate_digest"],
            "review_media_digest": canonical_json_digest(review_media),
            "review_frame_count": sum(len(row["frames"]) for row in review_media),
        },
        "all_selected_tracks_accepted": accepted,
        "claim_boundary": {
            "track_selection_reviewed": accepted,
            "human_review_completed": False,
            "ai_visual_review_completed": True,
            "object_identity_qualified": False,
            "gaussian_ownership_qualified": False,
            "physical_evidence": False,
        },
        "receipt_digest": "",
    }
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    destination = Path(output_path).expanduser().resolve()
    if destination.exists() or destination.is_symlink():
        raise Sam31TrackSelectionReviewError("sam31_review_output_exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(canonical_json(receipt) + "\n", encoding="utf-8")
    return receipt


def validate_sam31_track_selection_review(
    *,
    receipt_path: str | Path,
    task_freeze_paths: Sequence[str | Path],
    task_inputs: Mapping[str, Mapping[str, Any]],
    selected_track_ids_by_task: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    """Reopen a review receipt and prove it accepts these exact selection inputs."""

    _path, receipt = _read(receipt_path, code="sam31_review_receipt_invalid")
    schema_version = receipt.get("schema_version")
    if schema_version == RECEIPT_SCHEMA_VERSION:
        if (
            receipt.get("status") != "selected_tracks_human_review_accepted"
            or receipt.get("all_selected_tracks_accepted") is not True
            or receipt.get("agent_selected_tracks_without_human_review") is not False
            or receipt.get("receipt_digest")
            != canonical_digest(receipt, digest_field="receipt_digest")
        ):
            raise Sam31TrackSelectionReviewError("sam31_review_receipt_invalid")
    elif schema_version == AI_RECEIPT_SCHEMA_VERSION:
        reviewer = receipt.get("reviewer")
        if (
            receipt.get("status") != "selected_tracks_ai_visual_review_accepted"
            or receipt.get("decision") != "accepted"
            or receipt.get("all_selected_tracks_accepted") is not True
            or not isinstance(reviewer, Mapping)
            or reviewer.get("kind") != "ai"
            or reviewer.get("identity") != AI_REVIEWER_ID
            or reviewer.get("method") != AI_REVIEW_METHOD
            or reviewer.get("runtime") != "openai_agents_sdk"
            or reviewer.get("model") != AI_REVIEW_MODEL
            or reviewer.get("model_version") != AI_REVIEW_MODEL
            or not all(
                str(reviewer.get(field) or "").strip()
                for field in ("model", "model_version", "sdk_version")
            )
            or not str(receipt.get("reviewed_at") or "").strip()
            or not isinstance(receipt.get("review_execution_receipt"), Mapping)
            or any(
                field in receipt
                for field in (
                    "reviewed_by",
                    "reviewed_on",
                    "agent_selected_tracks_without_human_review",
                )
            )
            or receipt.get("claim_boundary", {}).get("human_review_completed") is not False
            or receipt.get("claim_boundary", {}).get("ai_visual_review_completed") is not True
            or receipt.get("receipt_digest")
            != canonical_digest(receipt, digest_field="receipt_digest")
        ):
            raise Sam31TrackSelectionReviewError("sam31_review_receipt_invalid")
    else:
        raise Sam31TrackSelectionReviewError("sam31_review_receipt_invalid")
    candidate_record = receipt.get("candidate")
    if not isinstance(candidate_record, Mapping):
        raise Sam31TrackSelectionReviewError("sam31_review_receipt_invalid")
    candidate_path, candidate = _read(
        str(candidate_record.get("path") or ""), code="sam31_review_candidate_invalid"
    )
    if (
        _record(candidate_path)
        != {key: candidate_record.get(key) for key in ("path", "size_bytes", "sha256")}
        or candidate.get("candidate_digest") != candidate_record.get("candidate_digest")
        or candidate.get("candidate_digest")
        != canonical_digest(candidate, digest_field="candidate_digest")
    ):
        raise Sam31TrackSelectionReviewError("sam31_review_candidate_invalid")
    _validate_candidate_file(candidate_path, candidate)
    if schema_version == AI_RECEIPT_SCHEMA_VERSION:
        review_scope = receipt.get("review_scope")
        execution_record = receipt.get("review_execution_receipt")
        if (
            not isinstance(review_scope, Mapping)
            or review_scope.get("candidate_digest") != candidate["candidate_digest"]
            or review_scope.get("review_media_digest")
            != canonical_json_digest(candidate["review_media"])
            or review_scope.get("review_frame_count")
            != sum(len(row["frames"]) for row in candidate["review_media"])
            or not isinstance(execution_record, Mapping)
        ):
            raise Sam31TrackSelectionReviewError("sam31_review_receipt_invalid")
        execution_path, execution = _read(
            str(execution_record.get("path") or ""),
            code="sam31_review_execution_receipt_invalid",
        )
        if (
            _record(execution_path)
            != {
                key: execution_record.get(key)
                for key in ("path", "size_bytes", "sha256")
            }
            or execution.get("decision") != "accepted"
            or execution.get("execution_receipt_digest")
            != execution_record.get("execution_receipt_digest")
        ):
            raise Sam31TrackSelectionReviewError("sam31_review_execution_receipt_invalid")
        _validate_ai_execution_receipt(
            execution_path=execution_path,
            execution=execution,
            candidate_file=candidate_path,
            candidate=candidate,
        )
    expected = _selection_bindings(
        task_freeze_paths=task_freeze_paths,
        task_inputs=task_inputs,
        selected_track_ids_by_task=selected_track_ids_by_task,
    )
    if (
        receipt.get("selection_bindings") != expected
        or candidate.get("selection_bindings") != expected
    ):
        raise Sam31TrackSelectionReviewError("sam31_review_selection_mismatch")
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    candidate = commands.add_parser("candidate")
    candidate.add_argument("--task-freeze", action="append", required=True)
    candidate.add_argument("--task-inputs", required=True)
    candidate.add_argument("--selected-tracks", required=True)
    candidate.add_argument("--output-root", required=True)
    accept = commands.add_parser("accept")
    accept.add_argument("--candidate", required=True)
    accept.add_argument("--reviewed-by", required=True)
    accept.add_argument("--reviewed-on", required=True)
    accept.add_argument("--output", required=True)
    accept_ai = commands.add_parser("accept-ai")
    accept_ai.add_argument("--candidate", required=True)
    accept_ai.add_argument("--review-execution-receipt", required=True)
    accept_ai.add_argument("--output", required=True)
    authorize_ai = commands.add_parser("authorize-ai")
    authorize_ai.add_argument("--candidate", required=True)
    authorize_ai.add_argument("--accepted-by", required=True)
    authorize_ai.add_argument("--accepted-on", required=True)
    authorize_ai.add_argument("--human-authority-reference", required=True)
    authorize_ai.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.command == "candidate":
        _task_inputs_path, task_inputs = _read(
            args.task_inputs, code="sam31_review_task_inputs_invalid"
        )
        _selected_path, selected_tracks = _read(
            args.selected_tracks, code="sam31_review_selected_tracks_invalid"
        )
        materialize_sam31_track_selection_review_candidate(
            task_freeze_paths=args.task_freeze,
            task_inputs=task_inputs,
            selected_track_ids_by_task=selected_tracks,
            output_root=args.output_root,
        )
    elif args.command == "accept":
        seal_sam31_track_selection_review(
            candidate_path=args.candidate,
            reviewed_by=args.reviewed_by,
            reviewed_on=args.reviewed_on,
            output_path=args.output,
        )
    elif args.command == "accept-ai":
        seal_sam31_track_selection_ai_review(
            candidate_path=args.candidate,
            review_execution_receipt_path=args.review_execution_receipt,
            output_path=args.output,
        )
    else:
        materialize_sam31_ai_visual_review_rights(
            candidate_path=args.candidate,
            accepted_by=args.accepted_by,
            accepted_on=args.accepted_on,
            human_authority_reference=args.human_authority_reference,
            output_path=args.output,
        )
    return 0


__all__ = [
    "AI_RECEIPT_SCHEMA_VERSION",
    "AI_EXECUTION_SCHEMA_VERSION",
    "AI_RIGHTS_SCHEMA_VERSION",
    "AI_REVIEW_METHOD",
    "AI_REVIEW_CAPABILITY",
    "AI_REVIEW_ACCEPTED_BY",
    "AI_REVIEW_DECLARED_USE",
    "AI_REVIEW_FRAME_COUNT",
    "AI_REVIEW_INPUT_TOKEN_CEILING",
    "AI_REVIEW_MAX_COST_USD",
    "AI_REVIEW_MODEL",
    "AI_REVIEWER_ID",
    "CANDIDATE_SCHEMA_VERSION",
    "RECEIPT_SCHEMA_VERSION",
    "Sam31TrackSelectionReviewError",
    "materialize_sam31_track_selection_review_candidate",
    "materialize_sam31_ai_visual_review_rights",
    "build_sam31_ai_visual_review_input",
    "load_validated_sam31_track_selection_review_candidate",
    "resolve_sam31_review_media_path",
    "seal_sam31_track_selection_ai_review",
    "seal_sam31_track_selection_review",
    "validate_sam31_track_selection_review",
    "validate_sam31_ai_structured_decision",
    "validate_sam31_ai_visual_review_rights",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
