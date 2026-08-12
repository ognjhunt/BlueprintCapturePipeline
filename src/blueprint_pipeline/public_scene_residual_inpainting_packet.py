"""Seal exact residual-mask inpainting inputs for one through five replacements.

This module deliberately sits *before* an inpainting backend.  A source-layer
coverage audit can establish that some deleted source contribution is not
occluded by the replacement, but it cannot safely hand a broad image crop to a
generative model.  The packet below binds the common retained scene, every
co-present replacement, exact full-resolution masks, and every rendered input
frame.  It never runs an editor or upgrades a packet into an inpainting claim.

The common-scene requirement is intentional.  Per-object cutouts are useful
diagnostics, but a scene with several replacements needs a single retained
Gaussian layer and depth coverage that includes every co-present replacement.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .decision_evidence_contracts import canonical_digest, canonical_json
from .dual_task_rehearsal_contract import MAX_REPLACEMENT_OBJECTS, validate_task_freeze
from .gaussian_splat_decode import (
    read_standard_3dgs_ply,
    verify_standard_3dgs_ply_subset_exact,
)
from .public_scene_replacement_depth_composition import (
    COMPOSITION_SCHEMA as DEPTH_COMPOSITION_SCHEMA,
    ReplacementDepthCompositionError,
    validate_replacement_depth_composition,
)


REQUEST_SCHEMA = "public_scene_residual_inpainting_input_request.v1"
PACKET_SCHEMA = "public_scene_residual_inpainting_input_packet.v1"
BACKEND_ADMISSION_SCHEMA = "public_scene_released_code_inpainting_admission.v1"
SOURCE_COVERAGE_SCHEMA = "adp009b_source_layer_replacement_coverage_audit.v1"
RENDER_SCHEMA = "sealed_camera_render_manifest.v1"
SUPPORTED_CANDIDATE_SET_SCHEMAS = frozenset(
    {
        "adp009b_direct_evidence_expansion_set.v1",
        "adp009b_ownership_coverage_cutout_set.v1",
    }
)
QUALIFIED_RENDER_CLASSES = frozenset({"method_input", "evaluation_authorized"})


class ResidualInpaintingInputPacketError(ValueError):
    """Stable fail-closed errors for residual inpainting packet construction."""

    def __init__(self, codes: Sequence[str]) -> None:
        self.codes = tuple(sorted(set(str(code) for code in codes if str(code))))
        super().__init__(";".join(self.codes))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _digest(value: Any) -> bool:
    text = str(value or "")
    return len(text) == 71 and text.startswith("sha256:") and all(
        character in "0123456789abcdef" for character in text[7:]
    )


def _clone(value: Mapping[str, Any], *, code: str) -> dict[str, Any]:
    try:
        cloned = json.loads(json.dumps(dict(value), allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ResidualInpaintingInputPacketError([code]) from exc
    if not isinstance(cloned, dict):
        raise ResidualInpaintingInputPacketError([code])
    return cloned


def _read_object(path: Path, *, code: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ResidualInpaintingInputPacketError([code]) from exc
    if not isinstance(value, dict):
        raise ResidualInpaintingInputPacketError([code])
    return value


def _path(value: str | Path, *, code: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_file() or path.is_symlink():
        raise ResidualInpaintingInputPacketError([code])
    return path


def _under(path: Path, root: Path, *, code: str) -> Path:
    resolved = path.expanduser().resolve()
    root = root.expanduser().resolve()
    if resolved != root and root not in resolved.parents:
        raise ResidualInpaintingInputPacketError([code])
    return resolved


def _relative_record(root: Path, value: Any, *, code: str) -> tuple[Path, dict[str, Any]]:
    if not isinstance(value, Mapping):
        raise ResidualInpaintingInputPacketError([code])
    relative = str(value.get("relative_path") or "")
    if not relative or relative.startswith("/") or ".." in Path(relative).parts:
        raise ResidualInpaintingInputPacketError([code])
    path = _under(root / relative, root, code=code)
    if (
        not path.is_file()
        or path.is_symlink()
        or path.stat().st_size != value.get("size_bytes")
        or _sha256(path) != value.get("sha256")
    ):
        raise ResidualInpaintingInputPacketError([code])
    return path, {
        "relative_path": relative,
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _absolute_record(value: Any, *, code: str) -> tuple[Path, dict[str, Any]]:
    if not isinstance(value, Mapping):
        raise ResidualInpaintingInputPacketError([code])
    path = _path(str(value.get("path") or ""), code=code)
    if path.stat().st_size != value.get("size_bytes") or _sha256(path) != value.get("sha256"):
        raise ResidualInpaintingInputPacketError([code])
    return path, {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def build_residual_inpainting_input_request(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a frozen request without reading a backend or render artifact."""

    request = _clone(value, code="residual_inpainting_request_not_json")
    supplied_digest = request.pop("request_digest", None)
    errors: list[str] = []
    if request.get("schema_version") != REQUEST_SCHEMA:
        errors.append("residual_inpainting_request_schema_invalid")
    if request.get("program_id") != "arm-decision-proof-v1":
        errors.append("residual_inpainting_request_program_invalid")
    if request.get("adp_item") != "ADP-009D":
        errors.append("residual_inpainting_request_adp_item_invalid")
    if request.get("frozen_before_inpainting_execution") is not True:
        errors.append("residual_inpainting_request_not_frozen")
    if request.get("learned_policy_outcomes_accessed") is not False:
        errors.append("residual_inpainting_request_policy_outcome_leakage")
    if any(key in request for key in ("status", "inpainting_result_qualified", "method_succeeded")):
        errors.append("residual_inpainting_request_caller_outcome_forbidden")
    for key in (
        "candidate_set_path",
        "backend_admission_path",
    ):
        if not str(request.get(key) or "").strip():
            errors.append(f"residual_inpainting_request_{key}_missing")
    privacy = request.get("private_upload_policy")
    if not isinstance(privacy, Mapping):
        errors.append("residual_inpainting_request_private_upload_policy_missing")
    elif (
        privacy.get("raw_dataset_bytes_upload") is not False
        or privacy.get("private_derived_upload") is not True
        or privacy.get("provider_training") is not False
        or privacy.get("publication") is not False
        or isinstance(privacy.get("maximum_retention_days"), bool)
        or not isinstance(privacy.get("maximum_retention_days"), int)
        or not 1 <= int(privacy["maximum_retention_days"]) <= 30
    ):
        errors.append("residual_inpainting_request_private_upload_policy_invalid")
    lanes = request.get("task_lanes")
    if not isinstance(lanes, list) or not 1 <= len(lanes) <= MAX_REPLACEMENT_OBJECTS:
        errors.append("residual_inpainting_request_task_lane_count_invalid")
        lanes = []
    seen: set[str] = set()
    for lane in lanes:
        if not isinstance(lane, Mapping):
            errors.append("residual_inpainting_request_task_lane_invalid")
            continue
        task_id = str(lane.get("task_id") or "").strip()
        if not task_id or task_id in seen:
            errors.append("residual_inpainting_request_task_id_invalid_or_duplicate")
        seen.add(task_id)
        if lane.get("co_present_replacements_required") is not True:
            errors.append("residual_inpainting_request_co_present_replacements_required")
        for key in ("coverage_audit_path", "retained_render_manifest_path"):
            if not str(lane.get(key) or "").strip():
                errors.append(f"residual_inpainting_request_lane_{key}_missing")
    if errors:
        raise ResidualInpaintingInputPacketError(errors)
    request["request_digest"] = canonical_digest(request, digest_field="request_digest")
    if supplied_digest is not None and supplied_digest != request["request_digest"]:
        raise ResidualInpaintingInputPacketError(["residual_inpainting_request_digest_mismatch"])
    return request


def _validate_backend_admission(
    path: Path, *, privacy: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    admission = _read_object(path, code="residual_inpainting_backend_admission_unreadable")
    if (
        admission.get("schema_version") != BACKEND_ADMISSION_SCHEMA
        or admission.get("status") != "rights_admitted_for_private_derived_inpainting"
        or admission.get("receipt_digest")
        != canonical_digest(admission, digest_field="receipt_digest")
    ):
        raise ResidualInpaintingInputPacketError(["residual_inpainting_backend_admission_invalid"])
    for key in (
        "backend_id",
        "source_repository",
        "source_revision",
        "source_archive_sha256",
        "environment_lock_sha256",
        "model_identity",
    ):
        if not str(admission.get(key) or "").strip():
            raise ResidualInpaintingInputPacketError(
                ["residual_inpainting_backend_identity_incomplete"]
            )
    if not _digest(admission.get("source_archive_sha256")) or not _digest(
        admission.get("environment_lock_sha256")
    ):
        raise ResidualInpaintingInputPacketError(
            ["residual_inpainting_backend_digest_invalid"]
        )
    allowed = admission.get("private_derived_upload_policy")
    if not isinstance(allowed, Mapping) or (
        allowed.get("raw_dataset_bytes_upload") is not False
        or allowed.get("private_derived_upload") is not True
        or allowed.get("provider_training") is not False
        or allowed.get("publication") is not False
        or isinstance(allowed.get("maximum_retention_days"), bool)
        or not isinstance(allowed.get("maximum_retention_days"), int)
        or int(allowed["maximum_retention_days"]) > int(privacy["maximum_retention_days"])
    ):
        raise ResidualInpaintingInputPacketError(
            ["residual_inpainting_backend_private_upload_policy_invalid"]
        )
    return admission, {
        **_file_record(path),
        "receipt_digest": admission["receipt_digest"],
    }


def _validate_candidate_set(
    path: Path,
) -> tuple[dict[str, Any], dict[str, Any], Path, dict[str, Any], Path, dict[str, Any]]:
    candidate_set = _read_object(path, code="residual_inpainting_candidate_set_unreadable")
    digest_field = "receipt_digest" if "receipt_digest" in candidate_set else "set_digest"
    expected = canonical_digest(candidate_set, digest_field=digest_field)
    claim_boundary = candidate_set.get("claim_boundary")
    if (
        candidate_set.get("schema_version") not in SUPPORTED_CANDIDATE_SET_SCHEMAS
        or candidate_set.get(digest_field) != expected
        or not isinstance(claim_boundary, Mapping)
        or claim_boundary.get("candidate_derived_layers_only") is not True
    ):
        raise ResidualInpaintingInputPacketError(["residual_inpainting_candidate_set_invalid"])
    candidates = candidate_set.get("task_candidates")
    if not isinstance(candidates, list) or not 1 <= len(candidates) <= MAX_REPLACEMENT_OBJECTS:
        raise ResidualInpaintingInputPacketError(["residual_inpainting_candidate_set_count_invalid"])
    union = candidate_set.get("shared_scene_union")
    if not isinstance(union, Mapping):
        raise ResidualInpaintingInputPacketError(["residual_inpainting_shared_scene_union_missing"])
    outputs = union.get("outputs")
    if not isinstance(outputs, Mapping):
        raise ResidualInpaintingInputPacketError(["residual_inpainting_shared_scene_outputs_missing"])
    root = path.parent
    source_path, source_record = _absolute_record(
        candidate_set.get("source_standard_splat"),
        code="residual_inpainting_candidate_source_invalid",
    )
    retained_path, retained_record = _relative_record(
        root,
        outputs.get("retained_scene_gaussians"),
        code="residual_inpainting_shared_retained_scene_invalid",
    )
    deleted_path, deleted_record = _relative_record(
        root,
        outputs.get("deleted_source_gaussians"),
        code="residual_inpainting_shared_deleted_source_layer_invalid",
    )
    counts = union.get("counts")
    deleted_indices_path, _deleted_indices_record = _relative_record(
        root,
        outputs.get("deleted_source_indices"),
        code="residual_inpainting_shared_deleted_indices_invalid",
    )
    retained_indices_path, _retained_indices_record = _relative_record(
        root,
        outputs.get("retained_source_indices"),
        code="residual_inpainting_shared_retained_indices_invalid",
    )
    try:
        deleted_indices = np.asarray(np.load(deleted_indices_path, allow_pickle=False), dtype=np.int64)
        retained_indices = np.asarray(np.load(retained_indices_path, allow_pickle=False), dtype=np.int64)
    except (OSError, ValueError) as exc:
        raise ResidualInpaintingInputPacketError(
            ["residual_inpainting_shared_index_arrays_unreadable"]
        ) from exc
    try:
        source_count = read_standard_3dgs_ply(source_path).count
        retained_count = read_standard_3dgs_ply(retained_path).count
        deleted_count = read_standard_3dgs_ply(deleted_path).count
        retained_byte_exact = verify_standard_3dgs_ply_subset_exact(
            source_path, retained_path, retained_indices
        ).get("retained_rows_byte_exact")
        deleted_byte_exact = verify_standard_3dgs_ply_subset_exact(
            source_path, deleted_path, deleted_indices
        ).get("retained_rows_byte_exact")
    except (OSError, ValueError) as exc:
        raise ResidualInpaintingInputPacketError(
            ["residual_inpainting_shared_retained_scene_not_byte_exact"]
        ) from exc
    expected_source_indices = np.arange(source_count, dtype=np.int64)
    if (
        deleted_indices.ndim != 1
        or retained_indices.ndim != 1
        or not np.array_equal(np.unique(deleted_indices), deleted_indices)
        or not np.array_equal(np.unique(retained_indices), retained_indices)
        or np.intersect1d(deleted_indices, retained_indices, assume_unique=True).size
        or not np.array_equal(
            np.union1d(deleted_indices, retained_indices), expected_source_indices
        )
        or retained_byte_exact is not True
        or deleted_byte_exact is not True
    ):
        raise ResidualInpaintingInputPacketError(
            ["residual_inpainting_shared_retained_scene_not_byte_exact"]
        )
    if (
        not isinstance(counts, Mapping)
        or counts.get("source") != source_count
        or counts.get("deleted_total") != int(deleted_indices.size)
        or counts.get("retained_total") != retained_count
        or deleted_count != int(deleted_indices.size)
        or retained_count <= 0
        or deleted_count <= 0
    ):
        raise ResidualInpaintingInputPacketError(["residual_inpainting_shared_retained_count_invalid"])
    return candidate_set, {
        **_file_record(path),
        digest_field: candidate_set[digest_field],
    }, deleted_path, {
        **deleted_record,
        "deleted_gaussian_count": deleted_count,
        "source_layer_role": "shared_deleted_source_union",
    }, retained_path, retained_record


def _candidate_tasks(candidate_set: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for candidate in candidate_set["task_candidates"]:
        if not isinstance(candidate, Mapping):
            raise ResidualInpaintingInputPacketError(["residual_inpainting_candidate_row_invalid"])
        task_id = str(candidate.get("task_id") or "").strip()
        if not task_id or task_id in result:
            raise ResidualInpaintingInputPacketError(
                ["residual_inpainting_candidate_task_id_invalid_or_duplicate"]
            )
        task_record = candidate.get("task_freeze")
        task_path, verified_task_record = _absolute_record(
            task_record, code="residual_inpainting_task_freeze_record_invalid"
        )
        task = _read_object(task_path, code="residual_inpainting_task_freeze_unreadable")
        try:
            freeze = validate_task_freeze(task)
        except Exception as exc:
            raise ResidualInpaintingInputPacketError(
                ["residual_inpainting_task_freeze_invalid"]
            ) from exc
        removal = freeze.get("removal_plan")
        if (
            freeze.get("task_id") != task_id
            or candidate.get("task_freeze_digest") != freeze.get("task_freeze_digest")
            or not isinstance(removal, Mapping)
            or candidate.get("removal_id") != removal.get("removal_id")
            or candidate.get("mask_set_id") != removal.get("mask_set_id")
            or not str(removal.get("replacement_asset_id") or "")
        ):
            raise ResidualInpaintingInputPacketError(
                ["residual_inpainting_candidate_task_join_invalid"]
            )
        result[task_id] = {
            "candidate": dict(candidate),
            "task_freeze": freeze,
            "task_freeze_record": verified_task_record,
            "replacement_asset_id": str(removal["replacement_asset_id"]),
        }
    return result


def _validate_coverage_audit(
    path: Path,
    *,
    lane: Mapping[str, Any],
    candidate: Mapping[str, Any],
    deleted_source_splat_digest: str,
    expected_asset_ids: Sequence[str],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[str, Any]], dict[str, Any]]:
    audit = _read_object(path, code="residual_inpainting_coverage_audit_unreadable")
    if (
        audit.get("schema_version") != SOURCE_COVERAGE_SCHEMA
        or audit.get("status") != "source_layer_coverage_measured"
        or audit.get("manifest_digest")
        != canonical_digest(audit, digest_field="manifest_digest")
        or audit.get("uncovered_source_support_masks_are_inpainting_authority") is not True
        or audit.get("source_layer_splat_digest") != deleted_source_splat_digest
    ):
        raise ResidualInpaintingInputPacketError(["residual_inpainting_coverage_audit_invalid"])
    task_freeze = candidate["task_freeze"]
    removal = task_freeze["removal_plan"]
    if (
        audit.get("task_id") != task_freeze.get("task_id")
        or audit.get("task_freeze_digest") != task_freeze.get("task_freeze_digest")
        or audit.get("removal_id") != removal.get("removal_id")
        or audit.get("mask_set_id") != removal.get("mask_set_id")
        or audit.get("replacement_asset_id") != candidate.get("replacement_asset_id")
    ):
        raise ResidualInpaintingInputPacketError(
            ["residual_inpainting_coverage_task_join_invalid"]
        )
    eligibility = audit.get("inpainting_mask_eligibility")
    if not isinstance(eligibility, Mapping) or (
        eligibility.get("full_resolution_source_frames") is not True
        or eligibility.get("full_resolution_replacement_depth") is not True
        or eligibility.get("calibrated_method_input_pair") is not True
        or eligibility.get("authorizes_only")
        != "future_exact_mask_contained_multi_view_edit_input"
        or eligibility.get("inpainting_result_qualified") is not False
    ):
        raise ResidualInpaintingInputPacketError(
            ["residual_inpainting_mask_eligibility_invalid"]
        )
    composition_path, composition_record = _absolute_record(
        audit.get("replacement_depth_composition"),
        code="residual_inpainting_depth_composition_receipt_invalid",
    )
    composition = _read_object(
        composition_path, code="residual_inpainting_depth_composition_unreadable"
    )
    if (
        composition.get("schema_version") != DEPTH_COMPOSITION_SCHEMA
        or composition.get("status") != "co_present_replacement_depth_rasterized"
        or composition.get("receipt_digest")
        != canonical_digest(composition, digest_field="receipt_digest")
        or composition.get("task_id") != task_freeze.get("task_id")
        or composition.get("task_freeze_digest")
        != task_freeze.get("task_freeze_digest")
        or sorted(composition.get("replacement_asset_ids") or [])
        != sorted(expected_asset_ids)
    ):
        raise ResidualInpaintingInputPacketError(
            ["residual_inpainting_co_present_depth_coverage_missing"]
        )
    try:
        validate_replacement_depth_composition(
            composition, receipt_path=composition_path
        )
    except ReplacementDepthCompositionError as exc:
        raise ResidualInpaintingInputPacketError(
            ["residual_inpainting_depth_composition_receipt_invalid", *exc.codes]
        ) from exc
    composition_record["receipt_digest"] = composition["receipt_digest"]
    masks = audit.get("uncovered_source_support_masks")
    camera_ids = audit.get("camera_ids")
    if (
        not isinstance(camera_ids, list)
        or not camera_ids
        or len(camera_ids) != len(set(camera_ids))
        or not isinstance(masks, list)
        or len(masks) != len(camera_ids)
    ):
        raise ResidualInpaintingInputPacketError(["residual_inpainting_mask_camera_set_invalid"])
    mask_root = path.parent
    masks_by_camera: dict[str, dict[str, Any]] = {}
    for row in masks:
        if not isinstance(row, Mapping):
            raise ResidualInpaintingInputPacketError(["residual_inpainting_mask_row_invalid"])
        camera_id = str(row.get("camera_id") or "")
        if camera_id not in camera_ids or camera_id in masks_by_camera:
            raise ResidualInpaintingInputPacketError(["residual_inpainting_mask_camera_id_invalid"])
        mask_path, mask_record = _relative_record(
            mask_root, row, code="residual_inpainting_mask_bytes_changed"
        )
        with Image.open(mask_path) as mask:
            if mask.mode not in {"1", "L"}:
                raise ResidualInpaintingInputPacketError(
                    ["residual_inpainting_mask_empty_or_invalid"]
                )
            values = set(mask.convert("L").tobytes())
            if not values.issubset({0, 255}) or values != {0, 255}:
                raise ResidualInpaintingInputPacketError(
                    ["residual_inpainting_mask_empty_or_invalid"]
                )
            nonzero_count = int(np.count_nonzero(np.asarray(mask.convert("L"))))
        if nonzero_count != row.get("pixel_count") or int(
            row.get("derived_from_all_state_cells") or 0
        ) <= 0:
            raise ResidualInpaintingInputPacketError(
                ["residual_inpainting_mask_measurement_invalid"]
            )
        masks_by_camera[camera_id] = {
            **mask_record,
            "camera_id": camera_id,
            "pixel_count": nonzero_count,
            "derived_from_all_state_cells": int(
                row.get("derived_from_all_state_cells") or 0
            ),
            "_mask_path": mask_path,
        }
    if set(masks_by_camera) != set(camera_ids):
        raise ResidualInpaintingInputPacketError(["residual_inpainting_mask_camera_set_invalid"])
    return audit, {
        **_file_record(path),
        "manifest_digest": audit["manifest_digest"],
    }, masks_by_camera, composition_record


def _validate_retained_render(
    path: Path,
    *,
    shared_splat_digest: str,
    shared_retained_count: int,
    masks_by_camera: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[str, Any]]]:
    manifest = _read_object(path, code="residual_inpainting_retained_render_unreadable")
    if (
        manifest.get("schema_version") != RENDER_SCHEMA
        or manifest.get("status") != "rendered_exact_cameras"
        or manifest.get("sealed_camera_render_manifest_digest")
        != canonical_digest(manifest, digest_field="sealed_camera_render_manifest_digest")
        or manifest.get("authorization_class") not in QUALIFIED_RENDER_CLASSES
        or manifest.get("splat_digest") != shared_splat_digest
    ):
        raise ResidualInpaintingInputPacketError(["residual_inpainting_retained_render_invalid"])
    source = manifest.get("source_splat")
    calibration = manifest.get("calibrated_camera_file")
    settings = manifest.get("render_settings")
    if (
        not isinstance(source, Mapping)
        or source.get("retained_gaussian_count") != shared_retained_count
        or not isinstance(calibration, Mapping)
        or calibration.get("binding") != "caller_file_exact_match"
        or not isinstance(settings, Mapping)
        or not isinstance(settings.get("dimensions"), Mapping)
    ):
        raise ResidualInpaintingInputPacketError(
            ["residual_inpainting_retained_render_binding_invalid"]
        )
    cameras = manifest.get("calibrated_cameras")
    renders = manifest.get("renders")
    if not isinstance(cameras, list) or not isinstance(renders, list):
        raise ResidualInpaintingInputPacketError(["residual_inpainting_retained_render_rows_invalid"])
    camera_ids = [str(row.get("id") or "") for row in cameras if isinstance(row, Mapping)]
    if len(camera_ids) != len(cameras) or set(camera_ids) != set(masks_by_camera):
        raise ResidualInpaintingInputPacketError(
            ["residual_inpainting_retained_render_camera_set_mismatch"]
        )
    render_root = path.parent
    frames: dict[str, dict[str, Any]] = {}
    dimensions = settings["dimensions"]
    width = dimensions.get("width")
    height = dimensions.get("height")
    if (
        isinstance(width, bool)
        or isinstance(height, bool)
        or not isinstance(width, int)
        or not isinstance(height, int)
        or width <= 0
        or height <= 0
    ):
        raise ResidualInpaintingInputPacketError(
            ["residual_inpainting_retained_render_dimensions_invalid"]
        )
    for row in renders:
        if not isinstance(row, Mapping):
            raise ResidualInpaintingInputPacketError(["residual_inpainting_retained_frame_row_invalid"])
        camera_id = str(row.get("camera_id") or "")
        if camera_id not in masks_by_camera or camera_id in frames:
            raise ResidualInpaintingInputPacketError(["residual_inpainting_retained_frame_camera_invalid"])
        frame_path, frame_record = _relative_record(
            render_root, row, code="residual_inpainting_retained_frame_bytes_changed"
        )
        with Image.open(frame_path) as image:
            if image.size != (width, height):
                raise ResidualInpaintingInputPacketError(
                    ["residual_inpainting_retained_frame_dimensions_invalid"]
                )
        with Image.open(masks_by_camera[camera_id]["_mask_path"]) as mask:
            if mask.size != (width, height):
                raise ResidualInpaintingInputPacketError(
                    ["residual_inpainting_mask_frame_dimensions_mismatch"]
                )
        frames[camera_id] = {**frame_record, "camera_id": camera_id}
    if set(frames) != set(masks_by_camera):
        raise ResidualInpaintingInputPacketError(
            ["residual_inpainting_retained_frame_camera_set_mismatch"]
        )
    return manifest, {
        **_file_record(path),
        "sealed_camera_render_manifest_digest": manifest[
            "sealed_camera_render_manifest_digest"
        ],
    }, frames


def materialize_residual_inpainting_input_packet(
    *, request_path: str | Path, output_root: str | Path
) -> dict[str, Any]:
    """Materialize an execution-ready *input* packet for up to five objects.

    All files are verified before the packet is written.  The resulting packet
    authorizes neither an upload nor an inpainting invocation; those require a
    separate backend adapter that preserves the exact masks and enforces the
    receipt-bound privacy policy.
    """

    request_file = _path(request_path, code="residual_inpainting_request_missing")
    request = build_residual_inpainting_input_request(
        _read_object(request_file, code="residual_inpainting_request_unreadable")
    )
    output = Path(output_root).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise ResidualInpaintingInputPacketError(["residual_inpainting_packet_output_not_empty"])
    candidate_path = _path(
        request["candidate_set_path"], code="residual_inpainting_candidate_set_missing"
    )
    (
        candidate_set,
        candidate_record,
        deleted_ply,
        deleted_ply_record,
        shared_ply,
        shared_ply_record,
    ) = _validate_candidate_set(candidate_path)
    candidate_tasks = _candidate_tasks(candidate_set)
    requested_ids = {str(lane["task_id"]) for lane in request["task_lanes"]}
    if requested_ids != set(candidate_tasks):
        raise ResidualInpaintingInputPacketError(
            ["residual_inpainting_request_candidate_task_set_mismatch"]
        )
    privacy = request["private_upload_policy"]
    backend_path = _path(
        request["backend_admission_path"], code="residual_inpainting_backend_admission_missing"
    )
    backend, backend_record = _validate_backend_admission(backend_path, privacy=privacy)
    shared_digest = shared_ply_record["sha256"]
    shared_count = read_standard_3dgs_ply(shared_ply).count
    expected_asset_ids = sorted(
        row["replacement_asset_id"] for row in candidate_tasks.values()
    )
    output.mkdir(parents=True, exist_ok=True)
    lanes: list[dict[str, Any]] = []
    for lane in sorted(request["task_lanes"], key=lambda row: str(row["task_id"])):
        task_id = str(lane["task_id"])
        candidate = candidate_tasks[task_id]
        coverage_path = _path(
            lane["coverage_audit_path"], code="residual_inpainting_coverage_audit_missing"
        )
        audit, audit_record, masks, composition_record = _validate_coverage_audit(
            coverage_path,
            lane=lane,
            candidate=candidate,
            deleted_source_splat_digest=deleted_ply_record["sha256"],
            expected_asset_ids=expected_asset_ids,
        )
        render_path = _path(
            lane["retained_render_manifest_path"],
            code="residual_inpainting_retained_render_missing",
        )
        render, render_record, frames = _validate_retained_render(
            render_path,
            shared_splat_digest=shared_digest,
            shared_retained_count=shared_count,
            masks_by_camera=masks,
        )
        lane_root = output / f"lane_{len(lanes) + 1:02d}"
        lane_root.mkdir()
        lane_receipt = {
            "task_id": task_id,
            "task_freeze_digest": candidate["task_freeze"]["task_freeze_digest"],
            "removal_id": candidate["task_freeze"]["removal_plan"]["removal_id"],
            "mask_set_id": candidate["task_freeze"]["removal_plan"]["mask_set_id"],
            "replacement_asset_id": candidate["replacement_asset_id"],
            "co_present_replacement_asset_ids": expected_asset_ids,
            "coverage_audit": audit_record,
            "retained_scene_render": render_record,
            "replacement_depth_composition": composition_record,
            "exact_residual_masks": [
                {key: value for key, value in masks[camera_id].items() if key != "_mask_path"}
                for camera_id in sorted(masks)
            ],
            "retained_scene_frames": [frames[camera_id] for camera_id in sorted(frames)],
            "inpainting_execution_authorized": False,
            "inpainting_result_qualified": False,
        }
        lane_receipt["lane_digest"] = canonical_digest(
            lane_receipt, digest_field="lane_digest"
        )
        lane_path = lane_root / "residual_inpainting_input_lane.v1.json"
        lane_path.write_text(canonical_json(lane_receipt) + "\n", encoding="utf-8")
        lanes.append(
            {**_file_record(lane_path), "lane_digest": lane_receipt["lane_digest"]}
        )
    packet: dict[str, Any] = {
        "schema_version": PACKET_SCHEMA,
        "status": "exact_mask_contained_inpainting_input_packet_materialized",
        "request": {**_file_record(request_file), "request_digest": request["request_digest"]},
        "candidate_set": candidate_record,
        "shared_retained_scene": {
            **shared_ply_record,
            "retained_gaussian_count": shared_count,
            "all_replacements_co_present": True,
        },
        "shared_deleted_source_layer": {
            **deleted_ply_record,
            "all_replacements_co_present": True,
            "source_mask_authority_only": True,
        },
        "backend_admission": backend_record,
        "private_upload_policy": dict(privacy),
        "replacement_object_count": len(lanes),
        "maximum_replacement_objects": MAX_REPLACEMENT_OBJECTS,
        "lanes": lanes,
        "claim_boundary": {
            "raw_dataset_bytes_upload_authorized": False,
            "private_derived_upload_not_yet_performed": True,
            "released_code_inpainting_executed": False,
            "inpainting_result_qualified": False,
            "outside_mask_locality_measured": False,
            "source_gaussian_removal_qualified": False,
            "native_simulator_import_qualified": False,
        },
        "packet_digest": "",
    }
    packet["packet_digest"] = canonical_digest(packet, digest_field="packet_digest")
    packet_path = output / f"{PACKET_SCHEMA}.json"
    packet_path.write_text(canonical_json(packet) + "\n", encoding="utf-8")
    return packet


__all__ = [
    "BACKEND_ADMISSION_SCHEMA",
    "PACKET_SCHEMA",
    "REQUEST_SCHEMA",
    "ResidualInpaintingInputPacketError",
    "build_residual_inpainting_input_request",
    "materialize_residual_inpainting_input_packet",
]
