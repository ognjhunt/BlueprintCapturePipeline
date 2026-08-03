"""Admit Capture V3.2 retained observations to reconstruction preparation.

The capture manifest indexes immutable raw observations.  This module validates
that index and applies only an explicit, digest-bound task/site selection
profile.  It does not decode pixels, select a provider, authorize an external
upload, or qualify reconstruction, geometry, physics, task success, or physical
deployment.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .common import join_gs_uri, read_json
from .decision_evidence_contracts import canonical_digest, canonical_json


CANDIDATE_SCHEMA_VERSION = "downstream_candidate_manifest.v1"
SELECTION_PROFILE_SCHEMA_VERSION = "task_site_frame_selection_profile.v1"
ADMISSION_SCHEMA_VERSION = "capture_v32_reconstruction_admission.v1"
MISSING_SELECTION_PROFILE = "task_site_evidence_profile_with_frame_selection_parameters"
ALLOWED_CAPTURE_SELECTORS = (
    "explicit_encoded_frame_ordinals",
    "profile_bound_even_decoded_pts_coverage",
    "profile_bound_quality_filter",
)


class CaptureV32CandidateAdmissionError(ValueError):
    """Stable fail-closed validation error."""

    def __init__(self, codes: Sequence[str]) -> None:
        self.codes = tuple(sorted(set(str(code) for code in codes if str(code))))
        super().__init__("; ".join(self.codes))


def _is_digest(value: Any) -> bool:
    text = str(value or "")
    return len(text) == 71 and text.startswith("sha256:") and all(
        character in "0123456789abcdef" for character in text[7:]
    )


def _safe_relative(value: Any) -> bool:
    text = str(value or "").replace("\\", "/")
    return bool(
        text
        and not text.startswith("/")
        and all(part not in {"", ".", ".."} for part in text.split("/"))
    )


def _matrix4(value: Any) -> bool:
    return bool(
        isinstance(value, list)
        and len(value) == 4
        and all(
            isinstance(row, list)
            and len(row) == 4
            and all(_finite_number(cell) for cell in row)
            for row in value
        )
    )


def _finite_number(value: Any) -> bool:
    return bool(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def validate_capture_v32_candidate_manifest(
    value: Mapping[str, Any],
    *,
    expected_source_video_digest: str | None = None,
    expected_source_video_uri: str | None = None,
    expected_coordinate_frame_session_id: str | None = None,
) -> dict[str, Any]:
    """Return a normalized manifest or raise stable contract errors."""

    manifest = json.loads(canonical_json(dict(value)))
    errors: list[str] = []
    if manifest.get("schema_version") != CANDIDATE_SCHEMA_VERSION:
        errors.append("capture_v32_candidate_schema_invalid")
    if manifest.get("manifest_digest") != canonical_digest(
        manifest, digest_field="manifest_digest"
    ):
        errors.append("capture_v32_candidate_digest_mismatch")
    source_digest = str(manifest.get("source_video_sha256") or "")
    normalized_source_digest = source_digest if source_digest.startswith("sha256:") else f"sha256:{source_digest}"
    if not _is_digest(normalized_source_digest):
        errors.append("capture_v32_candidate_source_video_digest_invalid")
    if expected_source_video_digest and normalized_source_digest != expected_source_video_digest:
        errors.append("capture_v32_candidate_source_video_digest_mismatch")
    source_video_uri = str(manifest.get("source_video_uri") or "").strip()
    if not _safe_relative(source_video_uri):
        errors.append("capture_v32_candidate_source_video_uri_missing")
    if expected_source_video_uri and source_video_uri.removeprefix(
        "raw/"
    ) != expected_source_video_uri.removeprefix("raw/"):
        errors.append("capture_v32_candidate_source_video_uri_mismatch")
    coordinate_frame = str(manifest.get("coordinate_frame_session_id") or "").strip()
    if not coordinate_frame:
        errors.append("capture_v32_candidate_coordinate_frame_missing")
    if expected_coordinate_frame_session_id and (
        coordinate_frame != expected_coordinate_frame_session_id
    ):
        errors.append("capture_v32_candidate_coordinate_frame_mismatch")
    if (
        manifest.get("source_video_authority") != "immutable_raw_capture_video"
        or manifest.get("decoded_timing_authority")
        != "sync_map_decoded_sample_presentation_timestamps"
        or manifest.get("candidate_order") != "encoded_frame_index_ascending"
    ):
        errors.append("capture_v32_candidate_authority_invalid")

    selection = manifest.get("selection_contract")
    if (
        not isinstance(selection, Mapping)
        or selection.get("selection_authority")
        != "blueprint_pipeline_task_site_profile"
        or selection.get("capture_default_selection") is not None
        or selection.get("selection_parameters_required") is not True
        or selection.get("allowed_deterministic_selectors")
        != list(ALLOWED_CAPTURE_SELECTORS)
        or selection.get("smallest_missing_input_when_unselectable")
        != MISSING_SELECTION_PROFILE
    ):
        errors.append("capture_v32_candidate_selection_contract_invalid")

    neutrality = manifest.get("provider_neutrality")
    if not isinstance(neutrality, Mapping) or any(
        (
            neutrality.get("mobile_app_direct_provider_upload_allowed") is not False,
            neutrality.get("third_party_provider_upload_authorized") is not False,
            neutrality.get("provider_selection_authority") != "blueprint_pipeline",
            neutrality.get("provider_authorization_status")
            != "not_granted_by_capture_manifest",
            "provider_selected" in neutrality,
        )
    ):
        errors.append("capture_v32_candidate_provider_neutrality_invalid")
    scope = manifest.get("allowed_use_scope")
    if (
        not isinstance(scope, Mapping)
        or scope.get("raw_observation_indexing_allowed") is not True
        or not isinstance(scope.get("derived_processing_allowed"), bool)
        or not isinstance(scope.get("data_licensing_allowed"), bool)
        or scope.get("latest_revocation_check_required") is not True
        or scope.get("provider_upload_requires_separate_downstream_authorization")
        is not True
        or not isinstance(scope.get("redaction_required_before_derived_use"), bool)
        or not isinstance(scope.get("requested_outputs"), list)
    ):
        errors.append("capture_v32_candidate_use_scope_invalid")
    claim_boundary = manifest.get("claim_boundary")
    if (
        not isinstance(claim_boundary, Mapping)
        or claim_boundary.get("raw_capture_remains_authoritative") is not True
        or claim_boundary.get("candidate_manifest_qualifies_reconstruction") is not False
        or claim_boundary.get("candidate_manifest_qualifies_metric_scale") is not False
        or claim_boundary.get("candidate_manifest_qualifies_collision_or_physics") is not False
        or claim_boundary.get("candidate_manifest_proves_task_success") is not False
    ):
        errors.append("capture_v32_candidate_claim_boundary_invalid")

    rows = manifest.get("candidates")
    if not isinstance(rows, list):
        rows = []
        errors.append("capture_v32_candidate_rows_missing")
    if manifest.get("candidate_count") != len(rows):
        errors.append("capture_v32_candidate_count_mismatch")
    candidate_ids: set[str] = set()
    output_paths: set[str] = set()
    previous_pts = -1.0
    previous_encoded_frame_index = -1
    for ordinal, raw in enumerate(rows):
        row = raw if isinstance(raw, Mapping) else {}
        candidate_id = str(row.get("candidate_id") or "")
        output_path = str(row.get("output_image_relative_path") or "")
        if candidate_id != f"rgb_{ordinal:06d}" or candidate_id in candidate_ids:
            errors.append(f"capture_v32_candidate_id_invalid:{ordinal}")
        if (
            output_path != f"candidate_rgb/{ordinal:06d}.png"
            or not _safe_relative(output_path)
            or output_path in output_paths
        ):
            errors.append(f"capture_v32_candidate_output_path_invalid:{ordinal}")
        candidate_ids.add(candidate_id)
        output_paths.add(output_path)
        encoded_frame_index = row.get("encoded_frame_index")
        if (
            row.get("decoded_frame_ordinal") != ordinal
            or not isinstance(encoded_frame_index, int)
            or isinstance(encoded_frame_index, bool)
            or encoded_frame_index <= previous_encoded_frame_index
        ):
            errors.append(f"capture_v32_candidate_ordinal_invalid:{ordinal}")
        if isinstance(encoded_frame_index, int) and not isinstance(encoded_frame_index, bool):
            previous_encoded_frame_index = encoded_frame_index
        try:
            pts = float(row.get("decoded_pts_sec"))
        except (TypeError, ValueError):
            pts = -1.0
        if not math.isfinite(pts) or pts < 0 or pts <= previous_pts and ordinal > 0:
            errors.append(f"capture_v32_candidate_pts_invalid:{ordinal}")
        previous_pts = pts
        if (
            not str(row.get("frame_id") or "")
            or not str(row.get("pose_frame_id") or "")
            or row.get("coordinate_frame_session_id") != coordinate_frame
            or row.get("site_frame_id") != coordinate_frame
            or row.get("site_frame_definition") != "arkit_world_origin_at_session_start"
            or row.get("transform_semantics") != "row_major_camera_to_site"
            or row.get("units") != "meters"
            or row.get("handedness") != "right_handed"
            or row.get("up_axis") != "Y"
            or row.get("gravity_aligned") is not True
            or not _matrix4(row.get("T_site_camera"))
            or row.get("T_site_camera") != row.get("T_world_camera")
        ):
            errors.append(f"capture_v32_candidate_capture_binding_invalid:{ordinal}")
        if row.get("source_video_uri") != source_video_uri:
            errors.append(f"capture_v32_candidate_row_video_binding_invalid:{ordinal}")
        if any(
            not _finite_number(row.get(key)) or float(row[key]) < 0
            for key in ("decoded_source_pts_sec", "t_capture_sec")
        ):
            errors.append(f"capture_v32_candidate_time_binding_invalid:{ordinal}")
        intrinsics = row.get("camera_intrinsics")
        if (
            not isinstance(intrinsics, Mapping)
            or any(
                not _finite_number(intrinsics.get(key))
                for key in ("fx", "fy", "cx", "cy", "width", "height")
            )
            or any(float(intrinsics.get(key, 0)) <= 0 for key in ("fx", "fy", "width", "height"))
            or intrinsics.get("authority") != "arkit_arframe_exact_per_observation"
            or not isinstance(intrinsics.get("matrix_column_major"), list)
            or len(intrinsics["matrix_column_major"]) != 9
            or not all(_finite_number(cell) for cell in intrinsics["matrix_column_major"])
            or row.get("camera_calibration_digest") != canonical_digest(intrinsics)
        ):
            errors.append(f"capture_v32_candidate_intrinsics_invalid:{ordinal}")
        tracking_state = str(row.get("tracking_state") or "unknown")
        relocalization = row.get("relocalization_event") is True
        if (
            not isinstance(row.get("arkit_frame_row_ordinal"), int)
            or not isinstance(row.get("arkit_pose_row_ordinal"), int)
            or row.get("pose_assisted_eligible")
            is not (tracking_state == "normal" and not relocalization)
            or row.get("raw_observation_authority") is not True
            or row.get("downstream_artifact_authority") is not False
        ):
            errors.append(f"capture_v32_candidate_observation_authority_invalid:{ordinal}")
        for path_key in ("depth_path", "confidence_path"):
            if path_key in row and not _safe_relative(row[path_key]):
                errors.append(f"capture_v32_candidate_support_path_invalid:{ordinal}:{path_key}")
    if errors:
        raise CaptureV32CandidateAdmissionError(errors)
    return manifest


def attach_capture_v32_candidate_sidecar(
    sidecars: Mapping[str, Any],
    *,
    raw_root: Path,
    raw_prefix_uri: str,
    manifest: Mapping[str, Any],
    source: str,
) -> dict[str, Any]:
    """Attach bounded V3.2 registry status to existing media metadata."""

    result = dict(sidecars)
    media_metadata = dict(result.get("media_metadata") or {})
    candidate_path = raw_root / "downstream_candidate_manifest.json"
    required = source == "iphone" and str(
        manifest.get("capture_schema_version") or ""
    ).startswith("3.2")
    projection: dict[str, Any] = {
        "uri": (
            join_gs_uri(raw_prefix_uri, candidate_path.name)
            if candidate_path.is_file()
            else None
        ),
        "digest": None,
        "validation_status": "not_required",
        "blockers": [],
        "selection_status": "not_applicable",
        "selection_blocker": None,
        "provider_upload_authorized": False,
        "claim_ceiling": "retained_observation_registry_only",
    }
    if candidate_path.is_file():
        try:
            expected_video_digest: str | None = None
            hashes_path = raw_root / "hashes.json"
            if hashes_path.is_file():
                artifacts = read_json(hashes_path).get("artifacts")
                video_uri = str(manifest.get("video_uri") or "").removeprefix("raw/")
                if isinstance(artifacts, Mapping) and video_uri:
                    raw_digest = str(artifacts.get(video_uri) or "")
                    if raw_digest:
                        expected_video_digest = (
                            raw_digest
                            if raw_digest.startswith("sha256:")
                            else f"sha256:{raw_digest}"
                        )
            if required and expected_video_digest is None:
                raise CaptureV32CandidateAdmissionError(
                    ["capture_v32_source_video_hash_evidence_missing"]
                )
            candidate = validate_capture_v32_candidate_manifest(
                read_json(candidate_path),
                expected_source_video_digest=expected_video_digest,
                expected_source_video_uri=str(manifest.get("video_uri") or "") or None,
                expected_coordinate_frame_session_id=str(
                    manifest.get("coordinate_frame_session_id") or ""
                )
                or None,
            )
            projection.update(
                {
                    "digest": candidate["manifest_digest"],
                    "validation_status": "validated",
                    "selection_status": "awaiting_task_site_evidence_profile",
                    "selection_blocker": MISSING_SELECTION_PROFILE,
                }
            )
        except (CaptureV32CandidateAdmissionError, OSError, ValueError) as exc:
            projection["validation_status"] = "blocked"
            projection["blockers"] = list(
                exc.codes
                if isinstance(exc, CaptureV32CandidateAdmissionError)
                else ("capture_v32_candidate_manifest_unreadable",)
            )
    elif required:
        projection["validation_status"] = "blocked"
        projection["blockers"] = ["capture_v32_candidate_manifest_missing"]
    media_metadata["downstream_candidate_manifest"] = projection
    result["media_metadata"] = media_metadata
    return result


def _validate_profile(
    profile: Mapping[str, Any], *, candidate_manifest_digest: str
) -> dict[str, Any]:
    normalized = json.loads(canonical_json(dict(profile)))
    errors: list[str] = []
    if normalized.get("schema_version") != SELECTION_PROFILE_SCHEMA_VERSION:
        errors.append("task_site_frame_selection_profile_schema_invalid")
    if normalized.get("profile_digest") != canonical_digest(
        normalized, digest_field="profile_digest"
    ):
        errors.append("task_site_frame_selection_profile_digest_mismatch")
    if normalized.get("candidate_manifest_digest") != candidate_manifest_digest:
        errors.append("task_site_frame_selection_profile_capture_binding_mismatch")
    if not str(normalized.get("site_id") or "") or not str(normalized.get("task_id") or ""):
        errors.append("task_site_frame_selection_profile_identity_missing")
    if normalized.get("selector") not in ALLOWED_CAPTURE_SELECTORS:
        errors.append("task_site_frame_selection_profile_selector_unsupported")
    rights = normalized.get("rights_authorization")
    if (
        not isinstance(rights, Mapping)
        or rights.get("status") != "authorized"
        or rights.get("latest_revocation_check_status") != "clear"
        or not _is_digest(rights.get("rights_evidence_digest"))
    ):
        errors.append("latest_authoritative_revocation_check_required")
    if errors:
        raise CaptureV32CandidateAdmissionError(errors)
    return normalized


def _maximum_frames(parameters: Mapping[str, Any]) -> int:
    maximum = parameters.get("maximum_frames")
    if not isinstance(maximum, int) or isinstance(maximum, bool) or maximum < 1:
        raise CaptureV32CandidateAdmissionError([MISSING_SELECTION_PROFILE])
    return maximum


def _even_pts_coverage(candidates: Sequence[Mapping[str, Any]], maximum: int) -> list[dict[str, Any]]:
    rows = [dict(row) for row in candidates]
    if maximum >= len(rows):
        return rows
    if maximum == 1:
        midpoint = (float(rows[0]["decoded_pts_sec"]) + float(rows[-1]["decoded_pts_sec"])) / 2
        return [min(rows, key=lambda row: (abs(float(row["decoded_pts_sec"]) - midpoint), row["encoded_frame_index"]))]
    start = float(rows[0]["decoded_pts_sec"])
    stop = float(rows[-1]["decoded_pts_sec"])
    targets = [start + (stop - start) * index / (maximum - 1) for index in range(maximum)]
    selected = {
        min(
            range(len(rows)),
            key=lambda ordinal: (
                abs(float(rows[ordinal]["decoded_pts_sec"]) - target),
                rows[ordinal]["encoded_frame_index"],
            ),
        )
        for target in targets
    }
    return [rows[ordinal] for ordinal in sorted(selected)]


def _select_candidates(
    profile: Mapping[str, Any], candidates: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    selector = profile["selector"]
    parameters = profile.get("parameters")
    if not isinstance(parameters, Mapping):
        raise CaptureV32CandidateAdmissionError([MISSING_SELECTION_PROFILE])
    if selector == "explicit_encoded_frame_ordinals":
        raw = parameters.get("encoded_frame_ordinals")
        if not isinstance(raw, list) or not raw:
            raise CaptureV32CandidateAdmissionError([MISSING_SELECTION_PROFILE])
        encoded_ordinals = sorted(set(raw))
        if any(
            not isinstance(value, int) or isinstance(value, bool)
            for value in encoded_ordinals
        ):
            raise CaptureV32CandidateAdmissionError(["task_site_frame_selection_ordinal_invalid"])
        by_encoded_ordinal = {row["encoded_frame_index"]: dict(row) for row in candidates}
        if any(value not in by_encoded_ordinal for value in encoded_ordinals):
            raise CaptureV32CandidateAdmissionError(
                ["task_site_frame_selection_ordinal_out_of_range"]
            )
        return [by_encoded_ordinal[value] for value in encoded_ordinals]
    if selector == "profile_bound_even_decoded_pts_coverage":
        return _even_pts_coverage(candidates, _maximum_frames(parameters))

    allowed_tracking_states = parameters.get("allowed_tracking_states")
    require_pose_eligible = parameters.get("require_pose_assisted_eligible")
    exclude_relocalization = parameters.get("exclude_relocalization_events")
    if (
        not isinstance(allowed_tracking_states, list)
        or not allowed_tracking_states
        or any(not isinstance(value, str) or not value for value in allowed_tracking_states)
        or not isinstance(require_pose_eligible, bool)
        or not isinstance(exclude_relocalization, bool)
    ):
        raise CaptureV32CandidateAdmissionError([MISSING_SELECTION_PROFILE])
    qualified = [
        row
        for row in candidates
        if row.get("tracking_state") in allowed_tracking_states
        and (not require_pose_eligible or row.get("pose_assisted_eligible") is True)
        and (not exclude_relocalization or row.get("relocalization_event") is not True)
    ]
    if not qualified:
        return []
    return _even_pts_coverage(qualified, _maximum_frames(parameters))


def build_capture_v32_reconstruction_admission(
    *,
    candidate_manifest: Mapping[str, Any],
    task_site_selection_profile: Mapping[str, Any] | None,
    expected_source_video_digest: str | None = None,
) -> dict[str, Any]:
    """Return an admitted plan or an explicit smallest-input abstention."""

    manifest = validate_capture_v32_candidate_manifest(
        candidate_manifest,
        expected_source_video_digest=expected_source_video_digest,
    )
    base = {
        "schema_version": ADMISSION_SCHEMA_VERSION,
        "candidate_manifest_digest": manifest["manifest_digest"],
        "source_video_uri": manifest["source_video_uri"],
        "source_video_digest": (
            manifest["source_video_sha256"]
            if str(manifest["source_video_sha256"]).startswith("sha256:")
            else f"sha256:{manifest['source_video_sha256']}"
        ),
        "coordinate_frame_session_id": manifest["coordinate_frame_session_id"],
        "provider_selected": None,
        "provider_upload_authorized": False,
        "raw_capture_remains_authoritative": True,
        "claim_ceiling": "retained_observation_selection_only",
        "next_stage": "materialize_exact_decoded_candidate_images",
    }
    if task_site_selection_profile is None:
        result = {
            **base,
            "status": "abstained",
            "blockers": [MISSING_SELECTION_PROFILE],
            "selected_candidates": [],
        }
    else:
        allowed_scope = manifest.get("allowed_use_scope")
        if not isinstance(allowed_scope, Mapping) or allowed_scope.get(
            "derived_processing_allowed"
        ) is not True:
            result = {
                **base,
                "status": "abstained",
                "blockers": ["derived_processing_not_authorized"],
                "selected_candidates": [],
            }
            result["admission_digest"] = canonical_digest(
                result, digest_field="admission_digest"
            )
            return result
        profile = _validate_profile(
            task_site_selection_profile,
            candidate_manifest_digest=manifest["manifest_digest"],
        )
        selected = _select_candidates(profile, manifest["candidates"])
        result = {
            **base,
            "status": "admitted" if selected else "abstained",
            "blockers": [] if selected else ["task_site_frame_selection_no_candidates_qualified"],
            "task_site_selection_profile_digest": profile["profile_digest"],
            "site_id": profile["site_id"],
            "task_id": profile["task_id"],
            "selector": profile["selector"],
            "selected_candidates": selected,
        }
    result["admission_digest"] = canonical_digest(result, digest_field="admission_digest")
    return result


__all__ = [
    "ADMISSION_SCHEMA_VERSION",
    "CANDIDATE_SCHEMA_VERSION",
    "CaptureV32CandidateAdmissionError",
    "MISSING_SELECTION_PROFILE",
    "SELECTION_PROFILE_SCHEMA_VERSION",
    "build_capture_v32_reconstruction_admission",
    "attach_capture_v32_candidate_sidecar",
    "validate_capture_v32_candidate_manifest",
]
