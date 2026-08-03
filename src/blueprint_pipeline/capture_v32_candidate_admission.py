"""Admit Capture V3.2 retained observations to reconstruction preparation.

The capture manifest indexes immutable raw observations.  This module validates
that index and applies only an explicit, digest-bound task/site selection
profile.  It does not decode pixels, select a provider, authorize an external
upload, or qualify reconstruction, geometry, physics, task success, or physical
deployment.
"""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

from .decision_evidence_contracts import canonical_digest, canonical_json


CANDIDATE_SCHEMA_VERSION = "downstream_candidate_manifest.v1"
SELECTION_PROFILE_SCHEMA_VERSION = "task_site_frame_selection_profile.v1"
ADMISSION_SCHEMA_VERSION = "capture_v32_reconstruction_admission.v1"
MISSING_SELECTION_PROFILE = "task_site_evidence_profile_with_frame_selection_parameters"


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
            and all(isinstance(cell, (int, float)) and not isinstance(cell, bool) for cell in row)
            for row in value
        )
    )


def validate_capture_v32_candidate_manifest(
    value: Mapping[str, Any],
    *,
    expected_source_video_digest: str | None = None,
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
    if not str(manifest.get("source_video_uri") or "").strip():
        errors.append("capture_v32_candidate_source_video_uri_missing")
    if not str(manifest.get("coordinate_frame_session_id") or "").strip():
        errors.append("capture_v32_candidate_coordinate_frame_missing")

    neutrality = manifest.get("provider_neutrality")
    if not isinstance(neutrality, Mapping) or any(
        (
            neutrality.get("mobile_app_direct_provider_upload_allowed") is not False,
            neutrality.get("third_party_provider_upload_authorized") is not False,
            neutrality.get("provider_selection_authority") != "blueprint_pipeline",
            isinstance(neutrality.get("provider_selected"), str),
        )
    ):
        errors.append("capture_v32_candidate_provider_neutrality_invalid")
    scope = manifest.get("allowed_use_scope")
    if not isinstance(scope, Mapping) or scope.get("latest_revocation_check_required") is not True:
        errors.append("capture_v32_candidate_revocation_gate_missing")

    rows = manifest.get("candidates")
    if not isinstance(rows, list):
        rows = []
        errors.append("capture_v32_candidate_rows_missing")
    if manifest.get("candidate_count") != len(rows):
        errors.append("capture_v32_candidate_count_mismatch")
    candidate_ids: set[str] = set()
    output_paths: set[str] = set()
    previous_pts = -1.0
    coordinate_frame = manifest.get("coordinate_frame_session_id")
    for ordinal, raw in enumerate(rows):
        row = raw if isinstance(raw, Mapping) else {}
        candidate_id = str(row.get("candidate_id") or "")
        output_path = str(row.get("output_image_relative_path") or "")
        if not candidate_id or candidate_id in candidate_ids:
            errors.append(f"capture_v32_candidate_id_invalid:{ordinal}")
        if not _safe_relative(output_path) or output_path in output_paths:
            errors.append(f"capture_v32_candidate_output_path_invalid:{ordinal}")
        candidate_ids.add(candidate_id)
        output_paths.add(output_path)
        if row.get("decoded_frame_ordinal") != ordinal or row.get("encoded_frame_index") != ordinal:
            errors.append(f"capture_v32_candidate_ordinal_invalid:{ordinal}")
        try:
            pts = float(row.get("decoded_pts_sec"))
        except (TypeError, ValueError):
            pts = -1.0
        if pts < 0 or pts <= previous_pts and ordinal > 0:
            errors.append(f"capture_v32_candidate_pts_invalid:{ordinal}")
        previous_pts = pts
        if (
            not str(row.get("frame_id") or "")
            or not str(row.get("pose_frame_id") or "")
            or row.get("coordinate_frame_session_id") != coordinate_frame
            or not _matrix4(row.get("T_site_camera"))
            or row.get("T_site_camera") != row.get("T_world_camera")
        ):
            errors.append(f"capture_v32_candidate_capture_binding_invalid:{ordinal}")
        intrinsics = row.get("camera_intrinsics")
        if not isinstance(intrinsics, Mapping) or any(
            not isinstance(intrinsics.get(key), (int, float))
            for key in ("fx", "fy", "cx", "cy", "width", "height")
        ):
            errors.append(f"capture_v32_candidate_intrinsics_invalid:{ordinal}")
    if errors:
        raise CaptureV32CandidateAdmissionError(errors)
    return manifest


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
    if normalized.get("selector") not in {
        "explicit_encoded_frame_ordinals",
        "profile_bound_even_decoded_pts_coverage",
    }:
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


def _select_ordinals(profile: Mapping[str, Any], count: int) -> list[int]:
    selector = profile["selector"]
    parameters = profile.get("parameters")
    if not isinstance(parameters, Mapping):
        raise CaptureV32CandidateAdmissionError([MISSING_SELECTION_PROFILE])
    if selector == "explicit_encoded_frame_ordinals":
        raw = parameters.get("encoded_frame_ordinals")
        if not isinstance(raw, list) or not raw:
            raise CaptureV32CandidateAdmissionError([MISSING_SELECTION_PROFILE])
        ordinals = sorted(set(raw))
        if any(not isinstance(value, int) or isinstance(value, bool) for value in ordinals):
            raise CaptureV32CandidateAdmissionError(["task_site_frame_selection_ordinal_invalid"])
    else:
        maximum = parameters.get("maximum_frames")
        if not isinstance(maximum, int) or isinstance(maximum, bool) or maximum < 1:
            raise CaptureV32CandidateAdmissionError([MISSING_SELECTION_PROFILE])
        if maximum >= count:
            ordinals = list(range(count))
        elif maximum == 1:
            ordinals = [count // 2]
        else:
            ordinals = sorted({round(index * (count - 1) / (maximum - 1)) for index in range(maximum)})
    if any(value < 0 or value >= count for value in ordinals):
        raise CaptureV32CandidateAdmissionError(["task_site_frame_selection_ordinal_out_of_range"])
    return ordinals


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
        ordinals = _select_ordinals(profile, len(manifest["candidates"]))
        selected = [manifest["candidates"][ordinal] for ordinal in ordinals]
        result = {
            **base,
            "status": "admitted",
            "blockers": [],
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
    "validate_capture_v32_candidate_manifest",
]
