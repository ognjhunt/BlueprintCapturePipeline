"""Normalize persistent 2D mask tracks without inventing 3D authority.

This module is a provider-neutral seam between a SAM-class video tracker (or a
human-reviewed equivalent) and the Gaussian contribution lifter.  It binds
every mask observation to an encoder-retained source frame, decoded PTS, and
camera record, while preserving compact run-length masks for large captures.
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, Mapping, Sequence

from .semantic_gaussian_lifting import canonical_json_digest


REQUEST_SCHEMA_VERSION = "semantic_source_track_import_request.v1"
PROVIDER_RESULT_SCHEMA_VERSION = "semantic_source_track_provider_result.v1"
RESULT_SCHEMA_VERSION = "semantic_source_track_import_result.v1"
MASK_ENCODING = "sparse_probability_rle.v1"

_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_MAX_FRAMES = 100_000
_MAX_TRACKS = 10_000
_MAX_OBSERVATIONS = 1_000_000
_MAX_RUNS_PER_OBSERVATION = 1_000_000


def _valid_digest(value: Any) -> bool:
    text = str(value or "").strip().lower()
    return (
        len(text) == 71
        and text.startswith("sha256:")
        and all(character in "0123456789abcdef" for character in text[7:])
    )


def _same_digest(left: Any, right: Any) -> bool:
    return str(left or "").strip().lower() == str(right or "").strip().lower()


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _positive_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return value


def _identifier(value: Any) -> str:
    text = str(value or "").strip()
    return text if _IDENTIFIER.fullmatch(text) else ""


def _blocked(request: Mapping[str, Any], blockers: Sequence[str]) -> Dict[str, Any]:
    bindings = request.get("bindings")
    result: Dict[str, Any] = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "blocked",
        "bindings": dict(bindings) if isinstance(bindings, Mapping) else {},
        "track_registry": [],
        "frame_masks": [],
        "blockers": sorted(set(blockers)),
        "warnings": [],
        "claim_ceiling": "none_invalid_or_unbound_source_tracks",
        "directly_observed_object_fact": False,
        "canonical_object_geometry": False,
        "metric_box_ready": False,
        "collision_ready": False,
        "physics_ready": False,
        "physical_task_success_established": False,
        "generated_regions_can_upgrade_claims": False,
        "comparative_policy_ranking_verdict": "thesis_not_supported",
    }
    result["result_digest"] = canonical_json_digest(result)
    return result


def blocked_semantic_source_track_import(
    request: Mapping[str, Any], blockers: Sequence[str]
) -> Dict[str, Any]:
    """Build a deterministic terminal artifact for file/admission failures."""

    return _blocked(request, blockers)


def _validate_frame_registry(
    request: Mapping[str, Any], blockers: list[str]
) -> Dict[str, Dict[str, Any]]:
    bindings = request.get("bindings") if isinstance(request.get("bindings"), Mapping) else {}
    raw = request.get("frame_registry")
    if not isinstance(raw, list) or not raw or len(raw) > _MAX_FRAMES:
        blockers.append("frame_registry_missing_empty_or_too_large")
        return {}
    try:
        digest = canonical_json_digest(raw)
    except (TypeError, ValueError):
        blockers.append("frame_registry_not_canonical_json")
        return {}
    if not _same_digest(bindings.get("frame_registry_digest"), digest):
        blockers.append("frame_registry_digest_mismatch")
    frames: Dict[str, Dict[str, Any]] = {}
    for row in raw:
        if not isinstance(row, Mapping):
            blockers.append("frame_registry_row_invalid")
            continue
        frame_id = _identifier(row.get("source_frame_id"))
        if not frame_id or frame_id in frames:
            blockers.append("frame_registry_id_invalid_or_duplicate")
            continue
        for field in (
            "source_frame_digest",
            "retained_video_digest",
            "sync_map_row_digest",
            "camera_record_digest",
        ):
            if not _valid_digest(row.get(field)):
                blockers.append(f"frame_registry_digest_invalid:{frame_id}:{field}")
        if not _same_digest(
            row.get("retained_video_digest"), bindings.get("retained_video_digest")
        ):
            blockers.append(f"frame_registry_retained_video_mismatch:{frame_id}")
        pts = _finite(row.get("decoded_pts_seconds"))
        if pts is None or pts < 0.0:
            blockers.append(f"frame_registry_pts_invalid:{frame_id}")
        if row.get("encoder_retained") is not True:
            blockers.append(f"frame_registry_encoder_retention_not_proven:{frame_id}")
        frames[frame_id] = dict(row)
    return frames


def _validate_profile(request: Mapping[str, Any], blockers: list[str]) -> Dict[str, Any]:
    raw = request.get("provider_profile")
    if not isinstance(raw, Mapping):
        blockers.append("provider_profile_missing")
        return {}
    profile = dict(raw)
    supplied_digest = profile.get("profile_digest")
    if not _valid_digest(supplied_digest) or supplied_digest != canonical_json_digest(
        {key: value for key, value in profile.items() if key != "profile_digest"}
    ):
        blockers.append("provider_profile_digest_mismatch")
    for field in ("method_id", "method_version"):
        if not str(profile.get(field) or "").strip():
            blockers.append(f"provider_profile_{field}_missing")
    for field in ("runtime_digest", "model_digest"):
        if not _valid_digest(profile.get(field)):
            blockers.append(f"provider_profile_{field}_invalid")
    if profile.get("persistent_track_ids") is not True:
        blockers.append("provider_profile_persistent_track_ids_required")
    if profile.get("mask_encoding") != MASK_ENCODING:
        blockers.append("provider_profile_mask_encoding_unsupported")
    if profile.get("model_self_grading_forbidden") is not True:
        blockers.append("provider_profile_self_grading_boundary_missing")
    execution_mode = str(profile.get("execution_mode") or "").strip()
    if execution_mode not in {"local", "configured_external", "owner_attested_import"}:
        blockers.append("provider_profile_execution_mode_invalid")
    if execution_mode == "configured_external" and not _valid_digest(
        profile.get("execution_authorization_digest")
    ):
        blockers.append("external_execution_authorization_missing")
    allowed_uses = request.get("allowed_evidence_uses")
    if not isinstance(allowed_uses, list) or "semantic_analysis" not in allowed_uses:
        blockers.append("semantic_analysis_use_not_permitted")
    if not isinstance(profile.get("customer_data_training_allowed"), bool):
        blockers.append("provider_profile_training_use_declaration_missing")
    return profile


def _normalize_runs(
    raw: Any,
    *,
    pixel_count: int,
    track_id: str,
    frame_id: str,
    blockers: list[str],
) -> list[dict[str, Any]]:
    if not isinstance(raw, list) or not raw or len(raw) > _MAX_RUNS_PER_OBSERVATION:
        blockers.append(f"mask_runs_missing_empty_or_too_large:{track_id}:{frame_id}")
        return []
    normalized: list[dict[str, Any]] = []
    previous_end = 0
    for row in raw:
        if not isinstance(row, Mapping):
            blockers.append(f"mask_run_invalid:{track_id}:{frame_id}")
            continue
        start = row.get("start")
        length = row.get("length")
        probability = _finite(row.get("probability"))
        if (
            isinstance(start, bool)
            or not isinstance(start, int)
            or start < previous_end
            or isinstance(length, bool)
            or not isinstance(length, int)
            or length <= 0
            or start + length > pixel_count
            or probability is None
            or not 0.0 < probability <= 1.0
        ):
            blockers.append(f"mask_run_bounds_or_probability_invalid:{track_id}:{frame_id}")
            continue
        normalized.append(
            {"start": start, "length": length, "probability": probability}
        )
        previous_end = start + length
    return normalized


def import_semantic_source_tracks(
    request: Mapping[str, Any], provider_result: Mapping[str, Any]
) -> Dict[str, Any]:
    """Return compact, source-bound mask tracks or one explicit abstention."""

    blockers: list[str] = []
    warnings: list[str] = []
    if request.get("schema_version") != REQUEST_SCHEMA_VERSION:
        blockers.append("request_schema_version_mismatch")
    bindings = request.get("bindings")
    if not isinstance(bindings, Mapping):
        blockers.append("bindings_missing")
        bindings = {}
    for field in (
        "capture_digest",
        "retained_video_digest",
        "camera_solution_digest",
        "frame_registry_digest",
        "provider_result_digest",
    ):
        if not _valid_digest(bindings.get(field)):
            blockers.append(f"binding_digest_invalid:{field}")
    frames = _validate_frame_registry(request, blockers)
    profile = _validate_profile(request, blockers)

    if provider_result.get("schema_version") != PROVIDER_RESULT_SCHEMA_VERSION:
        blockers.append("provider_result_schema_version_mismatch")
    supplied_provider_digest = provider_result.get("result_digest")
    if not _valid_digest(supplied_provider_digest) or supplied_provider_digest != (
        canonical_json_digest(
            {key: value for key, value in provider_result.items() if key != "result_digest"}
        )
    ):
        blockers.append("provider_result_digest_mismatch")
    if not _same_digest(bindings.get("provider_result_digest"), supplied_provider_digest):
        blockers.append("provider_result_request_binding_mismatch")
    provider_bindings = (
        provider_result.get("bindings")
        if isinstance(provider_result.get("bindings"), Mapping)
        else {}
    )
    for field in (
        "capture_digest",
        "retained_video_digest",
        "camera_solution_digest",
        "frame_registry_digest",
    ):
        if not _same_digest(provider_bindings.get(field), bindings.get(field)):
            blockers.append(f"provider_result_binding_mismatch:{field}")
    if not _same_digest(provider_result.get("profile_digest"), profile.get("profile_digest")):
        blockers.append("provider_result_profile_mismatch")
    if not _same_digest(provider_result.get("model_digest"), profile.get("model_digest")):
        blockers.append("provider_result_model_mismatch")
    if not _same_digest(provider_result.get("runtime_digest"), profile.get("runtime_digest")):
        blockers.append("provider_result_runtime_mismatch")

    raw_tracks = provider_result.get("tracks")
    if not isinstance(raw_tracks, list) or len(raw_tracks) > _MAX_TRACKS:
        blockers.append("provider_tracks_missing_or_too_large")
        raw_tracks = []
    seen_tracks: set[str] = set()
    observation_count = 0
    normalized_tracks: list[dict[str, Any]] = []
    frame_masks: Dict[str, dict[str, Any]] = {}
    for raw_track in raw_tracks:
        if not isinstance(raw_track, Mapping):
            blockers.append("provider_track_invalid")
            continue
        track_id = _identifier(raw_track.get("track_id"))
        label = str(raw_track.get("label") or "").strip()
        if not track_id or track_id in seen_tracks or not label or len(label) > 256:
            blockers.append("provider_track_identity_invalid_or_duplicate")
            continue
        seen_tracks.add(track_id)
        label_source = str(raw_track.get("label_source") or "").strip()
        if label_source not in {"model_inferred", "owner_supplied", "operator_reviewed"}:
            blockers.append(f"track_label_source_invalid:{track_id}")
        raw_observations = raw_track.get("observations")
        if not isinstance(raw_observations, list) or not raw_observations:
            blockers.append(f"track_observations_missing:{track_id}")
            continue
        normalized_observations: list[dict[str, Any]] = []
        seen_frame_ids: set[str] = set()
        for raw_observation in raw_observations:
            observation_count += 1
            if observation_count > _MAX_OBSERVATIONS:
                blockers.append("provider_observations_exceed_limit")
                break
            if not isinstance(raw_observation, Mapping):
                blockers.append(f"track_observation_invalid:{track_id}")
                continue
            frame_id = _identifier(raw_observation.get("source_frame_id"))
            frame = frames.get(frame_id)
            if not frame_id or frame_id in seen_frame_ids or frame is None:
                blockers.append(f"track_observation_frame_invalid:{track_id}")
                continue
            seen_frame_ids.add(frame_id)
            width = _positive_int(raw_observation.get("width"))
            height = _positive_int(raw_observation.get("height"))
            if width is None or height is None or width * height > 100_000_000:
                blockers.append(f"track_observation_dimensions_invalid:{track_id}:{frame_id}")
                continue
            if raw_observation.get("mask_encoding") != MASK_ENCODING:
                blockers.append(f"track_observation_mask_encoding_invalid:{track_id}:{frame_id}")
            if not _same_digest(
                raw_observation.get("source_frame_digest"), frame.get("source_frame_digest")
            ):
                blockers.append(f"track_observation_frame_digest_mismatch:{track_id}:{frame_id}")
            if not _same_digest(
                raw_observation.get("camera_record_digest"), frame.get("camera_record_digest")
            ):
                blockers.append(f"track_observation_camera_digest_mismatch:{track_id}:{frame_id}")
            pts = _finite(raw_observation.get("decoded_pts_seconds"))
            if pts is None or pts != _finite(frame.get("decoded_pts_seconds")):
                blockers.append(f"track_observation_pts_mismatch:{track_id}:{frame_id}")
            runs = _normalize_runs(
                raw_observation.get("runs"),
                pixel_count=width * height,
                track_id=track_id,
                frame_id=frame_id,
                blockers=blockers,
            )
            observation = {
                "source_frame_id": frame_id,
                "source_frame_digest": frame.get("source_frame_digest"),
                "decoded_pts_seconds": frame.get("decoded_pts_seconds"),
                "camera_record_digest": frame.get("camera_record_digest"),
                "width": width,
                "height": height,
                "mask_encoding": MASK_ENCODING,
                "runs": runs,
            }
            observation["observation_digest"] = canonical_json_digest(observation)
            normalized_observations.append(observation)
            frame_mask = frame_masks.setdefault(
                frame_id,
                {
                    "source_frame_id": frame_id,
                    "source_frame_digest": frame.get("source_frame_digest"),
                    "decoded_pts_seconds": frame.get("decoded_pts_seconds"),
                    "camera_record_digest": frame.get("camera_record_digest"),
                    "width": width,
                    "height": height,
                    "mask_encoding": MASK_ENCODING,
                    "track_masks": [],
                },
            )
            if frame_mask["width"] != width or frame_mask["height"] != height:
                blockers.append(f"frame_mask_dimensions_disagree:{frame_id}")
            frame_mask["track_masks"].append(
                {"track_id": track_id, "runs": runs}
            )
        normalized_observations.sort(key=lambda row: row["source_frame_id"])
        if len(normalized_observations) < 2:
            warnings.append(f"track_has_single_view_support:{track_id}")
        evidence = {
            "track_id": track_id,
            "label": label,
            "label_source": label_source,
            "observations": normalized_observations,
        }
        normalized_tracks.append(
            {
                "track_id": track_id,
                "label": label,
                "label_source": label_source,
                "mask_model_digest": profile.get("model_digest"),
                "track_evidence_digest": canonical_json_digest(evidence),
                "supporting_frame_ids": [
                    row["source_frame_id"] for row in normalized_observations
                ],
                "observation_count": len(normalized_observations),
                "semantic_authority": "inferred_candidate",
            }
        )
    if blockers:
        return _blocked(request, blockers)
    normalized_tracks.sort(key=lambda row: row["track_id"])
    normalized_frame_masks: list[dict[str, Any]] = []
    for frame_id in sorted(frame_masks):
        row = frame_masks[frame_id]
        row["track_masks"].sort(key=lambda item: item["track_id"])
        row["mask_artifact_digest"] = canonical_json_digest(row["track_masks"])
        normalized_frame_masks.append(row)
    if not normalized_tracks:
        status = "abstained"
        claim_ceiling = "no_source_tracks_detected"
        warnings.append("provider_returned_no_tracks")
    else:
        status = "completed"
        claim_ceiling = "source_bound_2d_mask_tracks_only"
    output_bindings = {
        **dict(bindings),
        "provider_profile_digest": profile.get("profile_digest"),
        "track_registry_digest": canonical_json_digest(normalized_tracks),
        "frame_masks_digest": canonical_json_digest(normalized_frame_masks),
    }
    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": status,
        "bindings": output_bindings,
        "provider_profile": profile,
        "track_registry": normalized_tracks,
        "frame_masks": normalized_frame_masks,
        "blockers": [],
        "warnings": sorted(set(warnings)),
        "claim_ceiling": claim_ceiling,
        "directly_observed_object_fact": False,
        "canonical_object_geometry": False,
        "metric_box_ready": False,
        "collision_ready": False,
        "physics_ready": False,
        "physical_task_success_established": False,
        "generated_regions_can_upgrade_claims": False,
        "comparative_policy_ranking_verdict": "thesis_not_supported",
    }
    result["result_digest"] = canonical_json_digest(result)
    return result


__all__ = [
    "MASK_ENCODING",
    "PROVIDER_RESULT_SCHEMA_VERSION",
    "REQUEST_SCHEMA_VERSION",
    "RESULT_SCHEMA_VERSION",
    "blocked_semantic_source_track_import",
    "import_semantic_source_tracks",
]
