"""Site Reference Database v1 contract helpers.

The site reference database is a derived support layer. These helpers keep the
local contract executable without calling provider or live WebApp services.
"""

from __future__ import annotations

from pathlib import Path
import math
import re
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

from .common import read_json, utc_now_iso, write_json

SITE_REFERENCE_DATABASE_SCHEMA_VERSION = "site_reference_database.v1"
WEBAPP_PROJECTION_SCHEMA_VERSION = "site_reference_webapp_projection.v1"
EVALUATION_SITE_ADMISSION_SCHEMA_VERSION = "evaluation_site_admission.v2"

REFERENCE_RECORD_REQUIRED_FIELDS = (
    "reference_id",
    "site_id",
    "scene_id",
    "capture_id",
    "authority_level",
    "storage_class",
    "capture_session_id",
    "coordinate_frame_session_id",
    "frame_id",
    "frame_index",
    "t_capture_sec",
    "T_world_camera",
    "T_site_camera",
    "intrinsics",
    "depth_uri",
    "confidence_uri",
    "embedding_uri",
    "frame_uri",
    "thumbnail_uri",
    "privacy_source",
    "geometry_source",
    "provenance_lineage",
    "privacy_lineage",
    "rights_lineage",
    "quality",
    "retrieval_signals",
    "visibility_cells",
    "anchor_observations",
    "captured_at",
    "indexed_at",
)

MANIFEST_REQUIRED_FIELDS = (
    "schema_version",
    "site_id",
    "authority_level",
    "storage_class",
    "raw_capture_authority",
    "total_reference_frames",
    "capture_count",
    "chunk_count",
    "captures",
    "coverage_summary",
    "readiness",
    "artifact_uris",
    "last_updated",
)

DENSE_RECORD_FIELD_KEYS = frozenset(
    {
        "reference_records",
        "records",
        "references",
        "T_world_camera",
        "T_site_camera",
        "intrinsics",
        "depth_uri",
        "confidence_uri",
        "embedding_uri",
        "frame_uri",
        "thumbnail_uri",
        "splat_uri",
        "plucker_map_uri",
        "visibility_cells",
        "geometry_fingerprint",
    }
)


class SiteReferenceContractError(ValueError):
    """Raised when a site-reference artifact violates the local v1 contract."""


_SHA256_RE = re.compile(r"^(?:sha256:)?[0-9a-f]{64}$")


def _admission_mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _admission_rows(value: Any) -> tuple[list[dict[str, Any]], bool]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return [], False
    if any(not isinstance(item, Mapping) for item in value):
        return [], False
    return [dict(item) for item in value], True


def _admission_string_list(value: Any) -> tuple[list[str], bool]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return [], False
    if any(not isinstance(item, str) or not item.strip() for item in value):
        return [], False
    return [item.strip() for item in value], True


def _admission_digest(value: Any) -> bool:
    return bool(_SHA256_RE.fullmatch(str(value or "").strip().lower()))


def _admission_normalized_digest(value: Any) -> str:
    digest = str(value or "").strip().lower()
    if not _SHA256_RE.fullmatch(digest):
        return ""
    return digest.removeprefix("sha256:")


def _admission_finite(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _gravity_matches_up_axis(gravity: Sequence[Any], up_axis: Any) -> bool:
    values = [_admission_finite(item) for item in gravity]
    if any(value is None for value in values):
        return False
    vector = [float(value) for value in values if value is not None]
    magnitude = math.sqrt(sum(value * value for value in vector))
    expected = {
        "+Z": (2, -1.0),
        "-Z": (2, 1.0),
        "+Y": (1, -1.0),
        "-Y": (1, 1.0),
    }.get(up_axis)
    if expected is None or not 9.0 <= magnitude <= 10.5:
        return False
    axis, sign = expected
    return vector[axis] * sign >= 0.95 * magnitude


def validate_evaluation_site_admission(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Derive evaluation readiness for a real-site scan without trusting labels.

    Assisted import, indexing, visual geometry, and review readiness are support
    states.  Only this derived contract may mark a site evaluation-ready.
    """

    blockers: list[str] = []
    if manifest.get("schema_version") != EVALUATION_SITE_ADMISSION_SCHEMA_VERSION:
        blockers.append("site_admission_schema_missing_or_unsupported")

    identity = _admission_mapping(manifest.get("immutable_source_identity"))
    for field in ("site_id", "scene_id", "capture_id", "source_bundle_id"):
        if not str(identity.get(field) or "").strip():
            blockers.append(f"immutable_source_identity_missing:{field}")
    for field in ("capture_sha256", "source_bundle_sha256", "manifest_sha256"):
        if not _admission_digest(identity.get(field)):
            blockers.append(f"immutable_source_digest_missing_or_invalid:{field}")

    verification = _admission_mapping(manifest.get("independent_evidence_verification"))
    if verification.get("status") != "verified":
        blockers.append("independent_evidence_verification_not_verified")
    if verification.get("independent_of_importer_and_model_backend") is not True:
        blockers.append("site_evidence_verifier_independence_not_proven")
    for field in ("verifier_id", "verifier_version"):
        if not str(verification.get(field) or "").strip():
            blockers.append(f"independent_evidence_verifier_identity_missing:{field}")
    for field in ("verification_report_sha256", "source_artifact_index_sha256"):
        if not _admission_digest(verification.get(field)):
            blockers.append(f"independent_evidence_verification_digest_invalid:{field}")
    if _admission_normalized_digest(verification.get("verified_source_manifest_sha256")) != (
        _admission_normalized_digest(identity.get("manifest_sha256"))
    ):
        blockers.append("independent_verification_source_manifest_digest_mismatch")

    rights = _admission_mapping(manifest.get("rights_privacy_provenance"))
    for field in (
        "consent_active",
        "rights_verified",
        "privacy_review_passed",
        "provenance_verified",
        "commercial_sim_evaluation_allowed",
    ):
        if rights.get(field) is not True:
            blockers.append(f"rights_privacy_provenance_not_proven:{field}")
    if not _admission_digest(rights.get("rights_manifest_sha256")):
        blockers.append("rights_manifest_digest_missing_or_invalid")
    for field in ("consent_scope_id", "privacy_policy_id", "provenance_chain_id"):
        if not str(rights.get(field) or "").strip():
            blockers.append(f"rights_privacy_provenance_identity_missing:{field}")
    commercial_scopes, commercial_scopes_valid = _admission_string_list(
        rights.get("commercial_use_scope")
    )
    if not commercial_scopes_valid or "sim_evaluation" not in commercial_scopes:
        blockers.append("commercial_use_scope_missing_sim_evaluation")

    geometry = _admission_mapping(manifest.get("metric_coordinate_contract"))
    if geometry.get("scale_status") != "verified_metric" or geometry.get("length_unit") != "m":
        blockers.append("metric_scale_not_verified_in_meters")
    if geometry.get("up_axis") not in {"+Z", "-Z", "+Y", "-Y"}:
        blockers.append("up_axis_missing_or_invalid")
    gravity = geometry.get("gravity_m_s2")
    if not (
        isinstance(gravity, Sequence)
        and not isinstance(gravity, (str, bytes, bytearray))
        and len(gravity) == 3
        and all(_admission_finite(item) is not None for item in gravity)
    ):
        blockers.append("gravity_vector_missing_or_invalid")
    elif not _gravity_matches_up_axis(gravity, geometry.get("up_axis")):
        blockers.append("gravity_vector_inconsistent_with_up_axis")
    if not _admission_digest(geometry.get("coordinate_frame_manifest_sha256")):
        blockers.append("coordinate_frame_manifest_digest_missing_or_invalid")
    for field in ("world_frame_id", "site_frame_id", "capture_frame_id"):
        if not str(geometry.get(field) or "").strip():
            blockers.append(f"coordinate_frame_identity_missing:{field}")
    for field in ("scale_evidence_sha256", "gravity_alignment_sha256"):
        if not _admission_digest(geometry.get(field)):
            blockers.append(f"metric_coordinate_evidence_digest_invalid:{field}")
    uncertainty = _admission_mapping(geometry.get("uncertainty"))
    for field in ("scale_sigma", "translation_sigma_m", "rotation_sigma_deg"):
        value = _admission_finite(uncertainty.get(field))
        if value is None or value < 0:
            blockers.append(f"coordinate_uncertainty_missing_or_invalid:{field}")

    calibration = _admission_mapping(manifest.get("camera_time_calibration"))
    for field in (
        "intrinsics_calibrated",
        "extrinsics_calibrated",
        "timestamps_synchronized",
        "reprojection_check_passed",
    ):
        if calibration.get(field) is not True:
            blockers.append(f"camera_time_calibration_not_proven:{field}")
    reprojection_error = _admission_finite(calibration.get("reprojection_rmse_px"))
    reprojection_limit = _admission_finite(calibration.get("maximum_reprojection_rmse_px"))
    if (
        reprojection_error is None
        or reprojection_limit is None
        or reprojection_error < 0
        or reprojection_limit < 0
        or reprojection_error > reprojection_limit
    ):
        blockers.append("camera_reprojection_error_missing_or_above_limit")
    if not _admission_digest(calibration.get("calibration_manifest_sha256")):
        blockers.append("camera_calibration_manifest_digest_missing_or_invalid")
    for field in ("intrinsics_sha256", "extrinsics_sha256", "timestamps_sha256"):
        if not _admission_digest(calibration.get(field)):
            blockers.append(f"camera_calibration_component_digest_invalid:{field}")

    viewpoints, viewpoints_payload_valid = _admission_rows(
        manifest.get("static_robot_evaluation_viewpoints")
    )
    if not viewpoints_payload_valid:
        blockers.append("static_robot_evaluation_viewpoints_payload_invalid")
    if not viewpoints:
        blockers.append("static_robot_evaluation_viewpoints_missing")
    viewpoint_ids: set[str] = set()
    source_viewpoints: set[tuple[str, str]] = set()
    for index, viewpoint in enumerate(viewpoints):
        if viewpoint.get("derived_from_moving_scan") is not True:
            blockers.append(f"static_viewpoint_not_derived_from_scan:{index}")
        if viewpoint.get("status") != "calibrated_static_viewpoint":
            blockers.append(f"static_viewpoint_not_calibrated:{index}")
        for field in (
            "viewpoint_id",
            "camera_profile_id",
            "robot_profile_id",
            "source_capture_id",
            "source_frame_id",
        ):
            if not str(viewpoint.get(field) or "").strip():
                blockers.append(f"static_viewpoint_missing:{index}:{field}")
        viewpoint_id = str(viewpoint.get("viewpoint_id") or "").strip()
        source_viewpoint = (
            str(viewpoint.get("source_capture_id") or "").strip(),
            str(viewpoint.get("source_frame_id") or "").strip(),
        )
        if viewpoint_id in viewpoint_ids:
            blockers.append(f"static_viewpoint_duplicate_identity:{index}")
        viewpoint_ids.add(viewpoint_id)
        if source_viewpoint in source_viewpoints:
            blockers.append(f"static_viewpoint_duplicate_source_frame:{index}")
        source_viewpoints.add(source_viewpoint)
        if (
            str(viewpoint.get("source_capture_id") or "").strip()
            != str(identity.get("capture_id") or "").strip()
        ):
            blockers.append(f"static_viewpoint_source_capture_mismatch:{index}")
        if not _admission_digest(viewpoint.get("pose_sha256")):
            blockers.append(f"static_viewpoint_pose_digest_invalid:{index}")
        if not _admission_digest(viewpoint.get("source_trajectory_sha256")):
            blockers.append(f"static_viewpoint_source_trajectory_digest_invalid:{index}")

    embodiment = _admission_mapping(manifest.get("robot_camera_embodiment"))
    for field in ("robot_profile_id", "camera_profile_id", "embodiment_id"):
        if not str(embodiment.get(field) or "").strip():
            blockers.append(f"robot_camera_embodiment_identity_missing:{field}")
    for field in (
        "robot_profile_sha256",
        "camera_profile_sha256",
        "embodiment_manifest_sha256",
    ):
        if not _admission_digest(embodiment.get(field)):
            blockers.append(f"robot_camera_embodiment_digest_invalid:{field}")
    for index, viewpoint in enumerate(viewpoints):
        if (
            str(viewpoint.get("robot_profile_id") or "").strip()
            != str(embodiment.get("robot_profile_id") or "").strip()
        ):
            blockers.append(f"static_viewpoint_robot_profile_mismatch:{index}")
        if (
            str(viewpoint.get("camera_profile_id") or "").strip()
            != str(embodiment.get("camera_profile_id") or "").strip()
        ):
            blockers.append(f"static_viewpoint_camera_profile_mismatch:{index}")

    grounding = _admission_mapping(manifest.get("task_scene_grounding"))
    scene_identity = str(grounding.get("scene_identity") or "").strip()
    if not scene_identity:
        blockers.append("task_scene_grounding_missing:scene_identity")
    elif scene_identity != str(identity.get("scene_id") or "").strip():
        blockers.append("task_scene_grounding_scene_identity_mismatch")
    grounding_id_fields = {
        "task_objects": "object_id",
        "articulated_parts": "part_id",
        "target_zones": "zone_id",
    }
    for field, identity_field in grounding_id_fields.items():
        rows, rows_payload_valid = _admission_rows(grounding.get(field))
        if not rows_payload_valid:
            blockers.append(f"task_scene_grounding_payload_invalid:{field}")
        if not rows:
            blockers.append(f"task_scene_grounding_missing:{field}")
        else:
            row_ids = [str(row.get(identity_field) or "").strip() for row in rows]
            if any(not row_id for row_id in row_ids):
                blockers.append(f"task_scene_grounding_identity_missing:{field}")
            if len(row_ids) != len(set(row_ids)):
                blockers.append(f"task_scene_grounding_duplicate_identity:{field}")
    if not _admission_digest(grounding.get("grounding_manifest_sha256")):
        blockers.append("task_scene_grounding_digest_missing_or_invalid")

    task_contracts, task_contracts_payload_valid = _admission_rows(manifest.get("task_contracts"))
    if not task_contracts_payload_valid:
        blockers.append("task_contracts_payload_invalid")
    if not task_contracts:
        blockers.append("explicit_task_contracts_missing")
    if not _admission_digest(manifest.get("task_contract_manifest_sha256")):
        blockers.append("task_contract_manifest_digest_missing_or_invalid")
    task_criterion_ids: set[tuple[str, str]] = set()
    for index, contract in enumerate(task_contracts):
        for field in ("task_id", "criterion_id", "evidence_type", "evaluator_mapping"):
            if not str(contract.get(field) or "").strip():
                blockers.append(f"task_contract_missing:{index}:{field}")
        tolerance = _admission_finite(contract.get("tolerance"))
        if (
            tolerance is None
            or tolerance < 0
            or not str(contract.get("tolerance_unit") or "").strip()
        ):
            blockers.append(f"task_contract_tolerance_invalid:{index}")
        task_criterion_id = (
            str(contract.get("task_id") or "").strip(),
            str(contract.get("criterion_id") or "").strip(),
        )
        if task_criterion_id in task_criterion_ids:
            blockers.append(f"task_contract_duplicate_task_criterion:{index}")
        task_criterion_ids.add(task_criterion_id)

    truth = _admission_mapping(manifest.get("truth_layers"))
    visual = _admission_mapping(truth.get("visual_geometry"))
    if visual.get("status") != "verified" or not _admission_digest(visual.get("evidence_sha256")):
        blockers.append("visual_geometry_truth_not_verified")
    for layer in ("collision", "contact", "dynamics"):
        detail = _admission_mapping(truth.get(layer))
        if detail.get("status") != "verified" or not _admission_digest(
            detail.get("evidence_sha256")
        ):
            blockers.append(f"{layer}_truth_not_verified")

    dedup = _admission_mapping(manifest.get("deduplication"))
    if dedup.get("status") != "passed":
        blockers.append("site_task_trajectory_dedup_not_passed")
    for field in ("site_dedup_id", "task_dedup_id", "trajectory_dedup_id"):
        if not str(dedup.get(field) or "").strip():
            blockers.append(f"deduplication_identity_missing:{field}")
    if not _admission_digest(dedup.get("dedup_report_sha256")):
        blockers.append("deduplication_report_digest_missing_or_invalid")

    splits = _admission_mapping(manifest.get("frozen_splits"))
    if splits.get("locked_before_evaluation") is not True:
        blockers.append("evaluation_splits_not_frozen")
    if not _admission_digest(splits.get("split_manifest_sha256")):
        blockers.append("split_manifest_digest_missing_or_invalid")
    partition_sites: dict[str, set[str]] = {}
    for name in ("train_sites", "dev_sites", "held_out_sites"):
        site_rows, site_rows_valid = _admission_string_list(splits.get(name))
        if not site_rows_valid:
            blockers.append(f"site_split_payload_invalid:{name}")
        if len(site_rows) != len(set(site_rows)):
            blockers.append(f"site_split_duplicate_site:{name}")
        partition_sites[name] = set(site_rows)
    if not partition_sites["held_out_sites"]:
        blockers.append("held_out_site_split_missing")
    overlaps = (
        partition_sites["train_sites"] & partition_sites["dev_sites"]
        | partition_sites["train_sites"] & partition_sites["held_out_sites"]
        | partition_sites["dev_sites"] & partition_sites["held_out_sites"]
    )
    if overlaps:
        blockers.append("site_split_overlap_detected")
    site_id = str(identity.get("site_id") or "")
    if site_id and sum(site_id in sites for sites in partition_sites.values()) != 1:
        blockers.append("site_not_assigned_to_exactly_one_frozen_split")

    ood = _admission_mapping(manifest.get("ood_abstention"))
    ood_axes, ood_axes_payload_valid = _admission_rows(ood.get("axes"))
    if not ood_axes_payload_valid:
        blockers.append("ood_abstention_axes_payload_invalid")
    if ood.get("abstention_enabled") is not True or not ood_axes:
        blockers.append("ood_abstention_contract_missing")
    if ood.get("out_of_distribution_behavior") != "abstain":
        blockers.append("ood_behavior_must_abstain")
    if not _admission_digest(ood.get("calibration_manifest_sha256")):
        blockers.append("ood_calibration_manifest_digest_missing_or_invalid")
    ood_axis_ids = [str(row.get("axis") or "").strip() for row in ood_axes]
    if any(not axis for axis in ood_axis_ids) or len(ood_axis_ids) != len(set(ood_axis_ids)):
        blockers.append("ood_axis_identity_missing_or_duplicate")

    blockers = sorted(set(blockers))
    assisted_import = str(manifest.get("importer_kind") or "").startswith("scaniverse")
    return {
        "schema_version": EVALUATION_SITE_ADMISSION_SCHEMA_VERSION,
        "status": "evaluation_ready" if not blockers else "blocked",
        "evaluation_ready": not blockers,
        "site_id": identity.get("site_id"),
        "scene_id": identity.get("scene_id"),
        "importer_kind": manifest.get("importer_kind"),
        "scaniverse_assisted_import": assisted_import,
        "blockers": blockers,
        "claim_boundary": {
            "site_reference_indexing_is_not_evaluation_admission": True,
            "assisted_import_is_not_evaluation_readiness": True,
            "visual_geometry_is_separate_from_collision_contact_and_dynamics": True,
            "physical_robot_performance_proven": False,
        },
    }


def validate_site_reference_record(record: Mapping[str, Any]) -> None:
    """Validate the shape of one `site_reference_index.jsonl` row."""
    missing = [field for field in REFERENCE_RECORD_REQUIRED_FIELDS if field not in record]
    if missing:
        raise SiteReferenceContractError(
            "site_reference_record_missing_fields:" + ",".join(missing)
        )
    if record.get("authority_level") != "derived_reference_record":
        raise SiteReferenceContractError("site_reference_record_authority_level_invalid")
    if record.get("storage_class") != "jsonl_reference_record":
        raise SiteReferenceContractError("site_reference_record_storage_class_invalid")
    _validate_matrix_or_null(record.get("T_world_camera"), field="T_world_camera", allow_null=False)
    _validate_matrix_or_null(record.get("T_site_camera"), field="T_site_camera", allow_null=True)
    if not isinstance(record.get("intrinsics"), Mapping) or not record.get("intrinsics"):
        raise SiteReferenceContractError("site_reference_record_intrinsics_missing")
    for lineage_field in ("provenance_lineage", "privacy_lineage", "rights_lineage"):
        if not isinstance(record.get(lineage_field), Mapping):
            raise SiteReferenceContractError(f"site_reference_record_{lineage_field}_invalid")


def validate_site_reference_manifest(payload: Mapping[str, Any]) -> None:
    """Validate the site-reference manifest summary shape."""
    missing = [field for field in MANIFEST_REQUIRED_FIELDS if field not in payload]
    if missing:
        raise SiteReferenceContractError(
            "site_reference_manifest_missing_fields:" + ",".join(missing)
        )
    if payload.get("schema_version") != SITE_REFERENCE_DATABASE_SCHEMA_VERSION:
        raise SiteReferenceContractError("site_reference_manifest_schema_version_invalid")
    if payload.get("authority_level") != "derived_site_reference_manifest":
        raise SiteReferenceContractError("site_reference_manifest_authority_level_invalid")
    if payload.get("storage_class") != "object_storage_manifest":
        raise SiteReferenceContractError("site_reference_manifest_storage_class_invalid")
    if not isinstance(payload.get("artifact_uris"), Mapping):
        raise SiteReferenceContractError("site_reference_manifest_artifact_uris_invalid")
    if not isinstance(payload.get("readiness"), Mapping):
        raise SiteReferenceContractError("site_reference_manifest_readiness_invalid")


def build_reference_record_lineage(
    *,
    capture_prefix_uri: Optional[str],
    descriptor_uri: Optional[str],
    geometry_source: str,
    privacy_source: str,
    descriptor: Mapping[str, Any],
) -> Dict[str, Dict[str, Any]]:
    """Build conservative provenance/privacy/rights lineage for an index record."""
    raw_manifest_uri = f"{capture_prefix_uri}/raw/manifest.json" if capture_prefix_uri else None
    raw_rights_uri = f"{capture_prefix_uri}/raw/rights_consent.json" if capture_prefix_uri else None
    rights = _rights_payload_from_descriptor(descriptor)
    derived_generation_allowed = _first_present_bool(
        rights,
        (
            "derived_scene_generation_allowed",
            "derived_generation_allowed",
            "world_model_generation_allowed",
            "commercialization_allowed",
        ),
    )
    rights_status = str(
        rights.get("rights_status")
        or rights.get("consent_status")
        or rights.get("rights_profile")
        or "unknown"
    ).strip()
    return {
        "provenance_lineage": {
            "raw_capture_prefix_uri": capture_prefix_uri,
            "raw_manifest_uri": raw_manifest_uri,
            "capture_descriptor_uri": descriptor_uri,
            "derived_from": [
                "raw_capture",
                "capture_descriptor",
                "privacy_safe_video",
                "geometry_reference",
            ],
            "geometry_source": geometry_source,
        },
        "privacy_lineage": {
            "privacy_source": privacy_source,
            "privacy_safe_required": True,
            "privacy_status": "privacy_safe_source"
            if privacy_source.startswith("privacy/")
            else "raw_or_unknown_source",
        },
        "rights_lineage": {
            "rights_source_uri": raw_rights_uri,
            "rights_status": rights_status or "unknown",
            "derived_scene_generation_allowed": derived_generation_allowed,
            "claim_policy": "do_not_infer_rights_clearance",
        },
    }


def build_site_reference_manifest_payload(
    *,
    site_id: str,
    total_reference_frames: int,
    capture_count: int,
    chunk_count: int,
    captures: Iterable[Mapping[str, Any]],
    coverage_summary: Mapping[str, Any],
    artifact_uris: Mapping[str, Any],
    readiness: Mapping[str, Any],
    site_frame_established: bool,
    last_updated: Optional[str] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "schema_version": SITE_REFERENCE_DATABASE_SCHEMA_VERSION,
        "site_id": site_id,
        "authority_level": "derived_site_reference_manifest",
        "storage_class": "object_storage_manifest",
        "raw_capture_authority": {
            "authority": "BlueprintCapture raw bundle",
            "rule": "Raw capture, provenance, rights, privacy, timestamps, poses, and device metadata remain authoritative.",
        },
        "total_reference_frames": int(total_reference_frames),
        "capture_count": int(capture_count),
        "chunk_count": int(chunk_count),
        "captures": [dict(item) for item in captures],
        "coverage_summary": dict(coverage_summary),
        "readiness": dict(readiness),
        "artifact_uris": {str(key): value for key, value in artifact_uris.items() if value},
        "last_updated": last_updated or utc_now_iso(),
        "site_frame_established": bool(site_frame_established),
    }
    validate_site_reference_manifest(payload)
    return payload


def build_site_reference_summary_projection(
    *,
    site_id: str,
    site_root: Path,
    site_index_path: Path,
    storage_root: Path,
    manifest_payload: Optional[Mapping[str, Any]] = None,
    validation_payload: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a WebApp/Firestore-safe summary projection from local artifacts."""
    manifest = dict(
        manifest_payload or _read_optional_json(site_root / "site_reference_manifest.json")
    )
    validation = dict(
        validation_payload or _read_optional_json(site_root / "retrieval_validation.json")
    )
    counts = {
        "total_reference_frames": int(manifest.get("total_reference_frames") or 0),
        "capture_count": int(manifest.get("capture_count") or 0),
        "chunk_count": int(manifest.get("chunk_count") or validation.get("chunk_count") or 0),
    }
    coverage_summary = (
        dict(manifest.get("coverage_summary") or {})
        if isinstance(manifest.get("coverage_summary"), Mapping)
        else {}
    )
    readiness = _site_reference_readiness(
        manifest=manifest,
        validation=validation,
        counts=counts,
    )
    payload: Dict[str, Any] = {
        "schema_version": WEBAPP_PROJECTION_SCHEMA_VERSION,
        "site_id": site_id,
        "authority_level": "derived_summary_projection",
        "storage_class": "firestore_summary_safe",
        "artifact_uris": {
            "site_reference_manifest_uri": _path_to_gs_uri(
                site_root / "site_reference_manifest.json",
                storage_root=storage_root,
            ),
            "site_reference_index_uri": _path_to_gs_uri(site_index_path, storage_root=storage_root),
            "site_reference_summary_projection_uri": _path_to_gs_uri(
                site_root / "site_reference_summary_projection.json",
                storage_root=storage_root,
            ),
            "retrieval_validation_uri": _path_to_gs_uri(
                site_root / "retrieval_validation.json",
                storage_root=storage_root,
            ),
            "coverage_map_uri": _path_to_gs_uri(
                site_root / "coverage" / "coverage_map.json",
                storage_root=storage_root,
            ),
            "indices_manifest_uri": _path_to_gs_uri(
                site_root / "indices" / "manifest.json",
                storage_root=storage_root,
            ),
            "site_overlap_graph_uri": _path_to_gs_uri(
                site_root / "site_overlap_graph.json",
                storage_root=storage_root,
            ),
        },
        "readiness": readiness,
        "counts": counts,
        "scores": {
            "coverage_fraction": coverage_summary.get("coverage_fraction"),
            "geometry_fingerprint_coverage": validation.get("geometry_fingerprint_coverage"),
            "mean_staticness_score": validation.get("mean_staticness_score"),
            "aligned_fraction": validation.get("aligned_fraction"),
        },
        "blockers": readiness["blockers"],
        "last_updated": str(
            manifest.get("last_updated") or validation.get("generated_at") or utc_now_iso()
        ),
    }
    assert_summary_projection_safe(payload)
    return payload


def write_site_reference_summary_projection(
    *,
    site_id: str,
    site_root: Path,
    site_index_path: Path,
    storage_root: Path,
    manifest_payload: Optional[Mapping[str, Any]] = None,
    validation_payload: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    payload = build_site_reference_summary_projection(
        site_id=site_id,
        site_root=site_root,
        site_index_path=site_index_path,
        storage_root=storage_root,
        manifest_payload=manifest_payload,
        validation_payload=validation_payload,
    )
    write_json(site_root / "site_reference_summary_projection.json", payload)
    return payload


def assert_summary_projection_safe(payload: Mapping[str, Any]) -> None:
    """Reject dense per-record fields from a WebApp/Firestore summary payload."""
    if payload.get("schema_version") != WEBAPP_PROJECTION_SCHEMA_VERSION:
        raise SiteReferenceContractError("site_reference_projection_schema_version_invalid")
    if payload.get("storage_class") != "firestore_summary_safe":
        raise SiteReferenceContractError("site_reference_projection_storage_class_invalid")
    violations = sorted(_find_dense_field_violations(payload))
    if violations:
        raise SiteReferenceContractError(
            "site_reference_projection_contains_dense_fields:" + ",".join(violations)
        )


def _find_dense_field_violations(value: Any, *, path: str = "$") -> set[str]:
    violations: set[str] = set()
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            child_path = f"{path}.{key_text}"
            if key_text in DENSE_RECORD_FIELD_KEYS:
                violations.add(child_path)
            violations.update(_find_dense_field_violations(child, path=child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            violations.update(_find_dense_field_violations(child, path=f"{path}[{index}]"))
    return violations


def _site_reference_readiness(
    *,
    manifest: Mapping[str, Any],
    validation: Mapping[str, Any],
    counts: Mapping[str, int],
) -> Dict[str, Any]:
    blockers: list[str] = []
    if int(counts.get("total_reference_frames") or 0) <= 0:
        blockers.append("no_reference_frames")
    if int(counts.get("capture_count") or 0) <= 0:
        blockers.append("no_captures_indexed")
    geometry_coverage = _optional_float(validation.get("geometry_fingerprint_coverage"))
    if geometry_coverage is not None and geometry_coverage < 0.5:
        blockers.append("low_geometry_fingerprint_coverage")
    if not bool(manifest.get("site_frame_established")):
        blockers.append("site_frame_not_established")

    if not blockers:
        state = "ready"
    elif int(counts.get("total_reference_frames") or 0) > 0:
        state = "degraded"
    else:
        state = "blocked"
    return {
        "state": state,
        "blockers": blockers,
        "operational_launch_ready": False,
        "claim_policy": "local_site_reference_readiness_only",
    }


def _rights_payload_from_descriptor(descriptor: Mapping[str, Any]) -> Mapping[str, Any]:
    candidates = [
        descriptor.get("capture_rights"),
        descriptor.get("rights"),
        descriptor.get("rights_consent"),
    ]
    metadata = descriptor.get("metadata")
    if isinstance(metadata, Mapping):
        candidates.extend(
            [
                metadata.get("capture_rights"),
                metadata.get("rights"),
                metadata.get("rights_consent"),
            ]
        )
    for candidate in candidates:
        if isinstance(candidate, Mapping):
            return candidate
    return {}


def _first_present_bool(payload: Mapping[str, Any], keys: Iterable[str]) -> Optional[bool]:
    for key in keys:
        if key in payload:
            value = payload.get(key)
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                normalized = value.strip().lower()
                if normalized in {"true", "yes", "1", "allowed"}:
                    return True
                if normalized in {"false", "no", "0", "blocked", "denied"}:
                    return False
    return None


def _read_optional_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return read_json(path)
    except Exception:
        return {}


def _path_to_gs_uri(path: Path, *, storage_root: Path) -> Optional[str]:
    try:
        rel = path.resolve().relative_to(storage_root.resolve())
    except ValueError:
        return str(path)
    parts = rel.parts
    if len(parts) < 2:
        return None
    bucket = parts[0]
    key = "/".join(parts[1:])
    return f"gs://{bucket}/{key}"


def _validate_matrix_or_null(value: Any, *, field: str, allow_null: bool) -> None:
    if value is None and allow_null:
        return
    if not isinstance(value, list) or len(value) != 4:
        raise SiteReferenceContractError(f"site_reference_record_{field}_invalid")
    for row in value:
        if not isinstance(row, list) or len(row) != 4:
            raise SiteReferenceContractError(f"site_reference_record_{field}_invalid")


def _optional_float(value: Any) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None
