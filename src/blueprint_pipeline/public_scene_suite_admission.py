"""Fail-closed ADP-009A admission for a public scene test-suite manifest.

This contract is additive to :mod:`public_reference_admission`.  It admits the
metadata needed to materialize a development-only scene test suite; it does
not open remote artifacts, qualify their geometry, authorize dataset use, or
promote a public scene into partner or physical evidence.

JSON-shaped manifests always produce a digest-bound ``admitted`` or ``blocked``
receipt.  Only a non-object or non-JSON-serializable input raises an exception.
That keeps expected research gaps visible as data instead of turning them into
an opaque control-flow failure.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from .decision_evidence_contracts import canonical_digest


MANIFEST_SCHEMA_VERSION = "public_scene_suite_manifest.v1"
RECEIPT_SCHEMA_VERSION = "public_scene_suite_admission_receipt.v1"
PROGRAM_ID = "arm-decision-proof-v1"
ADP_ITEM = "ADP-009A"
PHASE_LABEL = "public_scene_qualification"
CLAIM_CEILING = "development_only"

_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
_GIT_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:+-]{0,191}$")
_MOVING_REVISIONS = {"current", "head", "latest", "main", "master", "tip", "trunk"}
_SOURCE_KINDS = {
    "collision_companion",
    "other_public_scene_source",
    "real_metric_scene",
    "simready_task_object",
    "synthetic_metric_scene",
}
_SOURCE_PROJECT_KINDS = {
    "Blueprint-controlled": _SOURCE_KINDS,
    "InteriorGS": {"synthetic_metric_scene"},
    "SAGE-3D": {"collision_companion"},
    "ScanNet++": {"real_metric_scene"},
}
_ARTIFACT_ROLES = {
    "appearance_3dgs",
    "camera_model_bundle",
    "calibration_observation",
    "calibration_metadata",
    "clean_background_depth_truth",
    "clean_background_geometry_truth",
    "clean_background_rgb_truth",
    "metric_geometry",
    "metric_scale_evidence",
    "method_input_mask",
    "method_input_rgb",
    "method_input_splat_depth",
    "semantic_metadata",
    "simready_usd_package",
    "static_collision_geometry",
    "task_object_collision_geometry",
    "task_object_physics_metadata",
    "task_object_visual_geometry",
    "test_observation",
    "unit_conversion_receipt",
    "validation_depth_oracle",
}
_CODE_CAPABILITY_ROLES = {
    "background_completion_effect_ablation",
    "background_completion_primary_adapter",
    "background_completion_quality_challenger",
    "background_completion_reproducibility_control",
    "scene_materialization",
    "segmentation",
    "simready_authoring",
    "simulator_integration",
}
_USE_SCOPES = {
    "commercial_and_redistribution",
    "commercial_internal_development",
    "noncommercial_internal_research",
}
_CODE_AVAILABILITY = {"paper_only", "proprietary_unverified", "released"}
_ALLOWED_SCALE_AUTHORITIES = {
    "authored_metric_environment",
    "calibrated_stereo",
    "dataset_metric_camera_poses",
    "known_length_reference",
    "laser_scan",
    "rgbd_sensor",
    "surveyed_control",
}
_FORBIDDEN_SCALE_AUTHORITY_MARKERS = {
    "3d_gaussian",
    "3dgs",
    "da3",
    "depth_anything",
    "gaussian",
    "learned_monocular_depth",
    "sam",
    "segmentation_anything",
}
_FORBIDDEN_TRUE_CLAIMS = {
    "artifact_bytes_verified",
    "customer_value",
    "deployment_readiness",
    "digital_twin",
    "general_sim_to_real_fidelity",
    "metric_geometry_qualified",
    "partner_capture_qualified",
    "physical_safety",
    "prospective_validation",
    "public_scene_software_qualification",
    "task_physics_qualified",
}


class PublicSceneSuiteAdmissionError(ValueError):
    """Input could not be represented as a JSON object."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(sorted(set(str(error) for error in errors if str(error))))
        super().__init__("; ".join(self.errors))


def _clone(value: Mapping[str, Any]) -> dict[str, Any]:
    try:
        cloned = json.loads(json.dumps(dict(value)))
    except (TypeError, ValueError) as exc:
        raise PublicSceneSuiteAdmissionError(["manifest:not_json_serializable"]) from exc
    if not isinstance(cloned, dict):
        raise PublicSceneSuiteAdmissionError(["manifest:not_mapping"])
    return cloned


def _string(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _reject_unknown(
    value: Mapping[str, Any],
    *,
    allowed: set[str],
    path: str,
    blockers: list[str],
) -> None:
    for key in sorted(set(value) - allowed):
        blockers.append(f"{path}.{key}:unknown_property" if path else f"{key}:unknown_property")


def _is_identifier(value: Any) -> bool:
    return bool(_IDENTIFIER.fullmatch(_string(value)))


def _is_sha256(value: Any) -> bool:
    return bool(_SHA256.fullmatch(_string(value)))


def _is_https_url(value: Any) -> bool:
    return _string(value).startswith("https://") and len(_string(value)) > len("https://")


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _numeric_vector(value: Any, *, length: int) -> list[float] | None:
    if not isinstance(value, list) or len(value) != length:
        return None
    result = [_finite_number(item) for item in value]
    if any(item is None for item in result):
        return None
    return [float(item) for item in result if item is not None]


def _matrix3(value: Any) -> list[list[float]] | None:
    if not isinstance(value, list) or len(value) != 3:
        return None
    result: list[list[float]] = []
    for row in value:
        normalized = _numeric_vector(row, length=3)
        if normalized is None:
            return None
        result.append(normalized)
    return result


def _rows(value: Any, *, path: str, blockers: list[str], nonempty: bool = True) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        blockers.append(f"{path}:must_be_array")
        return []
    if nonempty and not value:
        blockers.append(f"{path}:missing")
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(value):
        if not isinstance(row, Mapping):
            blockers.append(f"{path}[{index}]:must_be_object")
        else:
            rows.append(dict(row))
    return rows


def _identifier_list(value: Any, *, path: str, blockers: list[str]) -> list[str]:
    if not isinstance(value, list) or not value:
        blockers.append(f"{path}:missing_or_not_array")
        return []
    result: list[str] = []
    for index, item in enumerate(value):
        text = _string(item)
        if not _is_identifier(text):
            blockers.append(f"{path}[{index}]:invalid")
        result.append(text)
    if len(result) != len(set(result)):
        blockers.append(f"{path}:duplicate")
    return result


def _identifier_array(
    value: Any, *, path: str, blockers: list[str], allow_empty: bool = False
) -> list[str]:
    if not isinstance(value, list) or (not value and not allow_empty):
        blockers.append(f"{path}:missing_or_not_array")
        return []
    result: list[str] = []
    for index, item in enumerate(value):
        text = _string(item)
        if not _is_identifier(text):
            blockers.append(f"{path}[{index}]:invalid")
        result.append(text)
    if len(result) != len(set(result)):
        blockers.append(f"{path}:duplicate")
    return result


def _validate_revision(value: Any, *, path: str, blockers: list[str]) -> None:
    revision = _mapping(value)
    _reject_unknown(
        revision,
        allowed={"kind", "value"},
        path=path,
        blockers=blockers,
    )
    kind = _string(revision.get("kind"))
    revision_value = _string(revision.get("value"))
    if kind not in {"content_digest", "git_commit", "release_tag"}:
        blockers.append(f"{path}.kind:invalid")
        return
    if kind == "git_commit" and not _GIT_COMMIT.fullmatch(revision_value):
        blockers.append(f"{path}.value:not_exact_git_commit")
    elif kind == "content_digest" and not _is_sha256(revision_value):
        blockers.append(f"{path}.value:not_exact_content_digest")
    elif kind == "release_tag" and (
        not _is_identifier(revision_value) or revision_value.lower() in _MOVING_REVISIONS
    ):
        blockers.append(f"{path}.value:not_exact_release_tag")


def _validate_relative_path(value: Any, *, path: str, blockers: list[str]) -> None:
    text = _string(value)
    candidate = Path(text)
    if not text or candidate.is_absolute() or ".." in candidate.parts:
        blockers.append(f"{path}:invalid")


def _matrix4(value: Any) -> list[list[float]] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    result: list[list[float]] = []
    for row in value:
        if not isinstance(row, list) or len(row) != 4:
            return None
        normalized_row: list[float] = []
        for item in row:
            if isinstance(item, bool):
                return None
            try:
                number = float(item)
            except (TypeError, ValueError):
                return None
            if not math.isfinite(number):
                return None
            normalized_row.append(number)
        result.append(normalized_row)
    return result


def _matrix_product(left: list[list[float]], right: list[list[float]]) -> list[list[float]]:
    return [
        [sum(left[row][index] * right[index][column] for index in range(4)) for column in range(4)]
        for row in range(4)
    ]


def _is_identity(value: list[list[float]], *, tolerance: float = 1e-8) -> bool:
    return all(
        abs(value[row][column] - (1.0 if row == column else 0.0)) <= tolerance
        for row in range(4)
        for column in range(4)
    )


def _is_metric_isometry(value: list[list[float]], *, tolerance: float = 1e-8) -> bool:
    if any(abs(value[3][index] - (1.0 if index == 3 else 0.0)) > tolerance for index in range(4)):
        return False
    basis = [[value[row][column] for row in range(3)] for column in range(3)]
    for left_index in range(3):
        for right_index in range(3):
            observed = sum(
                basis[left_index][axis] * basis[right_index][axis] for axis in range(3)
            )
            expected = 1.0 if left_index == right_index else 0.0
            if abs(observed - expected) > tolerance:
                return False
    return True


def _determinant3(value: list[list[float]]) -> float:
    return (
        value[0][0] * (value[1][1] * value[2][2] - value[1][2] * value[2][1])
        - value[0][1] * (value[1][0] * value[2][2] - value[1][2] * value[2][0])
        + value[0][2] * (value[1][0] * value[2][1] - value[1][1] * value[2][0])
    )


def _is_physically_valid_inertia(
    value: list[list[float]], *, tolerance: float = 1e-12
) -> bool:
    """Return whether a 3x3 inertia tensor is symmetric and physically realizable.

    Positive diagonal entries alone are insufficient: a symmetric tensor may
    still have a negative principal moment.  Sylvester's criterion establishes
    positive definiteness.  The covariance-equivalent matrix
    ``trace(I) / 2 * identity - I`` must also be positive semidefinite, which is
    equivalent to the triangle inequalities on the principal moments of a
    physically realizable rigid body.
    """

    if any(
        abs(value[row][column] - value[column][row]) > tolerance
        for row in range(3)
        for column in range(3)
    ):
        return False
    if value[0][0] <= tolerance:
        return False
    if value[0][0] * value[1][1] - value[0][1] ** 2 <= tolerance:
        return False
    if _determinant3(value) <= tolerance:
        return False

    half_trace = 0.5 * sum(value[index][index] for index in range(3))
    covariance = [
        [
            (half_trace if row == column else 0.0) - value[row][column]
            for column in range(3)
        ]
        for row in range(3)
    ]
    principal_minors = [
        covariance[index][index] for index in range(3)
    ] + [
        covariance[left][left] * covariance[right][right]
        - covariance[left][right] ** 2
        for left, right in ((0, 1), (0, 2), (1, 2))
    ]
    return all(minor >= -tolerance for minor in principal_minors) and (
        _determinant3(covariance) >= -tolerance
    )


def _source_up_maps_to_world_up(
    source_to_world: list[list[float]], source_up_axis: str, *, tolerance: float = 1e-8
) -> bool:
    source_up_vectors = {
        "+X": (1.0, 0.0, 0.0),
        "-X": (-1.0, 0.0, 0.0),
        "+Y": (0.0, 1.0, 0.0),
        "-Y": (0.0, -1.0, 0.0),
        "+Z": (0.0, 0.0, 1.0),
        "-Z": (0.0, 0.0, -1.0),
    }
    source_up = source_up_vectors.get(source_up_axis)
    if source_up is None:
        return False
    mapped = [
        sum(source_to_world[row][column] * source_up[column] for column in range(3))
        for row in range(3)
    ]
    return all(
        abs(observed - expected) <= tolerance
        for observed, expected in zip(mapped, (0.0, 0.0, 1.0))
    )


def _validate_date(value: Any, *, path: str, blockers: list[str]) -> None:
    try:
        dt.date.fromisoformat(_string(value))
    except ValueError:
        blockers.append(f"{path}:invalid")


def _date(value: Any) -> dt.date | None:
    try:
        return dt.date.fromisoformat(_string(value))
    except ValueError:
        return None


def _validate_manifest(
    value: Mapping[str, Any], *, evaluation_date: dt.date
) -> list[str]:
    blockers: list[str] = []

    _reject_unknown(
        value,
        allowed={
            "schema_version",
            "program_id",
            "adp_item",
            "phase_label",
            "suite_id",
            "admission_as_of",
            "suite_purpose",
            "component_scope",
            "sources",
            "scene_pairings",
            "rights_reviews",
            "code_dependencies",
            "coordinate_frames",
            "splits",
            "observation_bundle",
            "representations",
            "claim_ceiling",
            "claim_boundaries",
            "manifest_digest",
        },
        path="",
        blockers=blockers,
    )

    constants = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "program_id": PROGRAM_ID,
        "adp_item": ADP_ITEM,
        "phase_label": PHASE_LABEL,
        "suite_purpose": "public_scene_software_qualification",
        "component_scope": "hybrid_edit_replacement_case",
        "claim_ceiling": CLAIM_CEILING,
    }
    for key, expected in constants.items():
        if _string(value.get(key)) != expected:
            blockers.append(f"{key}:must_be:{expected}")
    if not _is_identifier(value.get("suite_id")):
        blockers.append("suite_id:invalid")
    admission_as_of = _date(value.get("admission_as_of"))
    if admission_as_of is None:
        blockers.append("admission_as_of:invalid")
    elif admission_as_of != evaluation_date:
        blockers.append("admission_as_of:does_not_match_authoritative_evaluation_date")

    sources = _rows(value.get("sources"), path="sources", blockers=blockers)
    source_by_id: dict[str, dict[str, Any]] = {}
    artifact_by_id: dict[str, dict[str, Any]] = {}
    artifact_count = 0
    for source_index, source in enumerate(sources):
        prefix = f"sources[{source_index}]"
        _reject_unknown(
            source,
            allowed={
                "source_id",
                "upstream_project_id",
                "source_kind",
                "scene_id",
                "revision",
                "source_url",
                "artifacts",
            },
            path=prefix,
            blockers=blockers,
        )
        source_id = _string(source.get("source_id"))
        upstream_project_id = _string(source.get("upstream_project_id"))
        scene_id = _string(source.get("scene_id"))
        if not _is_identifier(source_id):
            blockers.append(f"{prefix}.source_id:invalid")
        elif source_id in source_by_id:
            blockers.append(f"{prefix}.source_id:duplicate")
        else:
            source_by_id[source_id] = source
        if not _is_identifier(upstream_project_id):
            blockers.append(f"{prefix}.upstream_project_id:invalid")
        elif upstream_project_id not in _SOURCE_PROJECT_KINDS:
            blockers.append(f"{prefix}.upstream_project_id:not_active_program_source")
        source_kind = _string(source.get("source_kind"))
        if source_kind not in _SOURCE_KINDS:
            blockers.append(f"{prefix}.source_kind:invalid")
        elif upstream_project_id in _SOURCE_PROJECT_KINDS and source_kind not in (
            _SOURCE_PROJECT_KINDS[upstream_project_id]
        ):
            blockers.append(f"{prefix}.source_kind:project_role_mismatch")
        if not _is_identifier(scene_id):
            blockers.append(f"{prefix}.scene_id:invalid")
        _validate_revision(source.get("revision"), path=f"{prefix}.revision", blockers=blockers)
        if not _is_https_url(source.get("source_url")):
            blockers.append(f"{prefix}.source_url:invalid")

        artifacts = _rows(source.get("artifacts"), path=f"{prefix}.artifacts", blockers=blockers)
        artifact_count += len(artifacts)
        for artifact_index, artifact in enumerate(artifacts):
            artifact_prefix = f"{prefix}.artifacts[{artifact_index}]"
            _reject_unknown(
                artifact,
                allowed={
                    "artifact_id",
                    "scene_id",
                    "role",
                    "relative_path",
                    "sha256",
                    "size_bytes",
                },
                path=artifact_prefix,
                blockers=blockers,
            )
            artifact_id = _string(artifact.get("artifact_id"))
            if not _is_identifier(artifact_id):
                blockers.append(f"{artifact_prefix}.artifact_id:invalid")
            elif artifact_id in artifact_by_id:
                blockers.append(f"{artifact_prefix}.artifact_id:duplicate")
            else:
                artifact_by_id[artifact_id] = {
                    **artifact,
                    "source_id": source_id,
                    "source_scene_id": scene_id,
                }
            if _string(artifact.get("scene_id")) != scene_id:
                blockers.append(f"{artifact_prefix}.scene_id:source_scene_mismatch")
            if _string(artifact.get("role")) not in _ARTIFACT_ROLES:
                blockers.append(f"{artifact_prefix}.role:invalid")
            _validate_relative_path(
                artifact.get("relative_path"),
                path=f"{artifact_prefix}.relative_path",
                blockers=blockers,
            )
            if not _is_sha256(artifact.get("sha256")):
                blockers.append(f"{artifact_prefix}.sha256:invalid")
            size = artifact.get("size_bytes")
            if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
                blockers.append(f"{artifact_prefix}.size_bytes:invalid")
            if artifact.get("role") == "simready_usd_package" and Path(
                _string(artifact.get("relative_path"))
            ).suffix.lower() not in {".usd", ".usda", ".usdc", ".usdz"}:
                blockers.append(f"{artifact_prefix}.relative_path:not_usd_package")

        if _string(source.get("source_kind")) == "simready_task_object":
            source_roles = {_string(artifact.get("role")) for artifact in artifacts}
            required_roles = {
                "simready_usd_package",
                "task_object_visual_geometry",
                "task_object_collision_geometry",
                "task_object_physics_metadata",
            }
            for missing_role in sorted(required_roles - source_roles):
                blockers.append(f"{prefix}.artifacts:missing_role:{missing_role}")

    pairings = _rows(
        value.get("scene_pairings"), path="scene_pairings", blockers=blockers
    )
    pairing_ids: list[str] = []
    for index, pairing in enumerate(pairings):
        prefix = f"scene_pairings[{index}]"
        _reject_unknown(
            pairing,
            allowed={
                "pairing_id",
                "appearance_source_id",
                "appearance_scene_id",
                "appearance_artifact_id",
                "collision_source_id",
                "collision_scene_id",
                "collision_artifact_id",
                "exact_scene_match_required",
            },
            path=prefix,
            blockers=blockers,
        )
        pairing_id = _string(pairing.get("pairing_id"))
        if not _is_identifier(pairing_id):
            blockers.append(f"{prefix}.pairing_id:invalid")
        pairing_ids.append(pairing_id)
        if pairing.get("exact_scene_match_required") is not True:
            blockers.append(f"{prefix}.exact_scene_match_required:must_be_true")

        appearance_source_id = _string(pairing.get("appearance_source_id"))
        collision_source_id = _string(pairing.get("collision_source_id"))
        appearance_scene_id = _string(pairing.get("appearance_scene_id"))
        collision_scene_id = _string(pairing.get("collision_scene_id"))
        appearance_source = source_by_id.get(appearance_source_id)
        collision_source = source_by_id.get(collision_source_id)
        if appearance_source is None:
            blockers.append(f"{prefix}.appearance_source_id:unknown")
        elif appearance_scene_id != _string(appearance_source.get("scene_id")):
            blockers.append(f"{prefix}.appearance_scene_id:source_scene_mismatch")
        if collision_source is None:
            blockers.append(f"{prefix}.collision_source_id:unknown")
        elif collision_scene_id != _string(collision_source.get("scene_id")):
            blockers.append(f"{prefix}.collision_scene_id:source_scene_mismatch")
        if appearance_scene_id != collision_scene_id:
            blockers.append(f"{prefix}:exact_scene_pair_mismatch")

        appearance_artifact_id = _string(pairing.get("appearance_artifact_id"))
        collision_artifact_id = _string(pairing.get("collision_artifact_id"))
        appearance_artifact = artifact_by_id.get(appearance_artifact_id)
        collision_artifact = artifact_by_id.get(collision_artifact_id)
        if appearance_artifact is None:
            blockers.append(f"{prefix}.appearance_artifact_id:unknown")
        elif (
            appearance_artifact.get("source_id") != appearance_source_id
            or appearance_artifact.get("role") != "appearance_3dgs"
        ):
            blockers.append(f"{prefix}.appearance_artifact_id:source_or_role_mismatch")
        if collision_artifact is None:
            blockers.append(f"{prefix}.collision_artifact_id:unknown")
        elif (
            collision_artifact.get("source_id") != collision_source_id
            or collision_artifact.get("role") != "static_collision_geometry"
        ):
            blockers.append(f"{prefix}.collision_artifact_id:source_or_role_mismatch")
    if len(pairing_ids) != len(set(pairing_ids)):
        blockers.append("scene_pairings:pairing_id_duplicate")

    rights_reviews = _rows(
        value.get("rights_reviews"), path="rights_reviews", blockers=blockers
    )
    reviewed_source_ids: list[str] = []
    for index, review in enumerate(rights_reviews):
        prefix = f"rights_reviews[{index}]"
        _reject_unknown(
            review,
            allowed={
                "source_id",
                "license_expression",
                "terms_url",
                "terms_text_sha256",
                "use_scope",
                "reviewer_status",
                "reviewer_id",
                "reviewed_on",
                "access_authority_reference",
                "expiration_policy",
                "valid_through",
                "agent_accepted_terms",
            },
            path=prefix,
            blockers=blockers,
        )
        source_id = _string(review.get("source_id"))
        reviewed_source_ids.append(source_id)
        if source_id not in source_by_id:
            blockers.append(f"{prefix}.source_id:unknown")
        if not _string(review.get("license_expression")):
            blockers.append(f"{prefix}.license_expression:missing")
        if not _is_https_url(review.get("terms_url")):
            blockers.append(f"{prefix}.terms_url:invalid")
        if not _is_sha256(review.get("terms_text_sha256")):
            blockers.append(f"{prefix}.terms_text_sha256:invalid")
        if _string(review.get("use_scope")) not in _USE_SCOPES:
            blockers.append(f"{prefix}.use_scope:invalid")
        if _string(review.get("reviewer_status")) != "approved_for_declared_use":
            blockers.append(f"{prefix}.reviewer_status:not_approved")
        if not _is_identifier(review.get("reviewer_id")):
            blockers.append(f"{prefix}.reviewer_id:invalid")
        reviewed_on = _date(review.get("reviewed_on"))
        if reviewed_on is None:
            blockers.append(f"{prefix}.reviewed_on:invalid")
        elif reviewed_on > evaluation_date:
            blockers.append(f"{prefix}.reviewed_on:after_evaluation_date")
        if not _is_identifier(review.get("access_authority_reference")):
            blockers.append(f"{prefix}.access_authority_reference:invalid")
        expiration_policy = _string(review.get("expiration_policy"))
        valid_through = review.get("valid_through")
        if expiration_policy == "no_expiration_declared":
            if valid_through is not None:
                blockers.append(f"{prefix}.valid_through:must_be_null_without_expiration")
        elif expiration_policy == "expires_on_valid_through":
            expiration_date = _date(valid_through)
            if expiration_date is None:
                blockers.append(f"{prefix}.valid_through:invalid")
            elif admission_as_of is not None and expiration_date < admission_as_of:
                blockers.append(f"{prefix}.valid_through:expired")
        else:
            blockers.append(f"{prefix}.expiration_policy:invalid")
        if review.get("agent_accepted_terms") is not False:
            blockers.append(f"{prefix}.agent_accepted_terms:must_be_false")
    if len(reviewed_source_ids) != len(set(reviewed_source_ids)):
        blockers.append("rights_reviews:source_id_duplicate")
    for source_id in sorted(set(source_by_id) - set(reviewed_source_ids)):
        blockers.append(f"rights_reviews:missing_source:{source_id}")

    dependencies = _rows(
        value.get("code_dependencies"), path="code_dependencies", blockers=blockers
    )
    dependency_ids: list[str] = []
    dependency_bindings: list[tuple[str, str]] = []
    dependency_by_id: dict[str, dict[str, Any]] = {}
    for index, dependency in enumerate(dependencies):
        prefix = f"code_dependencies[{index}]"
        _reject_unknown(
            dependency,
            allowed={
                "dependency_id",
                "purpose",
                "capability_role",
                "upstream_project_id",
                "repository_url",
                "availability",
                "revision",
                "license",
                "smoke_status",
                "smoke_receipt_digest",
                "runtime_lock_digest",
                "dependency_license_inventory_digest",
            },
            path=prefix,
            blockers=blockers,
        )
        dependency_id = _string(dependency.get("dependency_id"))
        dependency_ids.append(dependency_id)
        if not _is_identifier(dependency_id):
            blockers.append(f"{prefix}.dependency_id:invalid")
        elif dependency_id in dependency_by_id:
            blockers.append(f"{prefix}.dependency_id:duplicate")
        else:
            dependency_by_id[dependency_id] = dependency
        if not _string(dependency.get("purpose")):
            blockers.append(f"{prefix}.purpose:missing")
        capability_role = _string(dependency.get("capability_role"))
        upstream_project_id = _string(dependency.get("upstream_project_id"))
        dependency_bindings.append((capability_role, upstream_project_id.lower()))
        if capability_role not in _CODE_CAPABILITY_ROLES:
            blockers.append(f"{prefix}.capability_role:invalid")
        if not _is_identifier(upstream_project_id):
            blockers.append(f"{prefix}.upstream_project_id:invalid")
        if not _is_https_url(dependency.get("repository_url")):
            blockers.append(f"{prefix}.repository_url:invalid")
        availability = _string(dependency.get("availability"))
        if availability not in _CODE_AVAILABILITY:
            blockers.append(f"{prefix}.availability:invalid")
        elif availability != "released":
            blockers.append(f"{prefix}.availability:not_released:{availability}")
        _validate_revision(
            dependency.get("revision"), path=f"{prefix}.revision", blockers=blockers
        )
        license_value = _mapping(dependency.get("license"))
        _reject_unknown(
            license_value,
            allowed={"license_expression", "terms_url", "text_sha256"},
            path=f"{prefix}.license",
            blockers=blockers,
        )
        if not _string(license_value.get("license_expression")):
            blockers.append(f"{prefix}.license.license_expression:missing")
        if not _is_https_url(license_value.get("terms_url")):
            blockers.append(f"{prefix}.license.terms_url:invalid")
        if not _is_sha256(license_value.get("text_sha256")):
            blockers.append(f"{prefix}.license.text_sha256:invalid")
        if _string(dependency.get("smoke_status")) != "passed":
            blockers.append(f"{prefix}.smoke_status:not_passed")
        if not _is_sha256(dependency.get("smoke_receipt_digest")):
            blockers.append(f"{prefix}.smoke_receipt_digest:invalid")
        if not _is_sha256(dependency.get("runtime_lock_digest")):
            blockers.append(f"{prefix}.runtime_lock_digest:invalid")
        if not _is_sha256(dependency.get("dependency_license_inventory_digest")):
            blockers.append(f"{prefix}.dependency_license_inventory_digest:invalid")
    if len(dependency_ids) != len(set(dependency_ids)):
        blockers.append("code_dependencies:dependency_id_duplicate")
    required_completion_bindings = {
        ("background_completion_primary_adapter", "infusion"),
        ("background_completion_quality_challenger", "aurafusion360"),
    }
    for capability_role, project_id in sorted(
        required_completion_bindings - set(dependency_bindings)
    ):
        blockers.append(
            "code_dependencies:missing_required_released_binding:"
            f"{capability_role}:{project_id}"
        )

    frames = _rows(
        value.get("coordinate_frames"), path="coordinate_frames", blockers=blockers
    )
    framed_source_ids: list[str] = []
    all_round_trips_valid = True
    for index, frame in enumerate(frames):
        prefix = f"coordinate_frames[{index}]"
        _reject_unknown(
            frame,
            allowed={
                "source_id",
                "scene_id",
                "native_units",
                "native_unit_scale_to_meters",
                "units",
                "handedness",
                "up_axis",
                "world_frame",
                "normalization_history",
                "unit_conversion_artifact_ids",
                "source_to_world",
                "world_to_source",
                "metric_scale_authority",
            },
            path=prefix,
            blockers=blockers,
        )
        source_id = _string(frame.get("source_id"))
        framed_source_ids.append(source_id)
        source = source_by_id.get(source_id)
        if source is None:
            blockers.append(f"{prefix}.source_id:unknown")
        elif _string(frame.get("scene_id")) != _string(source.get("scene_id")):
            blockers.append(f"{prefix}.scene_id:source_scene_mismatch")
        if _string(frame.get("units")) != "meters":
            blockers.append(f"{prefix}.units:must_be:meters")
        native_units = _string(frame.get("native_units"))
        native_scale = _finite_number(frame.get("native_unit_scale_to_meters"))
        expected_native_scales = {
            "meters": 1.0,
            "centimeters": 0.01,
            "millimeters": 0.001,
        }
        if native_units not in {*expected_native_scales, "unitless_normalized"}:
            blockers.append(f"{prefix}.native_units:unknown")
        elif native_scale is None or native_scale <= 0.0:
            blockers.append(f"{prefix}.native_unit_scale_to_meters:invalid")
        elif native_units in expected_native_scales and not math.isclose(
            native_scale,
            expected_native_scales[native_units],
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            blockers.append(f"{prefix}.native_unit_scale_to_meters:unit_mismatch")

        normalization = _mapping(frame.get("normalization_history"))
        _reject_unknown(
            normalization,
            allowed={"status", "reference", "inverse_transform"},
            path=f"{prefix}.normalization_history",
            blockers=blockers,
        )
        normalization_status = _string(normalization.get("status"))
        if not _string(normalization.get("reference")):
            blockers.append(f"{prefix}.normalization_history.reference:missing")
        inverse_normalization = normalization.get("inverse_transform")
        if normalization_status == "none":
            if inverse_normalization is not None:
                blockers.append(
                    f"{prefix}.normalization_history.inverse_transform:must_be_null"
                )
        elif normalization_status == "applied_and_inverted":
            inverse_matrix = _matrix4(inverse_normalization)
            if inverse_matrix is None or abs(_determinant3(inverse_matrix)) <= 1e-12:
                blockers.append(
                    f"{prefix}.normalization_history.inverse_transform:invalid"
                )
        else:
            blockers.append(f"{prefix}.normalization_history.status:invalid")
        if native_units == "unitless_normalized" and normalization_status != (
            "applied_and_inverted"
        ):
            blockers.append(
                f"{prefix}.normalization_history:required_for_unitless_normalized"
            )

        conversion_ids = _identifier_list(
            frame.get("unit_conversion_artifact_ids"),
            path=f"{prefix}.unit_conversion_artifact_ids",
            blockers=blockers,
        )
        for artifact_id in conversion_ids:
            artifact = artifact_by_id.get(artifact_id)
            if artifact is None:
                blockers.append(
                    f"{prefix}.unit_conversion_artifact_ids:unknown:{artifact_id}"
                )
            elif artifact.get("source_id") != source_id:
                blockers.append(
                    f"{prefix}.unit_conversion_artifact_ids:wrong_source:{artifact_id}"
                )
            elif artifact.get("role") != "unit_conversion_receipt":
                blockers.append(
                    f"{prefix}.unit_conversion_artifact_ids:role_mismatch:{artifact_id}"
                )
        handedness = _string(frame.get("handedness"))
        up_axis = _string(frame.get("up_axis"))
        if handedness not in {"left_handed", "right_handed"}:
            blockers.append(f"{prefix}.handedness:unknown")
        if up_axis not in {"+X", "+Y", "+Z", "-X", "-Y", "-Z"}:
            blockers.append(f"{prefix}.up_axis:unknown")
        if _string(frame.get("world_frame")) != "blueprint_world_right_handed_z_up_meters":
            blockers.append(f"{prefix}.world_frame:invalid")

        source_to_world = _matrix4(frame.get("source_to_world"))
        world_to_source = _matrix4(frame.get("world_to_source"))
        if source_to_world is None:
            blockers.append(f"{prefix}.source_to_world:invalid_matrix")
            all_round_trips_valid = False
        elif not _is_metric_isometry(source_to_world):
            blockers.append(f"{prefix}.source_to_world:not_metric_isometry")
            all_round_trips_valid = False
        else:
            determinant = _determinant3(source_to_world)
            expected_determinant_sign = -1.0 if handedness == "left_handed" else 1.0
            if handedness in {"left_handed", "right_handed"} and (
                determinant * expected_determinant_sign <= 0.0
            ):
                blockers.append(f"{prefix}.source_to_world:handedness_conversion_mismatch")
                all_round_trips_valid = False
            if up_axis in {"+X", "+Y", "+Z", "-X", "-Y", "-Z"} and not (
                _source_up_maps_to_world_up(source_to_world, up_axis)
            ):
                blockers.append(f"{prefix}.source_to_world:up_axis_mapping_mismatch")
                all_round_trips_valid = False
        if world_to_source is None:
            blockers.append(f"{prefix}.world_to_source:invalid_matrix")
            all_round_trips_valid = False
        elif not _is_metric_isometry(world_to_source):
            blockers.append(f"{prefix}.world_to_source:not_metric_isometry")
            all_round_trips_valid = False
        if source_to_world is not None and world_to_source is not None:
            if not (
                _is_identity(_matrix_product(source_to_world, world_to_source))
                and _is_identity(_matrix_product(world_to_source, source_to_world))
            ):
                blockers.append(f"{prefix}:source_world_inverse_round_trip_failed")
                all_round_trips_valid = False

        authority = _mapping(frame.get("metric_scale_authority"))
        _reject_unknown(
            authority,
            allowed={"kind", "authority_reference", "evidence_artifact_ids"},
            path=f"{prefix}.metric_scale_authority",
            blockers=blockers,
        )
        authority_kind = _string(authority.get("kind")).lower().replace("-", "_").replace(" ", "_")
        if any(marker in authority_kind for marker in _FORBIDDEN_SCALE_AUTHORITY_MARKERS):
            blockers.append(f"{prefix}.metric_scale_authority.kind:forbidden:{authority_kind}")
        elif authority_kind not in _ALLOWED_SCALE_AUTHORITIES:
            blockers.append(f"{prefix}.metric_scale_authority.kind:unsupported")
        if not _string(authority.get("authority_reference")):
            blockers.append(f"{prefix}.metric_scale_authority.authority_reference:missing")
        evidence_ids = _identifier_list(
            authority.get("evidence_artifact_ids"),
            path=f"{prefix}.metric_scale_authority.evidence_artifact_ids",
            blockers=blockers,
        )
        for artifact_id in evidence_ids:
            artifact = artifact_by_id.get(artifact_id)
            if artifact is None:
                blockers.append(
                    f"{prefix}.metric_scale_authority.evidence_artifact_ids:unknown:{artifact_id}"
                )
            elif artifact.get("source_id") != source_id:
                blockers.append(
                    f"{prefix}.metric_scale_authority.evidence_artifact_ids:wrong_source:{artifact_id}"
                )
            elif artifact.get("role") not in {
                "calibration_metadata",
                "metric_scale_evidence",
            }:
                blockers.append(
                    f"{prefix}.metric_scale_authority.evidence_artifact_ids:not_metric_evidence:{artifact_id}"
                )
    if len(framed_source_ids) != len(set(framed_source_ids)):
        blockers.append("coordinate_frames:source_id_duplicate")
    for source_id in sorted(set(source_by_id) - set(framed_source_ids)):
        blockers.append(f"coordinate_frames:missing_source:{source_id}")

    splits = _mapping(value.get("splits"))
    _reject_unknown(
        splits,
        allowed={"calibration_trajectory_ids", "test_trajectory_ids"},
        path="splits",
        blockers=blockers,
    )
    calibration_ids = _identifier_list(
        splits.get("calibration_trajectory_ids"),
        path="splits.calibration_trajectory_ids",
        blockers=blockers,
    )
    test_ids = _identifier_list(
        splits.get("test_trajectory_ids"),
        path="splits.test_trajectory_ids",
        blockers=blockers,
    )
    overlap = sorted(set(calibration_ids) & set(test_ids))
    if overlap:
        blockers.append("splits:calibration_test_overlap:" + ",".join(overlap))
    for artifact_id in calibration_ids:
        artifact = artifact_by_id.get(artifact_id)
        if artifact is None:
            blockers.append(
                f"splits.calibration_trajectory_ids:unknown:{artifact_id}"
            )
        elif artifact.get("role") != "calibration_observation":
            blockers.append(
                f"splits.calibration_trajectory_ids:role_mismatch:{artifact_id}"
            )
    for artifact_id in test_ids:
        artifact = artifact_by_id.get(artifact_id)
        if artifact is None:
            blockers.append(f"splits.test_trajectory_ids:unknown:{artifact_id}")
        elif artifact.get("role") != "test_observation":
            blockers.append(
                f"splits.test_trajectory_ids:role_mismatch:{artifact_id}"
            )

    observation_bundle = _mapping(value.get("observation_bundle"))
    _reject_unknown(
        observation_bundle,
        allowed={
            "bundle_id",
            "origin",
            "camera_model",
            "camera_model_artifact_id",
            "appearance_artifact_id",
            "materialization_dependency_id",
            "rgb_authority",
            "camera_preparation",
            "calibration_artifact_ids",
            "test_artifact_ids",
            "method_profiles",
            "validation_oracle",
            "truth_contract",
            "object_present_inputs",
            "unscaled_sfm_rerun",
            "partitions_disjoint",
            "independent_capture_evidence",
        },
        path="observation_bundle",
        blockers=blockers,
    )
    if not _is_identifier(observation_bundle.get("bundle_id")):
        blockers.append("observation_bundle.bundle_id:invalid")
    origin = _string(observation_bundle.get("origin"))
    if origin not in {"render_derived_synthetic", "source_captured"}:
        blockers.append("observation_bundle.origin:invalid")
    if _string(observation_bundle.get("camera_model")) not in {
        "COLMAP_OPENCV",
        "COLMAP_PINHOLE",
    }:
        blockers.append("observation_bundle.camera_model:invalid")
    camera_model_artifact_id = _string(
        observation_bundle.get("camera_model_artifact_id")
    )
    camera_model_artifact = artifact_by_id.get(camera_model_artifact_id)
    if camera_model_artifact is None:
        blockers.append("observation_bundle.camera_model_artifact_id:unknown")
    elif camera_model_artifact.get("role") != "camera_model_bundle":
        blockers.append("observation_bundle.camera_model_artifact_id:role_mismatch")
    observation_calibration_ids = _identifier_list(
        observation_bundle.get("calibration_artifact_ids"),
        path="observation_bundle.calibration_artifact_ids",
        blockers=blockers,
    )
    observation_test_ids = _identifier_list(
        observation_bundle.get("test_artifact_ids"),
        path="observation_bundle.test_artifact_ids",
        blockers=blockers,
    )
    if set(observation_calibration_ids) != set(calibration_ids):
        blockers.append("observation_bundle.calibration_artifact_ids:split_mismatch")
    if set(observation_test_ids) != set(test_ids):
        blockers.append("observation_bundle.test_artifact_ids:split_mismatch")
    for field, expected in (
        ("object_present_inputs", True),
        ("unscaled_sfm_rerun", False),
        ("partitions_disjoint", True),
    ):
        if observation_bundle.get(field) is not expected:
            blockers.append(
                f"observation_bundle.{field}:must_be:{str(expected).lower()}"
            )

    observation_appearance_id = observation_bundle.get("appearance_artifact_id")
    materialization_dependency_id = observation_bundle.get(
        "materialization_dependency_id"
    )
    if origin == "render_derived_synthetic":
        if _string(observation_bundle.get("rgb_authority")) != (
            "appearance_3dgs_render"
        ):
            blockers.append(
                "observation_bundle.rgb_authority:must_be:appearance_3dgs_render"
            )
        if _string(observation_bundle.get("camera_preparation")) != (
            "prebuilt_metric_colmap_without_mapper"
        ):
            blockers.append(
                "observation_bundle.camera_preparation:must_be:prebuilt_metric_colmap_without_mapper"
            )
        if observation_bundle.get("independent_capture_evidence") is not False:
            blockers.append(
                "observation_bundle.independent_capture_evidence:must_be_false"
            )
        if not _is_identifier(observation_appearance_id):
            blockers.append("observation_bundle.appearance_artifact_id:invalid")
        render_dependency = dependency_by_id.get(
            _string(materialization_dependency_id)
        )
        if render_dependency is None:
            blockers.append(
                "observation_bundle.materialization_dependency_id:unknown"
            )
        elif _string(render_dependency.get("capability_role")) != (
            "scene_materialization"
        ):
            blockers.append(
                "observation_bundle.materialization_dependency_id:not_scene_materialization"
            )
    elif origin == "source_captured":
        if _string(observation_bundle.get("rgb_authority")) != "source_rgb":
            blockers.append("observation_bundle.rgb_authority:must_be:source_rgb")
        if _string(observation_bundle.get("camera_preparation")) != (
            "source_calibration"
        ):
            blockers.append(
                "observation_bundle.camera_preparation:must_be:source_calibration"
            )
        if observation_bundle.get("independent_capture_evidence") is not True:
            blockers.append(
                "observation_bundle.independent_capture_evidence:must_be_true"
            )
        if observation_appearance_id is not None:
            blockers.append("observation_bundle.appearance_artifact_id:must_be_null")
        if materialization_dependency_id is not None:
            blockers.append(
                "observation_bundle.materialization_dependency_id:must_be_null"
            )

    method_profiles = _rows(
        observation_bundle.get("method_profiles"),
        path="observation_bundle.method_profiles",
        blockers=blockers,
    )
    method_input_ids: set[str] = set()
    method_roles: list[str] = []
    profile_ids: list[str] = []
    expected_profiles = {
        "primary_interface_adapter": (
            "InFusion",
            "background_completion_primary_adapter",
            {
                "object_present_rgb",
                "multiview_object_masks",
                "splat_rendered_inverse_depth",
                "camera_intrinsics",
                "camera_to_world",
            },
        ),
        "multiview_quality_challenger": (
            "AuraFusion360",
            "background_completion_quality_challenger",
            {
                "object_present_rgb",
                "multiview_object_masks",
                "camera_intrinsics",
                "camera_to_world",
                "source_3dgs",
            },
        ),
    }
    allowed_method_input_roles = {
        "appearance_3dgs",
        "camera_model_bundle",
        "method_input_mask",
        "method_input_rgb",
        "method_input_splat_depth",
    }
    for index, profile in enumerate(method_profiles):
        prefix = f"observation_bundle.method_profiles[{index}]"
        _reject_unknown(
            profile,
            allowed={
                "profile_id",
                "method_role",
                "dependency_id",
                "upstream_project_id",
                "input_artifact_ids",
                "input_modalities",
                "input_mount_policy",
                "writes_delta_layer",
                "preserves_source_world_frame",
                "external_validation_oracle_access",
            },
            path=prefix,
            blockers=blockers,
        )
        profile_id = _string(profile.get("profile_id"))
        profile_ids.append(profile_id)
        if not _is_identifier(profile_id):
            blockers.append(f"{prefix}.profile_id:invalid")
        method_role = _string(profile.get("method_role"))
        method_roles.append(method_role)
        expected_profile = expected_profiles.get(method_role)
        if expected_profile is None:
            blockers.append(f"{prefix}.method_role:invalid")
            expected_project = ""
            expected_capability = ""
            expected_modalities: set[str] = set()
        else:
            expected_project, expected_capability, expected_modalities = expected_profile
        project_id = _string(profile.get("upstream_project_id"))
        if expected_project and project_id != expected_project:
            blockers.append(
                f"{prefix}.upstream_project_id:must_be:{expected_project}"
            )
        dependency_id = _string(profile.get("dependency_id"))
        dependency = dependency_by_id.get(dependency_id)
        if dependency is None:
            blockers.append(f"{prefix}.dependency_id:unknown")
        elif (
            _string(dependency.get("upstream_project_id")) != project_id
            or _string(dependency.get("capability_role")) != expected_capability
        ):
            blockers.append(f"{prefix}.dependency_id:project_or_role_mismatch")
        input_ids = _identifier_list(
            profile.get("input_artifact_ids"),
            path=f"{prefix}.input_artifact_ids",
            blockers=blockers,
        )
        method_input_ids.update(input_ids)
        for artifact_id in input_ids:
            artifact = artifact_by_id.get(artifact_id)
            if artifact is None:
                blockers.append(f"{prefix}.input_artifact_ids:unknown:{artifact_id}")
            elif _string(artifact.get("role")) not in allowed_method_input_roles:
                blockers.append(
                    f"{prefix}.input_artifact_ids:forbidden_role:{artifact_id}"
                )
        modalities = profile.get("input_modalities")
        if not isinstance(modalities, list) or not modalities:
            blockers.append(f"{prefix}.input_modalities:missing_or_not_array")
            modality_set: set[str] = set()
        else:
            modality_set = {_string(item) for item in modalities}
            if len(modality_set) != len(modalities):
                blockers.append(f"{prefix}.input_modalities:duplicate")
        if expected_modalities and modality_set != expected_modalities:
            blockers.append(f"{prefix}.input_modalities:profile_mismatch")
        for field, expected in (
            ("input_mount_policy", "allowlist_only"),
            ("writes_delta_layer", True),
            ("preserves_source_world_frame", True),
            ("external_validation_oracle_access", False),
        ):
            actual = profile.get(field)
            if (isinstance(expected, bool) and actual is not expected) or (
                isinstance(expected, str) and _string(actual) != expected
            ):
                blockers.append(
                    f"{prefix}.{field}:must_be:{str(expected).lower()}"
                )
        required_input_roles = {
            "camera_model_bundle",
            "method_input_mask",
            "method_input_rgb",
        }
        if method_role == "primary_interface_adapter":
            required_input_roles.add("method_input_splat_depth")
        if method_role == "multiview_quality_challenger":
            required_input_roles.add("appearance_3dgs")
        observed_input_roles = {
            _string(artifact_by_id[artifact_id].get("role"))
            for artifact_id in input_ids
            if artifact_id in artifact_by_id
        }
        for missing_role in sorted(required_input_roles - observed_input_roles):
            blockers.append(f"{prefix}.input_artifact_ids:missing_role:{missing_role}")
    if len(profile_ids) != len(set(profile_ids)):
        blockers.append("observation_bundle.method_profiles:profile_id_duplicate")
    if set(method_roles) != set(expected_profiles) or len(method_roles) != len(
        expected_profiles
    ):
        blockers.append("observation_bundle.method_profiles:required_roles_mismatch")

    validation_oracle = _mapping(observation_bundle.get("validation_oracle"))
    _reject_unknown(
        validation_oracle,
        allowed={
            "availability",
            "depth_artifact_id",
            "geometry_artifact_id",
            "authority",
            "usage",
            "method_access",
            "independent_of_method_inputs",
        },
        path="observation_bundle.validation_oracle",
        blockers=blockers,
    )
    oracle_availability = _string(validation_oracle.get("availability"))
    oracle_depth_id = validation_oracle.get("depth_artifact_id")
    oracle_geometry_id = validation_oracle.get("geometry_artifact_id")
    oracle_ids = {
        _string(artifact_id)
        for artifact_id in (oracle_depth_id, oracle_geometry_id)
        if artifact_id is not None
    }
    if validation_oracle.get("method_access") is not False:
        blockers.append("observation_bundle.validation_oracle.method_access:must_be:false")
    if validation_oracle.get("independent_of_method_inputs") is not True:
        blockers.append(
            "observation_bundle.validation_oracle.independent_of_method_inputs:must_be:true"
        )
    if oracle_availability == "unavailable":
        if oracle_ids:
            blockers.append(
                "observation_bundle.validation_oracle:unavailable_but_artifacts_bound"
            )
        if _string(validation_oracle.get("authority")) != "none":
            blockers.append(
                "observation_bundle.validation_oracle.authority:must_be:none"
            )
        if _string(validation_oracle.get("usage")) != "not_available":
            blockers.append(
                "observation_bundle.validation_oracle.usage:must_be:not_available"
            )
    elif oracle_availability == "available":
        if not oracle_ids:
            blockers.append(
                "observation_bundle.validation_oracle:available_without_artifact"
            )
        if _string(validation_oracle.get("authority")) not in {
            "controlled_ground_truth",
            "source_sensor_or_laser",
        }:
            blockers.append("observation_bundle.validation_oracle.authority:invalid")
        if _string(validation_oracle.get("usage")) != "evaluation_only":
            blockers.append(
                "observation_bundle.validation_oracle.usage:must_be:evaluation_only"
            )
    else:
        blockers.append("observation_bundle.validation_oracle.availability:invalid")
    expected_oracle_roles = {
        _string(oracle_depth_id): "validation_depth_oracle",
        _string(oracle_geometry_id): "metric_geometry",
    }
    for artifact_id in sorted(oracle_ids):
        artifact = artifact_by_id.get(artifact_id)
        if artifact is None:
            blockers.append(
                f"observation_bundle.validation_oracle:unknown:{artifact_id}"
            )
        elif _string(artifact.get("role")) != expected_oracle_roles[artifact_id]:
            blockers.append(
                f"observation_bundle.validation_oracle:role_mismatch:{artifact_id}"
            )
    for artifact_id in sorted(oracle_ids & method_input_ids):
        blockers.append(
            "observation_bundle.validation_oracle:leaked_to_method_input:"
            f"{artifact_id}"
        )

    truth_contract = _mapping(observation_bundle.get("truth_contract"))
    _reject_unknown(
        truth_contract,
        allowed={
            "availability",
            "clean_background_artifact_ids",
            "method_access",
            "edit_result_digest",
            "edit_seal_digest",
            "truth_release_join_digest",
        },
        path="observation_bundle.truth_contract",
        blockers=blockers,
    )
    truth_ids_value = truth_contract.get("clean_background_artifact_ids")
    if not isinstance(truth_ids_value, list):
        blockers.append(
            "observation_bundle.truth_contract.clean_background_artifact_ids:must_be_array"
        )
        truth_ids: list[str] = []
    else:
        truth_ids = []
        for index, artifact_id in enumerate(truth_ids_value):
            text = _string(artifact_id)
            if not _is_identifier(text):
                blockers.append(
                    "observation_bundle.truth_contract.clean_background_artifact_ids"
                    f"[{index}]:invalid"
                )
            truth_ids.append(text)
        if len(truth_ids) != len(set(truth_ids)):
            blockers.append(
                "observation_bundle.truth_contract.clean_background_artifact_ids:duplicate"
            )
    truth_availability = _string(truth_contract.get("availability"))
    truth_digests = [
        truth_contract.get("edit_result_digest"),
        truth_contract.get("edit_seal_digest"),
        truth_contract.get("truth_release_join_digest"),
    ]
    if truth_contract.get("method_access") is not False:
        blockers.append("observation_bundle.truth_contract.method_access:must_be:false")
    if truth_availability == "unavailable":
        if truth_ids:
            blockers.append(
                "observation_bundle.truth_contract:unavailable_but_artifacts_bound"
            )
        if any(digest is not None for digest in truth_digests):
            blockers.append(
                "observation_bundle.truth_contract:unavailable_but_release_bound"
            )
    elif truth_availability == "available_withheld":
        if not truth_ids:
            blockers.append(
                "observation_bundle.truth_contract:withheld_without_artifact"
            )
        if any(digest is not None for digest in truth_digests):
            blockers.append(
                "observation_bundle.truth_contract:withheld_but_release_bound"
            )
    elif truth_availability == "released_after_edit_seal":
        if not truth_ids:
            blockers.append(
                "observation_bundle.truth_contract:released_without_artifact"
            )
        if not all(_is_sha256(digest) for digest in truth_digests):
            blockers.append(
                "observation_bundle.truth_contract:release_chain_incomplete"
            )
        elif len({_string(digest) for digest in truth_digests}) != 3:
            blockers.append(
                "observation_bundle.truth_contract:release_chain_digest_reuse"
            )
    else:
        blockers.append("observation_bundle.truth_contract.availability:invalid")
    allowed_truth_roles = {
        "clean_background_depth_truth",
        "clean_background_geometry_truth",
        "clean_background_rgb_truth",
    }
    for artifact_id in truth_ids:
        artifact = artifact_by_id.get(artifact_id)
        if artifact is None:
            blockers.append(
                "observation_bundle.truth_contract.clean_background_artifact_ids:"
                f"unknown:{artifact_id}"
            )
        elif _string(artifact.get("role")) not in allowed_truth_roles:
            blockers.append(
                "observation_bundle.truth_contract.clean_background_artifact_ids:"
                f"role_mismatch:{artifact_id}"
            )
    for artifact_id in sorted(set(truth_ids) & method_input_ids):
        blockers.append(
            "observation_bundle.truth_contract:leaked_to_method_input:"
            f"{artifact_id}"
        )

    representations = _mapping(value.get("representations"))
    _reject_unknown(
        representations,
        allowed={
            "active_pairing_id",
            "appearance",
            "metric_geometry",
            "collision",
            "task_objects",
        },
        path="representations",
        blockers=blockers,
    )
    active_pairing_id = _string(representations.get("active_pairing_id"))
    if active_pairing_id not in pairing_ids:
        blockers.append("representations.active_pairing_id:unknown")
    active_pairing = next(
        (
            pairing
            for pairing in pairings
            if _string(pairing.get("pairing_id")) == active_pairing_id
        ),
        None,
    )
    active_scene_id = (
        _string(active_pairing.get("appearance_scene_id"))
        if active_pairing is not None
        else ""
    )
    active_appearance_source_id = (
        _string(active_pairing.get("appearance_source_id"))
        if active_pairing is not None
        else ""
    )
    active_appearance_source = source_by_id.get(active_appearance_source_id)
    active_appearance_project_id = (
        _string(active_appearance_source.get("upstream_project_id"))
        if active_appearance_source is not None
        else ""
    )
    active_scene_source_ids = (
        {
            _string(active_pairing.get("appearance_source_id")),
            _string(active_pairing.get("collision_source_id")),
        }
        if active_pairing is not None
        else set()
    )
    for split_name, split_artifact_ids in (
        ("calibration", calibration_ids),
        ("test", test_ids),
    ):
        for artifact_id in split_artifact_ids:
            artifact = artifact_by_id.get(artifact_id)
            if artifact is not None and (
                artifact.get("source_id") != active_appearance_source_id
                or artifact.get("source_scene_id") != active_scene_id
            ):
                blockers.append(
                    f"splits.{split_name}_trajectory_ids:not_active_appearance_scene:{artifact_id}"
                )
    if camera_model_artifact is not None and (
        camera_model_artifact.get("source_id") != active_appearance_source_id
        or camera_model_artifact.get("source_scene_id") != active_scene_id
    ):
        blockers.append(
            "observation_bundle.camera_model_artifact_id:not_active_appearance_scene"
        )
    appearance = _mapping(representations.get("appearance"))
    _reject_unknown(
        appearance,
        allowed={
            "kind",
            "usage",
            "artifact_ids",
            "metric_measurement_authority",
            "collision_authority",
        },
        path="representations.appearance",
        blockers=blockers,
    )
    appearance_ids = _identifier_list(
        appearance.get("artifact_ids"),
        path="representations.appearance.artifact_ids",
        blockers=blockers,
    )
    if _string(appearance.get("kind")) != "3dgs":
        blockers.append("representations.appearance.kind:must_be:3dgs")
    if _string(appearance.get("usage")) != "appearance_only":
        blockers.append("representations.appearance.usage:must_be:appearance_only")
    if appearance.get("metric_measurement_authority") is not False:
        blockers.append("representations.appearance.metric_measurement_authority:must_be_false")
    if appearance.get("collision_authority") is not False:
        blockers.append("representations.appearance.collision_authority:must_be_false")
    for artifact_id in appearance_ids:
        artifact = artifact_by_id.get(artifact_id)
        if artifact is None:
            blockers.append(f"representations.appearance.artifact_ids:unknown:{artifact_id}")
        elif artifact.get("role") != "appearance_3dgs":
            blockers.append(f"representations.appearance.artifact_ids:role_mismatch:{artifact_id}")
    if active_pairing is not None and set(appearance_ids) != {
        _string(active_pairing.get("appearance_artifact_id"))
    }:
        blockers.append("representations.appearance:does_not_match_active_pairing")
    if origin == "render_derived_synthetic" and _string(
        observation_appearance_id
    ) not in set(appearance_ids):
        blockers.append(
            "observation_bundle.appearance_artifact_id:not_active_appearance"
        )

    metric_geometry = _mapping(representations.get("metric_geometry"))
    _reject_unknown(
        metric_geometry,
        allowed={"kind", "usage", "artifact_ids", "measurement_authority"},
        path="representations.metric_geometry",
        blockers=blockers,
    )
    metric_geometry_ids = _identifier_array(
        metric_geometry.get("artifact_ids"),
        path="representations.metric_geometry.artifact_ids",
        blockers=blockers,
        allow_empty=True,
    )
    metric_geometry_kind = _string(metric_geometry.get("kind"))
    metric_geometry_usage = _string(metric_geometry.get("usage"))
    metric_measurement_authority = metric_geometry.get("measurement_authority")
    if metric_geometry_kind not in {
        "publisher_metric_frame_and_boxes",
        "authored_metric_geometry",
        "calibrated_depth_surface",
        "laser_mesh",
        "rgbd_surface",
    }:
        blockers.append("representations.metric_geometry.kind:invalid")
    if metric_geometry_usage not in {
        "measurement_authority",
        "metric_frame_reference_only",
    }:
        blockers.append("representations.metric_geometry.usage:invalid")
    if not isinstance(metric_measurement_authority, bool):
        blockers.append(
            "representations.metric_geometry.measurement_authority:must_be_boolean"
        )
    if metric_geometry_kind == "publisher_metric_frame_and_boxes":
        if metric_geometry_usage != "metric_frame_reference_only":
            blockers.append(
                "representations.metric_geometry.usage:must_be:metric_frame_reference_only"
            )
        if metric_measurement_authority is not False:
            blockers.append(
                "representations.metric_geometry.measurement_authority:must_be_false"
            )
        if metric_geometry_ids:
            blockers.append(
                "representations.metric_geometry.artifact_ids:must_be_empty_for_frame_reference"
            )
    else:
        if metric_geometry_usage != "measurement_authority":
            blockers.append(
                "representations.metric_geometry.usage:must_be:measurement_authority"
            )
        if metric_measurement_authority is not True:
            blockers.append(
                "representations.metric_geometry.measurement_authority:must_be_true"
            )
        if not metric_geometry_ids:
            blockers.append(
                "representations.metric_geometry.artifact_ids:required_for_measurement_authority"
            )
    if active_appearance_project_id == "InteriorGS" and (
        metric_geometry_kind != "publisher_metric_frame_and_boxes"
        or metric_geometry_usage != "metric_frame_reference_only"
        or metric_measurement_authority is not False
    ):
        blockers.append(
            "representations.metric_geometry:interiorgs_cannot_claim_local_measurement_authority"
        )
    if active_appearance_project_id == "ScanNet++" and (
        metric_geometry_kind not in {"laser_mesh", "rgbd_surface"}
        or metric_geometry_usage != "measurement_authority"
        or metric_measurement_authority is not True
    ):
        blockers.append(
            "representations.metric_geometry:scannetpp_requires_admitted_surface_authority"
        )
    for artifact_id in metric_geometry_ids:
        artifact = artifact_by_id.get(artifact_id)
        if artifact is None:
            blockers.append(
                f"representations.metric_geometry.artifact_ids:unknown:{artifact_id}"
            )
        elif artifact.get("role") != "metric_geometry":
            blockers.append(
                f"representations.metric_geometry.artifact_ids:role_mismatch:{artifact_id}"
            )
        elif (
            artifact.get("source_id") not in active_scene_source_ids
            or artifact.get("source_scene_id") != active_scene_id
        ):
            blockers.append(
                f"representations.metric_geometry.artifact_ids:not_active_scene:{artifact_id}"
            )
    if oracle_geometry_id is not None and _string(oracle_geometry_id) not in set(
        metric_geometry_ids
    ):
        blockers.append(
            "observation_bundle.validation_oracle.geometry_artifact_id:not_active_metric_geometry"
        )

    collision = _mapping(representations.get("collision"))
    _reject_unknown(
        collision,
        allowed={"kind", "usage", "artifact_ids", "separate_from_appearance"},
        path="representations.collision",
        blockers=blockers,
    )
    collision_ids = _identifier_list(
        collision.get("artifact_ids"),
        path="representations.collision.artifact_ids",
        blockers=blockers,
    )
    if _string(collision.get("kind")) not in {"mesh", "openusd_collision"}:
        blockers.append("representations.collision.kind:invalid")
    if _string(collision.get("usage")) != "collision_only":
        blockers.append("representations.collision.usage:must_be:collision_only")
    if collision.get("separate_from_appearance") is not True:
        blockers.append("representations.collision.separate_from_appearance:must_be_true")
    if set(appearance_ids) & set(collision_ids):
        blockers.append("representations:appearance_collision_artifacts_overlap")
    for artifact_id in collision_ids:
        artifact = artifact_by_id.get(artifact_id)
        if artifact is None:
            blockers.append(f"representations.collision.artifact_ids:unknown:{artifact_id}")
        elif artifact.get("role") != "static_collision_geometry":
            blockers.append(f"representations.collision.artifact_ids:role_mismatch:{artifact_id}")
    if active_pairing is not None and set(collision_ids) != {
        _string(active_pairing.get("collision_artifact_id"))
    }:
        blockers.append("representations.collision:does_not_match_active_pairing")

    task_objects = _rows(
        representations.get("task_objects"),
        path="representations.task_objects",
        blockers=blockers,
    )
    task_object_ids: list[str] = []
    for index, task_object in enumerate(task_objects):
        prefix = f"representations.task_objects[{index}]"
        _reject_unknown(
            task_object,
            allowed={
                "task_object_id",
                "source_object_id",
                "replacement_mode",
                "asset_format",
                "asset_source_id",
                "usd_artifact_id",
                "visual_artifact_ids",
                "collision_artifact_ids",
                "physics_artifact_ids",
                "physics_properties",
                "visual_and_collision_are_separate",
                "source_object_pose_world",
                "replacement_pose_world",
                "reset_pose_world",
                "pose_authority",
                "dimensions_uncertainty_m",
                "support_contact_point_world_m",
                "support_normal_world",
                "semantic_label",
                "reset_state_id",
            },
            path=prefix,
            blockers=blockers,
        )
        task_object_id = _string(task_object.get("task_object_id"))
        task_object_ids.append(task_object_id)
        if not _is_identifier(task_object_id):
            blockers.append(f"{prefix}.task_object_id:invalid")
        if not _is_identifier(task_object.get("source_object_id")):
            blockers.append(f"{prefix}.source_object_id:invalid")
        if _string(task_object.get("replacement_mode")) != (
            "remove_source_then_insert_exact_usd"
        ):
            blockers.append(f"{prefix}.replacement_mode:invalid")
        if _string(task_object.get("asset_format")) != "simready_usd":
            blockers.append(f"{prefix}.asset_format:must_be:simready_usd")
        asset_source_id = _string(task_object.get("asset_source_id"))
        asset_source = source_by_id.get(asset_source_id)
        if asset_source is None:
            blockers.append(f"{prefix}.asset_source_id:unknown")
        elif _string(asset_source.get("source_kind")) != "simready_task_object":
            blockers.append(f"{prefix}.asset_source_id:not_simready_task_object_source")
        usd_artifact_id = _string(task_object.get("usd_artifact_id"))
        usd_artifact = artifact_by_id.get(usd_artifact_id)
        if usd_artifact is None:
            blockers.append(f"{prefix}.usd_artifact_id:unknown")
        elif (
            usd_artifact.get("source_id") != asset_source_id
            or usd_artifact.get("role") != "simready_usd_package"
        ):
            blockers.append(f"{prefix}.usd_artifact_id:source_or_role_mismatch")
        visual_ids = _identifier_list(
            task_object.get("visual_artifact_ids"),
            path=f"{prefix}.visual_artifact_ids",
            blockers=blockers,
        )
        object_collision_ids = _identifier_list(
            task_object.get("collision_artifact_ids"),
            path=f"{prefix}.collision_artifact_ids",
            blockers=blockers,
        )
        physics_ids = _identifier_list(
            task_object.get("physics_artifact_ids"),
            path=f"{prefix}.physics_artifact_ids",
            blockers=blockers,
        )
        if task_object.get("visual_and_collision_are_separate") is not True:
            blockers.append(f"{prefix}.visual_and_collision_are_separate:must_be_true")
        if (
            set(visual_ids) & set(object_collision_ids)
            or set(visual_ids) & set(physics_ids)
            or set(object_collision_ids) & set(physics_ids)
        ):
            blockers.append(f"{prefix}:visual_collision_physics_artifacts_overlap")
        for artifact_id in visual_ids:
            artifact = artifact_by_id.get(artifact_id)
            if artifact is None:
                blockers.append(f"{prefix}.visual_artifact_ids:unknown:{artifact_id}")
            elif (
                artifact.get("role") != "task_object_visual_geometry"
                or artifact.get("source_id") != asset_source_id
            ):
                blockers.append(
                    f"{prefix}.visual_artifact_ids:source_or_role_mismatch:{artifact_id}"
                )
        for artifact_id in object_collision_ids:
            artifact = artifact_by_id.get(artifact_id)
            if artifact is None:
                blockers.append(f"{prefix}.collision_artifact_ids:unknown:{artifact_id}")
            elif (
                artifact.get("role") != "task_object_collision_geometry"
                or artifact.get("source_id") != asset_source_id
            ):
                blockers.append(
                    f"{prefix}.collision_artifact_ids:source_or_role_mismatch:{artifact_id}"
                )
        for artifact_id in physics_ids:
            artifact = artifact_by_id.get(artifact_id)
            if artifact is None:
                blockers.append(f"{prefix}.physics_artifact_ids:unknown:{artifact_id}")
            elif (
                artifact.get("role") != "task_object_physics_metadata"
                or artifact.get("source_id") != asset_source_id
            ):
                blockers.append(
                    f"{prefix}.physics_artifact_ids:source_or_role_mismatch:{artifact_id}"
                )

        for field in (
            "source_object_pose_world",
            "replacement_pose_world",
            "reset_pose_world",
        ):
            pose = _matrix4(task_object.get(field))
            if pose is None or not _is_metric_isometry(pose):
                blockers.append(f"{prefix}.{field}:not_metric_isometry")
        if _string(task_object.get("pose_authority")) not in {
            "measured",
            "preregistered_candidate",
            "source_annotation",
        }:
            blockers.append(f"{prefix}.pose_authority:invalid")
        uncertainty = _numeric_vector(
            task_object.get("dimensions_uncertainty_m"), length=3
        )
        if uncertainty is None or any(value < 0.0 for value in uncertainty):
            blockers.append(f"{prefix}.dimensions_uncertainty_m:invalid")
        if _numeric_vector(
            task_object.get("support_contact_point_world_m"), length=3
        ) is None:
            blockers.append(f"{prefix}.support_contact_point_world_m:invalid")
        support_normal = _numeric_vector(
            task_object.get("support_normal_world"), length=3
        )
        if support_normal is None or not math.isclose(
            math.sqrt(sum(value * value for value in support_normal)),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-8,
        ):
            blockers.append(f"{prefix}.support_normal_world:not_unit_vector")
        if not _string(task_object.get("semantic_label")):
            blockers.append(f"{prefix}.semantic_label:missing")
        if not _is_identifier(task_object.get("reset_state_id")):
            blockers.append(f"{prefix}.reset_state_id:invalid")

        physics = _mapping(task_object.get("physics_properties"))
        _reject_unknown(
            physics,
            allowed={
                "dimensions_m",
                "mass_kg",
                "center_of_mass_m",
                "inertia_tensor_kg_m2",
                "static_friction",
                "dynamic_friction",
                "restitution",
                "authority",
                "contact_material_id",
            },
            path=f"{prefix}.physics_properties",
            blockers=blockers,
        )
        dimensions = _numeric_vector(physics.get("dimensions_m"), length=3)
        if dimensions is None or any(value <= 0.0 for value in dimensions):
            blockers.append(f"{prefix}.physics_properties.dimensions_m:invalid")
        mass = _finite_number(physics.get("mass_kg"))
        if mass is None or mass <= 0.0:
            blockers.append(f"{prefix}.physics_properties.mass_kg:invalid")
        if _numeric_vector(physics.get("center_of_mass_m"), length=3) is None:
            blockers.append(f"{prefix}.physics_properties.center_of_mass_m:invalid")
        inertia = _matrix3(physics.get("inertia_tensor_kg_m2"))
        if inertia is None:
            blockers.append(f"{prefix}.physics_properties.inertia_tensor_kg_m2:invalid")
        elif not _is_physically_valid_inertia(inertia):
            blockers.append(
                f"{prefix}.physics_properties.inertia_tensor_kg_m2:not_physically_valid"
            )
        static_friction = _finite_number(physics.get("static_friction"))
        dynamic_friction = _finite_number(physics.get("dynamic_friction"))
        if static_friction is None or static_friction < 0.0:
            blockers.append(f"{prefix}.physics_properties.static_friction:invalid")
        if dynamic_friction is None or dynamic_friction < 0.0:
            blockers.append(f"{prefix}.physics_properties.dynamic_friction:invalid")
        if (
            static_friction is not None
            and dynamic_friction is not None
            and dynamic_friction > static_friction
        ):
            blockers.append(
                f"{prefix}.physics_properties.dynamic_friction:exceeds_static_friction"
            )
        restitution = _finite_number(physics.get("restitution"))
        if restitution is None or not 0.0 <= restitution <= 1.0:
            blockers.append(f"{prefix}.physics_properties.restitution:invalid")
        if _string(physics.get("authority")) not in {
            "manufacturer_specification",
            "measured",
            "preregistered_candidate",
        }:
            blockers.append(f"{prefix}.physics_properties.authority:invalid")
        if not _is_identifier(physics.get("contact_material_id")):
            blockers.append(f"{prefix}.physics_properties.contact_material_id:invalid")
    if len(task_object_ids) != len(set(task_object_ids)):
        blockers.append("representations.task_objects:task_object_id_duplicate")

    boundaries = _mapping(value.get("claim_boundaries"))
    _reject_unknown(
        boundaries,
        allowed={"public_scene_manifest_admission", *_FORBIDDEN_TRUE_CLAIMS},
        path="claim_boundaries",
        blockers=blockers,
    )
    if boundaries.get("public_scene_manifest_admission") is not True:
        blockers.append("claim_boundaries.public_scene_manifest_admission:must_be_true")
    for claim in sorted(_FORBIDDEN_TRUE_CLAIMS):
        if boundaries.get(claim) is not False:
            blockers.append(f"claim_boundaries.{claim}:must_be_false")

    supplied_digest = _string(value.get("manifest_digest"))
    expected_digest = canonical_digest(value, digest_field="manifest_digest")
    if not supplied_digest:
        blockers.append("manifest_digest:missing")
    elif not _is_sha256(supplied_digest):
        blockers.append("manifest_digest:invalid")
    elif supplied_digest != expected_digest:
        blockers.append("manifest_digest:mismatch")

    # Keep this derived value in scope so future edits cannot accidentally
    # remove the round-trip calculation while leaving only per-matrix checks.
    if frames and not all_round_trips_valid:
        blockers.append("coordinate_frames:round_trip_not_verified")
    if artifact_count == 0:
        blockers.append("sources:artifacts_missing")
    return sorted(set(blockers))


def build_public_scene_suite_admission_receipt(
    value: Mapping[str, Any], *, evaluated_on: dt.date | str
) -> dict[str, Any]:
    """Return a deterministic ADP-009A admission receipt for *value*.

    ``admitted`` means the manifest is internally complete and ready for a
    separate materialization/byte-verification step.  It never means the
    declared artifacts were downloaded, opened, or scientifically qualified.
    """

    if not isinstance(value, Mapping):
        raise PublicSceneSuiteAdmissionError(["manifest:not_mapping"])
    evaluation_date = (
        evaluated_on if isinstance(evaluated_on, dt.date) else _date(evaluated_on)
    )
    if evaluation_date is None:
        raise PublicSceneSuiteAdmissionError(["evaluation_date:invalid"])
    normalized = _clone(value)
    blockers = _validate_manifest(normalized, evaluation_date=evaluation_date)
    manifest_digest = canonical_digest(normalized, digest_field="manifest_digest")

    sources = [dict(row) for row in normalized.get("sources", []) if isinstance(row, Mapping)]
    dependencies = [
        dict(row)
        for row in normalized.get("code_dependencies", [])
        if isinstance(row, Mapping)
    ]
    splits = _mapping(normalized.get("splits"))
    observation_bundle = _mapping(normalized.get("observation_bundle"))
    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "program_id": PROGRAM_ID,
        "adp_item": ADP_ITEM,
        "gate_id": "public_scene_suite_admission",
        "phase_label": PHASE_LABEL,
        "suite_id": _string(normalized.get("suite_id")) or None,
        "manifest_digest": manifest_digest,
        "supplied_manifest_digest": _string(normalized.get("manifest_digest")) or None,
        "status": "blocked" if blockers else "component_admitted",
        "blockers": blockers,
        "evaluated_on": evaluation_date.isoformat(),
        "qualification_role": "component_admission_only",
        "adp009a_matrix_complete": False,
        "claim_ceiling": CLAIM_CEILING,
        "source_bindings": [
            {
                "source_id": _string(source.get("source_id")) or None,
                "upstream_project_id": _string(source.get("upstream_project_id"))
                or None,
                "scene_id": _string(source.get("scene_id")) or None,
                "revision": _mapping(source.get("revision")),
                "artifact_count": len(
                    [row for row in source.get("artifacts", []) if isinstance(row, Mapping)]
                ),
            }
            for source in sources
        ],
        "code_dependency_ids": sorted(
            _string(row.get("dependency_id"))
            for row in dependencies
            if _string(row.get("dependency_id"))
        ),
        "observation_bundle_id": _string(observation_bundle.get("bundle_id")) or None,
        "observation_origin": _string(observation_bundle.get("origin")) or None,
        "independent_capture_evidence": (
            observation_bundle.get("independent_capture_evidence") is True
        ),
        "calibration_trajectory_ids": sorted(
            _string(item)
            for item in splits.get("calibration_trajectory_ids", [])
            if _string(item)
        ),
        "test_trajectory_ids": sorted(
            _string(item)
            for item in splits.get("test_trajectory_ids", [])
            if _string(item)
        ),
        "manifest_ready_for_materialization": not blockers,
        "artifact_bytes_opened": False,
        "artifact_bytes_verified": False,
        "rights_terms_bytes_opened_by_builder": False,
        "public_scene_software_qualified": False,
        "metric_geometry_qualified": False,
        "task_physics_qualified": False,
        "partner_capture_qualified": False,
        "prospective_validation": False,
        "physical_evidence_created": False,
        "existing_adp_008_artifacts_modified": False,
    }
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    return receipt


__all__ = [
    "ADP_ITEM",
    "CLAIM_CEILING",
    "MANIFEST_SCHEMA_VERSION",
    "PHASE_LABEL",
    "PublicSceneSuiteAdmissionError",
    "build_public_scene_suite_admission_receipt",
]
