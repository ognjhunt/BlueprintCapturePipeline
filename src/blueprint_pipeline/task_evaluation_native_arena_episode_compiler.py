"""Closed production compiler for configured-scene episode evaluations.

Robot teams submit independent, immutable scene, robot, controller, sensor, and
runtime references through the Website contract.  This module joins those
verified bytes into the legacy native-Arena packet only inside Blueprint's
production preparation service.  It never allocates a provider and it never
accepts a customer-built Arena packet.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import stat
import zipfile
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any, Callable

from .common import write_json
from .decision_evidence_contracts import canonical_digest
from .gaussian_field_quality import (
    gaussian_quality_is_qualified,
    measure_gaussian_field_quality,
)
from .native_task_arena_packet import (
    materialize_native_task_arena_appearance_variant_request,
    materialize_native_task_arena_packet,
)
from .native_task_wrist_camera_mount_sweep import (
    OVERVIEW_RENDER_RESOLUTION,
    POLICY_RENDER_RESOLUTION,
    materialize_wrist_camera_mount_sweep_request,
)
from .particlefield_usd import write_particlefield_usd_from_nurec
from .particlefield_runtime_asset_cache import materialize_cached_particlefield
from .task_evaluation_configured_scene_revision import (
    validate_configured_scene_revision,
)
from .task_evaluation_native_arena_preparation_adapter import (
    RUNTIME_SOURCE_LAYER_CONTRACT_PREFIX,
    build_task_evaluation_adapter_bundle,
    materialize_native_arena_adapter,
)
from .task_evaluation_rigid_relocation_native_adapter import (
    adapt_rigid_relocation_task_template,
)
from .task_evaluation_native_construction_feedback_controller import (
    validate_native_construction_inventory,
)


OUTPUT_SCHEMA_VERSION = "task_evaluation_episode_compiler_output.v1"
MAXIMUM_INLINE_NUREC_CONVERSION_BYTES = 32 * 1024 * 1024
DESTINATION_ASSET_CONTRACT_PATH = "task.destination.asset"
DESTINATION_RIGHTS_CONTRACT_PATH = "task.destination.rights_admission"
DESTINATION_STATIC_CONTRACT_PATH = "task.destination.static_qualification"
DESTINATION_NATIVE_CONTRACT_PATH = "task.destination.native_import_qualification"
DESTINATION_GEOMETRY_CONTRACT_PATH = "task.destination.geometry"
DESTINATION_PLACEMENT_CONTRACT_PATH = "task.destination.placement_qualification"


class TaskEvaluationNativeArenaEpisodeCompilerError(RuntimeError):
    """Verified robot-team inputs could not be compiled fail-closed."""


NativeAppearanceMaterializer = Callable[..., Mapping[str, Any]]


def _appearance_render_backend(
    *,
    kind: str,
    source_digest: str,
    particlefield_digest: str,
    authoring_receipt_digest: str | None,
    upstream_converter: Mapping[str, Any] | None = None,
    projection_mode_hint: str | None = None,
    sorting_mode_hint: str | None = None,
    color_space: str | None = None,
) -> dict[str, Any]:
    backend: dict[str, Any] = {
        "schema_version": "task_evaluation_appearance_render_backend.v1",
        "kind": str(kind),
        "source_configured_appearance_digest": source_digest,
        "particlefield_digest": particlefield_digest,
        "authoring_receipt_digest": authoring_receipt_digest,
        "upstream_converter": (
            dict(upstream_converter) if upstream_converter is not None else None
        ),
        "projection_mode_hint": projection_mode_hint,
        "sorting_mode_hint": sorting_mode_hint,
        "color_space": color_space,
        "backend_digest": "",
    }
    backend["backend_digest"] = canonical_digest(
        backend, digest_field="backend_digest"
    )
    return backend


def _runtime_subject_task_spec(value: Mapping[str, Any]) -> dict[str, Any]:
    """Derive a native-safe asset ID while retaining product identity."""

    task_spec = json.loads(json.dumps(dict(value)))
    source_subject_id = str(task_spec.get("subject_asset_id") or "")
    runtime_subject_id = re.sub(r"[^A-Za-z0-9_]", "_", source_subject_id)
    if not runtime_subject_id or not runtime_subject_id.replace("_", "a").isalnum():
        raise TaskEvaluationNativeArenaEpisodeCompilerError(
            "episode_compiler_subject_runtime_id_invalid"
        )
    interaction_affordance = task_spec.get("interaction_affordance")
    if (
        not isinstance(interaction_affordance, Mapping)
        or interaction_affordance.get("subject_asset_id") != source_subject_id
    ):
        raise TaskEvaluationNativeArenaEpisodeCompilerError(
            "episode_compiler_interaction_affordance_invalid"
        )
    task_spec["subject_asset_id"] = runtime_subject_id
    task_spec["source_subject_identity"] = source_subject_id
    interaction_affordance = dict(interaction_affordance)
    interaction_affordance["subject_asset_id"] = runtime_subject_id
    interaction_affordance["affordance_digest"] = canonical_digest(
        interaction_affordance, digest_field="affordance_digest"
    )
    task_spec["interaction_affordance"] = interaction_affordance
    return task_spec


def _sha256_and_size(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return "sha256:" + digest.hexdigest(), size


def _reference_path(references: Mapping[str, Mapping[str, Any]], contract_path: str) -> Path:
    row = references.get(contract_path)
    unresolved_path = Path(str((row or {}).get("materialized_path") or "")).expanduser()
    path = unresolved_path.resolve()
    if (
        row is None
        or unresolved_path.is_symlink()
        or not path.is_file()
        or _sha256_and_size(path) != (row.get("digest"), row.get("size_bytes"))
        or row.get("full_byte_service_account_readback_passed") is not True
    ):
        raise TaskEvaluationNativeArenaEpisodeCompilerError(
            f"episode_compiler_reference_invalid:{contract_path}"
        )
    return path


def _link_policy_observation_asset(*, source: Path, output_root: Path) -> Path:
    """Put one verified immutable appearance inside the compiler evidence root."""

    destination_root = output_root / "policy-observation-appearance"
    destination_root.mkdir(mode=0o750)
    destination = destination_root / "scene_appearance.usdc"
    if destination.exists() or destination.is_symlink():
        raise TaskEvaluationNativeArenaEpisodeCompilerError(
            "episode_compiler_policy_observation_asset_output_conflict"
        )
    try:
        os.link(source, destination, follow_symlinks=False)
    except OSError:
        shutil.copyfile(source, destination)
        destination.chmod(0o440)
    if _sha256_and_size(destination) != _sha256_and_size(source):
        raise TaskEvaluationNativeArenaEpisodeCompilerError(
            "episode_compiler_policy_observation_asset_copy_mismatch"
        )
    return destination


def _json_reference(
    references: Mapping[str, Mapping[str, Any]],
    contract_path: str,
    schema_version: str,
) -> dict[str, Any]:
    path = _reference_path(references, contract_path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TaskEvaluationNativeArenaEpisodeCompilerError(
            f"episode_compiler_reference_json_invalid:{contract_path}"
        ) from exc
    if not isinstance(value, Mapping) or value.get("schema_version") != schema_version:
        raise TaskEvaluationNativeArenaEpisodeCompilerError(
            f"episode_compiler_reference_contract_invalid:{contract_path}"
        )
    return dict(value)


def _safe_member(value: Any) -> PurePosixPath:
    member = PurePosixPath(str(value or ""))
    if member.is_absolute() or member.as_posix() in {"", ".", ".."} or ".." in member.parts:
        raise TaskEvaluationNativeArenaEpisodeCompilerError(
            "episode_compiler_configured_bundle_member_invalid"
        )
    return member


def _extract_configured_assets(bundle_path: Path, *, output_root: Path) -> dict[str, Path]:
    destination = output_root / "configured-scene"
    destination.mkdir(mode=0o750)
    try:
        archive = zipfile.ZipFile(bundle_path)
    except (OSError, zipfile.BadZipFile) as exc:
        raise TaskEvaluationNativeArenaEpisodeCompilerError(
            "episode_compiler_configured_bundle_invalid"
        ) from exc
    with archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        manifest_name = "configured_scene_bundle_candidate.v1.json"
        if len(names) != len(set(names)) or manifest_name not in names:
            raise TaskEvaluationNativeArenaEpisodeCompilerError(
                "episode_compiler_configured_bundle_invalid"
            )
        for info in infos:
            mode = info.external_attr >> 16
            _safe_member(info.filename)
            if info.is_dir() or info.flag_bits & 0x1 or stat.S_ISLNK(mode):
                raise TaskEvaluationNativeArenaEpisodeCompilerError(
                    "episode_compiler_configured_bundle_member_invalid"
                )
        try:
            manifest = json.loads(archive.read(manifest_name).decode("utf-8"))
        except (KeyError, UnicodeError, json.JSONDecodeError) as exc:
            raise TaskEvaluationNativeArenaEpisodeCompilerError(
                "episode_compiler_configured_bundle_manifest_invalid"
            ) from exc
        if (
            not isinstance(manifest, Mapping)
            or manifest.get("schema_version")
            != "task_evaluation_configured_scene_bundle_candidate.v1"
            or manifest.get("status") != "assembled_pending_control_plane_publication"
            or manifest.get("robot_neutral") is not True
            or manifest.get("robot_specific_base_registration_included") is not False
            or manifest.get("manifest_digest")
            != canonical_digest(manifest, digest_field="manifest_digest")
        ):
            raise TaskEvaluationNativeArenaEpisodeCompilerError(
                "episode_compiler_configured_bundle_manifest_invalid"
            )
        rows = manifest.get("assets")
        if not isinstance(rows, list) or {
            row.get("role") for row in rows if isinstance(row, Mapping)
        } != {
            "appearance",
            "collision",
            "replacement",
        }:
            raise TaskEvaluationNativeArenaEpisodeCompilerError(
                "episode_compiler_configured_bundle_asset_set_invalid"
            )
        extracted: dict[str, Path] = {}
        expected_names = {manifest_name}
        for row in rows:
            if not isinstance(row, Mapping):
                raise TaskEvaluationNativeArenaEpisodeCompilerError(
                    "episode_compiler_configured_bundle_asset_set_invalid"
                )
            role = str(row["role"])
            member = _safe_member(row["relative_path"])
            expected_names.add(member.as_posix())
            target = destination / member.name
            if target.exists():
                raise TaskEvaluationNativeArenaEpisodeCompilerError(
                    "episode_compiler_configured_bundle_asset_set_invalid"
                )
            try:
                payload = archive.read(member.as_posix())
            except KeyError as exc:
                raise TaskEvaluationNativeArenaEpisodeCompilerError(
                    "episode_compiler_configured_bundle_asset_missing"
                ) from exc
            target.write_bytes(payload)
            if _sha256_and_size(target) != (row.get("digest"), row.get("size_bytes")):
                raise TaskEvaluationNativeArenaEpisodeCompilerError(
                    "episode_compiler_configured_bundle_asset_identity_mismatch"
                )
            target.chmod(0o440)
            extracted[role] = target
        if set(names) != expected_names:
            raise TaskEvaluationNativeArenaEpisodeCompilerError(
                "episode_compiler_configured_bundle_member_set_invalid"
            )
    return extracted


def _identity(value: Mapping[str, Any], expected: Mapping[str, Any], *, label: str) -> None:
    if value.get("identity") != expected:
        raise TaskEvaluationNativeArenaEpisodeCompilerError(
            f"episode_compiler_identity_mismatch:{label}"
        )


def _runtime_asset_id(value: Any, *, label: str) -> str:
    source = str(value or "")
    runtime = re.sub(r"[^A-Za-z0-9_]", "_", source)
    if not runtime or not runtime.replace("_", "a").isalnum():
        raise TaskEvaluationNativeArenaEpisodeCompilerError(
            f"episode_compiler_runtime_asset_id_invalid:{label}"
        )
    return runtime


def _numeric_bounds(value: Any) -> tuple[list[float], list[float]]:
    try:
        lower = [float(item) for item in value["minimum"]]
        upper = [float(item) for item in value["maximum"]]
    except (KeyError, TypeError, ValueError) as exc:
        raise TaskEvaluationNativeArenaEpisodeCompilerError(
            "episode_compiler_destination_geometry_invalid"
        ) from exc
    if (
        len(lower) != 3
        or len(upper) != 3
        or any(low >= high for low, high in zip(lower, upper, strict=True))
        or not all(math.isfinite(item) for item in (*lower, *upper))
    ):
        raise TaskEvaluationNativeArenaEpisodeCompilerError(
            "episode_compiler_destination_geometry_invalid"
        )
    return lower, upper


def _rotate_xyzw(vector: list[float], quaternion: list[float]) -> list[float]:
    x, y, z, w = quaternion
    vx, vy, vz = vector
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return [
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx),
    ]


def _multiply_xyzw(left: list[float], right: list[float]) -> list[float]:
    lx, ly, lz, lw = left
    rx, ry, rz, rw = right
    result = [
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
        lw * rw - lx * rx - ly * ry - lz * rz,
    ]
    norm = math.sqrt(sum(value * value for value in result))
    return [value / norm for value in result]


def _subject_bounds_in_scoring_frame(
    *, bounds: Any, transform: Any
) -> tuple[list[float], list[float]]:
    lower, upper = _numeric_bounds(bounds)
    try:
        offset = [float(item) for item in transform["position_m"]]
        orientation = [float(item) for item in transform["orientation_xyzw"]]
    except (KeyError, TypeError, ValueError) as exc:
        raise TaskEvaluationNativeArenaEpisodeCompilerError(
            "episode_compiler_destination_subject_geometry_invalid"
        ) from exc
    if (
        len(offset) != 3
        or len(orientation) != 4
        or not math.isclose(
            sum(item * item for item in orientation),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-6,
        )
    ):
        raise TaskEvaluationNativeArenaEpisodeCompilerError(
            "episode_compiler_destination_subject_geometry_invalid"
        )
    inverse = [-orientation[0], -orientation[1], -orientation[2], orientation[3]]
    corners = [
        _rotate_xyzw(
            [x - offset[0], y - offset[1], z - offset[2]], inverse
        )
        for x in (lower[0], upper[0])
        for y in (lower[1], upper[1])
        for z in (lower[2], upper[2])
    ]
    return (
        [min(point[axis] for point in corners) for axis in range(3)],
        [max(point[axis] for point in corners) for axis in range(3)],
    )


def _stage_destination_asset(
    *,
    request: Mapping[str, Any],
    materialized_references: Mapping[str, Mapping[str, Any]],
    output_root: Path,
    configured_collision_path: Path,
    task_spec: Mapping[str, Any],
) -> dict[str, Any] | None:
    task = request.get("task")
    destination = task.get("destination") if isinstance(task, Mapping) else None
    if destination is None:
        return None
    if (
        not isinstance(destination, Mapping)
        or task.get("strategy") != "pick_and_place"
    ):
        raise TaskEvaluationNativeArenaEpisodeCompilerError(
            "episode_compiler_destination_contract_invalid"
        )
    identity = destination.get("identity")
    source_id = str(identity.get("id") or "") if isinstance(identity, Mapping) else ""
    runtime_id = _runtime_asset_id(source_id, label="destination")
    asset_source = _reference_path(
        materialized_references, DESTINATION_ASSET_CONTRACT_PATH
    )
    rights = _json_reference(
        materialized_references,
        DESTINATION_RIGHTS_CONTRACT_PATH,
        "task_evaluation_rigid_destination_rights_admission.v1",
    )
    static = _json_reference(
        materialized_references,
        DESTINATION_STATIC_CONTRACT_PATH,
        "task_evaluation_rigid_replacement_static_qualification.v1",
    )
    native = _json_reference(
        materialized_references,
        DESTINATION_NATIVE_CONTRACT_PATH,
        "task_evaluation_replacement_native_import_result.v1",
    )
    geometry = _json_reference(
        materialized_references,
        DESTINATION_GEOMETRY_CONTRACT_PATH,
        "task_evaluation_rigid_destination_geometry.v1",
    )
    placement = _json_reference(
        materialized_references,
        DESTINATION_PLACEMENT_CONTRACT_PATH,
        "task_evaluation_rigid_destination_placement_qualification.v1",
    )
    subject_static = _json_reference(
        materialized_references,
        "scene.configured_revision.replacement.static_qualification",
        "task_evaluation_rigid_replacement_static_qualification.v1",
    )
    asset_digest, asset_size = _sha256_and_size(asset_source)
    static_digest = _sha256_and_size(
        _reference_path(materialized_references, DESTINATION_STATIC_CONTRACT_PATH)
    )[0]
    native_digest = _sha256_and_size(
        _reference_path(materialized_references, DESTINATION_NATIVE_CONTRACT_PATH)
    )[0]
    subject_static_digest = _sha256_and_size(
        _reference_path(
            materialized_references,
            "scene.configured_revision.replacement.static_qualification",
        )
    )[0]
    configured_collision_digest = _sha256_and_size(configured_collision_path)[0]
    rigid_paths = (static.get("observed_structure") or {}).get("rigid_body_paths")
    intended_paths = geometry.get("intended_support_prim_paths")
    pose = destination.get("pose_world")
    subject_bounds = geometry.get("subject_collision_bounds_scoring_frame_m")
    interior_bounds = geometry.get("destination_interior_bounds_body_frame_m")
    bounds = geometry.get("destination_position_bounds_destination_frame_m")
    support_interval = geometry.get("support_height_interval_m")
    try:
        orientation = [float(value) for value in pose["orientation_xyzw"]]
        position = [float(value) for value in pose["position_world_m"]]
        subject_lower, subject_upper = _numeric_bounds(subject_bounds)
        interior_lower, interior_upper = _numeric_bounds(interior_bounds)
        lower, upper = _numeric_bounds(bounds)
        support = [float(value) for value in support_interval]
        withdrawal = [
            float(value)
            for value in geometry["insertion_withdrawal_unit_destination_frame"]
        ]
        subject_destination_orientation = [
            float(value)
            for value in geometry[
                "subject_orientation_destination_frame_xyzw"
            ]
        ]
    except (KeyError, TypeError, ValueError) as exc:
        raise TaskEvaluationNativeArenaEpisodeCompilerError(
            "episode_compiler_destination_geometry_invalid"
        ) from exc
    if (
        destination.get("schema_version")
        != "task_evaluation_rigid_destination_asset.v1"
        or destination.get("identity") is None
        or destination.get("relation") not in {"inside", "on"}
        or not str(destination.get("visible_label") or "").strip()
        or destination.get("provider_disclosure_allowed") is not True
        or rights.get("status") != "admitted"
        or rights.get("destination_identity") != identity
        or rights.get("private_provider_processing_allowed") is not True
        or rights.get("rights_admission_digest")
        != canonical_digest(rights, digest_field="rights_admission_digest")
        or static.get("status") != "authored_structure_statically_qualified"
        or static.get("replacement_identity") != identity
        or (static.get("replacement_usd") or {}).get("sha256") != asset_digest
        or (static.get("replacement_usd") or {}).get("size_bytes") != asset_size
        or static.get("result_digest")
        != canonical_digest(static, digest_field="result_digest")
        or not isinstance(rigid_paths, list)
        or not rigid_paths
        or native.get("status") != "qualified"
        or native.get("replacement_identity") != identity
        or native.get("native_simulator_import_qualified") is not True
        or native.get("native_isaac_executed") is not True
        or native.get("asset_digest") != asset_digest
        or native.get("static_qualification_digest") != static_digest
        or native.get("support_contact_observed") is not True
        or native.get("deterministic_reset_state_digest_repeat_count", 0) < 3
        or native.get("blockers") not in ([], ())
        or native.get("result_digest")
        != canonical_digest(native, digest_field="result_digest")
        or geometry.get("status") != "qualified"
        or geometry.get("subject_identity") != task.get("subject", {}).get("identity")
        or geometry.get("destination_identity") != identity
        or geometry.get("relation") != destination.get("relation")
        or geometry.get("pose_world") != pose
        or geometry.get("subject_static_qualification_digest")
        != subject_static_digest
        or geometry.get("destination_static_qualification_digest") != static_digest
        or geometry.get("whole_subject_containment_encoded_by_shrunk_bounds")
        is not True
        or geometry.get("geometry_digest")
        != canonical_digest(geometry, digest_field="geometry_digest")
        or not isinstance(intended_paths, list)
        or not intended_paths
        or any(path not in rigid_paths for path in intended_paths)
        or len(position) != 3
        or len(orientation) != 4
        or len(lower) != 3
        or len(upper) != 3
        or len(support) != 2
        or len(withdrawal) != 3
        or len(subject_destination_orientation) != 4
        or any(low >= high for low, high in zip(lower, upper, strict=True))
        or support[0] >= support[1]
        or not all(
            math.isfinite(value)
            for value in (
                *position,
                *orientation,
                *subject_lower,
                *subject_upper,
                *interior_lower,
                *interior_upper,
                *lower,
                *upper,
                *support,
                *withdrawal,
                *subject_destination_orientation,
            )
        )
        or not math.isclose(
            sum(value * value for value in orientation),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-6,
        )
        or not math.isclose(
            sum(value * value for value in withdrawal),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-6,
        )
        or not math.isclose(
            sum(value * value for value in subject_destination_orientation),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-6,
        )
    ):
        raise TaskEvaluationNativeArenaEpisodeCompilerError(
            "episode_compiler_destination_geometry_invalid"
        )
    computed_subject_lower, computed_subject_upper = _subject_bounds_in_scoring_frame(
        bounds=(subject_static.get("observed_structure") or {}).get(
            "collision_bounds_body_frame_m"
        ),
        transform=(task_spec.get("interaction_affordance") or {}).get(
            "asset_root_from_scoring_frame"
        ),
    )
    oriented_subject_corners = [
        _rotate_xyzw([x, y, z], subject_destination_orientation)
        for x in (computed_subject_lower[0], computed_subject_upper[0])
        for y in (computed_subject_lower[1], computed_subject_upper[1])
        for z in (computed_subject_lower[2], computed_subject_upper[2])
    ]
    oriented_subject_lower = [
        min(point[axis] for point in oriented_subject_corners) for axis in range(3)
    ]
    oriented_subject_upper = [
        max(point[axis] for point in oriented_subject_corners) for axis in range(3)
    ]
    expected_lower = [
        interior_lower[axis] - oriented_subject_lower[axis] for axis in range(3)
    ]
    expected_upper = [
        interior_upper[axis] - oriented_subject_upper[axis] for axis in range(3)
    ]
    reset = placement.get("repeated_reset_readback") or {}
    try:
        reset_repeat_count = int(reset["repeat_count"])
        reset_translation_error = float(reset["maximum_translation_error_m"])
        reset_rotation_error = float(reset["maximum_rotation_error_rad"])
        reset_translation_tolerance = float(reset["translation_tolerance_m"])
        reset_rotation_tolerance = float(reset["rotation_tolerance_rad"])
    except (KeyError, TypeError, ValueError) as exc:
        raise TaskEvaluationNativeArenaEpisodeCompilerError(
            "episode_compiler_destination_placement_invalid"
        ) from exc
    if (
        any(
            not math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-9)
            for observed, expected in zip(
                (*subject_lower, *subject_upper),
                (*computed_subject_lower, *computed_subject_upper),
                strict=True,
            )
        )
        or any(
            not math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-9)
            for observed, expected in zip(
                (*lower, *upper), (*expected_lower, *expected_upper), strict=True
            )
        )
        or placement.get("status") != "qualified"
        or placement.get("destination_identity") != identity
        or placement.get("configured_scene_revision_digest")
        != task.get("configured_scene_revision_digest")
        or placement.get("configured_scene_collision_digest")
        != configured_collision_digest
        or placement.get("destination_asset_digest") != asset_digest
        or placement.get("destination_static_qualification_digest") != static_digest
        or placement.get("destination_native_import_qualification_digest")
        != native_digest
        or placement.get("destination_geometry_digest")
        != geometry.get("geometry_digest")
        or placement.get("pose_world") != pose
        or placement.get("nonpenetration_passed") is not True
        or placement.get("support_stability_passed") is not True
        or placement.get("camera_visibility")
        != {"external": True, "wrist": True, "overview": True}
        or isinstance(reset.get("repeat_count"), bool)
        or reset_repeat_count < 3
        or not all(
            math.isfinite(value)
            for value in (
                reset_translation_error,
                reset_rotation_error,
                reset_translation_tolerance,
                reset_rotation_tolerance,
            )
        )
        or min(reset_translation_tolerance, reset_rotation_tolerance) <= 0.0
        or min(reset_translation_error, reset_rotation_error) < 0.0
        or reset_translation_error > reset_translation_tolerance
        or reset_rotation_error > reset_rotation_tolerance
        or placement.get("placement_qualification_digest")
        != canonical_digest(
            placement, digest_field="placement_qualification_digest"
        )
    ):
        raise TaskEvaluationNativeArenaEpisodeCompilerError(
            "episode_compiler_destination_placement_invalid"
        )
    destination_root = output_root / "task-destination"
    destination_root.mkdir(mode=0o750)
    suffix = asset_source.suffix.lower()
    target = destination_root / f"task_support{suffix}"
    if target.exists() or target.is_symlink():
        raise TaskEvaluationNativeArenaEpisodeCompilerError(
            "episode_compiler_destination_asset_output_conflict"
        )
    try:
        os.link(asset_source, target, follow_symlinks=False)
    except OSError:
        shutil.copyfile(asset_source, target)
    target.chmod(0o440)
    if _sha256_and_size(target) != _sha256_and_size(asset_source):
        raise TaskEvaluationNativeArenaEpisodeCompilerError(
            "episode_compiler_destination_asset_copy_mismatch"
        )
    target_local = [
        (low + high) / 2.0 for low, high in zip(lower, upper, strict=True)
    ]
    target_offset_world = _rotate_xyzw(target_local, orientation)
    return {
        "asset_id": runtime_id,
        "source_asset_id": source_id,
        "path": target,
        "pose_world": pose,
        "relation": destination["relation"],
        "visible_label": str(destination["visible_label"]).strip(),
        "destination_position_bounds_destination_frame_m": {
            "minimum": lower,
            "maximum": upper,
        },
        "target_position_world_m": [
            position[axis] + target_offset_world[axis] for axis in range(3)
        ],
        "destination_pose_world": [*position, *orientation],
        "destination_orientation_world_xyzw": _multiply_xyzw(
            orientation, subject_destination_orientation
        ),
        "destination_reset_translation_tolerance_m": float(
            reset_translation_tolerance
        ),
        "destination_reset_rotation_tolerance_rad": float(
            reset_rotation_tolerance
        ),
        "support_height_interval_m": support,
        "intended_support_prim_paths": list(intended_paths),
        "insertion_withdrawal_unit_world": _rotate_xyzw(
            withdrawal, orientation
        ),
        "rights_digest": rights["rights_admission_digest"],
        "geometry_digest": geometry["geometry_digest"],
    }


def _materialize_native_particlefield_appearance(
    *, source_path: Path, output_root: Path
) -> dict[str, Any]:
    """Make Isaac's proven ParticleField representation the episode default."""

    try:
        from pxr import Usd
    except Exception as exc:  # noqa: BLE001
        raise TaskEvaluationNativeArenaEpisodeCompilerError(
            "episode_compiler_appearance_usd_runtime_unavailable"
        ) from exc
    try:
        stage = Usd.Stage.Open(str(source_path))
    except Exception as exc:  # noqa: BLE001
        raise TaskEvaluationNativeArenaEpisodeCompilerError(
            "episode_compiler_configured_appearance_invalid"
        ) from exc
    if stage is None or not stage.GetDefaultPrim():
        raise TaskEvaluationNativeArenaEpisodeCompilerError(
            "episode_compiler_configured_appearance_invalid"
        )
    particlefields = [
        prim
        for prim in stage.Traverse()
        if str(prim.GetTypeName()) == "ParticleField3DGaussianSplat"
    ]
    nurec_volumes = [
        prim
        for prim in stage.Traverse()
        if bool(prim.GetAttribute("omni:nurec:isNuRecVolume").Get())
    ]
    source_digest, source_size = _sha256_and_size(source_path)
    if len(particlefields) == 1 and not nurec_volumes:
        field = particlefields[0]
        try:
            field_quality = measure_gaussian_field_quality(
                positions=field.GetAttribute("positions").Get(),
                activated_scales=field.GetAttribute("scales").Get(),
                opacities=field.GetAttribute("opacities").Get(),
            )
        except ValueError as exc:
            raise TaskEvaluationNativeArenaEpisodeCompilerError(
                "episode_compiler_configured_appearance_gaussian_field_quality_invalid"
            ) from exc
        if not gaussian_quality_is_qualified(field_quality):
            raise TaskEvaluationNativeArenaEpisodeCompilerError(
                "episode_compiler_configured_appearance_gaussian_field_quality_invalid"
            )
        return {
            "status": "existing_particlefield_selected",
            "path": str(source_path),
            "representation": "particlefield_3d_gaussian_splat",
            "source_configured_appearance_digest": source_digest,
            "source_configured_appearance_size_bytes": source_size,
            "representation_conversion_performed": False,
            "exact_learned_arrays_preserved": True,
            "gaussian_field_quality": field_quality,
            "appearance_render_backend": _appearance_render_backend(
                kind="preauthored_particlefield",
                source_digest=source_digest,
                particlefield_digest=source_digest,
                authoring_receipt_digest=None,
            ),
        }
    if len(nurec_volumes) != 1 or particlefields:
        raise TaskEvaluationNativeArenaEpisodeCompilerError(
            "episode_compiler_configured_appearance_representation_unsupported"
        )
    try:
        cached = materialize_cached_particlefield(
            source_digest=source_digest,
            output_root=output_root,
        )
    except ValueError as exc:
        raise TaskEvaluationNativeArenaEpisodeCompilerError(
            f"episode_compiler_official_particlefield_cache_invalid:{exc}"
        ) from exc
    if cached is not None:
        receipt = cached["authoring_receipt"]
        return {
            "status": "nurec_reused_cached_official_particlefield",
            "path": cached["asset_path"],
            "representation": "particlefield_3d_gaussian_splat",
            "source_configured_appearance_digest": source_digest,
            "source_configured_appearance_size_bytes": source_size,
            "particlefield_digest": receipt["output_sha256"],
            "particlefield_size_bytes": receipt["output_bytes"],
            "particlefield_authoring_receipt_path": cached[
                "authoring_receipt_path"
            ],
            "particlefield_authoring_receipt_digest": receipt["receipt_digest"],
            "particlefield_runtime_cache_manifest_digest": cached[
                "cache_manifest_digest"
            ],
            "representation_conversion_performed": True,
            "exact_learned_arrays_preserved": True,
            "gaussian_field_quality": receipt["gaussian_field_quality"],
            "appearance_render_backend": _appearance_render_backend(
                kind=receipt["particlefield_authoring_implementation"],
                source_digest=source_digest,
                particlefield_digest=receipt["output_sha256"],
                authoring_receipt_digest=receipt["receipt_digest"],
                upstream_converter=receipt.get("upstream_converter"),
                projection_mode_hint=receipt.get(
                    "upstream_projection_mode_hint"
                ),
                sorting_mode_hint=receipt.get("upstream_sorting_mode_hint"),
                color_space=receipt.get("upstream_color_space"),
            ),
        }
    if source_size > MAXIMUM_INLINE_NUREC_CONVERSION_BYTES:
        raise TaskEvaluationNativeArenaEpisodeCompilerError(
            "episode_compiler_official_particlefield_cache_required"
        )
    output_root.mkdir(mode=0o750)
    output_path = output_root / "scene_appearance.usdc"
    receipt_path = output_root / "particlefield_authoring_receipt.v1.json"
    receipt = dict(
        write_particlefield_usd_from_nurec(
            source_path,
            output_path,
            expected_source_sha256=source_digest,
            receipt_path=receipt_path,
        )
    )
    if (
        receipt.get("status") != "completed"
        or receipt.get("schema_version") != "particlefield_3dgs_authoring_receipt.v1"
        or receipt.get("schema") != "ParticleField3DGaussianSplat"
        or receipt.get("source_sha256") != source_digest
        or receipt.get("source_kind") != "nurec_usdz"
        or receipt.get("exact_learned_arrays_preserved") is not True
        or receipt.get("representation_conversion_only") is not True
        or not gaussian_quality_is_qualified(receipt.get("gaussian_field_quality"))
        or receipt.get("receipt_digest") != canonical_digest(receipt, digest_field="receipt_digest")
        or output_path.is_symlink()
        or not output_path.is_file()
        or _sha256_and_size(output_path)
        != (receipt.get("output_sha256"), receipt.get("output_bytes"))
    ):
        raise TaskEvaluationNativeArenaEpisodeCompilerError(
            "episode_compiler_particlefield_derivation_invalid"
        )
    output_path.chmod(0o440)
    receipt_path.chmod(0o440)
    return {
        "status": "nurec_converted_to_particlefield",
        "path": str(output_path),
        "representation": "particlefield_3d_gaussian_splat",
        "source_configured_appearance_digest": source_digest,
        "source_configured_appearance_size_bytes": source_size,
        "particlefield_digest": receipt["output_sha256"],
        "particlefield_size_bytes": receipt["output_bytes"],
        "particlefield_authoring_receipt_path": str(receipt_path),
        "particlefield_authoring_receipt_digest": receipt["receipt_digest"],
        "representation_conversion_performed": True,
        "exact_learned_arrays_preserved": True,
        "gaussian_field_quality": receipt["gaussian_field_quality"],
        "appearance_render_backend": _appearance_render_backend(
            kind=receipt["particlefield_authoring_implementation"],
            source_digest=source_digest,
            particlefield_digest=receipt["output_sha256"],
            authoring_receipt_digest=receipt["receipt_digest"],
            upstream_converter=receipt.get("upstream_converter"),
            projection_mode_hint=receipt.get("upstream_projection_mode_hint"),
            sorting_mode_hint=receipt.get("upstream_sorting_mode_hint"),
            color_space=receipt.get("upstream_color_space"),
        ),
    }


def compile_native_arena_episode(
    *,
    envelope: Mapping[str, Any],
    materialized_references: Mapping[str, Mapping[str, Any]],
    output_root: str | Path,
    native_appearance_materializer: NativeAppearanceMaterializer = (
        _materialize_native_particlefield_appearance
    ),
) -> dict[str, Any]:
    """Compile one production-owned native packet without provider mutation."""

    request = envelope["request"]
    root = Path(output_root).resolve()
    revision = validate_configured_scene_revision(
        _json_reference(
            materialized_references,
            "scene.configured_revision",
            "task_evaluation_configured_scene_revision.v1",
        )
    )
    if (
        revision["revision_digest"] != envelope["configured_scene_revision_digest"]
        or revision["scene_identity"] != request["scene"]["identity"]
        or revision["task_template"]["identity"] != request["task"]["identity"]
        or revision["replacement"]["identity"] != request["task"]["subject"]["identity"]
    ):
        raise TaskEvaluationNativeArenaEpisodeCompilerError(
            "episode_compiler_configured_revision_binding_mismatch"
        )
    robot = _json_reference(
        materialized_references,
        "robot.configuration",
        "task_evaluation_native_robot_configuration.v1",
    )
    kinematics = _json_reference(
        materialized_references,
        "robot.kinematics",
        "task_evaluation_native_robot_kinematics.v1",
    )
    joint_bounds = _json_reference(
        materialized_references,
        "robot.joint_bounds",
        "task_evaluation_native_robot_joint_bounds.v1",
    )
    base = _json_reference(
        materialized_references,
        "robot.base_registration",
        "task_evaluation_robot_to_scene_registration.v1",
    )
    controller = _json_reference(
        materialized_references,
        "controller.configuration",
        "task_evaluation_native_controller_configuration.v1",
    )
    sensors = _json_reference(
        materialized_references,
        "sensors.configuration",
        "task_evaluation_native_sensor_configuration.v1",
    )
    task_adapter = adapt_rigid_relocation_task_template(
        request=request,
        configured_revision=revision,
        materialized_references=materialized_references,
    )
    task_definition = task_adapter["native_task_definition"]
    success = task_adapter["native_success_criteria"]
    execution = task_adapter["native_episode_execution"]
    for label, value in (
        ("robot.configuration", robot),
        ("robot.kinematics", kinematics),
        ("robot.joint_bounds", joint_bounds),
    ):
        _identity(value, request["robot"]["identity"], label=label)
    _identity(controller, request["controller"]["identity"], label="controller")
    if (
        controller.get("kind") != request["controller"]["kind"]
        or base.get("robot_identity") != request["robot"]["identity"]
        or base.get("scene_identity") != request["scene"]["identity"]
        or base.get("robot_mount_interface_digest")
        != revision["registration"]["robot_mount_interface"]["digest"]
        or sensors.get("scene_camera_calibration_digest")
        != revision["registration"]["camera_calibration"]["digest"]
        or task_definition.get("identity") != request["task"]["identity"]
        or success.get("identity") != request["task"]["identity"]
        or execution.get("identity") != request["task"]["identity"]
    ):
        raise TaskEvaluationNativeArenaEpisodeCompilerError(
            "episode_compiler_request_binding_mismatch"
        )
    task_spec = task_definition.get("task_spec")
    if not isinstance(task_spec, Mapping):
        raise TaskEvaluationNativeArenaEpisodeCompilerError(
            "episode_compiler_task_definition_invalid"
        )
    task_spec = dict(task_spec)
    task_spec["subject_asset_id"] = request["task"]["subject"]["identity"]["id"]
    task_spec["manipulation_strategy"] = request["task"]["strategy"]
    task_spec["success_criteria"] = success.get("criteria")
    task_spec = _runtime_subject_task_spec(task_spec)
    configured_assets = _extract_configured_assets(
        _reference_path(
            materialized_references,
            "scene.configured_revision.configured_scene_bundle",
        ),
        output_root=root,
    )
    destination_asset = _stage_destination_asset(
        request=request,
        materialized_references=materialized_references,
        output_root=root,
        configured_collision_path=configured_assets["collision"],
        task_spec=task_spec,
    )
    if destination_asset is not None:
        affordance = dict(task_spec["interaction_affordance"])
        affordance["intended_support_prim_paths"] = destination_asset[
            "intended_support_prim_paths"
        ]
        affordance["insertion_withdrawal_unit_world"] = destination_asset[
            "insertion_withdrawal_unit_world"
        ]
        affordance["affordance_digest"] = canonical_digest(
            affordance, digest_field="affordance_digest"
        )
        task_spec.update(
            destination_relation=destination_asset["relation"],
            destination_support_asset_id=destination_asset["asset_id"],
            destination_position_bounds_destination_frame_m=destination_asset[
                "destination_position_bounds_destination_frame_m"
            ],
            destination_pose_world=destination_asset["destination_pose_world"],
            destination_orientation_xyzw=destination_asset[
                "destination_orientation_world_xyzw"
            ],
            destination_reset_translation_tolerance_m=destination_asset[
                "destination_reset_translation_tolerance_m"
            ],
            destination_reset_rotation_tolerance_rad=destination_asset[
                "destination_reset_rotation_tolerance_rad"
            ],
            target_position_world_m=destination_asset["target_position_world_m"],
            support_height_interval_m=destination_asset[
                "support_height_interval_m"
            ],
            visible_target_label=destination_asset["visible_label"],
            prompt=(
                f"Pick up the configured rigid object and place it "
                f"{destination_asset['relation']} the "
                f"{destination_asset['visible_label']}."
            ),
            interaction_affordance=affordance,
        )
    native_candidate_universe = robot.get("native_construction_candidate_universe")
    if native_candidate_universe is not None:
        if not isinstance(native_candidate_universe, Mapping):
            raise TaskEvaluationNativeArenaEpisodeCompilerError(
                "episode_compiler_native_candidate_universe_invalid"
            )
        try:
            native_candidate_universe = validate_native_construction_inventory(
                native_candidate_universe,
                expected_run_id=str(native_candidate_universe.get("run_id") or ""),
                expected_round_index=0,
                expected_feedback_digest=None,
                maximum_candidates=64,
            )
        except ValueError as exc:
            raise TaskEvaluationNativeArenaEpisodeCompilerError(
                "episode_compiler_native_candidate_universe_invalid"
            ) from exc
    source_subject_asset_id = task_spec["source_subject_identity"]
    runtime_subject_asset_id = task_spec["subject_asset_id"]
    policy_observation_setup = request["execution_adapter"].get(
        "policy_observation_setup"
    )
    if policy_observation_setup is not None:
        if (
            policy_observation_setup.get("schema_version")
            != "task_evaluation_policy_observation_setup.v1"
            or policy_observation_setup.get("fresh_native_mount_sweep_required")
            is not True
            or policy_observation_setup.get("policy_master_resolution_wh")
            != list(POLICY_RENDER_RESOLUTION)
            or policy_observation_setup.get("overview_review_resolution_wh")
            != list(OVERVIEW_RENDER_RESOLUTION)
        ):
            raise TaskEvaluationNativeArenaEpisodeCompilerError(
                "episode_compiler_policy_observation_setup_invalid"
            )
        native_appearance_path = _link_policy_observation_asset(
            source=_reference_path(
                materialized_references,
                "execution_adapter.policy_observation_setup.appearance_asset",
            ),
            output_root=root,
        )
        native_appearance: dict[str, Any] = {}
    else:
        native_appearance = dict(
            native_appearance_materializer(
                source_path=configured_assets["appearance"],
                output_root=root / "native-appearance",
            )
        )
        native_appearance_path = Path(str(native_appearance.get("path") or ""))
        if (
            native_appearance.get("representation")
            != "particlefield_3d_gaussian_splat"
            or native_appearance.get("exact_learned_arrays_preserved") is not True
            or native_appearance_path.is_symlink()
            or not native_appearance_path.is_file()
            or (
                native_appearance_path != root
                and root not in native_appearance_path.resolve().parents
            )
        ):
            raise TaskEvaluationNativeArenaEpisodeCompilerError(
                "episode_compiler_native_appearance_invalid"
            )
    object_pose = task_definition.get("task_object_pose_world")
    packet_assets = []
    for role, semantic_role in (
        ("appearance", "scene_appearance"),
        ("collision", "scene_collision"),
        ("replacement", "task_object"),
    ):
        path = native_appearance_path if role == "appearance" else configured_assets[role]
        digest, size = _sha256_and_size(path)
        row: dict[str, Any] = {
            "semantic_role": semantic_role,
            "filename": path.name,
            "source": {
                "root": "evidence",
                "relative_path": path.relative_to(root).as_posix(),
                "size_bytes": size,
                "sha256": digest,
            },
            "pose_world": object_pose
            if role == "replacement"
            else {
                "position_world_m": [0.0, 0.0, 0.0],
                "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
        }
        if role == "replacement":
            row.update(
                asset_id=runtime_subject_asset_id,
                source_asset_id=source_subject_asset_id,
                object_type="RIGID",
                reset_state={"root_pose_world": object_pose, "joint_positions": {}},
            )
        packet_assets.append(row)
    if destination_asset is not None:
        path = destination_asset["path"]
        digest, size = _sha256_and_size(path)
        packet_assets.append(
            {
                "semantic_role": "task_support",
                "filename": path.name,
                "asset_id": destination_asset["asset_id"],
                "source_asset_id": destination_asset["source_asset_id"],
                "object_type": "RIGID",
                "source": {
                    "root": "evidence",
                    "relative_path": path.relative_to(root).as_posix(),
                    "size_bytes": size,
                    "sha256": digest,
                },
                "pose_world": destination_asset["pose_world"],
                "reset_state": {
                    "root_pose_world": destination_asset["pose_world"],
                    "joint_positions": {},
                },
            }
        )
    control_search = None
    if native_candidate_universe is not None:
        control_search = {
            "schema_version": "task_evaluation_control_search_authority.v1",
            "enabled": True,
            "claim_ceiling": "development_only_control_search",
            "provider_allocations_performed": 0,
            "requested_vector_env_count": 256,
            "maximum_vector_env_count": 1_024,
            "seeds_per_candidate": 1,
            "shortlist_size": 16,
            "appearance_mode": "omitted",
            "camera_mode": "disabled",
            "full_fidelity_replay_required": True,
            "authority_digest": "",
        }
        control_search["authority_digest"] = canonical_digest(
            control_search, digest_field="authority_digest"
        )
    packet_request: dict[str, Any] = {
        "schema_version": "native_task_arena_packet_request.v1",
        "scene_id": request["scene"]["identity"]["id"],
        "task_id": request["task"]["identity"]["id"],
        "task_spec": task_spec,
        "task_joint_bindings": task_definition.get("task_joint_bindings") or [],
        "task_state_binding": task_definition.get("task_state_binding"),
        "assets": packet_assets,
        "robot_base_pose_world": base.get("pose_world"),
        "robot_joint_reset_positions_rad": robot.get("joint_reset_positions_rad"),
        "cameras": sensors.get("cameras"),
        "scenario": execution.get("scenario"),
        "physics_frequency_hz": execution.get("physics_frequency_hz"),
        "configured_task_template_adapter": task_adapter,
        "appearance_variant": (
            {
                "representation": "particlefield_3d_gaussian_splat",
                "source_configured_appearance_digest": native_appearance[
                    "source_configured_appearance_digest"
                ],
                "representation_conversion_performed": native_appearance[
                    "representation_conversion_performed"
                ],
                "exact_learned_arrays_preserved": True,
                "gaussian_field_quality": native_appearance[
                    "gaussian_field_quality"
                ],
                "render_backend": native_appearance[
                    "appearance_render_backend"
                ],
            }
            if policy_observation_setup is None
            else None
        ),
        "native_construction_feedback": (
            {
                "selected_placement_candidate_id": robot.get("selected_placement_candidate_id"),
                "candidate_universe": native_candidate_universe,
                "candidate_generator_authority": {
                    "generator": "remote_curobo_v2_motion_generation",
                    "package_version": "0.8.0",
                    "source_revision": ("4ea77366ca48ee453e7df139e39fa6532af49f3b"),
                    "required_on_retained_gpu": True,
                    "deterministic_cpu_prefilter_required": True,
                    "silent_fallback_permitted": False,
                },
                "allocator_retry_cap": 0,
                "maximum_rounds": 8,
                "native_gates_unchanged": True,
                "control_search": control_search,
            }
            if native_candidate_universe is not None
            else None
        ),
        "request_digest": "",
    }
    packet_request["request_digest"] = canonical_digest(
        packet_request, digest_field="request_digest"
    )
    if policy_observation_setup is not None:
        base_request_path = root / "policy_observation_base_packet_request.v1.json"
        write_json(base_request_path, packet_request)
        receipt_path = _reference_path(
            materialized_references,
            "execution_adapter.policy_observation_setup.appearance_authoring_receipt",
        )
        packet_request = materialize_native_task_arena_appearance_variant_request(
            base_request_path=base_request_path,
            appearance_authoring_receipt_path=receipt_path,
            appearance_asset_path=native_appearance_path,
            evidence_root=root,
            output_path=root / "policy_observation_appearance_packet_request.v1.json",
        )
        registry = _json_reference(
            materialized_references,
            "execution_adapter.policy_observation_setup.wrist_camera_mount_registry",
            "policy_canary_wrist_camera_mount_registry.v1",
        )
        packet_request = materialize_wrist_camera_mount_sweep_request(
            base_request=packet_request,
            registry=registry,
            output_path=root / "policy_observation_camera_packet_request.v1.json",
        )
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        appearance_variant = packet_request["appearance_variant"]
        appearance_digest, appearance_size = _sha256_and_size(native_appearance_path)
        native_appearance = {
            "status": "policy_observation_override_materialized",
            "path": str(native_appearance_path),
            "representation": "particlefield_3d_gaussian_splat",
            "source_configured_appearance_digest": appearance_variant[
                "source_gaussian_sha256"
            ],
            "source_configured_appearance_size_bytes": None,
            "particlefield_digest": appearance_digest,
            "particlefield_size_bytes": appearance_size,
            "representation_conversion_performed": True,
            "exact_learned_arrays_preserved": True,
            "gaussian_field_quality": appearance_variant["gaussian_field_quality"],
            "appearance_render_backend": _appearance_render_backend(
                kind=appearance_variant["particlefield_authoring_implementation"],
                source_digest=appearance_variant["source_gaussian_sha256"],
                particlefield_digest=appearance_digest,
                authoring_receipt_digest=appearance_variant[
                    "authoring_receipt_digest"
                ],
                upstream_converter=appearance_variant.get("upstream_converter"),
                projection_mode_hint=appearance_variant.get(
                    "upstream_projection_mode_hint"
                ),
                sorting_mode_hint=appearance_variant.get(
                    "upstream_sorting_mode_hint"
                ),
                color_space=appearance_variant.get("upstream_color_space"),
            ),
            "policy_observation_setup_bound": True,
            "appearance_authoring_receipt_digest": receipt["receipt_digest"],
            "wrist_camera_mount_registry_digest": registry["registry_digest"],
        }
    packet_root = root / "native-task-packet"
    # The extracted assets and the packet share this per-run root and are
    # retired together, so the packet hard-links the verified bytes instead of
    # writing a second multi-gigabyte copy onto the control plane's disk.
    materialize_native_task_arena_packet(
        request=packet_request,
        evidence_root=root,
        output_dir=packet_root,
        link_sources_within=root,
    )
    packet_zip = root / "native-task-arena-bundle.zip"
    build_task_evaluation_adapter_bundle(
        source_root=packet_root,
        output_path=packet_zip,
        request=request,
        role="construction_packet",
    )
    digest, size = _sha256_and_size(packet_zip)
    adapter_result = materialize_native_arena_adapter(
        request=request,
        compiled_episode_packet_path=packet_zip,
        compiled_episode_packet_reference={
            "uri": "production-internal://compiled-episode-packet",
            "digest": digest,
            "size_bytes": size,
        },
        configured_revision=revision,
        runtime_source_bundle_path=_reference_path(
            materialized_references,
            "execution_adapter.runtime_source_bundle",
        ),
        output_root=root / "native-arena-adapter",
        content_store_root=root.parent / "content-addressed" / "adapter-members" / "sha256",
        external_layers={
            str(row["digest"]): Path(str(row["materialized_path"]))
            for contract_path, row in materialized_references.items()
            if contract_path.startswith(RUNTIME_SOURCE_LAYER_CONTRACT_PREFIX)
        },
    )
    output: dict[str, Any] = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "status": "completed",
        "run_id": envelope["run_id"],
        "configured_scene_revision_digest": revision["revision_digest"],
        "configured_task_template_adapter": {
            "schema_version": task_adapter["schema_version"],
            "adapter_digest": task_adapter["adapter_digest"],
            "source_documents_digest": task_adapter["source_documents"]["source_documents_digest"],
            "manipulation_strategy": task_adapter["manipulation_strategy"],
        },
        "compiled_episode_packet": {
            "format": "native_task_arena_bundle_zip",
            "path": str(packet_zip),
            "digest": digest,
            "size_bytes": size,
        },
        "adapter_result": {
            "path": str(
                root
                / "native-arena-adapter"
                / "task_evaluation_native_arena_adapter_result.v1.json"
            ),
            "digest": adapter_result["result_digest"],
            "packet_receipt_digest": adapter_result["packet_receipt_digest"],
            "runtime_source_receipt_digest": adapter_result["runtime_source_receipt_digest"],
        },
        "native_scene_appearance": {
            key: value for key, value in native_appearance.items() if key != "path"
        },
        "compiled_by_production": True,
        "customer_supplied_prebuilt_episode_packet": False,
        "provider_mutation_performed": False,
        "paid_execution_requested": False,
        "raw_secret_values_recorded": False,
        "compiler_output_digest": "",
    }
    output["compiler_output_digest"] = canonical_digest(
        output, digest_field="compiler_output_digest"
    )
    return output


__all__ = [
    "OUTPUT_SCHEMA_VERSION",
    "TaskEvaluationNativeArenaEpisodeCompilerError",
    "compile_native_arena_episode",
]
