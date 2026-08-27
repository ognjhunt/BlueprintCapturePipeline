"""No-spend cross-checks for immutable scene-configuration source inputs."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest


class TaskEvaluationSceneConfigurationSourcePreflightError(ValueError):
    """Immutable source inputs disagree before provider allocation."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _reference(envelope: Mapping[str, Any], contract_path: str) -> tuple[Mapping[str, Any], Path]:
    rows = [
        row
        for row in envelope.get("materialized_references") or []
        if isinstance(row, Mapping) and row.get("contract_path") == contract_path
    ]
    if len(rows) != 1:
        raise TaskEvaluationSceneConfigurationSourcePreflightError(
            f"scene_configuration_source_preflight_reference_invalid:{contract_path}"
        )
    row = rows[0]
    path = Path(str(row.get("materialized_path") or "")).resolve()
    if (
        path.is_symlink()
        or not path.is_file()
        or path.stat().st_size != row.get("size_bytes")
        or _sha256(path) != row.get("digest")
        or row.get("full_byte_service_account_readback_passed") is not True
    ):
        raise TaskEvaluationSceneConfigurationSourcePreflightError(
            f"scene_configuration_source_preflight_reference_invalid:{contract_path}"
        )
    return row, path


def _json(path: Path, *, code: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TaskEvaluationSceneConfigurationSourcePreflightError(code) from exc
    if not isinstance(value, Mapping):
        raise TaskEvaluationSceneConfigurationSourcePreflightError(code)
    return dict(value)


def _request_reference_matches(
    request: Mapping[str, Any], path: tuple[str, ...], row: Mapping[str, Any]
) -> bool:
    value: Any = request
    for part in path:
        value = value.get(part) if isinstance(value, Mapping) else None
    return isinstance(value, Mapping) and all(
        value.get(field) == row.get(field) for field in ("uri", "digest", "size_bytes")
    )


def validate_scene_configuration_source_preflight(
    *,
    envelope: Mapping[str, Any],
    configurations: Mapping[str, Mapping[str, Any]],
) -> None:
    """Prove source, render, and exact collision-target identity before spend."""

    stages = (envelope.get("recipe") or {}).get("stage_sequence") or []
    stage_by_capability = {
        str(stage.get("capability") or ""): str(stage.get("stage_id") or "")
        for stage in stages
        if isinstance(stage, Mapping)
    }
    stage_one = configurations.get(
        stage_by_capability.get("observed_appearance_object_removal", "")
    )
    stage_two = configurations.get(stage_by_capability.get("collision_object_excision", ""))
    if not isinstance(stage_one, Mapping) or not isinstance(stage_two, Mapping):
        # Other bundle fixtures and future recipes without these capabilities
        # are outside this source contract.
        return

    manifest_row, manifest_path = _reference(envelope, "scene.source_manifest")
    appearance_row, _appearance_path = _reference(
        envelope, "scene.appearance.representation"
    )
    collision_row, _collision_path = _reference(envelope, "scene.geometry.collision")
    validation_row, validation_path = _reference(envelope, "scene.geometry.validation")
    request = envelope.get("request") or {}
    bindings = (
        (("scene", "source_manifest"), manifest_row),
        (("scene", "appearance", "representation"), appearance_row),
        (("scene", "geometry", "collision"), collision_row),
        (("scene", "geometry", "validation"), validation_row),
    )
    if not isinstance(request, Mapping) or any(
        not _request_reference_matches(request, path, row) for path, row in bindings
    ):
        raise TaskEvaluationSceneConfigurationSourcePreflightError(
            "scene_configuration_source_preflight_request_binding_invalid"
        )

    manifest = _json(
        manifest_path, code="scene_configuration_source_preflight_manifest_invalid"
    )
    validation = _json(
        validation_path, code="scene_configuration_source_preflight_validation_invalid"
    )
    source_collision = manifest.get("source_collision_object") or {}
    source_object = manifest.get("source_task_object") or {}
    source_artifacts = manifest.get("artifacts") or []
    collision_artifacts = [
        row
        for row in source_artifacts
        if isinstance(row, Mapping) and row.get("role") == "sage_collision_source"
    ]
    appearance_artifacts = [
        row
        for row in source_artifacts
        if isinstance(row, Mapping) and row.get("role") == "interiorgs_source_splat"
    ]
    render = envelope.get("render_inputs_result") or {}
    stage_one_source = stage_one.get("source_object") or {}
    masks = render.get("source_object_masks") or {}
    recipe = envelope.get("recipe") or {}
    if (
        manifest.get("schema_version") != "task_evaluation_scene_source_manifest.v1"
        or manifest.get("status") != "candidate_source_bytes_retained"
        or not str(manifest.get("scene_id") or "")
        or manifest.get("publisher_scene_id") != manifest.get("scene_id")
        or recipe.get("source_manifest_digest") != manifest_row.get("digest")
        or recipe.get("scene_identity") != request.get("scene", {}).get("identity")
        or len(collision_artifacts) != 1
        or collision_artifacts[0].get("sha256") != collision_row.get("digest")
        or collision_artifacts[0].get("size_bytes") != collision_row.get("size_bytes")
        or len(appearance_artifacts) != 1
        or appearance_artifacts[0].get("sha256") != appearance_row.get("digest")
        or appearance_artifacts[0].get("size_bytes") != appearance_row.get("size_bytes")
        or render.get("source_splat_digest") != appearance_row.get("digest")
        or stage_one_source.get("publisher_instance_id")
        != source_object.get("publisher_instance_id")
        or stage_one_source.get("aabb_min_xyz_m")
        != source_object.get("source_aabb_min_xyz_m")
        or stage_one_source.get("aabb_max_xyz_m")
        != source_object.get("source_aabb_max_xyz_m")
        or masks.get("source_object_identity", {}).get("publisher_instance_id")
        != stage_one_source.get("publisher_instance_id")
    ):
        raise TaskEvaluationSceneConfigurationSourcePreflightError(
            "scene_configuration_source_preflight_manifest_binding_invalid"
        )

    expected_target = stage_two.get("expected_target") or {}
    target_path = stage_two.get("exact_target_prim")
    validation_source = (validation.get("source_files") or {}).get(
        "sage_collision_usd"
    ) or {}
    matches = [
        row
        for row in validation.get("whole_object_matches") or []
        if isinstance(row, Mapping) and row.get("prim_path") == target_path
    ]
    if (
        stage_two.get("collision_source_digest") != collision_row.get("digest")
        or source_collision.get("prim_path") != target_path
        or source_collision.get("aabb_min_xyz_m") != expected_target.get("aabb_min_xyz_m")
        or source_collision.get("aabb_max_xyz_m") != expected_target.get("aabb_max_xyz_m")
        or source_collision.get("point_count") != expected_target.get("point_count")
        or source_collision.get("face_count") != expected_target.get("face_count")
        or validation.get("schema_version") != "interiorgs_sage_collision_identity.v1"
        or validation.get("receipt_digest")
        != canonical_digest(validation, digest_field="receipt_digest")
        or validation.get("whole_object_collision_identity_passed") is not True
        or validation_source.get("sha256") != collision_row.get("digest")
        or validation_source.get("size_bytes") != collision_row.get("size_bytes")
        or len(matches) != 1
        or matches[0].get("point_count") != expected_target.get("point_count")
        or matches[0].get("face_count") != expected_target.get("face_count")
    ):
        raise TaskEvaluationSceneConfigurationSourcePreflightError(
            "scene_configuration_source_preflight_collision_target_invalid"
        )


__all__ = [
    "TaskEvaluationSceneConfigurationSourcePreflightError",
    "validate_scene_configuration_source_preflight",
]
