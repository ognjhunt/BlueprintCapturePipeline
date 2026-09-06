"""Readback and source-specific preparation for completed mesh and splat inputs."""
from __future__ import annotations

import json
import math
from pathlib import Path

from .decision_evidence_contracts import canonical_digest, canonical_json
from .task_evaluation_scene_configuration_disclosure import MESH_INPUT_STATUS, MATERIALIZED_STATUS
from .task_evaluation_scene_configuration_submission_inputs import require


def validate_completed_scene_inputs(*, envelope: dict, configurations: dict,
                                   require_render_inputs: bool) -> None:
    from .provided_scene_mesh import inspect_mesh
    from .task_evaluation_scene_configuration_source_preflight import _reference, _json, _request_reference_matches
    request = envelope["request"]
    manifest_row, manifest_path = _reference(envelope, "scene.source_manifest")
    collision_row, collision_path = _reference(envelope, "scene.geometry.collision")
    appearance_row, _ = _reference(envelope, "scene.appearance.representation")
    normalization_row, normalization_path = _reference(envelope, "scene.geometry.validation")
    for keys, row in [(("scene", "source_manifest"), manifest_row),
                      (("scene", "geometry", "collision"), collision_row),
                      (("scene", "geometry", "validation"), normalization_row),
                      (("scene", "appearance", "representation"), appearance_row)]:
        require(_request_reference_matches(request, keys, row), "completed_scene_request_reference_mismatch")
    manifest = _json(manifest_path, code="completed_scene_manifest_invalid")
    normalization = _json(normalization_path, code="completed_scene_normalization_invalid")
    subject = manifest.get("source_task_object", {})
    support = manifest.get("source_support_object", {})
    recipe = envelope["recipe"]
    first, second, third = (configurations[stage["stage_id"]] for stage in recipe["stage_sequence"][:3])
    require(manifest.get("schema_version") == "task_evaluation_completed_scene_source_manifest.v1"
            and manifest.get("source") == "owner_provided_completed_asset"
            and recipe["source_manifest_digest"] == manifest_row["digest"]
            and normalization.get("schema_version") == "task_evaluation_completed_mesh_normalization.v1"
            and normalization.get("normalization_digest") == canonical_digest(normalization, digest_field="normalization_digest")
            and normalization.get("output", {}).get("sha256") == collision_row["digest"]
            and normalization.get("output", {}).get("size_bytes") == collision_row["size_bytes"]
            and normalization.get("physical_scale_measured") is False
            and normalization.get("source_bytes_unchanged") is True,
            "completed_scene_normalization_binding_invalid")
    rows = {row["role"]: row for row in manifest.get("artifacts", [])}
    appearance_role = manifest.get("runtime_appearance_role", "owner_appearance_source")
    require(appearance_role in {"owner_appearance_source", "normalized_owner_appearance"}, "completed_scene_appearance_role_invalid")
    require(rows.get(appearance_role, {}).get("sha256") == appearance_row["digest"]
            and rows.get(appearance_role, {}).get("size_bytes") == appearance_row["size_bytes"]
            and rows.get("owner_collision_source", {}).get("sha256") == normalization.get("source_digest")
            and rows.get("normalized_owner_collision", {}).get("sha256") == collision_row["digest"],
            "completed_scene_source_digest_mismatch")
    if appearance_role == "normalized_owner_appearance":
        value = manifest.get("appearance_normalization", {})
        require(value.get("normalization_digest") == canonical_digest(value, digest_field="normalization_digest")
                and value.get("source_digest") == rows["owner_appearance_source"]["sha256"]
                and value.get("output", {}).get("sha256") == appearance_row["digest"]
                and value.get("physical_scale_measured") is False,
                "completed_scene_appearance_normalization_invalid")
    # CAS filenames do not necessarily retain the source format. The normalized
    # layer is always USDA; inspect_mesh keeps its exact bytes under an alias.
    inspected = inspect_mesh(collision_path, original_filename="normalized_scene.usda",
        coordinate_frame_declaration={"meters_per_unit": 1.0, "up_axis": "Z"})
    require(inspected == normalization.get("normalized_inspection"), "completed_scene_geometry_readback_changed")
    objects = {row["source_object_id"]: row for row in inspected["objects"]}
    object_row = objects.get(subject.get("runtime_prim_path"))
    support_row = objects.get(support.get("runtime_prim_path"))
    require(object_row is not None and support_row is not None and object_row != support_row
            and normalization["object_mapping"].get(subject.get("source_object_id")) == subject.get("runtime_prim_path")
            and normalization["object_mapping"].get(support.get("source_object_id")) == support.get("runtime_prim_path")
            and all(math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-7)
                    for key, expected in (("world_aabb_min_m", "source_aabb_min_xyz_m"),
                                          ("world_aabb_max_m", "source_aabb_max_xyz_m"))
                    for a, b in zip(object_row[key], subject.get(expected, []), strict=True))
            and second.get("exact_target_prim") == subject["runtime_prim_path"]
            and second.get("support_prim_must_remain_active") == support["runtime_prim_path"]
            and second.get("collision_source_digest") == collision_row["digest"]
            and third.get("source_object_identity") == subject["source_object_id"],
            "completed_scene_object_binding_invalid")
    require(first.get("source_origin") == "owner_provided_completed_asset"
            and first.get("source_object", {}).get("source_object_id") == subject["source_object_id"],
            "completed_scene_appearance_binding_invalid")
    require(second.get("expected_target", {}).get("aabb_min_xyz_m") == subject["source_aabb_min_xyz_m"]
            and second.get("expected_target", {}).get("aabb_max_xyz_m") == subject["source_aabb_max_xyz_m"]
            and third.get("metric_envelope", {}).get("minimum_xyz_m") == subject["source_aabb_min_xyz_m"]
            and third.get("metric_envelope", {}).get("maximum_xyz_m") == subject["source_aabb_max_xyz_m"],
            "completed_scene_stage_geometry_mismatch")
    mesh = request["scene"]["appearance"]["kind"] == "other_observed"
    if not mesh:
        require(first["source_object"].get("aabb_min_xyz_m") == subject["source_aabb_min_xyz_m"]
                and first["source_object"].get("aabb_max_xyz_m") == subject["source_aabb_max_xyz_m"],
                "completed_scene_removal_bounds_mismatch")
    require(recipe["stage_sequence"][0]["adapter"]["id"] == (
        "provided_mesh_appearance_excision" if mesh else "artifixer3d_observed_object_removal"),
        "completed_scene_appearance_adapter_mismatch")
    if require_render_inputs:
        render = envelope.get("render_inputs_result", {})
        require(render.get("status") == (MESH_INPUT_STATUS if mesh else MATERIALIZED_STATUS)
                and render.get("source_appearance_digest", render.get("source_splat_digest")) == appearance_row["digest"],
                "completed_scene_method_input_binding_invalid")
        if mesh:
            require(render.get("derived_visual_geometry", {}).get("digest") == collision_row["digest"]
                    and render.get("normalization_digest") == normalization["normalization_digest"],
                    "completed_scene_visual_geometry_binding_invalid")


def materialize_completed_mesh_inputs(*, envelope: dict, stage_one_configuration: dict,
                                      output_root: str | Path) -> dict:
    from .task_evaluation_completed_scene_adapters import mesh_appearance_configuration_refusal
    from .task_evaluation_scene_configuration_render_inputs import _materialized
    require(mesh_appearance_configuration_refusal(stage_one_configuration, envelope) is None,
            "completed_scene_mesh_input_configuration_invalid")
    appearance, _ = _materialized(envelope, contract_path="scene.appearance.representation")
    collision, path = _materialized(envelope, contract_path="scene.geometry.collision")
    _, normalization_path = _materialized(envelope, contract_path="scene.geometry.validation")
    normalization = json.loads(normalization_path.read_text())
    require(normalization.get("normalization_digest") == canonical_digest(normalization, digest_field="normalization_digest")
            and normalization.get("output", {}).get("sha256") == collision["digest"],
            "completed_scene_mesh_input_normalization_invalid")
    result = {"schema_version": "task_evaluation_scene_configuration_render_inputs.v1",
        "status": MESH_INPUT_STATUS, "input_kind": "provided_mesh", "run_id": envelope["request"]["run_id"],
        "source_appearance_digest": appearance["digest"], "raw_interiorgs_bytes_in_provider_packet": False,
        "provider_disclosure_scope": "derived_runtime_visual_geometry_only",
        "normalization_digest": normalization["normalization_digest"],
        "derived_visual_geometry": {"path": str(path), "digest": collision["digest"], "size_bytes": collision["size_bytes"]},
        "derived_frames": [], "derived_frame_count": 0, "renderer_qualified": False,
        "physical_truth_claimed": False, "provider_mutation_performed": False,
        "paid_execution_requested": False, "provider_render_required": False}
    result["result_digest"] = canonical_digest(result, digest_field="result_digest")
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True, mode=0o750)
    (root / "task_evaluation_scene_configuration_render_inputs.v1.json").write_text(canonical_json(result) + "\n")
    return result
