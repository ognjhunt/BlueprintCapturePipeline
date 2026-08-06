"""Compose the ADP-009 SAGE collision proxy with an exact SimReady replacement.

The appearance edit remains a separate 3DGS artifact.  This module only derives a
collision-and-replacement USD from inspected bytes and records the support-plane
measurement used for placement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence

from .decision_evidence_contracts import canonical_digest


REQUEST_SCHEMA_VERSION = "adp009b_simready_replacement_request.v1"
RECEIPT_SCHEMA_VERSION = "adp009b_simready_replacement_receipt.v1"


class PublicSceneSimReadyReplacementError(ValueError):
    """The observed scene or replacement evidence cannot support composition."""


def _read_object(path: Path, *, error: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PublicSceneSimReadyReplacementError(error) from exc
    if not isinstance(value, dict):
        raise PublicSceneSimReadyReplacementError(error)
    return value


def _under(path: Path, root: Path, *, error: str) -> Path:
    resolved = path.expanduser().resolve()
    root = root.expanduser().resolve()
    if resolved != root and root not in resolved.parents:
        raise PublicSceneSimReadyReplacementError(error)
    return resolved


def _required_file(path: Path, root: Path, *, error: str) -> Path:
    resolved = _under(path, root, error=error)
    if resolved.is_symlink() or not resolved.is_file() or resolved.stat().st_size <= 0:
        raise PublicSceneSimReadyReplacementError(error)
    return resolved


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _file_record(path: Path, *, root: Path) -> dict[str, Any]:
    return {
        "relative_path": path.relative_to(root).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _verify_digest(value: Mapping[str, Any], field: str, *, error: str) -> str:
    supplied = value.get(field)
    if supplied != canonical_digest(dict(value), digest_field=field):
        raise PublicSceneSimReadyReplacementError(error)
    return str(supplied)


def _artifact_by_role(manifest: Mapping[str, Any], role: str) -> Mapping[str, Any]:
    matches = [
        row
        for row in manifest.get("materialized_artifacts", [])
        if isinstance(row, Mapping) and row.get("role") == role
    ]
    if len(matches) != 1:
        raise PublicSceneSimReadyReplacementError(f"component_artifact_missing:{role}")
    return matches[0]


def _normalize_instance_id(value: Any) -> str:
    result = str(value or "")
    return result[3:] if result.startswith("ins") else result


def _obb_bounds(corners: Any) -> tuple[list[float], list[float]]:
    if not isinstance(corners, list) or len(corners) != 8:
        raise PublicSceneSimReadyReplacementError("target_obb_corners_invalid")
    parsed: list[list[float]] = []
    for point in corners:
        if not isinstance(point, list) or len(point) != 3:
            raise PublicSceneSimReadyReplacementError("target_obb_corners_invalid")
        parsed.append([float(value) for value in point])
    lower = [min(point[index] for point in parsed) for index in range(3)]
    upper = [max(point[index] for point in parsed) for index in range(3)]
    if min(upper[index] - lower[index] for index in range(3)) <= 0:
        raise PublicSceneSimReadyReplacementError("target_obb_degenerate")
    return lower, upper


def _relative_asset(from_path: Path, target: Path) -> str:
    return Path(os.path.relpath(target, start=from_path.parent)).as_posix()


def _support_probe(
    stage: Any,
    *,
    support_path: str,
    lower: Sequence[float],
    upper: Sequence[float],
    margin_m: float,
    maximum_flatness_error_m: float,
    maximum_tilt_degrees: float,
    minimum_overlapping_area_m2: float,
    maximum_support_correction_m: float,
) -> dict[str, Any]:
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    prim = stage.GetPrimAtPath(support_path)
    if not prim or not prim.IsA(UsdGeom.Mesh) or not prim.HasAPI(UsdPhysics.CollisionAPI):
        raise PublicSceneSimReadyReplacementError("support_collider_identity_invalid")
    mesh = UsdGeom.Mesh(prim)
    points = mesh.GetPointsAttr().Get()
    counts = mesh.GetFaceVertexCountsAttr().Get()
    indices = mesh.GetFaceVertexIndicesAttr().Get()
    if not points or not counts or not indices:
        raise PublicSceneSimReadyReplacementError("support_collider_mesh_empty")
    transform = UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(prim)
    world = [transform.Transform(Gf.Vec3d(point)) for point in points]
    probe_min = [float(lower[0]) - margin_m, float(lower[1]) - margin_m]
    probe_max = [float(upper[0]) + margin_m, float(upper[1]) + margin_m]

    cursor = 0
    selected: list[tuple[Any, Any, Any, float, float]] = []
    for count in counts:
        face = [world[int(index)] for index in indices[cursor : cursor + int(count)]]
        cursor += int(count)
        for offset in range(1, len(face) - 1):
            a, b, c = face[0], face[offset], face[offset + 1]
            tri_min = [min(float(a[i]), float(b[i]), float(c[i])) for i in range(2)]
            tri_max = [max(float(a[i]), float(b[i]), float(c[i])) for i in range(2)]
            if any(tri_max[i] < probe_min[i] or tri_min[i] > probe_max[i] for i in range(2)):
                continue
            cross = Gf.Cross(b - a, c - a)
            norm = float(cross.GetLength())
            if norm <= 1e-12:
                continue
            tilt = math.degrees(math.acos(min(1.0, abs(float(cross[2])) / norm)))
            mean_height = sum(float(vertex[2]) for vertex in (a, b, c)) / 3.0
            if (
                tilt <= maximum_tilt_degrees
                and abs(mean_height - float(lower[2])) <= maximum_support_correction_m
            ):
                selected.append((a, b, c, norm * 0.5, tilt))
    if not selected:
        raise PublicSceneSimReadyReplacementError("horizontal_support_triangles_missing")

    vertices = [vertex for row in selected for vertex in row[:3]]
    heights = [float(vertex[2]) for vertex in vertices]
    flatness = max(heights) - min(heights)
    maximum_tilt = max(float(row[4]) for row in selected)
    area = sum(float(row[3]) for row in selected)
    if flatness > maximum_flatness_error_m:
        raise PublicSceneSimReadyReplacementError("support_surface_not_flat")
    if maximum_tilt > maximum_tilt_degrees:
        raise PublicSceneSimReadyReplacementError("support_surface_not_horizontal")
    if area < minimum_overlapping_area_m2:
        raise PublicSceneSimReadyReplacementError("support_surface_area_insufficient")
    xy_min = [min(float(vertex[index]) for vertex in vertices) for index in range(2)]
    xy_max = [max(float(vertex[index]) for vertex in vertices) for index in range(2)]
    if any(xy_min[i] > lower[i] or xy_max[i] < upper[i] for i in range(2)):
        raise PublicSceneSimReadyReplacementError("support_surface_does_not_cover_target")
    return {
        "support_prim_path": support_path,
        "triangle_count": len(selected),
        "overlapping_triangle_area_m2": round(area, 12),
        "height_min_m": min(heights),
        "height_max_m": max(heights),
        "height_span_m": flatness,
        "maximum_tilt_degrees": maximum_tilt,
        "measured_support_height_m": sum(heights) / len(heights),
        "probe_margin_m": margin_m,
        "maximum_flatness_error_m": maximum_flatness_error_m,
        "maximum_allowed_tilt_degrees": maximum_tilt_degrees,
        "minimum_overlapping_area_m2": minimum_overlapping_area_m2,
        "measurement_source": "materialized_sage_collision_mesh_vertices_and_faces",
        "geometry_modified": False,
    }


def materialize_simready_replacement(
    *,
    request_path: Path,
    repo_root: Path,
    evidence_root: Path,
    output_usda: Path,
    output_receipt: Path,
) -> dict[str, Any]:
    repo_root = repo_root.expanduser().resolve()
    evidence_root = evidence_root.expanduser().resolve()
    request_path = _required_file(request_path, repo_root, error="replacement_request_invalid")
    output_usda = _under(output_usda, evidence_root, error="replacement_usd_outside_evidence_root")
    output_receipt = _under(
        output_receipt, repo_root, error="replacement_receipt_outside_repo_root"
    )
    request = _read_object(request_path, error="replacement_request_invalid")
    if request.get("schema_version") != REQUEST_SCHEMA_VERSION:
        raise PublicSceneSimReadyReplacementError("replacement_request_schema_invalid")
    if {"status", "admitted", "qualified"}.intersection(request):
        raise PublicSceneSimReadyReplacementError("caller_asserted_admission_forbidden")

    def repo_json(key: str, digest_field: str, error: str) -> tuple[Path, dict[str, Any], str]:
        path = _required_file(repo_root / str(request.get(key)), repo_root, error=error)
        value = _read_object(path, error=error)
        return path, value, _verify_digest(value, digest_field, error=error)

    _, sage, sage_digest = repo_json(
        "sage_component_manifest_path", "manifest_digest", "sage_component_manifest_invalid"
    )
    _, simready, simready_digest = repo_json(
        "simready_receipt_path", "receipt_digest", "simready_receipt_invalid"
    )
    _, edit, edit_digest = repo_json(
        "edit_input_receipt_path", "receipt_digest", "edit_input_receipt_invalid"
    )
    _, aura, aura_digest = repo_json(
        "aura_execution_receipt_path", "receipt_digest", "aura_execution_receipt_invalid"
    )

    if sage.get("role") != "sage3d_collision_companion":
        raise PublicSceneSimReadyReplacementError("sage_collision_component_required")
    if simready.get("status") not in {"statically_validated", "prepared_for_independent_validation"}:
        raise PublicSceneSimReadyReplacementError("simready_control_not_statically_prepared")
    if aura.get("status") != "executed_candidate":
        raise PublicSceneSimReadyReplacementError("aura_executed_candidate_required")

    scene_id = str(sage.get("scene_mapping", {}).get("publisher_scene_id") or "")
    target = sage.get("target_binding")
    scene = edit.get("scene")
    aura_scene = aura.get("scene")
    if not scene_id or not isinstance(target, Mapping) or not isinstance(scene, Mapping):
        raise PublicSceneSimReadyReplacementError("scene_target_binding_missing")
    target_instance = _normalize_instance_id(target.get("interiorgs_instance_id"))
    joined = [
        scene_id,
        str(scene.get("publisher_scene_id") or ""),
        str(aura_scene.get("publisher_scene_id") or "") if isinstance(aura_scene, Mapping) else "",
        str(simready.get("source_scene_id") or ""),
    ]
    if len(set(joined)) != 1:
        raise PublicSceneSimReadyReplacementError("replacement_scene_id_mismatch")
    instances = [
        target_instance,
        _normalize_instance_id(scene.get("target_instance_id")),
        _normalize_instance_id(aura_scene.get("target_instance_id"))
        if isinstance(aura_scene, Mapping)
        else "",
        _normalize_instance_id(simready.get("source_instance_id")),
    ]
    if not target_instance or len(set(instances)) != 1:
        raise PublicSceneSimReadyReplacementError("replacement_target_instance_mismatch")
    if target.get("separately_removable") is not True:
        raise PublicSceneSimReadyReplacementError("target_collider_not_separately_removable")

    lower, upper = _obb_bounds(scene.get("target_obb_corners_m"))
    manifest_lower = [float(value) for value in target.get("obb_aabb_min_m", [])]
    manifest_upper = [float(value) for value in target.get("obb_aabb_max_m", [])]
    if len(manifest_lower) != 3 or len(manifest_upper) != 3 or any(
        abs(manifest_lower[i] - lower[i]) > 1e-8 or abs(manifest_upper[i] - upper[i]) > 1e-8
        for i in range(3)
    ):
        raise PublicSceneSimReadyReplacementError("replacement_target_obb_mismatch")

    collision_artifact = _artifact_by_role(sage, "static_collision_geometry")
    collision_path = _required_file(
        evidence_root / str(collision_artifact.get("external_relative_path")),
        evidence_root,
        error="sage_collision_usd_missing",
    )
    if _sha256(collision_path) != collision_artifact.get("sha256"):
        raise PublicSceneSimReadyReplacementError("sage_collision_usd_digest_mismatch")
    asset_path = _required_file(
        repo_root / str(simready.get("usd", {}).get("relative_path")),
        repo_root,
        error="simready_asset_missing",
    )
    if _sha256(asset_path) != simready.get("usd", {}).get("sha256"):
        raise PublicSceneSimReadyReplacementError("simready_asset_digest_mismatch")

    aura_root = _under(
        evidence_root / str(request.get("aura_execution_root_relative_path")),
        evidence_root,
        error="aura_execution_root_outside_evidence_root",
    )
    final_ply = _required_file(
        aura_root / str(aura.get("execution", {}).get("final_point_cloud", {}).get("relative_path")),
        aura_root,
        error="aura_final_point_cloud_missing",
    )
    if _sha256(final_ply) != aura.get("execution", {}).get("final_point_cloud", {}).get("sha256"):
        raise PublicSceneSimReadyReplacementError("aura_final_point_cloud_digest_mismatch")

    try:
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdUtils
    except ImportError as exc:  # pragma: no cover - environment failure
        raise PublicSceneSimReadyReplacementError("openusd_runtime_missing") from exc

    source_stage = Usd.Stage.Open(str(collision_path))
    if source_stage is None:
        raise PublicSceneSimReadyReplacementError("sage_collision_usd_open_failed")
    if abs(float(UsdGeom.GetStageMetersPerUnit(source_stage)) - 1.0) > 1e-9:
        raise PublicSceneSimReadyReplacementError("sage_collision_units_not_meters")
    if UsdGeom.GetStageUpAxis(source_stage) != UsdGeom.Tokens.z:
        raise PublicSceneSimReadyReplacementError("sage_collision_up_axis_not_z")
    _layers, _assets, unresolved = UsdUtils.ComputeAllDependencies(Sdf.AssetPath(str(collision_path)))
    if unresolved:
        raise PublicSceneSimReadyReplacementError("sage_collision_unresolved_dependencies")
    target_path = str(target.get("collision_prim_path") or "")
    support_path = str(target.get("support_collision_prim_path") or "")
    target_prim = source_stage.GetPrimAtPath(target_path)
    if not target_prim or not target_prim.HasAPI(UsdPhysics.CollisionAPI):
        raise PublicSceneSimReadyReplacementError("target_collider_identity_invalid")

    probe = request.get("support_probe")
    if not isinstance(probe, Mapping):
        raise PublicSceneSimReadyReplacementError("support_probe_contract_missing")
    support = _support_probe(
        source_stage,
        support_path=support_path,
        lower=lower,
        upper=upper,
        margin_m=float(probe.get("footprint_margin_m")),
        maximum_flatness_error_m=float(probe.get("maximum_flatness_error_m")),
        maximum_tilt_degrees=float(probe.get("maximum_tilt_degrees")),
        minimum_overlapping_area_m2=float(probe.get("minimum_overlapping_area_m2")),
        maximum_support_correction_m=float(probe.get("maximum_support_correction_m")),
    )
    base_placement = [
        (lower[0] + upper[0]) / 2.0,
        (lower[1] + upper[1]) / 2.0,
        support["measured_support_height_m"],
    ]
    support_correction = base_placement[2] - lower[2]
    if abs(support_correction) > float(probe.get("maximum_support_correction_m")):
        raise PublicSceneSimReadyReplacementError("support_alignment_correction_too_large")

    output_usda.parent.mkdir(parents=True, exist_ok=True)
    copied_asset = output_usda.parent / "assets" / asset_path.name
    copied_asset.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(asset_path, copied_asset)
    if _sha256(copied_asset) != _sha256(asset_path):
        raise PublicSceneSimReadyReplacementError("simready_asset_copy_digest_mismatch")

    stage = Usd.Stage.CreateNew(str(output_usda))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    stage.GetRootLayer().subLayerPaths.append(_relative_asset(output_usda, collision_path))
    stage.OverridePrim(target_path).SetActive(False)
    replacement = UsdGeom.Xform.Define(stage, "/BlueprintReplacement")
    replacement.GetPrim().GetReferences().AddReference(
        _relative_asset(output_usda, copied_asset), "/canned_beverage"
    )
    replacement.AddTranslateOp().Set(Gf.Vec3d(*base_placement))
    replacement.GetPrim().SetCustomDataByKey("blueprint:source_scene_id", scene_id)
    replacement.GetPrim().SetCustomDataByKey("blueprint:source_instance_id", target_instance)
    replacement.GetPrim().SetCustomDataByKey("blueprint:aura_final_ply_sha256", _sha256(final_ply))
    stage.GetRootLayer().customLayerData = {
        "blueprint:appearance_layer_separate": True,
        "blueprint:sage_geometry_modified": False,
        "blueprint:target_collider_deactivated": target_path,
        "blueprint:support_collider_preserved": support_path,
    }
    stage.GetRootLayer().Save()

    reopened = Usd.Stage.Open(str(output_usda))
    if reopened is None:
        raise PublicSceneSimReadyReplacementError("replacement_usd_readback_failed")
    if reopened.GetPrimAtPath(target_path).IsActive():
        raise PublicSceneSimReadyReplacementError("source_target_collider_still_active")
    if not reopened.GetPrimAtPath(support_path).IsActive():
        raise PublicSceneSimReadyReplacementError("support_collider_not_preserved")
    replacement_prim = reopened.GetPrimAtPath("/BlueprintReplacement")
    replacement_collider = reopened.GetPrimAtPath(
        "/BlueprintReplacement/colliders/body_collider"
    )
    if not replacement_prim.HasAPI(UsdPhysics.RigidBodyAPI) or not replacement_collider.HasAPI(
        UsdPhysics.CollisionAPI
    ):
        raise PublicSceneSimReadyReplacementError("replacement_physics_api_readback_failed")
    _layers, _assets, unresolved = UsdUtils.ComputeAllDependencies(Sdf.AssetPath(str(output_usda)))
    if unresolved:
        raise PublicSceneSimReadyReplacementError("replacement_usd_unresolved_dependencies")
    transform = UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(
        replacement_prim
    )
    observed_translation = [float(value) for value in transform.ExtractTranslation()]
    if any(abs(observed_translation[i] - base_placement[i]) > 1e-9 for i in range(3)):
        raise PublicSceneSimReadyReplacementError("replacement_transform_readback_mismatch")

    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "program_id": "arm-decision-proof-v1",
        "adp_item": "ADP-009B",
        "scene": {
            "publisher_scene_id": scene_id,
            "target_instance_id": target_instance,
            "target_semantic_label": target.get("semantic_label"),
            "source_target_collision_prim_path": target_path,
            "support_collision_prim_path": support_path,
        },
        "bindings": {
            "sage_component_manifest_digest": sage_digest,
            "simready_control_receipt_digest": simready_digest,
            "edit_input_receipt_digest": edit_digest,
            "aura_execution_receipt_digest": aura_digest,
            "aura_final_point_cloud": _file_record(final_ply, root=evidence_root),
            "sage_collision_usd": _file_record(collision_path, root=evidence_root),
            "exact_simready_asset": _file_record(asset_path, root=repo_root),
        },
        "support_surface_measurement": support,
        "placement": {
            "local_origin": "center_of_base_datum",
            "obb_center_m": [(lower[i] + upper[i]) / 2.0 for i in range(3)],
            "nominal_obb_base_placement_m": [
                (lower[0] + upper[0]) / 2.0,
                (lower[1] + upper[1]) / 2.0,
                lower[2],
            ],
            "support_aligned_base_placement_m": base_placement,
            "support_alignment_correction_m": support_correction,
            "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
            "authority": "publisher_obb_xy_plus_measured_sage_support_mesh_z",
        },
        "composition": {
            **_file_record(output_usda, root=evidence_root),
            "replacement_asset_copy": _file_record(copied_asset, root=evidence_root),
            "appearance_3dgs_composed_into_usd": False,
            "appearance_3dgs_separate_sha256": _sha256(final_ply),
            "source_target_collider_active": False,
            "support_collider_active": True,
            "replacement_rigid_body_active": True,
            "replacement_collision_api_present": True,
            "unresolved_dependency_count": 0,
            "sage_geometry_modified": False,
        },
        "nvidia_agent_routing": {
            "geometry_smoothing_required": False,
            "reason": "observed support triangles pass frozen flatness and tilt gates",
            "next_static_gates": [
                "omni_asset_validate",
                "omni_asset_validate_geometry",
                "omni_asset_validate_physics",
                "simready_validate",
            ],
            "next_dynamic_gate": "native_ovphysx_drop_contact_settle",
            "next_visual_gate": "native_ovrtx_rgb_depth_normal_multiview_review",
        },
        "status": "composed_static_candidate",
        "blockers": [
            "native_ovphysx_drop_contact_settle_missing",
            "native_ovrtx_composite_visual_review_missing",
        ],
        "claim_ceiling": "internal_noncommercial_static_composition_candidate_only",
        "successful_inpainting_admitted": False,
        "dynamic_contact_proven": False,
        "physical_evidence": False,
    }
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    output_receipt.parent.mkdir(parents=True, exist_ok=True)
    output_receipt.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--output-usda", type=Path, required=True)
    parser.add_argument("--output-receipt", type=Path, required=True)
    args = parser.parse_args()
    materialize_simready_replacement(
        request_path=args.request,
        repo_root=args.repo_root,
        evidence_root=args.evidence_root,
        output_usda=args.output_usda,
        output_receipt=args.output_receipt,
    )


if __name__ == "__main__":
    main()
