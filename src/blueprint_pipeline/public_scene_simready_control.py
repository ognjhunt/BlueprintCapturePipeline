"""Build the deterministic CAD-derived SimReady control for ADP-009A."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
from typing import Any, Mapping, Sequence

from .decision_evidence_contracts import canonical_digest


REQUEST_SCHEMA_VERSION = "adp009a_parametric_simready_request.v1"
RECEIPT_SCHEMA_VERSION = "adp009a_parametric_simready_receipt.v1"


class PublicSceneSimReadyControlError(ValueError):
    """The source component or requested control is not safe to materialize."""


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise PublicSceneSimReadyControlError("json_object_required")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _require_under(path: Path, root: Path) -> Path:
    path = path.expanduser().resolve()
    root = root.expanduser().resolve()
    if path != root and root not in path.parents:
        raise PublicSceneSimReadyControlError(f"path_outside_approved_root:{path}")
    return path


def _required_file(path: Path, root: Path, *, name: str) -> Path:
    resolved = _require_under(path, root)
    if not resolved.is_file():
        raise PublicSceneSimReadyControlError(f"required_file_missing:{name}")
    if resolved.stat().st_size <= 0:
        raise PublicSceneSimReadyControlError(f"required_file_empty:{name}")
    return resolved


def _number(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise PublicSceneSimReadyControlError(f"invalid_number:{name}") from exc
    if result <= 0:
        raise PublicSceneSimReadyControlError(f"nonpositive_number:{name}")
    return result


def _unit_interval(value: Any, name: str) -> float:
    result = float(value)
    if not 0.0 <= result <= 1.0:
        raise PublicSceneSimReadyControlError(f"unit_interval_required:{name}")
    return result


def _file_evidence(path: Path, *, root: Path) -> dict[str, Any]:
    return {
        "relative_path": path.relative_to(root).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _lab_to_srgb(lab: Sequence[float]) -> list[float]:
    lightness, channel_a, channel_b = (float(value) for value in lab)
    fy = (lightness + 16.0) / 116.0
    fx = channel_a / 500.0 + fy
    fz = fy - channel_b / 200.0

    def inverse(value: float) -> float:
        return value**3 if value**3 > 0.008856 else (value - 16.0 / 116.0) / 7.787

    x, y, z = 0.95047 * inverse(fx), inverse(fy), 1.08883 * inverse(fz)
    linear = (
        3.2404542 * x - 1.5371385 * y - 0.4985314 * z,
        -0.969266 * x + 1.8760108 * y + 0.041556 * z,
        0.0556434 * x - 0.2040259 * y + 1.0572252 * z,
    )

    def encode(value: float) -> float:
        value = max(0.0, min(1.0, value))
        encoded = 12.92 * value if value <= 0.0031308 else 1.055 * value ** (1 / 2.4) - 0.055
        return max(0.0, min(1.0, encoded))

    return [encode(value) for value in linear]


def _visual_match_evidence(
    request: Mapping[str, Any], *, evidence_root: Path
) -> dict[str, Any] | None:
    value = request.get("visual_match_evidence")
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise PublicSceneSimReadyControlError("visual_match_evidence_invalid")
    path = _required_file(
        evidence_root / str(value.get("relative_path")),
        evidence_root,
        name="visual_match_evidence",
    )
    receipt = _read_json(path)
    supplied = receipt.get("receipt_digest")
    if supplied != canonical_digest(receipt, digest_field="receipt_digest"):
        raise PublicSceneSimReadyControlError("visual_match_evidence_digest_invalid")
    if supplied != value.get("receipt_digest"):
        raise PublicSceneSimReadyControlError("visual_match_evidence_identity_mismatch")
    aggregate = receipt.get("aggregate")
    rows = receipt.get("camera_results")
    if (
        receipt.get("status") not in {"diagnosed_mismatch", "diagnosed_match_candidate"}
        or not isinstance(aggregate, Mapping)
        or aggregate.get("projected_scale_and_pose_gate_passed") is not True
        or not isinstance(rows, list)
        or len(rows) < 3
    ):
        raise PublicSceneSimReadyControlError("visual_match_evidence_not_usable")
    labs: list[list[float]] = []
    for row in rows:
        appearance = row.get("appearance") if isinstance(row, Mapping) else None
        lab = appearance.get("reference_median_lab") if isinstance(appearance, Mapping) else None
        if not isinstance(lab, list) or len(lab) != 3 or not all(
            isinstance(item, (int, float)) and math.isfinite(float(item)) for item in lab
        ):
            raise PublicSceneSimReadyControlError("visual_match_reference_lab_missing")
        labs.append([float(item) for item in lab])
    median_lab = [statistics.median(row[index] for row in labs) for index in range(3)]
    return {
        "receipt": _file_evidence(path, root=evidence_root),
        "receipt_digest": supplied,
        "camera_count": len(labs),
        "projected_scale_and_pose_gate_passed": True,
        "reference_median_lab": median_lab,
        "derived_srgb_diffuse_color": _lab_to_srgb(median_lab),
        "derivation": "median_of_camera_reference_median_lab_then_cie_lab_d65_to_srgb",
        "authority": "synthetic_interiorgs_multiview_appearance_diagnostic_not_physical_material_truth",
    }


def _git_value(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise PublicSceneSimReadyControlError(f"validation_source_git_failed:{args[0]}")
    return completed.stdout.strip()


def _run_foundation_validation(
    request: Mapping[str, Any],
    *,
    asset_path: Path,
    evidence_root: Path,
    validator_path: Path,
    foundation_root: Path,
) -> dict[str, Any]:
    value = request.get("simready_foundation_validation")
    if not isinstance(value, Mapping):
        raise PublicSceneSimReadyControlError("simready_foundation_validation_missing")
    validator_path = validator_path.expanduser().resolve()
    foundation_root = foundation_root.expanduser().resolve()
    if not validator_path.is_file():
        raise PublicSceneSimReadyControlError("simready_validator_cli_missing")
    if not (foundation_root / ".git").exists():
        raise PublicSceneSimReadyControlError("simready_foundation_git_checkout_required")

    expected_commit = str(value.get("commit") or "")
    expected_tree = str(value.get("tree") or "")
    observed_commit = _git_value(foundation_root, "rev-parse", "HEAD")
    observed_tree = _git_value(foundation_root, "rev-parse", "HEAD^{tree}")
    if observed_commit != expected_commit or observed_tree != expected_tree:
        raise PublicSceneSimReadyControlError("simready_foundation_revision_mismatch")
    if _git_value(foundation_root, "status", "--porcelain"):
        raise PublicSceneSimReadyControlError("simready_foundation_checkout_dirty")

    version_run = subprocess.run(
        [str(validator_path), "--show-version"],
        check=False,
        capture_output=True,
        text=True,
    )
    version_text = version_run.stdout.strip()
    if version_run.returncode != 0 or version_text != str(value.get("validator_version")):
        raise PublicSceneSimReadyControlError("simready_validator_version_mismatch")

    result_path = _require_under(
        evidence_root / str(value.get("result_relative_path")), evidence_root
    )
    log_path = _require_under(evidence_root / str(value.get("log_relative_path")), evidence_root)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    pending_path = result_path.with_name(f".{result_path.name}.pending")
    pending_path.unlink(missing_ok=True)

    profile = str(value.get("profile") or "")
    profile_version = str(value.get("profile_version") or "")
    specs_root = foundation_root / "nv_core" / "sr_specs" / "docs"
    command = [
        str(validator_path),
        "--profile",
        profile,
        "--version",
        profile_version,
        "--rules-path",
        str(specs_root / "capabilities"),
        "--features-path",
        str(specs_root / "features"),
        "--profiles-path",
        str(specs_root / "profiles" / "profiles.toml"),
        "--output",
        str(pending_path),
        "-v",
        str(asset_path),
    ]
    started_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    completed = subprocess.run(
        command,
        cwd=foundation_root,
        check=False,
        capture_output=True,
        text=True,
    )
    finished_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    combined_log = completed.stdout + completed.stderr
    log_path.write_text(combined_log, encoding="utf-8")
    if completed.returncode != 0 or not pending_path.is_file():
        raise PublicSceneSimReadyControlError("simready_foundation_validation_execution_failed")

    payload = _read_json(pending_path)
    asset_result = payload.get(str(asset_path))
    if not isinstance(asset_result, Mapping):
        raise PublicSceneSimReadyControlError("simready_validation_asset_identity_mismatch")
    if asset_result.get("profile_id") != profile or asset_result.get("profile_version") != profile_version:
        raise PublicSceneSimReadyControlError("simready_validation_profile_mismatch")
    features = asset_result.get("features_summary")
    if not isinstance(features, Mapping) or not features:
        raise PublicSceneSimReadyControlError("simready_validation_features_missing")
    failed = sorted(
        str(name)
        for name, feature in features.items()
        if not isinstance(feature, Mapping) or feature.get("passed") is not True
    )
    if failed:
        raise PublicSceneSimReadyControlError(
            "simready_foundation_profile_failed:" + ",".join(failed)
        )
    pending_path.replace(result_path)
    license_path = _required_file(
        foundation_root / "LICENSE.txt", foundation_root, name="foundation_license"
    )
    return {
        "repository": str(value.get("repository")),
        "commit": observed_commit,
        "tree": observed_tree,
        "license": str(value.get("license")),
        "license_file": _file_evidence(license_path, root=foundation_root),
        "validator_version": version_text,
        "profile": profile,
        "profile_version": profile_version,
        "command": command,
        "started_at": started_at,
        "finished_at": finished_at,
        "exit_status": completed.returncode,
        "asset_sha256": _sha256_file(asset_path),
        "result": _file_evidence(result_path, root=evidence_root),
        "log": _file_evidence(log_path, root=evidence_root),
        "features": {str(name): dict(feature) for name, feature in features.items()},
        "passed": True,
    }


def _load_mesh(stl_path: Path) -> Any:
    try:
        import trimesh
    except ImportError as exc:  # pragma: no cover - environment failure
        raise PublicSceneSimReadyControlError("trimesh_runtime_missing") from exc
    try:
        mesh = trimesh.load_mesh(stl_path, force="mesh", process=True)
    except Exception as exc:  # pragma: no cover - backend error detail varies
        raise PublicSceneSimReadyControlError("cad_stl_load_failed") from exc
    if not isinstance(mesh, trimesh.Trimesh):
        raise PublicSceneSimReadyControlError("cad_stl_single_mesh_required")
    if len(mesh.vertices) < 8 or len(mesh.faces) < 8:
        raise PublicSceneSimReadyControlError("cad_stl_mesh_too_small")
    if not mesh.is_watertight or not mesh.is_winding_consistent:
        raise PublicSceneSimReadyControlError("cad_stl_watertight_winding_required")
    if int(mesh.body_count) != 1:
        raise PublicSceneSimReadyControlError("cad_stl_one_connected_body_required")
    return mesh


def _extract_cad_evidence(
    request: Mapping[str, Any], *, repo_root: Path, evidence_root: Path, expected_size_m: Sequence[float]
) -> tuple[Any, dict[str, Any]]:
    cad = request.get("cad_evidence")
    if not isinstance(cad, Mapping):
        raise PublicSceneSimReadyControlError("cad_evidence_missing")
    if cad.get("length_unit") != "millimeter":
        raise PublicSceneSimReadyControlError("cad_length_unit_millimeter_required")

    generator_path = _required_file(
        repo_root / str(cad.get("generator_path")), repo_root, name="cad_generator"
    )
    step_path = _required_file(repo_root / str(cad.get("step_path")), repo_root, name="cad_step")
    stl_path = _required_file(
        evidence_root / str(cad.get("stl_relative_path")), evidence_root, name="cad_stl"
    )
    inspection_path = _required_file(
        evidence_root / str(cad.get("inspection_relative_path")),
        evidence_root,
        name="cad_inspection",
    )
    snapshot_path = _required_file(
        evidence_root / str(cad.get("snapshot_relative_path")), evidence_root, name="cad_snapshot"
    )
    contact_sheet_path = _required_file(
        evidence_root / str(cad.get("scene_target_contact_sheet_relative_path")),
        evidence_root,
        name="scene_target_contact_sheet",
    )

    inspection = _read_json(inspection_path)
    if inspection.get("ok") is not True or inspection.get("errors") != []:
        raise PublicSceneSimReadyControlError("cad_inspection_not_passed")
    tokens = inspection.get("tokens")
    if not isinstance(tokens, list) or len(tokens) != 1 or not isinstance(tokens[0], Mapping):
        raise PublicSceneSimReadyControlError("cad_inspection_single_token_required")
    summary = tokens[0].get("summary")
    if not isinstance(summary, Mapping):
        raise PublicSceneSimReadyControlError("cad_inspection_summary_missing")
    if summary.get("kind") != "part" or summary.get("shapeCount") != 1:
        raise PublicSceneSimReadyControlError("cad_inspection_one_part_shape_required")

    mesh = _load_mesh(stl_path)
    bounds_m = [[float(value) * 0.001 for value in row] for row in mesh.bounds.tolist()]
    mesh_size_m = [bounds_m[1][index] - bounds_m[0][index] for index in range(3)]
    if abs(bounds_m[0][2]) > 1e-7:
        raise PublicSceneSimReadyControlError("cad_mesh_base_not_at_origin")
    for index, (actual, expected) in enumerate(zip(mesh_size_m, expected_size_m, strict=True)):
        if abs(actual - expected) > 0.0002:
            raise PublicSceneSimReadyControlError(f"cad_mesh_dimension_mismatch:{index}")

    for image_path, name in ((snapshot_path, "cad_snapshot"), (contact_sheet_path, "contact_sheet")):
        try:
            from PIL import Image

            with Image.open(image_path) as image:
                image.verify()
        except Exception as exc:  # pragma: no cover - Pillow error detail varies
            raise PublicSceneSimReadyControlError(f"image_evidence_invalid:{name}") from exc

    source_skill = cad.get("source_skill")
    if not isinstance(source_skill, Mapping):
        raise PublicSceneSimReadyControlError("cad_source_skill_missing")
    commit = str(source_skill.get("commit") or "")
    tree = str(source_skill.get("tree") or "")
    if len(commit) != 40 or len(tree) != 40:
        raise PublicSceneSimReadyControlError("cad_source_skill_revision_invalid")

    evidence = {
        "source_skill": dict(source_skill),
        "generator": _file_evidence(generator_path, root=repo_root),
        "step": _file_evidence(step_path, root=repo_root),
        "stl": _file_evidence(stl_path, root=evidence_root),
        "inspection": _file_evidence(inspection_path, root=evidence_root),
        "snapshot": _file_evidence(snapshot_path, root=evidence_root),
        "scene_target_contact_sheet": _file_evidence(contact_sheet_path, root=evidence_root),
        "mesh": {
            "vertex_count": len(mesh.vertices),
            "triangle_count": len(mesh.faces),
            "body_count": int(mesh.body_count),
            "watertight": bool(mesh.is_watertight),
            "winding_consistent": bool(mesh.is_winding_consistent),
            "bounds_m": bounds_m,
            "size_m": mesh_size_m,
            "source_length_unit": "millimeter",
        },
        "mesh_linear_deflection_mm": float(cad.get("mesh_linear_deflection_mm")),
        "mesh_angular_deflection_rad": float(cad.get("mesh_angular_deflection_rad")),
    }
    return mesh, evidence


def _grasp_selection(
    request: Mapping[str, Any], *, diameter: float, height: float
) -> tuple[list[list[float]], dict[str, Any]]:
    value = request.get("grasp_selection")
    if not isinstance(value, Mapping):
        raise PublicSceneSimReadyControlError("grasp_selection_missing")
    points_raw = value.get("points_local_m")
    if not isinstance(points_raw, list) or len(points_raw) < 2:
        raise PublicSceneSimReadyControlError("grasp_points_missing")
    points: list[list[float]] = []
    for point in points_raw:
        if not isinstance(point, list) or len(point) != 3:
            raise PublicSceneSimReadyControlError("grasp_point_xyz_required")
        points.append([float(item) for item in point])
    first, last = points[0], points[-1]
    if first == last:
        raise PublicSceneSimReadyControlError("grasp_line_degenerate")
    if not 0.25 * height <= first[2] <= 0.75 * height or abs(first[2] - last[2]) > 1e-6:
        raise PublicSceneSimReadyControlError("grasp_height_not_stable_mid_body")
    if first[0] * last[0] >= 0 or abs(last[0] - first[0]) < 0.9 * diameter:
        raise PublicSceneSimReadyControlError("grasp_diameter_crossing_required")
    if max(abs(first[1]), abs(last[1])) > diameter * 0.1:
        raise PublicSceneSimReadyControlError("grasp_line_not_through_body")
    rationale = str(value.get("rationale") or "").strip()
    coordinate_note = str(value.get("coordinate_note") or "").strip()
    if not rationale or not coordinate_note:
        raise PublicSceneSimReadyControlError("grasp_evidence_notes_required")
    return points, {"points_local_m": points, "rationale": rationale, "coordinate_note": coordinate_note}


def materialize_parametric_simready_control(
    *,
    request_path: Path,
    repo_root: Path,
    evidence_root: Path,
    output_usda: Path,
    output_receipt: Path,
    simready_validator: Path | None = None,
    simready_foundation_root: Path | None = None,
) -> dict[str, Any]:
    repo_root = repo_root.expanduser().resolve()
    evidence_root = evidence_root.expanduser().resolve()
    request_path = _require_under(request_path, repo_root)
    output_usda = _require_under(output_usda, repo_root)
    output_receipt = _require_under(output_receipt, repo_root)
    if not evidence_root.is_dir():
        raise PublicSceneSimReadyControlError("evidence_root_missing")
    request = _read_json(request_path)
    if request.get("schema_version") != REQUEST_SCHEMA_VERSION:
        raise PublicSceneSimReadyControlError("request_schema_invalid")
    if {"status", "admitted", "qualified"}.intersection(request):
        raise PublicSceneSimReadyControlError("caller_asserted_qualification_forbidden")

    component_path = _require_under(repo_root / str(request["scene_component_manifest_path"]), repo_root)
    component = _read_json(component_path)
    if component.get("role") != "interiorgs_appearance_scene":
        raise PublicSceneSimReadyControlError("interiorgs_component_required")
    supplied_digest = component.get("manifest_digest")
    expected_digest = canonical_digest(component, digest_field="manifest_digest")
    if supplied_digest != expected_digest:
        raise PublicSceneSimReadyControlError("scene_component_digest_mismatch")
    target = component.get("target_binding")
    if not isinstance(target, Mapping):
        raise PublicSceneSimReadyControlError("target_binding_missing")
    if target.get("semantic_label") != "canned_beverage":
        raise PublicSceneSimReadyControlError("sealed_target_not_canned_beverage")

    lower = [float(value) for value in target["obb_aabb_min_m"]]
    upper = [float(value) for value in target["obb_aabb_max_m"]]
    size = [upper[index] - lower[index] for index in range(3)]
    if min(size) <= 0:
        raise PublicSceneSimReadyControlError("target_dimensions_degenerate")
    if abs(size[0] - size[1]) > 0.005:
        raise PublicSceneSimReadyControlError("target_not_cylindrical_within_tolerance")
    world_center = [(lower[index] + upper[index]) / 2.0 for index in range(3)]
    nominal_base_placement = [world_center[0], world_center[1], lower[2]]
    diameter = (size[0] + size[1]) / 2.0
    radius = diameter / 2.0
    height = size[2]
    mesh, cad_evidence = _extract_cad_evidence(
        request, repo_root=repo_root, evidence_root=evidence_root, expected_size_m=(diameter, diameter, height)
    )
    grasp_points, grasp_evidence = _grasp_selection(request, diameter=diameter, height=height)

    mass = _number(request.get("mass_kg"), "mass_kg")
    static_friction = _unit_interval(request.get("static_friction"), "static_friction")
    dynamic_friction = _unit_interval(request.get("dynamic_friction"), "dynamic_friction")
    restitution = _unit_interval(request.get("restitution"), "restitution")
    visual_material = request.get("visual_material")
    if not isinstance(visual_material, Mapping):
        raise PublicSceneSimReadyControlError("visual_material_missing")
    diffuse = [_unit_interval(value, "diffuse_color") for value in visual_material.get("diffuse_color", [])]
    if len(diffuse) != 3:
        raise PublicSceneSimReadyControlError("diffuse_color_rgb_required")
    visual_match = _visual_match_evidence(request, evidence_root=evidence_root)
    if visual_match is not None and any(
        abs(diffuse[index] - visual_match["derived_srgb_diffuse_color"][index]) > 0.02
        for index in range(3)
    ):
        raise PublicSceneSimReadyControlError("diffuse_color_not_derived_from_visual_match")
    roughness = _unit_interval(visual_material.get("roughness"), "roughness")
    metallic = _unit_interval(visual_material.get("metallic"), "metallic")

    try:
        from pxr import Gf, Kind, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade
    except ImportError as exc:  # pragma: no cover
        raise PublicSceneSimReadyControlError("openusd_runtime_missing") from exc

    vertices_m = [[float(value) * 0.001 for value in row] for row in mesh.vertices.tolist()]
    faces = [[int(value) for value in row] for row in mesh.faces.tolist()]
    face_normals = [[float(value) for value in row] for row in mesh.face_normals.tolist()]
    extent = [
        Gf.Vec3f(*cad_evidence["mesh"]["bounds_m"][0]),
        Gf.Vec3f(*cad_evidence["mesh"]["bounds_m"][1]),
    ]

    output_usda.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(output_usda))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/canned_beverage")
    stage.SetDefaultPrim(root.GetPrim())
    Usd.ModelAPI(root.GetPrim()).SetKind(Kind.Tokens.component)
    root.GetPrim().SetCustomDataByKey("blueprint:source_scene_id", component["scene_mapping"]["publisher_scene_id"])
    root.GetPrim().SetCustomDataByKey("blueprint:source_instance_id", target["interiorgs_instance_id"])
    root.GetPrim().SetCustomDataByKey("blueprint:source_component_digest", supplied_digest)

    UsdGeom.Scope.Define(stage, "/canned_beverage/visuals")
    visual = UsdGeom.Mesh.Define(stage, "/canned_beverage/visuals/body")
    visual.CreatePointsAttr([Gf.Vec3f(*vertex) for vertex in vertices_m])
    visual.CreateFaceVertexCountsAttr([3] * len(faces))
    visual.CreateFaceVertexIndicesAttr([index for face in faces for index in face])
    visual.CreateNormalsAttr([Gf.Vec3f(*normal) for normal in face_normals for _ in range(3)])
    visual.SetNormalsInterpolation(UsdGeom.Tokens.faceVarying)
    visual.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    visual.CreateOrientationAttr(UsdGeom.Tokens.rightHanded)
    visual.CreateExtentAttr(extent)
    visual.CreatePurposeAttr(UsdGeom.Tokens.render)
    visual.CreateDoubleSidedAttr(False)
    visual.CreateDisplayColorAttr([Gf.Vec3f(*diffuse)])

    UsdGeom.Scope.Define(stage, "/canned_beverage/colliders")
    collider = UsdGeom.Mesh.Define(stage, "/canned_beverage/colliders/body_collider")
    collider.CreatePointsAttr([Gf.Vec3f(*vertex) for vertex in vertices_m])
    collider.CreateFaceVertexCountsAttr([3] * len(faces))
    collider.CreateFaceVertexIndicesAttr([index for face in faces for index in face])
    collider.CreateNormalsAttr([Gf.Vec3f(*normal) for normal in face_normals for _ in range(3)])
    collider.SetNormalsInterpolation(UsdGeom.Tokens.faceVarying)
    collider.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    collider.CreateOrientationAttr(UsdGeom.Tokens.rightHanded)
    collider.CreateExtentAttr(extent)
    collider.CreatePurposeAttr(UsdGeom.Tokens.guide)
    collider.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
    UsdPhysics.CollisionAPI.Apply(collider.GetPrim())
    mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(collider.GetPrim())
    mesh_collision.CreateApproximationAttr("sdf")

    UsdGeom.Scope.Define(stage, "/canned_beverage/materials")
    render_material = UsdShade.Material.Define(stage, "/canned_beverage/materials/green_can")
    render_shader = UsdShade.Shader.Define(stage, "/canned_beverage/materials/green_can_shader")
    render_shader.CreateIdAttr("UsdPreviewSurface")
    render_shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*diffuse))
    render_shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    render_shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
    render_material.CreateSurfaceOutput().ConnectToSource(render_shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(visual.GetPrim()).Bind(render_material)

    contact_material = UsdShade.Material.Define(stage, "/canned_beverage/materials/contact")
    material_api = UsdPhysics.MaterialAPI.Apply(contact_material.GetPrim())
    material_api.CreateStaticFrictionAttr(static_friction)
    material_api.CreateDynamicFrictionAttr(dynamic_friction)
    material_api.CreateRestitutionAttr(restitution)
    UsdShade.MaterialBindingAPI.Apply(collider.GetPrim()).Bind(contact_material, materialPurpose="physics")

    rigid_body = UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    rigid_body.CreateRigidBodyEnabledAttr(True)
    mass_api = UsdPhysics.MassAPI.Apply(root.GetPrim())
    mass_api.CreateMassAttr(mass)
    center_of_mass = [0.0, 0.0, height / 2.0]
    mass_api.CreateCenterOfMassAttr(Gf.Vec3f(*center_of_mass))
    inertia_xy = mass * (3.0 * radius * radius + height * height) / 12.0
    inertia_z = mass * radius * radius / 2.0
    mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(inertia_xy, inertia_xy, inertia_z))
    mass_api.CreatePrincipalAxesAttr(Gf.Quatf(1.0))

    grasp = UsdGeom.BasisCurves.Define(stage, "/canned_beverage/grasp_identifier_01")
    grasp.CreateTypeAttr(UsdGeom.Tokens.linear)
    grasp.CreateCurveVertexCountsAttr([len(grasp_points)])
    grasp.CreatePointsAttr([Gf.Vec3f(*point) for point in grasp_points])
    grasp.CreateExtentAttr(
        [
            Gf.Vec3f(*(min(point[index] for point in grasp_points) for index in range(3))),
            Gf.Vec3f(*(max(point[index] for point in grasp_points) for index in range(3))),
        ]
    )
    grasp.CreateWidthsAttr([0.002])
    grasp.SetWidthsInterpolation(UsdGeom.Tokens.constant)
    grasp.CreatePurposeAttr(UsdGeom.Tokens.guide)
    grasp.CreateDisplayColorAttr([Gf.Vec3f(1.0, 0.15, 0.05)])

    stage.GetRootLayer().customLayerData = {
        "SimReady_Metadata": {
            "asset_name": "canned_beverage",
            "asset_type": "prop",
            "source_file": str(request["cad_evidence"]["step_path"]),
            "usd_date_generated": str(request["usd_date_generated"]),
        },
        "blueprint:claim_ceiling": "development_only",
        "blueprint:physics_values_authority": "frozen_authoring_priors_not_measurements",
        "blueprint:visual_geometry_source": "text_to_cad_step_derived_mesh",
        "blueprint:world_placement_separate": True,
    }
    stage.GetRootLayer().Save()

    reopened = Usd.Stage.Open(str(output_usda))
    if reopened is None or reopened.GetDefaultPrim().GetPath().pathString != "/canned_beverage":
        raise PublicSceneSimReadyControlError("usd_readback_failed")
    collider_prim = reopened.GetPrimAtPath("/canned_beverage/colliders/body_collider")
    if not collider_prim.HasAPI(UsdPhysics.CollisionAPI) or not collider_prim.HasAPI(
        UsdPhysics.MeshCollisionAPI
    ):
        raise PublicSceneSimReadyControlError("collision_api_readback_failed")
    if not reopened.GetPrimAtPath("/canned_beverage").HasAPI(UsdPhysics.RigidBodyAPI):
        raise PublicSceneSimReadyControlError("rigid_body_api_readback_failed")
    if not reopened.GetPrimAtPath("/canned_beverage/grasp_identifier_01").IsA(UsdGeom.BasisCurves):
        raise PublicSceneSimReadyControlError("grasp_curve_readback_failed")

    output_digest = _sha256_file(output_usda)
    foundation_validation = None
    if simready_validator is not None or simready_foundation_root is not None:
        if simready_validator is None or simready_foundation_root is None:
            raise PublicSceneSimReadyControlError("simready_validation_paths_must_be_paired")
        foundation_validation = _run_foundation_validation(
            request,
            asset_path=output_usda,
            evidence_root=evidence_root,
            validator_path=simready_validator,
            foundation_root=simready_foundation_root,
        )
    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "program_id": "arm-decision-proof-v1",
        "adp_item": "ADP-009A",
        "control_id": str(request["control_id"]),
        "source_component_manifest_digest": supplied_digest,
        "source_scene_id": component["scene_mapping"]["publisher_scene_id"],
        "source_instance_id": target["interiorgs_instance_id"],
        "source_semantic_label": target["semantic_label"],
        "cad_evidence": cad_evidence,
        "geometry": {
            "kind": "text_to_cad_step_derived_triangle_mesh",
            "diameter_m": diameter,
            "height_m": height,
            "local_origin": "center_of_base_datum",
            "obb_center_m": world_center,
            "nominal_base_placement_m": nominal_base_placement,
            "world_placement_m": nominal_base_placement,
            "world_placement_datum": "center_of_base_datum",
            "world_placement_authored_into_asset": False,
            "source": "publisher_semantic_obb_plus_materialized_cad",
            "measurement_authoritative": False,
        },
        "visual_material": {
            "strategy": visual_material["strategy"],
            "diffuse_color": diffuse,
            "roughness": roughness,
            "metallic": metallic,
            "authority": visual_material["authority"],
        },
        "visual_match_evidence": visual_match,
        "grasp_selection": grasp_evidence,
        "physics": {
            "mass_kg": mass,
            "center_of_mass_m": center_of_mass,
            "diagonal_inertia_kg_m2": [inertia_xy, inertia_xy, inertia_z],
            "static_friction": static_friction,
            "dynamic_friction": dynamic_friction,
            "restitution": restitution,
            "authority": "frozen_authoring_priors_not_measurements",
        },
        "usd": {
            "relative_path": output_usda.relative_to(repo_root).as_posix(),
            "size_bytes": output_usda.stat().st_size,
            "sha256": output_digest,
            "meters_per_unit": 1.0,
            "up_axis": "Z",
            "default_prim": "/canned_beverage",
            "visual_prim": "/canned_beverage/visuals/body",
            "collision_prim": "/canned_beverage/colliders/body_collider",
            "grasp_prim": "/canned_beverage/grasp_identifier_01",
            "visual_and_collision_prims_distinct": True,
        },
        "checks": {
            "scene_component_digest_verified": True,
            "target_dimensions_derived_not_caller_asserted": True,
            "cad_source_and_step_hashed": True,
            "cad_inspection_passed": True,
            "cad_mesh_watertight_single_body": True,
            "cad_snapshot_and_scene_contact_sheet_verified": True,
            "usd_readback_passed": True,
            "mesh_geometry_authored": True,
            "collision_and_mesh_collision_apis_present": True,
            "rigid_body_api_present": True,
            "mass_and_inertia_authored": True,
            "contact_material_authored": True,
            "visual_material_bound": True,
            "vision_selected_grasp_curve_authored": True,
            "world_placement_separate_from_asset": True,
            "simready_foundation_profile_passed": bool(foundation_validation),
        },
        "simready_foundation_validation": foundation_validation,
        "status": "statically_validated" if foundation_validation else "prepared_for_independent_validation",
        "blockers": ["isaac_dynamic_contact_drop_slide_tip_gripper_probes_missing"]
        if foundation_validation
        else [
            "simready_foundation_profile_validation_missing",
            "isaac_dynamic_contact_drop_slide_tip_gripper_probes_missing",
        ],
        "claim_ceiling": "development_only",
        "physical_material_authority": False,
        "inpainting_result": False,
    }
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    output_receipt.parent.mkdir(parents=True, exist_ok=True)
    output_receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--output-usda", type=Path, required=True)
    parser.add_argument("--output-receipt", type=Path, required=True)
    parser.add_argument("--simready-validator", type=Path)
    parser.add_argument("--simready-foundation-root", type=Path)
    args = parser.parse_args(argv)
    receipt = materialize_parametric_simready_control(
        request_path=args.request,
        repo_root=args.repo_root,
        evidence_root=args.evidence_root,
        output_usda=args.output_usda,
        output_receipt=args.output_receipt,
        simready_validator=args.simready_validator,
        simready_foundation_root=args.simready_foundation_root,
    )
    print(json.dumps({"status": receipt["status"], "receipt_digest": receipt["receipt_digest"]}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
