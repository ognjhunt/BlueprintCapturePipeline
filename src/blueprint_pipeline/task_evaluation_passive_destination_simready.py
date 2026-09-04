"""Complete an agent-authored passive destination into a static-qualified USDZ."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest
from .task_evaluation_passive_destination_cad_agent import (
    RESULT_SCHEMA_VERSION as CAD_RESULT_SCHEMA_VERSION,
    validate_passive_destination_cad_request,
)
from .task_evaluation_scene_configuration_static_qualification import (
    qualify_scene_configuration_rigid_asset_static,
)


SCHEMA_VERSION = "task_evaluation_passive_destination_simready.v1"
MASS_KG = 0.75
STATIC_FRICTION = 0.60
DYNAMIC_FRICTION = 0.45
RESTITUTION = 0.05
INTENDED_SUPPORT_PRIM = "/Asset/Colliders/Bottom"


class PassiveDestinationSimReadyError(RuntimeError):
    """The CAD result, projection, physics completion, or static gate failed."""


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _read(path: str | Path) -> tuple[Path, dict[str, Any]]:
    source = Path(path).expanduser().resolve()
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PassiveDestinationSimReadyError("passive_destination_input_invalid") from exc
    if source.is_symlink() or not isinstance(value, Mapping):
        raise PassiveDestinationSimReadyError("passive_destination_input_invalid")
    return source, dict(value)


def _record(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        raise PassiveDestinationSimReadyError("passive_destination_artifact_invalid")
    return {"path": str(path), "sha256": _sha(path), "size_bytes": path.stat().st_size}


def _matches(record: Any) -> bool:
    if not isinstance(record, Mapping):
        return False
    path = Path(str(record.get("path") or "")).expanduser().resolve()
    return (
        path.is_file()
        and not path.is_symlink()
        and record.get("size_bytes") == path.stat().st_size
        and record.get("sha256") == _sha(path)
    )


def _box(stage: Any, path: str, size: Sequence[float], center: Sequence[float]) -> None:
    from pxr import Gf, UsdGeom, UsdPhysics

    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    xform = UsdGeom.Xformable(cube.GetPrim())
    xform.AddTranslateOp().Set(Gf.Vec3d(*center))
    xform.AddScaleOp().Set(Gf.Vec3d(*size))
    cube.CreatePurposeAttr(UsdGeom.Tokens.guide)
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())


def materialize_passive_destination_simready(
    *,
    cad_result_path: str | Path,
    projection_receipt_path: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    from pxr import Gf, Sdf, Usd, UsdPhysics, UsdShade, UsdUtils

    cad_path, cad = _read(cad_result_path)
    projection_path, projection = _read(projection_receipt_path)
    if (
        cad.get("schema_version") != CAD_RESULT_SCHEMA_VERSION
        or cad.get("status")
        != "candidate_authored_pending_visual_static_native_qualification"
        or cad.get("result_digest") != canonical_digest(cad, digest_field="result_digest")
        or not all(_matches((cad.get("artifacts") or {}).get(name)) for name in ("step", "inspection", "generator_source", "agent_invocation"))
        or projection.get("schema_version") != "cad_agent_mesh_usd_projection.v1"
        or projection.get("status") != "mesh_working_copy_authored"
        or projection.get("receipt_digest")
        != canonical_digest(projection, digest_field="receipt_digest")
        or projection.get("step", {}).get("sha256")
        != cad["artifacts"]["step"]["sha256"]
        or not _matches(projection.get("output_usd"))
    ):
        raise PassiveDestinationSimReadyError("passive_destination_cad_join_invalid")
    request_path, request = _read(cad["request"]["path"])
    request = validate_passive_destination_cad_request(request)
    dimensions = request["dimensions_m"]
    outer_x = float(dimensions["outer_x"])
    outer_y = float(dimensions["outer_y"])
    base = float(dimensions["base_thickness"])
    wall = float(dimensions["wall_thickness"])
    height = float(dimensions["wall_height_above_base"])
    interior_x = outer_x - 2.0 * wall
    interior_y = outer_y - 2.0 * wall
    root = Path(output_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=False, mode=0o750)
    authored = root / "passive_destination_simready.usda"
    source_stage = Usd.Stage.Open(projection["output_usd"]["path"], load=Usd.Stage.LoadAll)
    if source_stage is None or not source_stage.Export(str(authored)):
        raise PassiveDestinationSimReadyError("passive_destination_projection_invalid")
    stage = Usd.Stage.Open(str(authored), load=Usd.Stage.LoadAll)
    asset = stage.GetDefaultPrim()
    if not asset.IsValid() or str(asset.GetPath()) != "/Asset":
        raise PassiveDestinationSimReadyError("passive_destination_projection_invalid")
    UsdPhysics.RigidBodyAPI.Apply(asset)
    mass = UsdPhysics.MassAPI.Apply(asset)
    mass.CreateMassAttr(MASS_KG)
    mass.CreateCenterOfMassAttr(Gf.Vec3f(0.0, 0.0, 0.0046))
    inertia = [
        MASS_KG * (outer_y**2 + (base + height) ** 2) / 12.0,
        MASS_KG * (outer_x**2 + (base + height) ** 2) / 12.0,
        MASS_KG * (outer_x**2 + outer_y**2) / 12.0,
    ]
    mass.CreateDiagonalInertiaAttr(Gf.Vec3f(*inertia))
    material = UsdShade.Material.Define(stage, "/Asset/Materials/BlueTray")
    shader = UsdShade.Shader.Define(stage, "/Asset/Materials/BlueTray/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(0.035, 0.18, 0.70)
    )
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.65)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    for path in projection["mesh_prim_paths"]:
        UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath(path)).Bind(material)
    physics_prim = stage.DefinePrim("/Asset/Materials/Physics", "Material")
    physics = UsdPhysics.MaterialAPI.Apply(physics_prim)
    physics.CreateStaticFrictionAttr(STATIC_FRICTION)
    physics.CreateDynamicFrictionAttr(DYNAMIC_FRICTION)
    physics.CreateRestitutionAttr(RESTITUTION)
    _box(stage, INTENDED_SUPPORT_PRIM, (outer_x, outer_y, base), (0.0, 0.0, base / 2.0))
    _box(stage, "/Asset/Colliders/Left", (wall, outer_y, height), (-(outer_x - wall) / 2.0, 0.0, base + height / 2.0))
    _box(stage, "/Asset/Colliders/Right", (wall, outer_y, height), ((outer_x - wall) / 2.0, 0.0, base + height / 2.0))
    _box(stage, "/Asset/Colliders/Front", (interior_x, wall, height), (0.0, -(outer_y - wall) / 2.0, base + height / 2.0))
    _box(stage, "/Asset/Colliders/Back", (interior_x, wall, height), (0.0, (outer_y - wall) / 2.0, base + height / 2.0))
    asset.SetCustomDataByKey("blueprint:intendedSupportPrim", INTENDED_SUPPORT_PRIM)
    asset.SetCustomDataByKey("blueprint:sourceStepSha256", cad["artifacts"]["step"]["sha256"])
    asset.SetCustomDataByKey("blueprint:passiveDestination", True)
    stage.GetRootLayer().Save()
    usdz = root / "passive_destination_simready.usdz"
    if not UsdUtils.CreateNewUsdzPackage(Sdf.AssetPath(str(authored)), str(usdz)):
        raise PassiveDestinationSimReadyError("passive_destination_usdz_package_failed")
    identity = request["destination_identity"]
    bounds = {"minimum": [-outer_x / 2.0, -outer_y / 2.0, 0.0], "maximum": [outer_x / 2.0, outer_y / 2.0, base + height]}
    physics_bounds = {"mass_kg": [0.5, 1.0], "static_friction": [0.4, 0.8], "dynamic_friction": [0.3, 0.7], "restitution": [0.0, 0.1]}
    graph = {"schema_version": "task_evaluation_rigid_replacement_graph.v1", "asset_id": identity["id"], "asset_version": identity["version"], "articulation_graph": {"joints": []}, "single_rigid_candidate": True, "physics_bounds": physics_bounds, "physics_authority_granted": False}
    paths = [INTENDED_SUPPORT_PRIM, "/Asset/Colliders/Left", "/Asset/Colliders/Right", "/Asset/Colliders/Front", "/Asset/Colliders/Back"]
    completion = {"schema_version": "task_evaluation_rigid_candidate_physics_completion.v1", "status": "bounded_candidate_completed", "physics_bounds": physics_bounds, "candidate_prior_only": True, "physical_truth_claimed": False, "mass_kg": MASS_KG, "center_of_mass_m": [0.0, 0.0, 0.0046], "diagonal_inertia_kg_m2": inertia, "collision_bounds_body_frame_m": bounds, "collision_dimensions_m": [outer_x, outer_y, base + height], "collision_prim_paths": paths, "physics_materials": [{"path": "/Asset/Materials/Physics", "static_friction": STATIC_FRICTION, "dynamic_friction": DYNAMIC_FRICTION, "restitution": RESTITUTION}], "completion_digest": ""}
    completion["completion_digest"] = canonical_digest(completion, digest_field="completion_digest")
    authoring = {"schema_version": "task_evaluation_rigid_replacement_authoring_result.v1", "status": "authored_candidate_pending_qualification", "replacement_identity": identity, "physics_authority_granted": False, "output_usd": {"sha256": _sha(usdz), "size_bytes": usdz.stat().st_size}, "candidate_physics_completion": completion, "source_cad_result": _record(cad_path), "source_projection": _record(projection_path), "source_request": _record(request_path), "result_digest": ""}
    authoring["result_digest"] = canonical_digest(authoring, digest_field="result_digest")
    authoring_path = root / "passive_destination_authoring_receipt.v1.json"
    authoring_path.write_text(json.dumps(authoring, sort_keys=True, separators=(",", ":")) + "\n")
    static_path = root / "passive_destination_static_qualification.v1.json"
    static = qualify_scene_configuration_rigid_asset_static(asset_path=usdz, graph_spec=graph, authoring_receipt=authoring, replacement_identity=identity, output_path=static_path)
    rights = {"schema_version": "task_evaluation_rigid_destination_rights_admission.v1", "status": "admitted", "destination_identity": identity, "private_provider_processing_allowed": True, "provider_training_allowed": False, "public_redistribution_allowed": False, "license_identifier": "Blueprint-generated-development-asset", "rights_admission_digest": ""}
    rights["rights_admission_digest"] = canonical_digest(rights, digest_field="rights_admission_digest")
    rights_path = root / "passive_destination_rights_admission.v1.json"
    rights_path.write_text(json.dumps(rights, sort_keys=True, separators=(",", ":")) + "\n")
    result = {"schema_version": SCHEMA_VERSION, "status": "static_qualified_pending_native_import_and_placement", "destination_identity": identity, "asset": _record(usdz), "authoring_receipt": _record(authoring_path), "static_qualification": _record(static_path), "rights_admission": _record(rights_path), "intended_support_prim_paths": [INTENDED_SUPPORT_PRIM], "interior_bounds_body_frame_m": {"minimum": [-interior_x / 2.0, -interior_y / 2.0, base], "maximum": [interior_x / 2.0, interior_y / 2.0, base + height]}, "static_result_digest": static["result_digest"], "native_import_qualified": False, "placement_qualified": False, "result_digest": ""}
    result["result_digest"] = canonical_digest(result, digest_field="result_digest")
    (root / "passive_destination_simready_result.v1.json").write_text(json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cad-result", required=True)
    parser.add_argument("--projection-receipt", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args(argv)
    result = materialize_passive_destination_simready(cad_result_path=args.cad_result, projection_receipt_path=args.projection_receipt, output_root=args.output_root)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
