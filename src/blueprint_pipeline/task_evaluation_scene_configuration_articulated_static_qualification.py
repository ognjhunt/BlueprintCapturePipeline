"""Static qualification of an authored articulated assembly (stage 4, articulated kind).

Sibling of the rigid qualifier. It proves, from the exact stage-3 USDZ bytes and
the sealed authoring receipt and graph spec, that the assembly is portable, is
one articulation root of dynamic links, has exactly one passive task joint with
finite limits and a closed reset, keeps every other joint fixed, carries a
tagged handle on the moving link, has per-link colliders, masses, inertia and
bounded physics materials, and labels generated geometry. Passing it yields a
statically admitted candidate only; native import and physical equivalence
remain separate claims.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .articulation_graph_contract import ArticulationGraphContractError, validate_articulation_graph
from .decision_evidence_contracts import canonical_digest, canonical_json
from .task_evaluation_scene_configuration_static_qualification import (
    TaskEvaluationSceneConfigurationStaticQualificationError,
    _close_sequence,
    _finite,
    _is_exact_package_member,
    _portable_package_findings,
    _sha256,
)
from .task_evaluation_scene_configuration_submission_records import ARTICULATED_STATIC_CHECKS
from .task_object_articulated_packaging import (
    GENERATED_PROVENANCE,
    HANDLE_ROLE,
    OBSERVED_PROVENANCE,
    PROVENANCE_ATTRIBUTE,
    TASK_CONTACT_ROLE_ATTRIBUTE,
)

SCHEMA_VERSION = "task_evaluation_articulated_replacement_static_qualification.v1"
GRAPH_SCHEMA_VERSION = "task_evaluation_articulated_replacement_graph.v1"
RECEIPT_SCHEMA_VERSION = "task_evaluation_articulated_replacement_authoring_result.v1"
COMPLETION_SCHEMA_VERSION = "task_evaluation_articulated_candidate_physics_completion.v1"
_DYNAMIC_MESH_COLLISION_APPROXIMATIONS = {"convexDecomposition", "convexHull"}


def _bounds_valid(value: Any) -> bool:
    return (isinstance(value, Mapping) and _finite(value.get("mass_kg") or [])
            and all(_finite(value.get(name) or []) and len(value.get(name) or []) == 2
                    for name in ("mass_kg", "static_friction", "dynamic_friction", "restitution")))


def _usd_findings(path: Path, *, graph: Mapping[str, Any], physics_bounds: Mapping[str, Mapping[str, Sequence[float]]],
                  link_parts: Mapping[str, str]) -> tuple[list[str], dict[str, Any]]:
    try:
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, UsdUtils
    except ImportError as exc:  # pragma: no cover - provider image owns OpenUSD
        raise TaskEvaluationSceneConfigurationStaticQualificationError(["replacement_openusd_runtime_missing"]) from exc

    findings = _portable_package_findings(path)
    try:
        layers, external_assets, unresolved = UsdUtils.ComputeAllDependencies(Sdf.AssetPath(str(path)))
    except Exception:
        layers, external_assets, unresolved = [], [], [str(path)]
    package_identifier = str(path)
    layer_identifiers = [str(layer.identifier) for layer in layers]
    unpinned = [asset for asset in external_assets
                if not _is_exact_package_member(asset, package_identifier=package_identifier)]
    if (not layer_identifiers or unpinned or unresolved
            or any(identifier != package_identifier and not identifier.startswith(package_identifier + "[")
                   for identifier in layer_identifiers)):
        findings.append("replacement_external_or_unresolved_dependency")
    stage = Usd.Stage.Open(str(path), load=Usd.Stage.LoadAll)
    if stage is None or not stage.GetDefaultPrim().IsValid():
        return [*findings, "replacement_usd_unreadable"], {}
    if float(UsdGeom.GetStageMetersPerUnit(stage)) != 1.0 or str(UsdGeom.GetStageUpAxis(stage)).upper() != "Z":
        findings.append("replacement_stage_frame_invalid")
    prims = list(stage.Traverse())
    root = stage.GetDefaultPrim()
    roots = [prim for prim in prims if prim.HasAPI(UsdPhysics.ArticulationRootAPI)]
    if len(roots) != 1 or roots[0] != root:
        findings.append("replacement_single_articulation_root_required")
    bodies = [prim for prim in prims if prim.HasAPI(UsdPhysics.RigidBodyAPI)]
    body_by_path = {str(prim.GetPath()): prim for prim in bodies}
    if len(bodies) < 2:
        findings.append("replacement_articulated_links_missing")
    for prim in bodies:
        api = UsdPhysics.RigidBodyAPI(prim)
        if api.GetRigidBodyEnabledAttr().Get() is False or api.GetKinematicEnabledAttr().Get() is True:
            findings.append("replacement_articulated_link_not_dynamic:" + prim.GetName())
    graph_links = {row["link_id"] for row in graph["links"]}
    if {prim.GetName() for prim in bodies} != graph_links:
        findings.append("replacement_link_set_disagrees_with_graph")

    joints = [prim for prim in prims if prim.IsA(UsdPhysics.Joint)]
    moving = [prim for prim in joints if not prim.IsA(UsdPhysics.FixedJoint)]
    target_rows = [row for row in graph["joints"] if row["role"] == "target"]
    target_row = target_rows[0] if len(target_rows) == 1 else None
    task_joint = moving[0] if len(moving) == 1 else None
    task_child: Any = None
    observed_joint: dict[str, Any] = {}
    if task_joint is None or target_row is None:
        findings.append("replacement_single_target_joint_required")
    else:
        expected_type = {"prismatic": UsdPhysics.PrismaticJoint, "revolute": UsdPhysics.RevoluteJoint}.get(target_row["joint_type"])
        if expected_type is None or not task_joint.IsA(expected_type) or task_joint.GetName() != target_row["joint_id"]:
            findings.append("replacement_target_joint_type_or_name_mismatch")
        typed = expected_type(task_joint) if expected_type is not None and task_joint.IsA(expected_type) else None
        lower = upper = None
        if typed is not None:
            lower, upper = typed.GetLowerLimitAttr().Get(), typed.GetUpperLimitAttr().Get()
            if target_row["joint_type"] == "revolute" and lower is not None and upper is not None:
                lower, upper = math.radians(float(lower)), math.radians(float(upper))
        if (lower is None or upper is None or not _finite([lower, upper]) or float(lower) >= float(upper)
                or not _close_sequence([float(lower), float(upper)], list(target_row["limits"]))):
            findings.append("replacement_target_joint_limits_invalid")
        reset = task_joint.GetCustomDataByKey("blueprint:resetPosition")
        if (not isinstance(reset, (int, float)) or isinstance(reset, bool) or lower is None or upper is None
                or not float(lower) <= float(reset) <= float(upper) or abs(float(reset) - float(target_row["reset_position"])) > 1e-9):
            findings.append("replacement_closed_reset_outside_limits")
        for name in ("linear", "angular"):
            if task_joint.HasAPI(UsdPhysics.DriveAPI, name):
                drive = UsdPhysics.DriveAPI(task_joint, name)
                stiffness = drive.GetStiffnessAttr().Get()
                if stiffness is not None and float(stiffness) > 0.0:
                    findings.append("replacement_target_joint_position_servo_forbidden")
        targets0 = UsdPhysics.Joint(task_joint).GetBody0Rel().GetTargets()
        targets1 = UsdPhysics.Joint(task_joint).GetBody1Rel().GetTargets()
        if (len(targets0) != 1 or len(targets1) != 1 or str(targets0[0]) not in body_by_path
                or str(targets1[0]) not in body_by_path
                or body_by_path[str(targets1[0])].GetName() != target_row["child_link_id"]):
            findings.append("replacement_target_joint_bodies_invalid")
        else:
            task_child = body_by_path[str(targets1[0])]
        observed_joint = {"prim_path": str(task_joint.GetPath()), "joint_type": target_row["joint_type"],
                          "limits": [float(lower), float(upper)] if lower is not None and upper is not None else None,
                          "reset_position": reset}
    for prim in joints:
        if prim is task_joint:
            continue
        if not prim.IsA(UsdPhysics.FixedJoint):
            findings.append("replacement_non_target_joint_not_fixed:" + prim.GetName())
        joint = UsdPhysics.Joint(prim)
        if any(len(rel.GetTargets()) != 1 or str(rel.GetTargets()[0]) not in body_by_path
               for rel in (joint.GetBody0Rel(), joint.GetBody1Rel())):
            findings.append("replacement_fixed_joint_bodies_invalid:" + prim.GetName())
    if len(joints) != len(graph["joints"]):
        findings.append("replacement_joint_count_disagrees_with_graph")

    collision_prims = [prim for prim in prims if prim.HasAPI(UsdPhysics.CollisionAPI)]
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(),
                              [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy, UsdGeom.Tokens.guide],
                              useExtentsHint=False)
    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    link_rows: dict[str, dict[str, Any]] = {}
    material_bounds_ok = True
    handle_found = False
    for prim in bodies:
        link_id = prim.GetName()
        part_id = link_parts.get(link_id, "")
        bounds = physics_bounds.get(part_id)
        own = [c for c in collision_prims if str(c.GetPath()).startswith(str(prim.GetPath()) + "/")]
        if not own:
            findings.append("replacement_link_collision_missing:" + link_id)
        corners: list[Any] = []
        inverse = xform_cache.GetLocalToWorldTransform(prim).GetInverse()
        for collider in own:
            aligned = cache.ComputeWorldBound(collider).ComputeAlignedRange()
            if aligned.IsEmpty():
                findings.append("replacement_collision_geometry_invalid:" + link_id)
                continue
            lo, hi = aligned.GetMin(), aligned.GetMax()
            for x in (lo[0], hi[0]):
                for y in (lo[1], hi[1]):
                    for z in (lo[2], hi[2]):
                        corners.append(inverse.Transform(Gf.Vec3d(float(x), float(y), float(z))))
            if collider.IsA(UsdGeom.Mesh):
                approximation = (UsdPhysics.MeshCollisionAPI(collider).GetApproximationAttr().Get()
                                 if collider.HasAPI(UsdPhysics.MeshCollisionAPI) else None)
                if str(approximation or "") not in _DYNAMIC_MESH_COLLISION_APPROXIMATIONS:
                    findings.append("replacement_dynamic_mesh_collision_approximation_invalid")
            if collider.GetCustomDataByKey(PROVENANCE_ATTRIBUTE) not in {OBSERVED_PROVENANCE, GENERATED_PROVENANCE}:
                findings.append("replacement_geometry_provenance_untagged:" + str(collider.GetPath()))
            if collider.GetCustomDataByKey(TASK_CONTACT_ROLE_ATTRIBUTE) == HANDLE_ROLE:
                if task_child is not None and prim == task_child:
                    handle_found = True
                else:
                    findings.append("replacement_handle_on_non_target_link:" + str(collider.GetPath()))
            bound, _ = UsdShade.MaterialBindingAPI(collider).ComputeBoundMaterial(materialPurpose="physics")
            if not bound:
                findings.append("replacement_collision_physics_material_unbound")
            else:
                material = UsdPhysics.MaterialAPI(bound.GetPrim())
                try:
                    row = {name: float(getattr(material, getter)().Get()) for name, getter in (
                        ("static_friction", "GetStaticFrictionAttr"), ("dynamic_friction", "GetDynamicFrictionAttr"),
                        ("restitution", "GetRestitutionAttr"))}
                except (TypeError, ValueError):
                    row = {}
                if (not row or bounds is None or row["dynamic_friction"] > row["static_friction"]
                        or any(not bounds[name][0] <= row[name] <= bounds[name][1] for name in row)):
                    material_bounds_ok = False
        visual = stage.GetPrimAtPath(prim.GetPath().AppendChild("visual"))
        if not visual or visual.GetCustomDataByKey(PROVENANCE_ATTRIBUTE) not in {OBSERVED_PROVENANCE, GENERATED_PROVENANCE}:
            findings.append("replacement_visual_provenance_untagged:" + link_id)
        link_bounds = None
        if corners and all(_finite([float(c[i]) for i in range(3)]) for c in corners):
            link_bounds = {"minimum": [min(float(c[i]) for c in corners) for i in range(3)],
                           "maximum": [max(float(c[i]) for c in corners) for i in range(3)]}
        mass_value = None
        center: list[float] = []
        inertia: list[float] = []
        if prim.HasAPI(UsdPhysics.MassAPI):
            api = UsdPhysics.MassAPI(prim)
            try:
                mass_value = float(api.GetMassAttr().Get())
            except (TypeError, ValueError):
                mass_value = None
            raw_center, raw_inertia = api.GetCenterOfMassAttr().Get(), api.GetDiagonalInertiaAttr().Get()
            center = [float(raw_center[i]) for i in range(3)] if raw_center is not None else []
            inertia = [float(raw_inertia[i]) for i in range(3)] if raw_inertia is not None else []
        if (mass_value is None or not math.isfinite(mass_value) or mass_value <= 0.0 or bounds is None
                or not bounds["mass_kg"][0] <= mass_value <= bounds["mass_kg"][1] or not _finite(center)
                or not _finite(inertia, positive=True)
                or any(inertia[i] > sum(inertia) - inertia[i] + 1e-12 for i in range(3))):
            findings.append("replacement_link_mass_or_inertia_invalid:" + link_id)
        elif link_bounds is not None and any(center[i] < link_bounds["minimum"][i] - 1e-6
                                             or center[i] > link_bounds["maximum"][i] + 1e-6 for i in range(3)):
            findings.append("replacement_link_center_of_mass_outside_collision:" + link_id)
        link_rows[link_id] = {"prim_path": str(prim.GetPath()), "part_id": part_id, "mass_kg": mass_value,
                              "center_of_mass_m": center, "diagonal_inertia_kg_m2": inertia,
                              "collision_prim_paths": [str(c.GetPath()) for c in own],
                              "collision_bounds_link_frame_m": link_bounds}
    if not material_bounds_ok:
        findings.append("replacement_physics_material_bounds_invalid")
    if task_child is not None and not handle_found:
        findings.append("replacement_handle_contact_role_missing")
    return findings, {
        "default_prim": str(root.GetPath()), "articulation_root": str(root.GetPath()),
        "links": link_rows, "task_joint": observed_joint,
        "joint_prim_paths": [str(prim.GetPath()) for prim in joints],
        "collision_prim_paths": [str(prim.GetPath()) for prim in collision_prims],
        "dependency_layer_count": len(layers), "external_asset_count": len(unpinned),
        "embedded_package_asset_count": len(external_assets) - len(unpinned),
        "unresolved_dependency_count": len(unresolved),
    }


def qualify_scene_configuration_articulated_asset_static(
    *, asset_path: str | Path, graph_spec: Mapping[str, Any], authoring_receipt: Mapping[str, Any],
    replacement_identity: Mapping[str, Any], output_path: str | Path,
) -> dict[str, Any]:
    """Prove the exact stage-3 articulated USDZ is portable, passively jointed and structurally safe."""

    asset = Path(asset_path).expanduser().resolve()
    findings: list[str] = []
    if asset.is_symlink() or not asset.is_file():
        findings.append("replacement_asset_missing")
    digest = _sha256(asset) if asset.is_file() else ""
    size = asset.stat().st_size if asset.is_file() else 0
    physics_bounds = graph_spec.get("physics_bounds")
    bounds_valid = isinstance(physics_bounds, Mapping) and physics_bounds and all(_bounds_valid(v) for v in physics_bounds.values())
    graph: dict[str, Any] | None = None
    try:
        graph = validate_articulation_graph(graph_spec.get("articulation_graph") or {})
    except (ArticulationGraphContractError, ValueError, TypeError):
        findings.append("replacement_articulation_graph_invalid")
    plan = graph_spec.get("assembly_plan")
    link_parts = ({row["link_id"]: row["part_id"] for row in plan.get("links", []) if isinstance(row, Mapping)}
                  if isinstance(plan, Mapping) else {})
    if (graph_spec.get("schema_version") != GRAPH_SCHEMA_VERSION
            or graph_spec.get("asset_id") != replacement_identity.get("id")
            or graph_spec.get("asset_version") != replacement_identity.get("version")
            or not bounds_valid or graph_spec.get("physics_authority_granted") is not False
            or not link_parts or not str(graph_spec.get("task_joint_prim_path") or "").startswith("/Asset/joints/")
            or not str(graph_spec.get("fixed_base_body_prim_path") or "").startswith("/Asset/links/")):
        findings.append("replacement_graph_spec_invalid")
    output_usd = authoring_receipt.get("output_usd")
    completion = authoring_receipt.get("candidate_physics_completion")
    if (authoring_receipt.get("schema_version") != RECEIPT_SCHEMA_VERSION
            or authoring_receipt.get("status") != "authored_candidate_pending_qualification"
            or authoring_receipt.get("asset_kind") != "articulated_assembly"
            or authoring_receipt.get("replacement_identity") != replacement_identity
            or authoring_receipt.get("physics_authority_granted") is not False
            or authoring_receipt.get("result_digest") != canonical_digest(authoring_receipt, digest_field="result_digest")
            or not isinstance(output_usd, Mapping) or output_usd.get("sha256") != digest or output_usd.get("size_bytes") != size):
        findings.append("replacement_authoring_receipt_invalid")
    observed: dict[str, Any] = {}
    if asset.is_file() and graph is not None and bounds_valid and link_parts:
        usd_findings, observed = _usd_findings(asset, graph=graph, physics_bounds=physics_bounds, link_parts=link_parts)
        findings.extend(usd_findings)
    if (not isinstance(completion, Mapping) or completion.get("schema_version") != COMPLETION_SCHEMA_VERSION
            or completion.get("status") != "bounded_candidate_completed"
            or completion.get("candidate_prior_only") is not True or completion.get("physical_truth_claimed") is not False
            or completion.get("completion_digest") != canonical_digest(completion, digest_field="completion_digest")
            or completion.get("task_joint_prim_path") != graph_spec.get("task_joint_prim_path")
            or completion.get("fixed_base_body_prim_path") != graph_spec.get("fixed_base_body_prim_path")
            or not observed
            or completion.get("task_joint_prim_path") != observed.get("task_joint", {}).get("prim_path")
            or {row["link_id"]: row for row in completion.get("links", []) if isinstance(row, Mapping)}.keys() != observed["links"].keys()
            or any(not _close_sequence([row["mass_kg"]], [observed["links"][row["link_id"]]["mass_kg"]])
                   or not _close_sequence(row["center_of_mass_m"], observed["links"][row["link_id"]]["center_of_mass_m"])
                   or not _close_sequence(row["diagonal_inertia_kg_m2"], observed["links"][row["link_id"]]["diagonal_inertia_kg_m2"])
                   or sorted(row["collision_prim_paths"]) != sorted(observed["links"][row["link_id"]]["collision_prim_paths"])
                   for row in completion.get("links", []) if isinstance(row, Mapping))):
        findings.append("replacement_physics_completion_invalid")
    if findings:
        raise TaskEvaluationSceneConfigurationStaticQualificationError(findings)
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "authored_structure_statically_qualified",
        "asset_kind": "articulated_assembly",
        "replacement_identity": dict(replacement_identity),
        "replacement_usd": {"path": str(asset), "sha256": digest, "size_bytes": size},
        "checks": dict(ARTICULATED_STATIC_CHECKS),
        "observed_structure": observed,
        "task_joint": {"joint_id": graph["joints"][[row["role"] for row in graph["joints"]].index("target")]["joint_id"],
                       **observed["task_joint"]},
        "authored_structure_statically_qualified": True,
        "structural_findings": [],
        "claim_boundary": {"native_simulator_import_qualified": False, "physical_equivalence_proven": False,
                           "generated_geometry_is_observed_truth": False, "joint_travel_is_measured": False},
        "result_digest": "",
    }
    result["result_digest"] = canonical_digest(result, digest_field="result_digest")
    destination = Path(output_path).expanduser().resolve()
    if destination.exists() or destination.is_symlink():
        raise TaskEvaluationSceneConfigurationStaticQualificationError(["replacement_static_qualification_output_exists"])
    destination.write_text(canonical_json(result) + "\n", encoding="utf-8")
    return result


__all__ = ["SCHEMA_VERSION", "qualify_scene_configuration_articulated_asset_static"]
