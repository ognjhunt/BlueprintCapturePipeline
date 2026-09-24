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

import json
import math
import struct
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
    CAVITY_COLLISION_WALL_BOXES,
    GENERATED_PROVENANCE,
    HANDLE_ROLE,
    HANDLE_PROTRUSION_M,
    OBSERVED_PROVENANCE,
    PROVENANCE_ATTRIBUTE,
    REQUIRED_PARTS_ATTRIBUTE,
    TASK_CONTACT_ROLE_ATTRIBUTE,
    box_collision_piece,
    collision_cavity_findings,
    interior_cavity_findings,
    legacy_drawer_plan,
    required_part_findings,
)

SCHEMA_VERSION = "task_evaluation_articulated_replacement_static_qualification.v1"
GRAPH_SCHEMA_VERSION = "task_evaluation_articulated_replacement_graph.v1"
RECEIPT_SCHEMA_VERSION = "task_evaluation_articulated_replacement_authoring_result.v1"
COMPLETION_SCHEMA_VERSION = "task_evaluation_articulated_candidate_physics_completion.v1"
_DYNAMIC_MESH_COLLISION_APPROXIMATIONS = {"convexDecomposition", "convexHull"}


def _close_authored_float32_sequence(expected: Any, observed: Any) -> bool:
    """Compare a receipt value with the exact precision authored into USD float attrs."""
    try:
        rounded = [struct.unpack('f', struct.pack('f', float(value)))[0] for value in expected]
    except (OverflowError, ValueError, TypeError, struct.error):
        return False
    return all(math.isfinite(value) for value in rounded) and _close_sequence(rounded, observed)


def _collision_piece(collider: Any, link: Any, xform_cache: Any, *, UsdGeom: Any, UsdPhysics: Any) -> dict[str, Any]:
    """One collider as link-frame triangles tagged with the approximation PhysX will use."""
    from pxr import Gf
    to_link = xform_cache.GetLocalToWorldTransform(collider) * xform_cache.GetLocalToWorldTransform(link).GetInverse()
    if collider.IsA(UsdGeom.Cube):
        half = float(UsdGeom.Cube(collider).GetSizeAttr().Get()) / 2
        box = box_collision_piece([0.0, 0.0, 0.0], [2 * half] * 3)
        return {**box, "vertices": [list(to_link.Transform(Gf.Vec3d(*point))) for point in box["vertices"]]}
    if collider.IsA(UsdGeom.Mesh):
        approximation = (UsdPhysics.MeshCollisionAPI(collider).GetApproximationAttr().Get()
                         if collider.HasAPI(UsdPhysics.MeshCollisionAPI) else None)
        points = UsdGeom.Mesh(collider).GetPointsAttr().Get() or []
        indices = list(UsdGeom.Mesh(collider).GetFaceVertexIndicesAttr().Get() or [])
        return {"approximation": str(approximation or "none"),
                "vertices": [list(to_link.Transform(Gf.Vec3d(*[float(c) for c in point]))) for point in points],
                "faces": indices if len(indices) % 3 == 0 else []}
    return {"approximation": "analytic_" + str(collider.GetTypeName()), "vertices": [], "faces": []}


def _bounds_valid(value: Any) -> bool:
    return (isinstance(value, Mapping)
            and all(_finite(value.get(name) or []) and len(value.get(name) or []) == 2
                    and value[name][0] <= value[name][1]
                    for name in ("mass_kg", "static_friction", "dynamic_friction", "restitution")))


def _development_hypothesis_findings(
    *, plan: Mapping[str, Any], completion: Mapping[str, Any], observed: Mapping[str, Any],
    graph: Mapping[str, Any], physics_bounds: Mapping[str, Any]
) -> list[str]:
    """Keep a depth prior distinct from the retained thin source envelope."""

    hypothesis = plan.get("development_geometry_hypothesis")
    if hypothesis is None:
        validation = completion.get("metric_envelope_validation")
        return (["replacement_development_geometry_hypothesis_invalid"]
                if isinstance(validation, Mapping)
                and validation.get("status") == "within_development_geometry_hypothesis"
                else [])  # Historical v1 candidates did not carry this explicit prior.
    source = plan.get("source_geometry")
    dimensions = plan.get("assembly_dimensions_m")
    validation = completion.get("metric_envelope_validation")
    actual = observed.get("collision_dimensions_m")
    if (not isinstance(hypothesis, Mapping) or not isinstance(source, Mapping)
            or not isinstance(dimensions, Mapping) or not isinstance(validation, Mapping)
            or not isinstance(actual, list)):
        return ["replacement_development_geometry_hypothesis_invalid"]
    source_depth = source.get("projected_depth_m")
    estimated_depth = hypothesis.get("estimated_depth_m")
    interval = hypothesis.get("depth_interval_m")
    mass_interval = hypothesis.get("whole_assembly_mass_interval_kg")
    part_mass_bounds = hypothesis.get("revised_part_mass_bounds_kg")
    disagreement = hypothesis.get("depth_disagreement_m")
    try:
        expected = [float(dimensions["depth_x"]) + HANDLE_PROTRUSION_M,
                    float(dimensions["width_y"]), float(dimensions["height_z"])]
        source_depth = float(source_depth)
        estimated_depth = float(estimated_depth)
        disagreement = float(disagreement)
        stroke = float(hypothesis["estimated_usable_stroke_m"])
        opening_fraction = float(hypothesis["minimum_opening_fraction"])
        minimum_opening = float(hypothesis["estimated_minimum_opening_m"])
        tolerance = float(validation["maximum_dimension_relative_error"])
        relative_errors = [abs(float(actual[i]) - expected[i]) / expected[i] for i in range(3)]
    except (KeyError, TypeError, ValueError, IndexError, ZeroDivisionError):
        return ["replacement_development_geometry_hypothesis_invalid"]
    source_disagreement = validation.get("source_aabb_disagreement")
    source_min = source.get("aabb_min_xyz_m")
    source_max = source.get("aabb_max_xyz_m")
    try:
        total_mass = sum(float(row["mass_kg"]) for row in observed.get("links", {}).values())
    except (KeyError, TypeError, ValueError):
        return ["replacement_development_geometry_hypothesis_invalid"]
    target = next((row for row in graph.get("joints", []) if row["role"] == "target"), None)
    success = graph.get("success_predicate") or {}
    if (hypothesis.get("schema_version") != "articulated_cabinet_depth_hypothesis.v1"
            or hypothesis.get("claim_ceiling") != "development_only"
            or hypothesis.get("basis") not in {"bounded_cabinet_prior", "original_capture_frames"}
            or hypothesis.get("prior_record_schema_version") != "website_drawer_depth_prior.v1"
            or not isinstance(hypothesis.get("reference_retrieved_date"), str)
            or len(hypothesis["reference_retrieved_date"]) != 10
            or not isinstance(hypothesis.get("manufacturer_examples"), list)
            or not hypothesis["manufacturer_examples"]
            or not isinstance(hypothesis.get("prior_comparison"), Mapping)
            or hypothesis["prior_comparison"].get("reference_models_are_exact_match") is not False
            or (hypothesis["prior_comparison"].get("owner_reported_dimensions_m") is not None
                and (hypothesis["prior_comparison"].get("owner_reported_source") != "cabinet_owner_chat_2026-09-24"
                     or not _close_sequence(
                         [dimensions.get("depth_x"), dimensions.get("width_y"), dimensions.get("height_z")],
                         [hypothesis["prior_comparison"]["owner_reported_dimensions_m"].get(axis)
                          for axis in ("depth", "width", "height")])
                     or dimensions.get("authority") != "development_only_owner_reported_dimensions"))
            or hypothesis.get("status") != "development_only_estimate_disagrees_with_source"
            or hypothesis.get("physical_measurement_proven") is not False
            or source.get("authority") != "retained_source_envelope_not_physical_measurement"
            or not isinstance(source_min, list) or not isinstance(source_max, list)
            or not _close_sequence(source_min, hypothesis.get("source_aabb_min_xyz_m"))
            or not _close_sequence(source_max, hypothesis.get("source_aabb_max_xyz_m"))
            or not isinstance(interval, list) or len(interval) != 2 or not _finite(interval)
            or not interval[0] <= estimated_depth <= interval[1]
            or not isinstance(mass_interval, list) or len(mass_interval) != 2
            or not _finite(mass_interval, positive=True)
            or not mass_interval[0] <= total_mass <= mass_interval[1]
            or not isinstance(part_mass_bounds, Mapping)
            or part_mass_bounds != {part: row["mass_kg"] for part, row in physics_bounds.items()}
            or target is None
            or not _close_sequence(target["limits"], [0.0, stroke])
            or not _close_sequence(
                success.get("joint_intervals", {}).get(target["joint_id"]),
                [minimum_opening, stroke])
            or not _close_sequence([minimum_opening], [stroke * opening_fraction])
            or not _close_sequence([source_depth + disagreement], [estimated_depth])
            or not _close_sequence([estimated_depth], [float(dimensions["depth_x"])])
            or source_depth >= estimated_depth
            or validation.get("status") != "within_development_geometry_hypothesis"
            or validation.get("frame") != "assembly_frame_closed_plus_handle_protrusion"
            or not _close_sequence(validation.get("expected_dimensions_m"), expected)
            or not _close_sequence(validation.get("observed_collision_dimensions_m"), actual)
            or not _close_sequence(completion.get("collision_dimensions_m"), actual)
            or not _close_sequence(validation.get("dimension_relative_errors"), relative_errors)
            or not math.isfinite(tolerance) or tolerance < 0.0
            or any(error > tolerance for error in relative_errors)
            or not isinstance(source_disagreement, Mapping)
            or not _close_sequence([source_disagreement.get("source_projected_depth_m")], [source_depth])
            or not _close_sequence([source_disagreement.get("estimated_depth_m")], [estimated_depth])
            or not _close_sequence([source_disagreement.get("depth_disagreement_m")], [disagreement])
            or source_disagreement.get("physical_measurement_proven") is not False):
        return ["replacement_development_geometry_hypothesis_invalid"]
    return []


def _usd_findings(path: Path, *, graph: Mapping[str, Any], physics_bounds: Mapping[str, Mapping[str, Sequence[float]]],
                  link_parts: Mapping[str, str], plan: Mapping[str, Any]) -> tuple[list[str], dict[str, Any]]:
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
    if root.GetCustomDataByKey("blueprint:articulatedAssemblyPlanDigest") != canonical_digest(dict(plan)):
        findings.append("replacement_assembly_plan_disagrees_with_usd")
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
    graph_link_rows = {row["link_id"]: row for row in graph["links"]}
    plan_link_rows = {row["link_id"]: row for row in plan["links"]}

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
                or body_by_path[str(targets0[0])].GetName() != target_row["parent_link_id"]
                or body_by_path[str(targets1[0])].GetName() != target_row["child_link_id"]):
            findings.append("replacement_target_joint_bodies_invalid")
        else:
            task_child = body_by_path[str(targets1[0])]
        if typed is not None:
            axis_token = str(typed.GetAxisAttr().Get() or "").upper()
            axis_value = task_joint.GetCustomDataByKey("blueprint:graphAxis")
            declared_axis = list(target_row["axis"])
            token_vector = {"X": [1.0, 0.0, 0.0], "Y": [0.0, 1.0, 0.0],
                            "Z": [0.0, 0.0, 1.0]}.get(axis_token)
            try:
                axis_components = [float(v) for v in axis_value]
            except (TypeError, ValueError):
                axis_components = []
            # The token is in the joint frame; both body frames carry the same
            # joint rotation, which maps it onto the declared asset-frame axis.
            joint_api = UsdPhysics.Joint(task_joint)
            rot0, rot1 = joint_api.GetLocalRot0Attr().Get(), joint_api.GetLocalRot1Attr().Get()
            quaternions = [[q.GetReal(), *q.GetImaginary()] if q is not None else [1.0, 0.0, 0.0, 0.0]
                           for q in (rot0, rot1)]
            body_axis = (list(Gf.Rotation(Gf.Quatd(quaternions[0][0], Gf.Vec3d(*quaternions[0][1:])))
                              .TransformDir(Gf.Vec3d(*token_vector))) if token_vector is not None else None)
            if (token_vector is None or not _close_sequence(axis_components, declared_axis)
                    or not _close_sequence(body_axis, declared_axis)
                    or not _close_sequence(quaternions[0], quaternions[1])):
                findings.append("replacement_target_joint_axis_mismatch")
        declared_drive = target_row["drive"]
        revolute = target_row["joint_type"] == "revolute"
        drive_name, other_name = ("angular", "linear") if revolute else ("linear", "angular")
        passive_drive = (UsdPhysics.DriveAPI(task_joint, drive_name)
                         if task_joint.HasAPI(UsdPhysics.DriveAPI, drive_name) else None)
        # USD angular damping is per degree; the graph declares SI per radian.
        usd_damping = math.radians(declared_drive["damping"]) if revolute else declared_drive["damping"]
        if (task_joint.GetCustomDataByKey("blueprint:declaredDriveType") != "none"
                or declared_drive["drive_type"] != "none"
                or task_joint.HasAPI(UsdPhysics.DriveAPI, other_name)
                or (passive_drive is None and declared_drive["damping"] != 0.0)
                or (passive_drive is not None and (
                    str(passive_drive.GetTypeAttr().Get()) != "force"
                    or not _close_sequence([passive_drive.GetStiffnessAttr().Get()], [0.0])
                    or not _close_sequence([passive_drive.GetDampingAttr().Get()], [usd_damping])))):
            findings.append("replacement_target_joint_drive_mismatch")
        observed_joint = {"prim_path": str(task_joint.GetPath()), "joint_type": target_row["joint_type"],
                          "limits": [float(lower), float(upper)] if lower is not None and upper is not None else None,
                          "reset_position": reset}
    expected_joints = {row["joint_id"]: row for row in graph["joints"]}
    for prim in joints:
        expected = expected_joints.get(prim.GetName())
        joint = UsdPhysics.Joint(prim)
        targets0 = joint.GetBody0Rel().GetTargets()
        targets1 = joint.GetBody1Rel().GetTargets()
        if (expected is None or str(prim.GetPath()) != "/Asset/joints/" + prim.GetName()
                or len(targets0) != 1 or len(targets1) != 1
                or str(targets0[0]) not in body_by_path or str(targets1[0]) not in body_by_path
                or body_by_path[str(targets0[0])].GetName() != expected["parent_link_id"]
                or body_by_path[str(targets1[0])].GetName() != expected["child_link_id"]):
            findings.append("replacement_joint_graph_topology_mismatch:" + prim.GetName())
        if prim is task_joint:
            continue
        if not prim.IsA(UsdPhysics.FixedJoint):
            findings.append("replacement_non_target_joint_not_fixed:" + prim.GetName())
    if len(joints) != len(graph["joints"]):
        findings.append("replacement_joint_count_disagrees_with_graph")
    for pair in graph["collision_pairs"]:
        first, second = (body_by_path.get("/Asset/links/" + pair[name]) for name in ("link_a", "link_b"))
        if first is None or second is None:
            findings.append("replacement_collision_pair_link_missing")
            continue
        filtered = (first.HasAPI(UsdPhysics.FilteredPairsAPI)
                    and second.GetPath() in UsdPhysics.FilteredPairsAPI(first).GetFilteredPairsRel().GetTargets())
        filtered = filtered or (second.HasAPI(UsdPhysics.FilteredPairsAPI)
                                and first.GetPath() in UsdPhysics.FilteredPairsAPI(second).GetFilteredPairsRel().GetTargets())
        if filtered == pair["collision_enabled"]:
            findings.append("replacement_collision_filter_disagrees_with_graph")

    collision_prims = [prim for prim in prims if prim.HasAPI(UsdPhysics.CollisionAPI)]
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(),
                              [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy, UsdGeom.Tokens.guide],
                              useExtentsHint=False)
    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    asset_inverse = xform_cache.GetLocalToWorldTransform(root).GetInverse()
    asset_corners: list[Any] = []
    for collider in collision_prims:
        aligned = cache.ComputeWorldBound(collider).ComputeAlignedRange()
        if aligned.IsEmpty():
            continue
        lo, hi = aligned.GetMin(), aligned.GetMax()
        asset_corners.extend(
            asset_inverse.Transform(Gf.Vec3d(float(x), float(y), float(z)))
            for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2]))
    asset_bounds = ({"minimum": [min(float(c[i]) for c in asset_corners) for i in range(3)],
                     "maximum": [max(float(c[i]) for c in asset_corners) for i in range(3)]}
                    if asset_corners else None)
    asset_dimensions = ([asset_bounds["maximum"][i] - asset_bounds["minimum"][i]
                         for i in range(3)] if asset_bounds else None)
    link_rows: dict[str, dict[str, Any]] = {}
    material_bounds_ok = True
    handle_found = False
    handle_paths: list[str] = []
    handle_link_corners: list[Any] = []
    for prim in bodies:
        link_id = prim.GetName()
        part_id = link_parts.get(link_id, "")
        bounds = physics_bounds.get(part_id)
        if (str(prim.GetPath()) != "/Asset/links/" + link_id
                or prim.GetCustomDataByKey("blueprint:partId") != part_id
                or prim.GetCustomDataByKey("blueprint:semanticRole")
                != (graph_link_rows.get(link_id) or {}).get("semantic_role")
                or prim.GetCustomDataByKey("blueprint:semanticRole")
                != (plan_link_rows.get(link_id) or {}).get("semantic_role")):
            findings.append("replacement_link_identity_or_part_mismatch:" + link_id)
        own = [c for c in collision_prims if str(c.GetPath()).startswith(str(prim.GetPath()) + "/")]
        if not own:
            findings.append("replacement_link_collision_missing:" + link_id)
        corners: list[Any] = []
        materials: list[dict[str, float]] = []
        inverse = xform_cache.GetLocalToWorldTransform(prim).GetInverse()
        for collider in own:
            if UsdPhysics.CollisionAPI(collider).GetCollisionEnabledAttr().Get() is False:
                findings.append("replacement_link_collision_disabled:" + link_id)
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
                    handle_paths.append(str(collider.GetPath()))
                    handle_bound = cache.ComputeWorldBound(collider).ComputeAlignedRange()
                    if not handle_bound.IsEmpty():
                        lo, hi = handle_bound.GetMin(), handle_bound.GetMax()
                        handle_link_corners.extend(
                            inverse.Transform(Gf.Vec3d(float(x), float(y), float(z)))
                            for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2]))
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
                if row:
                    materials.append(row)
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
                              "collision_bounds_link_frame_m": link_bounds,
                              "physics_materials": materials}
    if not material_bounds_ok:
        findings.append("replacement_physics_material_bounds_invalid")
    # Every contract required part must be carried by its planned link.
    placed: dict[str, Any] = {}
    for prim in bodies:
        try:
            placed[prim.GetName()] = json.loads(str(prim.GetCustomDataByKey(REQUIRED_PARTS_ATTRIBUTE) or "{}"))
        except json.JSONDecodeError:
            placed[prim.GetName()] = {}
    findings.extend("replacement_" + code for code in required_part_findings(plan, placed))
    # The body must be hollow behind an open front on the exact collision bytes.
    cavities = plan.get("interior_cavities")
    legacy = legacy_drawer_plan(plan)  # origin/main's drawer plan: no cavity was ever planned or checked
    if legacy:
        cavities = []
    elif not isinstance(cavities, list) or not cavities:
        findings.append("replacement_body_interior_cavity_unplanned")
        cavities = []
    cavity_collision_policy = plan.get("cavity_collision_approximation")
    cavity_collision: dict[str, Any] = {}
    if cavity_collision_policy not in {None, CAVITY_COLLISION_WALL_BOXES}:
        findings.append("replacement_body_cavity_collision_policy_unsupported")
    for link_id in sorted({str(row.get("link_id")) for row in cavities if isinstance(row, Mapping)}):
        if cavity_collision_policy is not None:
            # The declared approximation, not the render mesh, is what PhysX collides with.
            link_prim = body_by_path.get("/Asset/links/" + link_id)
            own = [c for c in collision_prims
                   if link_prim is not None and str(c.GetPath()).startswith(str(link_prim.GetPath()) + "/")]
            pieces = [_collision_piece(c, link_prim, xform_cache, UsdGeom=UsdGeom, UsdPhysics=UsdPhysics)
                      for c in own]
            findings.extend("replacement_" + code for code in collision_cavity_findings(
                pieces, [row for row in cavities if row.get("link_id") == link_id]))
            cavity_collision[link_id] = {"collider_prim_paths": sorted(str(c.GetPath()) for c in own),
                                         "approximations": sorted({p["approximation"] for p in pieces})}
            continue
        shape = stage.GetPrimAtPath(f"/Asset/links/{link_id}/collision/FinalVisualShape")
        points = UsdGeom.Mesh(shape).GetPointsAttr().Get() if shape and shape.IsA(UsdGeom.Mesh) else None
        indices = UsdGeom.Mesh(shape).GetFaceVertexIndicesAttr().Get() if points is not None else None
        if not points or not indices or len(indices) % 3:
            findings.append("replacement_body_interior_cavity_mesh_missing:" + link_id)
            continue
        try:
            codes = interior_cavity_findings([[float(c) for c in p] for p in points], list(indices),
                                             [row for row in cavities if row.get("link_id") == link_id])
        except (KeyError, TypeError, ValueError, IndexError):
            codes = ["body_interior_cavity_unplanned"]
        findings.extend("replacement_" + code for code in codes)
    if task_child is not None and not handle_found:
        findings.append("replacement_handle_contact_role_missing")
    handle_bounds = None
    if handle_link_corners and all(_finite([float(c[i]) for i in range(3)]) for c in handle_link_corners):
        handle_bounds = {"minimum": [min(float(c[i]) for c in handle_link_corners) for i in range(3)],
                         "maximum": [max(float(c[i]) for c in handle_link_corners) for i in range(3)]}
    return findings, {
        "default_prim": str(root.GetPath()), "articulation_root": str(root.GetPath()),
        "task_contact": {"contact_link_id": task_child.GetName() if task_child is not None else "",
                         "handle_prim_paths": sorted(handle_paths),
                         "handle_bounds_link_frame_m": handle_bounds},
        "links": link_rows, "task_joint": observed_joint,
        "joint_prim_paths": [str(prim.GetPath()) for prim in joints],
        "collision_prim_paths": [str(prim.GetPath()) for prim in collision_prims],
        "collision_bounds_asset_frame_closed_m": asset_bounds,
        "collision_dimensions_m": asset_dimensions,
        "whole_assembly_mass_kg": sum(row["mass_kg"] for row in link_rows.values()
                                      if row["mass_kg"] is not None),
        "dependency_layer_count": len(layers), "external_asset_count": len(unpinned),
        "embedded_package_asset_count": len(external_assets) - len(unpinned),
        "unresolved_dependency_count": len(unresolved),
        **({"body_cavity_collision": {"approximation": cavity_collision_policy, "links": cavity_collision,
                                      "cavity_preserved_on_collision_geometry": True}}
           if cavity_collision_policy is not None else {}),
        **({"body_interior_cavity_check": "not_planned_legacy_drawer_configuration"} if legacy else {}),
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
    plan_links = plan.get("links") if isinstance(plan, Mapping) else None
    graph_links = graph.get("links") if isinstance(graph, Mapping) else None
    if (not isinstance(plan_links, list) or not isinstance(graph_links, list)
            or len(link_parts) != len(plan_links)
            or {row["link_id"] for row in graph_links} != set(link_parts)):
        findings.append("replacement_assembly_plan_links_invalid")
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
    if asset.is_file() and graph is not None and bounds_valid and link_parts and "replacement_assembly_plan_links_invalid" not in findings:
        usd_findings, observed = _usd_findings(asset, graph=graph, physics_bounds=physics_bounds,
                                               link_parts=link_parts, plan=plan)
        findings.extend(usd_findings)
    expected_link_paths = {link_id: row["prim_path"] for link_id, row in observed.get("links", {}).items()}
    root_link_id = next((row["link_id"] for row in graph["links"] if row["is_root"]), None) if graph else None
    if (not observed or graph_spec.get("link_prim_paths") != expected_link_paths
            or graph_spec.get("fixed_base_body_prim_path") != expected_link_paths.get(root_link_id)
            or graph_spec.get("task_link_prim_path")
            != expected_link_paths.get(next((row["child_link_id"] for row in graph["joints"]
                                            if row["role"] == "target"), None) if graph else None)):
        findings.append("replacement_graph_link_paths_disagree_with_usd")
    completion_links = completion.get("links") if isinstance(completion, Mapping) else None
    completion_joints = completion.get("joints") if isinstance(completion, Mapping) else None
    completion_rows_valid = (isinstance(completion_links, list) and isinstance(completion_joints, list)
        and all(isinstance(row, Mapping) and isinstance(row.get("joint_id"), str)
                for row in completion_joints)
        and all(isinstance(row, Mapping) and {
            "link_id", "part_id", "mass_kg", "center_of_mass_m", "diagonal_inertia_kg_m2",
            "collision_prim_paths", "collision_bounds_link_frame_m", "physics_material",
            "mass_uncertainty_interval_kg"} <= row.keys()
            and isinstance(row["link_id"], str) and isinstance(row["part_id"], str)
            and row["part_id"] in (physics_bounds or {})
            and isinstance(row["mass_kg"], (int, float)) and not isinstance(row["mass_kg"], bool)
            and _finite([row["mass_kg"]])
            and isinstance(row["collision_prim_paths"], list)
            and isinstance(row["collision_bounds_link_frame_m"], Mapping)
            and {"minimum", "maximum"} <= row["collision_bounds_link_frame_m"].keys()
            and isinstance(row["physics_material"], Mapping)
            and {"static_friction", "dynamic_friction", "restitution"} <= row["physics_material"].keys()
            and all(isinstance(row["physics_material"][key], (int, float))
                    and _finite([row["physics_material"][key]])
                    for key in ("static_friction", "dynamic_friction", "restitution"))
            and isinstance(row["mass_uncertainty_interval_kg"], list)
            and len(row["mass_uncertainty_interval_kg"]) == 2
            and all(isinstance(value, (int, float)) and not isinstance(value, bool)
                    for value in row["mass_uncertainty_interval_kg"])
            for row in completion_links))
    if (not isinstance(completion, Mapping) or completion.get("schema_version") != COMPLETION_SCHEMA_VERSION
            or completion.get("status") != "bounded_candidate_completed"
            or completion.get("candidate_prior_only") is not True or completion.get("physical_truth_claimed") is not False
            or completion.get("completion_digest") != canonical_digest(completion, digest_field="completion_digest")
            or completion.get("task_joint_prim_path") != graph_spec.get("task_joint_prim_path")
            or completion.get("task_link_prim_path") != graph_spec.get("task_link_prim_path")
            or completion.get("fixed_base_body_prim_path") != graph_spec.get("fixed_base_body_prim_path")
            or completion.get("physics_bounds") != graph_spec.get("physics_bounds")
            or completion.get("intra_assembly_collision_filtered") is not True
            or completion.get("handle_prim_paths") != graph_spec.get("handle_prim_paths")
            or completion.get("handle_grasp_point_link_m") != graph_spec.get("handle_grasp_point_link_m")
            or not completion_rows_valid
            or not observed
            or not _close_sequence(completion.get("collision_dimensions_m"),
                                   observed.get("collision_dimensions_m"))
            or not isinstance(completion.get("collision_bounds_asset_frame_closed_m"), Mapping)
            or not all(_close_sequence(
                completion["collision_bounds_asset_frame_closed_m"].get(bound),
                (observed.get("collision_bounds_asset_frame_closed_m") or {}).get(bound))
                for bound in ("minimum", "maximum"))
            or completion.get("task_joint_prim_path") != observed.get("task_joint", {}).get("prim_path")
            or {row["joint_id"] for row in completion.get("joints", []) if isinstance(row, Mapping)}
            != {row["joint_id"] for row in graph["joints"]}
            or len(completion.get("joints", [])) != len(graph["joints"])
            or any(row.get("prim_path") != "/Asset/joints/" + row["joint_id"]
                   or any(row.get(key) != expected.get(key) for key in (
                       "joint_type", "parent_link_id", "child_link_id", "role", "limits", "reset_position"))
                   for row in completion.get("joints", []) if isinstance(row, Mapping)
                   for expected in graph["joints"] if expected["joint_id"] == row["joint_id"])
            or {row["link_id"]: row for row in completion.get("links", []) if isinstance(row, Mapping)}.keys() != observed["links"].keys()
            or len(completion_links) != len(observed["links"])
            or any(row.get("prim_path") != observed["links"][row["link_id"]]["prim_path"]
                   or row.get("part_id") != observed["links"][row["link_id"]]["part_id"]
                   or not _close_authored_float32_sequence([row["mass_kg"]], [observed["links"][row["link_id"]]["mass_kg"]])
                   or not _close_authored_float32_sequence(row["center_of_mass_m"], observed["links"][row["link_id"]]["center_of_mass_m"])
                   or not _close_authored_float32_sequence(row["diagonal_inertia_kg_m2"], observed["links"][row["link_id"]]["diagonal_inertia_kg_m2"])
                   or sorted(row["collision_prim_paths"]) != sorted(observed["links"][row["link_id"]]["collision_prim_paths"])
                   or not all(_close_sequence(row["collision_bounds_link_frame_m"][bound],
                                              observed["links"][row["link_id"]]["collision_bounds_link_frame_m"][bound])
                              for bound in ("minimum", "maximum"))
                   or any(not all(abs(material[key] - row["physics_material"][key]) <= 1e-6
                                  for key in ("static_friction", "dynamic_friction", "restitution"))
                          for material in observed["links"][row["link_id"]]["physics_materials"])
                   or (not isinstance(row.get("mass_uncertainty_interval_kg"), list)
                       or len(row["mass_uncertainty_interval_kg"]) != 2
                       or not _finite(row["mass_uncertainty_interval_kg"], positive=True)
                       or not row["mass_uncertainty_interval_kg"][0] <= row["mass_kg"] <= row["mass_uncertainty_interval_kg"][1]
                       or row.get("mass_interval_exceeds_admitted_simulation_bounds") is not (
                           row["mass_uncertainty_interval_kg"][0] < physics_bounds[row["part_id"]]["mass_kg"][0]
                           or row["mass_uncertainty_interval_kg"][1] > physics_bounds[row["part_id"]]["mass_kg"][1]))
                   for row in completion.get("links", []) if isinstance(row, Mapping))):
        findings.append("replacement_physics_completion_invalid")
    if isinstance(completion, Mapping) and isinstance(plan, Mapping):
        findings.extend(_development_hypothesis_findings(
            plan=plan, completion=completion, observed=observed,
            graph=graph or {}, physics_bounds=physics_bounds or {}))
    # The grasp point the runtime will reach for must sit inside the handle the
    # qualifier actually found, in the moving link's own frame.
    grasp_point = graph_spec.get("handle_grasp_point_link_m")
    contact = (observed.get("task_contact") or {}) if observed else {}
    handle_bounds = contact.get("handle_bounds_link_frame_m")
    if (not isinstance(grasp_point, Sequence) or isinstance(grasp_point, (str, bytes))
            or len(list(grasp_point)) != 3 or not _finite(list(grasp_point))
            or sorted(graph_spec.get("handle_prim_paths") or []) != contact.get("handle_prim_paths")
            or not isinstance(handle_bounds, Mapping)
            or any(not handle_bounds["minimum"][index] - 1e-6 <= float(list(grasp_point)[index])
                   <= handle_bounds["maximum"][index] + 1e-6 for index in range(3))):
        findings.append("replacement_handle_grasp_point_outside_handle")
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
        # Published so downstream compilation binds a graph this qualifier has
        # already checked against the exact bytes, not the authoring proposal.
        "articulation_graph": graph,
        "link_prim_paths": {link_id: row["prim_path"] for link_id, row in observed["links"].items()},
        "task_joint": {"joint_id": graph["joints"][[row["role"] for row in graph["joints"]].index("target")]["joint_id"],
                       **observed["task_joint"]},
        "task_contact": {**contact, "contact_point_link_m": [float(value) for value in grasp_point]},
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
