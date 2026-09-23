"""Compose reviewed per-part Astra candidates into one articulated SimReady assembly.

A drawer or door never leaves its cabinet, so the replacement is an assembly:
an open-front carcass plus one moving part on a single passive task joint, with
every other moving part fixed closed. Each part is authored, reviewed and
measured exactly like a rigid candidate; this module only places those exact
solids in one assembly frame (+X out of the front face, Z up, origin at the
carcass centre-XY / bottom-Z), authors the articulation, filters the
intra-assembly contacts that the joints already constrain, and seals a
development-only candidate. Interiors the footage never showed are candidate
geometry and are tagged as such; nothing here grants physics authority.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from .decision_evidence_contracts import canonical_digest
from .task_object_astra_authoring import (
    AssetAuthoringError, AuthoringRequest, file_record, save_json, validate_geometry_readback,
)
from .task_object_physical_property_review import (
    PhysicalPropertyReviewInput, PhysicalPropertyReviewResult, review_physical_properties,
)
from .task_object_simready_packaging import _final_mass_consistency, _final_visual_mesh, _verified

PLAN_SCHEMA_VERSION = "articulated_assembly_plan.v1"
COMPLETION_SCHEMA_VERSION = "task_evaluation_articulated_candidate_physics_completion.v1"
AUTHORING_RESULT_SCHEMA_VERSION = "task_object_astra_articulated_authoring_result.v1"
PROVENANCE_ATTRIBUTE = "blueprint:articulatedReplacement:provenance"
OBSERVED_PROVENANCE = "observed_source_derived"
GENERATED_PROVENANCE = "generated_candidate_geometry"
TASK_CONTACT_ROLE_ATTRIBUTE = "blueprint:articulatedReplacement:taskContactRole"
HANDLE_ROLE = "handle"
TARGET_JOINT_ID = "task_part_joint"
CARCASS_LINK_ID = "carcass"
ASSET_ROOT = "/Asset"

# Object-prior construction assumptions for an unobserved interior. Every value
# is recorded in the plan; none is a measurement.
PANEL_THICKNESS_M = 0.018
FRONT_PANEL_THICKNESS_M = 0.018
HANDLE_PROTRUSION_M = 0.03
HANDLE_SECTION_M = 0.012
HANDLE_LENGTH_FRACTION_OF_FRONT = 0.6
BAY_CLEARANCE_M = 0.004
SLIDE_CLEARANCE_M = 0.013
BACK_CLEARANCE_M = 0.02
MINIMUM_RETAINED_DEPTH_M = 0.05
PASSIVE_JOINT_DAMPING_N_S_PER_M = 5.0

_ORDINALS = {
    0: ("top", "upper", "uppermost", "first", "1st", "highest"),
    1: ("middle", "center", "centre", "second", "2nd", "mid"),
    2: ("bottom", "lower", "lowest", "third", "3rd", "last"),
}
_COUNT_WORDS = {"two": 2, "2": 2, "three": 3, "3": 3, "four": 4, "4": 4}


def _resolve_bay_layout(assembly_label: str, part_label: str) -> tuple[int, int, list[str]]:
    """Resolve how many stacked bays exist and which one the task names (recorded assumptions)."""
    assumptions: list[str] = []
    label = assembly_label.lower()
    match = re.search(r"\b(two|three|four|2|3|4)[\s-]*drawer", label)
    count = _COUNT_WORDS[match.group(1)] if match else 0
    words = re.findall(r"[a-z0-9]+", part_label.lower())
    index = next((k for k, names in _ORDINALS.items() if any(w in names for w in words)), None)
    if index is None:
        raise AssetAuthoringError("articulated_task_part_position_unresolved:" + part_label)
    if count == 0:
        count = 3 if index == 1 else max(index + 1, 2)
        assumptions.append(f"bay_count_assumed_{count}_from_part_label")
    if index == 2 and count > 3:
        index = count - 1  # "bottom" of a taller stack
    if index >= count:
        raise AssetAuthoringError("articulated_task_part_outside_assembly:" + part_label)
    return count, index, assumptions


def plan_articulated_assembly(configuration: Mapping[str, Any]) -> dict[str, Any]:
    """Derive exact part envelopes, rest poses and the task joint from the stage-3 configuration.

    The assembly frame has +X out of the front face and Z up. Width/depth come
    from the estimated world envelope projected onto the estimated front normal;
    they inherit that estimate's uncertainty and are labelled so.
    """
    mechanism = configuration["mechanism"]
    if mechanism.get("joint_type") != "prismatic":
        raise AssetAuthoringError("articulated_assembly_family_unsupported:" + str(mechanism.get("joint_type")))
    envelope = configuration["metric_envelope"]
    lower, upper = envelope["minimum_xyz_m"], envelope["maximum_xyz_m"]
    extents = [float(upper[i]) - float(lower[i]) for i in range(3)]
    normal = [float(v) for v in mechanism["estimated_front_normal_world"]]
    horizontal = math.hypot(normal[0], normal[1])
    if horizontal < 1e-9:
        raise AssetAuthoringError("articulated_front_normal_invalid")
    nx, ny = normal[0] / horizontal, normal[1] / horizontal
    depth = abs(nx) * extents[0] + abs(ny) * extents[1]
    width = abs(ny) * extents[0] + abs(nx) * extents[1]
    height = extents[2]
    if min(depth, width, height) <= 6 * PANEL_THICKNESS_M:
        raise AssetAuthoringError("articulated_assembly_envelope_too_small")
    count, task_index, assumptions = _resolve_bay_layout(
        str(configuration.get("authoring_target") or ""), str(mechanism["task_part_label"]))
    t = PANEL_THICKNESS_M
    bay_height = (height - (count + 1) * t) / count
    bay_width = width - 2 * t
    box_depth = depth - t - BACK_CLEARANCE_M
    if bay_height <= 0.04 or bay_width <= 0.05 or box_depth <= MINIMUM_RETAINED_DEPTH_M + 0.02:
        raise AssetAuthoringError("articulated_bay_geometry_infeasible")
    drawer_x = HANDLE_PROTRUSION_M + FRONT_PANEL_THICKNESS_M + box_depth
    drawer_y = bay_width - BAY_CLEARANCE_M
    drawer_z = bay_height - BAY_CLEARANCE_M
    stroke = min(float(mechanism["estimated_usable_stroke_m"]), box_depth - MINIMUM_RETAINED_DEPTH_M)
    if stroke <= 0.02:
        raise AssetAuthoringError("articulated_usable_stroke_infeasible")
    handle_length = round(HANDLE_LENGTH_FRACTION_OF_FRONT * drawer_y, 4)
    handle_center = [round(drawer_x / 2 - HANDLE_SECTION_M / 2, 5), 0.0, round(drawer_z / 2, 5)]
    bays = []
    for k in range(count):
        floor_z = t + (count - 1 - k) * (bay_height + t)
        bays.append({"bay_index": k, "link_id": f"drawer_{k}", "is_task_part": k == task_index,
                     "rest_translation_m": [round(depth / 2 - drawer_x / 2 + HANDLE_PROTRUSION_M, 5), 0.0,
                                            round(floor_z + BAY_CLEARANCE_M / 2, 5)]})
    yaw = math.atan2(ny, nx)
    return {
        "schema_version": PLAN_SCHEMA_VERSION,
        "family": "stacked_drawer_cabinet",
        "assembly_frame": {"front_axis": "+X", "up_axis": "Z", "origin": "carcass_center_xy_bottom_z",
                           "world_yaw_rad_from_estimated_front_normal": yaw,
                           "estimated_front_normal_world": normal},
        "assembly_dimensions_m": {"depth_x": round(depth, 5), "width_y": round(width, 5), "height_z": round(height, 5),
                                  "authority": "estimated_envelope_projected_on_estimated_front_normal"},
        "bay_count": count, "task_bay_index": task_index,
        "construction_assumptions": [
            f"panel_thickness_m={t}", f"front_panel_thickness_m={FRONT_PANEL_THICKNESS_M}",
            f"handle_protrusion_m={HANDLE_PROTRUSION_M}", f"handle_section_m={HANDLE_SECTION_M}",
            f"equal_bay_heights={round(bay_height, 5)}", f"slide_clearance_m={SLIDE_CLEARANCE_M}",
            "interior_and_drawer_boxes_unobserved_generated_candidate_geometry",
            "carcass_treated_as_grounded_base_casters_not_modelled",
            "static_breakaway_friction_not_modelled_only_viscous_damping",
            *assumptions,
        ],
        "parts": {
            CARCASS_LINK_ID: {
                "link_role": "carcass", "dimensions_m": [round(depth, 5), round(width, 5), round(height, 5)],
                "description": (f"Open-front cabinet carcass: top, bottom, back, left and right panels {t} m thick "
                                f"plus {count - 1} horizontal dividers forming {count} equal drawer bays "
                                f"({round(bay_width, 4)} m wide x {round(bay_height, 4)} m tall x {round(depth - t, 4)} m deep). "
                                "The front (+X face) is fully open; no drawer fronts, handles or feet belong to this part."),
            },
            "drawer": {
                "link_role": "task_part", "dimensions_m": [round(drawer_x, 5), round(drawer_y, 5), round(drawer_z, 5)],
                "description": (f"One drawer: a front panel {FRONT_PANEL_THICKNESS_M} m thick spanning the full Y width and Z height "
                                f"at the +X end, a centred horizontal bar handle {handle_length} m long with a {HANDLE_SECTION_M} m "
                                f"square section protruding {HANDLE_PROTRUSION_M} m in +X from the front panel (its bar centre at "
                                f"part-frame {handle_center}), and behind the front an open-top drawer box "
                                f"{round(box_depth, 4)} m deep, {round(drawer_y - 2 * SLIDE_CLEARANCE_M, 4)} m wide, "
                                f"{round(drawer_z - 0.03, 4)} m tall with 0.012 m walls. The same solid is instanced "
                                f"for all {count} bays."),
                "handle": {"center_m": handle_center, "length_m": handle_length, "section_m": HANDLE_SECTION_M,
                           "axis": "Y", "grasp_point_link_m": handle_center},
            },
        },
        "links": [{"link_id": CARCASS_LINK_ID, "part_id": CARCASS_LINK_ID, "is_root": True, "semantic_role": "cabinet_carcass",
                   "rest_translation_m": [0.0, 0.0, 0.0]},
                  *[{"link_id": bay["link_id"], "part_id": "drawer", "is_root": False,
                     "semantic_role": "task_drawer" if bay["is_task_part"] else "fixed_drawer",
                     "rest_translation_m": bay["rest_translation_m"], "bay_index": bay["bay_index"]} for bay in bays]],
        "task_joint": {"joint_id": TARGET_JOINT_ID, "joint_type": "prismatic", "parent_link_id": CARCASS_LINK_ID,
                       "child_link_id": f"drawer_{task_index}", "axis_asset_frame": [1.0, 0.0, 0.0],
                       "limits_m": [0.0, round(stroke, 5)], "reset_position_m": 0.0,
                       "stroke_authority": mechanism.get("travel_authority", "object_prior_estimate"),
                       "drive": {"drive_type": "none", "stiffness": 0.0, "damping": PASSIVE_JOINT_DAMPING_N_S_PER_M,
                                 "maximum_force": 0.0, "implementation": "passive_force_damper"}},
        "intra_assembly_collision": "filtered_joints_constrain_mechanism",
        "lock_status": str(mechanism.get("lock_status") or "unknown"),
        "physical_measurement_proven": False,
    }


def _part_physics(*, request: AuthoringRequest, authoring_result: Mapping[str, Any],
                  physics_bounds: Mapping[str, Sequence[float]]) -> dict[str, Any]:
    """Re-verify one part exactly as the rigid packager does, without writing USD."""
    import numpy as np
    from pxr import Sdf, Usd, UsdGeom, UsdPhysics

    if authoring_result.get("request_digest") != request.request_digest:
        raise AssetAuthoringError("authoring_packaging_request_mismatch")
    if authoring_result.get("status") != "candidate_authored_pending_native_qualification":
        raise AssetAuthoringError("authoring_packaging_candidate_not_reviewed")
    if authoring_result.get("result_digest") != canonical_digest(authoring_result, digest_field="result_digest"):
        raise AssetAuthoringError("authoring_packaging_result_digest_mismatch")
    review_path = _verified(authoring_result["physical_review"])
    review = PhysicalPropertyReviewResult.model_validate_json(review_path.read_text())
    review_input = PhysicalPropertyReviewInput.model_validate_json(
        _verified(authoring_result["physical_review_input"]).read_text())
    if review_physical_properties(review_input, review.proposed).model_dump(mode="json") != review.model_dump(mode="json"):
        raise AssetAuthoringError("authoring_packaging_physics_review_not_reproducible")
    if review.accepted is None or review.blockers or review.claim_ceiling != "development_only":
        raise AssetAuthoringError("authoring_packaging_physics_not_accepted")
    if (review_input.object_id != request.object_id
            or any(abs(getattr(review_input.dimensions, axis).value - expected) > 1e-12
                   for axis, expected in zip(("x_m", "y_m", "z_m"), request.dimensions_m, strict=True))):
        raise AssetAuthoringError("authoring_packaging_physics_identity_mismatch")
    properties = review.accepted.properties
    for name in ("mass_kg", "static_friction", "dynamic_friction", "restitution"):
        value = getattr(properties, name)
        lower, upper = physics_bounds[name]
        # USD receives one mass value. The uncertainty interval describes the
        # unknown photographed object's possible mass; it is not a set of
        # masses the simulator will silently sample. Keep that interval in the
        # completion receipt, including any portion outside the admitted
        # simulation value range. Contact parameters retain their stricter
        # full-interval admission because they affect the policy interaction.
        inside = (lower <= value.value <= upper if name == "mass_kg" and value.basis == "estimated"
                  else lower <= value.interval.lower <= value.value <= value.interval.upper <= upper)
        if not inside:
            raise AssetAuthoringError("authoring_packaging_estimate_outside_admitted_bounds:" + name)
    measurement = json.loads(_verified(authoring_result["geometry_readback"]).read_text())
    validate_geometry_readback(request, measurement, review_input.appearance)
    source_path = _verified(authoring_result["asset"])
    source = Usd.Stage.Open(str(source_path))
    if source is None or source.GetDefaultPrim().GetPath() != Sdf.Path(ASSET_ROOT):
        raise AssetAuthoringError("authoring_packaging_visual_root_invalid")
    if abs(UsdGeom.GetStageMetersPerUnit(source) - 1.0) > 1e-12 or UsdGeom.GetStageUpAxis(source) != "Z":
        raise AssetAuthoringError("authoring_packaging_visual_frame_invalid")
    for prim in source.Traverse():
        if (any(schema.startswith(("Physics", "Physx")) for schema in prim.GetAppliedSchemas())
                or prim.IsA(UsdPhysics.Joint)
                or any(prop.GetName().startswith(("physics:", "physx")) for prop in prim.GetProperties())):
            raise AssetAuthoringError("authoring_packaging_unreviewed_physics_in_visual")
    mesh, mesh_receipt, geometry_sources = _final_visual_mesh(request=request, authoring_result=authoring_result, source=source)
    consistency = _final_mass_consistency(review, mesh)
    mass_kg = float(properties.mass_kg.value)
    tensor = np.asarray(mesh.moment_inertia, dtype=float) * (mass_kg / mesh.mass)
    principal, rotation = np.linalg.eigh(tensor)
    if np.any(principal <= 0) or not np.isfinite(principal).all():
        raise AssetAuthoringError("authoring_packaging_inertia_invalid")
    if np.linalg.det(rotation) < 0:
        rotation[:, 0] *= -1
    return {"mesh": mesh, "source_path": source_path, "review_path": review_path, "mesh_receipt": mesh_receipt,
            "geometry_sources": geometry_sources, "consistency": consistency, "mass_kg": mass_kg,
            "mass_interval_kg": [float(properties.mass_kg.interval.lower), float(properties.mass_kg.interval.upper)],
            "mass_basis": properties.mass_kg.basis,
            "center_of_mass_m": [float(v) for v in mesh.center_mass], "principal_inertia": [float(v) for v in principal],
            "principal_rotation": rotation, "static_friction": float(properties.static_friction.value),
            "dynamic_friction": float(properties.dynamic_friction.value), "restitution": float(properties.restitution.value),
            "collision_bounds_part_frame_m": {"minimum": [float(v) for v in mesh.bounds[0]],
                                              "maximum": [float(v) for v in mesh.bounds[1]]},
            "cad_readback": authoring_result["cad"].get("readback")}


def _quat_from_matrix(rotation):
    from pxr import Gf
    return Gf.Matrix4d(Gf.Matrix3d(*rotation.T.reshape(-1).tolist()), Gf.Vec3d(0)).ExtractRotationQuat()


def package_astra_articulated_candidate(*, requests: Mapping[str, AuthoringRequest],
                                        authoring_results: Mapping[str, Mapping[str, Any]],
                                        plan: Mapping[str, Any], output_root: Path,
                                        physics_bounds: Mapping[str, Mapping[str, Sequence[float]]]) -> dict[str, Any]:
    """Compose the reviewed parts into one articulated USDZ and seal its candidate physics.

    ``physics_bounds`` maps part id to the admitted bounds for that part's
    reviewed properties. The carcass is the dynamic root; the runtime grounds it
    with its anchor joint at staging (no link is kinematic, which PhysX refuses).
    """
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, UsdUtils

    if plan.get("schema_version") != PLAN_SCHEMA_VERSION:
        raise AssetAuthoringError("articulated_plan_schema_invalid")
    if set(requests) != set(plan["parts"]) or set(authoring_results) != set(plan["parts"]):
        raise AssetAuthoringError("articulated_part_set_mismatch")
    parts = {part_id: _part_physics(request=requests[part_id], authoring_result=authoring_results[part_id],
                                    physics_bounds=physics_bounds[part_id]) for part_id in plan["parts"]}
    for part_id, spec in plan["parts"].items():
        if [round(v, 5) for v in requests[part_id].dimensions_m] != [round(v, 5) for v in spec["dimensions_m"]]:
            raise AssetAuthoringError("articulated_part_dimensions_changed:" + part_id)
    output_root.mkdir(parents=True, exist_ok=True)
    authored = output_root / "astra_articulated_candidate.usdc"
    if authored.exists():
        raise AssetAuthoringError("authoring_packaging_output_exists")
    stage = Usd.Stage.CreateNew(str(authored))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    root = UsdGeom.Xform.Define(stage, ASSET_ROOT)
    stage.SetDefaultPrim(root.GetPrim())
    UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())
    root.GetPrim().SetCustomDataByKey("blueprint:articulatedAssemblyPlanDigest", canonical_digest(dict(plan)))
    task_joint = plan["task_joint"]
    link_paths: dict[str, str] = {}
    link_rows: dict[str, dict[str, Any]] = {}
    materials: dict[str, Any] = {}
    for link in plan["links"]:
        link_id, part_id = link["link_id"], link["part_id"]
        physics = parts[part_id]
        path = f"{ASSET_ROOT}/links/{link_id}"
        link_paths[link_id] = path
        xform = UsdGeom.Xform.Define(stage, path)
        xform.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in link["rest_translation_m"]]))
        body = UsdPhysics.RigidBodyAPI.Apply(xform.GetPrim())
        body.CreateRigidBodyEnabledAttr(True)
        body.CreateKinematicEnabledAttr(False)
        mass = UsdPhysics.MassAPI.Apply(xform.GetPrim())
        mass.CreateMassAttr(physics["mass_kg"])
        mass.CreateCenterOfMassAttr(Gf.Vec3f(*physics["center_of_mass_m"]))
        mass.CreateDiagonalInertiaAttr(Gf.Vec3f(*physics["principal_inertia"]))
        mass.CreatePrincipalAxesAttr(Gf.Quatf(_quat_from_matrix(physics["principal_rotation"])))
        xform.GetPrim().SetCustomDataByKey("blueprint:semanticRole", link["semantic_role"])
        xform.GetPrim().SetCustomDataByKey("blueprint:partId", part_id)
        provenance = OBSERVED_PROVENANCE if link["semantic_role"] == "task_drawer" else GENERATED_PROVENANCE
        # Visual appearance: the reviewed part candidate, referenced then flattened.
        visual = stage.DefinePrim(f"{path}/visual", "Xform")
        visual.GetReferences().AddReference(str(physics["source_path"]), ASSET_ROOT)
        visual.SetCustomDataByKey(PROVENANCE_ATTRIBUTE, provenance)
        if part_id not in materials:
            material = UsdShade.Material.Define(stage, f"{ASSET_ROOT}/Looks/ReviewedPhysics_{part_id}")
            contact = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
            contact.CreateStaticFrictionAttr(physics["static_friction"])
            contact.CreateDynamicFrictionAttr(physics["dynamic_friction"])
            contact.CreateRestitutionAttr(physics["restitution"])
            materials[part_id] = material
        mesh = physics["mesh"]
        collision = UsdGeom.Mesh.Define(stage, f"{path}/collision/FinalVisualShape")
        collision.CreatePointsAttr([Gf.Vec3f(*[float(c) for c in v]) for v in mesh.vertices.tolist()])
        collision.CreateFaceVertexCountsAttr([3] * len(mesh.faces))
        collision.CreateFaceVertexIndicesAttr(mesh.faces.reshape(-1).tolist())
        collision.CreateSubdivisionSchemeAttr("none")
        collision.CreatePurposeAttr("guide")
        collision.CreateVisibilityAttr("invisible")
        UsdPhysics.CollisionAPI.Apply(collision.GetPrim()).CreateCollisionEnabledAttr(True)
        UsdPhysics.MeshCollisionAPI.Apply(collision.GetPrim()).CreateApproximationAttr("convexDecomposition")
        UsdShade.MaterialBindingAPI.Apply(collision.GetPrim()).Bind(materials[part_id], UsdShade.Tokens.weakerThanDescendants, "physics")
        collision.GetPrim().SetCustomDataByKey(PROVENANCE_ATTRIBUTE, GENERATED_PROVENANCE)
        collision.GetPrim().SetCustomDataByKey("blueprint:collisionGeometryOnly", True)
        collision_paths = [str(collision.GetPath())]
        handle_grasp_point = None
        if link["semantic_role"] == "task_drawer":
            handle = plan["parts"][part_id]["handle"]
            bar = UsdGeom.Cube.Define(stage, f"{path}/collision/handle")
            bar.CreateSizeAttr(1.0)
            bar.AddTranslateOp().Set(Gf.Vec3f(*[float(v) for v in handle["center_m"]]))
            bar.AddScaleOp().Set(Gf.Vec3f(float(handle["section_m"]), float(handle["length_m"]), float(handle["section_m"])))
            bar.CreatePurposeAttr("guide")
            bar.CreateVisibilityAttr("invisible")
            UsdPhysics.CollisionAPI.Apply(bar.GetPrim()).CreateCollisionEnabledAttr(True)
            UsdShade.MaterialBindingAPI.Apply(bar.GetPrim()).Bind(materials[part_id], UsdShade.Tokens.weakerThanDescendants, "physics")
            bar.GetPrim().SetCustomDataByKey(PROVENANCE_ATTRIBUTE, OBSERVED_PROVENANCE)
            bar.GetPrim().SetCustomDataByKey(TASK_CONTACT_ROLE_ATTRIBUTE, HANDLE_ROLE)
            bar.GetPrim().SetCustomDataByKey("blueprint:collisionGeometryOnly", True)
            collision_paths.append(str(bar.GetPath()))
            handle_grasp_point = [float(v) for v in handle["grasp_point_link_m"]]
        link_rows[link_id] = {"link_id": link_id, "part_id": part_id, "prim_path": path, "semantic_role": link["semantic_role"],
                              "rest_translation_m": [float(v) for v in link["rest_translation_m"]],
                              "mass_kg": physics["mass_kg"], "mass_basis": physics["mass_basis"],
                              "mass_uncertainty_interval_kg": physics["mass_interval_kg"],
                              "mass_interval_exceeds_admitted_simulation_bounds": (
                                  physics["mass_interval_kg"][0] < physics_bounds[part_id]["mass_kg"][0]
                                  or physics["mass_interval_kg"][1] > physics_bounds[part_id]["mass_kg"][1]),
                              "center_of_mass_m": physics["center_of_mass_m"],
                              "diagonal_inertia_kg_m2": physics["principal_inertia"],
                              "collision_bounds_link_frame_m": physics["collision_bounds_part_frame_m"],
                              "collision_prim_paths": collision_paths,
                              "physics_material": {"static_friction": physics["static_friction"],
                                                   "dynamic_friction": physics["dynamic_friction"],
                                                   "restitution": physics["restitution"]},
                              **({"handle_grasp_point_link_m": handle_grasp_point} if handle_grasp_point else {})}
    joint_rows = []
    for link in plan["links"]:
        if link["is_root"]:
            continue
        is_task = link["link_id"] == task_joint["child_link_id"]
        joint_id = task_joint["joint_id"] if is_task else f"{link['link_id']}_fixed"
        path = f"{ASSET_ROOT}/joints/{joint_id}"
        if is_task:
            joint = UsdPhysics.PrismaticJoint.Define(stage, path)
            joint.CreateAxisAttr("X")
            joint.CreateLowerLimitAttr(float(task_joint["limits_m"][0]))
            joint.CreateUpperLimitAttr(float(task_joint["limits_m"][1]))
        else:
            joint = UsdPhysics.FixedJoint.Define(stage, path)
        joint.CreateBody0Rel().SetTargets([Sdf.Path(link_paths[CARCASS_LINK_ID])])
        joint.CreateBody1Rel().SetTargets([Sdf.Path(link_paths[link["link_id"]])])
        joint.CreateLocalPos0Attr(Gf.Vec3f(*[float(v) for v in link["rest_translation_m"]]))
        joint.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
        joint.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        joint.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        prim = joint.GetPrim()
        prim.SetCustomDataByKey("blueprint:jointRole", "target" if is_task else "locked")
        prim.SetCustomDataByKey("blueprint:resetPosition", 0.0)
        drive_row: dict[str, Any] = {"declared_drive_type": "none", "usd_drive_authored": False, "implementation": "none"}
        if is_task:
            prim.SetCustomDataByKey("blueprint:graphAxis", Gf.Vec3d(*task_joint["axis_asset_frame"]))
            prim.SetCustomDataByKey("blueprint:declaredDriveType", "none")
            damping = float(task_joint["drive"]["damping"])
            if damping > 0.0:
                drive = UsdPhysics.DriveAPI.Apply(prim, "linear")
                drive.CreateTypeAttr().Set("force")
                drive.CreateStiffnessAttr().Set(0.0)
                drive.CreateDampingAttr().Set(damping)
                drive.CreateTargetPositionAttr().Set(0.0)
                prim.SetCustomDataByKey("blueprint:driveImplementation", "passive_force_damper")
                drive_row = {"declared_drive_type": "none", "usd_drive_authored": True, "usd_drive_type": "force",
                             "implementation": "passive_force_damper", "stiffness": 0.0, "damping": damping}
        joint_rows.append({"joint_id": joint_id, "prim_path": path, "joint_type": "prismatic" if is_task else "fixed",
                           "parent_link_id": CARCASS_LINK_ID, "child_link_id": link["link_id"],
                           "role": "target" if is_task else "locked",
                           "limits": [float(v) for v in task_joint["limits_m"]] if is_task else [0.0, 0.0],
                           "reset_position": 0.0, "drive": drive_row})
    # The joints and limits constrain the mechanism; approximate colliders of
    # nested parts must not fight them. Robot contacts are unaffected.
    for link_id, path in link_paths.items():
        if link_id == CARCASS_LINK_ID:
            continue
        filtered = UsdPhysics.FilteredPairsAPI.Apply(stage.GetPrimAtPath(link_paths[CARCASS_LINK_ID]))
        filtered.CreateFilteredPairsRel().AddTarget(Sdf.Path(path))
        for other_id, other_path in link_paths.items():
            if other_id not in {CARCASS_LINK_ID, link_id} and other_id > link_id:
                UsdPhysics.FilteredPairsAPI.Apply(stage.GetPrimAtPath(path)).CreateFilteredPairsRel().AddTarget(Sdf.Path(other_path))
    stage.GetRootLayer().documentation = ("Blueprint articulated SimReady candidate composed from reviewed parts; "
                                          "native behaviour and physical equivalence are not qualified by authoring")
    stage.GetRootLayer().Save()
    flattened = output_root / "astra_articulated_candidate.flat.usdc"
    Usd.Stage.Open(str(authored)).Flatten().Export(str(flattened))
    asset = output_root / "astra_articulated_replacement_candidate.usdz"
    if not UsdUtils.CreateNewUsdzPackage(Sdf.AssetPath(str(flattened)), str(asset)):
        raise AssetAuthoringError("authoring_usdz_packaging_failed")
    reopened = Usd.Stage.Open(str(asset))
    if reopened is None or not reopened.GetDefaultPrim().HasAPI(UsdPhysics.ArticulationRootAPI):
        raise AssetAuthoringError("authoring_usdz_readback_failed")
    prismatic = [p for p in reopened.Traverse() if p.IsA(UsdPhysics.PrismaticJoint)]
    if len(prismatic) != 1 or len([p for p in reopened.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]) != len(link_rows):
        raise AssetAuthoringError("authoring_usdz_readback_failed")
    task_link = link_rows[task_joint["child_link_id"]]
    lows = [min(row["collision_bounds_link_frame_m"]["minimum"][i] + row["rest_translation_m"][i] for row in link_rows.values()) for i in range(3)]
    highs = [max(row["collision_bounds_link_frame_m"]["maximum"][i] + row["rest_translation_m"][i] for row in link_rows.values()) for i in range(3)]
    completion = {
        "schema_version": COMPLETION_SCHEMA_VERSION, "status": "bounded_candidate_completed",
        "asset_kind": "articulated_assembly", "candidate_prior_only": True, "physical_truth_claimed": False,
        "physics_bounds": {part_id: {k: [float(a), float(b)] for k, (a, b) in bounds.items()} for part_id, bounds in physics_bounds.items()},
        "links": [link_rows[link["link_id"]] for link in plan["links"]],
        "joints": joint_rows,
        "task_joint_prim_path": f"{ASSET_ROOT}/joints/{task_joint['joint_id']}",
        "task_link_prim_path": task_link["prim_path"],
        "fixed_base_body_prim_path": link_paths[CARCASS_LINK_ID],
        "handle_prim_paths": [p for p in task_link["collision_prim_paths"] if p.endswith("/handle")],
        "handle_grasp_point_link_m": task_link["handle_grasp_point_link_m"],
        "collision_bounds_asset_frame_closed_m": {"minimum": lows, "maximum": highs},
        "collision_dimensions_m": [highs[i] - lows[i] for i in range(3)],
        "intra_assembly_collision_filtered": True,
        "center_of_mass_authority": "constant_density_final_visual_candidate_per_link",
        "inertia_authority": "constant_density_final_visual_candidate_scaled_to_reviewed_mass_per_link",
        "collision_approximation": "convexDecomposition_per_link_plus_tagged_handle_box",
        "native_collision_cooking_qualified": False,
        "part_reviews": {part_id: {"astra_review": file_record(parts[part_id]["review_path"]),
                                   "final_visual_solid_volume_m3": float(parts[part_id]["mesh"].volume),
                                   "final_visual_mesh_sources": parts[part_id]["geometry_sources"],
                                   "final_visual_mesh_receipt_digest": parts[part_id]["mesh_receipt"]["receipt_digest"],
                                   "mass_model_final_geometry_consistency": parts[part_id]["consistency"],
                                   "original_cad_readback": parts[part_id]["cad_readback"]} for part_id in plan["parts"]},
        "illumination_authority": "site_scene_only",
    }
    completion["completion_digest"] = canonical_digest(completion, digest_field="completion_digest")
    receipt = {"asset": file_record(asset), "physics_completion": completion, "plan": dict(plan),
               "request_digests": {part_id: requests[part_id].request_digest for part_id in plan["parts"]},
               "authoring_result_digests": {part_id: authoring_results[part_id].get("result_digest") for part_id in plan["parts"]},
               "asset_origin": "captured_assembly_reconstruction", "claim_ceiling": "development_only", "native_qualified": False}
    save_json(output_root / "astra_articulated_packaging_receipt.json", receipt)
    return receipt


def articulation_graph_from_plan(plan: Mapping[str, Any], *, opening_fraction: float = 0.6) -> dict[str, Any]:
    """The task-neutral graph contract for the composed assembly (adp_articulation_graph.v1)."""
    task_joint = plan["task_joint"]
    stroke = float(task_joint["limits_m"][1])
    links = [{"link_id": row["link_id"], "is_root": row["is_root"], "semantic_role": row["semantic_role"]} for row in plan["links"]]
    joints = []
    for row in plan["links"]:
        if row["is_root"]:
            continue
        is_task = row["link_id"] == task_joint["child_link_id"]
        joints.append({"joint_id": task_joint["joint_id"] if is_task else f"{row['link_id']}_fixed",
                       "parent_link_id": CARCASS_LINK_ID, "child_link_id": row["link_id"],
                       "joint_type": "prismatic" if is_task else "fixed", "role": "target" if is_task else "locked",
                       "axis": [1.0, 0.0, 0.0] if is_task else [0.0, 0.0, 0.0],
                       "limits": [0.0, stroke] if is_task else [0.0, 0.0], "reset_position": 0.0,
                       "reset_tolerance": 0.005,
                       "drive": {"drive_type": "none", "stiffness": 0.0,
                                 "damping": float(task_joint["drive"]["damping"]) if is_task else 0.0, "maximum_force": 0.0}})
    link_ids = [row["link_id"] for row in links]
    pairs = [{"link_a": a, "link_b": b, "collision_enabled": False}
             for i, a in enumerate(sorted(link_ids)) for b in sorted(link_ids)[i + 1:]]
    return {"schema_version": "adp_articulation_graph.v1", "links": links, "joints": joints, "collision_pairs": pairs,
            "success_predicate": {"combination": "all",
                                  "joint_intervals": {task_joint["joint_id"]: [round(opening_fraction * stroke, 5), stroke]}}}


__all__ = [
    "AUTHORING_RESULT_SCHEMA_VERSION", "COMPLETION_SCHEMA_VERSION", "PLAN_SCHEMA_VERSION",
    "PROVENANCE_ATTRIBUTE", "TASK_CONTACT_ROLE_ATTRIBUTE", "TARGET_JOINT_ID",
    "articulation_graph_from_plan", "package_astra_articulated_candidate", "plan_articulated_assembly",
]
