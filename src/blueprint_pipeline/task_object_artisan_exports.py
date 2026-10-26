"""Reopen ARTISAN's exact manifests and USD; enforce retained source constraints.

Output adaptations are namespace/provenance and declared material/passive-joint
properties. Geometry, collisions, joint frames and inertias are upstream.
Earlier physics reports remain bound to the original exports, never the overlay.
"""
from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

from .task_object_authoring_contracts import AuthoringContractError, require, file_record, read_json, verified_file, source_joint_constraints


def upstream_exports(terminal: dict, workspace: Path) -> dict:
    require(terminal.get("phase") == "DONE" and terminal.get("registered") is True
            and bool(terminal.get("canonical_record_id")) and terminal.get("returncode") == 0,
            "artisan_not_done_and_registered")
    paths = {key: Path(value) for key, value in terminal["paths"].items()}
    require(all(path.resolve().is_relative_to(workspace.resolve()) for path in paths.values()),
            "artisan_upstream_storage_escape")
    manifest_path = paths["exports"] / "export_manifest.json"
    manifest = read_json(manifest_path)
    candidate = read_json(paths["candidate_manifest"])
    require(manifest.get("schema_version") == "release-candidate-v7" and manifest.get("status") == "accepted"
            and manifest.get("candidate_id") == terminal.get("candidate_id")
            and manifest.get("source_manifest_sha256") == file_record(paths["candidate_manifest"])["sha256"][7:]
            and candidate.get("candidate_id") == manifest["candidate_id"], "artisan_export_candidate_binding_invalid")
    require(file_record(paths["model"])["sha256"][7:] == manifest["model_sha256"]
            and file_record(paths["blueprint"])["sha256"][7:] == manifest["blueprint_sha256"],
            "artisan_reviewed_program_or_blueprint_changed")
    records = {}
    for name, row in manifest["artifacts"].items():
        relative = Path(row["path"])
        require(not relative.is_absolute() and ".." not in relative.parts, "artisan_export_path_invalid")
        path = verified_file({"path": str(paths["exports"] / relative),
                              "sha256": "sha256:" + row["sha256"], "size_bytes": row["size_bytes"]},
                             root=paths["exports"])
        records[name] = file_record(path)
    usd = records.get("object.usd")
    require(usd is not None and "usd" in manifest["formats"], "artisan_reviewed_usd_export_missing")
    report_relative = Path(manifest["physics_report_path"])
    require(not report_relative.is_absolute() and ".." not in report_relative.parts,
            "artisan_export_physics_report_path_invalid")
    report_path = paths["exports"] / report_relative
    report = read_json(report_path)
    report_hash = file_record(report_path)["sha256"][7:]
    require(report_hash == manifest["physics_report_sha256"]
            == terminal.get("reviewed_physics_report_sha256")
            and file_record(paths["physics_report"])["sha256"][7:] == report_hash
            and report.get("candidate_id") == manifest["candidate_id"], "artisan_reviewed_physics_report_changed")
    return {"manifest": manifest, "export_manifest": file_record(manifest_path),
            "candidate_manifest": file_record(paths["candidate_manifest"]),
            "program": file_record(paths["model"]), "blueprint": file_record(paths["blueprint"]),
            "physics_report": file_record(report_path), "exports": records, "usd": usd}


def validated_sdk_exports(*, manifest_path: Path, model_path: Path, blueprint_path: Path) -> dict:
    """Read a real SDK validation export without inventing CLI registration.

    The SDK's accepted candidate is usable for independent static/native checks;
    it is not an ARTISAN controller DONE record or a registered library asset.
    """
    manifest = read_json(manifest_path)
    require(manifest.get("schema_version") == "release-candidate-v7"
            and manifest.get("status") == "accepted" and bool(manifest.get("candidate_id"))
            and "usd" in manifest.get("formats", []), "artisan_sdk_candidate_not_accepted")
    require(file_record(model_path)["sha256"][7:] == manifest.get("model_sha256")
            and file_record(blueprint_path)["sha256"][7:] == manifest.get("blueprint_sha256"),
            "artisan_reviewed_program_or_blueprint_changed")
    root, records = manifest_path.parent, {}
    for name, row in manifest.get("artifacts", {}).items():
        relative = Path(row["path"])
        require(not relative.is_absolute() and ".." not in relative.parts, "artisan_export_path_invalid")
        path = verified_file({"path": str(root / relative), "sha256": "sha256:" + row["sha256"],
                              "size_bytes": row["size_bytes"]}, root=root)
        records[name] = file_record(path)
    require("object.usd" in records, "artisan_reviewed_usd_export_missing")
    relative = Path(manifest["physics_report_path"])
    require(not relative.is_absolute() and ".." not in relative.parts,
            "artisan_export_physics_report_path_invalid")
    report_path = root / relative
    require(report_path.resolve().is_relative_to(root.resolve()), "artisan_upstream_storage_escape")
    report = read_json(report_path)
    require(file_record(report_path)["sha256"][7:] == manifest.get("physics_report_sha256")
            and report.get("candidate_id") == manifest["candidate_id"]
            and report.get("model_sha256") == manifest["model_sha256"]
            and bool(manifest.get("build_id")) and report.get("build_id") == manifest["build_id"]
            and report.get("status") == "pass" and report.get("blocking_findings") == [],
            "artisan_sdk_physics_report_invalid")
    for report_key, manifest_key in (("algorithm_version", "validation_algorithm_version"),
                                     ("validation_policy_sha256", "validation_policy_sha256"),
                                     ("simulator_collision_digest", "simulator_collision_digest"),
                                     ("accepted_fcl_pose_sha256", "accepted_fcl_pose_sha256")):
        require(bool(manifest.get(manifest_key)) and report.get(report_key) == manifest[manifest_key],
                "artisan_sdk_validation_binding_changed")
    # Both roles refer to the actual SDK manifest. There is no separate CLI
    # export manifest, canonical registry record or synthetic terminal status.
    return {"manifest": manifest, "export_manifest": file_record(manifest_path),
            "candidate_manifest": file_record(manifest_path), "program": file_record(model_path),
            "blueprint": file_record(blueprint_path), "physics_report": file_record(report_path),
            "exports": records, "usd": records["object.usd"]}


def _finite(values):
    return all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in values)


def _bounds(prims, *, cache, frame=None):
    from pxr import Gf
    points = []
    for prim in prims:
        box = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        require(not box.IsEmpty(), "artisan_empty_geometry")
        low, high = box.GetMin(), box.GetMax()
        for xyz in itertools.product(*[(low[i], high[i]) for i in range(3)]):
            point = Gf.Vec3d(*xyz)
            if frame is not None:
                point = frame.Transform(point)
            points.append(list(point))
    require(points and all(_finite(point) for point in points), "artisan_nonfinite_geometry")
    return {"minimum": [min(p[i] for p in points) for i in range(3)],
            "maximum": [max(p[i] for p in points) for i in range(3)]}


def _owner(prim, physics):
    current = prim
    while current.IsValid() and not current.IsPseudoRoot():
        if current.HasAPI(physics.RigidBodyAPI):
            return current
        current = current.GetParent()
    return None


def _measured(configuration, name):
    review = configuration.get("physical_review_input") or {}
    row = (configuration.get("measured_physical_properties") or review.get("measured") or {}).get(name)
    if row is None:
        return None
    require(row.get("basis") == "measured" and bool(row.get("evidence_ids")),
            "artisan_measured_property_provenance_missing:" + name)
    return float(row["value"])


def _fixed_descendants(root, joints):
    members = {root}
    while True:
        added = {row["child_link_id"] for row in joints
                 if row["joint_type"] == "fixed" and row["parent_link_id"] in members} - members
        if not added:
            return members
        members.update(added)


def _mass_groups(links, joints, configuration, task):
    """A split door's fixed features share its mass range, not the carcass range.

    Required fixed interior parts remain separate mass admissions even though
    they are connected to the carcass. Every link contributes exactly once.
    """
    parents = {row["child_link_id"]: row for row in joints}
    interior = {row["part_id"] for row in configuration.get("required_parts") or []
                if row.get("role") == "fixed_interior"}
    groups = {}
    for name, link in links.items():
        anchor = name
        while anchor in parents and anchor not in interior:
            edge = parents[anchor]
            if edge["joint_type"] != "fixed":
                break
            anchor = edge["parent_link_id"]
        group = groups.setdefault(anchor, {"link_ids": [], "mass_kg": 0.0})
        group["link_ids"].append(name)
        group["mass_kg"] += link["mass_kg"]
    for anchor, group in groups.items():
        key = "task_part_mass_kg_bounds" if task and anchor == task["child_link_id"] else "mass_kg_bounds"
        if anchor in interior and "fixed_part_mass_kg_bounds" in configuration["required_output"]:
            key = "fixed_part_mass_kg_bounds"
        interval = configuration["required_output"][key]
        require(float(interval[0]) <= group["mass_kg"] <= float(interval[1]), "artisan_mass_outside_admitted_bounds")
        group.update(bounds_key=key, admitted_bounds_kg=list(interval))
    return groups


def _material_overlay(stage, configuration, *, colliders):
    from pxr import UsdPhysics, UsdShade
    required = configuration["required_output"]
    values, provenance = {}, {}
    for name in ("static_friction", "dynamic_friction", "restitution"):
        measured = _measured(configuration, name)
        interval = required[name + "_bounds"]
        value = measured if measured is not None else (float(interval[0]) + float(interval[1])) / 2
        require(float(interval[0]) <= value <= float(interval[1]), "artisan_measured_property_outside_admitted_range")
        values[name] = value
        provenance[name] = {"basis": "measured" if measured is not None else "estimated",
                            "value": value, "admitted_range": interval,
                            "rationale": "retained measurement" if measured is not None else "admitted range midpoint candidate parameter"}
    require(values["static_friction"] >= values["dynamic_friction"], "artisan_friction_bounds_infeasible")
    material = UsdShade.Material.Define(stage, "/Asset/BlueprintSourceProperties")
    api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    api.CreateStaticFrictionAttr(values["static_friction"])
    api.CreateDynamicFrictionAttr(values["dynamic_friction"])
    api.CreateRestitutionAttr(values["restitution"])
    for prim in colliders:
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(material, materialPurpose="physics")
    return provenance


def _flatten_nested_physics_links(stage):
    """Keep body-local joint frames and asset placement valid in PhysX.

    A reset stack on each nested body would discard the scene's placement too.
    Instead, move nested bodies beside their ancestors and preserve their exact
    asset-space transforms. NamespaceEditor also updates joint relationships.
    """
    from pxr import Usd, UsdGeom, UsdPhysics, UsdShade
    cache = UsdGeom.XformCache()
    root_inverse = cache.GetLocalToWorldTransform(stage.GetDefaultPrim()).GetInverse()
    nested = []
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI) and _owner(prim.GetParent(), UsdPhysics) is not None:
            nested.append((str(prim.GetPath()), cache.GetLocalToWorldTransform(prim) * root_inverse))
    if nested:
        UsdGeom.Scope.Define(stage, '/Asset/BlueprintPhysicsLinks')
    moved = []
    for before, matrix in sorted(nested, key=lambda row: row[0].count('/'), reverse=True):
        prim = stage.GetPrimAtPath(before)
        # Make inherited material bindings explicit before leaving that parent.
        for purpose in ('', 'preview', 'full', 'physics'):
            material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial(materialPurpose=purpose)
            if material:
                UsdShade.MaterialBindingAPI.Apply(prim).Bind(material, materialPurpose=purpose)
        after = '/Asset/BlueprintPhysicsLinks/' + prim.GetName()
        require(not stage.GetPrimAtPath(after).IsValid(), 'artisan_physics_link_namespace_collision')
        editor = Usd.NamespaceEditor(stage)
        require(editor.MovePrimAtPath(before, after) and editor.CanApplyEdits()
                and editor.ApplyEdits(), 'artisan_physics_link_namespace_refused')
        body = UsdGeom.Xformable(stage.GetPrimAtPath(after))
        body.ClearXformOpOrder()
        body.AddTransformOp().Set(matrix)
        moved.append({'before': before, 'after': after})
    return moved


def _display_color_surface(stage):
    """Translate the SDK's authored display colours into a native render material."""
    from pxr import Sdf, UsdGeom, UsdPhysics, UsdShade
    material = None
    count = 0
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Gprim) or prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        existing, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
        if existing and existing.GetSurfaceOutput().HasConnectedSource():
            continue
        if not UsdGeom.Gprim(prim).GetDisplayColorAttr().Get():
            continue
        if material is None:
            material = UsdShade.Material.Define(stage, '/Asset/BlueprintDisplayColor')
            reader = UsdShade.Shader.Define(stage, '/Asset/BlueprintDisplayColor/reader')
            reader.CreateIdAttr('UsdPrimvarReader_float3')
            reader.CreateInput('varname', Sdf.ValueTypeNames.Token).Set('displayColor')
            reader.CreateOutput('result', Sdf.ValueTypeNames.Float3)
            shader = UsdShade.Shader.Define(stage, '/Asset/BlueprintDisplayColor/surface')
            shader.CreateIdAttr('UsdPreviewSurface')
            shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).ConnectToSource(reader.ConnectableAPI(), 'result')
            material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), 'surface')
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)
        count += 1
    return {'basis': 'upstream_display_color', 'bound_visual_count': count,
            'material_path': str(material.GetPath()) if material else None}


def _declared_joint_dynamics(configuration):
    from .articulation_graph_contract import validate_joint_gravity_response

    mechanism = configuration.get('mechanism') or {}
    policy = mechanism.get('passive_dynamics') or {}
    if not policy:
        return None
    require(policy.get('task_joint_drive', 'none') == 'none', 'artisan_passive_joint_drive_not_admitted')
    result = {'joint_id': mechanism.get('target_joint_id'), 'joint_type': mechanism.get('joint_type')}
    require(result['joint_id'] and result['joint_type'] in {'prismatic', 'revolute'},
            'artisan_passive_joint_identity_invalid')
    for key in ('joint_friction_bounds', 'joint_damping_bounds'):
        bounds = policy.get(key)
        require(isinstance(bounds, list) and len(bounds) == 2 and _finite(bounds)
                and 0 <= bounds[0] <= bounds[1], 'artisan_' + key + '_invalid')
        result[key] = [float(value) for value in bounds]
    result.update(static_effort=sum(result['joint_friction_bounds']) / 2,
                  dynamic_effort=result['joint_friction_bounds'][0],
                  damping=sum(result['joint_damping_bounds']) / 2,
                  units='N_m' if result['joint_type'] == 'revolute' else 'N',
                  basis='estimated_unobserved_joint_resistance_prior', physical_measurement_proven=False)
    try:
        response = validate_joint_gravity_response(policy.get('gravity_response'))
    except ValueError as exc:
        raise AuthoringContractError('artisan_joint_gravity_response_invalid') from exc
    if response is not None:
        require(response['mode'] != 'counterbalanced' or result['joint_type'] == 'revolute',
                'artisan_joint_gravity_response_unsupported')
        result['gravity_response'] = response
    return result


def _assembly_mounting_overlay(stage, request, *, author):
    """Carry an explicit capture-reviewed fixture constraint into the export."""
    from pxr import UsdPhysics
    from .task_object_required_motions import assembly_mounting

    declaration = (request.get("construction_constraints") or {}).get("assembly_mounting")
    if declaration is None:
        return None
    # Production source_frames are file records; frame identities live in the
    # retained handoff manifest. Bind cited identities to admitted image hashes.
    admitted = {row["sha256"] for row in request.get("source_frames", []) if isinstance(row.get("sha256"), str)}
    known_frames = [row["frame_id"] for row in (request.get("source_frame_manifest") or {}).get("frames", [])
                    if (row.get("image") or {}).get("sha256", (row.get("image") or {}).get("digest")) in admitted]
    try:
        mounting = assembly_mounting(declaration, source_frame_ids=known_frames)
    except ValueError as exc:
        raise AuthoringContractError(str(exc)) from exc
    bodies = {str(p.GetPath()): p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)}
    children = {str(path) for p in stage.Traverse() if p.IsA(UsdPhysics.Joint)
                for path in UsdPhysics.Joint(p).GetBody1Rel().GetTargets()}
    roots = set(bodies) - children
    require(len(roots) == 1, "artisan_mounting_root_ambiguous")
    base = bodies[roots.pop()]
    body = UsdPhysics.RigidBodyAPI(base)
    if author and mounting["mode"] == "fixed_in_scene":
        body.CreateKinematicEnabledAttr(True)
    if mounting["mode"] != "unknown":
        require(bool(body.GetKinematicEnabledAttr().Get()) == (mounting["mode"] == "fixed_in_scene"),
                "artisan_mounting_constraint_changed")
    root = stage.GetDefaultPrim()
    if author:
        root.SetCustomDataByKey("blueprint:assemblyMounting", json.dumps(mounting, sort_keys=True))
    else:
        require(root.GetCustomDataByKey("blueprint:assemblyMounting") == json.dumps(mounting, sort_keys=True),
                "artisan_mounting_annotation_changed")
    return {**mounting, "root_body_prim_path": str(base.GetPath()),
            "base_kinematic_enabled": bool(body.GetKinematicEnabledAttr().Get()),
            "basis": "capture_review_construction_candidate", "physical_mount_measured": False}


def _passive_joint_dynamics(stage, configuration, *, author):
    """Carry the admitted, unactuated resistance into imported and read-back USD."""
    from pxr import Sdf, UsdPhysics
    from .articulation_graph_contract import GRAVITY_RESPONSE_CUSTOM_DATA_KEY
    policy = _declared_joint_dynamics(configuration)
    if policy is None:
        return None
    matches = [prim for prim in stage.Traverse() if prim.IsA(UsdPhysics.Joint)
               and prim.GetCustomDataByKey('blueprint:jointId') == policy['joint_id']]
    require(len(matches) == 1, 'artisan_passive_joint_unresolved')
    prim = matches[0]
    response = policy.get('gravity_response')
    if response is not None:
        encoded = json.dumps(response, sort_keys=True)
        existing = prim.GetCustomDataByKey(GRAVITY_RESPONSE_CUSTOM_DATA_KEY)
        if author:
            require(existing is None or existing == encoded, 'artisan_joint_gravity_response_conflict')
            prim.SetCustomDataByKey(GRAVITY_RESPONSE_CUSTOM_DATA_KEY, encoded)
        require(prim.GetCustomDataByKey(GRAVITY_RESPONSE_CUSTOM_DATA_KEY) == encoded,
                'artisan_joint_gravity_response_missing_or_changed')
    angular = policy['joint_type'] == 'revolute'
    require(prim.IsA(UsdPhysics.RevoluteJoint if angular else UsdPhysics.PrismaticJoint),
            'artisan_passive_joint_type_changed')
    axis = 'angular' if angular else 'linear'
    for name in ('linear', 'angular'):
        if prim.HasAPI(UsdPhysics.DriveAPI, name):
            existing = UsdPhysics.DriveAPI(prim, name)
            require(float(existing.GetStiffnessAttr().Get() or 0) == 0
                    and float(existing.GetTargetVelocityAttr().Get() or 0) == 0,
                    'artisan_task_or_passive_drive_not_admitted')
    damping = math.radians(policy['damping']) if angular else policy['damping']
    attributes = {f'physxJointAxis:{axis}:staticFrictionEffort': policy['static_effort'],
                  f'physxJointAxis:{axis}:dynamicFrictionEffort': policy['dynamic_effort']}
    if author:
        prim.AddAppliedSchema(f'PhysxJointAxisAPI:{axis}')
        for name, value in attributes.items():
            prim.CreateAttribute(name, Sdf.ValueTypeNames.Float, custom=False).Set(value)
        drive = UsdPhysics.DriveAPI.Apply(prim, axis)
        drive.CreateTypeAttr().Set('force')
        drive.CreateStiffnessAttr().Set(0)
        drive.CreateDampingAttr().Set(damping)
        drive.CreateTargetPositionAttr().Set(0)
        drive.CreateTargetVelocityAttr().Set(0)
        prim.SetCustomDataByKey('blueprint:declaredDriveType', 'none')
        prim.SetCustomDataByKey('blueprint:driveImplementation',
                               'passive_torque_damper' if angular else 'passive_force_damper')
        prim.SetCustomDataByKey('blueprint:passiveFrictionBasis', policy['basis'])
    drive = UsdPhysics.DriveAPI(prim, axis)
    schemas = prim.GetMetadata('apiSchemas')
    require(schemas is not None and f'PhysxJointAxisAPI:{axis}' in schemas.GetAddedOrExplicitItems()
            and prim.HasAPI(UsdPhysics.DriveAPI, axis) and drive.GetTypeAttr().Get() == 'force'
            and math.isclose(float(drive.GetDampingAttr().Get()), damping, rel_tol=1e-6, abs_tol=1e-7)
            and (damping == 0 or float(drive.GetMaxForceAttr().Get()) > 0)
            and all(prim.GetAttribute(name).Get() is not None
                    and math.isclose(float(prim.GetAttribute(name).Get()), value, rel_tol=1e-6, abs_tol=1e-7)
                    for name, value in attributes.items()), 'artisan_declared_joint_dynamics_missing_or_changed')
    return {**policy, 'prim_path': str(prim.GetPath()), 'usd_axis': axis, 'usd_damping': damping,
            'motor_stiffness': 0.0, 'target_velocity': 0.0}


def adapt_usd(source: Path, destination: Path, request: dict, *, annotations_path: Path | None = None) -> dict:
    from pxr import Sdf, Usd, UsdGeom, UsdPhysics, UsdUtils
    stage = Usd.Stage.Open(str(source), load=Usd.Stage.LoadAll)
    require(stage is not None and stage.GetDefaultPrim().IsValid(), "artisan_usd_unreadable")
    # Flatten only layers; no geometry rescaling, inertial or joint conversion.
    working = destination.with_suffix(".usda")
    stage.Flatten().Export(str(working))
    stage = Usd.Stage.Open(str(working))
    original_root = str(stage.GetDefaultPrim().GetPath())
    if original_root != "/Asset":
        editor = Usd.NamespaceEditor(stage)
        require(editor.MovePrimAtPath(original_root, "/Asset") and editor.CanApplyEdits(), "artisan_root_namespace_refused")
        require(editor.ApplyEdits(), "artisan_root_namespace_refused")
        stage.SetDefaultPrim(stage.GetPrimAtPath("/Asset"))
    root = stage.GetDefaultPrim()
    root.SetCustomDataByKey("blueprint:sourceObjectIdentity", request["source_object_identity"])
    root.SetCustomDataByKey("blueprint:constructionRequestDigest", request["request_digest"])
    root.SetCustomDataByKey("blueprint:generatedGeometryIsObservedTruth", False)
    annotations = None
    if request["configuration"]["schema_version"].startswith("articulated_"):
        require(annotations_path is not None, "artisan_cavity_annotations_required")
        annotations = read_json(annotations_path)
        root.SetCustomDataByKey("blueprint:sourceConstraintAnnotations", json.dumps(annotations, allow_nan=False, sort_keys=True))
    physics_namespace = _flatten_nested_physics_links(stage)
    appearance = _display_color_surface(stage)
    colliders = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.CollisionAPI)]
    properties = _material_overlay(stage, request["configuration"], colliders=colliders)
    mechanism = request["configuration"].get("mechanism") or {}
    joint_original_paths = [str(p.GetPath()) for p in stage.Traverse() if p.IsA(UsdPhysics.Joint)]
    if joint_original_paths:
        UsdGeom.Xform.Define(stage, "/Asset/joints")
    for joint_path in joint_original_paths:
        prim = stage.GetPrimAtPath(joint_path)
        if prim.IsA(UsdPhysics.Joint):
            name = prim.GetName()
            joint_id = name.removeprefix("joint_")
            new_path = "/Asset/joints/" + joint_id
            if joint_path != new_path:
                editor = Usd.NamespaceEditor(stage)
                require(editor.MovePrimAtPath(joint_path, new_path) and editor.CanApplyEdits()
                        and editor.ApplyEdits(), "artisan_joint_namespace_refused")
                prim = stage.GetPrimAtPath(new_path)
            prim.SetCustomDataByKey("blueprint:jointId", joint_id)
            prim.SetCustomDataByKey("blueprint:jointRole", "locked" if prim.IsA(UsdPhysics.FixedJoint) else
                "target" if joint_id == mechanism.get("target_joint_id") else "dependent" if
                any(str(r.GetName()).startswith("physxMimicJoint:") for r in prim.GetRelationships()) else "passive")
            # The exported zero pose is the reviewed reset; a nonzero required
            # reset must be authored upstream rather than fabricated here.
            reset = float(mechanism.get("closed_reset_position", 0)) if joint_id == mechanism.get("target_joint_id") else 0.0
            require(reset == 0.0, "artisan_nonzero_reset_requires_upstream_pose")
            prim.SetCustomDataByKey("blueprint:resetPosition", reset)
            joint = UsdPhysics.Joint(prim)
            if not prim.IsA(UsdPhysics.FixedJoint):
                from pxr import Gf
                typed = UsdPhysics.RevoluteJoint(prim) if prim.IsA(UsdPhysics.RevoluteJoint) else UsdPhysics.PrismaticJoint(prim)
                token = str(typed.GetAxisAttr().Get()).upper()
                axis = list(Gf.Rotation(Gf.Quatd(joint.GetLocalRot0Attr().Get())).TransformDir(
                    Gf.Vec3d(*{"X": [1, 0, 0], "Y": [0, 1, 0], "Z": [0, 0, 1]}[token])))
                parent = list(joint.GetBody0Rel().GetTargets())
                require(len(parent) == 1, "artisan_joint_parent_missing")
                axis = list(UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(parent[0])).TransformDir(Gf.Vec3d(*axis)))
                prim.SetCustomDataByKey("blueprint:graphAxis", Gf.Vec3d(*axis))
    joint_dynamics = _passive_joint_dynamics(stage, request['configuration'], author=True)
    mounting = _assembly_mounting_overlay(stage, request, author=True)
    stage.GetRootLayer().Save()
    require(UsdUtils.CreateNewUsdzPackage(Sdf.AssetPath(str(working)), str(destination)), "artisan_usdz_packaging_failed")
    # Checks apply to the final package bytes after every overlay/namespace edit.
    observed = validate_final_usd(destination, request)
    return {"schema_version": "artisan_export_adaptation.v1", "original_usd": file_record(source),
            "final_usd": file_record(destination), "namespace_root_before": original_root,
            "namespace_root_after": "/Asset", "geometry_scale_applied": False,
            "physics_namespace_overlay": physics_namespace, "appearance_overlay": appearance,
            "material_overlay": properties, "joint_dynamics_overlay": joint_dynamics, "final_readback": observed,
            **({"assembly_mounting_overlay": mounting} if mounting is not None else {}),
            "candidate_annotations": file_record(annotations_path) if annotations is not None else None,
            "upstream_physics_report_certifies_overlay": False,
            "native_import_qualification_required": True}


def validate_final_usd(path: Path, request: dict) -> dict:
    from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade, UsdUtils, Sdf
    from .articulation_graph_contract import validate_articulation_graph
    from .task_evaluation_scene_configuration_static_qualification import _portable_package_findings, _is_exact_package_member
    require(not _portable_package_findings(path), "artisan_usdz_package_invalid")
    layers, assets, unresolved = UsdUtils.ComputeAllDependencies(Sdf.AssetPath(str(path)))
    require(layers and not unresolved and all(_is_exact_package_member(p, package_identifier=str(path)) for p in assets)
            and all(str(layer.identifier) == str(path) or _is_exact_package_member(layer.identifier, package_identifier=str(path)) for layer in layers),
            "artisan_usdz_external_dependency")
    stage = Usd.Stage.Open(str(path), load=Usd.Stage.LoadAll)
    require(stage is not None, "artisan_usd_unreadable")
    root = stage.GetDefaultPrim()
    require(str(root.GetPath()) == "/Asset" and UsdGeom.GetStageMetersPerUnit(stage) == 1.0
            and UsdGeom.GetStageUpAxis(stage) == "Z", "artisan_units_or_origin_invalid")
    require(UsdGeom.Xformable(root).GetLocalTransformation() == Gf.Matrix4d(1), "artisan_root_registration_transform_not_local_identity")
    require(root.GetCustomDataByKey("blueprint:sourceObjectIdentity") == request["source_object_identity"]
            and root.GetCustomDataByKey("blueprint:constructionRequestDigest") == request["request_digest"],
            "artisan_source_identity_changed")
    prims = list(stage.Traverse())
    bodies = [p for p in prims if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    require(all(_owner(p.GetParent(), UsdPhysics) is None for p in bodies),
            'artisan_native_nested_rigid_body')
    colliders = [p for p in prims if p.HasAPI(UsdPhysics.CollisionAPI)]
    visuals = [p for p in prims if p.IsA(UsdGeom.Gprim) and p not in colliders]
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render,
                                                   UsdGeom.Tokens.proxy, UsdGeom.Tokens.guide], useExtentsHint=False)
    transforms = UsdGeom.XformCache()
    require(bodies and colliders and visuals, "artisan_visual_collision_or_body_missing")
    bounds = _bounds(colliders, cache=cache)
    dimensions = [bounds["maximum"][i] - bounds["minimum"][i] for i in range(3)]
    tolerance = request["maximum_export_error_m"]
    require(all(abs(actual - expected) <= tolerance for actual, expected in zip(dimensions, request["dimensions_m"], strict=True)),
            "artisan_dimension_constraint_violation")
    require(abs(bounds["minimum"][2]) <= tolerance and all(abs((bounds["minimum"][i] + bounds["maximum"][i]) / 2) <= tolerance for i in (0, 1)),
            "artisan_origin_constraint_violation")
    visual_bounds = _bounds(visuals, cache=cache)
    require(all(abs(visual_bounds[bound][i] - bounds[bound][i]) <= tolerance for bound in ("minimum", "maximum") for i in range(3)),
            "artisan_visual_collision_dimension_drift")
    for prim in prims:
        if prim.IsA(UsdGeom.Mesh):
            mesh = UsdGeom.Mesh(prim)
            points = list(mesh.GetPointsAttr().Get() or [])
            indices = list(mesh.GetFaceVertexIndicesAttr().Get() or [])
            counts = list(mesh.GetFaceVertexCountsAttr().Get() or [])
            require(points and counts and all(n >= 3 for n in counts) and sum(counts) == len(indices)
                    and all(0 <= i < len(points) for i in indices) and all(_finite(list(p)) for p in points),
                    "artisan_mesh_topology_invalid")
        if prim in colliders and prim.IsA(UsdGeom.Mesh):
            require(prim.HasAPI(UsdPhysics.MeshCollisionAPI) and str(UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get())
                    in {"convexHull", "convexDecomposition"}, "artisan_uncooked_dynamic_collision")
    configuration = request["configuration"]
    _passive_joint_dynamics(stage, configuration, author=False)
    mounting = _assembly_mounting_overlay(stage, request, author=False)
    joints = [p for p in prims if p.IsA(UsdPhysics.Joint)]
    articulated = configuration["schema_version"] == "articulated_replacement_authoring_configuration.v1"
    articulation_roots = [p for p in prims if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
    require(not articulated or articulation_roots == [root], "artisan_articulation_root_invalid")
    require((articulated and len(bodies) >= 2) or (not articulated and len(bodies) == 1 and not joints
            and not any(p.HasAPI(UsdPhysics.ArticulationRootAPI) for p in prims)), "artisan_asset_kind_changed")
    body_by_path = {str(p.GetPath()): p.GetName() for p in bodies}
    graph_joints, task = [], None
    joint_paths = {}
    mechanism = configuration.get("mechanism") or {}
    for prim in joints:
        joint = UsdPhysics.Joint(prim)
        parent, child = list(joint.GetBody0Rel().GetTargets()), list(joint.GetBody1Rel().GetTargets())
        require(len(parent) == len(child) == 1 and str(parent[0]) in body_by_path and str(child[0]) in body_by_path,
                "artisan_joint_body_or_world_anchor_invalid")
        fixed = prim.IsA(UsdPhysics.FixedJoint)
        kind = "fixed" if fixed else "revolute" if prim.IsA(UsdPhysics.RevoluteJoint) else "prismatic" if prim.IsA(UsdPhysics.PrismaticJoint) else "unsupported"
        require(kind != "unsupported", "artisan_joint_type_unsupported")
        typed = None if fixed else (UsdPhysics.RevoluteJoint(prim) if kind == "revolute" else UsdPhysics.PrismaticJoint(prim))
        axis = [0., 0., 0.] if fixed else list(transforms.GetLocalToWorldTransform(stage.GetPrimAtPath(parent[0])).TransformDir(
            Gf.Rotation(Gf.Quatd(joint.GetLocalRot0Attr().Get())).TransformDir(
            Gf.Vec3d(*{"X": [1, 0, 0], "Y": [0, 1, 0], "Z": [0, 0, 1]}[str(typed.GetAxisAttr().Get()).upper()]))))
        limits = [0., 0.] if fixed else [float(typed.GetLowerLimitAttr().Get()), float(typed.GetUpperLimitAttr().Get())]
        if kind == "revolute":
            limits = [math.radians(v) for v in limits]
        identifier = prim.GetCustomDataByKey("blueprint:jointId") or prim.GetName().removeprefix("joint_")
        role = "locked" if fixed else "target" if identifier == mechanism.get("target_joint_id") else "passive"
        drives = [UsdPhysics.DriveAPI(prim, name) for name in ("linear", "angular") if prim.HasAPI(UsdPhysics.DriveAPI, name)]
        require(all(float(d.GetStiffnessAttr().Get() or 0) == 0
                    and float(d.GetTargetVelocityAttr().Get() or 0) == 0 for d in drives), "artisan_task_or_passive_drive_not_admitted")
        row = {"joint_id": identifier, "parent_link_id": body_by_path[str(parent[0])], "child_link_id": body_by_path[str(child[0])],
               "joint_type": kind, "role": role, "axis": axis, "limits": limits, "reset_position": 0.,
               "reset_tolerance": 0.001, "drive": {"drive_type": "none", "stiffness": 0., "damping": 0., "maximum_force": 0.}, "dependency": None}
        damping = sum(float(d.GetDampingAttr().Get() or 0) for d in drives)
        require(math.isfinite(damping) and damping >= 0, "artisan_joint_damping_invalid")
        row["drive"]["damping"] = damping * (180 / math.pi if kind == "revolute" else 1)
        mimics = [r for r in prim.GetRelationships() if str(r.GetName()).startswith("physxMimicJoint:")
                  and str(r.GetName()).endswith(":referenceJoint")]
        require(len(mimics) <= 1 and not (role == "target" and mimics), "artisan_mimic_joint_unsupported")
        if mimics:
            reference = list(mimics[0].GetTargets())
            require(len(reference) == 1, "artisan_mimic_joint_reference_invalid")
            driver = stage.GetPrimAtPath(reference[0])
            require(driver and driver.IsA(UsdPhysics.Joint), "artisan_mimic_joint_reference_invalid")
            prefix = str(mimics[0].GetName()).removesuffix("referenceJoint")
            gearing, offset = prim.GetAttribute(prefix + "gearing").Get(), prim.GetAttribute(prefix + "offset").Get()
            require(_finite([gearing, offset]) and driver.IsA(type(typed)), "artisan_mixed_unit_coupling_unsupported")
            row.update(role="dependent", dependency={"driver_joint_id": driver.GetCustomDataByKey("blueprint:jointId") or driver.GetName(),
                "multiplier": -float(gearing), "offset": -float(offset) * (math.pi / 180 if kind == "revolute" else 1), "tolerance": 0.001})
        require(prim.GetCustomDataByKey("blueprint:jointRole") == row["role"], "artisan_joint_role_changed")
        graph_joints.append(row)
        joint_paths[identifier] = str(prim.GetPath())
        if role == "target":
            task = {**row, "prim_path": str(prim.GetPath()), "usd_axis": str(typed.GetAxisAttr().Get()).upper()}
            require(kind == mechanism["joint_type"] and all(abs(a - b) <= 1e-6 for a, b in zip(limits, mechanism["joint_limits"], strict=True)),
                    "artisan_joint_travel_constraint_violation")
            constraints = request.get("joint_constraints") or source_joint_constraints(configuration)
            expected_axis = constraints["physical_joint_axis_asset_frame"]
            require(expected_axis is None or all(abs(a - b) <= 1e-6 for a, b in zip(axis, expected_axis, strict=True)), "artisan_joint_axis_constraint_violation")
            pivot = list(transforms.GetLocalToWorldTransform(stage.GetPrimAtPath(parent[0])).Transform(
                Gf.Vec3d(*joint.GetLocalPos0Attr().Get())))
            expected_pivot = constraints["pivot_asset_m"]
            require(expected_pivot is None or all(abs(a - b) <= tolerance for a, b in zip(pivot, expected_pivot, strict=True)),
                    "artisan_measured_joint_pivot_changed")
            task["pivot_asset_m"] = pivot
            task["source_joint_constraints"] = constraints
    require(not articulated or task is not None, "artisan_required_task_joint_missing")
    children = {r["child_link_id"] for r in graph_joints}
    intervals = {}
    if task:
        limits = task["limits"]
        # A task freeze can supply a stricter useful-travel interval. Otherwise
        # the candidate's last quarter is a labeled construction test, not task truth.
        intervals[task["joint_id"]] = mechanism.get("success_joint_interval") or [limits[0] + 0.75 * (limits[1] - limits[0]), limits[1]]
    filtered = {str(body.GetPath()): {str(p) for p in UsdPhysics.FilteredPairsAPI(body).GetFilteredPairsRel().GetTargets()}
                for body in bodies if body.HasAPI(UsdPhysics.FilteredPairsAPI)}
    pairs = [{"link_a": a.GetName(), "link_b": b.GetName(),
              "collision_enabled": str(b.GetPath()) not in filtered.get(str(a.GetPath()), set())
                    and str(a.GetPath()) not in filtered.get(str(b.GetPath()), set())}
             for a, b in itertools.combinations(bodies, 2)]
    graph = {"schema_version": "adp_articulation_graph.v1", "links": [{"link_id": p.GetName(), "is_root": p.GetName() not in children,
             "semantic_role": "generated_part_matching_retained_source"} for p in bodies], "joints": graph_joints,
             "collision_pairs": pairs, "success_predicate": {"combination": "all", "joint_intervals": intervals}}
    graph = validate_articulation_graph(graph, require_target_joint=articulated)
    link_rows = {}
    for body in bodies:
        name = body.GetName()
        owned = [p for p in colliders if _owner(p, UsdPhysics) == body]
        require(owned and body.HasAPI(UsdPhysics.MassAPI), "artisan_link_collision_or_mass_missing")
        api = UsdPhysics.MassAPI(body)
        mass = float(api.GetMassAttr().Get())
        center = list(api.GetCenterOfMassAttr().Get())
        inertia = list(api.GetDiagonalInertiaAttr().Get())
        principal = api.GetPrincipalAxesAttr().Get()
        principal_axes = [float(principal.GetReal()), *map(float, principal.GetImaginary())]
        require(_finite([mass, *center, *inertia]) and mass > 0 and all(v > 0 for v in inertia)
                and _finite(principal_axes) and math.isclose(sum(v*v for v in principal_axes), 1, abs_tol=1e-5)
                and (UsdPhysics.RigidBodyAPI(body).GetKinematicEnabledAttr().Get() is not True
                     or mounting is not None and mounting["mode"] == "fixed_in_scene"
                     and mounting["root_body_prim_path"] == str(body.GetPath()))
                and UsdPhysics.RigidBodyAPI(body).GetRigidBodyEnabledAttr().Get() is not False, "artisan_physics_properties_invalid")
        measured_mass = _measured(configuration, "mass_kg") if not articulated else None
        require(measured_mass is None or math.isclose(mass, measured_mass, rel_tol=1e-6), "artisan_measured_mass_changed")
        local_bounds = _bounds(owned, cache=cache, frame=transforms.GetLocalToWorldTransform(body).GetInverse())
        require(all(local_bounds["minimum"][i] <= center[i] <= local_bounds["maximum"][i] for i in range(3)), "artisan_com_outside_collision_bounds")
        materials = []
        for collider in owned:
            require(UsdPhysics.CollisionAPI(collider).GetCollisionEnabledAttr().Get() is not False,
                    "artisan_required_collision_disabled")
            material, _ = UsdShade.MaterialBindingAPI(collider).ComputeBoundMaterial(materialPurpose="physics")
            require(material and material.GetPrim().HasAPI(UsdPhysics.MaterialAPI), "artisan_collision_material_unbound")
            physics = UsdPhysics.MaterialAPI(material.GetPrim())
            values = {"static_friction": physics.GetStaticFrictionAttr().Get(), "dynamic_friction": physics.GetDynamicFrictionAttr().Get(), "restitution": physics.GetRestitutionAttr().Get()}
            require(_finite(list(values.values())), "artisan_material_nonfinite")
            for property_name, value in values.items():
                interval = configuration["required_output"][property_name + "_bounds"]
                measured = _measured(configuration, property_name)
                require(float(interval[0]) - 1e-7 <= value <= float(interval[1]) + 1e-7
                        and (measured is None or math.isclose(value, measured, abs_tol=1e-7)), "artisan_source_property_constraint_violation")
            materials.append({"path": str(material.GetPath()), **values})
        link_rows[name] = {"link_id": name, "part_id": name, "prim_path": str(body.GetPath()), "mass_kg": mass,
            "center_of_mass_m": center, "diagonal_inertia_kg_m2": inertia,
            "principal_axes_wxyz": principal_axes,
            "collision_prim_paths": sorted(str(p.GetPath()) for p in owned),
            "physics_materials": list({row["path"]: row for row in materials}.values()),
            "collision_bounds_link_frame_m": local_bounds}
    mass_groups = _mass_groups(link_rows, graph_joints, configuration, task)
    handles = []
    whole_mass = sum(row["mass_kg"] for row in link_rows.values())
    measured_assembly_mass = _measured(configuration, "mass_kg") if articulated else None
    require(measured_assembly_mass is None or math.isclose(whole_mass, measured_assembly_mass, rel_tol=1e-6),
            "artisan_measured_assembly_mass_changed")
    if articulated:
        moving_links = _fixed_descendants(task["child_link_id"], graph_joints)
        handle_uids = {p.GetCustomDataByKey("blueprint:visualUid") for p in visuals if p.GetCustomDataByKey("blueprint:visualName") == "task_handle"}
        handles = [p for p in colliders if p.GetCustomDataByKey("blueprint:sourceVisualUid") in handle_uids
                   and _owner(p, UsdPhysics).GetName() in moving_links]
        require(handle_uids and handles, "artisan_useful_task_handle_missing")
    required_parts = configuration.get("required_part_ids") or [row["part_id"] for row in configuration.get("required_parts") or []]
    names = set(link_rows) | {p.GetCustomDataByKey("blueprint:visualName") for p in visuals}
    require(set(required_parts) <= names, "artisan_required_observed_part_missing")
    from .task_object_required_motions import motion_graph_readback
    requirements = (request.get("joint_constraints") or source_joint_constraints(configuration)).get("required_motions", [])
    part_links = {}
    for part in {part for motion in requirements for part in motion["part_ids"]}:
        owners = ({part} if part in link_rows else {_owner(p, UsdPhysics).GetName() for p in visuals
            if p.GetCustomDataByKey("blueprint:visualName") == part})
        require(len(owners) == 1, "artisan_required_motion_part_owner_ambiguous:" + part)
        part_links[part] = next(iter(owners))
    try:
        motion_readback = motion_graph_readback(requirements, graph, part_links=part_links)
    except ValueError as exc:
        raise AuthoringContractError(str(exc)) from exc
    handle_point = None
    if handles:
        moving = stage.GetPrimAtPath(link_rows[task["child_link_id"]]["prim_path"])
        handle_bounds = _bounds(handles, cache=cache, frame=transforms.GetLocalToWorldTransform(moving).GetInverse())
        handle_point = [(handle_bounds["minimum"][i] + handle_bounds["maximum"][i]) / 2 for i in range(3)]
    cavity_readback = []
    if articulated:
        from .task_object_artisan_source_validation import validate_cavities
        annotations = json.loads(root.GetCustomDataByKey("blueprint:sourceConstraintAnnotations") or "{}")
        cavity_readback = validate_cavities(stage, annotations, request, bodies, colliders, transforms, _owner)
    return {"schema_version": "artisan_final_source_constraints.v1", "asset_sha256": file_record(path)["sha256"],
        "dimensions_m": dimensions, "bounds_m": bounds, "dimension_authority": request["dimension_authority"],
        "source_uncertainty_m": request["dimension_uncertainty_m"], "construction_tolerance_m": tolerance,
        "links": link_rows, "articulation_graph": graph, "task_joint": task,
        "whole_assembly_mass_kg": whole_mass,
        "mass_group_readback": mass_groups,
        "joint_prim_paths": joint_paths, "handle_prim_paths": sorted(str(p.GetPath()) for p in handles),
        "handle_grasp_point_link_m": handle_point,
        "interior_cavity_readback": cavity_readback,
        "required_motion_readback": motion_readback,
        **({"assembly_mounting": mounting} if mounting is not None else {}),
        "physics_authority_granted": False, "physical_equivalence_proven": False}
