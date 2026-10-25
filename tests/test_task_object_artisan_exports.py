"""Real USD bytes exercise ADP-009 adapter contracts, not a live ARTISAN run."""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import zipfile

import pytest
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, UsdUtils

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_object_authoring_contracts import AuthoringContractError, file_record
from blueprint_pipeline.task_object_artisan_exports import adapt_usd, upstream_exports, validate_final_usd


def construction_request(*, articulated=False):
    configuration = {"schema_version": "articulated_replacement_authoring_configuration.v1" if articulated else
                     "rigid_replacement_authoring_configuration.v1",
        "required_output": {"mass_kg_bounds": [0.1, 30], "task_part_mass_kg_bounds": [0.1, 10],
            "static_friction_bounds": [0.3, 0.6], "dynamic_friction_bounds": [0.1, 0.3], "restitution_bounds": [0, 0.1]}}
    if articulated:
        configuration["mechanism"] = {"target_joint_id": "door_hinge", "joint_type": "revolute",
            "joint_limits": [0, 1.5], "opening_axis_asset_frame": [1, 0, 0], "joint_axis_asset_frame": [1, 0, 0], "closed_reset_position": 0}
    value = {"schema_version": "task_object_construction_request.v1", "configuration": configuration,
             "source_object_identity": "retained-object", "dimensions_m": [1, 1, 1],
             "maximum_export_error_m": 0.001, "dimension_authority": "estimated",
             "dimension_uncertainty_m": [0.05] * 3}
    value["request_digest"] = canonical_digest(value, digest_field="request_digest")
    return value


def usd_fixture(root: Path, *, articulated=False, solid=False):
    root.mkdir(exist_ok=True)
    source = root / "object.usda"
    stage = Usd.Stage.CreateNew(str(source))
    model = UsdGeom.Xform.Define(stage, "/UpstreamObject")
    stage.SetDefaultPrim(model.GetPrim())
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    UsdGeom.SetStageUpAxis(stage, "Z")
    def body(name):
        prim = UsdGeom.Xform.Define(stage, "/UpstreamObject/" + name).GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(prim)
        mass = UsdPhysics.MassAPI.Apply(prim)
        mass.CreateMassAttr(2)
        mass.CreateCenterOfMassAttr(Gf.Vec3f(0.45 if name == "door" else 0, 0, 0.5))
        mass.CreateDiagonalInertiaAttr(Gf.Vec3f(0.2, 0.2, 0.2))
        mass.CreatePrincipalAxesAttr(Gf.Quatf(1))
        return prim
    def box(parent, name, center, size, *, handle=False):
        for collision in (True, False):
            cube = UsdGeom.Cube.Define(stage, str(parent.GetPath()) + "/" + ("col_" if collision else "vis_") + name)
            cube.CreateSizeAttr(1)
            cube.AddTranslateOp().Set(Gf.Vec3d(*center))
            cube.AddScaleOp().Set(Gf.Vec3d(*size))
            if collision:
                UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
                if handle:
                    cube.GetPrim().SetCustomDataByKey("blueprint:sourceVisualUid", "handle-uid")
            elif handle:
                cube.GetPrim().SetCustomDataByKey("blueprint:visualName", "task_handle")
                cube.GetPrim().SetCustomDataByKey("blueprint:visualUid", "handle-uid")
    base = body("body")
    if not articulated or solid:
        box(base, "solid", [0, 0, 0.5], [1, 1, 1])
    else:
        box(base, "back", [-0.45, 0, 0.5], [0.1, 1, 1])
        box(base, "left", [0, -0.45, 0.5], [1, 0.1, 1])
        box(base, "right", [0, 0.45, 0.5], [1, 0.1, 1])
        box(base, "bottom", [0, 0, 0.05], [1, 1, 0.1])
        box(base, "top", [0, 0, 0.95], [1, 1, 0.1])
    if articulated:
        UsdPhysics.ArticulationRootAPI.Apply(model.GetPrim())
        door = body("door")
        box(door, "door", [0.45, 0, 0.5], [0.1, 1, 1])
        box(door, "handle", [0.46, 0, 0.7], [0.05, 0.3, 0.05], handle=True)
        joint = UsdPhysics.RevoluteJoint.Define(stage, "/UpstreamObject/door/joint_door_hinge")
        joint.CreateBody0Rel().SetTargets([base.GetPath()])
        joint.CreateBody1Rel().SetTargets([door.GetPath()])
        joint.CreateAxisAttr("Z")
        joint.CreateLocalRot0Attr(Gf.Quatf(Gf.Rotation(Gf.Vec3d(0, 1, 0), 90).GetQuat()))
        joint.CreateLocalRot1Attr(Gf.Quatf(Gf.Rotation(Gf.Vec3d(0, 1, 0), 90).GetQuat()))
        joint.CreateLowerLimitAttr(0)
        joint.CreateUpperLimitAttr(1.5 * 180 / 3.141592653589793)
    stage.GetRootLayer().Save()
    return source


def annotations(root, request):
    path = root / "annotations.json"
    path.write_text(json.dumps({"schema_version": "artisan_candidate_annotations.v1", "request_digest": request["request_digest"],
        "interior_cavities": [{"cavity_id": "interior", "link_id": "body", "opening_center_link_m": [0.5, 0, 0.5],
            "opening_width_m": 0.6, "opening_height_m": 0.6, "inner_depth_m": 0.8, "basis": "generated_candidate"}]}))
    return path


def mounted_fixture(root, mode="fixed_in_scene"):
    from PIL import Image
    request = construction_request(articulated=True)
    root.mkdir(exist_ok=True)
    image = root / "installation.png"
    Image.new("RGB", (4, 4)).save(image)
    request["source_frames"] = [file_record(image)]
    image_reference = {"digest": file_record(image)["sha256"], "size_bytes": image.stat().st_size,
                       "path": image.name}
    request["source_frame_manifest"] = {"frames": [{"frame_id": "installation-view", "image": image_reference}]}
    request["source_frame_manifest_digest"] = canonical_digest(request["source_frame_manifest"])
    request["construction_constraints"] = {"assembly_mounting": {
        "mode": mode, "source_frame_ids": ["installation-view"],
        "reason": "Capture-reviewed construction candidate."}}
    request["request_digest"] = canonical_digest(request, digest_field="request_digest")
    source = usd_fixture(root, articulated=True)
    digest = file_record(source)["sha256"]
    destination = root / "mounted.usdz"
    result = adapt_usd(source, destination, request, annotations_path=annotations(root, request))
    assert file_record(source)["sha256"] == digest
    return destination, request, result


@pytest.mark.parametrize("mode", ["fixed_in_scene", "free", "unknown"])
def test_mounting_only_constrains_the_declared_base_and_preserves_geometry(tmp_path, mode):
    destination, request, result = mounted_fixture(tmp_path, mode)
    observed = result["final_readback"]
    mounting = observed["assembly_mounting"]
    assert mounting["mode"] == mode and mounting["physical_mount_measured"] is False
    assert mounting["base_kinematic_enabled"] is (mode == "fixed_in_scene")
    assert observed["dimensions_m"] == pytest.approx([1, 1, 1])
    assert observed["articulation_graph"]["joints"][0]["limits"] == pytest.approx([0, 1.5])
    stage = Usd.Stage.Open(str(destination))
    bodies = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    assert len(bodies) == 2
    assert [str(p.GetPath()) for p in bodies if UsdPhysics.RigidBodyAPI(p).GetKinematicEnabledAttr().Get()] == (
        [mounting["root_body_prim_path"]] if mode == "fixed_in_scene" else [])
    changed = deepcopy(request)
    changed["construction_constraints"]["assembly_mounting"]["mode"] = "free" if mode == "fixed_in_scene" else "fixed_in_scene"
    with pytest.raises(AuthoringContractError, match="mounting_constraint_changed|mounting_annotation_changed"):
        validate_final_usd(destination, changed)
    if mode != "unknown":
        changed = deepcopy(request)
        changed["source_frame_manifest"]["frames"][0]["image"]["digest"] = "sha256:" + "0" * 64
        with pytest.raises(AuthoringContractError, match="assembly_mounting_invalid"):
            validate_final_usd(destination, changed)


def test_rigid_final_bytes_preserve_dimensions_identity_and_do_not_invent_joint(tmp_path):
    request = construction_request()
    result = adapt_usd(usd_fixture(tmp_path), tmp_path / "final.usdz", request)
    observed = result["final_readback"]
    assert observed["dimensions_m"] == pytest.approx([1, 1, 1])
    assert observed["articulation_graph"]["joints"] == []
    assert observed["dimension_authority"] == "estimated"
    assert observed["source_uncertainty_m"] == [0.05] * 3
    assert result["upstream_physics_report_certifies_overlay"] is False
    changed = deepcopy(request)
    changed["dimensions_m"][0] += 0.002
    with pytest.raises(AuthoringContractError, match="dimension_constraint_violation"):
        validate_final_usd(tmp_path / "final.usdz", changed)


@pytest.mark.parametrize('kind', ['revolute', 'prismatic'])
def test_declared_passive_joint_dynamics_survive_import_and_packaged_readback(tmp_path, kind):
    request = construction_request(articulated=True)
    mechanism = request['configuration']['mechanism']
    mechanism.update(joint_type=kind, passive_dynamics={'task_joint_drive': 'none',
        'joint_friction_bounds': [1., 15.], 'joint_damping_bounds': [1., 30.]})
    source = usd_fixture(tmp_path, articulated=True)
    if kind == 'prismatic':
        stage = Usd.Stage.Open(str(source))
        joint = UsdPhysics.PrismaticJoint.Define(stage, '/UpstreamObject/door/joint_door_hinge')
        joint.CreateAxisAttr('X')
        joint.CreateLocalRot0Attr(Gf.Quatf(1))
        joint.CreateLocalRot1Attr(Gf.Quatf(1))
        joint.CreateUpperLimitAttr(.4)
        mechanism['joint_limits'] = [0., .4]
        stage.GetRootLayer().Save()
    request['request_digest'] = canonical_digest(request, digest_field='request_digest')
    final = tmp_path / 'final.usdz'
    result = adapt_usd(source, final, request, annotations_path=annotations(tmp_path, request))
    stage = Usd.Stage.Open(str(final))
    prim = stage.GetPrimAtPath('/Asset/joints/door_hinge')
    axis = 'angular' if kind == 'revolute' else 'linear'
    assert prim.GetAttribute(f'physxJointAxis:{axis}:staticFrictionEffort').Get() == 8.
    assert prim.GetAttribute(f'physxJointAxis:{axis}:dynamicFrictionEffort').Get() == 1.
    drive = UsdPhysics.DriveAPI(prim, axis)
    assert drive.GetStiffnessAttr().Get() == 0.
    assert drive.GetTargetVelocityAttr().Get() == 0.
    assert drive.GetDampingAttr().Get() == pytest.approx(math.radians(15.5) if kind == 'revolute' else 15.5)
    assert result['joint_dynamics_overlay']['physical_measurement_proven'] is False
    assert result['joint_dynamics_overlay']['units'] == ('N_m' if kind == 'revolute' else 'N')
    assert result['final_readback']['dimensions_m'] == pytest.approx([1., 1., 1.])
    assert result['final_readback']['task_joint']['drive']['damping'] == pytest.approx(15.5)
    # A later packaging step must not silently discard the configured resistance.
    working = Usd.Stage.Open(str(final.with_suffix('.usda')))
    working.GetPrimAtPath('/Asset/joints/door_hinge').GetAttribute(
        f'physxJointAxis:{axis}:staticFrictionEffort').Set(0.)
    working.GetRootLayer().Save()
    altered = tmp_path / 'altered.usdz'
    assert UsdUtils.CreateNewUsdzPackage(Sdf.AssetPath(str(final.with_suffix('.usda'))), str(altered))
    with pytest.raises(AuthoringContractError, match='declared_joint_dynamics_missing_or_changed'):
        validate_final_usd(altered, request)


def test_passive_joint_overlay_refuses_an_upstream_motor_instead_of_erasing_it(tmp_path):
    request = construction_request(articulated=True)
    request['configuration']['mechanism']['passive_dynamics'] = {'task_joint_drive': 'none',
        'joint_friction_bounds': [1., 15.], 'joint_damping_bounds': [1., 30.]}
    request['request_digest'] = canonical_digest(request, digest_field='request_digest')
    source = usd_fixture(tmp_path, articulated=True)
    stage = Usd.Stage.Open(str(source))
    drive = UsdPhysics.DriveAPI.Apply(stage.GetPrimAtPath('/UpstreamObject/door/joint_door_hinge'), 'angular')
    drive.CreateStiffnessAttr().Set(10.)
    stage.GetRootLayer().Save()
    with pytest.raises(AuthoringContractError, match='task_or_passive_drive_not_admitted'):
        adapt_usd(source, tmp_path / 'final.usdz', request, annotations_path=annotations(tmp_path, request))


def test_real_hollow_articulated_bytes_keep_joint_frame_handle_and_native_contract(tmp_path):
    from blueprint_pipeline.task_evaluation_scene_configuration_native_import_driver import _receipt_mechanism, _articulated_joint_observation
    request = construction_request(articulated=True)
    final = tmp_path / "final.usdz"
    result = adapt_usd(usd_fixture(tmp_path, articulated=True), final, request, annotations_path=annotations(tmp_path, request))
    observed = result["final_readback"]
    assert observed["task_joint"]["axis"] == pytest.approx([1, 0, 0], abs=1e-6)
    assert observed["task_joint"]["usd_axis"] == "Z"
    assert observed["interior_cavity_readback"][0]["basis"] == "generated_candidate"
    assert observed["handle_prim_paths"]
    static = {"backend": "artisan_v1", "articulation_graph": observed["articulation_graph"],
              "observed_structure": observed, "task_joint": observed["task_joint"]}
    assert _receipt_mechanism(static)["task_joint_usd_axis"] == "Z"
    joint = _articulated_joint_observation(stage=Usd.Stage.Open(str(final)), usd_physics=UsdPhysics, root_path="/Asset")
    assert joint["task_joint_name"] == "door_hinge"
    assert joint["task_joint_limits"] == pytest.approx([0, 1.5])


@pytest.mark.parametrize("requested_type,passes", [("revolute", True), ("prismatic", False)])
def test_real_usd_must_supply_the_requested_motion_not_only_a_named_part(tmp_path, requested_type, passes):
    request = construction_request(articulated=True)
    request["configuration"].update(required_part_ids=["door"], construction_constraints={
        "required_motions": [{"motion_id": "operate_panel", "part_ids": ["door"], "joint_type": requested_type,
                              "task_actions": ["open"], "source_frame_ids": []}]})
    request["request_digest"] = canonical_digest(request, digest_field="request_digest")
    source, final = usd_fixture(tmp_path, articulated=True), tmp_path / "final.usdz"
    if passes:
        result = adapt_usd(source, final, request, annotations_path=annotations(tmp_path, request))
        readback = result["final_readback"]["required_motion_readback"]
        assert readback["motions"][0]["joint_id"] == "door_hinge"
        assert readback["task_success_proven"] is False
    else:
        with pytest.raises(AuthoringContractError, match="required_task_motion_missing:operate_panel"):
            adapt_usd(source, final, request, annotations_path=annotations(tmp_path, request))


def test_solid_collider_cannot_claim_a_hollow_interior(tmp_path):
    request = construction_request(articulated=True)
    with pytest.raises(AuthoringContractError, match="collision_cavity_unproven"):
        adapt_usd(usd_fixture(tmp_path, articulated=True, solid=True), tmp_path / "final.usdz", request,
                  annotations_path=annotations(tmp_path, request))


def test_fixed_door_handle_has_assembly_mass_and_grasp_point_in_door_frame(tmp_path):
    request = construction_request(articulated=True)
    request["configuration"]["required_output"].update(mass_kg_bounds=[1.9, 2.1], task_part_mass_kg_bounds=[2.25, 2.35])
    request["request_digest"] = canonical_digest(request, digest_field="request_digest")
    source = usd_fixture(tmp_path, articulated=True)
    stage = Usd.Stage.Open(str(source))
    child = UsdGeom.Xform.Define(stage, "/UpstreamObject/door/handle_component").GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(child)
    mass = UsdPhysics.MassAPI.Apply(child)
    mass.CreateMassAttr(.3)
    mass.CreateCenterOfMassAttr(Gf.Vec3f(.46, 0, .7))
    mass.CreateDiagonalInertiaAttr(Gf.Vec3f(.001, .001, .001))
    mass.CreatePrincipalAxesAttr(Gf.Quatf(1))
    for name in ("vis_handle", "col_handle"):
        editor = Usd.NamespaceEditor(stage)
        assert editor.MovePrimAtPath("/UpstreamObject/door/" + name, str(child.GetPath()) + "/" + name)
        assert editor.ApplyEdits()
    joint = UsdPhysics.FixedJoint.Define(stage, "/UpstreamObject/door/joint_handle_mount")
    joint.CreateBody0Rel().SetTargets(["/UpstreamObject/door"])
    joint.CreateBody1Rel().SetTargets([child.GetPath()])
    stage.GetRootLayer().Save()
    final = tmp_path / "final.usdz"
    result = adapt_usd(source, final, request, annotations_path=annotations(tmp_path, request))["final_readback"]
    assert result["mass_group_readback"]["door"]["mass_kg"] == pytest.approx(2.3)
    assert set(result["mass_group_readback"]["door"]["link_ids"]) == {"door", "handle_component"}
    assert result["handle_grasp_point_link_m"] == pytest.approx([.46, 0, .7])
    # The native consumer must accept the same fixed handle assembly as the
    # static exporter, including a separate rigid body for the grasp feature.
    from blueprint_pipeline.task_evaluation_scene_configuration_native_import_driver import _articulated_structure_observation, _IMPORTED_ROOT
    receipt = {"backend": "artisan_v1", "articulation_graph": result["articulation_graph"],
        "observed_structure": {**result, "joint_prim_paths": list(result["joint_prim_paths"].values())},
        "task_joint": result["task_joint"], "task_contact": {"handle_prim_paths": result["handle_prim_paths"]}}
    imported = Usd.Stage.CreateInMemory()
    imported.DefinePrim(_IMPORTED_ROOT, "Xform").GetReferences().AddReference(str(final), "/Asset")
    readback = _articulated_structure_observation(stage=imported, usd_physics=UsdPhysics, static_receipt=receipt)
    assert set(readback["link_physics_readback"]) == {"body", "door", "handle_component"}
    assert readback["handle_prim_paths"] == [_IMPORTED_ROOT + p[len("/Asset"):] for p in result["handle_prim_paths"]]
    # Exercise the controller's real sealed-capsule preflight too: this exact
    # readback must happen before renting the native host, not first on a GPU.
    from blueprint_pipeline.task_evaluation_scene_configuration_native_continuation import preflight_native_prefix_composition
    receipt["asset_kind"] = "articulated_assembly"
    receipt["result_digest"] = canonical_digest(receipt, digest_field="result_digest")
    capsule = tmp_path / "completed-prefix.zip"
    blobs = [("statically_qualified_replacement_asset", "replacement.usdz", final.read_bytes()),
             ("static_qualification_receipt", "static.json", json.dumps(receipt).encode())]
    records = [{"role": role, "path": "/retained/stages/stage-4/adapter/" + name,
                "digest": "sha256:" + hashlib.sha256(data).hexdigest(), "size_bytes": len(data)}
               for role, name, data in blobs]
    with zipfile.ZipFile(capsule, "w") as archive:
        archive.writestr("stages/stage-4/completed_stage_checkpoint.json", json.dumps({"stage_result": {"output_artifacts": records}}))
        for role, name, data in blobs:
            archive.writestr("stages/stage-4/adapter/" + name, data)
    replay = preflight_native_prefix_composition(capsule, {"output_root": "/retained", "transport_digest": "sha256:" + "a" * 64})
    assert replay["status"] == "passed" and replay["link_count"] == 3
    assert replay["physics_executed"] is False
    wrong = deepcopy(receipt)
    wrong["task_contact"]["handle_prim_paths"] = result["links"]["body"]["collision_prim_paths"][:1]
    with pytest.raises(RuntimeError, match="handle_mismatch"):
        _articulated_structure_observation(stage=imported, usd_physics=UsdPhysics, static_receipt=wrong)
    changed = deepcopy(request)
    changed["configuration"]["required_output"]["task_part_mass_kg_bounds"] = [1.9, 2.1]
    with pytest.raises(AuthoringContractError, match="artisan_mass_outside_admitted_bounds"):
        validate_final_usd(final, changed)


def test_physics_namespace_keeps_joint_frames_and_parent_scene_placement(tmp_path):
    from blueprint_pipeline.task_object_artisan_exports import _flatten_nested_physics_links, _display_color_surface

    path = tmp_path / 'nested.usda'
    stage = Usd.Stage.CreateNew(str(path))
    root = UsdGeom.Xform.Define(stage, '/Asset')
    stage.SetDefaultPrim(root.GetPrim())
    parent = UsdGeom.Xform.Define(stage, '/Asset/base')
    parent.AddTranslateOp().Set(Gf.Vec3d(.1, .2, .3))
    parent.AddRotateZOp().Set(30)
    child = UsdGeom.Xform.Define(stage, '/Asset/base/panel')
    child.AddTranslateOp().Set(Gf.Vec3d(.4, .5, .6))
    for body in (parent, child):
        UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
    visual = UsdGeom.Cube.Define(stage, '/Asset/base/panel/visual')
    visual.CreateDisplayColorAttr([Gf.Vec3f(.2, .4, .6)])
    color_before = visual.GetDisplayColorAttr().Get()
    joint = UsdPhysics.RevoluteJoint.Define(stage, '/Asset/hinge')
    joint.CreateBody0Rel().SetTargets([parent.GetPath()])
    joint.CreateBody1Rel().SetTargets([child.GetPath()])
    joint.CreateLocalPos0Attr(Gf.Vec3f(.2, .3, .4))
    before = UsdGeom.XformCache().GetLocalToWorldTransform(child.GetPrim())
    overlay = _flatten_nested_physics_links(stage)
    assert overlay == [{'before': '/Asset/base/panel', 'after': '/Asset/BlueprintPhysicsLinks/panel'}]
    moved = stage.GetPrimAtPath(overlay[0]['after'])
    assert UsdGeom.XformCache().GetLocalToWorldTransform(moved) == before
    assert str(joint.GetBody1Rel().GetTargets()[0]) == str(moved.GetPath())
    assert joint.GetLocalPos0Attr().Get() == Gf.Vec3f(.2, .3, .4)
    placement = root.AddTranslateOp()
    placement.Set(Gf.Vec3d(1, 2, 3))
    expected = before * placement.GetOpTransform(Usd.TimeCode.Default())
    assert UsdGeom.XformCache().GetLocalToWorldTransform(moved) == expected
    assert _flatten_nested_physics_links(stage) == []
    appearance = _display_color_surface(stage)
    assert appearance['bound_visual_count'] == 1
    moved_visual = stage.GetPrimAtPath(str(moved.GetPath()) + '/visual')
    assert UsdGeom.Gprim(moved_visual).GetDisplayColorAttr().Get() == color_before
    material, _ = UsdShade.MaterialBindingAPI(moved_visual).ComputeBoundMaterial()
    assert material.GetSurfaceOutput().HasConnectedSource()
    reader = UsdShade.Shader(stage.GetPrimAtPath(appearance['material_path'] + '/reader'))
    assert reader.GetInput('varname').Get() == 'displayColor'
    assert _display_color_surface(stage)['bound_visual_count'] == 0


def test_fixed_interior_mass_keeps_its_own_admission():
    from blueprint_pipeline.task_object_artisan_exports import _mass_groups
    links = {"body": {"mass_kg": 20}, "rack": {"mass_kg": 2}, "rack_grip": {"mass_kg": .2}}
    joints = [{"parent_link_id": "body", "child_link_id": "rack", "joint_type": "fixed"},
              {"parent_link_id": "rack", "child_link_id": "rack_grip", "joint_type": "fixed"}]
    config = {"required_parts": [{"part_id": "rack", "role": "fixed_interior"}],
              "required_output": {"mass_kg_bounds": [19, 21], "fixed_part_mass_kg_bounds": [2.1, 2.3]}}
    groups = _mass_groups(links, joints, config, None)
    assert groups["body"]["mass_kg"] == 20
    assert groups["rack"]["mass_kg"] == pytest.approx(2.2)
    config["required_output"]["fixed_part_mass_kg_bounds"] = [2.4, 3]
    with pytest.raises(AuthoringContractError, match="artisan_mass_outside_admitted_bounds"):
        _mass_groups(links, joints, config, None)


@pytest.mark.parametrize("center,passes", [([.1, .4, .2], True), ([.1, 0, .5], False)])
def test_analytic_hinge_cylinder_can_be_excluded_only_outside_cavity_rays(tmp_path, center, passes):
    request = construction_request(articulated=True)
    source = usd_fixture(tmp_path, articulated=True)
    stage = Usd.Stage.Open(str(source))
    cylinder = UsdGeom.Cylinder.Define(stage, "/UpstreamObject/body/hinge")
    cylinder.CreateRadiusAttr(.02)
    cylinder.CreateHeightAttr(.03)
    cylinder.AddTranslateOp().Set(Gf.Vec3d(*center))
    UsdPhysics.CollisionAPI.Apply(cylinder.GetPrim())
    stage.GetRootLayer().Save()
    if passes:
        assert adapt_usd(source, tmp_path / "final.usdz", request,
                         annotations_path=annotations(tmp_path, request))["final_readback"]["interior_cavity_readback"]
    else:
        with pytest.raises(AuthoringContractError, match="analytic_Cylinder"):
            adapt_usd(source, tmp_path / "final.usdz", request, annotations_path=annotations(tmp_path, request))


def test_imported_composition_rechecks_materials_and_physical_joint_frame(tmp_path):
    from blueprint_pipeline.task_evaluation_scene_configuration_native_import_driver import _articulated_structure_observation, _IMPORTED_ROOT
    request = construction_request(articulated=True)
    final = tmp_path / "final.usdz"
    observed = adapt_usd(usd_fixture(tmp_path, articulated=True), final, request,
        annotations_path=annotations(tmp_path, request))["final_readback"]
    receipt = {"backend": "artisan_v1", "articulation_graph": observed["articulation_graph"],
        "observed_structure": {**observed, "joint_prim_paths": list(observed["joint_prim_paths"].values())},
        "task_joint": observed["task_joint"], "task_contact": {"handle_prim_paths": observed["handle_prim_paths"]}}
    imported = Usd.Stage.CreateInMemory()
    imported.DefinePrim(_IMPORTED_ROOT, "Xform").GetReferences().AddReference(str(final), "/Asset")
    readback = _articulated_structure_observation(stage=imported, usd_physics=UsdPhysics, static_receipt=receipt)
    assert readback["link_physics_readback"]["door"]["physics_materials"]
    joint_path = _IMPORTED_ROOT + "/joints/door_hinge"
    UsdPhysics.Joint(imported.GetPrimAtPath(joint_path)).GetLocalRot0Attr().Set(Gf.Quatf(1))
    with pytest.raises(RuntimeError, match="physical_joint_axis_mismatch"):
        _articulated_structure_observation(stage=imported, usd_physics=UsdPhysics, static_receipt=receipt)


def test_measured_mass_cannot_be_replaced_by_an_upstream_estimate(tmp_path):
    request = construction_request()
    request["configuration"]["measured_physical_properties"] = {"mass_kg": {"value": 3, "basis": "measured", "evidence_ids": ["scale"]}}
    with pytest.raises(AuthoringContractError, match="measured_mass_changed"):
        adapt_usd(usd_fixture(tmp_path), tmp_path / "final.usdz", request)


def export_fixture(root):
    exports = root / "exports"
    exports.mkdir()
    files = {"model": root / "model.py", "blueprint": root / "blueprint.json", "candidate_manifest": root / "candidate.json",
             "physics_report": root / "physics.json"}
    files["model"].write_text("# Synthetic protocol fixture, never submitted to generation")
    files["blueprint"].write_text("{}")
    files["candidate_manifest"].write_text('{"candidate_id":"fixture-candidate"}')
    files["physics_report"].write_text('{"candidate_id":"fixture-candidate","status":"passed"}')
    (exports / "physics.json").write_bytes(files["physics_report"].read_bytes())
    (exports / "object.usd").write_text("# synthetic manifest bytes, not native qualification")
    record = file_record(exports / "object.usd")
    manifest = {"schema_version": "release-candidate-v7", "status": "accepted", "candidate_id": "fixture-candidate",
        "source_manifest_sha256": file_record(files["candidate_manifest"])["sha256"][7:], "formats": ["usd"],
        "model_sha256": file_record(files["model"])["sha256"][7:], "blueprint_sha256": file_record(files["blueprint"])["sha256"][7:],
        "physics_report_path": "physics.json", "physics_report_sha256": file_record(files["physics_report"])["sha256"][7:],
        "artifacts": {"object.usd": {"path": "object.usd", "sha256": record["sha256"][7:], "size_bytes": record["size_bytes"]}}}
    (exports / "export_manifest.json").write_text(json.dumps(manifest))
    return {"phase": "DONE", "registered": True, "returncode": 0, "canonical_record_id": "fixture-record",
            "candidate_id": "fixture-candidate", "reviewed_physics_report_sha256": manifest["physics_report_sha256"],
            "paths": {**{k: str(v) for k, v in files.items()}, "exports": str(exports)}}, manifest


def test_upstream_manifest_preserves_reviewed_program_and_refuses_changed_or_missing_usd(tmp_path):
    terminal, manifest = export_fixture(tmp_path)
    assert upstream_exports(terminal, tmp_path)["usd"]["path"] == str(tmp_path / "exports/object.usd")
    terminal["phase"] = "FAILED"
    with pytest.raises(AuthoringContractError, match="not_done"):
        upstream_exports(terminal, tmp_path)
    terminal["phase"] = "DONE"
    Path(terminal["paths"]["model"]).write_text("changed")
    with pytest.raises(AuthoringContractError, match="reviewed_program"):
        upstream_exports(terminal, tmp_path)
