"""Real USD bytes exercise ADP-009 adapter contracts, not a live ARTISAN run."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
from pxr import Gf, Usd, UsdGeom, UsdPhysics

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


def test_solid_collider_cannot_claim_a_hollow_interior(tmp_path):
    request = construction_request(articulated=True)
    with pytest.raises(AuthoringContractError, match="collision_cavity_unproven"):
        adapt_usd(usd_fixture(tmp_path, articulated=True, solid=True), tmp_path / "final.usdz", request,
                  annotations_path=annotations(tmp_path, request))


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
