"""Reviewed parts compose into one passive-joint assembly; nothing is driven, nothing is measured."""
import json
from types import SimpleNamespace

import numpy as np
import pytest
import trimesh
from pxr import Gf, Usd, UsdGeom, UsdPhysics

from blueprint_pipeline.articulation_graph_contract import validate_articulation_graph
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_object_articulated_packaging import (
    PROVENANCE_ATTRIBUTE, TASK_CONTACT_ROLE_ATTRIBUTE, TARGET_JOINT_ID,
    articulation_graph_from_plan, derived_website_cabinet_depth_hypothesis,
    package_astra_articulated_candidate, plan_articulated_assembly,
)
from blueprint_pipeline.task_object_astra_authoring import AssetAuthoringError, file_record
from blueprint_pipeline.task_object_physical_property_review import (
    PhysicalPropertyReviewInput, PhysicalPropertyReviewProposal, review_physical_properties,
)
from blueprint_pipeline.task_evaluation_scene_configuration_submission_records import (
    articulated_stage_three_configuration,
)

MECHANISM = {"part_label": "middle drawer", "joint_type": "prismatic", "estimated_usable_stroke_m": 0.36,
             "estimated_front_normal_world": [0.0, -1.0, 0.0], "lock_status": "unknown",
             "travel_authority": "object_prior_estimate"}
PHYSICS = {"mass_kg_bounds": [4.0, 40.0], "task_part_mass_kg_bounds": [0.5, 6.0],
           "static_friction_bounds": [0.3, 0.8], "dynamic_friction_bounds": [0.2, 0.6],
           "restitution_bounds": [0.0, 0.2], "joint_friction_bounds": [1.0, 15.0], "joint_damping_bounds": [1.0, 30.0]}


def configuration():
    # World envelope: 0.42 m along X (width), 0.55 m along Y (depth: the front faces -Y), 0.62 m tall.
    return articulated_stage_three_configuration(
        scene_id="scene-1", replacement_identity={"id": "website-subject-cab", "version": "v1"},
        source_instance_id="cabinet", authoring_target="three-drawer wood cabinet",
        source_min=[-0.21, -0.275, 0.0], source_max=[0.21, 0.275, 0.62], dimension_tolerance=0.2,
        physics_bounds=PHYSICS, mechanism=MECHANISM)


def _part(root, *, object_id, dimensions, mass_kg, density, bounds, compound=False):
    root.mkdir(parents=True, exist_ok=True)
    if compound:
        halves = []
        for sign in (-1, 1):
            half = trimesh.creation.box(extents=[dimensions[0] / 2, *dimensions[1:]])
            half.apply_translation([sign * dimensions[0] / 4, 0, dimensions[2] / 2])
            halves.append(half)
        mesh = trimesh.util.concatenate(halves)
    else:
        mesh = trimesh.creation.box(extents=dimensions)
        mesh.apply_translation([0, 0, dimensions[2] / 2])
    stage = Usd.Stage.CreateNew(str(root / "candidate.usdc"))
    asset_root = UsdGeom.Xform.Define(stage, "/Asset")
    stage.SetDefaultPrim(asset_root.GetPrim())
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, "Z")
    visual = UsdGeom.Mesh.Define(stage, "/Asset/Visual")
    visual.CreatePointsAttr([Gf.Vec3f(*p) for p in mesh.vertices])
    visual.CreateFaceVertexCountsAttr([3] * len(mesh.faces))
    visual.CreateFaceVertexIndicesAttr(mesh.faces.reshape(-1).tolist())
    visual.CreateSubdivisionSchemeAttr("none")
    stage.GetRootLayer().Save()
    cad = trimesh.creation.box(extents=np.array(dimensions) * 1000)
    cad.export(root / "cad.stl")
    (root / "asset_program.py").write_text("# retained author program\n")
    (root / "final_visual_mesh.json").write_text(json.dumps(dict(
        schema_version="final_visual_mesh.v1", units="metres", coordinate_frame="center_XY_bottom_Z",
        vertices_m=mesh.vertices.tolist(), faces=mesh.faces.tolist())))
    receipt = dict(schema_version="final_visual_mesh_receipt.v1", claim_ceiling="development_only",
        mesh_file="final_visual_mesh.json", mesh_sha256=file_record(root / "final_visual_mesh.json")["sha256"],
        candidate_usd_file="candidate.usdc", candidate_usd_sha256=file_record(root / "candidate.usdc")["sha256"],
        author_program_file="asset_program.py", author_program_sha256=file_record(root / "asset_program.py")["sha256"],
        source_cad_stl_sha256=file_record(root / "cad.stl")["sha256"], dimensions_m=mesh.extents.tolist(),
        volume_m3=float(mesh.volume), watertight=True, winding_consistent=True,
        connected_components=mesh.body_count,
        physics_authority="packaging_accepted_physical_review_only", exported_rigid_body_count=0)
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    (root / "final_visual_mesh_receipt.json").write_text(json.dumps(receipt))

    def value(v, low, high):
        return dict(value=v, basis="estimated", interval=dict(lower=low, upper=high),
                    rationale="Fixture model", uncertainty="Explicit interval", evidence_ids=["fixture"])
    properties = dict(mass_kg=value(mass_kg, mass_kg * 0.95, mass_kg * 1.04), static_friction=value(0.6, 0.55, 0.7),
                      dynamic_friction=value(0.4, 0.3, 0.5), restitution=value(0.05, 0.0, 0.1))
    proposal = dict(object_id=object_id, dimensions={axis: value(v, v * 0.98, v * 1.02)
        for axis, v in zip(("x_m", "y_m", "z_m"), dimensions)}, properties=properties,
        optical_material=dict(name="Opaque", transmission=0.0, opacity=1.0),
        mass_model=dict(method="density_fill", density_kg_m3=dict(lower=density[0], upper=density[1]),
            envelope_fill_fraction=dict(lower=0.9, upper=1.0), sheet_count=None, sheet_area_m2=None,
            grammage_g_m2=None, cover_mass_kg=None, rationale="Fixture", uncertainty="Range", evidence_ids=["fixture"]),
        review_rationale="Independent fixture review")
    review_input = PhysicalPropertyReviewInput.model_validate(dict(object_id=object_id,
        object_description="Fixture part", material_description="Opaque laminate", appearance="opaque",
        dimensions=proposal["dimensions"], measured={k: None for k in properties}, proposed=None,
        optical_material=proposal["optical_material"], admitted_restitution=dict(lower=0.0, upper=0.2),
        evidence=[dict(evidence_id="fixture", uri="retained://fixture", sha256="a" * 64,
                       kind="primary_reference", excerpt="Fixture density range")]))
    review = review_physical_properties(review_input, PhysicalPropertyReviewProposal.model_validate(proposal)).model_dump(mode="json")
    assert review["accepted"] is not None, review["blockers"]
    (root / "review.json").write_text(json.dumps(review))
    (root / "review_input.json").write_text(review_input.model_dump_json())
    (root / "geometry.json").write_text(json.dumps(dict(dimensions_m=list(dimensions), minimum_z_m=0.0, center_xy_m=[0.0, 0.0],
                                                        materials=[dict(alpha=1.0, transmission=0.0)])))
    request = SimpleNamespace(request_digest="sha256:" + canonical_digest({"part": object_id})[7:], object_id=object_id,
                              role="task_object", dimensions_m=tuple(dimensions), maximum_export_error_m=0.00001,
                              physical_review_input=review_input)
    result = dict(request_digest=request.request_digest, status="candidate_authored_pending_native_qualification",
        physical_review=file_record(root / "review.json"), physical_review_input=file_record(root / "review_input.json"),
        geometry_readback=file_record(root / "geometry.json"), asset=file_record(root / "candidate.usdc"),
        cad={"stl": file_record(root / "cad.stl"), "readback": {"volume_mm3": float(cad.volume)}},
        final_visual_mesh=file_record(root / "final_visual_mesh.json"),
        final_visual_mesh_receipt=file_record(root / "final_visual_mesh_receipt.json"))
    result["result_digest"] = canonical_digest(result)
    return request, result, bounds


def fixture(tmp_path, *, compound_drawer=False, drawer_mass=2.0, drawer_density=(30.0, 120.0)):
    plan = plan_articulated_assembly(configuration())
    carcass_dims = plan["parts"]["carcass"]["dimensions_m"]
    drawer_dims = plan["parts"]["drawer"]["dimensions_m"]
    carcass = _part(tmp_path / "carcass", object_id="website-subject-cab__carcass", dimensions=carcass_dims,
                    mass_kg=12.0, density=(60.0, 140.0),
                    bounds={"mass_kg": [4.0, 40.0], "static_friction": [0.3, 0.8], "dynamic_friction": [0.2, 0.6], "restitution": [0.0, 0.2]})
    drawer = _part(tmp_path / "drawer", object_id="website-subject-cab__drawer", dimensions=drawer_dims,
                   mass_kg=drawer_mass, density=drawer_density,
                   bounds={"mass_kg": [0.5, 6.0], "static_friction": [0.3, 0.8], "dynamic_friction": [0.2, 0.6], "restitution": [0.0, 0.2]},
                   compound=compound_drawer)
    requests = {"carcass": carcass[0], "drawer": drawer[0]}
    results = {"carcass": carcass[1], "drawer": drawer[1]}
    bounds = {"carcass": carcass[2], "drawer": drawer[2]}
    return plan, requests, results, bounds


def test_articulated_drawer_keeps_closed_disconnected_visual_pieces_in_one_link(tmp_path):
    plan, requests, results, bounds = fixture(tmp_path, compound_drawer=True)
    receipt = package_astra_articulated_candidate(
        requests=requests, authoring_results=results, plan=plan,
        output_root=tmp_path / "packaged", physics_bounds=bounds)
    stage = Usd.Stage.Open(receipt["asset"]["path"])
    drawer_link = stage.GetPrimAtPath("/Asset/links/drawer_1")
    collision = UsdGeom.Mesh.Get(stage, "/Asset/links/drawer_1/collision/FinalVisualShape")
    mesh = trimesh.Trimesh(
        vertices=np.asarray(collision.GetPointsAttr().Get(), dtype=float),
        faces=np.asarray(collision.GetFaceVertexIndicesAttr().Get(), dtype=int).reshape(-1, 3),
        process=False)
    assert drawer_link.HasAPI(UsdPhysics.RigidBodyAPI)
    assert mesh.body_count == 2 and mesh.is_watertight
    assert len([prim for prim in stage.Traverse() if prim.IsA(UsdPhysics.PrismaticJoint)]) == 1


def test_plan_places_three_bays_with_one_prismatic_task_joint_and_records_assumptions():
    plan = plan_articulated_assembly(configuration())
    assert plan["bay_count"] == 3 and plan["task_bay_index"] == 1
    dims = plan["assembly_dimensions_m"]
    # The front faces -Y in the world, so the world Y extent becomes assembly depth (+X).
    assert dims["depth_x"] == pytest.approx(0.55) and dims["width_y"] == pytest.approx(0.42) and dims["height_z"] == pytest.approx(0.62)
    joint = plan["task_joint"]
    assert joint["child_link_id"] == "drawer_1" and joint["axis_asset_frame"] == [1.0, 0.0, 0.0]
    assert joint["limits_m"][0] == 0.0 and 0.0 < joint["limits_m"][1] <= 0.36
    assert joint["drive"] == {"drive_type": "none", "stiffness": 0.0, "damping": 5.0, "maximum_force": 0.0,
                              "implementation": "passive_force_damper"}
    assert "interior_and_drawer_boxes_unobserved_generated_candidate_geometry" in plan["construction_assumptions"]
    assert plan["physical_measurement_proven"] is False
    # Drawer fronts sit flush with the open carcass front, stacked bottom to top.
    z = [row["rest_translation_m"][2] for row in plan["links"] if not row["is_root"]]
    assert z == sorted(z, reverse=True) and all(v > 0 for v in z)
    with pytest.raises(AssetAuthoringError, match="family_unsupported"):
        plan_articulated_assembly({**configuration(), "mechanism": {**MECHANISM, "joint_type": "revolute",
                                                                     "estimated_usable_swing_rad": 1.2}})
    with pytest.raises(AssetAuthoringError, match="position_unresolved"):
        plan_articulated_assembly({**configuration(), "mechanism": {**MECHANISM, "task_part_label": "the drawer"}})


def _thin_website_cabinet():
    from blueprint_pipeline.website_drawer_depth_prior import PRIOR

    value = configuration()
    value["scene_id"] = PRIOR["scene_id"]
    value["replacement_identity"] = PRIOR["subject_identity"]
    value["source_observation_kind"] = "website_capture_frames"
    value["dimension_authority"] = "estimated"
    value["metric_envelope"] = {"minimum_xyz_m": [-0.21, -0.0815, 0.0],
                                "maximum_xyz_m": [0.21, 0.0815, 0.62],
                                "maximum_dimension_relative_error": 0.2}
    value["mechanism"]["estimated_usable_stroke_m"] = 0.12
    value["mechanism"]["joint_limits"] = [0.0, 0.12]
    return value


def _depth_prior(value):
    return derived_website_cabinet_depth_hypothesis(
        value, [{"role": "observed_source", "sha256": "sha256:" + "a" * 64}], 0.6)


def test_thin_website_drawer_requires_explicit_depth_hypothesis_before_authoring():
    value = _thin_website_cabinet()
    with pytest.raises(AssetAuthoringError, match="depth_implausible_hypothesis_required"):
        plan_articulated_assembly(value)
    value["development_geometry_hypothesis"] = _depth_prior(value)
    value["mechanism"]["estimated_usable_stroke_m"] = 0.4125
    value["mechanism"]["joint_limits"] = [0.0, 0.4125]
    value["required_output"]["mass_kg_bounds"] = [4.0, 30.0]
    value["required_output"]["task_part_mass_kg_bounds"] = [0.5, 9.0]
    plan = plan_articulated_assembly(value)
    assert plan["source_geometry"]["projected_depth_m"] == pytest.approx(0.163)
    assert plan["source_geometry"]["aabb_min_xyz_m"] == value["metric_envelope"]["minimum_xyz_m"]
    assert plan["assembly_dimensions_m"]["depth_x"] == pytest.approx(0.55)
    assert plan["assembly_dimensions_m"]["authority"] == "development_only_depth_hypothesis"
    assert plan["development_geometry_hypothesis"]["depth_disagreement_m"] == pytest.approx(0.387)
    assert plan["development_geometry_hypothesis"]["physical_measurement_proven"] is False
    assert plan["parts"]["carcass"]["dimensions_m"][0] == pytest.approx(0.55)
    assert plan["task_joint"]["limits_m"] == [0.0, 0.4125]
    assert plan["development_geometry_hypothesis"]["estimated_minimum_opening_m"] == 0.2475


@pytest.mark.parametrize("change", ["source", "interval", "basis", "nominal", "rationale", "scene"])
def test_depth_hypothesis_refuses_unbound_or_unbounded_estimates(change):
    value = _thin_website_cabinet()
    hypothesis = _depth_prior(value)
    if change == "source":
        hypothesis["source_aabb_max_xyz_m"] = [0.21, 0.5, 0.62]
    elif change == "interval":
        hypothesis["depth_interval_m"] = [0.35, 1.5]
    elif change == "basis":
        hypothesis["basis"] = "uncited_guess"
    elif change == "nominal":
        hypothesis["estimated_depth_m"] = 0.9
    elif change == "rationale":
        hypothesis["rationale"] = "assumed"
    value["development_geometry_hypothesis"] = hypothesis
    value["mechanism"]["estimated_usable_stroke_m"] = 0.4125
    value["mechanism"]["joint_limits"] = [0.0, 0.4125]
    value["required_output"]["mass_kg_bounds"] = [4.0, 30.0]
    value["required_output"]["task_part_mass_kg_bounds"] = [0.5, 9.0]
    if change == "scene":
        value["scene_id"] = "another-scene"
    with pytest.raises(AssetAuthoringError, match="depth_hypothesis"):
        plan_articulated_assembly(value)


def test_reviewed_parts_compose_into_a_passive_prismatic_assembly(tmp_path):
    plan, requests, results, bounds = fixture(tmp_path)
    receipt = package_astra_articulated_candidate(requests=requests, authoring_results=results, plan=plan,
                                                  output_root=tmp_path / "packaged", physics_bounds=bounds)
    stage = Usd.Stage.Open(receipt["asset"]["path"])
    root = stage.GetDefaultPrim()
    assert root.GetPath().pathString == "/Asset" and root.HasAPI(UsdPhysics.ArticulationRootAPI)
    bodies = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    assert sorted(p.GetName() for p in bodies) == ["carcass", "drawer_0", "drawer_1", "drawer_2"]
    assert all(UsdPhysics.RigidBodyAPI(p).GetKinematicEnabledAttr().Get() is False for p in bodies)
    prismatic = [p for p in stage.Traverse() if p.IsA(UsdPhysics.PrismaticJoint)]
    fixed = [p for p in stage.Traverse() if p.IsA(UsdPhysics.FixedJoint)]
    assert len(prismatic) == 1 and len(fixed) == 2
    joint = UsdPhysics.PrismaticJoint(prismatic[0])
    assert prismatic[0].GetName() == TARGET_JOINT_ID and joint.GetAxisAttr().Get() == "X"
    assert joint.GetLowerLimitAttr().Get() == 0.0 and joint.GetUpperLimitAttr().Get() == pytest.approx(plan["task_joint"]["limits_m"][1])
    assert [str(t) for t in joint.GetBody1Rel().GetTargets()] == ["/Asset/links/drawer_1"]
    drive = UsdPhysics.DriveAPI(prismatic[0], "linear")
    assert drive.GetStiffnessAttr().Get() == 0.0 and drive.GetDampingAttr().Get() == pytest.approx(5.0)
    assert prismatic[0].GetCustomDataByKey("blueprint:driveImplementation") == "passive_force_damper"
    handles = [p for p in stage.Traverse() if p.GetCustomDataByKey(TASK_CONTACT_ROLE_ATTRIBUTE) == "handle"]
    assert [p.GetPath().pathString for p in handles] == ["/Asset/links/drawer_1/collision/handle"]
    colliders = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.CollisionAPI)]
    assert len(colliders) == 5 and all(p.GetCustomDataByKey(PROVENANCE_ATTRIBUTE) for p in colliders)
    # Nested parts never fight the joints: every intra-assembly pair is filtered.
    carcass = stage.GetPrimAtPath("/Asset/links/carcass")
    filtered = {str(t) for t in UsdPhysics.FilteredPairsAPI(carcass).GetFilteredPairsRel().GetTargets()}
    assert filtered == {"/Asset/links/drawer_0", "/Asset/links/drawer_1", "/Asset/links/drawer_2"}
    completion = receipt["physics_completion"]
    assert completion["schema_version"] == "task_evaluation_articulated_candidate_physics_completion.v1"
    assert completion["fixed_base_body_prim_path"] == "/Asset/links/carcass"
    assert completion["task_joint_prim_path"] == "/Asset/joints/task_part_joint"
    assert completion["handle_prim_paths"] == ["/Asset/links/drawer_1/collision/handle"]
    assert completion["candidate_prior_only"] is True and completion["physical_truth_claimed"] is False
    assert completion["completion_digest"] == canonical_digest(completion, digest_field="completion_digest")
    assert [row["mass_kg"] for row in completion["links"]] == [12.0, 2.0, 2.0, 2.0]
    dims = completion["collision_dimensions_m"]
    assert dims[0] == pytest.approx(plan["assembly_dimensions_m"]["depth_x"] + 0.03, abs=1e-6)  # handles protrude
    assert dims[2] == pytest.approx(plan["assembly_dimensions_m"]["height_z"], abs=1e-6)
    graph = validate_articulation_graph(articulation_graph_from_plan(plan))
    assert [row["joint_id"] for row in graph["joints"] if row["role"] == "target"] == [TARGET_JOINT_ID]
    assert graph["success_predicate"]["joint_intervals"][TARGET_JOINT_ID][0] == pytest.approx(0.6 * plan["task_joint"]["limits_m"][1], abs=1e-5)
    assert all(row["collision_enabled"] is False for row in graph["collision_pairs"])
    assert receipt["native_qualified"] is False and receipt["claim_ceiling"] == "development_only"


def test_changed_part_dimensions_or_missing_part_are_refused(tmp_path):
    plan, requests, results, bounds = fixture(tmp_path)
    shrunk = dict(plan, parts={**plan["parts"], "drawer": {**plan["parts"]["drawer"], "dimensions_m": [0.1, 0.1, 0.1]}})
    with pytest.raises(AssetAuthoringError, match="part_dimensions_changed"):
        package_astra_articulated_candidate(requests=requests, authoring_results=results, plan=shrunk,
                                            output_root=tmp_path / "p1", physics_bounds=bounds)
    with pytest.raises(AssetAuthoringError, match="part_set_mismatch"):
        package_astra_articulated_candidate(requests={"carcass": requests["carcass"]}, authoring_results=results,
                                            plan=plan, output_root=tmp_path / "p2", physics_bounds=bounds)


def test_articulated_mass_bound_applies_to_simulated_value_and_retains_source_uncertainty(tmp_path):
    plan, requests, results, bounds = fixture(tmp_path)
    # The generated USD has one 12 kg carcass. Its reviewed 11.4–12.48 kg
    # uncertainty must remain visible even though this scene admits at most
    # 12 kg for the simulated value.
    bounds["carcass"]["mass_kg"] = [4.0, 12.0]
    receipt = package_astra_articulated_candidate(
        requests=requests, authoring_results=results, plan=plan,
        output_root=tmp_path / "bounded", physics_bounds=bounds)
    base = next(row for row in receipt["physics_completion"]["links"] if row["link_id"] == "carcass")
    assert base["mass_kg"] == 12.0
    assert base["mass_basis"] == "estimated"
    assert base["mass_uncertainty_interval_kg"] == pytest.approx([11.4, 12.48])
    assert base["mass_interval_exceeds_admitted_simulation_bounds"] is True
    bounds["carcass"]["mass_kg"] = [4.0, 11.9]
    with pytest.raises(AssetAuthoringError, match="estimate_outside_admitted_bounds:mass_kg"):
        package_astra_articulated_candidate(
            requests=requests, authoring_results=results, plan=plan,
            output_root=tmp_path / "rejected", physics_bounds=bounds)
