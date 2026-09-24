"""A hinged-door appliance plans, builds and qualifies as a hollow body, one revolute door and fixed racks."""
import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest
import trimesh
from pxr import Gf, Usd, UsdPhysics

from blueprint_pipeline.articulation_graph_contract import validate_articulation_graph
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_evaluation_scene_configuration_articulated_static_qualification import (
    qualify_scene_configuration_articulated_asset_static,
)
from blueprint_pipeline.task_evaluation_scene_configuration_static_qualification import (
    TaskEvaluationSceneConfigurationStaticQualificationError,
)
from blueprint_pipeline.task_evaluation_scene_configuration_submission_records import (
    ARTICULATED_STATIC_CHECKS, articulated_stage_three_configuration,
)
from blueprint_pipeline.task_object_articulated_packaging import (
    REQUIRED_PARTS_ATTRIBUTE, TARGET_JOINT_ID, articulation_graph_from_plan, interior_cavity_findings,
    package_astra_articulated_candidate, plan_articulated_assembly, required_part_findings,
)
from blueprint_pipeline.task_object_astra_authoring import AssetAuthoringError
from tests.test_task_object_articulated_packaging import _part, configuration as drawer_configuration, open_front_shell

IDENTITY = {"id": "website-subject-dw", "version": "v1"}
PHYSICS = {"mass_kg_bounds": [10.0, 60.0], "task_part_mass_kg_bounds": [2.0, 12.0],
           "static_friction_bounds": [0.3, 0.8], "dynamic_friction_bounds": [0.2, 0.6],
           "restitution_bounds": [0.0, 0.2], "joint_friction_bounds": [0.1, 5.0], "joint_damping_bounds": [0.1, 5.0]}
PARTS = [("body_front", "Body front frame", "body"), ("tub_interior", "Stainless tub interior", "body_feature"),
         ("left_side", "Left side panel", "body_feature"), ("right_side", "Right side panel", "body_feature"),
         ("kickplate", "Kickplate", "body_feature"), ("door_outer", "Door outer panel", "task_part"),
         ("door_inner", "Door inner liner", "door_feature"), ("handle", "Door handle", "door_feature"),
         ("control_panel", "Control panel", "door_feature"), ("brand_label", "Brand label", "door_feature"),
         ("upper_rack", "Upper dish rack", "fixed_interior"), ("lower_rack", "Lower dish rack", "fixed_interior"),
         ("cutlery_basket", "Cutlery basket", "fixed_interior")]


def _frame(frame_id, digest, state, visible, view="front", reason="whole-front coverage"):
    return {"path": f"/retained/{frame_id}.png", "sha256": digest, "frame_id": frame_id,
            "timestamp_seconds": 1.5, "visible_parts": visible, "part_state": state, "view": view, "reason": reason}


def dishwasher(hinge="bottom", frames=None):
    mechanism = {"part_label": "dishwasher door", "joint_type": "revolute", "estimated_usable_swing_rad": 1.4,
                 "estimated_front_normal_world": [0.0, -1.0, 0.0], "lock_status": "unknown",
                 "travel_authority": "object_prior_estimate"}
    value = articulated_stage_three_configuration(
        scene_id="scene-dw", replacement_identity=IDENTITY, source_instance_id="dishwasher",
        authoring_target="built-in dishwasher", source_min=[-0.3, -0.29, 0.0], source_max=[0.3, 0.29, 0.85],
        dimension_tolerance=0.1, physics_bounds=PHYSICS, mechanism=mechanism)
    frames = frames or [
        _frame("f_closed", "sha256:" + "1" * 64, "closed",
               ["body_front", "door_outer", "handle", "control_panel", "brand_label", "kickplate", "left_side"]),
        _frame("f_open", "sha256:" + "2" * 64, "open",
               ["tub_interior", "door_inner", "upper_rack", "lower_rack", "cutlery_basket", "right_side"],
               view="front-high", reason="interior observed with the door open")]
    seen = {part: [row["frame_id"] for row in frames if part in row["visible_parts"]] for part, _, _ in PARTS}
    value.update(assembly_family="hinged_door_appliance", hinge_edge=hinge, reference_frames=frames,
                 required_parts=[{"part_id": part, "label": label, "role": role, "observed_frame_ids": seen[part]}
                                 for part, label, role in PARTS],
                 body_depth={"value_m": 0.58, "basis": "interior_observed_open_state",
                             "frame_ids": [frames[-1]["frame_id"]]},
                 source_observation_kind="website_capture_frames", dimension_authority="estimated")
    value["required_output"]["fixed_part_mass_kg_bounds"] = [0.3, 4.0]
    return value


def test_dishwasher_plans_hollow_body_bottom_hinged_door_and_fixed_racks_inside_the_tub():
    plan = plan_articulated_assembly(dishwasher())
    assert plan["family"] == "hinged_door_appliance" and plan["root_link_id"] == "body"
    assert [row["link_id"] for row in plan["links"]] == ["body", "door", "upper_rack", "lower_rack", "cutlery_basket"]
    dims = plan["assembly_dimensions_m"]
    assert (dims["depth_x"], dims["width_y"], dims["height_z"]) == (0.58, 0.6, 0.85)
    assert plan["closed_collision_dimensions_m"] == [0.61, 0.6, 0.85]
    assert plan["parts"]["body"]["dimensions_m"] == [0.53, 0.6, 0.85]
    assert plan["parts"]["door"]["dimensions_m"] == [0.08, 0.6, 0.75]  # above a 0.10 m kickplate
    joint = plan["task_joint"]
    assert joint["joint_type"] == "revolute" and joint["child_link_id"] == "door" and joint["parent_link_id"] == "body"
    assert joint["axis_asset_frame"] == [0.0, 1.0, 0.0] and joint["usd_axis"] == "Y"
    assert joint["anchor_asset_frame_m"] == [0.29, 0.0, 0.1]  # bottom front edge
    assert joint["limits_rad"] == [0.0, 1.4] and joint["reset_position_rad"] == 0.0
    assert joint["drive"]["stiffness"] == 0.0 and joint["drive"]["implementation"] == "passive_torque_damper"
    # Opening by +swing about +Y swings the door top outward (+X), never into the body.
    top = np.array([0.0, 0.0, 0.75])
    c, s = math.cos(0.5), math.sin(0.5)
    assert (np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]) @ top)[0] > 0
    tub = plan["interior_cavities"][0]
    assert tub["link_id"] == "body" and tub["inner_depth_m"] == pytest.approx(0.48)
    body_rest = plan["links"][0]["rest_translation_m"]
    tub_lo = [tub["opening_center_link_m"][0] - tub["inner_depth_m"] + body_rest[0], -tub["opening_width_m"] / 2,
              tub["opening_center_link_m"][2] - tub["opening_height_m"] / 2]
    tub_hi = [tub["opening_center_link_m"][0] + body_rest[0], tub["opening_width_m"] / 2,
              tub["opening_center_link_m"][2] + tub["opening_height_m"] / 2]
    tub_mid_z = (tub_lo[2] + tub_hi[2]) / 2
    rest = {row["link_id"]: row["rest_translation_m"] for row in plan["links"]}
    for link_id in ("upper_rack", "lower_rack", "cutlery_basket"):
        size = plan["parts"][link_id]["dimensions_m"]
        lo = [rest[link_id][0] - size[0] / 2, rest[link_id][1] - size[1] / 2, rest[link_id][2]]
        hi = [rest[link_id][0] + size[0] / 2, rest[link_id][1] + size[1] / 2, rest[link_id][2] + size[2]]
        assert all(tub_lo[i] - 1e-9 <= lo[i] and hi[i] <= tub_hi[i] + 1e-9 for i in range(3)), link_id
    assert rest["upper_rack"][2] > tub_mid_z and rest["lower_rack"][2] < tub_lo[2] + 0.1
    mapped = {row["part_id"]: (row["link_id"], row["feature"]) for row in plan["required_parts"]}
    assert mapped == {"body_front": ("body", "front_frame"), "tub_interior": ("body", "tub_cavity"),
                      "left_side": ("body", "left_side_wall"), "right_side": ("body", "right_side_wall"),
                      "kickplate": ("body", "kickplate"), "door_outer": ("door", "outer_skin"),
                      "door_inner": ("door", "inner_liner"), "handle": ("door", "handle"),
                      "control_panel": ("door", "control_panel"), "brand_label": ("door", "brand_label"),
                      "upper_rack": ("upper_rack", "link"), "lower_rack": ("lower_rack", "link"),
                      "cutlery_basket": ("cutlery_basket", "link")}
    assert "open onto a hollow tub cavity" in plan["parts"]["body"]["description"] and "wood" not in json.dumps(plan).lower()
    graph = validate_articulation_graph(articulation_graph_from_plan(plan))
    target = next(row for row in graph["joints"] if row["role"] == "target")
    assert target["joint_type"] == "revolute" and target["limits"] == [0.0, 1.4]
    assert graph["success_predicate"]["joint_intervals"][TARGET_JOINT_ID] == [0.84, 1.4]
    assert sorted(row["joint_type"] for row in graph["joints"]) == ["fixed", "fixed", "fixed", "revolute"]


@pytest.mark.parametrize("hinge, axis, token, anchor_y", [("left", [0.0, 0.0, -1.0], "Z", -0.3),
                                                          ("right", [0.0, 0.0, 1.0], "Z", 0.3),
                                                          ("top", [0.0, -1.0, 0.0], "Y", 0.0)])
def test_side_and_top_hinges_open_outward_about_their_edge(hinge, axis, token, anchor_y):
    plan = plan_articulated_assembly(dishwasher(hinge))
    joint = plan["task_joint"]
    assert joint["axis_asset_frame"] == axis and joint["usd_axis"] == token
    assert joint["anchor_asset_frame_m"][1] == anchor_y
    # Free edge moves outward (+X) for a small positive rotation about the declared axis.
    handle = plan["parts"]["door"]["handle"]["center_m"]
    rest = plan["links"][1]["rest_translation_m"]
    arm = np.array([rest[i] + handle[i] for i in range(3)]) - np.array(joint["anchor_asset_frame_m"])
    moved = Gf.Rotation(Gf.Vec3d(*axis), 10.0).TransformDir(Gf.Vec3d(*arm.tolist()))
    assert moved[0] > arm[0]


@pytest.mark.parametrize("change, code", [
    ("unplaceable_interior", "required_part_unplanned:ice_maker"),
    ("unplaceable_feature", "required_part_unplanned:water_softener"),
    ("thin", "body_depth_implausibly_thin"),
    ("disagree", "body_depth_disagrees_with_envelope"),
    ("no_depth", "body_depth_missing_or_invalid"),
    ("thin_basis", "body_depth_missing_or_invalid"),
    ("no_hinge", "hinge_edge_missing_or_invalid"),
    ("no_parts", "required_parts_missing"),
    ("no_frames", "reference_frames_invalid"),
    ("unknown_frame", "required_parts_invalid"),
    ("swing", "hinge_swing_infeasible"),
    ("family_joint", "family_joint_mismatch"),
    ("no_family", "family_unsupported"),
])
def test_hinged_contract_fails_closed(change, code):
    value = dishwasher()
    if change == "unplaceable_interior":
        value["required_parts"].append({"part_id": "ice_maker", "label": "Ice maker", "role": "fixed_interior",
                                        "observed_frame_ids": []})
    elif change == "unplaceable_feature":
        value["required_parts"].append({"part_id": "water_softener", "label": "Water softener",
                                        "role": "body_feature", "observed_frame_ids": []})
    elif change == "thin":
        value["metric_envelope"]["minimum_xyz_m"][1], value["metric_envelope"]["maximum_xyz_m"][1] = -0.08, 0.08
        value["body_depth"]["value_m"] = 0.16
    elif change == "disagree":
        value["body_depth"]["value_m"] = 0.40
    elif change == "no_depth":
        del value["body_depth"]
    elif change == "thin_basis":
        value["body_depth"]["basis"] = "owner_or_catalog_specified"
    elif change == "no_hinge":
        del value["hinge_edge"]
    elif change == "no_parts":
        value["required_parts"] = []
    elif change == "no_frames":
        value["reference_frames"] = []
    elif change == "unknown_frame":
        value["required_parts"][0]["observed_frame_ids"] = ["never_retained"]
    elif change == "swing":
        value["mechanism"]["estimated_usable_swing_rad"] = 2.0
        value["mechanism"]["joint_limits"] = [0.0, 2.0]
    elif change == "family_joint":
        value["assembly_family"] = "stacked_drawer_cabinet"
    elif change == "no_family":
        del value["assembly_family"]
    with pytest.raises(AssetAuthoringError, match=code):
        plan_articulated_assembly(value)


def test_object_created_from_description_plans_without_invented_observation():
    value = dishwasher()
    value.update(source_observation_kind="not_captured_created_from_description", reference_frames=[],
                 body_depth={"value_m": 0.58, "basis": "owner_or_catalog_specified", "frame_ids": []})
    for row in value["required_parts"]:
        row["observed_frame_ids"] = []
    plan = plan_articulated_assembly(value)
    assert plan["source_observation"] == "not_captured_created_from_description"
    assert all(row["observed_frame_ids"] == [] for row in plan["required_parts"])
    value["required_parts"][0]["observed_frame_ids"] = ["f_closed"]
    with pytest.raises(AssetAuthoringError, match="required_parts_invalid"):
        plan_articulated_assembly(value)
    value["required_parts"][0]["observed_frame_ids"] = []
    value["body_depth"]["basis"] = "interior_observed_open_state"
    with pytest.raises(AssetAuthoringError, match="body_depth_missing_or_invalid"):
        plan_articulated_assembly(value)


def test_several_task_objects_plan_independently():
    alone = [plan_articulated_assembly(dishwasher()), plan_articulated_assembly(drawer_configuration())]
    together = [plan_articulated_assembly(value) for value in (dishwasher(), drawer_configuration(),
                                                               dishwasher("left"), drawer_configuration())]
    assert together[0] == alone[0] and together[1] == alone[1] and together[3] == alone[1]
    assert together[2]["task_joint"]["axis_asset_frame"] != alone[0]["task_joint"]["axis_asset_frame"]


def test_drawer_contract_parts_map_onto_carcass_and_drawers():
    value = drawer_configuration()
    value["required_parts"] = [
        {"part_id": "carcass", "label": "Cabinet body", "role": "body", "observed_frame_ids": []},
        {"part_id": "middle_drawer", "label": "Middle drawer", "role": "task_part", "observed_frame_ids": []},
        {"part_id": "middle_drawer_handle", "label": "Handle", "role": "door_feature", "observed_frame_ids": []},
        {"part_id": "top_drawer", "label": "Top drawer", "role": "fixed_interior", "observed_frame_ids": []}]
    plan = plan_articulated_assembly(value)
    assert {row["part_id"]: row["link_id"] for row in plan["required_parts"]} == {
        "carcass": "carcass", "middle_drawer": "drawer_1", "middle_drawer_handle": "drawer_1", "top_drawer": "drawer_0"}
    assert [row["cavity_id"] for row in plan["interior_cavities"]] == ["bay_0", "bay_1", "bay_2"]
    value["required_parts"].append({"part_id": "caster", "label": "Caster wheel", "role": "body_feature",
                                    "observed_frame_ids": []})
    with pytest.raises(AssetAuthoringError, match="required_part_unplanned:caster"):
        plan_articulated_assembly(value)


def test_cavity_probe_passes_hollow_open_front_and_fails_solid_slab_or_open_back():
    plan = plan_articulated_assembly(dishwasher())
    dims, cavities = plan["parts"]["body"]["dimensions_m"], plan["interior_cavities"]
    hollow = open_front_shell(dims, cavities)
    assert interior_cavity_findings(hollow.vertices, hollow.faces, cavities) == []
    solid = trimesh.creation.box(extents=dims)
    solid.apply_translation([0, 0, dims[2] / 2])
    assert interior_cavity_findings(solid.vertices, solid.faces, cavities) == ["body_interior_cavity_closed:tub"]
    slab = trimesh.creation.box(extents=[0.16, dims[1], dims[2]])
    slab.apply_translation([0, 0, dims[2] / 2])
    assert interior_cavity_findings(slab.vertices, slab.faces, cavities) == ["body_interior_cavity_closed:tub"]
    tube = open_front_shell(dims, [{**cavities[0], "inner_depth_m": dims[0]}])  # no back wall
    assert interior_cavity_findings(tube.vertices, tube.faces, cavities) == ["body_interior_cavity_back_open:tub"]
    shallow = open_front_shell(dims, [{**cavities[0], "inner_depth_m": 0.2}])
    assert interior_cavity_findings(shallow.vertices, shallow.faces, cavities) == ["body_interior_cavity_closed:tub"]
    assert interior_cavity_findings(hollow.vertices, hollow.faces, []) == ["body_interior_cavity_unplanned"]


def test_required_part_check_names_each_missing_part():
    plan = plan_articulated_assembly(dishwasher())
    carried = {"body": {"body_front": "front_frame", "tub_interior": "tub_cavity", "left_side": "left_side_wall",
                        "right_side": "right_side_wall", "kickplate": "kickplate"},
               "door": {"door_outer": "outer_skin", "door_inner": "inner_liner", "handle": "handle",
                        "control_panel": "control_panel", "brand_label": "brand_label"},
               "upper_rack": {"upper_rack": "link"}, "lower_rack": {"lower_rack": "link"},
               "cutlery_basket": {"cutlery_basket": "link"}}
    assert required_part_findings(plan, carried) == []
    del carried["cutlery_basket"]
    carried["door"]["handle"] = "outer_skin"
    assert required_part_findings(plan, carried) == ["required_part_missing:cutlery_basket",
                                                     "required_part_missing:handle"]


PART_MASS = {"body": 30.0, "door": 6.0, "upper_rack": 1.5, "lower_rack": 1.5, "cutlery_basket": 0.5}
BOUNDS = {"static_friction": [0.3, 0.8], "dynamic_friction": [0.2, 0.6], "restitution": [0.0, 0.2]}


def _package(tmp_path, plan, *, body_mesh=None):
    requests, results, bounds = {}, {}, {}
    for part_id, spec in plan["parts"].items():
        dims = spec["dimensions_m"]
        mass = PART_MASS[part_id]
        envelope = math.prod(dims)
        mass_bounds = {"body": [10.0, 60.0], "door": [2.0, 12.0]}.get(part_id, [0.3, 4.0])
        mesh = (body_mesh if body_mesh is not None else open_front_shell(dims, plan["interior_cavities"])
                ) if part_id == "body" else None
        request, result, bound = _part(tmp_path / part_id, object_id=f"{IDENTITY['id']}__{part_id}", dimensions=dims,
                                       mass_kg=mass, density=(0.5 * mass / envelope, 2.0 * mass / envelope),
                                       bounds={"mass_kg": mass_bounds, **BOUNDS}, mesh=mesh)
        requests[part_id], results[part_id], bounds[part_id] = request, result, bound
    receipt = package_astra_articulated_candidate(requests=requests, authoring_results=results, plan=plan,
                                                  output_root=tmp_path / "packaged", physics_bounds=bounds)
    return receipt, bounds


def _seal(plan, receipt, bounds):
    completion = dict(receipt["physics_completion"])
    completion["metric_envelope_validation"] = {"status": "within_preregistered_metric_envelope"}
    completion["completion_digest"] = canonical_digest(completion, digest_field="completion_digest")
    graph = {"schema_version": "task_evaluation_articulated_replacement_graph.v1", "asset_id": IDENTITY["id"],
             "asset_version": IDENTITY["version"], "articulation_graph": articulation_graph_from_plan(plan),
             "task_joint_prim_path": completion["task_joint_prim_path"],
             "task_link_prim_path": completion["task_link_prim_path"],
             "fixed_base_body_prim_path": completion["fixed_base_body_prim_path"],
             "handle_prim_paths": completion["handle_prim_paths"],
             "handle_grasp_point_link_m": completion["handle_grasp_point_link_m"],
             "link_prim_paths": {row["link_id"]: row["prim_path"] for row in completion["links"]},
             "assembly_plan": dict(plan), "physics_bounds": bounds, "physics_authority_granted": False,
             "authoring_backend": "astra_cad_blender_v1"}
    asset = Path(receipt["asset"]["path"])
    authoring = {"schema_version": "task_evaluation_articulated_replacement_authoring_result.v1",
                 "status": "authored_candidate_pending_qualification", "asset_kind": "articulated_assembly",
                 "replacement_identity": dict(IDENTITY),
                 "output_usd": {"sha256": "sha256:" + hashlib.sha256(asset.read_bytes()).hexdigest(),
                                "size_bytes": asset.stat().st_size},
                 "candidate_physics_completion": completion, "physics_authority_granted": False, "result_digest": ""}
    authoring["result_digest"] = canonical_digest(authoring, digest_field="result_digest")
    return asset, graph, authoring


@pytest.mark.parametrize("hinge", ["bottom", "left"])
def test_dishwasher_packages_one_passive_revolute_door_and_passes_static_qualification(tmp_path, hinge):
    plan = plan_articulated_assembly(dishwasher(hinge))
    receipt, bounds = _package(tmp_path, plan)
    stage = Usd.Stage.Open(receipt["asset"]["path"])
    revolute = [p for p in stage.Traverse() if p.IsA(UsdPhysics.RevoluteJoint)]
    assert len(revolute) == 1 and not [p for p in stage.Traverse() if p.IsA(UsdPhysics.PrismaticJoint)]
    joint = UsdPhysics.RevoluteJoint(revolute[0])
    assert joint.GetAxisAttr().Get() == plan["task_joint"]["usd_axis"]
    assert joint.GetUpperLimitAttr().Get() == pytest.approx(math.degrees(1.4))
    assert [str(t) for t in joint.GetBody0Rel().GetTargets()] == ["/Asset/links/body"]
    drive = UsdPhysics.DriveAPI(revolute[0], "angular")
    assert drive.GetStiffnessAttr().Get() == 0.0 and drive.GetDampingAttr().Get() == pytest.approx(math.radians(0.5))
    # Closed, both joint frames sit on the hinge line.
    body_rest, door_rest = (np.array(row["rest_translation_m"]) for row in plan["links"][:2])
    anchor = np.array(plan["task_joint"]["anchor_asset_frame_m"])
    assert np.allclose(np.array(joint.GetLocalPos0Attr().Get()) + body_rest, anchor, atol=1e-6)
    assert np.allclose(np.array(joint.GetLocalPos1Attr().Get()) + door_rest, anchor, atol=1e-6)
    fixed = sorted(p.GetName() for p in stage.Traverse() if p.IsA(UsdPhysics.FixedJoint))
    assert fixed == ["cutlery_basket_fixed", "lower_rack_fixed", "upper_rack_fixed"]
    carried = json.loads(stage.GetPrimAtPath("/Asset/links/door").GetCustomDataByKey(REQUIRED_PARTS_ATTRIBUTE))
    assert carried == {"brand_label": "brand_label", "control_panel": "control_panel", "door_inner": "inner_liner",
                       "door_outer": "outer_skin", "handle": "handle"}
    completion = receipt["physics_completion"]
    assert completion["fixed_base_body_prim_path"] == "/Asset/links/body"
    assert completion["collision_dimensions_m"] == pytest.approx(plan["closed_collision_dimensions_m"], abs=1e-6)
    assert completion["interior_cavity_check"]["cavity_ids"] == ["tub"]
    asset, graph, authoring = _seal(plan, receipt, bounds)
    result = qualify_scene_configuration_articulated_asset_static(
        asset_path=asset, graph_spec=graph, authoring_receipt=authoring,
        replacement_identity=IDENTITY, output_path=tmp_path / "static.json")
    assert result["checks"] == dict(ARTICULATED_STATIC_CHECKS)
    assert result["checks"]["articulated_required_parts_present"] is True
    assert result["checks"]["articulated_body_interior_cavity"] is True
    assert result["task_joint"]["joint_type"] == "revolute" and result["task_joint"]["limits"] == pytest.approx([0.0, 1.4])


def test_solid_body_or_unplanned_part_is_refused_before_any_usd(tmp_path):
    plan = plan_articulated_assembly(dishwasher())
    dims = plan["parts"]["body"]["dimensions_m"]
    solid = trimesh.creation.box(extents=dims)
    solid.apply_translation([0, 0, dims[2] / 2])
    with pytest.raises(AssetAuthoringError, match="articulated_body_interior_cavity_closed:tub"):
        _package(tmp_path / "solid", plan, body_mesh=solid)
    assert not (tmp_path / "solid" / "packaged").exists()
    orphan = copy.deepcopy(plan)
    orphan["required_parts"][-1]["link_id"] = "dishwasher_salt_hopper"
    with pytest.raises(AssetAuthoringError, match="required_part_unplanned:cutlery_basket"):
        _package(tmp_path / "orphan", orphan)


def test_static_qualification_refuses_a_hollow_plan_the_bytes_do_not_have(tmp_path):
    plan = plan_articulated_assembly(dishwasher())
    receipt, bounds = _package(tmp_path, plan)
    asset, graph, authoring = _seal(plan, receipt, bounds)
    stage = Usd.Stage.Open(str(asset))
    assert stage.GetPrimAtPath("/Asset/links/body/collision/FinalVisualShape")
    # A plan claiming a deeper cavity and an extra part that the authored bytes lack fails both new checks.
    deeper = copy.deepcopy(plan)
    deeper["interior_cavities"][0]["inner_depth_m"] = 0.9
    deeper["required_parts"].append({"part_id": "spray_arm", "label": "Spray arm", "role": "fixed_interior",
                                     "observed_frame_ids": [], "link_id": "body", "feature": "tub_cavity"})
    graph["assembly_plan"] = deeper
    with pytest.raises(TaskEvaluationSceneConfigurationStaticQualificationError) as error:
        qualify_scene_configuration_articulated_asset_static(
            asset_path=asset, graph_spec=graph, authoring_receipt=authoring,
            replacement_identity=IDENTITY, output_path=tmp_path / "static.json")
    assert "replacement_body_interior_cavity_closed:tub" in error.value.codes
    assert "replacement_required_part_missing:spray_arm" in error.value.codes
