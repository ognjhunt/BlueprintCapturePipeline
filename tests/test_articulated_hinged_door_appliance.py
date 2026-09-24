"""A hinged-door appliance plans, builds and qualifies as a hollow body, one revolute door and fixed racks."""
import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest
import trimesh
from pxr import Gf, Usd, UsdGeom, UsdPhysics

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
    CAVITY_COLLISION_WALL_BOXES, REQUIRED_PARTS_ATTRIBUTE, TARGET_JOINT_ID, articulation_graph_from_plan,
    box_collision_piece, cavity_wall_boxes, collision_cavity_findings, interior_cavity_findings,
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
    assert stage.GetPrimAtPath("/Asset/links/body/collision/wall_back")
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
    assert "replacement_body_cavity_collision_closed:tub" in error.value.codes
    assert "replacement_required_part_missing:spray_arm" in error.value.codes


# --- collision that PhysX will actually use -----------------------------------

def test_body_collision_is_exact_wall_boxes_that_keep_the_tub_hollow(tmp_path):
    plan = plan_articulated_assembly(dishwasher())
    assert plan["cavity_collision_approximation"] == CAVITY_COLLISION_WALL_BOXES
    dims, cavities = plan["parts"]["body"]["dimensions_m"], plan["interior_cavities"]
    bounds = {"minimum": [-dims[0] / 2, -dims[1] / 2, 0.0], "maximum": [dims[0] / 2, dims[1] / 2, dims[2]]}
    walls = cavity_wall_boxes(bounds, cavities)
    boxes = [box_collision_piece(row["center_m"], row["size_m"]) for row in walls]
    assert collision_cavity_findings(boxes, cavities) == []
    shell = open_front_shell(dims, cavities)
    # The render mesh is hollow, but PhysX's hull of it is a solid block and its
    # decomposition is unknown before cooking: neither proves the cavity.
    assert interior_cavity_findings(shell.vertices, shell.faces, cavities) == []
    mesh = {"vertices": shell.vertices.tolist(), "faces": shell.faces.tolist()}
    assert collision_cavity_findings([{**mesh, "approximation": "convexHull"}], cavities) == [
        "body_cavity_collision_closed:tub"]
    assert collision_cavity_findings([{**mesh, "approximation": "convexDecomposition"}], cavities) == [
        "body_cavity_collision_approximation_unproven:convexDecomposition"]
    assert collision_cavity_findings([{**mesh, "approximation": "none"}], cavities) == [
        "body_cavity_collision_approximation_unproven:none"]
    assert collision_cavity_findings([b for b, w in zip(boxes, walls) if w["name"] != "wall_back"],
                                     cavities) == ["body_cavity_collision_back_open:tub"]
    with pytest.raises(AssetAuthoringError, match="body_cavity_collision_infeasible"):
        cavity_wall_boxes(bounds, [{**cavities[0], "opening_width_m": dims[1]}])
    receipt, _ = _package(tmp_path, plan)
    stage = Usd.Stage.Open(receipt["asset"]["path"])
    colliders = sorted(str(p.GetPath()) for p in stage.Traverse()
                       if p.HasAPI(UsdPhysics.CollisionAPI) and str(p.GetPath()).startswith("/Asset/links/body/"))
    assert colliders == [f"/Asset/links/body/collision/wall_{n}" for n in ("back", "ceiling", "floor", "left", "right")]
    assert all(stage.GetPrimAtPath(path).IsA(UsdGeom.Cube) for path in colliders)
    completion = receipt["physics_completion"]
    assert completion["cavity_collision"]["approximation"] == CAVITY_COLLISION_WALL_BOXES
    assert sorted(completion["cavity_collision"]["collider_prim_paths"]["body"]) == colliders
    body = next(row for row in completion["links"] if row["link_id"] == "body")
    assert body["collision_bounds_link_frame_m"]["maximum"] == pytest.approx(bounds["maximum"], abs=1e-9)


@pytest.mark.parametrize("approximation, code", [
    ("convexHull", "replacement_body_cavity_collision_closed:tub"),
    ("convexDecomposition", "replacement_body_cavity_collision_approximation_unproven:convexDecomposition"),
])
def test_static_qualification_refuses_a_body_collider_that_fills_the_tub(tmp_path, approximation, code):
    from pxr import Sdf, UsdShade, UsdUtils
    plan = plan_articulated_assembly(dishwasher())
    receipt, bounds = _package(tmp_path, plan)
    flat = Path(receipt["asset"]["path"]).parent / "astra_articulated_candidate.flat.usdc"
    stage = Usd.Stage.Open(str(flat))
    for name in ("back", "ceiling", "floor", "left", "right"):
        stage.RemovePrim(f"/Asset/links/body/collision/wall_{name}")
    shell = open_front_shell(plan["parts"]["body"]["dimensions_m"], plan["interior_cavities"])
    mesh = UsdGeom.Mesh.Define(stage, "/Asset/links/body/collision/FinalVisualShape")
    mesh.CreatePointsAttr([Gf.Vec3f(*v) for v in shell.vertices.tolist()])
    mesh.CreateFaceVertexCountsAttr([3] * len(shell.faces))
    mesh.CreateFaceVertexIndicesAttr(shell.faces.reshape(-1).tolist())
    mesh.CreatePurposeAttr("guide")
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr(approximation)
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(
        UsdShade.Material(stage.GetPrimAtPath("/Asset/Looks/ReviewedPhysics_body")),
        UsdShade.Tokens.weakerThanDescendants, "physics")
    mesh.GetPrim().SetCustomDataByKey("blueprint:articulatedReplacement:provenance", "generated_candidate_geometry")
    edited = tmp_path / "edited.usdc"
    stage.GetRootLayer().Export(str(edited))
    asset = tmp_path / "filled_body.usdz"
    assert UsdUtils.CreateNewUsdzPackage(Sdf.AssetPath(str(edited)), str(asset))
    receipt = {**receipt, "asset": {"path": str(asset)}}
    asset, graph, authoring = _seal(plan, receipt, bounds)
    with pytest.raises(TaskEvaluationSceneConfigurationStaticQualificationError) as error:
        qualify_scene_configuration_articulated_asset_static(
            asset_path=asset, graph_spec=graph, authoring_receipt=authoring,
            replacement_identity=IDENTITY, output_path=tmp_path / "static.json")
    assert code in error.value.codes


# --- native import: one path for body/door/racks and carcass/drawers -----------

def _qualified(tmp_path, hinge="bottom"):
    plan = plan_articulated_assembly(dishwasher(hinge))
    receipt, bounds = _package(tmp_path, plan)
    asset, graph, authoring = _seal(plan, receipt, bounds)
    static_path = tmp_path / "static.json"
    static = qualify_scene_configuration_articulated_asset_static(
        asset_path=asset, graph_spec=graph, authoring_receipt=authoring,
        replacement_identity=IDENTITY, output_path=static_path)
    return plan, asset, graph, static, static_path


def _imported_structure(asset, static):
    from blueprint_pipeline import task_evaluation_scene_configuration_native_import_driver as driver
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World/Placement")
    stage.DefinePrim("/World/Placement/Replacement", "Xform").GetReferences().AddReference(str(asset), "/Asset")
    stage.Load()
    return driver._articulated_structure_observation(stage=stage, usd_physics=UsdPhysics, static_receipt=static)


def _native_environment(tmp_path, asset, static_path):
    from tests.test_task_evaluation_scene_configuration_native_import_driver import _environment
    from blueprint_pipeline.task_evaluation_scene_configuration_submission_records import stage_five_configuration
    tmp_path.mkdir(parents=True)
    environment = _environment(tmp_path)
    stage_input_path = Path(environment["BLUEPRINT_SCENE_CONFIGURATION_STAGE_INPUT"])
    stage_input = json.loads(stage_input_path.read_text())
    stage_input["configuration"] = {**stage_five_configuration(replacement_identity=IDENTITY, articulated=True),
                                    "schema_version": "replacement_native_import_qualification_configuration.v1"}
    stage_input_path.write_text(json.dumps(stage_input))
    rows = [{"role": role, "path": str(path), "size_bytes": path.stat().st_size,
             "digest": "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()}
            for role, path in (("statically_qualified_replacement_asset", Path(asset)),
                               ("static_qualification_receipt", static_path))]
    Path(environment["BLUEPRINT_SCENE_CONFIGURATION_STAGE_DEPENDENCIES"]).write_text(
        json.dumps([{"output_artifacts": rows}]))
    return environment


def _native_observation(structure, settled):
    joint = {**structure["task_joint"], "initial_task_joint_position": 0.0,
             "settled_task_joint_position": settled, "task_joint_returned_to_reset": True}
    state = {"position_m": [0.0, 0.0, 0.43], "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
             "task_joint_position_m": round(settled, 7)}
    repeat = {"asset_imported": True, "rigid_body_paths": structure["rigid_body_paths"],
              "articulation_root_paths": structure["articulation_root_paths"],
              "settle_measured_body_prim_path": structure["settle_measured_body_prim_path"],
              "collision_paths": structure["collision_paths"],
              "fixed_joint_prim_paths": structure["fixed_joint_prim_paths"],
              "link_physics_readback": structure["link_physics_readback"],
              "handle_prim_paths": structure["handle_prim_paths"], "task_joint": joint,
              "support_contact_observed": True, "contact_report_event_count": 5,
              "settle_translation_m": 0.001, "settle_rotation_rad": 0.002,
              "final_state": state, "final_state_digest": canonical_digest(state),
              "task_joint_trace_digest": "sha256:" + "b" * 64}
    return {"runtime_identity": {"engine_version": "6.0.1"},
            "repeats": [copy.deepcopy(repeat) for _ in range(3)]}


@pytest.mark.parametrize("hinge, token", [("bottom", "Y"), ("left", "Z")])
def test_dishwasher_passes_native_import_structure_and_seals_a_radian_reset(tmp_path, hinge, token):
    from blueprint_pipeline.task_evaluation_scene_configuration_native_import_driver import (
        TaskEvaluationSceneConfigurationNativeImportDriverError, execute_native_import_component,
    )
    _plan, asset, _graph, static, static_path = _qualified(tmp_path / "asset", hinge)
    structure = _imported_structure(asset, static)
    root = "/World/Placement/Replacement"
    assert set(structure["link_physics_readback"]) == {"body", "door", "upper_rack", "lower_rack", "cutlery_basket"}
    assert structure["settle_measured_body_prim_path"] == root + "/links/body"
    assert sorted(structure["fixed_joint_prim_paths"]) == [
        root + f"/joints/{name}_fixed" for name in ("cutlery_basket", "lower_rack", "upper_rack")]
    joint = structure["task_joint"]
    assert joint["task_joint_type"] == "revolute" and joint["task_joint_axis"] == token
    assert joint["task_joint_limits"] == pytest.approx([0.0, 1.4], abs=1e-6)  # radians, read from USD degrees
    assert joint["moving_link_prim_path"] == root + "/links/door"
    assert structure["handle_prim_paths"] == [root + "/links/door/collision/handle"]
    assert structure["link_physics_readback"]["body"]["collision_prim_paths"] == [
        root + f"/links/body/collision/wall_{n}" for n in ("back", "ceiling", "floor", "left", "right")]

    def runner(settled):
        return lambda *, observation_consumer, **kwargs: observation_consumer(_native_observation(structure, settled))

    # 0.015 rad sits inside the hinge's 0.02 rad graph reset tolerance.
    result = execute_native_import_component(
        environment=_native_environment(tmp_path / "ok", asset, static_path), native_runner=runner(0.015))
    runtime = json.loads(Path(result["artifacts"][0]["path"]).read_text())
    assert runtime["status"] == "qualified" and runtime["task_joint_coordinate_units"] == "rad"
    assert runtime["task_joint_readback"]["task_joint_type"] == "revolute"
    assert runtime["task_joint_readback"]["task_joint_axis"] == token
    assert runtime["task_joint_reset_numeric_readbacks"] == [0.015] * 3
    with pytest.raises(TaskEvaluationSceneConfigurationNativeImportDriverError, match="qualification_failed"):
        execute_native_import_component(
            environment=_native_environment(tmp_path / "ajar", asset, static_path), native_runner=runner(0.03))


def test_drawer_reset_tolerance_stays_five_millimetres(tmp_path):
    from tests.test_task_evaluation_scene_configuration_native_import_driver import (
        _articulated_environment, _articulated_observed, _native_runner,
    )
    from blueprint_pipeline.task_evaluation_scene_configuration_native_import_driver import (
        TaskEvaluationSceneConfigurationNativeImportDriverError, execute_native_import_component,
    )
    with pytest.raises(TaskEvaluationSceneConfigurationNativeImportDriverError, match="qualification_failed"):
        execute_native_import_component(
            environment=_articulated_environment(tmp_path),
            native_runner=_native_runner(_articulated_observed(settled_task_joint_position=0.015)))


# --- open/close task: radians, fraction of swing, jaw across the handle bar ---

def _revolute_case(tmp_path, hinge="bottom"):
    from tests.test_task_evaluation_articulated_open_close_native_adapter import _case
    plan, asset, graph, static, _ = _qualified(tmp_path / "asset", hinge)
    mechanism = {"part_label": "dishwasher door", "joint_type": "revolute", "estimated_usable_swing_rad": 1.4,
                 "estimated_front_normal_world": plan["assembly_frame"]["estimated_front_normal_world"],
                 "lock_status": "unknown", "travel_authority": "object_prior_estimate", "hinge_edge": hinge}
    return _case(tmp_path, assembly=(asset, graph, static), mechanism=mechanism)


@pytest.mark.parametrize("hinge, jaw", [("bottom", [0.0, 0.0, 1.0]), ("left", [0.0, 1.0, 0.0])])
def test_revolute_task_freezes_radian_thresholds_and_a_door_handle_grasp(tmp_path, hinge, jaw):
    from tests.test_task_evaluation_articulated_open_close_native_adapter import materialize_contract_and_plan
    from blueprint_pipeline.task_evaluation_articulated_open_close_native_adapter import (
        adapt_articulated_open_close_task_template,
    )
    case = _revolute_case(tmp_path, hinge)
    success = case["documents"]["scene.configured_revision.task_template.success_criteria"]
    assert success["joint_coordinate_units"] == "rad" and success["maximum_settled_target_speed"] == 0.03
    assert case["documents"]["scene.configured_revision.task_template.definition"][
        "interaction_affordance"]["jaw_unit_asset_frame"] == jaw
    adapted = adapt_articulated_open_close_task_template(
        configured_revision=case["configured"], materialized_references=case["references"])
    spec = adapted["native_task_definition"]["task_spec"]
    threshold = spec["executable_opening_threshold"]
    assert threshold["joint_type"] == "revolute" and threshold["coordinate_units"] == "rad"
    assert threshold["success_interval"] == pytest.approx([0.84, 1.4]) and threshold["opening_fraction_of_swing"] == 0.6
    assert threshold["qualified_joint_limits"] == pytest.approx([0.0, 1.4], abs=1e-6)
    plan = plan_articulated_assembly(dishwasher(hinge))
    affordance = spec["interaction_affordance"]
    assert affordance["contact_link_id"] == "door" and affordance["jaw_unit_asset_root"] == jaw
    assert affordance["handle_prim_paths"] == ["/Asset/links/door/collision/handle"]
    assert affordance["contact_point_link_m"] == plan["parts"]["door"]["handle"]["grasp_point_link_m"]
    assert affordance["pull_follows_arc_about_axis_asset_root"] == plan["task_joint"]["axis_asset_frame"]
    assert spec["movement_epsilon"] == pytest.approx(0.014, abs=1e-6)
    contract, arena = materialize_contract_and_plan(case, adapted, tmp_path)
    assert contract["task_sample_binding"]["native_coordinate_joint_ids"] == [TARGET_JOINT_ID]
    assert sorted(contract["task_sample_binding"]["fixed_joint_ids"]) == [
        "cutlery_basket_fixed", "lower_rack_fixed", "upper_rack_fixed"]
    assert arena["interaction_link_native_body_name"] == "door"
    assert arena["gpu_collision_qualification"]["status"] == "qualified"


def test_revolute_success_scores_in_radians(tmp_path):
    from blueprint_pipeline.adp_articulated_task_success_contract import compatibility_articulated_success_criteria
    from blueprint_pipeline.adp_task_scoring import TaskNeutralScoringError, score_articulated_task_episode
    from blueprint_pipeline.task_evaluation_articulated_open_close_native_adapter import (
        adapt_articulated_open_close_task_template,
    )
    case = _revolute_case(tmp_path)
    spec = adapt_articulated_open_close_task_template(
        configured_revision=case["configured"], materialized_references=case["references"],
    )["native_task_definition"]["task_spec"]
    criteria = compatibility_articulated_success_criteria(spec)
    assert criteria["opening"]["success_interval"] == pytest.approx([0.84, 1.4])
    assert criteria["opening"]["joint_hard_limits"] == pytest.approx([0.0, 1.4])
    assert criteria["reset"]["tolerance"] == 0.02 and criteria["hold"]["maximum_settled_target_speed"] == 0.03
    fixed = ["cutlery_basket_fixed", "lower_rack_fixed", "upper_rack_fixed"]

    def episode(angles, *, start=0.0):
        positions = [start, *angles]
        return score_articulated_task_episode(task_spec=spec, samples=[
            {"step_index": step,
             "joint_positions": {TARGET_JOINT_ID: angle, **dict.fromkeys(fixed, 0.0)},
             "joint_velocities_per_s": {TARGET_JOINT_ID: 0.0, **dict.fromkeys(fixed, 0.0)},
             "task_contact_active": step < len(positions) - 15, "joint_limit_violation": False,
             "containment_violation": False, "robot_collision_failure": False,
             "scene_collision_failure": False, "retreat_completed": step == len(positions) - 1}
            for step, angle in enumerate(positions)])

    ramp = [0.05 * k for k in range(1, 21)]
    opened = episode(ramp + [1.0] * 15)
    assert opened["task_succeeded"] is True and opened["outcome"] == "opened_and_settled"
    assert opened["thresholds"]["target_success_interval_rad"] == pytest.approx([0.84, 1.4])
    # 0.7 rad is half the swing: below the 0.6 fraction threshold.
    short = episode(ramp[:14] + [0.7] * 15)
    assert short["task_succeeded"] is False and short["outcome"] == "moved_below_threshold"
    assert episode(ramp + [1.0] * 15, start=0.015)["task_succeeded"] is True
    with pytest.raises(TaskNeutralScoringError, match="reset_readback_mismatch"):
        episode(ramp + [1.0] * 15, start=0.03)


def test_revolute_task_without_angular_units_is_refused(tmp_path, monkeypatch):
    import tests.test_task_evaluation_articulated_open_close_native_adapter as adapter_tests
    from blueprint_pipeline.task_evaluation_articulated_open_close_native_adapter import (
        TaskEvaluationArticulatedOpenCloseNativeAdapterError, adapt_articulated_open_close_task_template,
    )
    records = adapter_tests.articulated_open_close_task_records

    def metre_flavoured(**kwargs):
        template, success, execution = records(**kwargs)
        for document in (success, template["success"]):
            document.pop("joint_coordinate_units")
        return template, success, execution

    monkeypatch.setattr(adapter_tests, "articulated_open_close_task_records", metre_flavoured)
    case = _revolute_case(tmp_path)
    with pytest.raises(TaskEvaluationArticulatedOpenCloseNativeAdapterError, match="success_units_invalid"):
        adapt_articulated_open_close_task_template(
            configured_revision=case["configured"], materialized_references=case["references"])


def test_prismatic_task_records_keep_their_metre_fields():
    from blueprint_pipeline.task_evaluation_scene_configuration_submission_records import (
        articulated_open_close_task_records,
    )
    template, success, _ = articulated_open_close_task_records(
        task_identity={"id": "t", "version": "v1"}, object_identity=IDENTITY, start_center=[0, 0, 0.4],
        source_min=[-0.2, -0.2, 0.0], source_max=[0.2, 0.2, 0.8],
        mechanism={"part_label": "middle drawer", "joint_type": "prismatic", "estimated_usable_stroke_m": 0.3,
                   "estimated_front_normal_world": [1.0, 0.0, 0.0]},
        success={"control_frequency_hz": 15, "maximum_episode_seconds": 30,
                 "minimum_opening_fraction_of_estimated_stroke": 0.6, "minimum_hold_seconds": 1.0,
                 "maximum_retries": 0}, resolved_seed=1)
    assert success["maximum_settled_target_speed"] == 0.02 and success["locked_joint_motion_tolerance"] == 0.01
    assert "joint_coordinate_units" not in success and "pull_follows_hinge_arc" not in template["interaction_affordance"]
    assert template["interaction_affordance"]["jaw_unit_asset_frame"] == [0.0, 0.0, 1.0]


def test_one_unqualified_rack_is_the_lower_rack_but_two_stay_ambiguous():
    value = dishwasher()
    value["required_parts"] = [row for row in value["required_parts"]
                               if row["part_id"] not in {"upper_rack", "lower_rack", "cutlery_basket"}]
    value["required_parts"].append({"part_id": "rack", "label": "Dish rack", "role": "fixed_interior",
                                    "observed_frame_ids": ["f_open"]})
    plan = plan_articulated_assembly(value)
    rack = next(link for link in plan["links"] if link["link_id"] == "rack")
    assert rack["rest_translation_m"][2] < plan["parts"]["body"]["features"]["tub_cavity"]["minimum"][2] + 0.1
    value["required_parts"].append({"part_id": "rack_2", "label": "Another rack", "role": "fixed_interior",
                                    "observed_frame_ids": ["f_open"]})
    with pytest.raises(AssetAuthoringError, match="required_part_unplanned:rack"):
        plan_articulated_assembly(value)
