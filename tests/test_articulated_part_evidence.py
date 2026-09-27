"""Template-placed parts say they are priors; an observed part box replaces its prior, clamped to its cavity.

Family- and name-agnostic: exercised on a side-hinged storage cabinet with
two fixed shelves, and on the existing hinged-door and stacked-drawer families.
"""
import copy
import math

import pytest

from blueprint_pipeline.task_evaluation_scene_configuration_submission_records import (
    articulated_stage_three_configuration,
)
from blueprint_pipeline.task_object_articulated_packaging import (
    OBSERVED_ESTIMATE_BASIS, PART_ESTIMATE_CAVITY_MARGIN_M, TEMPLATE_PRIOR_BASIS,
    package_astra_articulated_candidate, plan_articulated_assembly,
)
from blueprint_pipeline.task_object_astra_authoring import AssetAuthoringError
from tests.test_articulated_hinged_door_appliance import PHYSICS, _frame, dishwasher
from tests.test_task_object_articulated_packaging import _part, open_front_shell

IDENTITY = {"id": "website-subject-store", "version": "v1"}
CABINET_PARTS = [("body_front", "Cabinet front frame", "body"), ("cabinet_interior", "Cabinet interior", "body"),
                 ("left_side", "Left side panel", "body_feature"), ("door_outer", "Door panel", "task_part"),
                 ("handle", "Door knob", "door_feature"), ("upper_shelf", "Upper shelf", "fixed_interior"),
                 ("lower_shelf", "Lower shelf", "fixed_interior")]


def cabinet_frames(digests=("1", "2", "3")):
    return [
        _frame("f_front", "sha256:" + digests[0] * 64, "closed", ["body_front", "door_outer", "handle", "left_side"]),
        _frame("f_upper", "sha256:" + digests[1] * 64, "open", ["cabinet_interior", "upper_shelf"],
               view="interior", reason="upper shelf seen through the open door"),
        _frame("f_lower", "sha256:" + digests[2] * 64, "open", ["cabinet_interior", "lower_shelf", "door_outer"],
               view="interior", reason="lower shelf seen through the open door")]


def storage_cabinet(frames=None, estimates=None):
    """A side-hinged storage cabinet: hollow body, one door, two fixed shelves (not an appliance)."""
    mechanism = {"part_label": "cabinet door", "joint_type": "revolute", "estimated_usable_swing_rad": 1.5,
                 "estimated_front_normal_world": [0.0, -1.0, 0.0], "lock_status": "unknown",
                 "travel_authority": "object_prior_estimate"}
    value = articulated_stage_three_configuration(
        scene_id="scene-store", replacement_identity=IDENTITY, source_instance_id="cabinet",
        authoring_target="tall storage cabinet", source_min=[-0.3, -0.25, 0.0], source_max=[0.3, 0.25, 0.9],
        dimension_tolerance=0.1, physics_bounds=PHYSICS, mechanism=mechanism)
    frames = frames or cabinet_frames()
    seen = {part: [row["frame_id"] for row in frames if part in row["visible_parts"]] for part, _, _ in CABINET_PARTS}
    value.update(assembly_family="hinged_door_appliance", hinge_edge="left", reference_frames=frames,
                 required_parts=[{"part_id": part, "label": label, "role": role, "observed_frame_ids": seen[part]}
                                 for part, label, role in CABINET_PARTS],
                 body_depth={"value_m": 0.5, "basis": "interior_observed_open_state", "frame_ids": ["f_upper"]},
                 source_observation_kind="website_capture_frames", dimension_authority="estimated")
    if estimates is not None:
        value["part_extent_estimates"] = estimates
    value["required_output"]["fixed_part_mass_kg_bounds"] = [0.3, 4.0]
    return value


def _cavity(plan):
    [cavity] = plan["interior_cavities"]
    rest = next(row["rest_translation_m"] for row in plan["links"] if row["link_id"] == cavity["link_id"])
    x, y, z = (c + r for c, r in zip(cavity["opening_center_link_m"], rest))
    return ([x - cavity["inner_depth_m"], y - cavity["opening_width_m"] / 2, z - cavity["opening_height_m"] / 2],
            [x, y + cavity["opening_width_m"] / 2, z + cavity["opening_height_m"] / 2])


def _link_box(plan, link_id):
    size = plan["parts"][link_id]["dimensions_m"]
    rest = next(row["rest_translation_m"] for row in plan["links"] if row["link_id"] == link_id)
    return ([rest[0] - size[0] / 2, rest[1] - size[1] / 2, rest[2]],
            [rest[0] + size[0] / 2, rest[1] + size[1] / 2, rest[2] + size[2]])


def _estimate(part_id, frame_ids, lo, hi, uncertainty=(0.01, 0.01, 0.02)):
    return {"part_id": part_id, "basis": OBSERVED_ESTIMATE_BASIS, "frame_ids": list(frame_ids),
            "box_assembly_m": {"minimum": list(lo), "maximum": list(hi)}, "uncertainty_m": list(uncertainty)}


def test_template_placed_parts_are_labelled_priors_in_any_family():
    plan = plan_articulated_assembly(storage_cabinet())
    bases = plan["part_dimension_bases"]
    # Fixed interior parts and features are template-placed; whole body and task part are not.
    assert set(bases) == {"left_side", "handle", "upper_shelf", "lower_shelf"}
    assert {row["basis"] for row in bases.values()} == {TEMPLATE_PRIOR_BASIS}
    assert all(row["frame_ids"] == [] for row in bases.values())
    assert bases["upper_shelf"]["appearance_frame_ids"] == ["f_upper"]
    assert bases["lower_shelf"] == {"link_id": "lower_shelf", "feature": "link", "appearance_frame_ids": ["f_lower"],
                                    "basis": TEMPLATE_PRIOR_BASIS, "frame_ids": [],
                                    "prior": "hinged_door_appliance_family_template"}
    for link_id in ("upper_shelf", "lower_shelf"):
        assert "template prior of the assembly family, not observed" in plan["parts"][link_id]["description"]
    # The hinged-door appliance template labels the same way, fixtures on the body included.
    config = dishwasher()
    config["required_parts"].append({"part_id": "lower_spray_arm", "label": "Lower spray arm",
                                     "role": "fixed_interior", "observed_frame_ids": ["f_open"]})
    appliance = plan_articulated_assembly(config)["part_dimension_bases"]
    assert appliance["lower_spray_arm"]["feature"] == "lower_spray_arm" and appliance["lower_spray_arm"]["link_id"] == "body"
    assert {row["basis"] for row in appliance.values()} == {TEMPLATE_PRIOR_BASIS}
    # And the stacked-drawer template: a fixed drawer is an instance of the shared drawer solid.
    from tests.test_task_object_articulated_packaging import configuration as drawer_configuration
    drawers = drawer_configuration()
    drawers.update(assembly_family="stacked_drawer_cabinet", required_parts=[
        {"part_id": "carcass", "label": "Cabinet body", "role": "body", "observed_frame_ids": []},
        {"part_id": "middle_drawer", "label": "Middle drawer", "role": "task_part", "observed_frame_ids": []},
        {"part_id": "top_drawer", "label": "Top drawer", "role": "fixed_interior", "observed_frame_ids": []}])
    drawer_plan = plan_articulated_assembly(drawers)
    assert drawer_plan["part_dimension_bases"] == {"top_drawer": {
        "link_id": "drawer_0", "feature": "link", "appearance_frame_ids": [], "basis": TEMPLATE_PRIOR_BASIS,
        "frame_ids": [], "prior": "stacked_drawer_cabinet_family_template"}}
    assert "template prior" not in drawer_plan["parts"]["drawer"]["description"]


def test_observed_part_box_replaces_the_prior_and_is_clamped_inside_its_cavity():
    template = plan_articulated_assembly(storage_cabinet())
    lo, hi = _cavity(template)
    inside = ([lo[0] + 0.05, lo[1] + 0.03, lo[2] + 0.4], [hi[0] - 0.04, hi[1] - 0.03, lo[2] + 0.43])
    # The lower shelf's observed box pokes through the cavity floor and past its open front.
    poking = ([lo[0] + 0.05, lo[1] + 0.03, lo[2] - 0.05], [hi[0] + 0.08, hi[1] - 0.03, lo[2] + 0.02])
    plan = plan_articulated_assembly(storage_cabinet(estimates=[
        _estimate("upper_shelf", ["f_upper"], *inside), _estimate("lower_shelf", ["f_lower"], *poking)]))
    bases = plan["part_dimension_bases"]
    upper, lower = bases["upper_shelf"], bases["lower_shelf"]
    assert upper["basis"] == lower["basis"] == OBSERVED_ESTIMATE_BASIS
    assert upper["frame_ids"] == ["f_upper"] and lower["frame_ids"] == ["f_lower"]
    assert upper["uncertainty_m"] == [0.01, 0.01, 0.02] and upper["physical_measurement_proven"] is False
    assert upper["clamped_axes"] == [] and lower["clamped_axes"] == ["x", "z"]
    assert upper["cavity_id"] == lower["cavity_id"] == "tub"
    # Used verbatim when inside the cavity ...
    got = _link_box(plan, "upper_shelf")
    assert got[0] == pytest.approx(inside[0], abs=1e-5) and got[1] == pytest.approx(inside[1], abs=1e-5)
    # ... and clamped to the cavity (minus a margin) where it is not: clear of the closed door.
    margin = PART_ESTIMATE_CAVITY_MARGIN_M
    got = _link_box(plan, "lower_shelf")
    assert got[0][2] == pytest.approx(lo[2] + margin, abs=1e-5) and got[1][0] == pytest.approx(hi[0] - margin, abs=1e-5)
    assert got[1][2] == pytest.approx(poking[1][2], abs=1e-5)
    for link_id in ("upper_shelf", "lower_shelf"):
        box = _link_box(plan, link_id)
        assert all(lo[i] <= box[0][i] and box[1][i] <= hi[i] for i in range(3)), link_id
        feature = plan["parts"][link_id]["features"]["link"]
        assert [b - a for a, b in zip(feature["minimum"], feature["maximum"])] == pytest.approx(
            plan["parts"][link_id]["dimensions_m"], abs=1e-5)
    # The prior it replaced travels with it; the description states the basis, not the template prior.
    assert lower["replaced_template_prior"]["dimensions_m"] == template["parts"]["lower_shelf"]["dimensions_m"]
    assert "estimated from frames f_lower" in plan["parts"]["lower_shelf"]["description"]
    assert "clamped inside the cavity on x, z" in plan["parts"]["lower_shelf"]["description"]
    assert "template prior" not in plan["parts"]["lower_shelf"]["description"]
    # Features and the body are untouched.
    assert plan["parts"]["body"] == template["parts"]["body"] and plan["parts"]["door"] == template["parts"]["door"]
    assert bases["handle"]["basis"] == TEMPLATE_PRIOR_BASIS


@pytest.mark.parametrize("change, code", [
    (lambda rows: rows[0].update(part_id="handle"), "part_extent_estimates_invalid"),  # a feature, not its own link
    (lambda rows: rows[0].update(frame_ids=["f_front"]), "part_extent_estimates_invalid"),  # frame does not show it
    (lambda rows: rows[0].update(frame_ids=[]), "part_extent_estimates_invalid"),
    (lambda rows: rows[0].update(basis="measured"), "part_extent_estimates_invalid"),
    (lambda rows: rows[0].update(uncertainty_m=[0.01, -0.01, 0.0]), "part_extent_estimates_invalid"),
    (lambda rows: rows[0]["box_assembly_m"].update(maximum=[0.0, 0.0, math.nan]), "part_extent_estimates_invalid"),
    (lambda rows: rows[0]["box_assembly_m"].update(maximum=rows[0]["box_assembly_m"]["minimum"]),
     "part_extent_estimates_invalid"),
    (lambda rows: rows.append(copy.deepcopy(rows[0])), "part_extent_estimates_invalid"),
    (lambda rows: rows[0]["box_assembly_m"].update(minimum=[2.0, 2.0, 2.0], maximum=[3.0, 3.0, 3.0]),
     "part_extent_estimate_outside_cavity:upper_shelf"),
])
def test_part_extent_estimates_fail_closed(change, code):
    template = plan_articulated_assembly(storage_cabinet())
    rows = [_estimate("upper_shelf", ["f_upper"], *_link_box(template, "upper_shelf"))]
    change(rows)
    with pytest.raises(AssetAuthoringError, match="articulated_" + code):
        plan_articulated_assembly(storage_cabinet(estimates=rows))


def test_estimates_need_a_captured_part_planned_as_its_own_link():
    template = plan_articulated_assembly(storage_cabinet())
    box = _link_box(template, "upper_shelf")
    created = storage_cabinet(estimates=[_estimate("upper_shelf", ["f_upper"], *box)])
    created.update(source_observation_kind="not_captured_created_from_description", reference_frames=[],
                   body_depth={"value_m": 0.5, "basis": "owner_or_catalog_specified", "frame_ids": []})
    for row in created["required_parts"]:
        row["observed_frame_ids"] = []
    plan_articulated_assembly({**created, "part_extent_estimates": []})
    with pytest.raises(AssetAuthoringError, match="articulated_part_extent_estimates_invalid"):
        plan_articulated_assembly(created)
    # A body fixture is a declared feature box, not its own link: an observed box has no seat.
    config = dishwasher()
    config["required_parts"].append({"part_id": "floor_filter", "label": "Filter", "role": "fixed_interior",
                                     "observed_frame_ids": ["f_open"]})
    config["part_extent_estimates"] = [_estimate("floor_filter", ["f_open"], [-0.1, -0.05, 0.12], [0.0, 0.05, 0.16])]
    with pytest.raises(AssetAuthoringError, match="articulated_part_extent_estimate_unplaceable:floor_filter"):
        plan_articulated_assembly(config)
    # Nor is a fixed drawer that instances the shared drawer solid.
    from tests.test_task_object_articulated_packaging import configuration as drawer_configuration
    drawers = drawer_configuration()
    drawers.update(assembly_family="stacked_drawer_cabinet", reference_frames=[
        _frame("f_top", "sha256:" + "4" * 64, "closed", ["top_drawer"])], required_parts=[
        {"part_id": "carcass", "label": "Cabinet body", "role": "body", "observed_frame_ids": []},
        {"part_id": "middle_drawer", "label": "Middle drawer", "role": "task_part", "observed_frame_ids": []},
        {"part_id": "top_drawer", "label": "Top drawer", "role": "fixed_interior", "observed_frame_ids": ["f_top"]}],
        source_observation_kind="website_capture_frames",
        part_extent_estimates=[_estimate("top_drawer", ["f_top"], [-0.2, -0.15, 0.45], [0.2, 0.15, 0.6])])
    with pytest.raises(AssetAuthoringError, match="articulated_part_extent_estimate_unplaceable:top_drawer"):
        plan_articulated_assembly(drawers)


def test_packaging_receipt_records_each_parts_dimension_basis(tmp_path):
    template = plan_articulated_assembly(storage_cabinet())
    plan = plan_articulated_assembly(storage_cabinet(estimates=[
        _estimate("lower_shelf", ["f_lower"], *_link_box(template, "lower_shelf"))]))
    requests, results, bounds = {}, {}, {}
    masses = {"body": 25.0, "door": 5.0, "upper_shelf": 1.0, "lower_shelf": 1.0}
    limits = {"body": [10.0, 60.0], "door": [2.0, 12.0]}
    for part_id, spec in plan["parts"].items():
        dims, mass = spec["dimensions_m"], masses[part_id]
        requests[part_id], results[part_id], bounds[part_id] = _part(
            tmp_path / part_id, object_id=f"{IDENTITY['id']}__{part_id}", dimensions=dims, mass_kg=mass,
            density=(0.5 * mass / math.prod(dims), 2.0 * mass / math.prod(dims)),
            bounds={"mass_kg": limits.get(part_id, [0.3, 4.0]), "static_friction": [0.3, 0.8],
                    "dynamic_friction": [0.2, 0.6], "restitution": [0.0, 0.2]},
            mesh=open_front_shell(dims, plan["interior_cavities"]) if part_id == "body" else None)
    receipt = package_astra_articulated_candidate(requests=requests, authoring_results=results, plan=plan,
                                                  output_root=tmp_path / "packaged", physics_bounds=bounds)
    recorded = receipt["physics_completion"]["part_dimension_bases"]
    assert recorded == plan["part_dimension_bases"]
    assert recorded["lower_shelf"]["basis"] == OBSERVED_ESTIMATE_BASIS
    assert recorded["upper_shelf"]["basis"] == TEMPLATE_PRIOR_BASIS
