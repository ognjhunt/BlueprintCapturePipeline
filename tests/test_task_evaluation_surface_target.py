"""ADP-009D: future scene geometry uses the same marker and measured score."""
from copy import deepcopy
from itertools import product
import json
import math

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_evaluation_surface_target import (
    bind_native_surface_target, derive_surface_target, marker_for_surface_target,
    score_surface_target, validate_surface_target, surface_target_for_web,
    minimum_container_carry_height,
)


def target_fixture(**overrides):
    destination = {"kind": "green_region", "relation": "on", "visible_label": "green spot",
                   "position_world_m": [0.4, 0.0, 0.75], "orientation_xyzw": [0, 0, 0, 1],
                   "radius_m": 0.07, **overrides}
    return derive_surface_target(destination=destination,
        support={"sage_prim_path": "/Room/Table", "bounds_min_xyz_m": [-1, -1, 0],
                 "bounds_max_xyz_m": [1, 1, 0.75]},
        source_min=[-0.04, -0.03, 0.75], source_max=[0.04, 0.03, 0.89], support_instance_id="table-17")


BOUNDS = {"minimum": [-0.04, -0.03, -0.07], "maximum": [0.04, 0.03, 0.07]}


def test_web_target_digest_survives_integral_float_json_normalization():
    from blueprint_pipeline.decision_evidence_contracts import cross_runtime_canonical_json
    original = target_fixture()
    retained = deepcopy(original)
    published = surface_target_for_web(original)
    decoded = json.loads(cross_runtime_canonical_json(published))
    assert validate_surface_target(decoded) == decoded
    assert original == retained
    assert published["target_digest"] != original["target_digest"]
    changed = deepcopy(decoded)
    changed["surface_position_world_m"][0] += 0.01
    with pytest.raises(ValueError, match="surface_target_digest_invalid"):
        validate_surface_target(changed)


def episode():
    poses = [[0, 0, 0.82], [0, 0, 0.89], [0.4, 0, 0.89]] + [[0.4, 0, 0.82]] * 17
    return [{"step_index": i, "task_object_pose_world": [*p, 0, 0, 0, 1],
             "destination_pose_world": [.4, 0, .751, 0, 0, 0, 1],
             "task_contact_active": i in (1, 2), "support_contact_active": i not in (1, 2)}
            for i, p in enumerate(poses)]


def score(rows, **kwargs):
    return score_surface_target(target=target_fixture(), bounds=BOUNDS, samples=rows,
                                frequency_hz=15, minimum_lift_m=0.05, **kwargs)


def test_real_lift_and_one_second_stable_release_passes():
    result = score(episode())
    assert result["satisfied"] and result["readback_complete"]


def test_the_scorer_applies_the_same_start_rule_as_the_region():
    # The start footprint reaches into the disc (x 0.29-0.37 against 0.33-0.47); its centre does not.
    rows = episode()
    rows[0]["task_object_pose_world"] = [0.325, 0, 0.82, 0, 0, 0, 1]
    assert score(rows)["initially_outside"] and score(rows)["satisfied"]
    rows[0]["task_object_pose_world"] = [0.335, 0, 0.82, 0, 0, 0, 1]
    assert not score(rows)["initially_outside"] and not score(rows)["satisfied"]


def test_container_target_keeps_inside_relation_and_checks_whole_object_height():
    target = target_fixture(relation="inside", interior_bounds_world_m={
        "minimum": [.25, -.15, .75], "maximum": [.55, .15, .95]})
    assert target["placement_relation"] == "inside"
    assert surface_target_for_web(target)["interior_bounds_world_m"] == target["interior_bounds_world_m"]
    result = score_surface_target(target=target, bounds=BOUNDS, samples=episode(),
                                  frequency_hz=15, minimum_lift_m=.05)
    assert result["satisfied"] and result["whole_volume_contained"]
    # A centre and whole footprint inside the bin cannot make a tall object fit.
    tall = {"minimum": BOUNDS["minimum"], "maximum": [.04, .03, .2]}
    result = score_surface_target(target=target, bounds=tall, samples=episode(),
                                  frequency_hz=15, minimum_lift_m=.05)
    assert result["whole_footprint_contained"] and not result["whole_volume_contained"]
    assert not result["satisfied"]


def test_container_target_cannot_be_derived_from_a_mask_without_interior_geometry():
    with pytest.raises(ValueError, match="surface_target_interior_invalid"):
        target_fixture(relation="inside")


def test_carry_height_bounds_oriented_rim_and_subject_rotations():
    spec = {"destination_relation": "inside", "destination_pose_world": [0., 0., .3, 0., 0., 0., 1.],
            "destination_interior_bounds_body_frame_m": {"minimum": [-.1, -.1, 0.], "maximum": [.1, .1, .4]},
            "subject_collision_bounds_scoring_frame_m": BOUNDS,
            "start_pose_world": [0., 0., .1, 0., 0., 0., 1.], "destination_orientation_xyzw": [0., 0., 0., 1.]}
    assert minimum_container_carry_height(spec, clearance_m=.02) == pytest.approx(.79)
    spec["destination_orientation_xyzw"] = [1., 0., 0., 0.]
    expected = .7 + math.sqrt(.04**2 + .03**2 + .07**2) + .02
    assert minimum_container_carry_height(spec, clearance_m=.02) == pytest.approx(expected)


@pytest.mark.parametrize("defect", ["never_lift", "pushed", "outside_edge", "tilted", "moving",
                                    "no_contact_readback", "no_support", "skipped_frame", "nan", "short", "marker_shift"])
def test_invalid_or_failed_placements_never_pass(defect):
    rows = episode()
    if defect == "never_lift":
        for row in rows:
            row["task_object_pose_world"][2] = 0.82
    elif defect == "pushed":
        for row in rows:
            row["task_contact_active"] = False
    elif defect == "outside_edge":
        for row in rows[3:]:
            row["task_object_pose_world"][0] = 0.43  # Center inside, corner outside.
    elif defect == "tilted":
        for row in rows[3:]:
            row["task_object_pose_world"][3:] = [math.sin(math.radians(10)), 0, 0, math.cos(math.radians(10))]
    elif defect == "moving":
        for i, row in enumerate(rows[3:]):
            row["task_object_pose_world"][0] += (i % 2) * 0.003
    elif defect == "no_contact_readback":
        rows[-1].pop("support_contact_active")
    elif defect == "no_support":
        rows[-1]["support_contact_active"] = False
    elif defect == "skipped_frame":
        rows[-1]["step_index"] += 1
    elif defect == "nan":
        rows[-1]["task_object_pose_world"][0] = math.nan
    elif defect == "short":
        rows = rows[:8]
    elif defect == "marker_shift":
        rows[-1]["destination_pose_world"][0] += .001
    result = score(rows)
    assert not result["satisfied"]
    if defect in {"no_contact_readback", "skipped_frame", "nan", "short", "marker_shift"}:
        assert not result["readback_complete"]


@pytest.mark.parametrize("position,radius,reason", [([0, 0, .75], .07, "initial_overlap"),
    ([.99, 0, .75], .07, "outside_support"), ([.4, 0, .9], .07, "height_mismatch"),
    ([.4, 0, .75], .01, "does_not_fit")])
def test_bad_regions_refused_before_execution(position, radius, reason):
    with pytest.raises(ValueError, match=reason):
        target_fixture(position_world_m=position, radius_m=radius)


def test_a_region_may_overlap_the_start_footprint_but_never_its_centre():
    """Owner decision 2026-10-31: the start must not already satisfy the region, not stay wholly clear of it."""
    from blueprint_pipeline.task_evaluation_surface_target import region_excludes_start

    # Subject 8x6 cm at the origin; a 7 cm region 7.5 cm away reaches over its footprint, not its centre.
    assert target_fixture(position_world_m=[0.075, 0.0, 0.75])["surface_position_world_m"][0] == 0.075
    with pytest.raises(ValueError, match="initial_overlap"):
        target_fixture(position_world_m=[0.06, 0.0, 0.75])
    assert region_excludes_start([-.04, -.03, 0], [.04, .03, .1], [.075, 0, 0], .07)
    assert not region_excludes_start([-.04, -.03, 0], [.04, .03, .1], [.07, 0, 0], .07)


def test_an_estimated_support_widens_only_the_height_check_to_its_placement_uncertainty():
    """capture-447ae4f4: the region sat 1.2 cm under the estimated table's highest face (7.2 cm uncertainty)."""
    def derive(position, **tolerance):
        return derive_surface_target(
            destination={"kind": "green_region", "relation": "on", "visible_label": "green spot",
                         "position_world_m": position, "orientation_xyzw": [0, 0, 0, 1], "radius_m": 0.07},
            support={"sage_prim_path": "/Room/Table", "bounds_min_xyz_m": [-1, -1, 0], "bounds_max_xyz_m": [1, 1, 0.75]},
            source_min=[-0.04, -0.03, 0.75], source_max=[0.04, 0.03, 0.89], support_instance_id="table-17",
            **tolerance)

    with pytest.raises(ValueError, match="height_mismatch"):
        derive([0.4, 0.0, 0.738])
    assert derive([0.4, 0.0, 0.738], height_tolerance_m=0.072)["surface_position_world_m"][2] == 0.738
    with pytest.raises(ValueError, match="height_mismatch"):
        derive([0.4, 0.0, 0.65], height_tolerance_m=0.072)
    for tolerance in (0.001, float("nan"), True, "0.07"):
        with pytest.raises(ValueError, match="height_tolerance_invalid"):
            derive([0.4, 0.0, 0.75], height_tolerance_m=tolerance)


def test_thresholds_are_task_inputs_and_marker_is_bound():
    target = target_fixture(success={"stable_seconds": 2.0, "maximum_tilt_rad": 0.1})
    assert target["stable_seconds"] == 2
    assert marker_for_surface_target(target)["surface_position_world_m"] == [0.4, 0, .75]
    changed = deepcopy(target)
    changed["radius_m"] *= 2
    with pytest.raises(ValueError, match="digest"):
        validate_surface_target(changed)


def test_explicit_full_tilt_domain_accepts_rotated_sphere_without_relaxing_settling():
    # The same conservative 80 mm collider envelope at ninety degrees is fully
    # contained and resting. Absolute orientation can be irrelevant to a task,
    # but geometry/labels alone must never override its sealed fifteen degrees.
    bounds = {"minimum": [-.04] * 3, "maximum": [.04] * 3}
    rows = episode()
    for row in rows:
        row["task_object_pose_world"][2] = .86 if row["task_contact_active"] else .79
    for row in rows[3:]:
        row["task_object_pose_world"][3:] = [math.sqrt(.5), 0, 0, math.sqrt(.5)]

    def result(target, samples):
        return score_surface_target(target=target, bounds=bounds, samples=samples,
                                    frequency_hz=15, minimum_lift_m=.05)

    default = target_fixture(visible_label="sphere destination")
    rejected = result(default, rows)
    assert default["maximum_tilt_rad"] == math.radians(15)
    assert not rejected["upright"] and not rejected["satisfied"]
    assert all(value for key, value in rejected.items() if key not in {"upright", "satisfied"})
    unrestricted = target_fixture(visible_label="sphere destination", success={"maximum_tilt_rad": math.pi})
    assert unrestricted["target_digest"] != default["target_digest"]
    assert result(unrestricted, rows)["satisfied"]
    # An explicit tighter owner criterion remains tighter, even for a sphere.
    assert not result(target_fixture(success={"maximum_tilt_rad": .1}), rows)["satisfied"]
    unsupported = deepcopy(rows)
    unsupported[-1]["support_contact_active"] = False
    assert not result(unrestricted, unsupported)["satisfied"]
    rolling = deepcopy(rows)
    rolling[-1]["task_object_pose_world"][3:] = [math.sin(.8), 0, 0, math.cos(.8)]
    assert not result(unrestricted, rolling)["stable_velocity"]


@pytest.mark.parametrize("tilt", [0, -.1, math.pi + .000001, math.nan, math.inf])
def test_surface_tilt_requires_a_finite_physical_angle(tilt):
    with pytest.raises(ValueError, match="surface_target_"):
        target_fixture(success={"maximum_tilt_rad": tilt})


def test_surface_target_never_accepts_unverified_shape_symmetry_as_a_criterion():
    with pytest.raises(ValueError, match="success_fields_invalid"):
        target_fixture(success={"rotational_symmetry": "sphere"})
    target = target_fixture()
    target["rotational_symmetry"] = "sphere"
    target["target_digest"] = canonical_digest(target, digest_field="target_digest")
    with pytest.raises(ValueError, match="surface_target_invalid"):
        validate_surface_target(target)


def test_published_surface_schemas_preserve_the_full_tilt_domain():
    from pathlib import Path
    import jsonschema

    root = Path(__file__).parents[1]
    paths = [root / "src/blueprint_pipeline/_catalog_schemas/rigid_task_success_contract.v1.schema.json"]
    paths.extend(root / "docs/schemas" / (name + ".schema.json") for name in (
        "rigid_task_success_contract.v1", "task_evaluation_configured_scene_revision.v1",
        "task_evaluation_launch_preparation_request.v1", "task_evaluation_retained_source_scene_preparation.v1",
        "task_evaluation_retained_source_astra_scene_preparation.v1"))
    target = target_fixture(success={"maximum_tilt_rad": math.pi})
    for path in paths:
        schema = json.loads(path.read_text())
        validator = jsonschema.Draft202012Validator({"$ref": "#/$defs/surface_target", "$defs": schema["$defs"]})
        assert not list(validator.iter_errors(target)), str(path)
        assert list(validator.iter_errors({**target, "maximum_tilt_rad": math.pi + .000001})), str(path)


def test_canonical_scorer_cannot_ignore_or_rebind_the_surface_criterion():
    from blueprint_pipeline.adp_task_scoring import (
        _compatibility_rigid_success_criteria, score_task_episode_from_spec,
        seal_rigid_task_success_contract, TaskNeutralScoringError,
    )
    from tests.test_adp_task_scoring import _rigid_v2_spec, _rigid_v2_sample
    spec = _rigid_v2_spec()
    target = target_fixture()
    spec.update(surface_target=target, visible_target_marker=marker_for_surface_target(target),
        subject_collision_bounds_scoring_frame_m=BOUNDS, control_frequency_hz=15,
        start_pose_world=[0, 0, .82, 0, 0, 0, 1], minimum_lift_m=.05,
        target_position_world_m=[.4, 0, .82], minimum_translation_m=.2, settle_window_samples=16,
        support_height_interval_m=[.815, .825], destination_position_bounds_world_m={
            "minimum": [.38, -.02, .815], "maximum": [.42, .02, .825]})
    rows = [{**_rigid_v2_sample(row["step_index"], row["task_object_pose_world"][:3]), **row} for row in episode()]
    with pytest.raises(TaskNeutralScoringError, match="surface_target_contract_required"):
        score_task_episode_from_spec(task_spec=spec, samples=rows)
    criteria = _compatibility_rigid_success_criteria(spec)
    criteria["surface_target"] = target
    spec["task_success_contract"] = seal_rigid_task_success_contract(task_spec=spec, site_id="fixture",
        task_id="fixture", author_source="task_owner", author_id="fixture", confirmation_status="confirmed",
        confirmed_by_team_id="fixture", criteria=criteria)
    report = score_task_episode_from_spec(task_spec=spec, samples=rows)
    assert report["task_succeeded"] is True, report
    assert report["surface_target_measurements"]["marker_pose_matches"] is True
    rows[-1]["destination_pose_world"][0] += .001
    report = score_task_episode_from_spec(task_spec=spec, samples=rows)
    assert report["status"] == "undetermined" and not report["task_succeeded"]
    spec["visible_target_marker"]["radius_m"] *= 2
    with pytest.raises(TaskNeutralScoringError, match="surface_target_binding_mismatch"):
        score_task_episode_from_spec(task_spec=spec, samples=rows)


def test_native_bounds_are_from_qualified_collider_not_source_label():
    spec = {"interaction_affordance": {"asset_root_from_scoring_frame": {
        "position_m": [0, 0, .07], "orientation_xyzw": [0, 0, 0, 1]}},
        "destination_position_tolerance_m": .02, "control_frequency_hz": 15}
    static = {"observed_structure": {"collision_bounds_body_frame_m": {
        "minimum": [-.04, -.03, 0], "maximum": [.04, .03, .14]}}}
    support = {"sage_prim_path": "/Room/Table", "top_z_m": .75}
    bound = bind_native_surface_target(task_spec=spec, target=target_fixture(), static=static, support=support)
    assert bound["subject_collision_bounds_scoring_frame_m"] == BOUNDS
    assert bound["target_position_world_m"] == pytest.approx([.4, 0, .82])
    assert bound["settle_window_samples"] == 16
    static["observed_structure"]["collision_bounds_body_frame_m"]["maximum"][0] = .2
    with pytest.raises(ValueError, match="does_not_fit"):
        bind_native_surface_target(task_spec=spec, target=target_fixture(), static=static, support=support)


@pytest.mark.parametrize("width,depth,height,shift", [(.08, .06, .14, 0), (.04, .05, .09, .2)])
def test_production_submission_supports_different_object_geometry_without_destination_asset(
        tmp_path, monkeypatch, width, depth, height, shift):
    from tests import test_task_evaluation_scene_configuration_submission as fixture_module
    center = [-2.03, -3.44, .275 + height/2]
    corners = [dict(zip("xyz", p, strict=True)) for p in product(
        [center[0]-width/2, center[0]+width/2], [center[1]-depth/2, center[1]+depth/2], [.275, .275+height])]
    monkeypatch.setattr(fixture_module, "BOOK_CORNERS", corners)
    fixture = fixture_module.production_fixture(tmp_path)
    task = json.loads(fixture["task_request"].read_text())
    task["destination"] = {"kind": "green_region", "relation": "on", "visible_label": "green spot",
        "position_world_m": [-2.03, -2.95+shift, .275], "orientation_xyzw": [0, 0, 0, 1], "radius_m": .07}
    task["subject"]["review_label"] = "small_opaque_object"
    task["instruction"] = "Pick up the small opaque object, place it on the green spot, release it, and move the gripper clear."
    task["request_digest"] = canonical_digest(task, digest_field="request_digest")
    fixture_module._write_json(fixture["task_request"], task)
    result = fixture_module._materialize(fixture, destination_simready_result_path=None)
    from pathlib import Path
    root = Path(result["staging_root"])
    request = json.loads((root / "scene_configuration_preparation_request.v1.json").read_text())
    template = json.loads((root / "configuration/task_template.v1.json").read_text())
    recipe = json.loads((root / "configuration/scene_construction_recipe.v1.json").read_text())
    assert "destination" not in request["task"] and "supplemental_destination" not in recipe
    assert request["task"]["surface_target"] == surface_target_for_web(template["surface_target"])
    native_target = template["surface_target"]
    assert native_target["target_digest"] == canonical_digest(native_target, digest_field="target_digest")
    success = json.loads((root / "configuration/task_success_criteria.v1.json").read_text())
    assert success["surface_target"] == native_target
    assert template["interaction_affordance"]["approach_unit_scoring_frame"] == [0, 0, 1]
    assert template["interaction_affordance"]["jaw_unit_scoring_frame"][2] == 0
    assert not (root / "destination").exists()


def test_retained_native_target_seal_survives_javascript_number_roundtrip():
    from blueprint_pipeline.decision_evidence_contracts import cross_runtime_canonical_json
    original = target_fixture(success={"stable_seconds": 2.0})
    decoded = json.loads(cross_runtime_canonical_json(original))
    assert type(decoded["stable_seconds"]) is int
    assert validate_surface_target(decoded) == decoded
    assert decoded["target_digest"] == original["target_digest"]
    for key in ("stable_seconds", "radius_m"):
        changed = deepcopy(decoded)
        changed[key] += 0.01
        with pytest.raises(ValueError, match="surface_target_digest_invalid"):
            validate_surface_target(changed)
