"""ADP-009D: future scene geometry uses the same marker and measured score."""
from copy import deepcopy
from itertools import product
import json
import math

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_evaluation_surface_target import (
    bind_native_surface_target, derive_surface_target, marker_for_surface_target,
    score_surface_target, validate_surface_target,
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


def test_thresholds_are_task_inputs_and_marker_is_bound():
    target = target_fixture(success={"stable_seconds": 2.0, "maximum_tilt_rad": 0.1})
    assert target["stable_seconds"] == 2
    assert marker_for_surface_target(target)["surface_position_world_m"] == [0.4, 0, .75]
    changed = deepcopy(target)
    changed["radius_m"] *= 2
    with pytest.raises(ValueError, match="digest"):
        validate_surface_target(changed)


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
    assert request["task"]["surface_target"] == template["surface_target"]
    assert template["interaction_affordance"]["approach_unit_scoring_frame"] == [0, 0, 1]
    assert template["interaction_affordance"]["jaw_unit_scoring_frame"][2] == 0
    assert not (root / "destination").exists()
