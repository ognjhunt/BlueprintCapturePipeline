"""Retreat must use native state, an oriented envelope and qualified direction."""
import copy
import math

import pytest

from blueprint_pipeline.adp_task_scoring import (
    TaskNeutralScoringError,
    score_task_episode_from_spec,
    seal_rigid_task_success_contract,
)
from tests.test_adp_task_scoring import _rigid_v2_sample, _rigid_v2_spec


def _fixture():
    spec = _rigid_v2_spec()
    pose = [1.15, 2.0, 0.8, 0.0, 0.0, 0.0, 1.0]
    spec.update(
        destination_relation="inside", destination_pose_world=pose,
        destination_position_bounds_destination_frame_m={
            "minimum": [-0.01]*3, "maximum": [0.01]*3},
        subject_collision_bounds_scoring_frame_m={
            "minimum": [-0.005, -0.005, -0.005], "maximum": [0.005]*3},
        destination_interior_bounds_body_frame_m={
            "minimum": [-0.02]*3, "maximum": [0.02]*3},
        destination_reset_translation_tolerance_m=0.002,
        destination_reset_rotation_tolerance_rad=0.01,
        retreat_clearance_m=0.05,
        interaction_affordance={"insertion_withdrawal_unit_world": [0.0, 0.0, 1.0]},
    )
    samples = [_rigid_v2_sample(i, pos) for i, pos in enumerate([
        [1., 2., .8], [1., 2., .84], [1.15, 2., .84],
        [1.15, 2., .8], [1.15, 2., .8], [1.15, 2., .8]])]
    for row in samples:
        row.update(destination_pose_world=pose, grasp_frame_position_world_m=[1.15, 2., .86])
    baseline = score_task_episode_from_spec(task_spec=spec, samples=samples)
    criteria = copy.deepcopy(baseline["task_success_contract"]["criteria"])
    criteria["retreat"] = {"mode": "required", "minimum_clearance_m": 0.05,
                           "withdrawal_unit_destination_frame": [0., 0., 1.]}
    criteria["terminal_task_contact"]["mode"] = "cleared"
    spec["task_success_contract"] = seal_rigid_task_success_contract(
        task_spec=spec, site_id="fixture_scene", task_id="fixture_task",
        author_source="task_owner", author_id="fixture_owner", confirmation_status="confirmed",
        confirmed_by_team_id="fixture_owner", criteria=criteria)
    return spec, samples


def test_retreat_scored_from_measured_grasp_clearance_for_complete_settle():
    spec, samples = _fixture()
    report = score_task_episode_from_spec(task_spec=spec, samples=samples)
    assert report["task_succeeded"] is True
    assert report["measurements"]["retreat"]["minimum_observed_clearance_m"] == pytest.approx(.055)
    # Only the middle settle sample falls short. A terminal-point check would pass.
    samples[-2]["grasp_frame_position_world_m"][2] = .83
    report = score_task_episode_from_spec(task_spec=spec, samples=samples)
    assert report["failed_criteria"] == ["retreat"]
    assert report["task_succeeded"] is False


@pytest.mark.parametrize("field", ["grasp_frame_position_world_m", "task_contact_active"])
def test_retreat_missing_readback_is_undetermined_even_with_claimed_or_commanded_success(field):
    spec, samples = _fixture()
    samples[-1].pop(field)
    samples[-1].update(retreat_completed=True, commanded_grasp_frame_position_world_m=[1., 2., 5.])
    report = score_task_episode_from_spec(task_spec=spec, samples=samples)
    assert report["status"] == "undetermined"
    assert report["task_succeeded"] is False
    assert report["measurements"]["retreat"]["readback_gap_steps"] == [5]


def test_retreat_uses_subject_orientation_and_destination_qualified_axis():
    spec, samples = _fixture()
    # Rotate destination 90deg around Y: its local withdrawal Z becomes world X.
    q = [0., math.sqrt(.5), 0., math.sqrt(.5)]
    spec["destination_pose_world"][3:] = q
    spec["interaction_affordance"]["insertion_withdrawal_unit_world"] = [1., 0., 0.]
    for row in samples:
        row["destination_pose_world"] = spec["destination_pose_world"]
        row["grasp_frame_position_world_m"] = [1.21, 2., .8]
    report = score_task_episode_from_spec(task_spec=spec, samples=samples)
    assert report["criteria_satisfied"]["retreat"] is True
    # An elongated subject rotates its long axis toward the gripper; center unchanged.
    spec["subject_collision_bounds_scoring_frame_m"]["minimum"][2] = -.015
    spec["subject_collision_bounds_scoring_frame_m"]["maximum"][2] = .015
    for row in samples[-3:]:
        row["task_object_pose_world"][3:] = q
    report = score_task_episode_from_spec(task_spec=spec, samples=samples)
    assert report["criteria_satisfied"]["retreat"] is False
    assert report["measurements"]["retreat"]["minimum_observed_clearance_m"] == pytest.approx(.045)


@pytest.mark.parametrize("mutation", ["direction", "clearance"])
def test_retreat_rejects_unqualified_direction_or_changed_clearance(mutation):
    spec, samples = _fixture()
    if mutation == "direction":
        spec["interaction_affordance"]["insertion_withdrawal_unit_world"] = [1., 0., 0.]
    else:
        spec["retreat_clearance_m"] = .01
    with pytest.raises(TaskNeutralScoringError, match="retreat_binding_mismatch"):
        score_task_episode_from_spec(task_spec=spec, samples=samples)


def test_owner_contract_materializer_seals_exact_configured_retreat_and_temporal_limits():
    from blueprint_pipeline.task_evaluation_rigid_owner_contract import (
        materialize_configured_owner_success_contract,
    )
    from blueprint_pipeline.rigid_task_success_contract_schema import rigid_task_success_contract_schema
    import jsonschema

    spec, samples = _fixture()
    spec["configured_success_criteria"] = {
        "owner_success_contract_required": True, "minimum_lift_m": .02,
        "retreat_clearance_m": .05, "drop_minimum_fall_m": .005,
        "maximum_task_contact_force_n": 10., "forbidden_contact_classes": ["robot_background"],
        "maximum_retries": 0, "maximum_regrasps": 0,
        "robot_workspace_position_bounds_world_m": {"minimum": [0., 0., 0.], "maximum": [3., 3., 3.]},
        "collision_failure_minimum_force_n": 1.0,
    }
    spec["robot_workspace_position_bounds_world_m"] = spec["configured_success_criteria"]["robot_workspace_position_bounds_world_m"]
    spec["collision_failure_minimum_force_n"] = 1.0
    spec["configured_owner_authority"] = {
        "confirmation_status": "confirmed", "accepted_by": "fixture_owner",
        "authority_reference": "fixture:owner-task-request",
    }
    contract = materialize_configured_owner_success_contract(spec, site_id="fixture_scene", task_id="fixture_task")
    jsonschema.validate(contract, rigid_task_success_contract_schema())
    assert contract["criteria"]["retreat"]["minimum_clearance_m"] == .05
    assert contract["criteria"]["temporal_invariants"]["no_drop"] == {"mode": "required", "minimum_fall_m": .005}
    assert contract["criteria"]["temporal_invariants"]["maximum_regrasps"] == 0
    spec["configured_success_criteria"].pop("retreat_clearance_m")
    with pytest.raises(TaskNeutralScoringError, match="explicit_field_missing:retreat_clearance_m"):
        materialize_configured_owner_success_contract(spec, site_id="fixture_scene", task_id="fixture_task")


def test_owner_contract_materializer_never_invents_confirmation_from_provider_authority():
    from blueprint_pipeline.task_evaluation_rigid_owner_contract import materialize_configured_owner_success_contract

    spec, _ = _fixture()
    spec["configured_success_criteria"] = {"owner_success_contract_required": True}
    with pytest.raises(TaskNeutralScoringError, match="authority_missing"):
        materialize_configured_owner_success_contract(spec, site_id="fixture_scene", task_id="fixture_task")
