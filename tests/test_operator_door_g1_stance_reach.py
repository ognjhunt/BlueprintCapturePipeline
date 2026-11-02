"""The door accepts the Pipeline's advisory G1 stance-reach note, and nothing stronger."""
from __future__ import annotations

import pytest

import tests.test_operator_door_public_policy_registration  # noqa: F401  (puts the door package on sys.path)
from operator_door import native_g1_registration_contract as contract
from blueprint_pipeline.native_g1_scene_reach_probe import STANCE_RULE, g1_stance_static_reach

STANCE = {"position_world_m": [0.0, 0.0, 0.79], "orientation_xyzw": [0.0, 0.0, 0.0, 1.0]}


def test_the_pipelines_own_records_are_accepted():
    _index, record = g1_stance_static_reach([STANCE], start=[0.3, -0.2, 0.9], target=[0.35, -0.1, 0.9], task_hand="right")
    assert contract.STANCE_REACH_RULE == STANCE_RULE
    assert contract.stance_reach_valid(record)
    assert contract.stance_reach_valid({"rule": STANCE_RULE, "status": "unassessed", "reason": "g1_stance_reach_inputs_invalid"})


@pytest.mark.parametrize("change", [
    {"claim_ceiling": "measured_robot_calibration"},
    {"rule": "measured_reach"},
    {"interpretation": "physically_validated"},
    {"within_nominal_arm_span": "yes"},
])
def test_a_stronger_or_unknown_claim_is_refused(change):
    _index, record = g1_stance_static_reach([STANCE], start=[0.3, -0.2, 0.9], target=[0.35, -0.1, 0.9], task_hand="right")
    assert not contract.stance_reach_valid({**record, **change})
    assert not contract.stance_reach_valid({"rule": STANCE_RULE, "status": "unassessed", "reason": "x", "extra": True})
