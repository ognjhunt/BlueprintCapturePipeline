from __future__ import annotations

from blueprint_pipeline import g1_microwave_lerobot_materialization as materialize
from blueprint_pipeline.g1_sonic_motion_token_conversion import (
    SOURCE_ACTION_JOINT_NAMES,
)


def test_minimal_sonic_features_match_training_contract() -> None:
    features = materialize.minimal_sonic_features()

    assert set(features) == {
        "observation.images.ego_view",
        "observation.state",
        "observation.projected_gravity",
        "action.motion_token",
        "teleop.left_hand_joints",
        "teleop.right_hand_joints",
    }
    assert features["observation.images.ego_view"]["shape"] == [480, 640, 3]
    assert features["observation.state"]["shape"] == (43,)
    assert features["observation.state"]["names"] == list(
        SOURCE_ACTION_JOINT_NAMES
    )
    assert features["action.motion_token"]["shape"] == (64,)
    assert features["teleop.left_hand_joints"]["shape"] == (7,)
    assert features["teleop.right_hand_joints"]["shape"] == (7,)


def test_pinned_materialization_versions_are_exact() -> None:
    assert materialize.PINNED_GEAR_SONIC_REVISION == (
        "6d8e931b9b10a4db2d8e7aba3ad6d5da3529ff3b"
    )
    assert materialize.PINNED_LEROBOT_REVISION == (
        "a445d9c9da6bea99a8972daa4fe1fdd053d711d2"
    )
    assert materialize.PINNED_DATASETS_VERSION == "3.6.0"
