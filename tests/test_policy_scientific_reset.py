from copy import deepcopy

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.policy_scientific_reset import (
    compare_reset_readbacks, seal_reset_readback, validate_reset_readback,
)


def snapshot(candidate="pi05_droid", **changes):
    observed = {"robot": {"joints": [0.0] * 7, "velocities": [0.0] * 7},
        "objects": {"object": {"pose": [1., 2., 3.], "mass": 1., "friction": .4}},
        "scene_assets": {"appearance": {"transform": [1., 0., 0., 0.]}},
        "cameras": {"external": {"pose": [0., 1., 2.], "resolution": [64, 32]}},
        "physics": {"dt": 1 / 120, "gravity": [0., 0., -9.81]},
        "lighting": {"key": {"intensity": 1500.0}},
        "colliders": {"/scene/object": {"enabled": True}},
        "contacts": {"support": [0., 0., 9.81]}}
    observed.update(changes)
    return seal_reset_readback(binding={"candidate_id": candidate, "cell_id": "cell-0", "seed": 31,
        "task_spec_digest": "sha256:" + "1" * 64, "resolved_scenario_digest": "sha256:" + "2" * 64},
        observed=observed, sources={key: "native_test_spy" for key in observed}, gaps=[])


def test_pair_compares_native_science_not_candidate_artifact_hash():
    left, right = snapshot(), snapshot("groot_n17_droid")
    assert left["receipt_digest"] != right["receipt_digest"]
    assert compare_reset_readbacks(left, right)["status"] == "matched"


@pytest.mark.parametrize("channel", ["robot", "objects", "scene_assets", "cameras", "physics", "lighting", "colliders", "contacts"])
def test_every_scientific_channel_remains_in_pair_comparison(channel):
    left = snapshot()
    observed = deepcopy(left["observed"])
    observed[channel]["changed"] = True
    right = snapshot("groot_n17_droid", **observed)
    assert compare_reset_readbacks(left, right)["status"] == "mismatch"
    assert compare_reset_readbacks(left, right)["comparison_eligible"] is False


def test_missing_native_channels_are_explicit_not_empty_equality():
    full = snapshot()
    left = seal_reset_readback(binding=full["binding"], observed={"cameras": full["observed"]["cameras"]},
        sources={"cameras": "sensor"}, gaps=[])
    assert left["complete"] is False
    assert compare_reset_readbacks(left, left)["status"] == "unverified"


def test_tolerance_must_be_frozen_equally_and_is_path_specific():
    left, right = snapshot(), snapshot("groot_n17_droid")
    right["observed"]["objects"]["object"]["mass"] += .001
    def seal(row, tolerance):
        return seal_reset_readback(binding=row["binding"], observed=row["observed"], sources=row["sources"],
            gaps=[], tolerances=tolerance)
    tolerance = {"/objects/object/mass": .002}
    assert compare_reset_readbacks(seal(left, tolerance), seal(right, tolerance))["status"] == "matched"
    assert compare_reset_readbacks(seal(left, {}), seal(right, tolerance))["status"] == "mismatch"


def test_resealed_completeness_claim_and_nonfinite_values_refuse():
    value = snapshot()
    value["gaps"] = ["unmeasured"]
    value["receipt_digest"] = canonical_digest(value, digest_field="receipt_digest")
    with pytest.raises(ValueError):
        validate_reset_readback(value)
    with pytest.raises(ValueError):
        snapshot(physics={"dt": float("nan")})


def test_native_reader_measures_physx_values_and_does_not_treat_static_assets_as_bodies():
    from types import SimpleNamespace as NS
    import numpy as np
    from blueprint_pipeline.policy_scientific_reset import read_native_reset_channels
    pose = np.array([[1., 2., 3., 0., 0., 0., 1.]])
    body = NS(data=NS(root_pose_w=pose, root_vel_w=np.zeros((1, 6))),
        root_physx_view=NS(get_masses=lambda: np.array([[1.25]]),
            get_inertias=lambda: np.ones((1, 9)), get_material_properties=lambda: np.array([[.5, .4, .0]])))
    robot = NS(data=NS(joint_pos=np.zeros((1, 7)), joint_vel=np.zeros((1, 7)),
        root_pose_w=pose, root_vel_w=np.zeros((1, 6)), joint_limits=np.array([[[-1., 1.]] * 7])))
    built = NS(env=NS(unwrapped=NS(scene={"robot": robot, "task_object": body, "room": object()})),
        plan={"objects": [{"name": "task_object", "object_type": "RIGID", "task_subject": True, "expected_mass": 999.},
                          {"name": "room", "object_type": "USD", "reset_state": {"root_pose_world": {}}}], "scenario": {}},
        scene_asset_names={"task_object": "task_object", "room": "room"}, contact_sensor_names={})
    episode = NS(read_control_observation_metadata=lambda: {"calibrations": {"external": {"resolution": [64, 32]}}})
    readback = read_native_reset_channels(built, episode)
    assert readback["observed"]["objects"]["task_object"]["masses"] == [[1.25]]
    assert readback["observed"]["objects"]["task_object"]["material_properties"] == [[.5, .4, .0]]
    assert "room" not in readback["observed"]["objects"]
    assert readback["gaps"]  # no USD stage or contact sensors in this deliberately bounded spy
