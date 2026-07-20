from __future__ import annotations

import numpy as np
import pytest

from blueprint_pipeline import g1_microwave_owned_training_seed as seed


def test_aligned_door_angle_schedule_places_pull_at_trajectory_end() -> None:
    schedule = seed.aligned_door_angle_schedule(6, [0.0, -0.1, -0.2])

    np.testing.assert_allclose(schedule, [0.0, 0.0, 0.0, 0.0, -0.1, -0.2])


def test_aligned_door_angle_schedule_rejects_non_opening_arc() -> None:
    with pytest.raises(
        ValueError,
        match="g1_microwave_training_seed_door_schedule_values_invalid",
    ):
        seed.aligned_door_angle_schedule(4, [0.0, -0.2, -0.1])


def test_split_sonic_training_actions_preserves_registered_fields() -> None:
    actions = np.arange(3 * 78, dtype=np.float32).reshape(3, 78)

    motion, left, right = seed.split_sonic_training_actions(actions)

    assert motion.shape == (3, 64)
    assert left.shape == (3, 7)
    assert right.shape == (3, 7)
    np.testing.assert_array_equal(motion, actions[:, :64])
    np.testing.assert_array_equal(left, actions[:, 64:71])
    np.testing.assert_array_equal(right, actions[:, 71:78])


def test_split_sonic_training_actions_rejects_wrong_width() -> None:
    with pytest.raises(
        ValueError,
        match="g1_microwave_training_seed_sonic_actions_invalid",
    ):
        seed.split_sonic_training_actions(np.zeros((2, 77), dtype=np.float32))
