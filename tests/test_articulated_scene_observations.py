"""The five observed predicates, and what each of them refuses."""

from __future__ import annotations

import pytest

from blueprint_pipeline.articulated_scene_observations import (
    ArticulatedSceneObservationError,
    build_scene_observations,
    max_contact_force_n,
)


AUTHORED_BASE = (1.9742142, 1.4792181, 0.0)
HANDLE = (1.62, 1.83, 1.02)


def _sources(**overrides):
    base = {
        "read_task_contact_forces": lambda: [[[0.0, 0.0, 0.0]]],
        "read_robot_contact_forces": lambda: [[[0.0, 0.0, 0.0]]],
        "read_scene_contact_forces": lambda: [[[0.0, 0.0, 0.0]]],
        "read_task_object_base_position_m": lambda: AUTHORED_BASE,
        "authored_task_object_base_position_m": AUTHORED_BASE,
        "read_end_effector_position_m": lambda: HANDLE,
        "read_handle_position_m": lambda: HANDLE,
    }
    base.update(overrides)
    return build_scene_observations(**base)


def test_resting_noise_is_not_contact():
    """A sensor reports small forces constantly; 'any force' is always true."""

    observations = _sources(
        read_task_contact_forces=lambda: [[[0.05, 0.0, 0.02]]]
    )

    assert observations["read_task_contact_active"]() is False


def test_a_grasped_handle_registers_contact():
    observations = _sources(read_task_contact_forces=lambda: [[[0.0, 18.0, 0.0]]])

    assert observations["read_task_contact_active"]() is True


def test_fingers_touching_the_handle_are_not_a_robot_collision():
    """The gripper contacting the handle is the task working, not a failure."""

    observations = _sources(
        # body 0 is a finger, body 1 is a link; only the finger is loaded.
        read_robot_contact_forces=lambda: [[[0.0, 30.0, 0.0], [0.0, 0.0, 0.0]]],
        finger_body_indices=[0],
        non_finger_body_indices=[1],
    )

    assert observations["read_robot_collision_failure"]() is False


def test_an_elbow_striking_the_cabinet_is_a_robot_collision():
    observations = _sources(
        read_robot_contact_forces=lambda: [[[0.0, 0.0, 0.0], [40.0, 0.0, 0.0]]],
        finger_body_indices=[0],
        non_finger_body_indices=[1],
    )

    assert observations["read_robot_collision_failure"]() is True


def test_a_dragged_appliance_is_a_containment_violation():
    """Joint angle alone would still read like success.

    If the arm pulls the whole unit across the floor instead of swinging the
    door, the hinge angle can look identical to a clean open.
    """

    moved = (AUTHORED_BASE[0] + 0.15, AUTHORED_BASE[1], AUTHORED_BASE[2])
    observations = _sources(read_task_object_base_position_m=lambda: moved)

    assert observations["read_containment_violation"]() is True


def test_an_appliance_that_stayed_put_is_contained():
    settled = (AUTHORED_BASE[0] + 0.002, AUTHORED_BASE[1], AUTHORED_BASE[2])
    observations = _sources(read_task_object_base_position_m=lambda: settled)

    assert observations["read_containment_violation"]() is False


def test_retreat_needs_real_distance_from_the_handle():
    near = _sources(read_end_effector_position_m=lambda: (HANDLE[0] + 0.05, *HANDLE[1:]))
    far = _sources(read_end_effector_position_m=lambda: (HANDLE[0] - 0.6, *HANDLE[1:]))

    assert near["read_retreat_completed"]() is False
    assert far["read_retreat_completed"]() is True


def test_an_absent_sensor_refuses_rather_than_reporting_no_collision():
    """False here would assert the robot was fine, not that we did not look."""

    observations = _sources(read_scene_contact_forces=lambda: None)

    with pytest.raises(ArticulatedSceneObservationError) as excinfo:
        observations["read_scene_collision_failure"]()

    assert any("scene_contact_unavailable" in e for e in excinfo.value.errors)


def test_force_reduction_takes_the_largest_magnitude():
    forces = [[[3.0, 4.0, 0.0], [0.0, 0.0, 1.0]]]

    assert max_contact_force_n(forces) == pytest.approx(5.0)


def test_a_body_index_outside_the_sensor_refuses():
    with pytest.raises(ArticulatedSceneObservationError) as excinfo:
        max_contact_force_n([[[0.0, 0.0, 0.0]]], body_indices=[7])

    assert any("body_index_out_of_range" in e for e in excinfo.value.errors)


def test_an_authored_base_position_must_be_three_dimensional():
    with pytest.raises(ArticulatedSceneObservationError):
        _sources(authored_task_object_base_position_m=(1.0, 2.0))


def test_indices_must_be_within_the_array_they_index():
    """rt29: body_index_out_of_range 9 and 14 on contact-sensor arrays.

    The indices came from the articulation's body list; the arrays came from a
    ContactSensor, which matches its own subset of bodies by prim-path regex
    and publishes its own ordering in ContactSensor.body_names. Two different
    index spaces that happen to be integers.

    The refusal is right - it is the source of the indices that was wrong - so
    this pins the refusal rather than relaxing it.
    """

    two_body_array = [[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]

    with pytest.raises(ArticulatedSceneObservationError) as excinfo:
        max_contact_force_n(two_body_array, body_indices=[14])

    assert any("body_index_out_of_range:14" in e for e in excinfo.value.errors)


def test_resolving_rows_from_a_sensors_own_body_names():
    """The helper that makes the two index spaces impossible to confuse."""

    from blueprint_pipeline.articulated_scene_observations import (
        resolve_contact_sensor_rows,
    )

    sensor_bodies = ["panda_link5", "left_inner_finger", "right_inner_finger"]

    rows = resolve_contact_sensor_rows(
        sensor_body_names=sensor_bodies,
        finger_body_names=("left_inner_finger", "right_inner_finger"),
    )

    assert rows["finger_rows"] == [1, 2]
    assert rows["non_finger_rows"] == [0]


def test_a_sensor_missing_its_finger_bodies_refuses():
    """No finger rows means task contact cannot be measured at all."""

    from blueprint_pipeline.articulated_scene_observations import (
        resolve_contact_sensor_rows,
    )

    with pytest.raises(ArticulatedSceneObservationError) as excinfo:
        resolve_contact_sensor_rows(
            sensor_body_names=["panda_link5", "panda_link6"],
            finger_body_names=("left_inner_finger", "right_inner_finger"),
        )

    assert any("finger_rows_absent" in e for e in excinfo.value.errors)


def test_an_empty_sensor_body_list_refuses():
    from blueprint_pipeline.articulated_scene_observations import (
        resolve_contact_sensor_rows,
    )

    with pytest.raises(ArticulatedSceneObservationError):
        resolve_contact_sensor_rows(sensor_body_names=[], finger_body_names=("a",))
