from __future__ import annotations


import pytest

from blueprint_pipeline.articulated_task_sample import (
    ARTICULATED_TASK_SAMPLE_SCHEMA_VERSION,
    ArticulatedTaskSampleError,
    build_articulated_task_sample,
)


def _reader(state=None):
    state = state or {
        "upper_door_hinge": (0.7854, 0.012),
        "lower_door_hinge": (0.0, 0.0),
    }

    def read(joint_id: str):
        if joint_id not in state:
            raise KeyError(joint_id)
        return state[joint_id]

    return read


# Every joint the default reader offers, so limit derivation has bounds for
# each; individual tests override where the limits are the subject.
_DEFAULT_LIMITS = {
    "lower_door_hinge": [0.0, 1.5707963267948966],
    "upper_door_hinge": [0.0, 1.5707963267948966],
}


def _sample(**overrides):
    arguments = {
        "joint_ids": ["lower_door_hinge", "upper_door_hinge"],
        "read_joint_state": _reader(),
        "step_index": 3,
        "joint_hard_limits_rad": _DEFAULT_LIMITS,
        "read_task_contact_active": lambda: False,
        "read_containment_violation": lambda: False,
        "read_robot_collision_failure": lambda: False,
        "read_scene_collision_failure": lambda: False,
        "read_retreat_completed": lambda: False,
    }
    arguments.update(overrides)
    return build_articulated_task_sample(**arguments)


def test_the_sample_carries_exactly_the_joints_the_scorer_demands() -> None:
    """The scorer rejects any sample whose joint set differs from the spec's.

    Discovering that mid-episode costs the run, so the set is built from the
    binding rather than from whatever the runtime happens to expose.
    """

    sample = _sample()

    assert set(sample["joint_positions_rad"]) == {
        "lower_door_hinge",
        "upper_door_hinge",
    }
    assert set(sample["joint_velocities_rad_s"]) == set(sample["joint_positions_rad"])
    assert sample["step_index"] == 3
    assert sample["schema_version"] == ARTICULATED_TASK_SAMPLE_SCHEMA_VERSION


def test_positions_and_velocities_come_back_in_radians() -> None:
    sample = _sample()

    assert sample["joint_positions_rad"]["upper_door_hinge"] == pytest.approx(0.7854)
    assert sample["joint_velocities_rad_s"]["upper_door_hinge"] == pytest.approx(0.012)


def test_a_joint_the_runtime_cannot_read_fails_closed_by_name() -> None:
    """A silently dropped joint would score a door that was never observed."""

    with pytest.raises(ArticulatedTaskSampleError) as excinfo:
        _sample(joint_ids=["upper_door_hinge", "ghost_hinge"])

    assert any("joint_unreadable:ghost_hinge" in e for e in excinfo.value.errors)


def test_a_non_finite_reading_fails_closed_rather_than_scoring() -> None:
    """NaN out of a diverged solver must not be recorded as a door angle."""

    with pytest.raises(ArticulatedTaskSampleError) as excinfo:
        _sample(
            read_joint_state=_reader(
                {"upper_door_hinge": (float("nan"), 0.0),
                 "lower_door_hinge": (0.0, 0.0)}
            )
        )

    assert any("joint_state_not_finite" in e for e in excinfo.value.errors)


def test_an_empty_binding_fails_closed() -> None:
    with pytest.raises(ArticulatedTaskSampleError) as excinfo:
        _sample(joint_ids=[])

    assert any("joint_ids_missing" in e for e in excinfo.value.errors)


def test_the_sample_is_json_safe_for_the_receipt() -> None:
    """Numpy scalars out of a physics view break canonical digesting."""

    class _Odd(float):
        pass

    sample = _sample(
        read_joint_state=_reader(
            {"upper_door_hinge": (_Odd(0.5), _Odd(0.1)),
             "lower_door_hinge": (_Odd(0.0), _Odd(0.0))}
        )
    )

    for value in sample["joint_positions_rad"].values():
        assert type(value) is float


def test_sampling_is_deterministic() -> None:
    assert _sample() == _sample()


def test_degrees_leaking_in_are_caught_by_the_plausibility_bound() -> None:
    """A hinge at 45 radians is 2578 degrees: a unit slip, not a door.

    This lane has already paid once for a degree/radian confusion in USD drive
    damping, so a reading far outside any plausible joint range is refused
    rather than scored.
    """

    with pytest.raises(ArticulatedTaskSampleError) as excinfo:
        _sample(
            read_joint_state=_reader(
                {"upper_door_hinge": (45.0, 0.0), "lower_door_hinge": (0.0, 0.0)}
            )
        )

    assert any("implausible" in e for e in excinfo.value.errors)


def _joint_state(positions, velocities=None):
    velocities = velocities or {name: 0.0 for name in positions}

    def _read(joint_id):
        return (positions[joint_id], velocities[joint_id])

    return _read


_LIMITS = {"upper_door_hinge": [0.0, 1.5707963267948966]}


def _observations(**overrides):
    base = {
        "read_task_contact_active": lambda: False,
        "read_robot_collision_failure": lambda: False,
        "read_scene_collision_failure": lambda: False,
        "read_containment_violation": lambda: False,
        "read_retreat_completed": lambda: False,
    }
    base.update(overrides)
    return base


def test_sample_carries_every_boolean_the_scorer_requires():
    """The scorer rejects a sample missing any of six booleans.

    Nothing in the runtime produced them, so the articulated scoring path had
    never been fed a real sample - every control run would have been refused
    regardless of how well the scene composed.
    """

    sample = build_articulated_task_sample(
        joint_ids=["upper_door_hinge"],
        read_joint_state=_joint_state({"upper_door_hinge": 0.4}),
        step_index=3,
        joint_hard_limits_rad=_LIMITS,
        **_observations(),
    )

    for field in (
        "task_contact_active",
        "joint_limit_violation",
        "containment_violation",
        "robot_collision_failure",
        "scene_collision_failure",
        "retreat_completed",
    ):
        assert isinstance(sample[field], bool), field


def test_a_missing_observation_refuses_rather_than_defaulting_to_false():
    """False on a collision flag is an assertion that there was no collision.

    Defaulting is the dangerous option here: it converts "we did not look" into
    "we looked and the robot was fine", which is exactly the claim this program
    must never make by accident.
    """

    observations = _observations()
    del observations["read_robot_collision_failure"]

    with pytest.raises(ArticulatedTaskSampleError) as excinfo:
        build_articulated_task_sample(
            joint_ids=["upper_door_hinge"],
            read_joint_state=_joint_state({"upper_door_hinge": 0.4}),
            joint_hard_limits_rad=_LIMITS,
            **observations,
        )

    assert any(
        "observation_missing:robot_collision_failure" in error
        for error in excinfo.value.errors
    )


def test_a_non_boolean_observation_is_refused():
    with pytest.raises(ArticulatedTaskSampleError) as excinfo:
        build_articulated_task_sample(
            joint_ids=["upper_door_hinge"],
            read_joint_state=_joint_state({"upper_door_hinge": 0.4}),
            joint_hard_limits_rad=_LIMITS,
            **_observations(read_task_contact_active=lambda: 1.0),
        )

    assert any(
        "observation_not_boolean:task_contact_active" in error
        for error in excinfo.value.errors
    )


def test_joint_limit_violation_is_derived_not_asked_for():
    """Position against the authored limits is knowable without a sensor."""

    sample = build_articulated_task_sample(
        joint_ids=["upper_door_hinge"],
        read_joint_state=_joint_state({"upper_door_hinge": 1.60}),
        joint_hard_limits_rad=_LIMITS,
        **_observations(),
    )

    assert sample["joint_limit_violation"] is True


def test_a_joint_inside_its_limits_does_not_violate():
    sample = build_articulated_task_sample(
        joint_ids=["upper_door_hinge"],
        read_joint_state=_joint_state({"upper_door_hinge": 0.9}),
        joint_hard_limits_rad=_LIMITS,
        **_observations(),
    )

    assert sample["joint_limit_violation"] is False


def test_the_limit_tolerance_absorbs_solver_overshoot_only():
    """PhysX settles a hard stop slightly past it; that is not a violation."""

    inside = build_articulated_task_sample(
        joint_ids=["upper_door_hinge"],
        read_joint_state=_joint_state({"upper_door_hinge": 1.5717963267948966}),
        joint_hard_limits_rad=_LIMITS,
        joint_limit_tolerance_rad=0.01,
        **_observations(),
    )
    outside = build_articulated_task_sample(
        joint_ids=["upper_door_hinge"],
        read_joint_state=_joint_state({"upper_door_hinge": 1.70}),
        joint_hard_limits_rad=_LIMITS,
        joint_limit_tolerance_rad=0.01,
        **_observations(),
    )

    assert inside["joint_limit_violation"] is False
    assert outside["joint_limit_violation"] is True


def test_limits_missing_for_a_commanded_joint_fail_closed():
    """Without limits the violation cannot be derived, so it is not guessed."""

    with pytest.raises(ArticulatedTaskSampleError) as excinfo:
        build_articulated_task_sample(
            joint_ids=["upper_door_hinge", "lower_door_hinge"],
            read_joint_state=_joint_state(
                {"upper_door_hinge": 0.4, "lower_door_hinge": 0.0}
            ),
            joint_hard_limits_rad=_LIMITS,
            **_observations(),
        )

    assert any(
        "joint_hard_limits_missing:lower_door_hinge" in error
        for error in excinfo.value.errors
    )


def test_an_observation_that_raises_is_reported_not_swallowed():
    def _explode():
        raise RuntimeError("contact sensor not initialised")

    with pytest.raises(ArticulatedTaskSampleError) as excinfo:
        build_articulated_task_sample(
            joint_ids=["upper_door_hinge"],
            read_joint_state=_joint_state({"upper_door_hinge": 0.4}),
            joint_hard_limits_rad=_LIMITS,
            **_observations(read_scene_collision_failure=_explode),
        )

    assert any(
        "observation_failed:scene_collision_failure" in error
        for error in excinfo.value.errors
    )


def test_an_unreadable_joint_says_why_it_was_unreadable():
    """A label without a cause costs a launch to diagnose.

    rt23 returned joint_unreadable for both joints and nothing else - no
    exception type, no message - from inside a physics buffer read. That is the
    same defect already fixed at the worker's top level, repeated one layer
    down: an error that names the symptom and discards the evidence.
    """

    def _explode(joint_id):
        raise IndexError("index 2 is out of bounds for dimension 0 with size 2")

    with pytest.raises(ArticulatedTaskSampleError) as excinfo:
        build_articulated_task_sample(
            joint_ids=["upper_door_hinge"],
            read_joint_state=_explode,
            joint_hard_limits_rad=_LIMITS,
            **_observations(),
        )

    joined = ";".join(excinfo.value.errors)
    assert "joint_unreadable:upper_door_hinge" in joined
    assert "IndexError" in joined
    assert "out of bounds" in joined


def test_the_cause_is_truncated_not_unbounded():
    """A physics traceback in an error string is unreadable, not helpful."""

    def _explode(joint_id):
        raise RuntimeError("x" * 4000)

    with pytest.raises(ArticulatedTaskSampleError) as excinfo:
        build_articulated_task_sample(
            joint_ids=["upper_door_hinge"],
            read_joint_state=_explode,
            joint_hard_limits_rad=_LIMITS,
            **_observations(),
        )

    assert all(len(error) < 400 for error in excinfo.value.errors)
