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


def _sample(**overrides):
    arguments = {
        "joint_ids": ["lower_door_hinge", "upper_door_hinge"],
        "read_joint_state": _reader(),
        "step_index": 3,
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
