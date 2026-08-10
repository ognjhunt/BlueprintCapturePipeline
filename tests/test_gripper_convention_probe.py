"""Which command closes the fingers is measured, never assumed."""

from __future__ import annotations

import pytest

from blueprint_pipeline.gripper_convention_probe import (
    GripperConventionProbeError,
    measure_gripper_convention,
)


class _Fingers:
    """A gripper whose fingers separate by `gap(command)`."""

    def __init__(self, gap, *, bodies=("left_inner_finger", "right_inner_finger")):
        self._gap = gap
        self._command = 0.0
        self.body_names = list(bodies)

    def apply(self, command: float) -> None:
        self._command = float(command)

    def separation(self) -> float:
        return float(self._gap(self._command))


def _probe(fingers, **overrides):
    kwargs = {
        "candidate_commands": (0.0, 1.0),
        "apply_command": fingers.apply,
        "read_finger_separation_m": fingers.separation,
        "body_names": fingers.body_names,
    }
    kwargs.update(overrides)
    return measure_gripper_convention(**kwargs)


def test_measures_which_command_closes_the_fingers():
    # 0.0 wide open, 1.0 closed - the DROID convention.
    probe = _probe(_Fingers(lambda c: 0.08 - 0.07 * c))

    assert probe["closed_command"] == 1.0
    assert probe["open_command"] == 0.0
    assert probe["gripper_closed_width_m"] == pytest.approx(0.01)
    assert probe["gripper_open_width_m"] == pytest.approx(0.08)


def test_an_inverted_convention_is_reported_not_corrected():
    """Arena's action dimension may invert DROID's meaning.

    An inverted convention turns every commanded grasp into a release, which
    reads as a policy failing the task rather than the harness driving the
    gripper backwards.
    """

    probe = _probe(_Fingers(lambda c: 0.01 + 0.07 * c))

    assert probe["closed_command"] == 0.0
    assert probe["open_command"] == 1.0
    assert probe["convention_matches_droid"] is False


def test_indistinguishable_commands_stay_unmeasured():
    """Below the travel floor the two commands are noise, not a convention."""

    with pytest.raises(GripperConventionProbeError) as excinfo:
        _probe(_Fingers(lambda c: 0.05 + 1e-6 * c))

    assert any("travel_below_floor" in error for error in excinfo.value.errors)


def test_missing_finger_bodies_fail_closed():
    with pytest.raises(GripperConventionProbeError) as excinfo:
        _probe(_Fingers(lambda c: 0.08 - 0.07 * c, bodies=("only_one_finger",)))

    assert any("finger_bodies_not_resolved" in e for e in excinfo.value.errors)


def test_every_candidate_command_is_actually_applied():
    seen: list[float] = []
    fingers = _Fingers(lambda c: 0.08 - 0.07 * c)
    original = fingers.apply

    def _record(command):
        seen.append(float(command))
        original(command)

    _probe(fingers, apply_command=_record)

    assert sorted(set(seen)) == [0.0, 1.0]
