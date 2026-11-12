"""Predicates over several objects: resting on, inside, and at least N of them.

Poses are built by hand: two cardboard boxes (0.20 × 0.15 × 0.10 m, origin at the bottom
centre) and a crate (outer 0.60 × 0.40 × 0.30 m, interior 0.56 × 0.36 × 0.27 m above a
3 cm floor, origin at the bottom centre) standing on the floor.
"""
from __future__ import annotations

import math

import pytest

from blueprint_pipeline import task_predicates as tp

BOX = {"minimum": [-0.10, -0.075, 0.0], "maximum": [0.10, 0.075, 0.10]}
CRATE_OUTER = {"minimum": [-0.30, -0.20, 0.0], "maximum": [0.30, 0.20, 0.30]}
CRATE_INSIDE = {"minimum": [-0.28, -0.18, 0.03], "maximum": [0.28, 0.18, 0.30]}
CRATE = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
IDENTITY = [0.0, 0.0, 0.0, 1.0]


def _yaw(angle: float) -> list[float]:
    return [0.0, 0.0, math.sin(angle / 2), math.cos(angle / 2)]


def _resting(**changes):
    return {"id": "a_on_crate", "kind": "resting_on", "subject": {"kind": "object", "object_id": "box_a"},
            "subject_extent": BOX, "support": {"kind": "object", "object_id": "crate"},
            "support_extent": CRATE_OUTER, "tolerance_m": 0.005, "settle_motion_m": 0.005,
            "hold_samples": 3, "when": "terminal", **changes}


def _inside(object_id="box_a", **changes):
    return {"id": f"{object_id}_in_crate", "kind": "inside", "subject": {"kind": "object", "object_id": object_id},
            "subject_extent": BOX, "container": {"kind": "object", "object_id": "crate"},
            "container_interior": CRATE_INSIDE, "tolerance_m": 0.005, "settle_motion_m": 0.005,
            "hold_samples": 3, "when": "terminal", **changes}


def _trace(paths: dict[str, list[list[float]]], crate=CRATE) -> list[dict]:
    steps = len(next(iter(paths.values())))
    return [{"step_index": index,
             "object_poses_world": {"crate": list(crate), **{name: list(path[index]) for name, path in paths.items()}},
             # Released: the robot touches none of them (the scorer's per-object contact channel).
             "object_robot_contact_active": {name: False for name in ("crate", "box_a", "box_b", "box_c")}}
            for index in range(steps)]


def _score(rows, trace):
    return tp.score_episode(predicates=tp.seal_predicates(rows), samples=trace)


def _still(position, orientation=IDENTITY, steps=5):
    return [[*position, *orientation] for _ in range(steps)]


def test_a_box_resting_on_the_crate_lid_is_satisfied() -> None:
    report = _score([_resting()], _trace({"box_a": _still([1.0, 0.0, 0.30])}))
    assert report["outcome"] == "success"
    assert report["predicate_results"][0]["terminal_gap_m"] == pytest.approx(0.0)


def test_touching_the_crate_side_on_the_floor_is_not_resting_on_it() -> None:
    # Box on the floor, its face against the crate's side: in contact, not on top.
    report = _score([_resting()], _trace({"box_a": _still([1.0 - 0.30 - 0.10, 0.0, 0.0])}))
    assert report["outcome"] == "failure"
    result = report["predicate_results"][0]
    assert result["terminal_gap_m"] == pytest.approx(-0.30) and result["terminal_over_support"] is False


def test_hovering_above_or_sunk_into_the_lid_beyond_the_tolerance_is_not_resting() -> None:
    assert _score([_resting()], _trace({"box_a": _still([1.0, 0.0, 0.31])}))["outcome"] == "failure"
    assert _score([_resting()], _trace({"box_a": _still([1.0, 0.0, 0.304])}))["outcome"] == "success"


def test_on_top_but_sliding_off_at_the_end_of_the_settle_window_fails() -> None:
    sliding = [[1.0 + 0.0, 0.0, 0.30, *IDENTITY], [1.0, 0.0, 0.30, *IDENTITY],
               [1.004, 0.0, 0.30, *IDENTITY], [1.012, 0.0, 0.30, *IDENTITY], [1.025, 0.0, 0.30, *IDENTITY]]
    report = _score([_resting()], _trace({"box_a": sliding}))
    # Every sample is still on the lid, but it moved 2.1 cm over the window.
    assert report["outcome"] == "failure" and report["predicate_results"][0]["terminal_over_support"] is True
    # "any" finds the settled window before the slide began.
    assert _score([_resting(when="any", hold_samples=2)], _trace({"box_a": sliding}))["outcome"] == "success"


def test_past_the_lid_edge_is_not_over_the_support() -> None:
    report = _score([_resting()], _trace({"box_a": _still([1.0 + 0.31, 0.0, 0.30])}))
    assert report["outcome"] == "failure" and report["predicate_results"][0]["terminal_over_support"] is False


def test_a_box_wholly_inside_the_crate_is_inside_and_half_in_is_not() -> None:
    assert _score([_inside()], _trace({"box_a": _still([1.0, 0.0, 0.03])}))["outcome"] == "success"
    half = _score([_inside()], _trace({"box_a": _still([1.0, 0.0, 0.25])}))  # top half above the rim
    assert half["outcome"] == "failure" and half["predicate_results"][0]["terminal_corners_outside"] == 4
    over_wall = _score([_inside()], _trace({"box_a": _still([1.0 + 0.24, 0.0, 0.03])}))  # through a wall
    assert over_wall["outcome"] == "failure"


def test_inside_reads_the_container_in_its_own_frame() -> None:
    turned = [1.0, 0.0, 0.0, *_yaw(math.pi / 2)]  # the crate is turned a quarter: its long side runs along y
    along_y = _still([1.0, 0.15, 0.03], orientation=_yaw(math.pi / 2))
    assert _score([_inside()], _trace({"box_a": along_y}, crate=turned))["outcome"] == "success"
    assert _score([_inside()], _trace({"box_a": _still([1.20, 0.0, 0.03])}, crate=turned))["outcome"] == "failure"


def test_at_least_counts_members_and_never_guesses_missing_readback() -> None:
    rows = [{"id": "two_of_three", "kind": "at_least", "minimum": 2,
             "predicates": [_inside("box_a"), _inside("box_b"), _inside("box_c")]}]
    inside, outside = _still([0.85, 0.0, 0.03]), _still([0.0, 0.0, 0.80])
    report = _score(rows, _trace({"box_a": inside, "box_b": _still([1.15, 0.0, 0.03]), "box_c": outside}))
    assert report["outcome"] == "success" and report["predicate_results"][0]["satisfied_count"] == 2
    assert _score(rows, _trace({"box_a": inside, "box_b": outside, "box_c": outside}))["outcome"] == "failure"
    unread = _trace({"box_a": inside, "box_b": outside})  # box_c has no readback
    assert _score(rows, unread)["status"] == "unscored"
    assert _score([{**rows[0], "minimum": 1}], unread)["outcome"] == "success"


def test_every_object_in_its_place_is_all_of_and_a_missing_body_is_unscored() -> None:
    rows = [_inside("box_a"), _inside("box_b")]
    both = _trace({"box_a": _still([0.85, 0.0, 0.03]), "box_b": _still([1.15, 0.0, 0.03])})
    assert _score(rows, both)["outcome"] == "success"
    one = _trace({"box_a": _still([0.85, 0.0, 0.03])})
    assert _score(rows, one)["status"] == "unscored"


@pytest.mark.parametrize("row, code", [
    (_resting(support={"kind": "object", "object_id": "box_a"}), "task_predicate_subject_invalid"),
    (_resting(support={"kind": "robot_root"}), "task_predicate_subject_invalid"),
    (_inside(tolerance_m=0.2), "task_predicate_tolerance_invalid"),
    (_inside(subject_extent={"minimum": [0, 0, 0], "maximum": [0, 1, 1]}), "task_predicate_extent_invalid"),
    ({"id": "n", "kind": "at_least", "minimum": 3, "predicates": [_inside("box_a"), _inside("box_b")]},
     "task_predicate_at_least_invalid"),
    ({"id": "n", "kind": "at_least", "minimum": 1, "predicates": [
        {"id": "m", "kind": "at_least", "minimum": 1, "predicates": [_inside("box_a")]}]},
     "task_predicate_at_least_member_invalid"),
    ({"id": "n", "kind": "at_least", "minimum": 1, "predicates": [_inside("box_a"), _inside("box_a")]},
     "task_predicate_ids_invalid"),
])
def test_malformed_several_object_predicates_are_refused(row, code) -> None:
    with pytest.raises(tp.TaskPredicateError, match=code):
        tp.seal_predicates([row])


# The independent review's probe (review28/p_pred.py): a 10 cm cube on a 1 m slab.
CUBE = {"minimum": [-0.05, -0.05, -0.05], "maximum": [0.05, 0.05, 0.05]}
SLAB = {"minimum": [-0.5, -0.5, -0.02], "maximum": [0.5, 0.5, 0.0]}


def _probe_trace(zs, *, tilt=None, held=False):
    q = [0, 0, 0, 1] if tilt is None else tilt
    return [{"step_index": index, "object_poses_world": {"a": [0, 0, z, 0, 0, 0, 1], "t": [0, 0, 0, *q]},
             "object_robot_contact_active": {"a": held, "t": False}} for index, z in enumerate(zs)]


def _probe_row(**changes):
    return {"id": "r", "kind": "resting_on", "subject": {"kind": "object", "object_id": "a"}, "subject_extent": CUBE,
            "support": {"kind": "object", "object_id": "t"}, "support_extent": SLAB, "tolerance_m": 0.005,
            "settle_motion_m": 0.01, "hold_samples": 3, "when": "terminal", **changes}


def test_a_5_cm_resting_tolerance_is_refused() -> None:
    with pytest.raises(tp.TaskPredicateError, match="task_predicate_tolerance_invalid"):
        tp.seal_predicates([_probe_row(tolerance_m=0.05)])


def test_held_still_above_the_support_is_not_resting_and_released_on_it_is() -> None:
    hover = _probe_trace([0.3, 0.2, 0.095, 0.095, 0.095])
    assert _score([_probe_row()], hover)["outcome"] == "failure"                       # 4.5 cm up
    held = _probe_trace([0.3, 0.2, 0.05, 0.05, 0.05], held=True)
    assert _score([_probe_row()], held)["outcome"] == "failure"                        # on it, still gripped
    assert _score([_probe_row()], _probe_trace([0.3, 0.2, 0.05, 0.05, 0.05]))["outcome"] == "success"


def test_a_box_resting_on_a_tilted_support_is_measured_along_its_normal() -> None:
    angle = math.radians(10)
    tilt = [math.sin(angle / 2), 0, 0, math.cos(angle / 2)]
    # The cube resting on the tilted slab: its bottom-face centre lies on the slab's top face.
    resting = _probe_trace([0.05] * 4, tilt=tilt)
    for sample in resting:
        sample["object_poses_world"]["a"] = [0, -0.05 * math.sin(angle), 0.05 * math.cos(angle), *tilt]
    result = _score([_probe_row()], resting)
    assert result["outcome"] == "success" and result["predicate_results"][0]["terminal_gap_m"] == pytest.approx(0.0)


def test_held_in_mid_air_inside_a_container_is_not_inside() -> None:
    inside = {"id": "i", "kind": "inside", "subject": {"kind": "object", "object_id": "a"}, "subject_extent": CUBE,
              "container": {"kind": "object", "object_id": "t"},
              "container_interior": {"minimum": [-0.2, -0.2, 0.0], "maximum": [0.2, 0.2, 0.3]},
              "tolerance_m": 0.01, "settle_motion_m": 0.01, "hold_samples": 2, "when": "terminal"}
    mid_air = _probe_trace([0.2, 0.2, 0.2])
    assert _score([inside], mid_air)["outcome"] == "failure"
    assert _score([inside], _probe_trace([0.2, 0.2, 0.2], held=True))["outcome"] == "failure"
    assert _score([inside], _probe_trace([0.05, 0.05, 0.05]))["outcome"] == "success"
    no_channel = [{k: v for k, v in sample.items() if k != "object_robot_contact_active"}
                  for sample in _probe_trace([0.05, 0.05, 0.05])]
    assert _score([inside], no_channel)["status"] == "unscored"


def test_a_nan_pose_is_unscored() -> None:
    assert _score([_probe_row()], _probe_trace([0.05, 0.05, float("nan")]))["status"] == "unscored"
