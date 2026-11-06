from __future__ import annotations

import copy
import math
import random

import pytest

from blueprint_pipeline.native_g1_navigation_goal import (
    PUBLISHED_TASK_INSTRUCTION,
    score_g1_navigation_episode,
)
from blueprint_pipeline.task_predicates import (
    TaskPredicateError,
    episode_behaviour,
    g1_navigation_goal_predicates,
    score_episode,
    seal_predicates,
    validate_predicates,
)


def _g1_task_spec(settle: int = 3, radius: float = 0.3, drift: float = 0.2) -> dict:
    return {
        "task_kind": "rigid_pick_place",
        "g1_navigation_goal": {
            "schema_version": "native_g1_navigation_goal.v1",
            "center_world_m": [2.0, 0.0, 0.0],
            "acceptance_radius_m": radius,
            "max_root_height_drift_m": drift,
            "settle_window_samples": settle,
            "task_instruction": PUBLISHED_TASK_INSTRUCTION,
            "visible_target_marker": {
                "schema_version": "native_task_target_marker.v1",
                "shape": "flat_yellow_disc",
                "non_colliding": True,
                "surface_position_world_m": [2.0, 0.0, 0.0],
                "radius_m": 0.4,
            },
        },
    }


def _root_trace(points: list[tuple[float, float, float]], **extra) -> list[dict]:
    return [{"step_index": index, "root_position_world_m": list(point), **extra} for index, point in enumerate(points)]


def test_g1_navigation_predicates_score_exactly_like_the_published_scorer() -> None:
    rng = random.Random(7)
    task = _g1_task_spec()
    predicates = g1_navigation_goal_predicates(task)
    compared = 0
    for _ in range(400):
        length = rng.randint(2, 12)
        points = [(rng.uniform(0.0, 2.6), rng.uniform(-0.5, 0.5), 0.85 + rng.uniform(-0.3, 0.3))
                  for _ in range(length)]
        trace = _root_trace(points)
        try:
            expected = score_g1_navigation_episode(task_spec=task, samples=trace)
        except ValueError as exc:
            assert str(exc) == "g1_navigation_start_already_at_goal"
            report = score_episode(predicates=predicates, samples=trace)
            assert report["status"] == "unscored" and report["outcome"] is None
            continue
        report = score_episode(predicates=predicates, samples=trace)
        assert report["outcome"] == expected["outcome"], (points, expected, report)
        goal = next(row for row in report["predicate_results"] if row["id"] == "goal")
        assert goal["minimum_distance_m"] == pytest.approx(max(0.0, expected["minimum_distance_m"] - 0.3))
        compared += 1
    assert compared > 200


def test_missing_readback_is_never_success() -> None:
    predicates = seal_predicates([
        {"id": "goal", "kind": "region_reached", "subject": {"kind": "robot_root"},
         "region": {"shape": "disc", "center_world_m": [1.0, 0.0, 0.0], "radius_m": 0.3},
         "hold_samples": 1, "when": "terminal"},
        {"id": "clear", "kind": "no_contact", "geometry": "scene_obstacles", "contact_force_threshold_n": 1.0},
    ])
    trace = _root_trace([(0.0, 0.0, 0.8), (1.0, 0.0, 0.8)])
    report = score_episode(predicates=predicates, samples=trace)
    assert report["status"] == "unscored"
    assert report["outcome"] is None
    assert report["unscored_predicates"] == ["clear"]


def test_obstacle_contact_fails_a_trace_that_reaches_its_goal() -> None:
    predicates = seal_predicates([
        {"id": "goal", "kind": "region_reached", "subject": {"kind": "robot_root"},
         "region": {"shape": "disc", "center_world_m": [1.0, 0.0, 0.0], "radius_m": 0.3},
         "hold_samples": 2, "when": "terminal"},
        {"id": "clear", "kind": "no_contact", "geometry": "scene_obstacles", "contact_force_threshold_n": 1.0,
         "minimum_clearance_m": 0.05},
    ])
    trace = _root_trace([(0.0, 0.0, 0.8), (0.5, 0.0, 0.8), (1.0, 0.0, 0.8), (1.0, 0.1, 0.8)])
    for sample, force, clearance in zip(trace, (0.0, 3.0, 0.0, 0.0), (0.3, 0.0, 0.2, 0.2), strict=True):
        sample["robot_scene_contact_peak_force_n"] = force
        sample["robot_obstacle_clearance_m"] = clearance
    report = score_episode(predicates=predicates, samples=trace)
    assert report["outcome"] == "failure"
    assert report["failed_predicates"] == ["clear"]
    clear = next(row for row in report["predicate_results"] if row["id"] == "clear")
    assert clear["contact_steps"] == [1] and clear["minimum_clearance_observed_m"] == 0.0
    navigation = report["behaviour"]["navigation"]
    assert navigation["path_length_m"] == pytest.approx(1.1)
    assert navigation["minimum_obstacle_clearance_m"] == 0.0
    assert navigation["obstacle_contact_steps"] == [1]
    assert navigation["goals"][0]["closest_distance_m"] == 0.0

    for sample in trace:
        sample["robot_scene_contact_peak_force_n"] = 0.0
        sample["robot_obstacle_clearance_m"] = 0.2
    assert score_episode(predicates=predicates, samples=trace)["outcome"] == "success"


def test_named_geometry_contact_and_joint_interval() -> None:
    predicates = seal_predicates([
        {"id": "drawer_open", "kind": "joint_in_interval", "joint": "drawer", "interval": [0.15, 0.4],
         "hold_samples": 2, "when": "terminal"},
        {"id": "shelf_clear", "kind": "no_contact", "geometry": ["shelf_left", "shelf_right"],
         "contact_force_threshold_n": 2.0},
    ])
    trace = [{"step_index": index, "joint_positions": {"drawer": value}, "robot_contact_geometry_ids": hits,
              "robot_contact_peak_force_n_by_geometry": {"shelf_left": 0.5, "floor": 9.0}}
             for index, (value, hits) in enumerate([(0.0, []), (0.2, ["floor"]), (0.3, ["shelf_left"])])]
    report = score_episode(predicates=predicates, samples=trace)
    assert report["outcome"] == "success"  # 0.5 N on the shelf is below its threshold; the floor is not named
    trace[2]["robot_contact_peak_force_n_by_geometry"]["shelf_left"] = 4.0
    assert score_episode(predicates=predicates, samples=trace)["failed_predicates"] == ["shelf_clear"]
    # A reported touch whose force is missing or not finite is unscored, never read as zero force.
    for missing in ({"floor": 9.0}, {"shelf_left": float("nan")}):
        trace[2]["robot_contact_peak_force_n_by_geometry"] = missing
        report = score_episode(predicates=predicates, samples=trace)
        assert report["outcome"] is None and "shelf_clear" not in report["failed_predicates"]


def test_gripper_behaviour_counts_real_closes_and_distance_to_the_subject() -> None:
    pose = [0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 1.0]
    widths = [0.085, 0.03, 0.03, 0.085, 0.02]
    grasp = [[0.5, 0.0, 1.2], [0.0, 0.0, 0.92], [0.0, 0.0, 0.92], [0.3, 0.0, 1.0], [0.4, 0.0, 0.8]]
    trace = [{"step_index": index, "gripper_width_m": width, "grasp_frame_position_world_m": point,
              "task_object_pose_world": pose} for index, (width, point) in enumerate(zip(widths, grasp, strict=True))]
    behaviour = episode_behaviour(samples=trace, context={"subject_half_extents_m": [0.15, 0.25, 0.02]})
    gripper = behaviour["gripper"]
    assert [event["step"] for event in gripper["close_events"]] == [1, 4]
    assert gripper["close_events"][0]["distance_to_subject_bounds_m"] == pytest.approx(0.10)
    assert gripper["close_events"][1]["distance_to_subject_bounds_m"] == pytest.approx(0.25)
    assert gripper["minimum_width_m"] == 0.02
    assert gripper["closest_approach_to_subject_bounds"] == {"distance_m": pytest.approx(0.10), "step": 1}
    assert "navigation" not in behaviour


def test_oriented_subject_bounds_follow_the_object_rotation() -> None:
    # A box 0.1 x 0.5 rotated 90 degrees about z: a point 0.2 m along x is inside its long side.
    quarter = [0.0, 0.0, 0.7071067811865476, 0.7071067811865476]
    trace = [{"step_index": 0, "gripper_width_m": 0.02, "grasp_frame_position_world_m": [0.2, 0.0, 0.0],
              "task_object_pose_world": [0.0, 0.0, 0.0, *quarter]},
             {"step_index": 1, "gripper_width_m": 0.02, "grasp_frame_position_world_m": [0.0, 0.2, 0.0],
              "task_object_pose_world": [0.0, 0.0, 0.0, *quarter]}]
    gripper = episode_behaviour(samples=trace, context={"subject_half_extents_m": [0.05, 0.25, 0.05]})["gripper"]
    assert gripper["close_events"][0]["distance_to_subject_bounds_m"] == pytest.approx(0.0, abs=1e-9)
    assert gripper["closest_approach_to_subject_bounds"]["step"] == 0


def test_predicate_sets_are_sealed_and_strict() -> None:
    spec = seal_predicates([{"id": "t", "kind": "time_limit", "max_steps": 3}])
    tampered = copy.deepcopy(spec)
    tampered["predicates"][0]["max_steps"] = 30
    with pytest.raises(TaskPredicateError, match="task_predicates_digest_invalid"):
        validate_predicates(tampered)
    with pytest.raises(TaskPredicateError, match="task_predicate_kind_unknown"):
        seal_predicates([{"id": "t", "kind": "reward_above", "threshold": 1}])
    with pytest.raises(TaskPredicateError, match="task_predicate_fields_invalid"):
        seal_predicates([{"id": "t", "kind": "time_limit", "max_steps": 3, "policy_says_done": True}])
    with pytest.raises(TaskPredicateError, match="task_predicate_sample_step_invalid"):
        score_episode(predicates=spec, samples=[{"step_index": 0}, {"step_index": 2}])
    assert score_episode(predicates=spec, samples=[{"step_index": i} for i in range(4)])["outcome"] == "failure"


def test_rigid_reports_carry_measured_gripper_behaviour() -> None:
    from blueprint_pipeline.adp_task_scoring import score_task_episode_from_spec
    from tests.test_adp_rigid_retreat_scoring import _fixture

    spec, samples = _fixture()
    for row in samples:
        row["gripper_width_m"] = 0.085
        row["task_scoring_pose_world"] = list(row["task_object_pose_world"])
    samples[1]["gripper_width_m"] = 0.02
    report = score_task_episode_from_spec(task_spec=spec, samples=samples)
    gripper = report["behaviour"]["gripper"]
    assert [event["step"] for event in gripper["close_events"]] == [1]
    # Grasp frame at z 0.86 over a subject centred at 0.84 with a 0.005 m half-height.
    assert gripper["close_events"][0]["distance_to_subject_bounds_m"] == pytest.approx(math.hypot(0.145, 0.015))
    assert report["measurements"]["gripper_closed"] is False  # the settle-window reading, unchanged


def test_a_compiled_spec_dispatches_straight_to_the_predicate_scorer() -> None:
    # What the compiler writes is what the dispatcher scores: no kind, no renamed field.
    from blueprint_pipeline.adp_task_scoring import TaskNeutralScoringError, score_task_episode_from_spec
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest
    from blueprint_pipeline.task_spec_compiler import compile_task_spec
    from tests.test_task_spec_compiler import _aisle_brief, _aisle_scene

    spec = compile_task_spec(brief=_aisle_brief(), scene=_aisle_scene("validated"),
                             task_requirements={"locomote": True, "manipulate": False})
    assert "task_kind" not in spec
    xs = [0.0, 0.5, 1.0, 1.5, *([2.0] * 10)]
    trace = [{"step_index": index, "robot_root_position_world_m": [x, -0.05, 0.78],
              "robot_scene_contact_peak_force_n": 0.0} for index, x in enumerate(xs)]
    report = score_task_episode_from_spec(task_spec=spec, samples=trace)
    assert report["schema_version"] == "blueprint.task_predicate_score.v1"
    assert report["task_succeeded"] is True
    with pytest.raises(TaskNeutralScoringError, match="task_spec_digest_invalid"):
        score_task_episode_from_spec(task_spec={**spec, "sentence": "another task"}, samples=trace)
    tampered = {**spec, "success": {**spec["success"], "predicates_digest": "x"}}
    tampered["spec_digest"] = canonical_digest(tampered, digest_field="spec_digest")
    with pytest.raises(TaskNeutralScoringError, match="task_predicates_digest_invalid"):
        score_task_episode_from_spec(task_spec=tampered, samples=trace)
