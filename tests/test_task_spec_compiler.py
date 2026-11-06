from __future__ import annotations

import copy
import math

import pytest

from blueprint_pipeline.task_predicates import score_episode
from blueprint_pipeline.task_spec_compiler import (
    TaskSpecCompileError,
    compile_task_spec,
    goal_clear_of_start,
    object_label,
    placement_instruction,
    requirements,
    validate_task_spec,
)

AISLE = "Navigate from the near end of the visible aisle to a stopping point before the end bookshelf, " \
        "avoiding contact with the shelves on either side."


def _aisle_scene(scale: str = "unverified") -> dict:
    return {
        "scene_revision_digest": "sha256:" + "a" * 64,
        "floor_height_world_m": 0.0,
        "metric_scale": {"status": scale, "evidence": [{"source": "atlas_scale_review", "value": "estimated"}]},
        # A forward walk down an aisle, as the camera filmed it (eye height above the floor).
        "filmed_path_world_m": [[0.0, 0.0, 1.5], [0.7, 0.05, 1.5], [1.4, 0.0, 1.5], [2.0, -0.05, 1.5]],
    }


def _aisle_brief() -> dict:
    return {"site_id": "site-library", "task_id": "task-aisle", "sentence": AISLE,
            "goal_label": "before the end bookshelf", "obstacle_label": "shelves on either side"}


def test_navigation_brief_compiles_to_locomotion_predicates_with_flagged_guesses() -> None:
    spec = compile_task_spec(brief=_aisle_brief(), scene=_aisle_scene(),
                             task_requirements={"locomote": True, "manipulate": False})
    assert spec["requirements"] == requirements(locomote=True, manipulate=False)
    kinds = [row["kind"] for row in spec["success"]["predicates"]]
    assert kinds == ["region_reached", "height_stable", "no_contact"]
    goal = spec["success"]["predicates"][0]
    # The goal comes from the filmed path's end, on the floor, widened for unvalidated scale.
    assert goal["region"]["center_world_m"] == [2.0, -0.05, 0.0]
    assert goal["region"]["radius_m"] == pytest.approx(0.25 + 0.1 * math.dist([0, 0], [2.0, -0.05]))
    assert spec["start"]["region"]["center_world_m"] == [0.0, 0.0, 0.0]
    assert spec["flags"] == ["goal_from_filmed_path", "goal_tolerance_widened_for_unvalidated_scale",
                             "metric_scale_unverified", "start_from_filmed_path"]
    marker = spec["markers"][0]
    assert marker["radius_m"] == goal["region"]["radius_m"] and marker["colour"] == "yellow"
    assert marker["sized_from"] == "goal_tolerance" and marker["policy_visible"] is True
    assert spec["instruction"]["canonical"] == (
        "Move to the yellow marked area before the end bookshelf without touching the shelves on either side.")
    assert spec["metric_scale"]["evidence"][0]["source"] == "atlas_scale_review"


def test_a_compiled_navigation_spec_scores_a_walk_and_a_collision() -> None:
    spec = compile_task_spec(brief=_aisle_brief(), scene=_aisle_scene("validated"),
                             task_requirements={"locomote": True, "manipulate": False})
    assert spec["success"]["predicates"][0]["region"]["radius_m"] == 0.25
    xs = [0.0, 0.5, 1.0, 1.5, *([2.0] * 10)]
    trace = [{"step_index": index, "robot_root_position_world_m": [x, -0.05, 0.78],
              "robot_scene_contact_peak_force_n": 0.0} for index, x in enumerate(xs)]
    assert score_episode(predicates=spec["success"], samples=trace)["outcome"] == "success"
    trace[4]["robot_scene_contact_peak_force_n"] = 25.0
    report = score_episode(predicates=spec["success"], samples=trace)
    assert report["outcome"] == "failure" and report["failed_predicates"] == ["no_obstacle_contact"]


def test_reviewed_anchors_win_over_the_filmed_path() -> None:
    scene = _aisle_scene("validated")
    scene["anchors"] = {"start_world_m": [0.2, 0.0, 0.0], "goal_world_m": [2.6, 0.0, 0.0]}
    spec = compile_task_spec(brief=_aisle_brief(), scene=scene,
                             task_requirements={"locomote": True, "manipulate": False})
    assert spec["goal_source"] == "reviewed_anchor" and spec["flags"] == []
    assert spec["start"]["source"] == "reviewed_anchor"


def test_placement_instruction_drops_location_phrases_and_names_the_destination() -> None:
    # capture-447's labels gave "Pick up and place the wooden tray on the dining table on the wooden dining table."
    assert placement_instruction(subject_label="wooden tray on the dining table",
                                 destination_label="wooden dining table", relation="on", colour="green") == (
        "Pick up the wooden tray and place it on the green marked area on the wooden dining table.")
    assert placement_instruction(subject_label="the red mug", destination_label="dish rack",
                                 relation="inside", colour="green") == "Pick up the red mug and put it inside the dish rack."
    assert object_label("stack of plates near the sink") == "stack of plates"


def test_placement_goal_never_covers_its_subject_and_its_marker_is_tolerance_sized() -> None:
    scene = {"scene_revision_digest": "sha256:" + "b" * 64, "floor_height_world_m": 0.0,
             "metric_scale": {"status": "provider_declared_not_independently_validated"},
             "objects": [{"object_id": "tray", "label": "wooden tray on the dining table",
                          "pose_world": [0.0, 0.0, 0.80, 0.0, 0.0, 0.0, 1.0], "half_extents_m": [0.155, 0.26, 0.02]}],
             # A goal 10 cm from the tray's centre: the old marker sizing put 31-38% of the tray over it.
             "anchors": {"goal_world_m": [0.1, 0.0, 0.80]}}
    brief = {"site_id": "s", "task_id": "t", "sentence": "Move the tray to the other end of the table.",
             "subject_object_id": "tray", "destination_label": "wooden dining table", "goal_tolerance_m": 0.08}
    spec = compile_task_spec(brief=brief, scene=scene, task_requirements={"locomote": False, "manipulate": True})
    goal = spec["success"]["predicates"][0]["region"]
    clearance = math.dist(goal["center_world_m"][:2], [0.0, 0.0])
    assert clearance >= 0.08 + math.hypot(0.155, 0.26)
    assert "goal_moved_off_start" in spec["flags"]
    assert spec["markers"][0]["radius_m"] == 0.08


def test_tampered_or_object_sized_markers_are_refused() -> None:
    spec = compile_task_spec(brief=_aisle_brief(), scene=_aisle_scene(),
                             task_requirements={"locomote": True, "manipulate": False})
    bad = copy.deepcopy(spec)
    bad["markers"][0]["radius_m"] = 0.39
    with pytest.raises(TaskSpecCompileError, match="task_spec_marker_not_goal_sized"):
        validate_task_spec(bad)
    bad = copy.deepcopy(spec)
    bad["sentence"] = "Walk somewhere else."
    with pytest.raises(TaskSpecCompileError, match="task_spec_digest_invalid"):
        validate_task_spec(bad)
    with pytest.raises(TaskSpecCompileError, match="task_spec_requirements_invalid"):
        compile_task_spec(brief=_aisle_brief(), scene=_aisle_scene(),
                          task_requirements={"locomote": False, "manipulate": False})
    with pytest.raises(TaskSpecCompileError, match="task_spec_start_unresolved"):
        compile_task_spec(brief=_aisle_brief(), scene={**_aisle_scene(), "filmed_path_world_m": []},
                          task_requirements={"locomote": True, "manipulate": False})


def test_a_long_unvalidated_walk_caps_its_goal_tolerance() -> None:
    scene = _aisle_scene()
    scene["filmed_path_world_m"] = [[0.0, 0.0, 1.5], [6.0, 0.0, 1.5]]
    spec = compile_task_spec(brief=_aisle_brief(), scene=scene,
                             task_requirements={"locomote": True, "manipulate": False})
    assert spec["success"]["predicates"][0]["region"]["radius_m"] == 0.5
    assert "goal_tolerance_capped" in spec["flags"]


def test_goal_clear_of_start_moves_a_coincident_goal_along_the_hint() -> None:
    goal, moved = goal_clear_of_start(goal=[0.0, 0.0, 0.0], radius=0.3, start=[0.0, 0.0, 0.0],
                                      start_footprint_m=0.25, direction_hint=[0.0, 1.0])
    assert moved and goal[0] == pytest.approx(0.0) and goal[1] > 0.55
