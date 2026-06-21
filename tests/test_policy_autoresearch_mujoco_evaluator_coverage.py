from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path

import pytest

from blueprint_pipeline import policy_autoresearch_mujoco_evaluator as mujoco_eval


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_mujoco_evaluator_helper_edges() -> None:
    assert mujoco_eval._string(None) == ""
    assert mujoco_eval._mapping({"a": 1}) == {"a": 1}
    assert mujoco_eval._mapping([]) == {}
    assert mujoco_eval._float(True, 2.5) == 2.5
    assert mujoco_eval._float(3) == 3.0
    assert mujoco_eval._float("4.5") == 4.5
    assert mujoco_eval._float("bad", 1.25) == 1.25
    assert mujoco_eval._int(True, 9) == 9
    assert mujoco_eval._int(4) == 4
    assert mujoco_eval._int("5.9") == 5
    assert mujoco_eval._int("bad", 3) == 3
    assert mujoco_eval._safe_id(" Policy:One! ") == "policy_one"
    assert mujoco_eval._safe_id("!!!", fallback="fallback") == "fallback"
    assert mujoco_eval._pose_triplet([1, "2", 3.5]) == [1.0, 2.0, 3.5]
    assert mujoco_eval._pose_triplet("1,2,3") is None
    assert mujoco_eval._pose_triplet([1, 2]) is None
    assert mujoco_eval._pose_triplet([1, object(), 3]) is None

    run = {
        "start_pose": [0, 0, 0],
        "concrete_mutation": {"goal_pose": [1, 1, 0]},
    }
    assert mujoco_eval._run_pose(run, "start_pose") == [0.0, 0.0, 0.0]
    assert mujoco_eval._run_pose(run, "goal_pose") == [1.0, 1.0, 0.0]
    assert mujoco_eval._run_pose(run, "missing") is None
    assert mujoco_eval._dedupe_route([[0, 0, 0], [0.01, 0.01, 0], ["bad"], [1, 1, 0]]) == [
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
    ]
    assert mujoco_eval._existing_route({"route_waypoints": [[0, 0, 0], [1, 1, 0]]}) == [
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
    ]
    assert mujoco_eval._existing_route({"route_waypoints": "bad"}) == []
    assert mujoco_eval._contact_event_count({"metrics": {"near_miss_event_count": 2, "collision_response_event_count": True}}) == 2
    assert mujoco_eval._safety_event_count({"metrics": {"fall_count": 1, "unsafe_proximity_event_count": 2}}) == 3


def test_mujoco_evaluator_route_generation_branches() -> None:
    recipe = {"candidate_id": "policy-1", "mutable_parameters": {"planner": "direct"}}
    missing_pose_run = {"route_waypoints": [[0, 0, 0], [1, 1, 0]]}
    route, strategy = mujoco_eval._route_from_recipe(run=missing_pose_run, recipe=recipe)
    assert strategy == "source_matrix_route_no_pose_available"
    assert route == [[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]]

    direct_route, direct_strategy = mujoco_eval._route_from_recipe(
        run={"start_pose": [0, 0, 0], "target_pose": [2, 2, 0]},
        recipe=recipe,
    )
    assert direct_strategy == "policy_direct_or_source_route"
    assert direct_route == [[0.0, 0.0, 0.0], [2.0, 2.0, 0.0]]

    detour_route, detour_strategy = mujoco_eval._route_from_recipe(
        run={"start_pose": [0, 0, 0], "target_pose": [2, 2, 0]},
        recipe={"mutable_parameters": {"planner": "clearance_aware", "detour_y": "3.5", "detour_x": "1.25"}},
    )
    assert detour_strategy == "policy_clearance_aware_detour_route"
    assert detour_route[1:4] == [[0.0, 3.5, 0.0], [1.25, 3.5, 0.0], [2.0, 3.5, 0.0]]

    default_detour, _ = mujoco_eval._route_from_recipe(
        run={"spawn_xyz": [0, 0, 0], "goal_xyz": [2, 2, 0]},
        recipe={"mutableParameters": {"planner": "safety_margin"}},
    )
    assert default_detour[1][1] == 8.8

    perimeter, perimeter_strategy = mujoco_eval._route_from_recipe(
        run={"robot_spawn_pose": [3, 0, 0], "robot_target_pose": [-2, 2, 0]},
        recipe={"mutable_parameters": {"planner": "route_replan", "retryBudget": 1, "routeStyle": "perimeter_south"}},
    )
    assert perimeter_strategy == "policy_perimeter_clearance_route"
    assert perimeter[1][1] == -9.0
    assert perimeter[1][0] > 0
    assert perimeter[3][0] < 0


def test_build_candidate_matrix_and_normalize_attempts(tmp_path: Path) -> None:
    candidate = mujoco_eval.build_candidate_matrix(
        recipe={"candidate_id": "policy-1", "mutable_parameters": {"planner": "clearance_aware"}},
        split_matrix={
            "schema_version": "source.v1",
            "runs": [
                {"scenario_eval_run_id": "run-1", "start_pose": [0, 0, 0], "target_pose": [1, 1, 0]},
                "bad",
            ],
        },
    )
    assert candidate["schema_version"] == "policy_autoresearch_mujoco_candidate_matrix.v1"
    assert candidate["source_schema_version"] == "source.v1"
    assert candidate["policy_id"] == "policy-1"
    assert candidate["scenario_eval_run_count"] == 1
    assert candidate["runs"][0]["policy_generated_route_waypoint_count"] == 4

    normalized = mujoco_eval._normalize_mujoco_attempts(
        attempts=[
            {
                "attempt_id": "",
                "success": True,
                "metrics": {
                    "fall_count": 1,
                    "robot_scene_contact_event_count": 2,
                },
                "claim_boundary": {"simulator": True},
            },
            {"task_success": False, "metrics": {"unsafe_proximity_event_count": 1}},
        ],
        recipe={"candidate_id": "Policy One", "policyKind": "route"},
        candidate_matrix_path=tmp_path / "candidate.json",
        simulator_output_path=tmp_path / "simulator.json",
        generated_at="2026-06-20T00:00:00Z",
    )

    assert normalized[0]["attempt_id"] == "policy_one_mujoco_0001"
    assert normalized[0]["success"] is True
    assert normalized[0]["metrics"]["safety_event_count"] == 1
    assert normalized[0]["metrics"]["contact_event_count"] == 2
    assert normalized[0]["claim_boundary"]["mujoco_attempt_claim_boundary"] == {"simulator": True}
    assert normalized[1]["task_success"] is False


def test_run_mujoco_policy_evaluator_completed_and_blocked(monkeypatch, tmp_path: Path) -> None:
    recipe_path = tmp_path / "recipe.json"
    matrix_path = tmp_path / "matrix.json"
    output_path = tmp_path / "out" / "mujoco_eval.json"
    capture_root = tmp_path / "capture"
    _write_json(recipe_path, {"candidate_id": "policy-1", "mutable_parameters": {"planner": "direct"}})
    _write_json(matrix_path, {"schema_version": "matrix.v1", "runs": [{"scenario_eval_run_id": "run-1"}]})

    calls: list[dict[str, object]] = []

    def fake_simulator(**kwargs):
        calls.append(kwargs)
        return {
            "status": "completed",
            "simulator_execution_proven": True,
            "attempts": [{"attempt_id": "sim-1", "task_success": True, "metrics": {}}],
        }

    monkeypatch.setenv("BLUEPRINT_POLICY_AUTORESEARCH_PHASE", "heldout")
    monkeypatch.setenv("BLUEPRINT_POLICY_AUTORESEARCH_VERIFIER_SHA256", "verifier")
    completed = mujoco_eval.run_mujoco_policy_evaluator(
        recipe_path=recipe_path,
        matrix_path=matrix_path,
        output_path=output_path,
        capture_root=capture_root,
        g1_model_root=tmp_path / "g1",
        steps=0,
        output_root=tmp_path / "mujoco-out",
        simulator_runner=fake_simulator,
        generated_at="2026-06-20T00:00:00Z",
    )

    assert completed["status"] == "completed"
    assert completed["phase"] == "heldout"
    assert completed["frozen_verifier_sha256"] == "verifier"
    assert completed["simulator_execution_proven"] is True
    assert completed["attempts"][0]["attempt_id"] == "sim-1"
    assert calls[0]["steps"] == 1
    assert calls[0]["g1_model_root"] == tmp_path / "g1"
    assert Path(completed["candidate_matrix_path"]).is_file()
    assert output_path.is_file()

    blocked = mujoco_eval.run_mujoco_policy_evaluator(
        recipe_path=recipe_path,
        matrix_path=matrix_path,
        output_path=tmp_path / "blocked.json",
        capture_root=capture_root,
        simulator_runner=lambda **_kwargs: {"status": "blocked", "attempts": {}},
        generated_at="2026-06-20T00:00:00Z",
    )
    assert blocked["status"] == "blocked_no_mujoco_attempts"
    assert blocked["attempts"] == []


def test_mujoco_policy_evaluator_main_and_module_guard(monkeypatch, tmp_path: Path, capsys) -> None:
    for name in [
        "BLUEPRINT_POLICY_AUTORESEARCH_RECIPE",
        "BLUEPRINT_POLICY_AUTORESEARCH_MATRIX",
        "BLUEPRINT_POLICY_AUTORESEARCH_OUTPUT",
        "BLUEPRINT_POLICY_AUTORESEARCH_CAPTURE_ROOT",
    ]:
        monkeypatch.delenv(name, raising=False)
    assert mujoco_eval.main([]) == 2
    assert "blocked_missing_env" in capsys.readouterr().out

    recipe_path = tmp_path / "recipe.json"
    matrix_path = tmp_path / "matrix.json"
    output_path = tmp_path / "out.json"
    capture_root = tmp_path / "capture"
    output_dir = tmp_path / "mujoco-output"
    for key, value in {
        "BLUEPRINT_POLICY_AUTORESEARCH_RECIPE": recipe_path,
        "BLUEPRINT_POLICY_AUTORESEARCH_MATRIX": matrix_path,
        "BLUEPRINT_POLICY_AUTORESEARCH_OUTPUT": output_path,
        "BLUEPRINT_POLICY_AUTORESEARCH_CAPTURE_ROOT": capture_root,
        "BLUEPRINT_POLICY_AUTORESEARCH_MUJOCO_G1_MODEL_ROOT": tmp_path / "g1",
        "BLUEPRINT_POLICY_AUTORESEARCH_MUJOCO_OUTPUT_DIR": output_dir,
        "BLUEPRINT_POLICY_AUTORESEARCH_MUJOCO_STEPS": "bad",
    }.items():
        monkeypatch.setenv(key, str(value))

    calls: list[dict[str, object]] = []

    def fake_run(**kwargs):
        calls.append(kwargs)
        return {"status": "completed"}

    monkeypatch.setattr(mujoco_eval, "run_mujoco_policy_evaluator", fake_run)
    assert mujoco_eval.main([]) == 0
    assert calls[-1]["g1_model_root"] == (tmp_path / "g1").resolve()
    assert calls[-1]["output_root"] == output_dir.resolve()
    assert calls[-1]["steps"] == 64

    monkeypatch.setattr(sys, "argv", ["mujoco-eval"])
    monkeypatch.delenv("BLUEPRINT_POLICY_AUTORESEARCH_RECIPE", raising=False)
    with pytest.warns(RuntimeWarning, match="found in sys.modules"):
        with pytest.raises(SystemExit) as excinfo:
            runpy.run_module("blueprint_pipeline.policy_autoresearch_mujoco_evaluator", run_name="__main__")
    assert excinfo.value.code == 2
