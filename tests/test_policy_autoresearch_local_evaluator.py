from __future__ import annotations

import json
import runpy
from pathlib import Path

import pytest

from blueprint_pipeline import policy_autoresearch_local_evaluator as evaluator


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_local_replay_scalar_and_failure_helpers_cover_edge_inputs() -> None:
    assert evaluator._string(None) == ""
    assert evaluator._string("  value ") == "value"
    assert evaluator._mapping({"a": 1}) == {"a": 1}
    assert evaluator._mapping(["not", "mapping"]) == {}
    assert evaluator._float(True, 7.5) == 7.5
    assert evaluator._float("3.25") == 3.25
    assert evaluator._float("bad", 1.25) == 1.25
    assert evaluator._int(True, 9) == 9
    assert evaluator._int(4) == 4
    assert evaluator._int("5.0") == 5
    assert evaluator._int("bad", 6) == 6
    assert evaluator._string_list("x") == ["x"]
    assert evaluator._string_list([" a ", "", None, "b"]) == ["a", "b"]
    assert evaluator._string_list(123) == []

    recipe = {
        "mutable_parameters": {
            "planner": "route_replan",
            "clearance_margin_m": "0.2",
            "dynamic_obstacle_yield": True,
            "perception_vote_count": "2",
            "retry_budget": "1",
            "grasp_alignment_correction": True,
        }
    }
    capabilities = evaluator._derive_policy_capabilities(recipe)
    assert capabilities == {
        "clearance_aware_navigation",
        "dynamic_obstacle_yield",
        "visual_recheck",
        "retry_recovery",
        "grasp_alignment_correction",
    }
    assert evaluator._failure_requires("failure_timeout") == "retry_recovery"
    assert (
        evaluator._failure_requires("failure_collision_probe_no_safe_pose")
        == "clearance_aware_navigation"
    )
    assert evaluator._failure_requires("failure_grasp_alignment") == "grasp_alignment_correction"
    assert evaluator._failure_requires("unknown") is None
    assert evaluator._remaining_failure_modes(
        [
            "failure_timeout",
            "failure_endpoint_not_clean",
            "failure_target_not_reached",
            "failure_perception_uncertainty",
        ],
        {"retry_recovery", "clearance_aware_navigation", "visual_recheck"},
    ) == []
    assert evaluator._safety_event_count(
        ["failure_dynamic_obstacle", "failure_safety_threshold_violation", "other"]
    ) == 2
    assert evaluator._contact_event_count(
        ["failure_clearance_near_miss", "failure_contact_collision", "other"]
    ) == 2


def test_local_replay_trace_loading_and_resolution(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    jsonl_path = tmp_path / "attempts.jsonl"
    jsonl_path.write_text(
        "\n".join(
            [
                "",
                json.dumps({"scenario_eval_run_id": "run-a", "success": False}),
                json.dumps(["not", "mapping"]),
            ]
        ),
        encoding="utf-8",
    )
    assert evaluator._read_jsonl(jsonl_path) == [
        {"scenario_eval_run_id": "run-a", "success": False}
    ]
    assert evaluator._load_attempts(jsonl_path) == [
        {"scenario_eval_run_id": "run-a", "success": False}
    ]

    payload_path = tmp_path / "trace.json"
    _write_json(payload_path, {"results": [{"scenario_eval_run_id": "run-b"}]})
    assert evaluator._load_attempts(payload_path) == [{"scenario_eval_run_id": "run-b"}]
    assert evaluator._attempts_from_payload({"episodes": [{"scenario_eval_run_id": "run-c"}]}) == [
        {"scenario_eval_run_id": "run-c"}
    ]
    assert evaluator._attempts_from_payload({"metadata": "no attempts"}) == []
    assert evaluator._attempts_from_payload([{"scenario_eval_run_id": "run-d"}, []]) == [
        {"scenario_eval_run_id": "run-d"}
    ]
    assert evaluator._attempts_from_payload("bad") == []

    assert evaluator._candidate_trace_paths(None) == []
    assert len(evaluator._candidate_trace_paths(tmp_path)) == len(evaluator.TRACE_CANDIDATE_NAMES) * 2
    monkeypatch.setenv("BLUEPRINT_POLICY_AUTORESEARCH_ATTEMPT_TRACE", str(tmp_path / "missing.json"))
    assert evaluator._resolve_attempt_trace() is None
    monkeypatch.setenv("BLUEPRINT_POLICY_AUTORESEARCH_ATTEMPT_TRACE", str(jsonl_path))
    assert evaluator._resolve_attempt_trace() == jsonl_path.resolve()
    monkeypatch.delenv("BLUEPRINT_POLICY_AUTORESEARCH_ATTEMPT_TRACE")
    nested = tmp_path / "simulation_automation" / "mujoco_g1_simulator_command"
    nested.mkdir(parents=True)
    nested_trace = nested / evaluator.TRACE_CANDIDATE_NAMES[0]
    nested_trace.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("BLUEPRINT_POLICY_AUTORESEARCH_JOB_DIR", str(tmp_path))
    assert evaluator._resolve_attempt_trace() == nested_trace.resolve()


def test_local_replay_attempt_building_uses_observed_and_fallback_failures(
    tmp_path: Path,
) -> None:
    observed = {
        "attempt_id": "attempt-a",
        "scenario_eval_run_id": "run-a",
        "scenario_variation_instance_id": "var-a",
        "task_id": "task-a",
        "scenario_id": "scenario-a",
        "variation_name": "blocked_path",
        "success": False,
        "failure_mode_ids": ["failure_timeout", "failure_dynamic_obstacle"],
        "metrics": {"stuck_event_count": 1},
        "task_outcome": {"original": True},
        "artifactPaths": {"trace": "trace.json"},
    }
    attempt = evaluator._build_attempt(
        run={"scenario_eval_run_id": "run-a"},
        observed_attempt=observed,
        recipe={"policy_id": "policy-a", "policy_kind": "navigation"},
        capabilities={"retry_recovery"},
        source_attempt_trace_path=tmp_path / "trace.json",
        generated_at="2026-06-20T00:00:00Z",
    )
    assert attempt["attempt_id"] == "attempt-a"
    assert attempt["status"] == "failed_counterfactual_replay"
    assert attempt["initial_failure_mode_ids"] == [
        "failure_dynamic_obstacle",
        "failure_timeout",
    ]
    assert attempt["failure_mode_ids"] == ["failure_dynamic_obstacle"]
    assert attempt["metrics"]["safety_event_count"] == 1
    assert attempt["metrics"]["contact_event_count"] == 0
    assert attempt["artifact_paths"] == {"trace": "trace.json"}

    inferred = evaluator._failure_modes_from_attempt(
        {
            "metrics": {
                "clearance_threshold_violation": True,
                "timeout_count": 1,
                "stuck_event_count": 1,
                "policy_instability_detected": True,
                "endpoint_clean": False,
                "goal_reached": False,
            }
        }
    )
    assert "failure_clearance_near_miss" in inferred
    assert "failure_target_not_reached" in inferred
    assert evaluator._failure_modes_from_attempt(
        {"task_outcome": {"failure_mode_ids": ["failure_grasp_alignment"]}}
    ) == ["failure_grasp_alignment"]

    fallback_modes = evaluator._fallback_failure_modes(
        {
            "scenario_eval_run_id": "run-blocked-human-glare-grasp",
            "required_policy_capabilities": ["clearance_aware_navigation"],
        },
        {"dynamic_obstacle_yield"},
    )
    assert fallback_modes == [
        "failure_clearance_near_miss",
        "failure_grasp_alignment",
        "failure_perception_uncertainty",
    ]
    fallback_attempt = evaluator._build_attempt(
        run={
            "scenario_eval_run_id": "run-blocked",
            "scenario_id": "scenario-a",
            "task_id": "task-a",
            "variation_name": "blocked_path",
        },
        observed_attempt=None,
        recipe={"policy_id": "policy-a"},
        capabilities={"clearance_aware_navigation"},
        source_attempt_trace_path=None,
        generated_at="2026-06-20T00:00:00Z",
    )
    assert fallback_attempt["attempt_id"] == "local_replay_run-blocked"
    assert fallback_attempt["success"] is True
    assert fallback_attempt["source_attempt_trace_path"] is None


def test_run_local_replay_evaluator_writes_counterfactual_payload(
    tmp_path: Path,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    recipe_path = tmp_path / "recipe.json"
    matrix_path = tmp_path / "matrix.json"
    trace_path = tmp_path / "attempts.jsonl"
    output_path = tmp_path / "output.json"
    _write_json(
        recipe_path,
        {
            "policy_id": "policy-replay",
            "mutableParameters": {
                "planner": "clearance_aware",
                "clearanceMarginM": 0.2,
                "retryBudget": 1,
            },
        },
    )
    _write_json(
        matrix_path,
        {
            "runs": [
                {
                    "scenario_eval_run_id": "run-a",
                    "scenario_id": "scenario-a",
                    "task_id": "task-a",
                    "variation_name": "blocked_path",
                },
                {
                    "scenario_eval_run_id": "run-b",
                    "scenario_id": "scenario-a",
                    "task_id": "task-a",
                    "variation_name": "human_crossing",
                },
            ]
        },
    )
    trace_path.write_text(
        json.dumps(
            {
                "attempt_id": "attempt-a",
                "scenario_eval_run_id": "run-a",
                "success": False,
                "failure_mode_ids": ["failure_timeout", "failure_target_not_reached"],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("BLUEPRINT_POLICY_AUTORESEARCH_ATTEMPT_TRACE", str(trace_path))
    monkeypatch.setenv("BLUEPRINT_POLICY_AUTORESEARCH_PHASE", "heldout")
    monkeypatch.setenv("BLUEPRINT_POLICY_AUTORESEARCH_SIMULATOR_ENGINE", "mujoco")
    monkeypatch.setenv("BLUEPRINT_POLICY_AUTORESEARCH_VERIFIER_SHA256", "abc123")

    payload = evaluator.run_local_replay_evaluator(
        recipe_path=recipe_path,
        matrix_path=matrix_path,
        output_path=output_path,
        generated_at="2026-06-20T00:00:00Z",
    )

    assert payload == json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["status"] == "completed"
    assert payload["phase"] == "heldout"
    assert payload["simulator_engine"] == "mujoco"
    assert payload["frozen_verifier_sha256"] == "abc123"
    assert payload["source_attempt_trace_found"] is True
    assert payload["observed_attempt_count"] == 1
    assert payload["split_run_count"] == 2
    assert payload["claim_boundary"]["simulator_execution_performed"] is False
    assert payload["attempts"][0]["success"] is True
    assert payload["attempts"][1]["status"] == "failed_counterfactual_replay"


def test_local_replay_main_handles_missing_and_present_env(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:  # type: ignore[no-untyped-def]
    for key in (
        "BLUEPRINT_POLICY_AUTORESEARCH_RECIPE",
        "BLUEPRINT_POLICY_AUTORESEARCH_MATRIX",
        "BLUEPRINT_POLICY_AUTORESEARCH_OUTPUT",
        "BLUEPRINT_POLICY_AUTORESEARCH_ATTEMPT_TRACE",
        "BLUEPRINT_POLICY_AUTORESEARCH_JOB_DIR",
    ):
        monkeypatch.delenv(key, raising=False)
    assert evaluator.main([]) == 2
    missing = json.loads(capsys.readouterr().out)
    assert missing["status"] == "blocked_missing_env"
    assert "BLUEPRINT_POLICY_AUTORESEARCH_RECIPE" in missing["missing_env"]

    recipe_path = tmp_path / "recipe.json"
    matrix_path = tmp_path / "matrix.json"
    output_path = tmp_path / "output.json"
    _write_json(recipe_path, {"policy_id": "policy-empty"})
    _write_json(matrix_path, {"runs": []})
    monkeypatch.setenv("BLUEPRINT_POLICY_AUTORESEARCH_RECIPE", str(recipe_path))
    monkeypatch.setenv("BLUEPRINT_POLICY_AUTORESEARCH_MATRIX", str(matrix_path))
    monkeypatch.setenv("BLUEPRINT_POLICY_AUTORESEARCH_OUTPUT", str(output_path))

    assert evaluator.main(None) == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["status"] == "blocked_no_split_matrix_runs"
    assert payload["attempts"] == []


def test_local_replay_module_main_guard_raises_system_exit(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    for key in (
        "BLUEPRINT_POLICY_AUTORESEARCH_RECIPE",
        "BLUEPRINT_POLICY_AUTORESEARCH_MATRIX",
        "BLUEPRINT_POLICY_AUTORESEARCH_OUTPUT",
    ):
        monkeypatch.delenv(key, raising=False)

    with pytest.warns(RuntimeWarning, match="found in sys.modules"), pytest.raises(
        SystemExit
    ) as exc:
        runpy.run_module("blueprint_pipeline.policy_autoresearch_local_evaluator", run_name="__main__")

    assert exc.value.code == 2
