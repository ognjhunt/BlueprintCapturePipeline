from __future__ import annotations

import json
from pathlib import Path

import pytest

from blueprint_pipeline import policy_autoresearch_wam_fixture_evaluator as evaluator


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_policy_autoresearch_wam_fixture_evaluator_writes_completed_payload(
    tmp_path: Path,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    recipe_path = tmp_path / "recipe.json"
    matrix_path = tmp_path / "matrix.json"
    output_path = tmp_path / "output.json"
    _write_json(
        recipe_path,
        {
            "policy_id": "candidate-a",
            "policy_kind": "navigation",
            "mutableParameters": {
                "planner": "route_replan",
                "clearanceMarginM": 0.2,
                "dynamicObstacleYield": True,
                "perceptionVoteCount": 2,
            },
        },
    )
    _write_json(
        matrix_path,
        {
            "phase": "train",
            "evaluationSubstrate": "wam_fixture",
            "runs": [
                {
                    "scenario_eval_run_id": "run-a",
                    "scenario_variation_instance_id": "var-a",
                    "scenario_id": "scenario-a",
                    "task_id": "task-a",
                    "variation_name": "blocked_path",
                    "required_policy_capabilities": [
                        "clearance_aware_navigation",
                        "visual_recheck",
                    ],
                },
                {
                    "scenario_eval_run_id": "run-b",
                    "scenario_id": "scenario-b",
                    "task_id": "task-b",
                    "variation_name": "human_crossing",
                    "required_policy_capabilities": ["retry_recovery"],
                },
                ["ignored"],
            ],
        },
    )
    monkeypatch.setenv("BLUEPRINT_POLICY_AUTORESEARCH_PHASE", "ignored-env-phase")

    payload = evaluator.run_policy_autoresearch_wam_fixture_evaluator(
        recipe_path=recipe_path,
        matrix_path=matrix_path,
        output_path=output_path,
        generated_at="2026-06-20T00:00:00Z",
    )

    assert payload == json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["status"] == "completed"
    assert payload["evaluation_substrate"] == "fixture_wam"
    assert payload["phase"] == "train"
    assert payload["attempt_count"] == 2
    assert payload["task_success_rate"] == 0.5
    first, second = payload["attempts"]
    assert first["task_success"] is True
    assert first["evaluation_substrate"] == "fixture_wam"
    assert first["metrics"]["world_model_uncertainty"] == 0.12
    assert first["claim_boundary"]["simulator_execution_performed"] is False
    assert second["task_success"] is False
    assert second["metrics"]["world_model_uncertainty"] == 0.36
    assert payload["claim_boundary"]["fixture_wam_is_deterministic_local_test_substrate"] is True


def test_policy_autoresearch_wam_fixture_evaluator_blocks_without_runs_and_uses_env(
    tmp_path: Path,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    recipe_path = tmp_path / "recipe.json"
    matrix_path = tmp_path / "matrix.json"
    output_path = tmp_path / "output.json"
    _write_json(recipe_path, {"policy_id": "candidate-empty"})
    _write_json(matrix_path, {"runs": []})
    monkeypatch.setenv("BLUEPRINT_POLICY_AUTORESEARCH_PHASE", "heldout")
    monkeypatch.setenv("BLUEPRINT_POLICY_AUTORESEARCH_EVALUATION_SUBSTRATE", "fixture_wam")

    payload = evaluator.run_policy_autoresearch_wam_fixture_evaluator(
        recipe_path=recipe_path,
        matrix_path=matrix_path,
        output_path=output_path,
        evaluation_substrate=None,
        generated_at="2026-06-20T00:00:00Z",
    )

    assert payload["status"] == "blocked_missing_eval_runs"
    assert payload["phase"] == "heldout"
    assert payload["attempt_count"] == 0
    assert payload["task_success_rate"] == 0.0


def test_policy_autoresearch_wam_fixture_evaluator_cli_success_and_error(
    tmp_path: Path,
    capsys,
) -> None:
    with pytest.raises(SystemExit) as exc:
        evaluator.main([])
    assert exc.value.code == 2

    recipe_path = tmp_path / "recipe.json"
    matrix_path = tmp_path / "matrix.json"
    output_path = tmp_path / "output.json"
    _write_json(recipe_path, {"policy_id": "candidate-cli"})
    _write_json(
        matrix_path,
        {
            "runs": [
                {
                    "scenario_eval_run_id": "run-cli",
                    "required_policy_capabilities": [],
                }
            ]
        },
    )

    assert (
        evaluator.main(
            [
                "--recipe",
                str(recipe_path),
                "--matrix",
                str(matrix_path),
                "--output",
                str(output_path),
                "--evaluation-substrate",
                "fixture_wam",
            ]
        )
        == 0
    )
    stdout = capsys.readouterr().out
    assert "[policy-autoresearch-wam-fixture] status=completed" in stdout
    assert str(output_path.resolve()) in stdout

    empty_matrix = tmp_path / "empty-matrix.json"
    blocked_output = tmp_path / "blocked-output.json"
    _write_json(empty_matrix, {"runs": []})
    assert (
        evaluator.main(
            [
                "--recipe",
                str(recipe_path),
                "--matrix",
                str(empty_matrix),
                "--output",
                str(blocked_output),
            ]
        )
        == 1
    )
