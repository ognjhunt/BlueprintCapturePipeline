from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from blueprint_pipeline import policy_autoresearch as pa
from tests.runpy_entrypoint import run_module_as_main


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _seed_recipe(path: Path, extra: dict[str, Any] | None = None) -> Path:
    payload: dict[str, Any] = {
        "schema_version": "policy_autoresearch_recipe.v1",
        "policy_id": "site_policy_seed",
        "policy_kind": "code_as_policy_navigation_heuristic",
        "mutable_parameters": {
            "planner": "direct",
            "clearance_margin_m": 0.05,
            "dynamic_obstacle_yield": False,
            "perception_vote_count": 1,
            "retry_budget": 0,
            "max_speed_mps": 0.9,
            "grasp_alignment_correction": False,
        },
    }
    if extra:
        payload.update(extra)
    recipe_path = path / "seed_policy_recipe.json"
    _write_json(recipe_path, payload)
    return recipe_path


def _job_paths(tmp_path: Path) -> tuple[Path, Path]:
    capture_root = tmp_path / "capture-root"
    job_dir = capture_root / "pipeline" / "robot_eval_jobs" / "job-policy-autoresearch"
    job_dir.mkdir(parents=True)
    return capture_root, job_dir


def _write_matrix(job_dir: Path, runs: list[dict[str, Any]]) -> Path:
    matrix_path = job_dir / "scenario_eval_matrix.json"
    _write_json(
        matrix_path,
        {
            "schema_version": "robot_eval_scenario_eval_matrix.v1",
            "runs": runs,
        },
    )
    return matrix_path


def test_policy_autoresearch_private_scalar_and_command_edges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert pa._float(True, 1.25) == 1.25
    assert pa._int(True, 7) == 7
    assert pa._string_list("single") == ["single"]
    assert pa._evaluation_substrate_for_engine("") == ""

    def reject_substrate(value: str, **kwargs: Any) -> str:
        del value, kwargs
        raise ValueError("unsupported")

    monkeypatch.setattr(pa, "normalize_evaluation_substrate", reject_substrate)
    assert pa._evaluation_substrate_for_engine("Bad Engine!") == "bad_engine"

    assert pa._parse_engine_evaluator_commands(["", "mujoco=python eval.py"]) == {
        "mujoco": "python eval.py"
    }
    with pytest.raises(ValueError, match="ENGINE=COMMAND"):
        pa._parse_engine_evaluator_commands(["mujoco"])
    with pytest.raises(ValueError, match="non-empty engine and command"):
        pa._parse_engine_evaluator_commands(["=python eval.py"])
    with pytest.raises(ValueError, match="non-empty engine and command"):
        pa._parse_engine_evaluator_commands(["mujoco="])


def test_policy_autoresearch_private_payload_capability_and_matrix_edges(
    tmp_path: Path,
) -> None:
    assert pa._payload_simulator_engines(
        [
            "ignored",
            {
                "simulatorEngine": "MuJoCo",
                "metrics": {"evaluationSubstrate": "classical_sim_mujoco"},
                "claimBoundary": {"simulator_backend": "mujoco"},
            },
        ]
    ) == ["classical_sim_mujoco", "mujoco"]
    assert pa._payload_simulator_engines("not-json-object") == []

    proven = pa._proven_simulator_engines(
        {
            "attempts": [
                {
                    "simulator_engine": "mujoco",
                    "metrics": {"simulator_execution_performed": True},
                    "claim_boundary": {"simulator_execution_performed": True},
                }
            ]
        }
    )
    assert proven == ["mujoco"]
    assert pa._proven_simulator_engines({"attempts": []}) == []
    assert pa._eval_has_simulator_execution({"attempts": ["bad"]}) is False
    assert pa._eval_has_simulator_execution(
        {
            "attempts": [
                {
                    "metrics": {"simulator_execution_performed": True},
                    "claim_boundary": {"simulator_execution_performed": True},
                }
            ]
        }
    ) is True

    assert pa._failure_capability_from_mode("failure_grasp_alignment") == (
        "grasp_alignment_correction"
    )
    inferred = pa._infer_required_capabilities(
        {
            "scenario_id": "blocked_path_human_crossing_occlusion",
            "variation_name": "grasp_place_insertion",
        }
    )
    assert inferred == [
        "clearance_aware_navigation",
        "dynamic_obstacle_yield",
        "grasp_alignment_correction",
        "visual_recheck",
    ]
    assert pa._derive_policy_capabilities(
        {"mutable_parameters": {"grasp_alignment_correction": True}}
    ) == ["grasp_alignment_correction"]
    assert pa._find_forbidden_recipe_keys(
        [{"safe": {"verifier": "do not allow"}}]
    ) == ["[0].safe.verifier"]

    invalid_matrix = tmp_path / "invalid_matrix.json"
    _write_json(invalid_matrix, {"not_runs": []})
    with pytest.raises(ValueError, match="does not contain runs"):
        pa._load_matrix_runs(invalid_matrix)

    generated_id_matrix = tmp_path / "generated_id_matrix.json"
    _write_json(generated_id_matrix, {"runs": [{"scenario_id": "blocked_path"}]})
    loaded = pa._load_matrix_runs(generated_id_matrix)
    assert loaded[0]["scenario_eval_run_id"] == "scenario_eval_run_0001"
    assert loaded[0]["required_policy_capabilities"] == ["clearance_aware_navigation"]

    train, heldout, split_source = pa._split_runs(
        [
            {"scenario_eval_run_id": "run-1"},
            {"scenario_eval_run_id": "run-2"},
            {"scenario_eval_run_id": "run-3"},
        ],
        heldout_ratio=0.34,
    )
    assert [run["scenario_eval_run_id"] for run in train] == ["run-1", "run-2"]
    assert [run["scenario_eval_run_id"] for run in heldout] == ["run-3"]
    assert split_source == "deterministic_tail_holdout_split"
    single_train, single_heldout, single_split = pa._split_runs(
        [{"scenario_eval_run_id": "single"}],
        heldout_ratio=0.5,
    )
    assert single_train == single_heldout == [{"scenario_eval_run_id": "single"}]
    assert single_split == "single_run_reused_as_heldout"

    attempt = pa._attempt_for_run(
        recipe={"policy_id": "seed", "mutable_parameters": {}},
        run={
            "scenario_eval_run_id": "run-grasp",
            "required_policy_capabilities": ["grasp_alignment_correction"],
        },
        phase="train",
        engine="mujoco",
        generated_at="2026-06-20T00:00:00+00:00",
    )
    assert attempt["failure_mode_ids"] == ["failure_grasp_alignment"]
    assert pa._apply_capability_mutation(
        {"mutable_parameters": {}}, "grasp_alignment_correction"
    )["mutable_parameters"] == {"grasp_alignment_correction": True}


def test_policy_autoresearch_external_attempt_and_evaluator_failure_edges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert pa._normalize_external_attempts(
        payload="not-a-list",
        recipe={"policy_id": "seed"},
        runs=[],
        phase="train",
        engine="mujoco",
        generated_at="2026-06-20T00:00:00+00:00",
    ) == []
    normalized = pa._normalize_external_attempts(
        payload=[
            "ignored",
            {
                "scenarioEvalRunId": "run-1",
                "success": True,
                "simulatorBackend": "mujoco",
            },
        ],
        recipe={"policy_id": "seed"},
        runs=[{"scenario_eval_run_id": "run-1", "task_id": "navigate"}],
        phase="heldout",
        engine="mujoco",
        generated_at="2026-06-20T00:00:00+00:00",
    )
    assert normalized[0]["attempt_id"] == "seed_heldout_external_0002"
    assert normalized[0]["task_id"] == "navigate"

    recipe = {"policy_id": "seed", "mutable_parameters": {}}
    runs = [{"scenario_eval_run_id": "run-1"}]

    monkeypatch.setattr(
        pa.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=2,
            stdout="failed stdout",
            stderr="failed stderr",
        ),
    )
    failed = pa._evaluate_recipe_with_command(
        recipe=recipe,
        runs=runs,
        phase="train",
        engine="mujoco",
        generated_at="2026-06-20T00:00:00+00:00",
        verifier_sha256="sha",
        evaluator_command="python evaluator.py",
        evaluator_timeout_seconds=1,
        eval_root_dir=tmp_path / "failed-evaluator",
    )
    assert failed["status"] == "failed_evaluator_command"
    assert failed["failure_mode_ids"] == ["external_evaluator_command_failed"]

    def write_invalid_output(*args: Any, **kwargs: Any) -> SimpleNamespace:
        output_path = Path(kwargs["env"]["BLUEPRINT_POLICY_AUTORESEARCH_OUTPUT"])
        output_path.write_text("{", encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(pa.subprocess, "run", write_invalid_output)
    invalid = pa._evaluate_recipe_with_command(
        recipe=recipe,
        runs=runs,
        phase="heldout",
        engine="mujoco",
        generated_at="2026-06-20T00:00:00+00:00",
        verifier_sha256="sha",
        evaluator_command="python evaluator.py",
        evaluator_timeout_seconds=1,
        eval_root_dir=tmp_path / "invalid-evaluator",
    )
    assert invalid["status"] == "failed_evaluator_output_invalid"
    assert invalid["failure_mode_ids"] == ["external_evaluator_output_invalid"]


def test_policy_autoresearch_remaining_helper_and_mutation_edges(tmp_path: Path) -> None:
    assert pa._requested_evaluation_substrate_cycle(["mujoco"]) == ["classical_sim_mujoco"]
    assert pa._payload_simulator_engines(
        {
            "simulatorEngine": "MuJoCo",
            "evaluationSubstrate": "classical_sim_mujoco",
            "attempts": [],
        }
    ) == ["classical_sim_mujoco", "mujoco"]
    assert pa._external_payload_engine_mismatch(
        {"simulator_engine": "isaac_sim"}, requested_engine="mujoco"
    ) == ["isaac_sim"]
    assert pa._external_payload_engine_mismatch({}, requested_engine="mujoco") == []
    assert (
        pa._external_payload_engine_mismatch(
            {"simulator_engine": "mujoco"}, requested_engine="mujoco"
        )
        == []
    )

    assert pa._failure_capability_from_mode("failure_dynamic_obstacle") == (
        "dynamic_obstacle_yield"
    )
    assert pa._failure_capability_from_mode("failure_perception_uncertainty") == "visual_recheck"
    assert pa._failure_capability_from_mode("failure_timeout") == "retry_recovery"
    assert pa._failure_capability_from_mode("unknown") is None
    assert pa._derive_policy_capabilities({"mutable_parameters": {"retry_budget": 1}}) == [
        "retry_recovery"
    ]

    attempt = pa._attempt_for_run(
        recipe={
            "policy_id": "seed",
            "mutable_parameters": {"max_speed_mps": 0.9, "clearance_margin_m": 0.2},
        },
        run={
            "scenario_eval_run_id": "dynamic",
            "required_policy_capabilities": ["dynamic_obstacle_yield", "visual_recheck"],
        },
        phase="train",
        engine="mujoco",
        generated_at="2026-06-20T00:00:00+00:00",
    )
    assert attempt["failure_mode_ids"] == [
        "failure_dynamic_obstacle",
        "failure_perception_uncertainty",
        "failure_safety_threshold_violation",
    ]

    dynamic = pa._apply_capability_mutation(
        {"mutable_parameters": {"max_speed_mps": 0.9}}, "dynamic_obstacle_yield"
    )
    assert dynamic["mutable_parameters"]["dynamic_obstacle_yield"] is True
    assert dynamic["mutable_parameters"]["max_speed_mps"] == 0.55
    assert pa._apply_capability_mutation(
        {"mutable_parameters": {"perception_vote_count": 1}}, "visual_recheck"
    )["mutable_parameters"]["perception_vote_count"] == 2
    assert pa._apply_capability_mutation(
        {"mutable_parameters": {"retry_budget": 0}}, "retry_recovery"
    )["mutable_parameters"]["retry_budget"] == 1

    assert pa._mutation_capabilities_from_failures({}, branch_index=0) == ["retry_recovery"]
    multi_failure_eval = {
        "failure_mode_ids": [
            "failure_dynamic_obstacle",
            "failure_perception_uncertainty",
            "failure_timeout",
        ]
    }
    assert pa._mutation_capabilities_from_failures(multi_failure_eval, branch_index=1) == [
        "dynamic_obstacle_yield",
        "visual_recheck",
    ]
    assert pa._mutation_capabilities_from_failures(multi_failure_eval, branch_index=2) == [
        "dynamic_obstacle_yield",
        "visual_recheck",
        "retry_recovery",
    ]

    blocked = pa._blocked_artifacts(
        output_dir=tmp_path / "blocked",
        generated_at="2026-06-20T00:00:00+00:00",
        blockers=["blocked"],
        verifier_manifest={"schema_version": "verifier"},
    )
    assert blocked["verifier_manifest"] == {"schema_version": "verifier"}
    assert (tmp_path / "blocked" / "verifier_manifest.json").is_file()


def test_policy_autoresearch_external_evaluator_success_and_mismatch_edges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recipe = {"policy_id": "seed", "policy_kind": "test", "mutable_parameters": {}}
    runs = [{"scenario_eval_run_id": "run-1", "task_id": "navigate"}]

    def write_mismatched_output(*args: Any, **kwargs: Any) -> SimpleNamespace:
        output_path = Path(kwargs["env"]["BLUEPRINT_POLICY_AUTORESEARCH_OUTPUT"])
        _write_json(output_path, {"simulator_engine": "isaac_sim", "attempts": []})
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(pa.subprocess, "run", write_mismatched_output)
    mismatch = pa._evaluate_recipe_with_command(
        recipe=recipe,
        runs=runs,
        phase="train",
        engine="mujoco",
        generated_at="2026-06-20T00:00:00+00:00",
        verifier_sha256="sha",
        evaluator_command="python evaluator.py",
        evaluator_timeout_seconds=1,
        eval_root_dir=tmp_path / "mismatch",
    )
    assert mismatch["status"] == "failed_evaluator_engine_mismatch"
    assert mismatch["failure_mode_ids"] == ["external_evaluator_engine_mismatch"]

    capture_root = tmp_path / "capture"
    job_dir = tmp_path / "job"
    matrix_path = tmp_path / "matrix.json"
    attempt_trace_path = tmp_path / "attempts.json"

    def write_success_output(*args: Any, **kwargs: Any) -> SimpleNamespace:
        env = kwargs["env"]
        assert env["BLUEPRINT_POLICY_AUTORESEARCH_CAPTURE_ROOT"] == str(capture_root)
        assert env["BLUEPRINT_POLICY_AUTORESEARCH_JOB_DIR"] == str(job_dir)
        assert env["BLUEPRINT_POLICY_AUTORESEARCH_SOURCE_MATRIX"] == str(matrix_path)
        assert env["BLUEPRINT_POLICY_AUTORESEARCH_ATTEMPT_TRACE"] == str(attempt_trace_path)
        output_path = Path(env["BLUEPRINT_POLICY_AUTORESEARCH_OUTPUT"])
        _write_json(
            output_path,
            {
                "simulatorEngine": "mujoco",
                "evaluationSubstrate": "classical_sim_mujoco",
                "attempts": [{"scenarioEvalRunId": "run-1", "task_success": True}],
            },
        )
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(pa.subprocess, "run", write_success_output)
    success = pa._evaluate_recipe_with_command(
        recipe=recipe,
        runs=runs,
        phase="heldout",
        engine="mujoco",
        generated_at="2026-06-20T00:00:00+00:00",
        verifier_sha256="sha",
        evaluator_command="python evaluator.py",
        evaluator_timeout_seconds=1,
        eval_root_dir=tmp_path / "success",
        source_capture_root=capture_root,
        source_job_dir=job_dir,
        source_matrix_path=matrix_path,
        source_attempt_trace_path=attempt_trace_path,
    )
    assert success["status"] == "completed"
    assert success["attempts"][0]["evaluation_substrate"] == "classical_sim_mujoco"

    monkeypatch.setattr(pa, "_evaluate_recipe_with_command", lambda **kwargs: {"status": "via-command"})
    assert (
        pa._evaluate_recipe(
            recipe=recipe,
            runs=runs,
            phase="train",
            engine="mujoco",
            generated_at="2026-06-20T00:00:00+00:00",
            verifier_sha256="sha",
            evaluator_command="python evaluator.py",
            eval_root_dir=tmp_path / "dispatch",
        )["status"]
        == "via-command"
    )


def test_policy_autoresearch_blocked_input_and_empty_engine_edges(tmp_path: Path) -> None:
    missing_output = tmp_path / "missing-output"
    blocked = pa.run_policy_autoresearch(
        capture_root=tmp_path / "missing-capture",
        job_dir=tmp_path / "missing-job",
        policy_recipe_path=tmp_path / "missing-recipe.json",
        reviewed_examples_path=tmp_path / "missing-reviewed.json",
        evaluator_attempt_trace_path=tmp_path / "missing-trace.jsonl",
        output_dir=missing_output,
        generated_at="2026-06-20T00:00:00+00:00",
    )
    assert blocked["report"]["status"] == "blocked"
    assert {
        "capture_root_missing",
        "scenario_eval_matrix_missing",
        "policy_recipe_missing",
        "reviewed_examples_missing",
        "evaluator_attempt_trace_missing",
    }.issubset(set(blocked["report"]["blockers"]))

    capture_root, job_dir = _job_paths(tmp_path / "invalid")
    invalid_matrix = job_dir / "scenario_eval_matrix.json"
    _write_json(invalid_matrix, {"not_runs": []})
    invalid = pa.run_policy_autoresearch(
        capture_root=capture_root,
        job_dir=job_dir,
        policy_recipe_path=_seed_recipe(tmp_path / "invalid"),
        generated_at="2026-06-20T00:00:00+00:00",
    )
    assert invalid["report"]["blockers"][0].startswith("scenario_eval_matrix_invalid:")

    capture_root, job_dir = _job_paths(tmp_path / "empty")
    _write_matrix(job_dir, [])
    empty = pa.run_policy_autoresearch(
        capture_root=capture_root,
        job_dir=job_dir,
        policy_recipe_path=_seed_recipe(tmp_path / "empty"),
        generated_at="2026-06-20T00:00:00+00:00",
    )
    assert empty["report"]["blockers"] == ["scenario_eval_matrix_empty"]

    capture_root, job_dir = _job_paths(tmp_path / "default-engine")
    _write_matrix(
        job_dir,
        [
            {"scenario_eval_run_id": "train", "split": "train"},
            {"scenario_eval_run_id": "heldout", "split": "heldout"},
        ],
    )
    class EmptyTruthyEngines:
        def __bool__(self) -> bool:
            return True

        def __iter__(self):
            return iter(())

    completed = pa.run_policy_autoresearch(
        capture_root=capture_root,
        job_dir=job_dir,
        policy_recipe_path=_seed_recipe(tmp_path / "default-engine"),
        simulator_engines=EmptyTruthyEngines(),
        max_iterations=0,
        target_success_rate=0.0,
        generated_at="2026-06-20T00:00:00+00:00",
    )
    assert completed["report"]["requested_simulator_engines"] == ["mujoco"]


def test_policy_autoresearch_blocked_substrate_and_recipe_guard_edges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture_root, job_dir = _job_paths(tmp_path / "bad-substrate")
    _write_matrix(job_dir, [{"scenario_eval_run_id": "run"}])
    recipe_path = _seed_recipe(tmp_path / "bad-substrate")

    def reject_substrate(value: str, **kwargs: Any) -> str:
        del kwargs
        raise ValueError(f"bad substrate {value}")

    monkeypatch.setattr(pa, "normalize_evaluation_substrate", reject_substrate)
    blocked_substrate = pa.run_policy_autoresearch(
        capture_root=capture_root,
        job_dir=job_dir,
        policy_recipe_path=recipe_path,
        evaluation_substrates=["bad"],
        generated_at="2026-06-20T00:00:00+00:00",
    )
    assert blocked_substrate["report"]["blockers"][0].startswith(
        "unsupported_evaluation_substrate:"
    )

    monkeypatch.undo()
    capture_root, job_dir = _job_paths(tmp_path / "forbidden")
    _write_matrix(job_dir, [{"scenario_eval_run_id": "run"}])
    forbidden_recipe = _seed_recipe(
        tmp_path / "forbidden",
        {"mutable_parameters": {"verifier": "not allowed"}},
    )
    forbidden_output = tmp_path / "forbidden-output"
    forbidden = pa.run_policy_autoresearch(
        capture_root=capture_root,
        job_dir=job_dir,
        policy_recipe_path=forbidden_recipe,
        output_dir=forbidden_output,
        generated_at="2026-06-20T00:00:00+00:00",
    )
    assert forbidden["report"]["blockers"] == ["forbidden_policy_recipe_keys"]
    verifier = json.loads((forbidden_output / "verifier_manifest.json").read_text())
    assert verifier["forbidden_policy_recipe_key_paths"] == ["mutable_parameters.verifier"]


def test_policy_autoresearch_budget_and_parallel_exception_edges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert set(
        pa._budget_limit_reached(
            {
                "limits": {
                    "token_budget": 10,
                    "compute_seconds_budget": 2.0,
                    "wall_time_budget_seconds": 1.0,
                },
                "usage": {
                    "estimated_tokens": 10,
                    "compute_seconds": 2.0,
                    "wall_time_seconds": 1.0,
                },
            }
        )
    ) == {
        "estimated_token_budget_exhausted",
        "compute_seconds_budget_exhausted",
        "wall_time_budget_exhausted",
    }

    capture_root, job_dir = _job_paths(tmp_path / "branch-budget")
    _write_matrix(
        job_dir,
        [
            {
                "scenario_eval_run_id": "train",
                "split": "train",
                "required_policy_capabilities": ["clearance_aware_navigation"],
            },
            {
                "scenario_eval_run_id": "heldout",
                "split": "heldout",
                "required_policy_capabilities": ["clearance_aware_navigation"],
            },
        ],
    )
    branch_budget = pa.run_policy_autoresearch(
        capture_root=capture_root,
        job_dir=job_dir,
        policy_recipe_path=_seed_recipe(tmp_path / "branch-budget"),
        max_iterations=1,
        agent_count=2,
        max_candidate_evaluations=1,
        target_success_rate=1.0,
        generated_at="2026-06-20T00:00:00+00:00",
    )
    assert any(
        event["event"] == "branch_not_planned_budget_exhausted"
        for event in branch_budget["budget_ledger"]["events"]
    )

    capture_root, job_dir = _job_paths(tmp_path / "no-candidates")
    _write_matrix(
        job_dir,
        [
            {
                "scenario_eval_run_id": "train",
                "split": "train",
                "required_policy_capabilities": ["clearance_aware_navigation"],
            },
            {
                "scenario_eval_run_id": "heldout",
                "split": "heldout",
                "required_policy_capabilities": ["clearance_aware_navigation"],
            },
        ],
    )
    budget_checks = iter([[], ["synthetic_budget_exhausted"]])
    monkeypatch.setattr(pa, "_budget_limit_reached", lambda ledger: next(budget_checks))
    no_candidates = pa.run_policy_autoresearch(
        capture_root=capture_root,
        job_dir=job_dir,
        policy_recipe_path=_seed_recipe(tmp_path / "no-candidates"),
        max_iterations=1,
        agent_count=1,
        target_success_rate=1.0,
        generated_at="2026-06-20T00:00:00+00:00",
    )
    assert no_candidates["report"]["iteration_records"] == []
    assert "synthetic_budget_exhausted" in no_candidates["report"]["blockers"]

    monkeypatch.undo()
    capture_root, job_dir = _job_paths(tmp_path / "branch-exception")
    _write_matrix(
        job_dir,
        [
            {
                "scenario_eval_run_id": "train",
                "split": "train",
                "required_policy_capabilities": ["clearance_aware_navigation"],
            },
            {
                "scenario_eval_run_id": "heldout",
                "split": "heldout",
                "required_policy_capabilities": ["clearance_aware_navigation"],
            },
        ],
    )
    original_evaluate_recipe = pa._evaluate_recipe

    def raise_for_candidate_recipe(**kwargs: Any) -> dict[str, Any]:
        recipe = kwargs["recipe"]
        if recipe.get("mutation_parent_policy_id"):
            raise RuntimeError("candidate branch failed")
        return original_evaluate_recipe(**kwargs)

    monkeypatch.setattr(pa, "_evaluate_recipe", raise_for_candidate_recipe)
    exception_result = pa.run_policy_autoresearch(
        capture_root=capture_root,
        job_dir=job_dir,
        policy_recipe_path=_seed_recipe(tmp_path / "branch-exception"),
        max_iterations=1,
        agent_count=1,
        target_success_rate=1.0,
        generated_at="2026-06-20T00:00:00+00:00",
    )
    assert exception_result["report"]["iteration_records"][0]["train_status"] == (
        "blocked_missing_eval_runs"
    )
    assert exception_result["report"]["iteration_records"][0]["parallel_branch"] is True


def test_policy_autoresearch_iteration_budget_and_target_break_edges(tmp_path: Path) -> None:
    capture_root, job_dir = _job_paths(tmp_path / "iteration-budget")
    _write_matrix(
        job_dir,
        [
            {
                "scenario_eval_run_id": "train",
                "split": "train",
                "required_policy_capabilities": ["clearance_aware_navigation"],
            },
            {
                "scenario_eval_run_id": "heldout",
                "split": "heldout",
                "required_policy_capabilities": ["clearance_aware_navigation"],
            },
        ],
    )
    budget_blocked = pa.run_policy_autoresearch(
        capture_root=capture_root,
        job_dir=job_dir,
        policy_recipe_path=_seed_recipe(tmp_path / "iteration-budget"),
        max_iterations=1,
        target_success_rate=1.0,
        token_budget=0,
        generated_at="2026-06-20T00:00:00+00:00",
    )
    assert any(
        event["event"] == "iteration_not_started_budget_exhausted"
        for event in budget_blocked["budget_ledger"]["events"]
    )

    capture_root, job_dir = _job_paths(tmp_path / "target-met")
    _write_matrix(
        job_dir,
        [
            {"scenario_eval_run_id": "train", "split": "train"},
            {"scenario_eval_run_id": "heldout", "split": "heldout"},
        ],
    )
    target_met = pa.run_policy_autoresearch(
        capture_root=capture_root,
        job_dir=job_dir,
        policy_recipe_path=_seed_recipe(tmp_path / "target-met"),
        max_iterations=1,
        target_success_rate=0.0,
        generated_at="2026-06-20T00:00:00+00:00",
    )
    assert target_met["report"]["iteration_records"] == []


def test_policy_autoresearch_cli_edges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    captured_kwargs: dict[str, Any] = {}

    def fake_run_policy_autoresearch(**kwargs: Any) -> dict[str, Any]:
        captured_kwargs.update(kwargs)
        return {"report": {"status": "completed_no_promotion"}}

    monkeypatch.setattr(pa, "run_policy_autoresearch", fake_run_policy_autoresearch)
    exit_code = pa.main(
        [
            "--capture-root",
            str(tmp_path / "capture"),
            "--job-dir",
            str(tmp_path / "job"),
            "--policy-recipe",
            str(tmp_path / "recipe.json"),
            "--scenario-eval-matrix",
            str(tmp_path / "matrix.json"),
            "--reviewed-examples",
            str(tmp_path / "reviewed.json"),
            "--output-dir",
            str(tmp_path / "out"),
            "--max-iterations",
            "2",
            "--agent-count",
            "3",
            "--target-success-rate",
            "0.8",
            "--heldout-ratio",
            "0.5",
            "--token-budget",
            "123",
            "--compute-seconds-budget",
            "4.5",
            "--wall-time-budget-seconds",
            "6.5",
            "--max-candidate-evaluations",
            "7",
            "--parallel-branch-limit",
            "2",
            "--evaluator-command",
            "python evaluator.py",
            "--evaluator-command-by-engine",
            "mujoco=python mujoco_eval.py",
            "--evaluator-timeout-seconds",
            "9",
            "--evaluator-attempt-trace",
            str(tmp_path / "attempts.jsonl"),
            "--simulator-engine",
            "isaac_sim",
            "--evaluation-substrate",
            "fixture_wam",
        ]
    )
    assert exit_code == 0
    assert captured_kwargs["max_iterations"] == 2
    assert captured_kwargs["agent_count"] == 3
    assert captured_kwargs["target_success_rate"] == 0.8
    assert captured_kwargs["heldout_ratio"] == 0.5
    assert captured_kwargs["token_budget"] == 123
    assert captured_kwargs["compute_seconds_budget"] == 4.5
    assert captured_kwargs["wall_time_budget_seconds"] == 6.5
    assert captured_kwargs["max_candidate_evaluations"] == 7
    assert captured_kwargs["parallel_branch_limit"] == 2
    assert captured_kwargs["evaluator_command"] == "python evaluator.py"
    assert captured_kwargs["evaluator_commands_by_engine"] == {
        "mujoco": "python mujoco_eval.py"
    }
    assert captured_kwargs["evaluator_timeout_seconds"] == 9
    assert captured_kwargs["simulator_engines"] == ("isaac_sim",)
    assert captured_kwargs["evaluation_substrates"] == ("fixture_wam",)
    assert json.loads(capsys.readouterr().out)["status"] == "completed_no_promotion"

    monkeypatch.setattr(
        pa,
        "run_policy_autoresearch",
        lambda **kwargs: {"report": {"status": "failed"}},
    )
    assert (
        pa.main(
            [
                "--capture-root",
                "capture",
                "--job-dir",
                "job",
                "--policy-recipe",
                "recipe.json",
            ]
        )
        == 1
    )

    with pytest.raises(SystemExit):
        pa.main(
            [
                "--capture-root",
                "capture",
                "--job-dir",
                "job",
                "--policy-recipe",
                "recipe.json",
                "--evaluator-command-by-engine",
                "bad",
            ]
        )


def test_policy_autoresearch_module_guard_runs_main(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    capture_root, job_dir = _job_paths(tmp_path)
    _write_matrix(
        job_dir,
        [
            {"scenario_eval_run_id": "train", "split": "train"},
            {"scenario_eval_run_id": "heldout", "split": "heldout"},
        ],
    )
    recipe_path = _seed_recipe(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "policy_autoresearch.py",
            "--capture-root",
            str(capture_root),
            "--job-dir",
            str(job_dir),
            "--policy-recipe",
            str(recipe_path),
            "--max-iterations",
            "0",
            "--target-success-rate",
            "0.0",
        ],
    )

    with pytest.raises(SystemExit) as exc:
        run_module_as_main("blueprint_pipeline.policy_autoresearch")

    assert exc.value.code == 0
    assert json.loads(capsys.readouterr().out)["status"] == "promoted"
