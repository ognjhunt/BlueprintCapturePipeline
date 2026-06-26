from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path

import pytest

from blueprint_pipeline import policy_autoresearch_owner_gpu_evaluator as owner_eval


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n\n", encoding="utf-8")


def test_owner_gpu_evaluator_helpers_cover_input_shapes(monkeypatch, tmp_path: Path) -> None:
    assert owner_eval._string(None) == ""
    assert owner_eval._mapping({"a": 1}) == {"a": 1}
    assert owner_eval._mapping([]) == {}
    assert owner_eval._int(True, 9) == 9
    assert owner_eval._int(5) == 5
    assert owner_eval._int("6.7") == 6
    assert owner_eval._int("bad", 3) == 3
    assert owner_eval._safe_id(" Policy:One! ") == "policy_one"
    assert owner_eval._safe_id("!!!", fallback="fallback") == "fallback"
    assert owner_eval._string_list("one") == ["one"]
    assert owner_eval._string_list(["one", "", b"two"]) == ["one", "b'two'"]
    assert owner_eval._string_list(None) == []

    jsonl_path = tmp_path / "attempts.jsonl"
    _write_jsonl(jsonl_path, [{"a": 1}, ["bad"], {"b": 2}])
    assert owner_eval._read_jsonl(jsonl_path) == [{"a": 1}, {"b": 2}]
    assert owner_eval._load_attempts(tmp_path / "missing.json") == []
    assert owner_eval._load_attempts(jsonl_path) == [{"a": 1}, {"b": 2}]

    for payload in [
        {"attempts": [{"id": 1}, "bad"]},
        {"results": [{"id": 2}]},
        {"episodes": [{"id": 3}]},
        [{"id": 4}, "bad"],
    ]:
        assert owner_eval._attempts_from_payload(payload)[0]["id"] in {1, 2, 3, 4}
    assert owner_eval._attempts_from_payload({"attempts": {}}) == []
    assert owner_eval._attempts_from_payload("bad") == []

    _write_json(tmp_path / "attempts.json", {"results": [{"id": 5}]})
    assert owner_eval._load_attempts(tmp_path / "attempts.json") == [{"id": 5}]
    _write_json(tmp_path / "matrix.json", {"runs": [{"scenario_eval_run_id": "run-1"}, "bad"]})
    assert owner_eval._runs_from_matrix(tmp_path / "matrix.json") == [{"scenario_eval_run_id": "run-1"}]
    _write_json(tmp_path / "empty_matrix.json", {"runs": {}})
    assert owner_eval._runs_from_matrix(tmp_path / "empty_matrix.json") == []

    assert owner_eval._task_success({"taskOutcome": {"success": True}}) is True
    assert owner_eval._failure_modes({"failureModeIds": ["direct"]}) == ["direct"]
    assert owner_eval._failure_modes({"taskOutcome": {"failureModeIds": ["nested"]}}) == ["nested"]
    assert owner_eval._contact_event_count({"metrics": {"near_miss_event_count": 2}}, []) == 2
    assert owner_eval._contact_event_count({}, ["failure_clearance_near_miss", "other"]) == 1
    assert owner_eval._safety_event_count({"metrics": {"unsafe_proximity_event_count": 3}}, []) == 3
    assert owner_eval._safety_event_count({}, ["failure_dynamic_obstacle", "other"]) == 1

    output_path = tmp_path / "out" / "result.json"
    assert owner_eval._attempt_trace_path(output_path) == output_path.parent / "owner_gpu_policy_attempt_trace.json"
    monkeypatch.setenv("BLUEPRINT_POLICY_AUTORESEARCH_OWNER_ATTEMPT_TRACE", str(tmp_path / "trace.json"))
    assert owner_eval._attempt_trace_path(output_path) == (tmp_path / "trace.json").resolve()
    monkeypatch.setenv("OWNER_GPU_PATH", str(tmp_path / "value"))
    assert owner_eval._env_path("OWNER_GPU_PATH") == (tmp_path / "value").resolve()
    monkeypatch.delenv("OWNER_GPU_PATH", raising=False)
    assert owner_eval._env_path("OWNER_GPU_PATH") is None


def test_owner_gpu_normalizes_attempts_for_success_failure_and_missing_trace(tmp_path: Path) -> None:
    runs = [
        {"scenario_eval_run_id": "run-1", "task_id": "task-1", "scenario_id": "scenario-1"},
        {"scenario_eval_run_id": "run-2", "scenario_variation_instance_id": "var-2", "variation_name": "turn"},
        {"scenario_eval_run_id": "run-3"},
    ]
    observed = [
        {
            "scenario_eval_run_id": "run-1",
            "attempt_id": "attempt-1",
            "taskSuccess": True,
            "metrics": {"contact_event_count": 1},
            "artifactPaths": {"video": "pov.mp4"},
        },
        {"scenario_eval_run_id": "run-2", "task_outcome": {"task_success": False}},
    ]

    attempts = owner_eval._normalize_owner_attempts(
        runs=runs,
        observed_attempts=observed,
        recipe={"candidate_id": "Policy One", "policyKind": "route"},
        simulator_engine="isaac_sim",
        attempt_trace_path=tmp_path / "trace.json",
        proof_result={"owner_gpu_simulator_execution_proven": True},
        validation={
            "simulator_backend": "isaac_lab",
            "isaac_sim_execution_proven": True,
            "owner_gpu_default_policy_execution_proven": True,
            "owner_gpu_sim_robot_pov_evidence_proven": True,
        },
        generated_at="2026-06-20T00:00:00Z",
    )

    assert attempts[0]["status"] == "completed"
    assert attempts[0]["success"] is True
    assert attempts[0]["artifact_paths"] == {"video": "pov.mp4"}
    assert attempts[1]["status"] == "failed_owner_gpu_policy_attempt"
    assert attempts[1]["failure_mode_ids"] == ["policy_task_not_successful"]
    assert attempts[2]["attempt_id"] == "policy_one_isaac_sim_0003"
    assert attempts[2]["failure_mode_ids"] == ["owner_gpu_policy_attempt_trace_missing"]
    assert attempts[2]["metrics"]["policy_attempt_trace_present"] is False

    unproven = owner_eval._normalize_owner_attempts(
        runs=[{"scenario_eval_run_id": "run-1"}],
        observed_attempts=[{"scenario_eval_run_id": "run-1", "failure_mode_ids": ["failure_contact_collision"]}],
        recipe={"policy_id": "p"},
        simulator_engine="mujoco",
        attempt_trace_path=tmp_path / "trace.json",
        proof_result={"owner_gpu_simulator_execution_proven": False},
        validation={},
        generated_at="2026-06-20T00:00:00Z",
    )
    assert "owner_gpu_simulator_execution_not_proven" in unproven[0]["failure_mode_ids"]
    assert unproven[0]["metrics"]["contact_event_count"] == 1


def test_run_owner_gpu_policy_evaluator_blocked_and_completed(monkeypatch, tmp_path: Path) -> None:
    recipe_path = tmp_path / "recipe.json"
    matrix_path = tmp_path / "matrix.json"
    capture_root = tmp_path / "capture"
    output_path = tmp_path / "out" / "owner_result.json"
    _write_json(recipe_path, {"candidate_id": "policy-1", "policy_kind": "route"})
    _write_json(matrix_path, {"runs": [{"scenario_eval_run_id": "run-1", "task_id": "task-1"}]})

    blocked = owner_eval.run_owner_gpu_policy_evaluator(
        recipe_path=recipe_path,
        matrix_path=matrix_path,
        output_path=output_path,
        capture_root=capture_root,
        generated_at="2026-06-20T00:00:00Z",
    )
    assert blocked["status"] == "blocked_owner_gpu_simulator_execution_not_proven"
    assert blocked["policy_attempt_trace_present"] is False
    assert blocked["attempts"][0]["failure_mode_ids"] == [
        "isaac_simulator_execution_not_proven",
        "owner_gpu_policy_attempt_trace_missing",
        "owner_gpu_simulator_execution_not_proven",
    ]
    assert output_path.is_file()

    def fake_owner_gpu_proof(**kwargs):
        trace_path = Path(kwargs["extra_env"]["BLUEPRINT_POLICY_AUTORESEARCH_OWNER_ATTEMPT_TRACE"])
        _write_json(trace_path, {"attempts": [{"scenario_eval_run_id": "run-1", "task_success": True}]})
        validation_path = tmp_path / "validation.json"
        _write_json(
            validation_path,
            {
                "status": "completed",
                "simulator_backend": "owner_backend",
                "owner_gpu_simulator_execution_proven": True,
                "isaac_sim_execution_proven": True,
            },
        )
        assert kwargs["command"] == "run-owner-policy"
        assert kwargs["timeout_seconds"] == 5
        return {
            "owner_gpu_simulator_execution_proven": True,
            "validation_manifest_path": str(validation_path),
        }

    monkeypatch.setattr(owner_eval, "run_owner_gpu_proof", fake_owner_gpu_proof)
    monkeypatch.setenv("BLUEPRINT_POLICY_AUTORESEARCH_PHASE", "heldout")
    monkeypatch.setenv("BLUEPRINT_POLICY_AUTORESEARCH_VERIFIER_SHA256", "abc123")
    completed = owner_eval.run_owner_gpu_policy_evaluator(
        recipe_path=recipe_path,
        matrix_path=matrix_path,
        output_path=tmp_path / "out" / "completed.json",
        capture_root=capture_root,
        owner_command=" run-owner-policy ",
        timeout_seconds=5,
        generated_at="2026-06-20T00:00:00Z",
    )

    assert completed["status"] == "completed"
    assert completed["phase"] == "heldout"
    assert completed["frozen_verifier_sha256"] == "abc123"
    assert completed["simulator_backend"] == "owner_backend"
    assert completed["isaac_sim_execution_proven"] is True
    assert completed["attempts"][0]["success"] is True

    def fake_generic_owner_gpu_proof(**kwargs):
        trace_path = Path(kwargs["extra_env"]["BLUEPRINT_POLICY_AUTORESEARCH_OWNER_ATTEMPT_TRACE"])
        _write_json(trace_path, {"attempts": [{"scenario_eval_run_id": "run-1", "task_success": True}]})
        validation_path = tmp_path / "generic_validation.json"
        _write_json(
            validation_path,
            {
                "status": "completed",
                "simulator_backend": "owner_backend",
                "owner_gpu_simulator_execution_proven": True,
                "isaac_sim_execution_proven": False,
            },
        )
        return {
            "owner_gpu_simulator_execution_proven": True,
            "validation_manifest_path": str(validation_path),
        }

    monkeypatch.setattr(owner_eval, "run_owner_gpu_proof", fake_generic_owner_gpu_proof)
    generic_only = owner_eval.run_owner_gpu_policy_evaluator(
        recipe_path=recipe_path,
        matrix_path=matrix_path,
        output_path=tmp_path / "out" / "generic-only.json",
        capture_root=capture_root,
        owner_command="run-owner-policy",
        simulator_engine="isaac_sim",
        timeout_seconds=5,
        generated_at="2026-06-20T00:00:00Z",
    )

    assert generic_only["status"] == "blocked_owner_gpu_simulator_execution_not_proven"
    assert generic_only["generic_owner_gpu_simulator_execution_proven"] is True
    assert generic_only["owner_gpu_simulator_execution_proven"] is False
    assert generic_only["attempts"][0]["task_success"] is False
    assert "isaac_simulator_execution_not_proven" in generic_only["attempts"][0]["failure_mode_ids"]


def test_owner_gpu_policy_evaluator_main_and_module_guard(monkeypatch, tmp_path: Path, capsys) -> None:
    for name in [
        "BLUEPRINT_POLICY_AUTORESEARCH_RECIPE",
        "BLUEPRINT_POLICY_AUTORESEARCH_MATRIX",
        "BLUEPRINT_POLICY_AUTORESEARCH_OUTPUT",
        "BLUEPRINT_POLICY_AUTORESEARCH_CAPTURE_ROOT",
    ]:
        monkeypatch.delenv(name, raising=False)
    assert owner_eval.main([]) == 2
    assert "blocked_missing_env" in capsys.readouterr().out

    recipe_path = tmp_path / "recipe.json"
    matrix_path = tmp_path / "matrix.json"
    output_path = tmp_path / "out.json"
    capture_root = tmp_path / "capture"
    _write_json(recipe_path, {"candidate_id": "policy-main"})
    _write_json(matrix_path, {"runs": [{"scenario_eval_run_id": "run-main"}]})
    for key, value in {
        "BLUEPRINT_POLICY_AUTORESEARCH_RECIPE": recipe_path,
        "BLUEPRINT_POLICY_AUTORESEARCH_MATRIX": matrix_path,
        "BLUEPRINT_POLICY_AUTORESEARCH_OUTPUT": output_path,
        "BLUEPRINT_POLICY_AUTORESEARCH_CAPTURE_ROOT": capture_root,
        "BLUEPRINT_POLICY_AUTORESEARCH_OWNER_COMMAND": "owner-cmd",
        "BLUEPRINT_POLICY_AUTORESEARCH_SIMULATOR_ENGINE": "mujoco",
        "BLUEPRINT_POLICY_AUTORESEARCH_OWNER_SYSTEM_ID": "owner-system",
        "BLUEPRINT_POLICY_AUTORESEARCH_OWNER_SIMULATOR_VERSION": "sim-v",
        "BLUEPRINT_POLICY_AUTORESEARCH_OWNER_GPU_MODEL": "gpu",
        "BLUEPRINT_POLICY_AUTORESEARCH_OPERATOR_ID": "operator",
        "BLUEPRINT_POLICY_AUTORESEARCH_OPERATOR_ATTESTATION": "attested",
        "BLUEPRINT_POLICY_AUTORESEARCH_OWNER_TIMEOUT_SECONDS": "bad",
    }.items():
        monkeypatch.setenv(key, str(value))

    calls: list[dict[str, object]] = []

    def fake_run(**kwargs):
        calls.append(kwargs)
        return {"status": "completed"}

    monkeypatch.setattr(owner_eval, "run_owner_gpu_policy_evaluator", fake_run)
    assert owner_eval.main([]) == 0
    assert calls[-1]["owner_command"] == "owner-cmd"
    assert calls[-1]["simulator_engine"] == "mujoco"
    assert calls[-1]["owner_system_id"] == "owner-system"
    assert calls[-1]["simulator_version"] == "sim-v"
    assert calls[-1]["gpu_model"] == "gpu"
    assert calls[-1]["operator_id"] == "operator"
    assert calls[-1]["operator_attestation"] == "attested"
    assert calls[-1]["timeout_seconds"] == 1800

    monkeypatch.setattr(sys, "argv", ["owner-gpu-eval"])
    monkeypatch.delenv("BLUEPRINT_POLICY_AUTORESEARCH_OWNER_COMMAND", raising=False)
    with pytest.warns(RuntimeWarning, match="found in sys.modules"):
        with pytest.raises(SystemExit) as excinfo:
            runpy.run_module("blueprint_pipeline.policy_autoresearch_owner_gpu_evaluator", run_name="__main__")
    assert excinfo.value.code == 0
