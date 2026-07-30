from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

import pytest

from blueprint_pipeline.task_evaluation_supervisor.candidate_policy import (
    CandidatePolicyError,
)
from blueprint_pipeline.task_evaluation_supervisor.pigey_candidate_runtime import (
    PIGEY_LICENSE_ATTESTATION_SCHEMA_VERSION,
    PigeyScenarioBinding,
    PigeySimCandidateRuntime,
)
from blueprint_pipeline.decision_evidence_contracts import canonical_digest


COMMIT = "b0cef8239dd2afb92827f05d76f16352635a36cb"


def _digest(path: Path) -> str:
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


def _license_attestation(*, issued_by_agent: bool = False) -> dict[str, Any]:
    value = {
        "schema_version": PIGEY_LICENSE_ATTESTATION_SCHEMA_VERSION,
        "status": "permission_granted",
        "source_repository": "https://github.com/lianegalanti/Pigey",
        "source_commit_sha": COMMIT,
        "reviewer_id": "fixture-independent-license-reviewer",
        "issued_by_agent": issued_by_agent,
        "commercial_use_authorized": True,
        "code_execution_authorized": True,
        "proof_effect": "none",
    }
    value["license_attestation_digest"] = canonical_digest(
        value,
        digest_field="license_attestation_digest",
    )
    return value


def _runtime(
    tmp_path: Path,
    command_runner,
    *,
    environment=None,
    max_llm_steps: int = 35,
    license_attestation=None,
):
    checkout = tmp_path / "pigey"
    script = checkout / "sim" / "agent_sim.py"
    script.parent.mkdir(parents=True)
    script.write_text("# frozen Pigey entrypoint\n", encoding="utf-8")
    return PigeySimCandidateRuntime(
        candidate_id="pigey-verify-recover",
        candidate_policy_manifest_digest="sha256:" + "a" * 64,
        checkout_root=checkout,
        expected_commit_sha=COMMIT,
        expected_agent_sim_digest=_digest(script),
        runtime_environment_digest="sha256:" + "e" * 64,
        terminal_signal_policy="shared_libero_task_done",
        python_executable=Path("/usr/bin/python3"),
        mode="harness",
        model_id="gpt-5-mini",
        policy_host="127.0.0.1",
        policy_port=8000,
        scenario_bindings=(
            PigeyScenarioBinding("occluded-target", "libero_goal_task", 0, 2),
            PigeyScenarioBinding("occupied-destination", "libero_goal_task", 1, 3),
        ),
        observation_schema_ref="pigey_libero_shared_task_done_observation.v1",
        action_schema_ref="pigey_libero_tool_action.v1",
        max_steps_per_rollout=500,
        max_llm_steps=max_llm_steps,
        replan_steps=5,
        timeout_seconds_per_scenario=600,
        max_cost_usd=10.0,
        input_cost_per_million_tokens_usd=1.0,
        output_cost_per_million_tokens_usd=2.0,
        environment=environment
        or {
            "PATH": "/usr/bin",
            "OPENAI_API_KEY": "secret-value-123",
            "OPENAI_PROJECT": "proj_pigey_eval",
        },
        paid_resource_admission_grant=None,
        openai_project_id="proj_pigey_eval",
        openai_api_key_id="key_pigey_eval",
        openai_api_key_scope_attestation_digest="sha256:" + "f" * 64,
        license_attestation=license_attestation or _license_attestation(),
        command_runner=command_runner,
    )


def _spec() -> dict[str, Any]:
    return {
        "policy_adapter": {
            "policy_id": "pigey-verify-recover",
            "observation_schema_ref": "pigey_libero_shared_task_done_observation.v1",
            "action_schema_ref": "pigey_libero_tool_action.v1",
        },
        "task_scenario_pack": {
            "scenario_ids": ["occupied-destination", "occluded-target"],
            "hidden_labels_included": False,
        },
    }


def test_pigey_runtime_invokes_exact_checkout_and_normalizes_trials_without_verdict(
    tmp_path: Path,
) -> None:
    calls: list[tuple[list[str], dict[str, Any]]] = []

    def command_runner(command: list[str], **kwargs: Any):
        calls.append((command, kwargs))
        if command[0] == "git":
            stdout = COMMIT + "\n" if "rev-parse" in command else ""
            return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")
        output_root = Path(command[command.index("--out-dir") + 1])
        scenario_id = output_root.name
        trial_dir = output_root / "gpt5" / scenario_id / "frozen-time"
        trial_dir.mkdir(parents=True)
        (trial_dir / "trial.json").write_text(
            json.dumps(
                {
                    "mode": "harness",
                    "model": "gpt-5-mini",
                    "suite": "libero_goal_task",
                    "task_id": 0 if scenario_id == "occluded-target" else 1,
                    "episode": 2 if scenario_id == "occluded-target" else 3,
                    "success": scenario_id == "occupied-destination",
                    "transcript": [
                        {
                            "role": "tool_call",
                            "tool": "Perceive",
                            "content": "inspect the public observation",
                        }
                    ],
                    "usage": {"input_tokens": 1_000_000, "output_tokens": 500_000},
                    "env_steps": 10,
                    "llm_steps": 2,
                    "duration_s": 3.5,
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, stdout="ignored", stderr="ignored")

    runtime = _runtime(tmp_path, command_runner)
    assert runtime.provider_execution_planned is True
    assert runtime.cost_accounting_authoritative is False
    output = tmp_path / "candidate-output"
    result = runtime.execute(evaluation_run_spec=_spec(), output_dir=output)

    assert result["status"] == "completed"
    assert result["provider_execution_started"] is True
    assert result["cost_usd"] == pytest.approx(4.0)
    assert result["duration_seconds"] == pytest.approx(7.0)
    trace = json.loads((output / result["trace_artifact_path"]).read_text(encoding="utf-8"))
    assert trace["source_commit_sha"] == COMMIT
    assert trace["runtime_configuration_digest"] == runtime.runtime_configuration_digest
    assert trace["candidate_reported_success_accepted_as_verdict"] is False
    assert trace["hidden_labels_received"] is False
    assert [row["scenario_id"] for row in trace["scenario_trials"]] == [
        "occluded-target",
        "occupied-destination",
    ]
    assert all(row["candidate_reported_success_value_excluded"] for row in trace["scenario_trials"])
    assert '"success": true' not in json.dumps(trace, sort_keys=True).lower()
    pigey_calls = [row for row in calls if row[0][0] != "git"]
    assert len(pigey_calls) == 2
    assert all("--no-video" in command for command, _kwargs in pigey_calls)
    assert all("shell" not in kwargs for _command, kwargs in pigey_calls)
    assert all(
        set(kwargs["env"])
        == {"PATH", "OPENAI_API_KEY", "OPENAI_PROJECT", "PYTHONDONTWRITEBYTECODE"}
        for _command, kwargs in pigey_calls
    )

    drifted = _runtime(
        tmp_path / "drifted-config",
        command_runner,
        max_llm_steps=36,
    )
    assert drifted.runtime_configuration_digest != runtime.runtime_configuration_digest


def test_pigey_runtime_rejects_scenario_or_checkout_drift(tmp_path: Path) -> None:
    calls: list[list[str]] = []

    def wrong_commit(command: list[str], **_kwargs: Any):
        calls.append(command)
        stdout = "f" * 40 + "\n" if "rev-parse" in command else ""
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

    runtime = _runtime(tmp_path, wrong_commit)
    with pytest.raises(CandidatePolicyError, match="pigey_checkout_commit_mismatch"):
        runtime.execute(evaluation_run_spec=_spec(), output_dir=tmp_path / "out")
    assert len(calls) == 1

    def right_commit(command: list[str], **_kwargs: Any):
        stdout = COMMIT + "\n" if "rev-parse" in command else ""
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

    runtime = _runtime(tmp_path / "scenario", right_commit)
    spec = _spec()
    spec["task_scenario_pack"]["scenario_ids"] = ["occluded-target"]
    with pytest.raises(CandidatePolicyError, match="pigey_scenario_set_mismatch"):
        runtime.execute(evaluation_run_spec=spec, output_dir=tmp_path / "scenario-out")


def test_pigey_runtime_fails_closed_on_timeout_and_unallowlisted_environment(
    tmp_path: Path,
) -> None:
    with pytest.raises(CandidatePolicyError, match="pigey_runtime_environment_not_allowlisted"):
        _runtime(
            tmp_path / "bad-env",
            lambda *_args, **_kwargs: None,
            environment={"UNREGISTERED_SECRET": "value"},
        )

    with pytest.raises(CandidatePolicyError, match="pigey_openai_cost_scope_not_bound"):
        _runtime(
            tmp_path / "wrong-project",
            lambda *_args, **_kwargs: None,
            environment={
                "PATH": "/usr/bin",
                "OPENAI_API_KEY": "secret-value-123",
                "OPENAI_PROJECT": "proj_wrong",
            },
        )

    with pytest.raises(
        CandidatePolicyError,
        match="pigey_license_or_permission_attestation_invalid",
    ):
        _runtime(
            tmp_path / "agent-issued-license",
            lambda *_args, **_kwargs: None,
            license_attestation=_license_attestation(issued_by_agent=True),
        )

    def timeout(command: list[str], **_kwargs: Any):
        if command[0] == "git":
            stdout = COMMIT + "\n" if "rev-parse" in command else ""
            return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")
        raise subprocess.TimeoutExpired(command, timeout=600)

    runtime = _runtime(tmp_path / "timeout", timeout)
    result = runtime.execute(
        evaluation_run_spec=_spec(),
        output_dir=tmp_path / "timeout-output",
    )
    assert result["status"] == "failed"
    assert result["blockers"] == ["pigey_scenario_timeout"]
    assert result["provider_execution_started"] is True


def test_pigey_runtime_rejects_dirty_checkout_and_secret_bearing_trial(
    tmp_path: Path,
) -> None:
    pigey_calls: list[list[str]] = []

    def dirty_checkout(command: list[str], **_kwargs: Any):
        if "rev-parse" in command:
            return subprocess.CompletedProcess(command, 0, stdout=COMMIT + "\n", stderr="")
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=" M sim/agent_sim.py\n",
            stderr="",
        )

    dirty = _runtime(tmp_path / "dirty", dirty_checkout)
    with pytest.raises(CandidatePolicyError, match="pigey_checkout_not_clean"):
        dirty.execute(evaluation_run_spec=_spec(), output_dir=tmp_path / "dirty-output")

    def secret_trial(command: list[str], **_kwargs: Any):
        if command[0] == "git":
            stdout = COMMIT + "\n" if "rev-parse" in command else ""
            return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")
        pigey_calls.append(command)
        output_root = Path(command[command.index("--out-dir") + 1])
        scenario_id = output_root.name
        trial_dir = output_root / "trial"
        trial_dir.mkdir(parents=True)
        (trial_dir / "trial.json").write_text(
            json.dumps(
                {
                    "mode": "harness",
                    "model": "gpt-5-mini",
                    "suite": "libero_goal_task",
                    "task_id": 0 if scenario_id == "occluded-target" else 1,
                    "episode": 2 if scenario_id == "occluded-target" else 3,
                    "success": False,
                    "transcript": [
                        {
                            "role": "error",
                            "content": "leaked secret OPENAI_API_KEY=secret-value-123",
                        }
                    ],
                    "usage": {"input_tokens": 0, "output_tokens": 0},
                    "env_steps": 0,
                    "llm_steps": 0,
                    "duration_s": 0,
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    leaking = _runtime(tmp_path / "leaking", secret_trial)
    with pytest.raises(CandidatePolicyError, match="pigey_trial_contains_secret"):
        leaking.execute(
            evaluation_run_spec=_spec(),
            output_dir=tmp_path / "leaking-output",
        )
    assert len(pigey_calls) == 1
