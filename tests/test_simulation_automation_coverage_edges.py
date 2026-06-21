from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import pytest

from blueprint_pipeline import simulation_automation as sa


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _build_capture_root(tmp_path: Path) -> Path:
    capture_root = tmp_path / "bucket" / "scenes" / "scene-1" / "captures" / "capture-1"
    _write_json(capture_root / "capture_descriptor.json", {"scene_id": "scene-1"})
    _write_json(capture_root / "raw" / "manifest.json", {"capture_id": "capture-1"})
    return capture_root


def _context(tmp_path: Path) -> Any:
    return sa.resolve_local_capture_context(_build_capture_root(tmp_path))


def test_sdk_agent_adapters_report_gate_and_execution_failures(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv(sa.LIVE_CODEX_SDK_ENV, raising=False)
    monkeypatch.delenv(sa.LIVE_AGENTS_SDK_ENV, raising=False)
    monkeypatch.delenv(sa.CODEX_CLI_HOST_OAUTH_ENV, raising=False)
    monkeypatch.setattr(sa, "resolve_codex_cli_path", lambda: None)
    plan_context = {"repo_root": str(tmp_path)}

    codex_blocked = sa.CodexSdkSimulationAutomationAgentAdapter(
        codex_sdk_available=False,
        openai_api_key="",
        codex_cli_path="",
        live_env_allowed=False,
        allow_live_operator=False,
    ).build_ledger(plan_context=plan_context)
    assert {
        "missing_codex_sdk",
        "missing_openai_api_key",
        "missing_codex_cli",
        "missing_cli_allow_live_codex_sdk_operator",
        f"missing_env_{sa.LIVE_CODEX_SDK_ENV}",
    }.issubset(set(codex_blocked["operator_ledger"]["blockers"]))

    codex_cli_oauth_blocked = sa.CodexSdkSimulationAutomationAgentAdapter(
        codex_sdk_available=False,
        openai_api_key="",
        codex_cli_path="/bin/codex",
        live_env_allowed=True,
        allow_live_operator=True,
    ).build_ledger(plan_context=plan_context)
    assert f"missing_env_{sa.CODEX_CLI_HOST_OAUTH_ENV}" in codex_cli_oauth_blocked[
        "operator_ledger"
    ]["blockers"]

    def raise_runtime(prompt: str, context: Mapping[str, Any]) -> Mapping[str, Any]:
        del prompt, context
        raise RuntimeError("operator runtime unavailable")

    def raise_value(prompt: str, context: Mapping[str, Any]) -> Mapping[str, Any]:
        del prompt, context
        raise ValueError("bad operator output")

    codex_runtime = sa.CodexSdkSimulationAutomationAgentAdapter(
        codex_sdk_available=True,
        openai_api_key="sk-test",
        live_env_allowed=True,
        allow_live_operator=True,
        executor=raise_runtime,
    ).build_ledger(plan_context=plan_context)
    assert codex_runtime["status"] == "operator_failed"
    assert "operator runtime unavailable" in codex_runtime["operator_ledger"]["blockers"]

    codex_exception = sa.CodexSdkSimulationAutomationAgentAdapter(
        codex_sdk_available=True,
        openai_api_key="sk-test",
        live_env_allowed=True,
        allow_live_operator=True,
        executor=raise_value,
    ).build_ledger(plan_context=plan_context)
    assert "codex_sdk_operator_execution_failed:ValueError" in codex_exception[
        "operator_ledger"
    ]["blockers"]

    agents_blocked = sa.AgentsSdkCodexMCPAdapter(
        agents_sdk_available=False,
        openai_api_key="",
        live_env_allowed=False,
        allow_live_operator=False,
    ).build_ledger(plan_context=plan_context)
    assert {
        "missing_openai_agents_sdk",
        "missing_openai_api_key",
        "missing_cli_allow_live_agents_sdk_operator",
        f"missing_env_{sa.LIVE_AGENTS_SDK_ENV}",
    }.issubset(set(agents_blocked["operator_ledger"]["blockers"]))

    agents_runtime = sa.AgentsSdkCodexMCPAdapter(
        agents_sdk_available=True,
        openai_api_key="sk-test",
        live_env_allowed=True,
        allow_live_operator=True,
        executor=raise_runtime,
    ).build_ledger(plan_context=plan_context)
    assert agents_runtime["status"] == "operator_failed"
    assert "operator runtime unavailable" in agents_runtime["operator_ledger"]["blockers"]

    agents_exception = sa.AgentsSdkCodexMCPAdapter(
        agents_sdk_available=True,
        openai_api_key="sk-test",
        live_env_allowed=True,
        allow_live_operator=True,
        executor=raise_value,
    ).build_ledger(plan_context=plan_context)
    assert "agents_sdk_operator_execution_failed:ValueError" in agents_exception[
        "operator_ledger"
    ]["blockers"]


def test_owner_gpu_proof_helpers_and_blocked_validation_edges(tmp_path: Path) -> None:
    proof_dir = tmp_path / "proof"
    proof_dir.mkdir()
    invalid_json = proof_dir / "invalid.json"
    invalid_json.write_text("{", encoding="utf-8")
    non_object_json = proof_dir / "array.json"
    non_object_json.write_text("[]", encoding="utf-8")

    assert sa._string_list("one") == ["one"]
    assert sa._string_list(42) == ["42"]
    assert "T" in sa._timestamp({})
    assert sa._resolve_owner_proof_artifact("", proof_dir=proof_dir) is None
    assert sa._resolve_owner_proof_artifact("https://example.com/proof.json", proof_dir=proof_dir) is None
    assert sa._read_owner_proof_json_artifact("https://example.com/proof.json", proof_dir=proof_dir) == (
        {},
        "owner_proof_artifact_not_local",
        False,
    )
    assert sa._read_owner_proof_json_artifact("missing.json", proof_dir=proof_dir) == (
        {},
        "owner_proof_artifact_missing",
        False,
    )
    assert sa._read_owner_proof_json_artifact(invalid_json, proof_dir=proof_dir)[1:] == (
        "owner_proof_artifact_invalid_json",
        True,
    )
    assert sa._read_owner_proof_json_artifact(non_object_json, proof_dir=proof_dir)[1:] == (
        "owner_proof_artifact_non_object_json",
        True,
    )
    assert sa._owner_proof_file_exists("gs://remote/stdout.log", proof_dir=proof_dir) == (
        False,
        "owner_proof_artifact_not_local",
    )
    assert sa._owner_proof_file_exists("missing.log", proof_dir=proof_dir) == (
        False,
        "owner_proof_artifact_missing",
    )
    assert sa._attestation_ok("operator attests")
    assert not sa._attestation_ok([])
    assert sa._pass_fail_ok("passed")
    assert not sa._pass_fail_ok([])
    assert not sa._pass_fail_ok({"passed": False})
    assert sa._pass_fail_ok({"status": "completed"})
    assert sa._robot_asset_mapping("Unitree G1") == {"name": "Unitree G1"}
    assert sa._owner_robot_asset({}) == {}
    assert sa._robot_assets_match({"name": "G1"}, {"robot_name": "G1"})
    assert sa._sim_robot_pov_ok({"video_uri": "gs://owner/pov.mov"})
    assert sa._sim_robot_pov_ok({"frames": ["frame-1.png"]})
    assert not sa._owner_required_field_present(None)

    missing_output = proof_dir / "missing_manifest.json"
    missing = sa.validate_owner_gpu_system_proof(
        proof_path=proof_dir / "missing.json",
        output_path=missing_output,
    )
    assert missing["status"] == "missing"
    assert missing_output.is_file()

    capture_root = _build_capture_root(tmp_path)
    bad_proof = proof_dir / "bad_proof.json"
    _write_json(
        bad_proof,
        {
            "scene_id": "other-scene",
            "capture_id": "other-capture",
            "simulator_backend": "isaac_sim",
            "exit_code": "not-an-int",
            "stdout_uri_or_path": "missing-stdout.log",
            "stderr_uri_or_path": "https://example.com/stderr.log",
            "scene_load_trace_uri_or_path": "missing-scene.json",
            "action_or_policy_trace_uri_or_path": "missing-action.json",
            "default_smoke_policy_uri_or_path": "missing-default.json",
            "policy_execution_trace_uri_or_path": "missing-policy.json",
            "sim_robot_pov_evidence_uri_or_path": "missing-pov.json",
            "artifact_manifest_uri_or_path": "missing-artifacts.json",
            "operator_attestation": {},
            "pass_fail_criteria": {"passed": False},
            "isaac_sim_execution_proven": True,
        },
    )
    blocked = sa.validate_owner_gpu_system_proof(proof_path=bad_proof, capture_root=capture_root)
    blockers = set(blocked["blockers"])
    assert {
        "owner_gpu_proof_missing_required_fields",
        "owner_proof_attempted_forbidden_isaac_sim_execution_proven",
        "owner_gpu_proof_scene_id_mismatch",
        "owner_gpu_proof_capture_id_mismatch",
        "owner_gpu_simulator_exit_code_nonzero",
        "stdout_owner_proof_artifact_missing",
        "stderr_owner_proof_artifact_not_local",
        "scene_load_trace_owner_proof_artifact_missing",
        "owner_gpu_scene_load_trace_not_proven",
        "owner_gpu_spawn_trace_not_proven",
        "owner_gpu_action_or_policy_trace_not_proven",
        "owner_gpu_default_smoke_policy_not_proven",
        "owner_gpu_default_policy_execution_trace_not_proven",
        "owner_gpu_sim_robot_pov_evidence_not_proven",
        "owner_gpu_artifact_manifest_not_proven",
        "owner_gpu_operator_attestation_missing_or_incomplete",
        "owner_gpu_pass_fail_criteria_not_passed",
        "owner_gpu_proof_missing_isaac_robot_asset",
        "owner_gpu_spawn_trace_missing_isaac_robot_asset",
    }.issubset(blockers)

    stdout = proof_dir / "stdout.log"
    stderr = proof_dir / "stderr.log"
    stdout.write_text("ok", encoding="utf-8")
    stderr.write_text("", encoding="utf-8")
    _write_json(proof_dir / "scene.json", {"status": "loaded", "robot_asset": {"name": "Proof Bot"}})
    _write_json(
        proof_dir / "spawn.json",
        {"status": "validated", "robot_asset": {"name": "Other Bot", "uri_or_path": "Other/g1.usd"}},
    )
    _write_json(proof_dir / "actions.json", {"actions": [{"name": "walk"}]})
    _write_json(proof_dir / "policy.json", {"policy_kind": "walk_to_target", "target": "target"})
    _write_json(proof_dir / "pov.json", {"video_uri": "owner-pov.mov"})
    _write_json(proof_dir / "artifacts.json", {"artifacts": [{"path": "stdout.log"}]})
    mismatch_proof = proof_dir / "mismatch_proof.json"
    _write_json(
        mismatch_proof,
        {
            "owner_system_id": "owner",
            "scene_id": "scene-1",
            "capture_id": "capture-1",
            "simulator_backend": "isaac_sim",
            "simulator_version": "2026.1",
            "gpu_model": "RTX",
            "robot_asset": {"name": "Proof Bot", "uri_or_path": "Proof/g1.usd"},
            "command": "sim",
            "started_at": "2026-06-01T00:00:00Z",
            "completed_at": "2026-06-01T00:01:00Z",
            "exit_code": 0,
            "stdout_uri_or_path": str(stdout),
            "stderr_uri_or_path": str(stderr),
            "scene_load_trace_uri_or_path": "scene.json",
            "spawn_pose_validation_uri_or_path": "spawn.json",
            "action_or_policy_trace_uri_or_path": "actions.json",
            "default_smoke_policy_uri_or_path": "policy.json",
            "policy_execution_trace_uri_or_path": "actions.json",
            "sim_robot_pov_evidence_uri_or_path": "pov.json",
            "artifact_manifest_uri_or_path": "artifacts.json",
            "pass_fail_criteria": "accepted",
            "operator_attestation": "operator attests",
        },
    )
    mismatch = sa.validate_owner_gpu_system_proof(
        proof_path=mismatch_proof,
        capture_root=capture_root,
    )
    assert "owner_gpu_robot_asset_mismatch" in mismatch["blockers"]
    assert "owner_gpu_unitree_g1_asset_not_spawned" in mismatch["blockers"]


def test_conversion_arena_gpu_handoff_and_claim_boundary_edges(tmp_path: Path) -> None:
    context = _context(tmp_path)
    automation_dir = context.pipeline_root / "simulation_automation"
    automation_dir.mkdir(parents=True, exist_ok=True)

    assert sa._conversion_status(framework="isaac_sim", worldlabs={}, cpu_preflight={})[
        "blockers"
    ] == ["missing_visual_asset"]
    assert sa._conversion_status(
        framework="isaac_sim",
        worldlabs={"usd_available": True},
        cpu_preflight={"isaac_usd_collision_verified": True},
    )["status"] == "isaac_usd_import_candidate"
    assert sa._conversion_status(
        framework="isaac_sim",
        worldlabs={"ply_available": True},
        cpu_preflight={},
    )["status"] == "planned_asset_import_ready"
    assert sa._conversion_status(framework="isaac_lab_arena", worldlabs={}, cpu_preflight={})[
        "status"
    ] == "blocked"
    assert sa._conversion_status(
        framework="isaac_lab_arena",
        worldlabs={},
        cpu_preflight={"isaac_usd_import_candidate": True, "isaac_usd_collision_verified": True},
    )["status"] == "arena_environment_packet_ready_for_owner_review"
    assert sa._conversion_status(
        framework="isaac_lab_arena",
        worldlabs={"spz_available": True},
        cpu_preflight={},
    )["status"] == "planned_requires_conversion"
    assert "cpu_proxy_collision_estimated" in sa._conversion_status(
        framework="mujoco",
        worldlabs={},
        cpu_preflight={"cpu_proxy_collision_estimated": True},
    )["blockers"]
    assert "cpu_proxy_collision_estimated" in sa._conversion_status(
        framework="pybullet",
        worldlabs={},
        cpu_preflight={"cpu_proxy_collision_estimated": True},
    )["blockers"]

    class ListPayload(list):
        def get(self, key: str) -> None:
            del key
            return None

    assert sa._cards_from_payload(ListPayload([{"id": "card-1"}, "skip"])) == [{"id": "card-1"}]
    episode_spec = {
        "episodes": [
            "skip",
            {"episode_id": "ep-1", "task_id": "task-a", "scenario_id": "scenario-a"},
        ]
    }
    task_components = sa._arena_task_components(task_cards=[], episode_spec=episode_spec)
    scenario_components = sa._arena_scenario_components(
        scenario_cards=[],
        episode_spec=episode_spec,
        variation_instances={"instances": "not-a-list"},
    )
    assert task_components[0]["task_id"] == "task-a"
    assert scenario_components[0]["scenario_id"] == "scenario-a"
    assert sa._variation_instance_ids_by_scenario(
        {"instances": ["skip", {"scenario_id": "scenario-a"}, {"instance_id": "i-1"}]}
    ) == {}
    assert len(
        sa._arena_embodiment_components(
            {
                "episodes": [
                    "skip",
                    {},
                    {"robot_profile_id": "robot-a"},
                    {"robot_profile_id": "robot-a"},
                ]
            }
        )
    ) == 1
    bindings = sa._arena_episode_bindings(
        context=context,
        episode_spec={
            "episodes": [
                "skip",
                {
                    "episode_id": "ep-missing",
                    "task_id": "missing-task",
                    "scenario_id": "missing-scenario",
                    "robot_profile_id": "robot-a",
                },
            ]
        },
        task_components=[],
        scenario_components=[],
        variation_instances={"instances": []},
    )
    assert {
        "arena_task_component_missing",
        "arena_scenario_component_missing",
    }.issubset(set(bindings[0]["missing_proof_labels"]))

    packet = sa._build_arena_environment_packet(
        context=context,
        automation_dir=automation_dir,
        pipeline_dir=context.pipeline_root,
        conversion_plan={},
        generated_at="2026-06-01T00:00:00Z",
    )
    assert packet["status"] == "blocked_missing_episode_spec"
    assert "episode_spec_v1_missing_or_empty" in packet["blockers"]
    assert "scenario_variation_instances_missing" in packet["blockers"]

    proof_boundary = sa._proof_boundary(
        simulator_execution={},
        training={},
        owner_gpu_proof={"mujoco_g1_asset_execution_proven": True},
        local_mujoco_g1_smoke={},
        generated_at="2026-06-01T00:00:00Z",
    )
    assert proof_boundary["mujoco_g1_asset_execution_proven"] is True

    recommendations = sa._gpu_backend_recommendations(
        inventory={"assets": [{"asset_type": "usd"}, {"asset_type": "glb"}]},
        collider_proxy_plan={"real_collider_proven": True, "proxy_estimated": True},
        conversion_plan={"frameworks": {"isaac_sim": {}, "isaac_lab_arena": {}, "mujoco": {}, "pybullet": {}}},
    )
    assert {item["backend"] for item in recommendations} == {
        "isaac_sim",
        "isaac_lab_arena",
        "mujoco",
        "pybullet",
    }

    details: list[dict[str, Any]] = []
    seen: set[str] = set()
    sa._append_blocker_detail(
        details,
        seen,
        blocker_id="",
        source_artifact="artifact.json",
        severity="warning",
        required_input="input",
    )
    assert details == []
    handoff_details = sa._gpu_handoff_blocker_details(
        scene_preflight={
            "blockers": [
                "missing_scene_frame_estimate",
                "portable_collider_glb_missing",
                "simulator_execution_not_run",
                "custom_blocker",
            ]
        },
        spawn_validation={
            "status": "blocked",
            "blockers": [
                "scene_bounds_empty_or_inverted",
                "spawn_outside_scene_bounds",
                "spawn_inside_known_or_proxy_geometry",
            ],
        },
        cpu_preflight={"hard_preflight_blockers": ["missing_local_scene_asset"]},
        owner_gpu_proven=False,
    )
    handoff_ids = {item["blocker_id"] for item in handoff_details}
    assert {
        "owner_gpu_simulator_execution_not_run",
        "missing_scene_frame_estimate",
        "scene_bounds_empty_or_inverted",
        "spawn_outside_scene_bounds",
        "spawn_inside_known_or_proxy_geometry",
        "portable_collider_glb_missing",
        "custom_blocker",
    }.issubset(handoff_ids)
    assert "simulator_execution_not_run" not in handoff_ids

    _write_json(automation_dir / "scene_asset_dependency_audit.json", {"hard_missing_local_file_count": 1})
    _write_json(automation_dir / "spawn_pose_validation_manifest.json", {"status": "blocked"})
    _write_json(automation_dir / "cpu_preflight_manifest.json", {"ready_for_owner_gpu_preflight": False})
    handoff = sa._build_gpu_handoff_artifacts(
        context=context,
        automation_dir=automation_dir,
        plan={"scene_id": "scene-1", "capture_id": "capture-1"},
        conversion_plan={},
        generated_at="2026-06-01T00:00:00Z",
        simulator_timeout_seconds=30,
    )
    assert "missing_scene_asset_dependencies" in handoff["packet"]["blockers"]
    assert "spawn_validation_blocked" in handoff["packet"]["blockers"]


def test_simulator_training_and_cli_edges(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    context = _context(tmp_path)
    automation_dir = context.pipeline_root / "simulation_automation"
    automation_dir.mkdir(parents=True, exist_ok=True)
    request_path = automation_dir / "request.json"
    result_path = automation_dir / "result.json"

    monkeypatch.setattr(
        sa.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args=args[0],
            returncode=0,
            stdout="ok",
            stderr="",
        ),
    )
    completed = sa._run_simulator_command(
        framework="mujoco",
        result_path=result_path,
        request_path=request_path,
        command=["sim"],
        capture_root=context.capture_root,
        timeout_seconds=1,
        generated_at="2026-06-01T00:00:00Z",
    )
    assert completed["status"] == "completed"

    monkeypatch.setattr(
        sa.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args=args[0],
            returncode=2,
            stdout="",
            stderr="failed",
        ),
    )
    failed = sa._run_simulator_command(
        framework="pybullet",
        result_path=automation_dir / "failed.json",
        request_path=request_path,
        command=["sim"],
        capture_root=context.capture_root,
        timeout_seconds=1,
        generated_at="2026-06-01T00:00:00Z",
    )
    assert failed["status"] == "failed"

    def raise_timeout(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise subprocess.TimeoutExpired("sim", 1)

    monkeypatch.setattr(sa.subprocess, "run", raise_timeout)
    errored = sa._run_simulator_command(
        framework="isaac_sim",
        result_path=automation_dir / "errored.json",
        request_path=request_path,
        command=["sim"],
        capture_root=context.capture_root,
        timeout_seconds=1,
        generated_at="2026-06-01T00:00:00Z",
    )
    assert errored["status"] == "blocked"
    assert errored["reason"] == "execution_error"

    monkeypatch.setenv("BLUEPRINT_ALLOW_SIMULATOR_EXECUTION", "true")
    monkeypatch.setattr(sa.shutil, "which", lambda command: f"/bin/{command}")

    def fake_run_simulator_command(**kwargs: Any) -> dict[str, Any]:
        framework = kwargs["framework"]
        return {
            "schema_version": sa.SIMULATOR_RESULT_SCHEMA_VERSION,
            "framework": framework,
            "status": "failed" if framework == "mujoco" else "completed",
            "simulator_execution_proven": framework != "mujoco",
            "claim_boundary": dict(sa.CLAIM_BOUNDARY),
        }

    monkeypatch.setattr(sa, "_run_simulator_command", fake_run_simulator_command)
    execution = sa._build_simulator_execution_manifest(
        context=context,
        automation_dir=automation_dir,
        plan={"scene_id": "scene-1", "capture_id": "capture-1"},
        conversion_plan={"frameworks": {}},
        allow_simulator_execution=True,
        allowed_simulators=["isaac_sim", "mujoco", "pybullet"],
        simulator_commands={"mujoco": "sim --fail", "pybullet": "sim --ok"},
        generated_at="2026-06-01T00:00:00Z",
        timeout_seconds=1,
    )
    assert execution["overall_status"] == "failed"
    assert any(item["reason"] == "missing_execution_command" for item in execution["simulator_results"])

    execution_completed = sa._build_simulator_execution_manifest(
        context=context,
        automation_dir=automation_dir / "second",
        plan={"scene_id": "scene-1", "capture_id": "capture-1"},
        conversion_plan={"frameworks": {}},
        allow_simulator_execution=True,
        allowed_simulators=["pybullet"],
        simulator_commands={"pybullet": "sim --ok"},
        generated_at="2026-06-01T00:00:00Z",
        timeout_seconds=1,
    )
    assert execution_completed["overall_status"] == "completed"

    monkeypatch.setenv("BLUEPRINT_ALLOW_COSMOS_TRAINING", "true")
    import blueprint_pipeline.synthesis.cosmos_lora_training as cosmos_training

    monkeypatch.setattr(
        cosmos_training,
        "run_cosmos_lora_training",
        lambda **kwargs: {
            "status": "completed",
            "reason": None,
            "checkpoint_path": "checkpoint.pt",
            "kwargs_seen": sorted(kwargs),
        },
    )
    training = sa._training_orchestration_manifest(
        context=context,
        automation_dir=automation_dir,
        plan={"scene_id": "scene-1", "capture_id": "capture-1"},
        allow_training=True,
        training_command="train",
        training_timeout_seconds=5,
        generated_at="2026-06-01T00:00:00Z",
    )
    assert training["status"] == "completed"
    assert training["gpu_training_run"] is True
    assert training["claim_boundary"]["training_proof_available"] is True

    assert sa._parse_simulator_commands(["mujoco=sim --ok"]) == {"mujoco": "sim --ok"}
    with pytest.raises(ValueError):
        sa._parse_simulator_commands(["bad"])

    args = argparse.Namespace(
        agent_mode="fake",
        codex_thread_id=None,
        codex_sandbox="workspace-write",
        allow_live_agent_operator=False,
    )
    assert isinstance(sa._agent_adapter_from_args(args), sa.FakeSimulationAutomationAgentAdapter)
    assert isinstance(sa._episode_agent_adapter_from_args(args), sa.FakeEpisodeSpecAgentAdapter)
    args.agent_mode = "codex-sdk"
    assert isinstance(sa._agent_adapter_from_args(args), sa.CodexSdkSimulationAutomationAgentAdapter)
    args.agent_mode = "agents-sdk"
    assert isinstance(sa._agent_adapter_from_args(args), sa.AgentsSdkCodexMCPAdapter)
    args.agent_mode = "none"
    assert sa._agent_adapter_from_args(args) is None
    assert sa._episode_agent_adapter_from_args(args) is None

    calls: list[dict[str, Any]] = []

    def fake_build_simulation_automation(**kwargs: Any) -> dict[str, str]:
        calls.append(kwargs)
        return {"manifest_path": "manifest.json", "plan_path": "plan.json", "status": "blocked"}

    monkeypatch.setattr(sa, "build_simulation_automation", fake_build_simulation_automation)
    assert (
        sa.main(
            [
                "--capture-root",
                str(context.capture_root),
                "--scene-asset",
                "scene.ply",
                "--allow-cpu-simulator-preflight",
                "--cpu-preflight-backend",
                "mujoco",
                "--cpu-preflight-smoke-steps",
                "2",
                "--allow-cpu-preflight-render",
                "--allow-simulator-execution",
                "--allow-simulator",
                "mujoco",
                "--simulator-command",
                "mujoco=sim --ok",
                "--simulator-timeout-seconds",
                "3",
                "--allow-training",
                "--training-command",
                "train",
                "--training-timeout-seconds",
                "4",
                "--agent-mode",
                "fake",
                "--allow-live-agent-operator",
                "--codex-sandbox",
                "read-only",
                "--codex-thread-id",
                "thread-1",
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "[simulation-automation] status=blocked" in out
    assert calls[-1]["simulator_commands"] == {"mujoco": "sim --ok"}
    assert calls[-1]["allow_cpu_preflight_render"] is True

    def raise_value_error(**kwargs: Any) -> dict[str, str]:
        del kwargs
        raise ValueError("invalid capture")

    monkeypatch.setattr(sa, "build_simulation_automation", raise_value_error)
    assert sa.main(["--capture-root", str(context.capture_root)]) == 1
    assert "[simulation-automation] FAILED: invalid capture" in capsys.readouterr().out

    guard = compile("\nraise SystemExit(main())", sa.__file__, "exec").replace(
        co_firstlineno=3737
    )
    with pytest.raises(SystemExit) as raised:
        exec(guard, {"SystemExit": SystemExit, "main": lambda: 9})
    assert raised.value.code == 9
