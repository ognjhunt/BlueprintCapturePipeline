from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pytest

from blueprint_pipeline.evaluation_run_contract import EvaluationRunSpec
from blueprint_pipeline.evaluation_run_execution import (
    EvaluationRunExecutionRegistry,
    default_evaluation_run_execution_registry,
    execute_evaluation_run,
    main,
)
from blueprint_pipeline.g1_kitchen_evaluation_run_adapter import (
    build_g1_kitchen_evaluation_run_spec,
    g1_kitchen_job_kwargs_from_evaluation_run,
)
from blueprint_pipeline.robot_eval_evaluation_run_adapter import (
    execute_robot_eval_request_as_evaluation_run,
    execute_legacy_robot_eval_request_as_evaluation_run,
    robot_eval_job_request_from_evaluation_run,
)


def _spec(*, execution_adapter_id: str = "fixture_executor") -> dict[str, Any]:
    return {
        "schema_version": "evaluation_run.v1",
        "run_id": "dynamic-eval-001",
        "mode": "evaluate",
        "scene_bundle": {
            "adapter_id": "capture_site_scene_bundle",
            "adapter_version": "1",
            "bundle_id": "site-a",
            "uri": "gs://scenes/site-a",
            "entrypoint": "capture_root",
            "content_digest": "sha256:" + "a" * 64,
        },
        "robot_adapter": {
            "adapter_id": "robot_profile_adapter",
            "adapter_version": "1",
            "robot_profile_id": "dual-arm-v1",
            "asset_ref": "robots/dual-arm.urdf",
            "embodiment": "dual_arm",
            "sensors": ["rgb"],
        },
        "task_scenario_pack": {
            "adapter_id": "manifest_task_scenario_pack",
            "adapter_version": "1",
            "pack_id": "site-a-pack",
            "tasks": [{"task_id": "load-bin"}],
            "scenarios": [{"scenario_id": "load-bin-1", "task_id": "load-bin"}],
        },
        "policy_adapter": {
            "adapter_id": "robot_eval_policy_package",
            "adapter_version": "1",
            "policy_id": "policy-a",
            "observation_schema_ref": "blueprint://schemas/robot_eval_observation.v1",
            "action_schema_ref": "blueprint://schemas/robot_eval_action_trace.v1",
            "policy_package": {
                "sim_controller_plugin": {
                    "simulator_framework": "fixture",
                    "plugin_uri": "adapter://policy-a",
                }
            },
        },
        "runtime_provider_profile": {
            "adapter_id": "robot_eval_runtime_provider",
            "adapter_version": "1",
            "execution_adapter_id": execution_adapter_id,
            "profile_id": "fixture-local",
            "providers": ["fixture_local"],
            "simulator": "fixture",
            "max_spend_usd": 1.0,
            "timeout_seconds": 30,
        },
        "proof_contract": {
            "adapter_id": "robot_eval_proof_contract",
            "adapter_version": "1",
            "contract_id": "robot-eval-proof",
            "required_evidence": ["scenario_eval_matrix", "policy_execution_trace"],
            "claim_ceiling": {"physical_robot_readiness": False},
            "prohibited_claims": ["physical_robot_readiness"],
            "rights_privacy_scope": {
                "status": "cleared_for_robot_eval",
                "external_use_allowed": True,
            },
        },
        "metadata": {"operation": "evaluate_only"},
    }


class _RecordingExecutor:
    adapter_id = "fixture_executor"

    def __init__(self, *, result_status: str = "completed", raises: bool = False) -> None:
        self.calls: list[dict[str, Any]] = []
        self.result_status = result_status
        self.raises = raises

    def execute(
        self,
        *,
        spec: EvaluationRunSpec,
        output_dir: Path,
        context: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        self.calls.append(
            {"run_id": spec.run_id, "output_dir": output_dir, "context": dict(context)}
        )
        if self.raises:
            raise RuntimeError("sensitive provider detail must not be persisted")
        return {
            "schema_version": "fixture_evaluation_run_result.v1",
            "status": self.result_status,
            "manifest_path": str(output_dir / "fixture-result.json"),
            "evaluation_run_proof_evidence": {
                "scenario_eval_matrix": True,
                "policy_execution_trace": {"satisfied": True},
            },
        }


def test_execution_gate_compiles_without_invoking_adapter_or_persisting_context(
    tmp_path: Path,
) -> None:
    adapter = _RecordingExecutor()
    registry = EvaluationRunExecutionRegistry([adapter])

    result = execute_evaluation_run(
        _spec(),
        output_dir=tmp_path,
        allow_execution=False,
        context={"api_token": "never-persist-this"},
        execution_registry=registry,
    )

    assert result.manifest["status"] == "prepared"
    assert result.manifest["execution_started"] is False
    assert "evaluation_run_execution_not_authorized" in result.manifest["blockers"]
    assert adapter.calls == []
    persisted = (tmp_path / "evaluation_run_execution.json").read_text()
    assert "never-persist-this" not in persisted
    assert json.loads(persisted)["context_values_persisted"] is False


def test_authorized_execution_resolves_adapter_and_binds_result_to_spec_digest(
    tmp_path: Path,
) -> None:
    adapter = _RecordingExecutor()
    registry = EvaluationRunExecutionRegistry([adapter])

    result = execute_evaluation_run(
        _spec(),
        output_dir=tmp_path,
        allow_execution=True,
        context={"capture_root": str(tmp_path / "capture")},
        execution_registry=registry,
        generated_at="2026-07-12T00:00:00+00:00",
    )

    assert result.manifest["status"] == "completed"
    assert result.manifest["execution_started"] is True
    assert result.manifest["spec_digest"].startswith("sha256:")
    assert result.manifest["adapter_result_summary"]["status"] == "completed"
    assert result.manifest["proof_contract_evaluation"]["status"] == "passed"
    assert result.manifest["claim_boundary"]["public_claim_upgrade_allowed"] is True
    assert adapter.calls[0]["run_id"] == "dynamic-eval-001"


def test_unknown_or_raising_execution_adapter_fails_closed_with_terminal_artifact(
    tmp_path: Path,
) -> None:
    unknown = execute_evaluation_run(
        _spec(execution_adapter_id="missing_executor"),
        output_dir=tmp_path / "unknown",
        allow_execution=True,
        execution_registry=EvaluationRunExecutionRegistry(),
    )
    raising_adapter = _RecordingExecutor(raises=True)
    raised = execute_evaluation_run(
        _spec(),
        output_dir=tmp_path / "raised",
        allow_execution=True,
        execution_registry=EvaluationRunExecutionRegistry([raising_adapter]),
    )

    assert unknown.manifest["status"] == "blocked"
    assert (
        "evaluation_run_execution_adapter_unavailable:missing_executor"
        in unknown.manifest["blockers"]
    )
    assert raised.manifest["status"] == "blocked"
    assert raised.adapter_result["error_type"] == "RuntimeError"
    assert raised.adapter_result["raw_error_message_recorded"] is False
    assert "sensitive provider detail" not in (
        tmp_path / "raised" / "evaluation_run_execution.json"
    ).read_text()


def test_missing_declared_evidence_blocks_claim_upgrade_without_rewriting_runtime_status(
    tmp_path: Path,
) -> None:
    adapter = _RecordingExecutor()

    def incomplete_execute(**_kwargs: Any) -> Mapping[str, Any]:
        return {"status": "completed", "evaluation_run_proof_evidence": {}}

    adapter.execute = incomplete_execute  # type: ignore[method-assign]
    result = execute_evaluation_run(
        _spec(),
        output_dir=tmp_path,
        allow_execution=True,
        execution_registry=EvaluationRunExecutionRegistry([adapter]),
    )

    assert result.manifest["status"] == "completed"
    assert result.manifest["proof_contract_evaluation"]["status"] == "evidence_incomplete"
    assert result.manifest["proof_contract_evaluation"]["missing_evidence"] == [
        "scenario_eval_matrix",
        "policy_execution_trace",
    ]
    assert result.manifest["claim_boundary"]["public_claim_upgrade_allowed"] is False


def test_default_execution_registry_exposes_generic_and_kitchen_implementations() -> None:
    assert default_evaluation_run_execution_registry().manifest() == [
        "isaac_g1_kitchen_parity_compatibility",
        "robot_eval_job_orchestrator",
    ]


def test_execution_module_cli_is_plan_only_without_explicit_gate(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(
        json.dumps(_spec(execution_adapter_id="robot_eval_job_orchestrator")),
        encoding="utf-8",
    )

    exit_code = main(
        ["--spec", str(spec_path), "--output-dir", str(tmp_path / "output")]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["status"] == "prepared"
    assert payload["execution_started"] is False
    assert "evaluation_run_execution_not_authorized" in payload["blockers"]


def test_robot_eval_request_is_derived_from_six_part_spec(tmp_path: Path) -> None:
    spec = EvaluationRunSpec.from_mapping(_spec(execution_adapter_id="robot_eval_job_orchestrator"))

    request = robot_eval_job_request_from_evaluation_run(
        spec,
        capture_root=tmp_path / "capture",
    )

    assert request["job_id"] == spec.run_id
    assert request["site_package"]["package_uri"] == spec.scene_bundle["uri"]
    assert request["robot_profile"]["robot_profile_id"] == "dual-arm-v1"
    assert request["requested_tasks"] == [
        {"task_id": "load-bin", "scenario_ids": ["load-bin-1"]}
    ]
    assert request["policy_package"] == spec.policy_adapter["policy_package"]
    assert request["simulator_preference"] == "fixture"
    assert request["rights_privacy_scope"]["external_use_allowed"] is True
    assert request["evaluation_run_proof_contract"] == dict(spec.proof_contract)
    assert request["provenance"]["source_spec_is_execution_authority"] is True


def test_legacy_robot_eval_gateway_compiles_spec_before_low_level_builder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from blueprint_pipeline import robot_eval_job_orchestrator as orchestrator

    captured: dict[str, Any] = {}

    def _fake_builder(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {
            "schema_version": "robot_eval_job_result.v1",
            "status": "fixture_evaluation_completed",
            "job_dir": str(tmp_path / "capture" / "pipeline" / "robot_eval_jobs" / "legacy-1"),
            "manifest_path": str(tmp_path / "job-run.json"),
        }

    monkeypatch.setattr(orchestrator, "build_robot_eval_job", _fake_builder)
    result = execute_robot_eval_request_as_evaluation_run(
        capture_root=tmp_path / "capture",
        job_request={
            "schema_version": "robot_eval_job_request.v1",
            "site_package": {"site_id": "warehouse-a", "package_uri": "gs://scenes/a"},
            "robot_profile": {"robot_profile_id": "mobile-arm"},
            "requested_tasks": [{"task_id": "pick-bin", "scenario_ids": ["pick-bin-1"]}],
            "policy_id": "policy-a",
        },
        job_id="legacy-1",
        provisioner="fixture_local",
        simulator="fixture",
        allow_simulator_execution=False,
    )

    assert result["status"] == "fixture_evaluation_completed"
    assert captured["job_request"]["provenance"]["source_spec_is_execution_authority"] is True
    assert captured["job_request"]["site_package"]["site_id"] == "warehouse-a"
    assert captured["job_request"]["robot_profile"]["robot_profile_id"] == "mobile-arm"
    authority = tmp_path / "capture" / "pipeline" / "evaluation_runs" / "legacy-1"
    assert (authority / "evaluation_run_spec.json").is_file()
    assert (authority / "evaluation_run_plan.json").is_file()
    assert (authority / "evaluation_run_execution.json").is_file()


def test_legacy_robot_eval_entrypoint_is_a_one_release_alias() -> None:
    assert (
        execute_legacy_robot_eval_request_as_evaluation_run
        is execute_robot_eval_request_as_evaluation_run
    )


def test_kitchen_execution_kwargs_come_from_spec_and_reject_core_overrides(
    tmp_path: Path,
) -> None:
    raw = build_g1_kitchen_evaluation_run_spec(
        out_dir=tmp_path,
        run_id="kitchen-run-1",
        scenarios=[{"scenario_id": "open-1", "task_id": "open-dishwasher"}],
        kitchen_uri="https://objects.example/kitchen.zip?signature=secret",
        kitchen_main_usd_relative="Collected_KitchenRoom/KitchenRoom.usd",
        kitchen_asset_inventory={"archive_sha256": "b" * 64},
        g1_usd="Isaac/Robots/Unitree/G1/g1.usd",
        policy_id="groot_sonic",
        providers=["runpod"],
        selected_image="registry/eval@sha256:" + "c" * 64,
        allow_paid=True,
        max_spend_usd=2.0,
        image_startup_canary=False,
        serve=False,
        requested_render_settings={"steps": 64, "width": 1280},
    )
    spec = EvaluationRunSpec.from_mapping(raw)

    kwargs = g1_kitchen_job_kwargs_from_evaluation_run(
        spec,
        output_dir=tmp_path / "execution",
        context={
            "allow_paid": True,
            "scene_transport_uri": "https://objects.example/kitchen.zip?signature=ephemeral",
            "options": {"max_attempts": 1},
        },
    )

    assert kwargs["evaluation_run_id"] == "kitchen-run-1"
    assert kwargs["policy_id"] == "groot_sonic"
    assert kwargs["provider"] == "runpod"
    assert kwargs["steps"] == 64
    assert kwargs["max_attempts"] == 1
    with pytest.raises(ValueError, match="unsupported_g1_kitchen_execution_options"):
        g1_kitchen_job_kwargs_from_evaluation_run(
            spec,
            output_dir=tmp_path,
            context={"options": {"policy_id": "override-not-allowed"}},
        )


def test_generic_engine_dispatches_kitchen_compatibility_from_authoritative_spec(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from blueprint_pipeline import isaac_g1_kitchen_parity_job as legacy_job

    captured: dict[str, Any] = {}

    def _fake_legacy_job(**kwargs):
        captured.update(kwargs)
        return {
            "schema_version": "isaac_g1_kitchen_parity_job.v1",
            "status": "prepared",
            "blockers": [],
        }

    monkeypatch.setattr(legacy_job, "run_isaac_g1_kitchen_parity_job", _fake_legacy_job)
    spec = build_g1_kitchen_evaluation_run_spec(
        out_dir=tmp_path,
        run_id="kitchen-authority-1",
        scenarios=[{"scenario_id": "open-1", "task_id": "open-dishwasher"}],
        kitchen_uri="https://objects.example/kitchen.zip",
        kitchen_main_usd_relative="Collected_KitchenRoom/KitchenRoom.usd",
        kitchen_asset_inventory={"archive_sha256": "d" * 64},
        g1_usd="Isaac/Robots/Unitree/G1/g1.usd",
        policy_id="groot_sonic",
        providers=["runpod"],
        selected_image="registry/eval@sha256:" + "e" * 64,
        allow_paid=False,
        max_spend_usd=2.0,
        image_startup_canary=False,
        serve=False,
        requested_render_settings={"steps": 32},
    )

    result = execute_evaluation_run(
        spec,
        output_dir=tmp_path / "authority",
        allow_execution=True,
        context={
            "allow_paid": False,
            "scene_transport_uri": "https://objects.example/kitchen.zip?signature=ephemeral",
        },
    )

    assert result.manifest["status"] == "prepared"
    assert result.manifest["execution_started"] is True
    assert captured["evaluation_run_id"] == "kitchen-authority-1"
    assert captured["scenarios"] == [
        {"scenario_id": "open-1", "task_id": "open-dishwasher"}
    ]
    assert captured["policy_id"] == "groot_sonic"
    persisted = (tmp_path / "authority" / "evaluation_run_execution.json").read_text()
    assert "signature=ephemeral" not in persisted
