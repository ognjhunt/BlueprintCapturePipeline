from __future__ import annotations

import json
from pathlib import Path

from blueprint_pipeline.evaluation_run import (
    EVALUATION_RUN_COMPONENTS,
    EvaluationRunAdapterDescriptor,
    default_evaluation_run_adapter_registry,
    compile_evaluation_run,
    main,
    validate_evaluation_run_spec,
)
from blueprint_pipeline.g1_kitchen_evaluation_run_adapter import (
    build_g1_kitchen_evaluation_run_spec,
)


def _warehouse_spec() -> dict:
    return {
        "schema_version": "evaluation_run.v1",
        "run_id": "warehouse-policy-eval-001",
        "mode": "evaluate",
        "scene_bundle": {
            "adapter_id": "openusd_scene_bundle",
            "adapter_version": "1",
            "bundle_id": "warehouse-a",
            "uri": "gs://blueprint-scenes/warehouse-a.zip",
            "entrypoint": "Warehouse.usd",
            "content_digest": "sha256:" + "a" * 64,
        },
        "robot_adapter": {
            "adapter_id": "isaac_robot_asset",
            "adapter_version": "1",
            "robot_profile_id": "mobile-manipulator-a",
            "asset_ref": "Robots/MobileManipulator/robot.usd",
        },
        "task_scenario_pack": {
            "adapter_id": "manifest_task_scenario_pack",
            "adapter_version": "1",
            "pack_id": "warehouse-pick-pack",
            "tasks": [{"task_id": "pick-tote"}],
            "scenarios": [
                {"scenario_id": "pick-tote-near-shelf", "task_id": "pick-tote"}
            ],
        },
        "policy_adapter": {
            "adapter_id": "http_policy_worker",
            "adapter_version": "1",
            "policy_id": "customer-policy-17",
            "observation_schema_ref": "blueprint://schemas/robot_eval_observation.v1",
            "action_schema_ref": "blueprint://schemas/robot_eval_action_trace.v1",
            "api_token": "must-not-be-recorded",
        },
        "runtime_provider_profile": {
            "adapter_id": "isaac_provider_runtime",
            "adapter_version": "1",
            "execution_adapter_id": "isaac_generic_evaluation",
            "profile_id": "isaac-a40",
            "providers": ["runpod", "vast"],
            "simulator": "isaac_sim",
            "max_spend_usd": 2.0,
        },
        "proof_contract": {
            "adapter_id": "declared_evidence_proof_contract",
            "adapter_version": "1",
            "contract_id": "warehouse-task-eval",
            "required_evidence": ["action_trace", "task_state_change", "teardown"],
            "claim_ceiling": {"simulator_task_success": True, "physical_readiness": False},
            "prohibited_claims": ["physical_robot_readiness"],
        },
        "metadata": {"customer_request_id": "request-17"},
    }


def test_compile_evaluation_run_materializes_six_part_provider_neutral_plan(
    tmp_path: Path,
) -> None:
    plan = compile_evaluation_run(_warehouse_spec(), output_dir=tmp_path)

    assert plan["status"] == "prepared"
    assert tuple(plan["component_bindings"]) == EVALUATION_RUN_COMPONENTS
    assert plan["execution_handoff"] == {
        "adapter_id": "isaac_generic_evaluation",
        "provider_mutation_allowed": False,
        "requires_explicit_runtime_gate": True,
    }
    assert plan["claim_boundary"]["plan_is_not_task_success_proof"] is True
    assert plan["spec_digest"].startswith("sha256:")
    persisted_spec = json.loads((tmp_path / "evaluation_run_spec.json").read_text())
    persisted_plan = json.loads((tmp_path / "evaluation_run_plan.json").read_text())
    assert persisted_spec["policy_adapter"]["api_token"] == "REDACTED_SECRET_FIELD"
    assert "must-not-be-recorded" not in json.dumps(persisted_spec)
    assert persisted_plan["artifacts"]["spec"].endswith("evaluation_run_spec.json")


def test_validation_fails_closed_for_missing_or_cross_inconsistent_parts() -> None:
    spec = _warehouse_spec()
    del spec["robot_adapter"]
    spec["task_scenario_pack"]["scenarios"].append(
        {"scenario_id": "pick-unknown", "task_id": "unknown-task"}
    )

    result = validate_evaluation_run_spec(spec)

    assert result["status"] == "blocked"
    assert "robot_adapter:missing" in result["errors"]
    assert "robot_adapter.adapter_id:missing" in result["errors"]
    assert "task_scenario_pack.scenarios:unknown_task_id" in result["errors"]


def test_new_adapter_can_be_registered_without_editing_the_compiler() -> None:
    spec = _warehouse_spec()
    spec["robot_adapter"]["adapter_id"] = "customer_dual_arm_adapter"

    blocked = compile_evaluation_run(spec)
    registry = default_evaluation_run_adapter_registry()
    registry.register(
        EvaluationRunAdapterDescriptor(
            component="robot_adapter",
            adapter_id="customer_dual_arm_adapter",
            adapter_version="1",
            capabilities=("dual_arm", "customer_owned"),
        )
    )
    prepared = compile_evaluation_run(spec, adapter_registry=registry)

    assert blocked["status"] == "blocked"
    assert (
        "robot_adapter.adapter:unsupported:customer_dual_arm_adapter@1"
        in blocked["validation"]["errors"]
    )
    assert prepared["status"] == "prepared"
    assert prepared["adapter_resolution"]["robot_adapter"] == {
        "status": "resolved",
        "adapter_id": "customer_dual_arm_adapter",
        "adapter_version": "1",
        "capabilities": ["dual_arm", "customer_owned"],
    }


def test_startup_canary_keeps_all_six_parts_without_claiming_task_execution() -> None:
    spec = _warehouse_spec()
    spec["run_id"] = "warehouse-image-canary"
    spec["mode"] = "startup_canary"
    spec["scene_bundle"].update(
        {"uri": None, "entrypoint": None, "content_digest": None}
    )
    spec["task_scenario_pack"].update({"tasks": [], "scenarios": []})
    spec["policy_adapter"]["policy_id"] = None

    plan = compile_evaluation_run(spec)

    assert plan["status"] == "prepared"
    assert plan["mode"] == "startup_canary"
    assert len(plan["component_bindings"]) == 6
    assert plan["claim_boundary"]["plan_is_not_policy_execution_proof"] is True


def test_kitchen_compatibility_adapter_produces_generic_contract_and_strips_signature(
    tmp_path: Path,
) -> None:
    spec = build_g1_kitchen_evaluation_run_spec(
        out_dir=tmp_path / "attempt-1",
        scenarios=[
            {
                "scenario_id": "dishwasher-open",
                "task_id": "open-dishwasher",
                "spawn_position_xyz": [0, 0, 0],
                "target_position_xyz": [1, 0, 0],
            }
        ],
        kitchen_uri="https://objects.example/kitchen.zip?X-Signature=private",
        kitchen_main_usd_relative="Collected_KitchenRoom/KitchenRoom.usd",
        kitchen_asset_inventory={
            "archive_sha256": "b" * 64,
            "file_count": 36,
            "total_bytes": 1234,
        },
        g1_usd="Isaac/Robots/Unitree/G1/g1.usd",
        policy_id="groot_sonic",
        providers=["runpod"],
        selected_image="registry.example/eval@sha256:" + "c" * 64,
        allow_paid=True,
        max_spend_usd=2.0,
        image_startup_canary=False,
        serve=False,
        requested_render_settings={"width": 1280, "height": 960},
    )

    plan = compile_evaluation_run(spec, output_dir=tmp_path / "compiled")

    assert plan["status"] == "prepared"
    assert spec["scene_bundle"]["uri"] == "https://objects.example/kitchen.zip"
    assert "private" not in json.dumps(spec)
    assert plan["component_bindings"]["scene_bundle"]["adapter_id"] == (
        "openusd_scene_bundle"
    )
    assert plan["component_bindings"]["runtime_provider_profile"][
        "adapter_id"
    ] == "isaac_provider_runtime"
    assert spec["metadata"]["scene_specific_name_is_not_platform_contract"] is True


def test_kitchen_startup_canary_compiles_without_scene_or_scenarios(tmp_path: Path) -> None:
    spec = build_g1_kitchen_evaluation_run_spec(
        out_dir=tmp_path,
        scenarios=[],
        kitchen_uri=None,
        kitchen_main_usd_relative="Collected_KitchenRoom/KitchenRoom.usd",
        kitchen_asset_inventory=None,
        g1_usd="Isaac/Robots/Unitree/G1/g1.usd",
        policy_id="blueprint_default_walk_to_target_smoke_policy",
        providers=["runpod"],
        selected_image="registry.example/eval@sha256:" + "d" * 64,
        allow_paid=True,
        max_spend_usd=2.0,
        image_startup_canary=True,
        serve=False,
        requested_render_settings={},
    )

    result = compile_evaluation_run(spec)

    assert result["status"] == "prepared"
    assert result["mode"] == "startup_canary"


def test_module_cli_compiles_spec_without_execution(tmp_path: Path, capsys) -> None:
    spec_path = tmp_path / "input.json"
    spec_path.write_text(json.dumps(_warehouse_spec()), encoding="utf-8")
    output_dir = tmp_path / "output"

    exit_code = main(["--spec", str(spec_path), "--output-dir", str(output_dir)])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["status"] == "prepared"
    assert payload["execution_handoff"]["provider_mutation_allowed"] is False
    assert (output_dir / "evaluation_run_plan.json").is_file()
