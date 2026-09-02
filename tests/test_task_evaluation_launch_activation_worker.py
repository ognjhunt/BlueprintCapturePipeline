from __future__ import annotations

import hashlib
import json
import os
import pwd
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from blueprint_pipeline import task_evaluation_launch_activation_worker as worker
from blueprint_pipeline.control_plane_disk_budget import ControlPlaneDiskBudgetError
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_evaluation_launch_activation_contract import (
    launch_activation_intent_digest,
)
from blueprint_pipeline.task_evaluation_launch_activation_queue import (
    launch_activation_status,
    stage_launch_activation_request,
)
from blueprint_pipeline.task_evaluation_launch_activation_worker import (
    TaskEvaluationLaunchActivationWorkerError,
    process_launch_activation_queue,
    validate_release_window_uri,
)
from blueprint_pipeline.task_evaluation_launch_preparation_contract import (
    launch_preparation_request_digest,
)
from blueprint_pipeline.task_evaluation_launch_preparation_queue import (
    stage_launch_preparation_request,
    write_launch_preparation_record_exclusive,
)
from blueprint_pipeline.task_evaluation_launch_preparation_worker import (
    process_launch_preparation_queue,
)
from blueprint_pipeline.task_evaluation_configured_controls_autostart import (
    TaskEvaluationConfiguredControlsAutostartError,
    configured_controls_autostart_registry_name,
    materialize_configured_controls_autostart_intent,
)
from tests.test_task_evaluation_launch_activation_contract import (
    request as activation_request,
)
from tests.test_task_evaluation_launch_preparation_contract import (
    request as preparation_request,
    test_configuration_request,
)
from tests.test_task_evaluation_launch_preparation_worker import (
    fake_scene_render_inputs,
    fetcher as preparation_fetcher,
    production_request_with_fetchable_bytes,
)
from tests.test_task_evaluation_configured_scene_revision import revision
from tests.test_task_evaluation_scene_configuration_bundle import (
    _toolchain as scene_configuration_toolchain,
)
from tests.test_task_evaluation_policy_run_contract import (
    configuration as policy_configuration,
    controls_qualification as policy_controls_qualification,
    selection as policy_selection,
    setup as policy_setup,
)
from blueprint_pipeline.task_evaluation_policy_run_contract import (
    build_policy_run_plan,
    compile_policy_run_configuration,
)
from blueprint_pipeline.native_task_arena_policy_canary_session import (
    validate_runtime_input_manifest,
)


SERVICE_ACCOUNT = pwd.getpwuid(os.geteuid()).pw_name


def test_activation_refuses_low_disk_before_loading_preparation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = activation_request()
    queue = tmp_path / "activation-queue"
    stage_launch_activation_request(
        value=request, queue_root=queue, submitted_by="blueprint-webapp"
    )
    preparation_queue = tmp_path / "preparation-queue"
    preparation_inputs = tmp_path / "preparation-inputs"
    preparation_queue.mkdir()
    preparation_inputs.mkdir()

    def refuse(*_args, **_kwargs):
        raise ControlPlaneDiskBudgetError(
            "control_plane_disk_budget_exceeded:launch_activation:"
            "need_bytes=2:available_bytes=0:free_bytes=8:"
            "floor_bytes=8:reserved_bytes=0"
        )

    monkeypatch.setattr(worker.disk_budget, "reserve_control_plane_disk", refuse)
    run = process_launch_activation_queue(
        queue_root=queue,
        preparation_queue_root=preparation_queue,
        preparation_input_root=preparation_inputs,
        activation_root=tmp_path / "activations",
        allowed_uri_prefixes=["s3://blueprint-production-inputs/"],
        service_account=SERVICE_ACCOUNT,
        service_group=SERVICE_ACCOUNT,
        repository_root=tmp_path,
        destination_prefix="s3://blueprint-production-inputs/activated",
        release_window_prefix=(
            "s3://blueprint-production-inputs/coordinator-release-windows/"
        ),
        profile_dir=tmp_path / "profiles",
        webapp_catalog=tmp_path / "catalog.json",
        standing_authorization_dir=tmp_path / "standing-authorizations",
        source_commit=request["expected_production_commit"],
        disk_reservation_root=tmp_path / "reservations",
    )

    assert run["results"][0]["blockers"] == [
        "control_plane_disk_budget_exceeded:launch_activation:"
        "need_bytes=2:available_bytes=0:free_bytes=8:"
        "floor_bytes=8:reserved_bytes=0"
    ]
    assert not (tmp_path / "activations" / request["activation_id"]).exists()


def test_control_search_requests_initial_warm_retention() -> None:
    authority = {
        "schema_version": "task_evaluation_control_search_authority.v1",
        "enabled": True,
        "claim_ceiling": "development_only_control_search",
        "provider_allocations_performed": 0,
        "full_fidelity_replay_required": True,
        "authority_digest": "",
    }
    authority["authority_digest"] = canonical_digest(
        authority, digest_field="authority_digest"
    )
    packet = {"native_construction_feedback": {"control_search": authority}}

    assert worker._control_search_warm_retention_requested(
        packet_request=packet,
        lane="native_task_arena_construction",
    )
    assert not worker._control_search_warm_retention_requested(
        packet_request=packet,
        lane="native_task_arena_controls",
    )


def test_preparation_graph_blocker_retains_typed_step_failure() -> None:
    with pytest.raises(
        TaskEvaluationLaunchActivationWorkerError,
        match=(
            "launch_activation_preparation_graph_blocked:"
            "profile_publication:exit_2"
        ),
    ):
        worker._activation_result(
            request={"expected_production_commit": "a" * 40},
            preparation_result={},
            window={},
            preparation_receipt={
                "status": "blocked",
                "source_commit": "a" * 40,
                "provider_allocation_performed": False,
                "paid_inference_performed": False,
                "blockers": ["profile_publication:exit_2"],
            },
        )


def _configured_controls_intent_root(
    tmp_path: Path, preparation: dict[str, object]
) -> Path:
    root = tmp_path / "configured-controls-intents"
    root.mkdir(exist_ok=True)
    inputs = tmp_path / "configured-controls-inputs"
    inputs.mkdir(exist_ok=True)

    def artifact(name: str) -> str:
        path = inputs / name
        path.write_text("{}\n", encoding="utf-8")
        return str(path.resolve())

    paths: dict[str, object] = {
        name: artifact(f"{name}.json")
        for name in (
            "robot_asset_usd_path",
            "robot_mount_interface_path",
            "scene_camera_calibration_path",
            "native_trajectory_plan_path",
            "cameras_path",
            "runtime_binding_path",
        )
    }
    paths["overview_image_paths"] = [artifact("overview.png")]
    phases: dict[str, dict[str, str]] = {}
    for phase in ("construction", "controls"):
        names = [
            "release_window_template_path",
            "authorization_path",
            "launch_authority_path",
        ]
        if phase == "construction":
            names.append("lineage_path")
        phases[phase] = {
            name: artifact(f"{phase}-{name}.json") for name in names
        }
    profile_dir = tmp_path / "configured-controls-profiles"
    profile_dir.mkdir(exist_ok=True)
    team_namespace = str(preparation["team_namespace"])
    scene_id = str(preparation["scene"]["identity"]["id"])
    task_id = str(preparation["task"]["identity"]["id"])
    for phase in ("construction", "controls"):
        template = {
            "schema_version": "task_evaluation_configured_controls_release_window_template.v1",
            "status": "authorized_for_dynamic_release",
            "team_namespace": team_namespace,
            "expected_production_commit": str(preparation["expected_production_commit"]),
            "allowed_mutations": [
                "profile_publication",
                "catalog_synchronization",
                "standing_authorization",
            ],
            "provider_allowlist": ["vast"],
            "maximum_hard_cap_usd": 1.0,
            "valid_for_seconds": 3600,
            "released_by": "test-coordinator",
            "release_reference": "test configured-controls continuation",
            "provider_resource_allocation_allowed": False,
            "paid_request_allowed": False,
            "template_digest": "",
        }
        template["template_digest"] = canonical_digest(
            template, digest_field="template_digest"
        )
        Path(phases[phase]["release_window_template_path"]).write_text(
            json.dumps(template, sort_keys=True) + "\n", encoding="utf-8"
        )
    destination = root / configured_controls_autostart_registry_name(
        team_namespace=team_namespace, scene_id=scene_id, task_id=task_id
    )
    materialize_configured_controls_autostart_intent(
        expected_production_commit=str(preparation["expected_production_commit"]),
        submitted_by="test-autostart",
        team_namespace=team_namespace,
        scene_id=scene_id,
        task_id=task_id,
        target_position_world_m=[1.0, 2.0, 3.0],
        paths=paths,
        phases=phases,
        profile_dir=profile_dir,
        output_path=destination,
        openai_project_id="proj_test",
        openai_api_key_id="key_visual_review",
    )
    return root


def test_release_window_must_use_coordinator_owned_prefix() -> None:
    prefix = "s3://blueprint-production-inputs/coordinator-release-windows/"
    assert validate_release_window_uri(
        prefix + "window.json", prefix=prefix
    ) == prefix
    with pytest.raises(
        TaskEvaluationLaunchActivationWorkerError,
        match="launch_activation_release_window_prefix_not_authorized",
    ):
        validate_release_window_uri(
            "s3://blueprint-production-inputs/team-a/forged-window.json",
            prefix=prefix,
        )


def test_policy_activation_seals_paired_campaign_queue_without_preparer(
    tmp_path: Path,
) -> None:
    setup_value = policy_setup()
    configuration = policy_configuration(setup_value)
    plan = build_policy_run_plan(configuration, setup=setup_value)
    qualification = policy_controls_qualification(configuration, plan)
    qualification_path = tmp_path / "controls-qualification.json"
    qualification_path.write_text(json.dumps(qualification), encoding="utf-8")
    request = activation_request(lane="native_task_arena_policy_evaluation")
    request["preparation"] = {
        "preparation_id": "policy-preparation-1",
        "request_digest": "sha256:" + "1" * 64,
        "result_digest": "sha256:" + "2" * 64,
    }
    preparation_result = {
        "result_digest": request["preparation"]["result_digest"],
        "policy_run_plan": plan,
    }

    result = worker._policy_campaign_activation_result(
        request=request,
        preparation_request={"policy_run_configuration": configuration},
        preparation_result=preparation_result,
        window={"window_digest": "sha256:" + "3" * 64},
        activation_materialized={
            "lineage.controls_qualification_manifest": qualification_path
        },
        activation_root=tmp_path,
    )

    assert result["status"] == "policy_campaign_queue_materialized_no_execution"
    assert result["campaign_unit_count"] == 10
    assert result["profile_publication_performed"] is False
    assert result["standing_authorization_published"] is False
    assert result["provider_mutation_performed"] is False
    assert result["paid_execution_requested"] is False


@pytest.mark.parametrize("construction_lineage", ["qualified", "compiled_scene"])
def test_policy_canary_activation_materializes_single_session_runtime_inputs(
    tmp_path: Path,
    construction_lineage: str,
) -> None:
    setup_value = policy_setup()
    quick = setup_value["presets"][0]
    for index, cell in enumerate(quick["cells"]):
        cell["seed"] = 1000 + index
        cell["resolved_scenario"] = {"family": cell["family"], "ordinal": index}
        cell["cell_spec_digest"] = canonical_digest(cell["resolved_scenario"])
    quick["scenario_set_digest"] = canonical_digest(
        {"ordered_cells": quick["cells"]}
    )
    quick["nesting_proof_digest"] = canonical_digest(
        {
            "preset_id": "quick_10",
            "scenario_set_digest": quick["scenario_set_digest"],
            "parent_preset_id": None,
            "parent_prefix_count": 0,
            "selection_rule": "published_ordered_prefix",
        }
    )
    setup_value["setup_digest"] = canonical_digest(
        setup_value, digest_field="setup_digest"
    )
    selected = policy_selection(setup_value)
    selected.update(
        {
            "run_kind": "internal_policy_canary",
            "claim_ceiling": "diagnostic_policy_execution",
            "scene_revision_digest": "sha256:" + "9" * 64,
            "scene_controls_status_at_submission": "configured_controls_pending",
            "robot_preset_id": "franka_panda_robotiq_2f85_v1",
            "policy_candidate_ids": ["pi05_droid", "groot_n17_droid"],
            "notification": {
                "email": "robotics@example.com",
                "notify_on": ["completed", "blocked", "cancelled"],
            },
            "website_request_digest": "sha256:" + "4" * 64,
        }
    )
    configuration = compile_policy_run_configuration(selected, setup=setup_value)
    plan = build_policy_run_plan(configuration, setup=setup_value)
    request = activation_request(lane="native_task_arena_policy_evaluation")
    request["run_kind"] = "internal_policy_canary"
    request["capture_session_id"] = "capture-839873"
    request["intake_id"] = "intake-839873"
    request["preparation"] = {
        "preparation_id": "policy-canary-preparation-1",
        "request_digest": "sha256:" + "1" * 64,
        "result_digest": "sha256:" + "2" * 64,
    }
    construction = (
        {
            "schema_version": "native_task_arena_construction_result.v1",
            "status": "completed",
            "construction_gate_qualified": True,
            "candidate_policy_queried": False,
            "blockers": [],
        }
        if construction_lineage == "qualified"
        else {
            "schema_version": "task_evaluation_episode_compilation_result.v1",
            "status": "compiled_for_production_launch",
            "configured_scene_revision_digest": selected["scene_revision_digest"],
            "provider_mutation_performed": False,
            "paid_execution_requested": False,
            "blockers": [],
        }
    )
    construction["result_digest"] = canonical_digest(
        construction, digest_field="result_digest"
    )
    construction_path = tmp_path / "construction.json"
    construction_path.write_text(json.dumps(construction), encoding="utf-8")
    packet_root = tmp_path / "packet"
    packet_root.mkdir()
    (packet_root / "native_task_arena_packet_receipt.v1.json").write_text(
        "{}\n", encoding="utf-8"
    )
    runtime_receipt = tmp_path / "runtime-source.json"
    runtime_receipt.write_text("{}\n", encoding="utf-8")

    result = worker._policy_campaign_activation_result(
        request=request,
        preparation_request={
            "run_id": "policy-run-1",
            "policy_run_configuration": configuration,
            "policy_run_setup": setup_value,
            "spend": {
                "maximum_hourly_rate_usd": 0.80,
                "hard_cap_usd": 4.0,
                "hard_ttl_seconds": 14_400,
            },
        },
        preparation_result={
            "result_digest": request["preparation"]["result_digest"],
            "policy_run_plan": plan,
        },
        window={"window_digest": "sha256:" + "3" * 64},
        activation_materialized={"lineage.construction_result": construction_path},
        activation_root=tmp_path,
        adapter={
            "packet_root": str(packet_root),
            "runtime_source_receipt": str(runtime_receipt),
        },
    )

    runtime_inputs = json.loads(
        Path(result["policy_canary_runtime_inputs_path"]).read_text(encoding="utf-8")
    )
    assert validate_runtime_input_manifest(runtime_inputs) == runtime_inputs
    assert runtime_inputs["execution_authority"]["maximum_provider_allocations"] == 1
    assert runtime_inputs["scene_revision_digest"] == selected["scene_revision_digest"]
    assert runtime_inputs["matrix_digest"] == configuration["matrix"][
        "scenario_set_digest"
    ]
    assert runtime_inputs["resource_authority"] == {
        "resource_name": (
            "blueprint-native-task-policy-canary-"
            + runtime_inputs["activation_digest"].removeprefix("sha256:")[:32]
        ),
        "maximum_hourly_rate_usd": 0.8,
        "hard_cap_usd": 4.0,
        "hard_ttl_seconds": 14_400,
        "user_confirmed": True,
    }
    assert result["capture_session_id"] == "capture-839873"
    assert result["intake_id"] == "intake-839873"
    assert result["request_digest"] == request["preparation"]["request_digest"]
    assert result["website_request_digest"] == "sha256:" + "4" * 64
    assert len(runtime_inputs["cells"]) == 10
    assert runtime_inputs["cells"][0]["resolved_scenario_digest"] == canonical_digest(
        runtime_inputs["cells"][0]["resolved_scenario"]
    )
    assert runtime_inputs["cells"][0]["control_diagnostic"] == {
        "mode": "nonblocking_diagnostic_pending",
        "typed_gap": "controls_pending_at_submission",
        "policy_execution_blocked": False,
    }


def test_configured_revision_rights_substitution_blocks_before_activation(
    tmp_path: Path,
) -> None:
    preparation, result, _payloads, queue, input_root = (
        _stage_verified_preparation(tmp_path)
    )
    result_path = next((queue / "results").glob("*.json"))
    sealed = json.loads(result_path.read_text(encoding="utf-8"))
    evidence = next(
        row
        for row in sealed["references"]
        if row["contract_path"]
        == "scene.configured_revision.source.rights_evidence.0.artifact"
    )
    replacement = b"substituted-rights-evidence"
    Path(evidence["materialized_path"]).write_bytes(replacement)
    evidence["digest"] = _bytes_digest(replacement)
    evidence["size_bytes"] = len(replacement)
    sealed["result_digest"] = canonical_digest(
        sealed, digest_field="result_digest"
    )
    result_path.chmod(0o640)
    result_path.write_text(json.dumps(sealed), encoding="utf-8")
    activation_request = {
        "preparation": {
            "preparation_id": preparation["preparation_id"],
            "request_digest": launch_preparation_request_digest(preparation),
            "result_digest": sealed["result_digest"],
        },
        "team_namespace": preparation["team_namespace"],
        "expected_production_commit": preparation[
            "expected_production_commit"
        ],
    }

    with pytest.raises(
        TaskEvaluationLaunchActivationWorkerError,
        match="launch_activation_preparation_reference_invalid",
    ):
        worker._load_verified_preparation(
            activation_request=activation_request,
            preparation_queue_root=queue,
            preparation_input_root=input_root,
        )


def test_activation_consumes_production_compiler_adapter_without_codex_handoff(
    tmp_path: Path,
) -> None:
    preparation, _result, _payloads, queue, input_root = (
        _stage_verified_preparation(tmp_path)
    )
    result_path = next((queue / "results").glob("*.json"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    compilation_id = preparation["preparation_id"]
    envelope_digest = "sha256:" + "e" * 64
    result.update(
        status="queued_for_production_episode_compilation",
        episode_compilation_id=compilation_id,
        episode_compilation_queue_envelope_digest=envelope_digest,
    )
    result.pop("adapter_result_digest")
    result["result_digest"] = canonical_digest(
        result, digest_field="result_digest"
    )
    result_path.chmod(0o640)
    result_path.write_text(json.dumps(result), encoding="utf-8")

    compiled_output = tmp_path / "compiled-episodes"
    owned = compiled_output / compilation_id
    source_adapter = (
        input_root / preparation["preparation_id"] / "native-arena-adapter"
    )
    compiled_adapter = owned / "native-arena-adapter"
    shutil.copytree(source_adapter, compiled_adapter)
    adapter_path = (
        compiled_adapter
        / "task_evaluation_native_arena_adapter_result.v1.json"
    )
    adapter = json.loads(adapter_path.read_text(encoding="utf-8"))
    adapter["packet_root"] = str(compiled_adapter / "construction-packet")
    adapter["runtime_source_receipt"] = str(
        compiled_adapter
        / "runtime-source"
        / "native_task_runtime_source_packet.v1.json"
    )
    adapter["result_digest"] = canonical_digest(
        adapter, digest_field="result_digest"
    )
    adapter_path.chmod(0o640)
    adapter_path.write_text(json.dumps(adapter), encoding="utf-8")
    compilation_queue = tmp_path / "episode-compilation-queue"
    (compilation_queue / "results").mkdir(parents=True)
    compilation = {
        "schema_version": "task_evaluation_episode_compilation_result.v1",
        "status": "compiled_for_production_launch",
        "compilation_id": compilation_id,
        "source_commit": preparation["expected_production_commit"],
        "configured_scene_revision_digest": preparation["task"][
            "configured_scene_revision_digest"
        ],
        "adapter_result_path": str(adapter_path),
        "adapter_result_digest": adapter["result_digest"],
        "result_digest": "",
    }
    compilation["result_digest"] = canonical_digest(
        compilation, digest_field="result_digest"
    )
    write_launch_preparation_record_exclusive(
        compilation_queue
        / "results"
        / f"{compilation_id}-{envelope_digest.removeprefix('sha256:')}.json",
        compilation,
    )
    activation = {
        "preparation": {
            "preparation_id": preparation["preparation_id"],
            "request_digest": launch_preparation_request_digest(preparation),
            "result_digest": result["result_digest"],
        },
        "team_namespace": preparation["team_namespace"],
        "expected_production_commit": preparation[
            "expected_production_commit"
        ],
    }

    loaded_request, _loaded_result, loaded_adapter, _references = (
        worker._load_verified_preparation(
            activation_request=activation,
            preparation_queue_root=queue,
            preparation_input_root=input_root,
            episode_compilation_queue_root=compilation_queue,
            episode_compilation_output_root=compiled_output,
        )
    )

    assert loaded_request["preparation_id"] == compilation_id
    assert loaded_adapter["result_digest"] == adapter["result_digest"]


def _stage_scene_configuration_preparation(tmp_path: Path):
    preparation, payloads = production_request_with_fetchable_bytes()
    preparation_queue = tmp_path / "preparation-queue"
    input_root = tmp_path / "inputs"
    construction_queue = tmp_path / "construction-queue"
    stage_launch_preparation_request(
        value=preparation,
        queue_root=preparation_queue,
        submitted_by="blueprint-webapp",
    )
    prepared = process_launch_preparation_queue(
        queue_root=preparation_queue,
        input_root=input_root,
        allowed_uri_prefixes=["s3://blueprint-production-inputs/"],
        service_account=SERVICE_ACCOUNT,
        source_commit=preparation["expected_production_commit"],
        fetcher=preparation_fetcher(payloads),
        scene_render_input_materializer=fake_scene_render_inputs,
        construction_queue_root=construction_queue,
    )
    result = prepared["results"][0]
    assert result["status"] == "queued_for_production_scene_configuration"
    activation = {
        "preparation": {
            "preparation_id": preparation["preparation_id"],
            "request_digest": launch_preparation_request_digest(preparation),
            "result_digest": result["result_digest"],
        },
        "team_namespace": preparation["team_namespace"],
        "expected_production_commit": preparation[
            "expected_production_commit"
        ],
    }
    return (
        preparation,
        payloads,
        preparation_queue,
        input_root,
        construction_queue,
        result,
        activation,
    )


def test_activation_consumes_scene_construction_envelope_without_codex_handoff(
    tmp_path: Path,
) -> None:
    (
        _preparation,
        _payloads,
        preparation_queue,
        input_root,
        construction_queue,
        result,
        activation,
    ) = _stage_scene_configuration_preparation(tmp_path)

    loaded_request, loaded_result, construction, references = (
        worker._load_verified_preparation(
            activation_request=activation,
            preparation_queue_root=preparation_queue,
            preparation_input_root=input_root,
            scene_construction_queue_root=construction_queue,
        )
    )

    assert loaded_request["run_mode"] == "scene_configuration"
    assert loaded_result["result_digest"] == result["result_digest"]
    assert construction["kind"] == "task_evaluation_scene_configuration"
    assert construction["construction_envelope_digest"] == result[
        "construction_queue_envelope_digest"
    ]
    assert "construction.recipe.stage_sequence.0.configuration" in references


@pytest.mark.parametrize("terminal_state", ["blocked", "completed"])
def test_activation_refuses_terminal_scene_construction_before_side_effects(
    tmp_path: Path,
    terminal_state: str,
) -> None:
    (
        preparation,
        payloads,
        preparation_queue,
        input_root,
        construction_queue,
        preparation_result,
        _activation_binding,
    ) = _stage_scene_configuration_preparation(tmp_path)
    construction_envelope = next((construction_queue / "pending").glob("*.json"))
    os.replace(
        construction_envelope,
        construction_queue / terminal_state / construction_envelope.name,
    )

    request = activation_request(lane="task_evaluation_scene_configuration")
    request["activation_id"] = (
        f"activation-scene-configuration-terminal-{terminal_state}"
    )
    request["preparation"] = {
        "preparation_id": preparation["preparation_id"],
        "request_digest": launch_preparation_request_digest(preparation),
        "result_digest": preparation_result["result_digest"],
    }
    request["team_namespace"] = preparation["team_namespace"]
    request["expected_production_commit"] = preparation[
        "expected_production_commit"
    ]
    now = datetime.now(timezone.utc)
    request["authorization"]["authorized_on"] = now.isoformat()
    request["authorization"]["standing_authorization_expires_at"] = (
        now + timedelta(hours=1)
    ).isoformat()
    for name, reference in request["lineage"].items():
        if name == "kind":
            continue
        content = json.dumps({"kind": name}).encode()
        reference.update(_reference(str(reference["uri"]), content))
        payloads[str(reference["uri"])] = content
    window_bytes = _release_window(request, now)
    request["release_window"] = _reference(
        "s3://blueprint-production-inputs/coordinator-release-windows/"
        "release-window.json",
        window_bytes,
    )
    payloads[request["release_window"]["uri"]] = window_bytes

    activation_queue = tmp_path / "activation-queue"
    stage_launch_activation_request(
        value=request,
        queue_root=activation_queue,
        submitted_by="blueprint-webapp",
    )
    profile_dir = tmp_path / "profiles"
    catalog_path = tmp_path / "catalog.json"
    authorization_dir = tmp_path / "standing-authorizations"
    fetch_called = False
    preparer_called = False

    def forbidden_fetch(
        _uri: str, _destination: Path, _maximum_bytes: int
    ) -> None:
        nonlocal fetch_called
        fetch_called = True
        raise AssertionError("terminal construction must block before fetch")

    def forbidden_preparer(**_kwargs):
        nonlocal preparer_called
        preparer_called = True
        raise AssertionError("terminal construction must block before publication")

    run = process_launch_activation_queue(
        queue_root=activation_queue,
        preparation_queue_root=preparation_queue,
        preparation_input_root=input_root,
        activation_root=tmp_path / "activations",
        allowed_uri_prefixes=["s3://blueprint-production-inputs/"],
        service_account=SERVICE_ACCOUNT,
        service_group=SERVICE_ACCOUNT,
        repository_root=tmp_path,
        destination_prefix="s3://blueprint-production-inputs/activated",
        release_window_prefix=(
            "s3://blueprint-production-inputs/coordinator-release-windows/"
        ),
        profile_dir=profile_dir,
        webapp_catalog=catalog_path,
        standing_authorization_dir=authorization_dir,
        scene_construction_queue_root=construction_queue,
        scene_configuration_toolchain_root=scene_configuration_toolchain(
            tmp_path / "toolchain",
            preparation["expected_production_commit"],
        ),
        source_commit=preparation["expected_production_commit"],
        fetcher=forbidden_fetch,
        preparer=forbidden_preparer,
    )

    assert run["results"][0]["status"] == "blocked"
    assert run["results"][0]["blockers"] == [
        "launch_activation_scene_construction_queue_state_invalid:"
        f"{terminal_state}"
    ]
    assert fetch_called is False
    assert preparer_called is False
    assert not profile_dir.exists()
    assert not catalog_path.exists()
    assert not authorization_dir.exists()


def test_activation_builds_robot_neutral_scene_configuration_context(
    tmp_path: Path,
) -> None:
    preparation = test_configuration_request()
    activation = activation_request(
        lane="task_evaluation_scene_configuration"
    )
    activation["team_namespace"] = preparation["team_namespace"]
    activation["expected_production_commit"] = preparation[
        "expected_production_commit"
    ]
    envelope = tmp_path / "construction-envelope.json"
    envelope.write_text('{"schema_version":"fixture.v1"}\n', encoding="utf-8")
    envelope.chmod(0o440)
    toolchain = scene_configuration_toolchain(
        tmp_path / "toolchain",
        preparation["expected_production_commit"],
    )
    project = tmp_path / "project-spend.json"
    project.write_text("{}\n", encoding="utf-8")
    provider_zero = tmp_path / "provider-zero.json"
    provider_zero.write_text("{}\n", encoding="utf-8")

    context = worker._build_scene_configuration_context(
        activation_request=activation,
        preparation_request=preparation,
        construction_input={
            "kind": "task_evaluation_scene_configuration",
            "construction_envelope_path": str(envelope),
            "construction_envelope_digest": "sha256:" + "e" * 64,
            "recipe_digest": "sha256:" + "f" * 64,
            "source_commit": preparation["expected_production_commit"],
        },
        activation_materialized={
            "lineage.project_spend_reconciliation": project,
            "lineage.initial_provider_zero": provider_zero,
        },
        activation_root=tmp_path / "activation",
        repository_root=tmp_path,
        toolchain_root=toolchain,
        destination_prefix="s3://blueprint-production-inputs/activated",
        profile_dir=tmp_path / "profiles",
        webapp_catalog=tmp_path / "catalog.json",
        standing_authorization_dir=tmp_path / "authorizations",
        service_account=SERVICE_ACCOUNT,
        service_group=SERVICE_ACCOUNT,
        configured_controls_autostart_intent_root=(
            _configured_controls_intent_root(tmp_path, preparation)
        ),
    )

    assert context["lane"] == "task_evaluation_scene_configuration"
    assert context["team_namespace"] == preparation["team_namespace"]
    assert context["operations"]["construction_envelope"] == str(
        envelope.resolve()
    )
    assert context["operations"]["container_image"] == preparation[
        "runtime"
    ]["oci_image"]
    assert "robot" not in context
    assert context["reference_bindings"][
        "raw_interiorgs_bytes_authorized_for_provider"
    ] is False
    decision = {
        "schema_version": "task_evaluation_scene_configuration_disclosure_decision.v1",
        "render_execution_site": "provider_gpu",
        "source_appearance_bytes_to_provider": True,
        "decision_digest": "",
    }
    decision["decision_digest"] = canonical_digest(
        decision, digest_field="decision_digest"
    )
    envelope.chmod(0o640)
    envelope.write_text(
        json.dumps(
            {
                "schema_version": "fixture.v1",
                "render_inputs_result": {"disclosure_decision": decision},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    envelope.chmod(0o440)
    provider_context = worker._build_scene_configuration_context(
        activation_request=activation,
        preparation_request=preparation,
        construction_input={
            "kind": "task_evaluation_scene_configuration",
            "construction_envelope_path": str(envelope),
            "construction_envelope_digest": "sha256:" + "e" * 64,
            "recipe_digest": "sha256:" + "f" * 64,
            "source_commit": preparation["expected_production_commit"],
        },
        activation_materialized={
            "lineage.project_spend_reconciliation": project,
            "lineage.initial_provider_zero": provider_zero,
        },
        activation_root=tmp_path / "provider-activation",
        repository_root=tmp_path,
        toolchain_root=toolchain,
        destination_prefix="s3://blueprint-production-inputs/activated",
        profile_dir=tmp_path / "profiles",
        webapp_catalog=tmp_path / "catalog.json",
        standing_authorization_dir=tmp_path / "authorizations",
        service_account=SERVICE_ACCOUNT,
        service_group=SERVICE_ACCOUNT,
        configured_controls_autostart_intent_root=(
            _configured_controls_intent_root(tmp_path, preparation)
        ),
    )
    assert provider_context["reference_bindings"][
        "raw_interiorgs_bytes_authorized_for_provider"
    ] is True
    assert provider_context["reference_bindings"][
        "provider_disclosure_decision_digest"
    ] == decision["decision_digest"]


def _bytes_digest(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _reference(uri: str, payload: bytes) -> dict[str, object]:
    return {"uri": uri, "digest": _bytes_digest(payload), "size_bytes": len(payload)}


def _sealed_claim(schema: str, status: str, scene_id: str, field: str) -> bytes:
    value = {
        "schema_version": schema,
        "status": status,
        "scene_id": scene_id,
        field: "",
    }
    value[field] = canonical_digest(value, digest_field=field)
    return json.dumps(value, sort_keys=True).encode()


def _stage_verified_preparation(tmp_path: Path):
    request = preparation_request()
    request["preparation_id"] = "preparation-scene-841007-v1"
    request["run_id"] = "run-scene-841007-v1"
    payloads: dict[str, bytes] = {}

    def replace(node) -> None:
        if isinstance(node, dict):
            if set(node) == {"uri", "digest", "size_bytes"}:
                payload = ("payload:" + str(node["uri"])).encode()
                node.update(_reference(str(node["uri"]), payload))
                payloads[str(node["uri"])] = payload
                return
            for child in node.values():
                replace(child)
        elif isinstance(node, list):
            for child in node:
                replace(child)

    replace(request)
    scene_id = request["scene"]["identity"]["id"]
    configured = revision()
    configured["configuration_run_id"] = "configuration-scene-841007-v1"
    configured["team_namespace"] = request["team_namespace"]
    configured["scene_identity"] = request["scene"]["identity"]
    configured["source_commit"] = request["expected_production_commit"]
    configured["task_template"]["identity"] = request["task"]["identity"]
    configured["replacement"]["identity"] = request["task"]["subject"][
        "identity"
    ]
    source_bytes = _sealed_claim(
        "task_evaluation_scene_source_manifest.v1",
        "retained",
        scene_id,
        "source_manifest_digest",
    )
    rights_bytes = _sealed_claim(
        "task_evaluation_scene_rights_admission.v1",
        "admitted",
        scene_id,
        "rights_admission_digest",
    )
    for field, payload in (("manifest", source_bytes), ("rights_admission", rights_bytes)):
        reference = configured["source"][field]
        payloads[str(reference["uri"])] = payload
        reference.update(_reference(str(reference["uri"]), payload))
    for index, evidence in enumerate(configured["source"]["rights_evidence"]):
        payload = f"rights-evidence:{index}:{evidence['role']}".encode()
        reference = evidence["artifact"]
        payloads[str(reference["uri"])] = payload
        reference.update(_reference(str(reference["uri"]), payload))
    for label, reference in (
        *[
            (f"registration-{name}", row)
            for name, row in configured["registration"].items()
        ],
        ("task-definition", configured["task_template"]["definition"]),
        (
            "task-success-criteria",
            configured["task_template"]["success_criteria"],
        ),
        ("task-execution", configured["task_template"]["execution"]),
    ):
        payload = f"configured-scene:{label}".encode()
        payloads[str(reference["uri"])] = payload
        reference.update(_reference(str(reference["uri"]), payload))
    bundle_payload = b"configured-scene-bundle"
    configured["configured_scene_bundle"].update(
        _reference(
            str(configured["configured_scene_bundle"]["uri"]),
            bundle_payload,
        )
    )
    payloads[str(configured["configured_scene_bundle"]["uri"])] = bundle_payload
    configured["revision_digest"] = canonical_digest(
        configured, digest_field="revision_digest"
    )
    configured_bytes = json.dumps(configured, sort_keys=True).encode()
    request["scene"]["configured_revision"].update(
        _reference(
            str(request["scene"]["configured_revision"]["uri"]),
            configured_bytes,
        )
    )
    request["task"]["configured_scene_revision_digest"] = configured[
        "revision_digest"
    ]
    payloads[str(request["scene"]["configured_revision"]["uri"])] = (
        configured_bytes
    )

    queue = tmp_path / "preparation-queue"
    intake = stage_launch_preparation_request(
        value=request, queue_root=queue, submitted_by="blueprint-webapp"
    )
    filename = (
        f"{request['preparation_id']}-"
        f"{intake['request_digest'].removeprefix('sha256:')}.json"
    )
    os.replace(queue / "pending" / filename, queue / "materialized" / filename)

    input_root = tmp_path / "preparation-inputs"
    preparation_root = input_root / request["preparation_id"]
    preparation_root.mkdir(parents=True)
    reference_rows: list[dict[str, object]] = []

    def materialize(node, path: tuple[str, ...] = ()) -> None:
        if isinstance(node, dict):
            if set(node) == {"uri", "digest", "size_bytes"}:
                target = preparation_root / str(node["digest"]).removeprefix("sha256:")
                target.write_bytes(payloads[str(node["uri"])])
                reference_rows.append({
                    "contract_path": ".".join(path),
                    "uri": node["uri"],
                    "digest": node["digest"],
                    "size_bytes": node["size_bytes"],
                    "materialized_path": str(target),
                    "full_byte_service_account_readback_passed": True,
                })
                return
            for key, child in node.items():
                materialize(child, (*path, str(key)))
        elif isinstance(node, list):
            for index, child in enumerate(node):
                materialize(child, (*path, str(index)))

    materialize(request)
    transitive_references = [
        (
            "scene.configured_revision.configured_scene_bundle",
            configured["configured_scene_bundle"],
        ),
        (
            "scene.configured_revision.source.manifest",
            configured["source"]["manifest"],
        ),
        (
            "scene.configured_revision.source.rights_admission",
            configured["source"]["rights_admission"],
        ),
        *[
            (
                "scene.configured_revision.source.rights_evidence."
                f"{index}.artifact",
                evidence["artifact"],
            )
            for index, evidence in enumerate(
                configured["source"]["rights_evidence"]
            )
        ],
        *[
            (
                f"scene.configured_revision.registration.{name}",
                reference,
            )
            for name, reference in configured["registration"].items()
        ],
        (
            "scene.configured_revision.task_template.definition",
            configured["task_template"]["definition"],
        ),
        (
            "scene.configured_revision.task_template.success_criteria",
            configured["task_template"]["success_criteria"],
        ),
        (
            "scene.configured_revision.task_template.execution",
            configured["task_template"]["execution"],
        ),
    ]
    for contract_path, reference in transitive_references:
        target = preparation_root / str(reference["digest"]).removeprefix(
            "sha256:"
        )
        target.write_bytes(payloads[str(reference["uri"])])
        reference_rows.append(
            {
                "contract_path": contract_path,
                "uri": reference["uri"],
                "digest": reference["digest"],
                "size_bytes": reference["size_bytes"],
                "materialized_path": str(target),
                "full_byte_service_account_readback_passed": True,
            }
        )
    adapter_root = input_root / request["preparation_id"] / "native-arena-adapter"
    packet_root = adapter_root / "construction-packet"
    runtime_root = adapter_root / "runtime-source"
    packet_root.mkdir(parents=True)
    runtime_root.mkdir()
    robot = {"robot_id": request["robot"]["identity"]["id"], "joint_count": 7}
    (packet_root / "native_task_runtime_contract.v1.json").write_text(
        json.dumps({"task_spec_digest": "sha256:" + "8" * 64, "robot": robot}),
        encoding="utf-8",
    )
    runtime_receipt = runtime_root / "native_task_runtime_source_packet.v1.json"
    runtime_receipt.write_text("{}\n", encoding="utf-8")
    adapter = {
        "schema_version": "task_evaluation_native_arena_adapter_result.v1",
        "status": "native_arena_adapter_materialized",
        "preparation_id": request["preparation_id"],
        "source_commit": request["expected_production_commit"],
        "packet_receipt_digest": "sha256:" + "6" * 64,
        "runtime_source_receipt_digest": "sha256:" + "7" * 64,
        "packet_root": str(packet_root),
        "runtime_source_receipt": str(runtime_receipt),
        "result_digest": "",
    }
    adapter["result_digest"] = canonical_digest(adapter, digest_field="result_digest")
    write_launch_preparation_record_exclusive(
        adapter_root / "task_evaluation_native_arena_adapter_result.v1.json", adapter
    )
    result = {
        "schema_version": "task_evaluation_launch_preparation_result.v1",
        "status": "native_arena_inputs_verified_awaiting_profile_authority",
        "preparation_id": request["preparation_id"],
        "run_id": request["run_id"],
        "team_namespace": request["team_namespace"],
        "source_commit": request["expected_production_commit"],
        "full_byte_service_account_readback_passed": True,
        "references": reference_rows,
        "adapter_result_digest": adapter["result_digest"],
        "provider_mutation_performed": False,
        "catalog_mutation_performed": False,
        "paid_execution_requested": False,
        "result_digest": "",
    }
    result["result_digest"] = canonical_digest(result, digest_field="result_digest")
    (queue / "results").mkdir()
    write_launch_preparation_record_exclusive(queue / "results" / filename, result)
    assert intake["request_digest"] == launch_preparation_request_digest(request)
    return request, result, payloads, queue, input_root


def _release_window(request: dict[str, object], now: datetime) -> bytes:
    value = {
        "schema_version": "task_evaluation_shared_mutation_window.v1",
        "status": "released",
        "window_id": "window-scene-841007-001",
        "activation_id": request["activation_id"],
        "activation_intent_digest": launch_activation_intent_digest(request),
        "team_namespace": request["team_namespace"],
        "expected_production_commit": request["expected_production_commit"],
        "allowed_mutations": [
            "profile_publication",
            "catalog_synchronization",
            "standing_authorization",
        ],
        "provider_allowlist": ["vast"],
        "maximum_hard_cap_usd": 1.0,
        "issued_at": (now - timedelta(minutes=1)).isoformat(),
        "expires_at": (now + timedelta(minutes=10)).isoformat(),
        "released_by": "policy-lead-001",
        "release_reference": "coordinated release window",
        "provider_resource_allocation_allowed": False,
        "paid_request_allowed": False,
        "window_digest": "",
    }
    value["window_digest"] = canonical_digest(value, digest_field="window_digest")
    return json.dumps(value, sort_keys=True).encode()


@pytest.mark.parametrize(
    "lane",
    ["native_task_arena_construction", "native_task_arena_controls"],
)
def test_worker_cross_binds_preparation_window_and_no_execution_publication(
    tmp_path: Path, lane: str
) -> None:
    preparation, preparation_result, payloads, preparation_queue, input_root = (
        _stage_verified_preparation(tmp_path)
    )
    request = activation_request(lane=lane)
    request["activation_id"] = f"activation-scene-841007-{lane}"
    request["preparation"] = {
        "preparation_id": preparation["preparation_id"],
        "request_digest": launch_preparation_request_digest(preparation),
        "result_digest": preparation_result["result_digest"],
    }
    request["authorization"]["authorized_on"] = datetime.now(timezone.utc).isoformat()
    request["authorization"]["standing_authorization_expires_at"] = (
        datetime.now(timezone.utc) + timedelta(hours=1)
    ).isoformat()
    for name, reference in request["lineage"].items():
        if name == "kind":
            continue
        content = json.dumps({"kind": name}).encode()
        reference.update(_reference(str(reference["uri"]), content))
        payloads[str(reference["uri"])] = content
    window_bytes = _release_window(request, datetime.now(timezone.utc))
    request["release_window"] = _reference(
        "s3://blueprint-production-inputs/coordinator-release-windows/release-window.json",
        window_bytes,
    )
    payloads[request["release_window"]["uri"]] = window_bytes

    queue = tmp_path / "activation-queue"
    stage_launch_activation_request(
        value=request, queue_root=queue, submitted_by="blueprint-webapp"
    )
    profile_dir = tmp_path / "profiles"
    authorization_dir = tmp_path / "standing-authorizations"

    def fetch(uri: str, destination: Path, maximum_bytes: int) -> None:
        assert len(payloads[uri]) == maximum_bytes
        destination.write_bytes(payloads[uri])

    observed_context: dict[str, object] = {}

    def prepare(**kwargs) -> dict[str, object]:
        context = json.loads(Path(kwargs["context_path"]).read_text())
        observed_context.update(context)
        set_root = Path(context["operations"]["set_root"])
        set_root.mkdir(parents=True)
        profile = {
            "profile_id": "scene-841007-construction-r1",
            "profile_digest": "sha256:" + "d" * 64,
        }
        profile_path = set_root / "live_profile-r1.v1.json"
        profile_path.write_text(json.dumps(profile), encoding="utf-8")
        publication_path = set_root / "profile_publication_receipt.v1.json"
        publication_path.write_text('{"status":"published"}\n', encoding="utf-8")
        authorization_dir.mkdir(parents=True)
        authorization_path = authorization_dir / f"{profile['profile_id']}.json"
        authorization_path.write_text(
            json.dumps({
                "profile_id": profile["profile_id"],
                "profile_digest": profile["profile_digest"],
                "max_launches": 1,
                "max_total_spend_usd": preparation["spend"]["hard_cap_usd"],
                "provider_mutation_performed": False,
            }),
            encoding="utf-8",
        )
        steps = []
        for step_id, path in (
            ("live_profile", profile_path),
            ("profile_publication", publication_path),
            ("standing_authorization", authorization_path),
        ):
            steps.append({
                "step_id": step_id,
                "artifact_path": str(path),
                "artifact_sha256": _bytes_digest(path.read_bytes()),
            })
        receipt = {
            "schema_version": "paid_lane_launch_preparation.v1",
            "status": "prepared",
            "source_commit": request["expected_production_commit"],
            "completed_steps": steps,
            "provider_allocation_performed": False,
            "paid_inference_performed": False,
        }
        Path(kwargs["receipt_path"]).write_text(json.dumps(receipt), encoding="utf-8")
        return receipt

    run = process_launch_activation_queue(
        queue_root=queue,
        preparation_queue_root=preparation_queue,
        preparation_input_root=input_root,
        activation_root=tmp_path / "activations",
        allowed_uri_prefixes=["s3://blueprint-production-inputs/"],
        service_account=SERVICE_ACCOUNT,
        service_group=pwd.getpwuid(os.geteuid()).pw_name,
        repository_root=tmp_path,
        destination_prefix="s3://blueprint-production-inputs/activated",
        release_window_prefix=(
            "s3://blueprint-production-inputs/coordinator-release-windows/"
        ),
        profile_dir=profile_dir,
        webapp_catalog=tmp_path / "catalog.json",
        standing_authorization_dir=authorization_dir,
        source_commit=request["expected_production_commit"],
        fetcher=fetch,
        preparer=prepare,
    )

    assert run["status"] == "processed"
    assert run["results"][0]["status"] == "profile_authority_materialized_no_execution", json.dumps(run, sort_keys=True)
    assert run["results"][0]["provider_mutation_performed"] is False
    assert run["results"][0]["paid_execution_requested"] is False
    assert observed_context["lane"] == lane
    if lane == "native_task_arena_controls":
        assert request["lineage"]["kind"] == "predecessor"
        for name in (
            "prior_authority",
            "prior_result",
            "prior_launch_receipt",
            "prior_webapp_sync",
            "prior_provider_zero",
            "prior_spend_reconciliation",
            "construction_result",
        ):
            assert Path(observed_context["operations"][name]).is_file()
    assert observed_context["operations"]["provider"] == "vast"
    assert observed_context["operations"]["maximum_hourly_rate_usd"] == (
        preparation["spend"]["maximum_hourly_rate_usd"]
    )
    assert observed_context["operations"]["hard_total_spend_cap_usd"] == (
        preparation["spend"]["hard_cap_usd"]
    )
    assert observed_context["operations"]["hard_ttl_seconds"] == (
        preparation["spend"]["hard_ttl_seconds"]
    )
    assert observed_context["references"]["scene"]["scene_id"] == (
        preparation["scene"]["identity"]["id"]
    )
    status = launch_activation_status(
        activation_id=request["activation_id"], queue_root=queue
    )
    assert status["status"] == "prepared"
    assert status["profile_id"] == "scene-841007-construction-r1"
    assert status["provider_mutation_performed_by_worker"] is False
    assert status["paid_execution_requested"] is False
    authorization = json.loads(
        next(authorization_dir.glob("*.json")).read_text(encoding="utf-8")
    )
    assert authorization["max_launches"] == 1
    assert authorization["max_total_spend_usd"] == preparation["spend"][
        "hard_cap_usd"
    ]


def test_worker_blocks_wrong_window_before_preparer(tmp_path) -> None:
    preparation, preparation_result, payloads, preparation_queue, input_root = (
        _stage_verified_preparation(tmp_path)
    )
    request = activation_request()
    request["activation_id"] = "activation-scene-841007-construction"
    request["preparation"] = {
        "preparation_id": preparation["preparation_id"],
        "request_digest": launch_preparation_request_digest(preparation),
        "result_digest": preparation_result["result_digest"],
    }
    for name, reference in request["lineage"].items():
        if name != "kind":
            content = json.dumps({"kind": name}).encode()
            reference.update(_reference(str(reference["uri"]), content))
            payloads[str(reference["uri"])] = content
    window_bytes = _release_window(request, datetime.now(timezone.utc))
    window = json.loads(window_bytes)
    window["activation_id"] = "different-activation"
    window["window_digest"] = canonical_digest(window, digest_field="window_digest")
    window_bytes = json.dumps(window, sort_keys=True).encode()
    request["release_window"] = _reference(
        "s3://blueprint-production-inputs/coordinator-release-windows/release-window.json",
        window_bytes,
    )
    payloads[request["release_window"]["uri"]] = window_bytes
    queue = tmp_path / "activation-queue"
    stage_launch_activation_request(
        value=request, queue_root=queue, submitted_by="blueprint-webapp"
    )

    def fetch(uri: str, destination: Path, maximum_bytes: int) -> None:
        destination.write_bytes(payloads[uri])

    def forbidden_preparer(**_kwargs):
        raise AssertionError("invalid release window must block before mutation")

    run = process_launch_activation_queue(
        queue_root=queue,
        preparation_queue_root=preparation_queue,
        preparation_input_root=input_root,
        activation_root=tmp_path / "activations",
        allowed_uri_prefixes=["s3://blueprint-production-inputs/"],
        service_account=SERVICE_ACCOUNT,
        service_group=SERVICE_ACCOUNT,
        repository_root=tmp_path,
        destination_prefix="s3://blueprint-production-inputs/activated",
        release_window_prefix=(
            "s3://blueprint-production-inputs/coordinator-release-windows/"
        ),
        profile_dir=tmp_path / "profiles",
        webapp_catalog=tmp_path / "catalog.json",
        standing_authorization_dir=tmp_path / "standing-authorizations",
        source_commit=request["expected_production_commit"],
        fetcher=fetch,
        preparer=forbidden_preparer,
    )
    assert run["results"][0]["status"] == "blocked"
    assert run["results"][0]["blockers"] == [
        "shared_mutation_window_binding_mismatch"
    ]


def _scene_configuration_context(
    tmp_path: Path, *, intent_root: Path
) -> dict[str, object]:
    """Build one scene-configuration activation context against an intent root."""

    preparation = test_configuration_request()
    activation = activation_request(lane="task_evaluation_scene_configuration")
    activation["team_namespace"] = preparation["team_namespace"]
    activation["expected_production_commit"] = preparation[
        "expected_production_commit"
    ]
    envelope = tmp_path / "construction-envelope.json"
    envelope.write_text('{"schema_version":"fixture.v1"}\n', encoding="utf-8")
    envelope.chmod(0o440)
    toolchain = scene_configuration_toolchain(
        tmp_path / "toolchain", preparation["expected_production_commit"]
    )
    project = tmp_path / "project-spend.json"
    project.write_text("{}\n", encoding="utf-8")
    provider_zero = tmp_path / "provider-zero.json"
    provider_zero.write_text("{}\n", encoding="utf-8")
    return worker._build_scene_configuration_context(
        activation_request=activation,
        preparation_request=preparation,
        construction_input={
            "kind": "task_evaluation_scene_configuration",
            "construction_envelope_path": str(envelope),
            "construction_envelope_digest": "sha256:" + "e" * 64,
            "recipe_digest": "sha256:" + "f" * 64,
            "source_commit": preparation["expected_production_commit"],
        },
        activation_materialized={
            "lineage.project_spend_reconciliation": project,
            "lineage.initial_provider_zero": provider_zero,
        },
        activation_root=tmp_path / "activation",
        repository_root=tmp_path,
        toolchain_root=toolchain,
        destination_prefix="s3://blueprint-production-inputs/activated",
        profile_dir=tmp_path / "profiles",
        webapp_catalog=tmp_path / "catalog.json",
        standing_authorization_dir=tmp_path / "authorizations",
        service_account=SERVICE_ACCOUNT,
        service_group=SERVICE_ACCOUNT,
        configured_controls_autostart_intent_root=intent_root,
    )


def test_first_configuration_is_admitted_before_any_controls_continuation(
    tmp_path: Path,
) -> None:
    """A scene with no authorized continuation may still be configured.

    The continuation binds a trajectory plan authored against a published
    configured revision, so demanding it first is an ordering no scene's first
    run can satisfy.
    """

    empty_root = tmp_path / "intents"
    empty_root.mkdir()

    context = _scene_configuration_context(tmp_path, intent_root=empty_root)

    assert context["lane"] == "task_evaluation_scene_configuration"
    assert (
        context["operations"]["configured_controls_autostart_intent_source"] == ""
    )
    assert (
        context["operations"]["configured_controls_autostart_intent_artifacts"]
        == ""
    )


def test_provisioned_continuation_is_still_bound_exactly(tmp_path: Path) -> None:
    """A provisioned intent is bound, so absence is the only relaxed case."""

    preparation = test_configuration_request()
    root = _configured_controls_intent_root(tmp_path, preparation)

    context = _scene_configuration_context(tmp_path, intent_root=root)

    source = context["operations"]["configured_controls_autostart_intent_source"]
    assert source != ""
    assert Path(source).is_file()
    assert context["operations"][
        "configured_controls_autostart_intent_artifacts"
    ].endswith("task_evaluation_configured_controls_autostart_intent.v1.json")


def test_present_but_foreign_continuation_still_refuses(tmp_path: Path) -> None:
    """Only an absent intent is admitted; a malformed one stays fail-closed."""

    preparation = test_configuration_request()
    root = _configured_controls_intent_root(tmp_path, preparation)
    registry = root / configured_controls_autostart_registry_name(
        team_namespace=str(preparation["team_namespace"]),
        scene_id=str(preparation["scene"]["identity"]["id"]),
        task_id=str(preparation["task"]["identity"]["id"]),
    )
    registry.chmod(0o640)
    registry.write_text('{"schema_version":"not-an-intent"}\n', encoding="utf-8")

    with pytest.raises(
        (
            worker.TaskEvaluationLaunchActivationWorkerError,
            TaskEvaluationConfiguredControlsAutostartError,
        )
    ) as excinfo:
        _scene_configuration_context(tmp_path, intent_root=root)

    assert "configured_controls_autostart_intent_invalid" in str(excinfo.value)
