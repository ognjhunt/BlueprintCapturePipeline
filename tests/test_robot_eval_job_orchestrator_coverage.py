from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from tests.runpy_entrypoint import run_module_as_main

from blueprint_pipeline import robot_eval_job_orchestrator as rejo


pytestmark = pytest.mark.slow


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _minimal_capture_root(tmp_path: Path) -> Path:
    capture_root = tmp_path / "local-blueprint" / "scenes" / "scene-1" / "captures" / "capture-1"
    (capture_root / "pipeline").mkdir(parents=True, exist_ok=True)
    _write_json(
        capture_root / "capture_descriptor.json",
        {
            "schema_version": "blueprint_capture_descriptor.v1",
            "scene_id": "scene-1",
            "capture_id": "capture-1",
            "artifacts": [],
        },
    )
    return capture_root


def _write_minimal_robot_eval_inputs(capture_root: Path) -> None:
    dataset = capture_root / "pipeline" / "robot_eval_dataset"
    _write_json(
        dataset / "site_card.json",
        {
            "scene_id": "scene-1",
            "capture_id": "capture-1",
            "provenance_rights_review_status": {"rights_privacy": {"blocked": False}},
        },
    )
    _write_json(
        dataset / "task_cards.json",
        {
            "cards": [{"task_id": "task-1", "task_statement": "Move the item"}],
            "task_card_count": 1,
        },
    )
    _write_json(
        dataset / "scenario_cards.json",
        {
            "cards": [{"task_id": "task-1", "scenario_id": "scenario-1"}],
            "scenario_card_count": 1,
        },
    )
    _write_json(dataset / "eval_cards.json", {"cards": [{"task_id": "task-1"}]})
    _write_json(dataset / "proof_boundaries.json", {"claim_boundary": "local_test_fixture"})


def test_robot_eval_job_small_helper_and_policy_edges(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rejo.importlib.util, "find_spec", lambda name: object() if name == "present" else None)
    assert rejo._module_available(["missing", "present"])
    assert "job" in rejo._agents_sdk_robot_eval_job_prompt({"job_id": "job"})
    monkeypatch.setenv("ROBOT_EVAL_TEST_BOOL", "yes")
    assert rejo._env_truthy("ROBOT_EVAL_TEST_BOOL")
    assert rejo._number("bad", 3.0) == 3.0
    assert rejo._number_field({"a": "", "b": "4"}, "a", "b") == 4.0
    assert rejo._string_list(None) == []
    assert rejo._string_list("a") == ["a"]
    assert rejo._string_list(7) == ["7"]
    assert rejo._first_allowed_backend(["bad"], ["pybullet"]) == "pybullet"
    assert rejo._first_allowed_backend([], []) == "mujoco"
    assert rejo._first_allowed_backend([], ["unsupported"]) == "fixture"
    assert rejo._simulator_role("newton", "mujoco", {}) == "operator_selected_backend"
    assert rejo._boolish(True)
    assert not rejo._boolish(False)

    existing = tmp_path / "file.json"
    _write_json(existing, {"ok": True})
    assert rejo._relative_if_file(tmp_path, existing) == "file.json"
    assert rejo._relative_if_file(tmp_path, tmp_path / "missing.json") is None
    assert rejo._read_optional_mapping(tmp_path / "missing.json") == {}
    bad_payload = tmp_path / "bad.json"
    bad_payload.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="Expected job request"):
        rejo._read_job_request(bad_payload)
    envelope = {"queue_contract": "robot_eval_job_request_inbox.v1", "job_request": {"job_id": "j"}}
    assert rejo._read_job_request(envelope)["job_id"] == "j"

    selection = rejo.resolve_simulator_selection_policy(
        {
            "execution_request": {
                "simulator_routing": {
                    "allowed_backends": ["fixture"],
                    "requested_backend": "isaac_sim",
                    "selection_policy": {"required_proof_classes": ["photoreal_contact_dynamics"]},
                }
            }
        },
        selected_simulator="mujoco",
    )
    assert "selected_simulator_not_allowed_by_request_policy" in selection["non_blocking_warnings"]
    assert rejo.resolve_simulator_selection_policy({"simulator_preference": "pybullet"}, selected_simulator="pybullet")[
        "recommended_backend"
    ] == "pybullet"

    policy_package = rejo._policy_package_from_payload(
        {"policyApiEndpoint": {"endpointUrl": "https://example.test/policy"}}
    )
    assert policy_package["policy_api_endpoint"]["endpointUrl"].startswith("https://")
    for modality in rejo.POLICY_MODALITY_ORDER:
        status, missing = rejo._validate_policy_modality(modality=modality, payload={"bad": "payload"})
        assert status in {"blocked", "reference_present_requires_owner_system_review"}
        if modality != "high_level_skill_trace":
            assert missing
    manifest, missing_inputs, statuses = rejo._policy_package_manifest(request={}, generated_at="now")
    assert manifest["status"] == "blocked"
    assert "policy_package.one_supported_modality" in missing_inputs
    assert "needs_robot_team_test_modality" in statuses

    adapter = rejo.AgentsSdkRobotEvalJobAdapter(
        agents_sdk_available=True,
        openai_api_key="sk-test",
        live_env_allowed=True,
        allow_live_operator=True,
    )
    monkeypatch.setattr(
        rejo,
        "run_agents_sdk_operator",
        lambda _config: (_ for _ in ()).throw(RuntimeError("operator down")),
    )
    assert adapter.build_plan(plan_context={"job_id": "job"})["status"] == "operator_failed"
    monkeypatch.setattr(
        rejo,
        "run_agents_sdk_operator",
        lambda _config: (_ for _ in ()).throw(ValueError("bad output")),
    )
    assert "agents_sdk_operator_execution_failed:ValueError" in adapter.build_plan(
        plan_context={"job_id": "job"}
    )["blockers"]

    staged_dir = tmp_path / "capture" / "pipeline" / "robot_eval_inputs" / "job-1"
    _write_json(staged_dir / "policy_package.json", {"job_id": "other"})
    assert rejo._load_staged_policy_package(capture_root=tmp_path / "capture", job_id="job-1") == {}
    _write_json(staged_dir / "policy_package.json", {"job_id": "job-1"})
    assert rejo._load_staged_policy_package(capture_root=tmp_path / "capture", job_id="job-1") == {}
    _write_json(
        staged_dir / "policy_package.json",
        {"job_id": "job-1", "policy_package": {"high_level_skill_trace": {"ordered_skill_sequence": ["walk"]}}},
    )
    request = {"policy_package": {"high_level_skill_trace": {"ordered_skill_sequence": ["keep"]}}, "external_input_sources": {}}
    assert rejo._apply_staged_policy_package(request=request, capture_root=tmp_path / "capture", job_id="job-1")[
        "policy_package"
    ]["high_level_skill_trace"]["ordered_skill_sequence"] == ["keep"]


def test_robot_eval_job_cards_validation_scheduler_and_worker_edges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = tmp_path / "pipeline"
    dataset = pipeline / "robot_eval_dataset"
    _write_json(dataset / "task_cards.json", {"cards": [], "task_card_count": 0})
    _write_json(dataset / "scenario_cards.json", {"cards": [{"task_id": "", "scenario_id": ""}], "scenario_card_count": 1})
    assert "robot_eval_task_cards_empty" in rejo._empty_robot_eval_card_inputs(pipeline)
    assert rejo._empty_robot_eval_card_inputs(tmp_path / "missing-pipeline") == []
    assert rejo._card_rows(dataset / "missing.json") == []
    assert rejo._default_requested_tasks_from_cards(pipeline) == []
    _write_json(dataset / "task_cards.json", {"cards": [{}], "task_card_count": 1})
    assert rejo._default_requested_tasks_from_cards(pipeline) == []
    assert rejo._default_robot_profile_from_cards(pipeline)["robot_profile_id"] == "unitree_g1"
    monkeypatch.setattr(rejo, "build_real_site_robot_eval_dataset", lambda **_kwargs: None)
    assert rejo._ensure_robot_eval_cards(capture_root=tmp_path, pipeline_dir=pipeline)

    blocked_request = {
        "operation": "bad",
        "rights_privacy_scope": {"external_use_allowed": "no"},
    }
    validation = rejo._job_validation(
        request=blocked_request,
        policy_missing_inputs=["policy"],
        policy_missing_statuses=["missing_policy"],
        missing_robot_eval_inputs=["cards"],
        generated_at="now",
        pipeline_dir=pipeline,
    )
    assert validation["missing_inputs"] == ["rights_privacy_clearance"]

    scheduler = rejo._build_scheduler_decision(
        request={
            "operation": "evaluate_only",
            "budget": {"budgetUsd": 5},
            "execution_request": {
                "webapp_role": "bad",
                "scheduler_owner": "other",
                "preflight": {"cpu_preflight_required_before_gpu": False},
                "gpu_allocation": {
                    "allocation_allowed_by_webapp": True,
                    "gpu_spend_approved": True,
                    "hard_timeout_seconds": 30,
                },
                "artifact_contract": {"public_claim_upgrade_allowed": True},
                "simulator_routing": {"allowed_backends": ["mujoco"]},
            },
        },
        job_id="job",
        provisioner="runpod",
        simulator="isaac_sim",
        pipeline_dir=pipeline,
        cpu_preflight={"status": "blocked"},
        budget_usd=None,
        timeout_seconds=30,
        generated_at="now",
    )
    assert "execution_request_webapp_role_not_queue_only" in scheduler["blockers"]
    assert "scheduler_selected_simulator_not_allowed_by_execution_request" in scheduler["blockers"]

    assert rejo._execution_worker_profile("newton")["worker_image_family"] == "newton-eval-worker"
    assert rejo._provider_credential_env_vars("gcp") == ["GOOGLE_APPLICATION_CREDENTIALS"]
    assert rejo._provider_launch_operation("gcp") == "create_gcp_gpu_worker_and_run_job"
    assert rejo._provider_launch_operation("docker_local") == "start_local_docker_worker"
    assert rejo._provider_launch_operation("local_process") == "start_local_process_worker"
    assert rejo._configured_worker_image_ref("mujoco")[0] == ""
    assert rejo._worker_image_ref_is_versioned("image@sha256:abc")
    assert not rejo._worker_image_ref_is_versioned("repo/image")
    assert not rejo._worker_image_ref_is_versioned("image:latest")
    assert not rejo._worker_image_ref_is_provider_fetchable("image:candidate-1", versioned=True)
    assert not rejo._worker_manifest_uri_is_fetchable_by_provider(
        "file:///tmp/worker.json", live_gpu_provider=False
    )
    assert not rejo._worker_manifest_uri_is_fetchable_by_provider("file:///tmp/worker.json", live_gpu_provider=True)
    assert rejo._provider_uri_is_fetchable("s3://bucket/input.zip", live_gpu_provider=True)
    assert rejo._provider_uri_is_fetchable("local://input.zip", live_gpu_provider=False)
    assert rejo._provider_artifact_output_uri_is_writable("local://out", live_gpu_provider=False)
    assert rejo._artifact_output_write_auth_contract(
        "local://out", external_provider=True, provider_writable=True
    )["write_auth_contract_ready"] is False
    monkeypatch.setenv("BLUEPRINT_PROVIDER_CACHE_ROOT", "/cache")
    assert rejo._persistent_cache_paths("mujoco")["mujoco_assets"].startswith("/cache")
    assert rejo._provider_gpu_priority_for_simulator("mujoco", {"provider_gpu_priority": ["A10"]}) == ["A10"]
    assert "latency_policy_does_not_justify_idle_cost" in rejo._warm_pool_policy(
        gpu_allocation={"warm_pool": {"enabled": True, "latency_slo_seconds": 100, "estimated_idle_cost_usd_per_hour": 2, "max_idle_cost_usd_per_hour": 1}}
    )["decision_reasons"]
    assert "latency_policy_justifies_idle_cost" in rejo._warm_pool_policy(
        gpu_allocation={"warm_pool": {"enabled": True, "latency_slo_seconds": 20, "estimated_idle_cost_usd_per_hour": 1, "max_idle_cost_usd_per_hour": 2}}
    )["decision_reasons"]
    assert rejo._runtime_preflight_contract(simulator="newton", provisioner="runpod", worker_profile={})[
        "renderer_context"
    ] == "gpu_physics_runtime"


def test_robot_eval_job_worker_manifest_gpu_and_remote_closure_edges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduler = {
        "status": "awaiting_explicit_gpu_and_simulator_gates",
        "selection": {"worker_profile": rejo._execution_worker_profile("mujoco")},
        "gpu_allocation": {"hard_timeout_seconds": 60, "requested_budget_usd": 4},
        "artifact_contract": {},
        "blockers": [],
    }
    monkeypatch.setenv(rejo.GENERIC_WORKER_IMAGE_REF_ENV, "repo/image:latest")
    plan = rejo._build_worker_launch_plan(
        request={},
        job_id="job",
        provisioner="runpod",
        simulator="mujoco",
        scheduler_decision=scheduler,
        timeout_seconds=60,
        generated_at="now",
    )
    assert "prebuilt_worker_image_ref_not_versioned" in plan["blockers"]
    monkeypatch.setenv(rejo.GENERIC_WORKER_IMAGE_REF_ENV, "repo/image:candidate-1")
    assert "prebuilt_worker_image_ref_not_provider_fetchable" in rejo._build_worker_launch_plan(
        request={},
        job_id="job",
        provisioner="runpod",
        simulator="mujoco",
        scheduler_decision=scheduler,
        timeout_seconds=60,
        generated_at="now",
    )["blockers"]
    monkeypatch.setenv(rejo.GENERIC_WORKER_IMAGE_REF_ENV, "repo/image:v1")
    monkeypatch.setenv(rejo.WORKER_MANIFEST_URI_ENV, "ftp://bad/manifest.json")
    monkeypatch.setenv(rejo.WORKER_CAPTURE_ROOT_BUNDLE_URI_ENV, "file:///local.zip")
    monkeypatch.setenv(rejo.WORKER_ARTIFACT_OUTPUT_URI_ENV, "s3://bucket/out")
    bad_uri_plan = rejo._build_worker_launch_plan(
        request={},
        job_id="job",
        provisioner="runpod",
        simulator="mujoco",
        scheduler_decision=scheduler,
        timeout_seconds=60,
        generated_at="now",
    )
    assert "worker_manifest_uri_not_fetchable_by_provider" in bad_uri_plan["blockers"]
    assert "capture_root_bundle_uri_not_fetchable_by_provider" in bad_uri_plan["blockers"]
    monkeypatch.setenv(rejo.WORKER_MANIFEST_URI_ENV, "file:///tmp/worker.json")
    monkeypatch.setenv(rejo.WORKER_CAPTURE_ROOT_BUNDLE_URI_ENV, "")
    monkeypatch.setattr(rejo, "REMOTE_ARTIFACT_OUTPUT_URI_SCHEMES", {"gs", "s3", "r2", "scratch"})
    monkeypatch.setenv(rejo.WORKER_ARTIFACT_OUTPUT_URI_ENV, "scratch://out")
    assert "worker_artifact_output_write_auth_contract_missing" in rejo._build_worker_launch_plan(
        request={},
        job_id="job",
        provisioner="docker_local",
        simulator="mujoco",
        scheduler_decision=scheduler,
        timeout_seconds=60,
        generated_at="now",
    )["blockers"]

    worker_manifest = rejo._build_worker_manifest(
        request={},
        job_id="job",
        capture_root=tmp_path,
        provisioner="docker_local",
        simulator="mujoco",
        evaluation_substrate=None,
        worker_launch_plan={
            "worker_manifest_input_contract": {"worker_manifest_uri_required_for_provider": True, "configured_worker_manifest_uri": "ftp://bad", "worker_manifest_uri_fetchable_by_provider": False},
            "artifact_upload_contract": {"artifact_output_uri_required_for_provider": True, "configured_artifact_output_uri": "local://out", "artifact_output_uri_provider_writable": True, "artifact_output_write_auth": {}},
            "input_bundle": {},
            "runtime_preflight_contract": {"command": "python -m smoke"},
        },
        allowed_simulators=["mujoco"],
        simulator_commands={"mujoco": "python run.py"},
        allow_wam_provider=True,
        wam_provider_commands={"cosmos3_wam": "python wam.py"},
        wam_artifact_output_uri=None,
        wam_provider_max_retries=2,
        wam_provider_timeout_seconds=9,
        timeout_seconds=10,
        budget_usd=1.0,
        generated_at="now",
    )
    assert "worker_manifest_uri_not_fetchable_by_provider" in worker_manifest["blockers"]
    assert "worker_artifact_output_write_auth_contract_missing" in worker_manifest["blockers"]

    provider_request = rejo._build_gpu_provider_launch_request(
        request_manifest={"provider": "fixture_local", "job_id": "job"},
        scheduler_decision={"blockers": []},
        worker_launch_plan={"blockers": [], "worker_image": {}, "gpu_selection": {}, "launch_mode": {}, "cache_plan": {}, "worker_entrypoint_contract": {}, "worker_manifest_input_contract": {}, "runtime_preflight_contract": {}, "artifact_upload_contract": {}, "cost_controls": {}},
        worker_manifest={"blockers": [], "status": "ready_for_worker_upload"},
        allow_gpu_provisioning=False,
        allow_simulator_execution=True,
        allowed_simulators=["mujoco"],
        simulator_commands={"mujoco": "python sim.py"},
        evaluation_substrate="fixture_wam",
        allow_wam_provider=True,
        wam_provider_commands={"cosmos3_wam": "python wam.py"},
        wam_artifact_output_uri="s3://bucket/out",
        wam_provider_max_retries=1,
        wam_provider_timeout_seconds=8,
        generated_at="now",
    )
    assert provider_request["status"] == "not_required_for_fixture_local"
    ledger = rejo._gpu_cost_control_ledger(
        request={"budget": {"timeoutSeconds": 10}},
        scheduler_decision={"gpu_allocation": {}, "blockers": []},
        worker_launch_plan={"launch_mode": {}, "cost_controls": {}},
        provider_launch_request={"provider": "runpod", "job_id": "job", "provider_request_shape": {"limits": {"hard_timeout_seconds": 10}}, "blockers": []},
        gpu_result={"status": "request_manifest_ready", "live_provider_calls_performed": True, "actual_gpu_seconds": 3},
        sim_result={},
        generated_at="now",
    )
    assert ledger["gpu_time"]["actual_gpu_time_source"] == "gpu_provisioning_result"

    remote = rejo._remote_cloud_execution_closure_manifest(
        job_id="job",
        provisioner="runpod",
        simulator="mujoco",
        worker_launch_plan={
            "worker_image": {},
            "input_bundle": {},
            "worker_manifest_input_contract": {},
            "artifact_upload_contract": {
                "configured_artifact_output_uri_present": True,
                "artifact_output_uri_provider_writable": True,
                "artifact_output_write_auth": {},
            },
            "launch_mode": {},
        },
        worker_manifest={},
        provider_launch_request={
            "status": "blocked",
            "provider_input_setup": {"status": "blocked", "blockers": ["missing_upload"], "provider_inputs_uploaded": False},
            "provider_request_shape": {"inputs": {}, "limits": {}},
        },
        gpu_result={"live_provider_calls_performed": True},
        gpu_cost_ledger={"live_provider_calls_performed": True, "gpu_time": {}},
        sim_result={},
        generated_at="now",
    )
    assert "remote_artifact_output_write_auth_contract_missing" in remote["contract_blockers"]
    assert "remote_actual_gpu_time_not_recorded" in remote["runtime_blockers"]


def test_robot_eval_job_fixture_expansion_and_closure_edge_blockers(tmp_path: Path) -> None:
    attempts = [{"task_id": "other", "scenario_id": "other", "score": 0.2}]
    assert rejo._attempt_for_matrix_run(
        attempts=attempts,
        matrix_run={"task_id": "task", "scenario_id": "scenario"},
        fallback_index=3,
    )["score"] == 0.2

    copied = {
        "normalized_attempt_trace": {
            "runner": "fixture",
            "attempts": [
                {"scenario_eval_run_id": "run-1", "task_id": "task", "scenario_id": "scenario"}
            ],
        }
    }
    assert rejo._expand_fixture_artifacts_to_scenario_eval_runs(
        copied_artifacts=copied,
        scenario_eval_matrix={"runs": [{"scenario_eval_run_id": "run-1"}]},
        job_dir=tmp_path,
        generated_at="now",
    ) == copied
    assert rejo._valid_explicitly_blocked_scenario_eval_run_ids(
        [{"scenario_eval_run_id": "other", "blockers": ["b"], "reason": "r", "stage": "s"}],
        missing_run_ids=["missing"],
    ) == ([], [])

    job_dir = tmp_path / "closure-job"
    job_dir.mkdir()
    common = {
        "job_dir": job_dir,
        "job_id": "job",
        "scene_id": "scene",
        "capture_id": "capture",
        "status": "blocked",
        "blockers": [],
        "scenario_eval_matrix": {
            "status": "completed",
            "scenario_eval_run_count": 2,
            "runs": [
                {"scenario_eval_run_id": "run-1", "task_id": "task", "scenario_id": "scenario-1"},
                {"scenario_eval_run_id": "run-2", "task_id": "task", "scenario_id": "scenario-2"},
            ],
        },
        "simulator_result": {},
        "copied_artifacts": {},
        "robot_pov_manifest": {},
        "policy_manifest": {},
        "policy_execution_manifest": {},
        "evaluation_result": {},
        "proof_boundary": {},
        "live_closure": {},
        "remote_cloud_closure": {},
        "webapp_status_projection": {},
        "data_package_export": {},
        "generated_at": "now",
    }

    _write_json(job_dir / "simulator_command_batch_metrics.json", {})
    _write_json(job_dir / "simulator_command_batch_trace_package_manifest.json", {})
    _write_json(job_dir / "simulator_command_batch_artifact_checksums.json", {})
    _write_json(job_dir / "simulator_command_digital_twin_fidelity_qa.json", {})
    manifest = rejo._robot_team_grade_eval_closure_manifest(**common)
    requirement_blockers = {
        blocker
        for requirement in manifest["requirements"]
        for blocker in requirement["blockers"]
    }
    assert "batch_metrics_manifest_missing_or_empty" in requirement_blockers
    assert "batch_trace_package_manifest_missing_or_empty" in requirement_blockers
    assert "artifact_checksums_manifest_missing_or_empty" in requirement_blockers
    assert "digital_twin_fidelity_qa_artifact_missing_or_empty" in requirement_blockers

    _write_json(
        job_dir / "simulator_command_batch_metrics.json",
        {
            "metric_coverage_complete": False,
            "attempt_metric_row_count": 1,
            "missing_metric_row_count": 1,
        },
    )
    _write_json(
        job_dir / "simulator_command_batch_trace_package_manifest.json",
        {"planner_state_coverage_complete": False, "control_stream_coverage_complete": False},
    )
    manifest = rejo._robot_team_grade_eval_closure_manifest(**common)
    requirement_blockers = {
        blocker
        for requirement in manifest["requirements"]
        for blocker in requirement["blockers"]
    }
    assert "batch_metric_coverage_incomplete" in requirement_blockers
    assert "batch_metric_row_count_mismatch" in requirement_blockers
    assert "batch_metric_rows_missing_required_keys" in requirement_blockers
    assert "contact_stream_record_count_missing" in requirement_blockers


def test_robot_eval_job_build_guards_and_inbox_dedupe(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    capture_root = _minimal_capture_root(tmp_path)
    with pytest.raises(ValueError, match="Unsupported provisioner"):
        rejo.build_robot_eval_job(capture_root=capture_root, job_request={}, job_id="job", provisioner="bad")
    with pytest.raises(ValueError, match="Unsupported simulator"):
        rejo.build_robot_eval_job(capture_root=capture_root, job_request={}, job_id="job", simulator="bad")

    inbox = tmp_path / "inbox"
    _write_json(inbox / "loose.json", {"job_id": "loose"})
    older = inbox / "a_older.json"
    newer = inbox / "z_newer.json"
    identity_request = {
        "source_kind": "webapp_route_forwarding_proof",
        "site_package": {
            "capture_root": str(capture_root),
            "capture_id": "cap",
            "site_slug": "slug",
        },
    }
    _write_json(older, {**identity_request, "job_id": "older"})
    _write_json(newer, {**identity_request, "job_id": "newer"})
    os.utime(older, (1, 1))
    os.utime(newer, (2, 2))

    def fake_build_robot_eval_job(**kwargs: object) -> dict[str, object]:
        job_id = str(kwargs["job_id"])
        job_dir = capture_root / "pipeline" / "robot_eval_jobs" / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = job_dir / "job_run_manifest.json"
        _write_json(manifest_path, {"job_id": job_id, "status": "completed"})
        return {"status": "completed", "job_dir": str(job_dir), "manifest_path": str(manifest_path)}

    monkeypatch.setattr(
        rejo,
        "execute_robot_eval_request_as_evaluation_run",
        fake_build_robot_eval_job,
    )
    manifest = rejo.run_robot_eval_job_request_inbox(capture_root=capture_root, inbox_dir=inbox)
    assert manifest["processed_count"] == 2
    assert manifest["superseded_request_count"] == 1


def test_robot_eval_job_build_propagates_wam_blockers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    capture_root = _minimal_capture_root(tmp_path)
    _write_minimal_robot_eval_inputs(capture_root)
    monkeypatch.setattr(
        rejo,
        "_run_simulator",
        lambda **_kwargs: (
            {"status": "completed", "artifact_paths": {}, "blockers": []},
            {},
            [],
        ),
    )
    monkeypatch.setattr(
        rejo,
        "run_wam_eval_job",
        lambda **_kwargs: {"status": "blocked", "blockers": ["wam_provider_fixture_blocked"]},
    )
    result = rejo.build_robot_eval_job(
        capture_root=capture_root,
        job_request={
            "customer": {"customer_id": "customer-1"},
            "robot_profile": {"robot_profile_id": "unitree_g1"},
            "requested_tasks": [{"task_id": "task-1"}],
            "policy_package": {
                "high_level_skill_trace": {"ordered_skill_sequence": ["move_to_target"]}
            },
        },
        job_id="wam-blocked-job",
        simulator="fixture",
        evaluation_substrate="fixture_wam",
    )
    manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
    assert "wam_evaluation_blocked" in manifest["blockers"]
    assert "wam_provider_fixture_blocked" in manifest["missing_inputs"]


def test_robot_eval_job_command_training_evaluation_projection_and_cli_edges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert rejo._command_executable("\"unterminated") is None
    monkeypatch.setattr(rejo.subprocess, "run", lambda *_args, **_kwargs: (_ for _ in ()).throw(FileNotFoundError()))
    assert rejo._run_command_simulator(simulator="mujoco", command_text="missing", timeout_seconds=1, generated_at="now")[
        "reason"
    ] == "missing_dependency"
    monkeypatch.setattr(
        rejo.subprocess,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(subprocess.TimeoutExpired(cmd=["sim"], timeout=1, output="out", stderr="err")),
    )
    assert rejo._run_command_simulator(simulator="mujoco", command_text="sim", timeout_seconds=1, generated_at="now")[
        "reason"
    ] == "timeout"
    output = tmp_path / "bad-output.json"
    output.write_text("{", encoding="utf-8")
    monkeypatch.setattr(rejo.subprocess, "run", lambda *_args, **_kwargs: SimpleNamespace(returncode=0, stdout="{bad", stderr=""))
    sim = rejo._run_command_simulator(
        simulator="isaac_sim",
        command_text="python sim.py",
        timeout_seconds=1,
        generated_at="now",
        output_path=output,
        capture_root=tmp_path,
        scenario_eval_matrix_path=tmp_path / "matrix.json",
    )
    assert sim["status"] == "completed"

    monkeypatch.setenv("BLUEPRINT_ALLOW_COSMOS_TRAINING", "true")
    monkeypatch.setattr(rejo.subprocess, "run", lambda *_args, **_kwargs: SimpleNamespace(returncode=0, stdout="ok", stderr=""))
    training = rejo._training_result(
        request_manifest={"status": "export_manifest_only", "preference": {"checkpoint_path": "/tmp/checkpoint"}},
        allow_training=True,
        training_command="python train.py",
        timeout_seconds=1,
        generated_at="now",
    )
    assert training["status"] == "completed"
    assert rejo._evaluation_result(
        evaluation_request={"status": "not_requested"},
        simulator_result={},
        copied_artifacts={},
        generated_at="now",
    )["status"] == "not_requested"
    records = rejo._explicitly_blocked_scenario_eval_run_records(
        {"blocked_scenario_eval_runs": {"run-a": {"reason": "bad", "stage": "sim", "blockers": ["b"]}}},
        {"scenario_eval_run_blockers": ["run-b", ""]},
    )
    valid, invalid = rejo._valid_explicitly_blocked_scenario_eval_run_ids(records, missing_run_ids=["run-a", "run-b"])
    assert valid == ["run-a"]
    assert invalid == ["run-b"]

    job_dir = tmp_path / "job"
    projection = rejo._webapp_robot_eval_status_projection(
        job_dir=job_dir,
        job_id="job",
        scene_id="scene",
        capture_id="capture",
        status="completed",
        blockers=[],
        request={},
        scenario_eval_matrix={"status": "completed", "runs": []},
        simulator_result={"robot_team_grade_package_complete": True},
        copied_artifacts={},
        robot_pov_manifest={},
        policy_manifest={},
        policy_execution_manifest={},
        evaluation_result={},
        proof_boundary={},
        live_closure={},
        data_package_export={},
        generated_at="now",
    )
    assert projection["buyer_display_state"] == "robot_team_package_ready_for_review"

    assert rejo._webapp_request_identity({}) is None
    assert rejo._webapp_request_identity({"source_kind": "webapp_route_forwarding_proof", "site_package": {"capture_root": "/cap", "capture_id": "cap", "site_slug": "slug"}})[0] == "webapp_route_forwarding_proof"
    with monkeypatch.context() as path_stat_patch:
        path_stat_patch.setattr(
            Path,
            "stat",
            lambda self: (_ for _ in ()).throw(OSError("no stat")),
        )
        assert rejo._inbox_request_sort_key(tmp_path / "missing.json")[0] == 0
    with pytest.raises(ValueError):
        rejo._parse_simulator_commands(["fixture=bad"])
    with pytest.raises(ValueError):
        rejo._parse_policy_execution_commands(["bad"])
    assert rejo._parse_simulator_commands(["mujoco=python sim.py"]) == {"mujoco": "python sim.py"}
    assert rejo._parse_policy_execution_commands(["teleop_demo=python demo.py"]) == {"teleop_demo": "python demo.py"}
    assert rejo._agent_adapter_from_mode("fake", allow_live_operator=False).__class__.__name__ == "FakeRobotEvalJobAgentAdapter"
    assert rejo._agent_adapter_from_mode("agents-sdk", allow_live_operator=True).__class__.__name__ == "AgentsSdkRobotEvalJobAdapter"
    assert rejo._agent_adapter_from_mode("none", allow_live_operator=False) is None

    monkeypatch.setattr(
        rejo,
        "execute_robot_eval_request_as_evaluation_run",
        lambda **_kwargs: {"manifest_path": "/tmp/manifest.json", "status": "completed"},
    )
    assert rejo.main(["--capture-root", str(tmp_path), "--job-request", str(tmp_path / "request.json"), "--job-id", "job"]) == 0
    assert "manifest=" in capsys.readouterr().out
    monkeypatch.setattr(rejo, "run_robot_eval_job_request_inbox", lambda **_kwargs: {"status": "completed", "processed_count": 2})
    assert rejo.main(["--capture-root", str(tmp_path), "--job-request-inbox", str(tmp_path / "inbox")]) == 0
    assert "processed_count=2" in capsys.readouterr().out
    assert rejo.main(["--capture-root", str(tmp_path)]) == 65
    assert "FAILED" in capsys.readouterr().out
    monkeypatch.setattr(sys, "argv", ["robot_eval_job_orchestrator", "--capture-root", str(tmp_path)])
    with pytest.raises(SystemExit) as entrypoint_exit:
        run_module_as_main("blueprint_pipeline.robot_eval_job_orchestrator")
    assert entrypoint_exit.value.code == 65


def test_robot_eval_cli_validates_typed_admission_before_execution(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        rejo,
        "execute_robot_eval_request_as_evaluation_run",
        lambda **kwargs: calls.append(kwargs),
    )
    monkeypatch.delenv("BLUEPRINT_ALLOW_SIMULATOR_EXECUTION", raising=False)

    exit_code = rejo.main(
        [
            "--capture-root",
            str(tmp_path),
            "--job-request",
            str(tmp_path / "request.json"),
            "--job-id",
            "job",
            "--allow-simulator-execution",
        ]
    )

    assert exit_code == 65
    assert calls == []
    assert "cli_admission_missing_environment_approval" in capsys.readouterr().out

    monkeypatch.setenv("BLUEPRINT_ALLOW_SIMULATOR_EXECUTION", "perhaps")
    assert rejo.main(
        [
            "--capture-root",
            str(tmp_path),
            "--job-request",
            str(tmp_path / "request.json"),
            "--job-id",
            "job",
        ]
    ) == 65
    assert calls == []
    assert "invalid_boolean_environment_value" in capsys.readouterr().out
