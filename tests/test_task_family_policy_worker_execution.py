"""Production family worker creates a registered runtime and steps native state."""
from __future__ import annotations

import base64
import io
import json
import shutil

import numpy as np
import pytest
from PIL import Image

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.native_task_arena_policy_canary_worker import _run_family_canary_episode
from blueprint_pipeline.task_family_policy_https_client import TaskFamilyPolicyHttpsClient, TaskFamilyPolicySpec
from blueprint_pipeline.task_family_policy_worker_execution import execute_family_policy_episode
from tests.test_task_family_native_runtime import installed_sdk as sdk_fixture
from tests.test_task_family_policy_execution import _spec


@pytest.fixture
def installed_sdk(tmp_path, monkeypatch):
    yield from sdk_fixture.__wrapped__(tmp_path, monkeypatch)


def _inputs(sdk):
    native = sdk.args["native_contract"]
    # Capture truth does not need the candidate's bounded evaluation budget.
    native["task_spec"].pop("maximum_action_steps", None)
    native["task_spec"].pop("control_frequency_hz", None)
    native["task_spec"]["prompt"] = "Press the captured button"
    native["contract_digest"] = canonical_digest(native, digest_field="contract_digest")
    scene = {"native_runtime_contract": native, "scene_id": native["scene_id"],
        "task_id": native["task_id"], "task_kind": native["task_kind"], "task_spec": native["task_spec"],
        "scenario": {"cell_id": native["scenario_cell_id"], "seed": native["seed"],
                     "scenario_instance_digest": native["scenario_instance_digest"]},
        "task_family_runtime_selection": sdk.args["qualified_runtime_selection"]}
    return scene, _spec(scene)


def _policy(spec, *, actions=None, fault=None):
    policy_spec = TaskFamilyPolicySpec(**spec["policy_spec"])
    requests = []

    def transport(**request):
        requests.append(request)
        if request["method"] == "GET":
            if fault == "preflight":
                raise PermissionError("hermetic readiness authentication failed")
            if fault == "identity" and len(requests) > 1:
                return json.dumps({**policy_spec.expected_metadata(), "policy_id": "wrong-policy"}).encode()
            return json.dumps(policy_spec.expected_metadata()).encode()
        if fault == "inference":
            raise TimeoutError("hermetic endpoint did not answer")
        if fault == "response":
            return json.dumps({"reward": 1}).encode()
        return json.dumps({"actions": [[1]] if actions is None else actions}).encode()

    return TaskFamilyPolicyHttpsClient(spec=policy_spec, prompt=spec["prompt"],
        bearer_token="hermetic-test-token", transport=transport), requests


def test_production_worker_dispatch_builds_registered_runtime_and_retains_native_evidence(
    installed_sdk, tmp_path, monkeypatch,
):
    sdk = installed_sdk
    scene, spec = _inputs(sdk)
    monkeypatch.setenv("BLUEPRINT_TASK_FAMILY_RUNTIME_CATALOG_PATH", str(sdk.args["runtime_catalog_path"]))
    client, requests = _policy(spec)
    output_root = tmp_path / "worker-output"
    result = _run_family_canary_episode(scene_plan=scene,
        policy={"client": client, "spec": spec, "checkpoint_digest": spec["checkpoint_digest"],
                "runtime_identity_digest": spec["runtime_identity_digest"]},
        context={"cell_id": "cell1", "candidate_id": spec["candidate_id"]},
        authority={"run_id": "worker-family", "authority_digest": "sha256:" + "a" * 64},
        inputs={"configuration_digest": "sha256:" + "b" * 64},
        packet_root=sdk.args["asset_root"], output_root=output_root, progress={})
    assert len(sdk.calls) == 1
    engine = sdk.module().instances[0]
    assert engine.actions == [[1.0], [1.0]] and engine.position == .04
    assert engine.close_count == 1
    episode = result["episode"]
    assert result["status"] == "completed" and result["candidate_policy_queried"]
    assert result["actions_reached_robot"] and not result["arm_moved"]
    assert episode["native_family_receipt"]["runtime_closed"]
    assert episode["score"]["task_succeeded"] and episode["score"]["status"] == "scored"
    assert episode["native_measurements"]["samples"][-1]["measurements"]["stage1"]["button_travel_m"] == .04
    assert episode["lifecycle"]["actual_action_steps"] == 2
    assert all(result["evidence_artifacts"][role] for role in ("frame_manifest", "review_video", "state_trace"))
    sent = [json.loads(request["body"]) for request in requests if request["method"] == "POST"]
    assert len(sent) == 2
    assert set(sent[0]["observation"]) == {"rgb", "proprioception", "prompt"}
    assert sent[0]["observation"]["prompt"] == spec["prompt"]
    pixels = np.asarray(Image.open(io.BytesIO(base64.b64decode(sent[0]["observation"]["rgb"]["data"]))))
    assert np.all(pixels == 0)
    assert "asset_states" not in json.dumps(sent)


@pytest.mark.parametrize("fault", ["missing_catalog", "action_bounds", "step", "delivery_readback"])
def test_worker_refuses_specific_gap_and_closes_registered_engine(
    installed_sdk, tmp_path, monkeypatch, fault,
):
    sdk = installed_sdk
    scene, spec = _inputs(sdk)
    monkeypatch.setenv("BLUEPRINT_TASK_FAMILY_RUNTIME_CATALOG_PATH", str(sdk.args["runtime_catalog_path"]))
    client, requests = _policy(spec, actions=[[2]] if fault == "action_bounds" else None)
    if fault == "missing_catalog":
        monkeypatch.delenv("BLUEPRINT_TASK_FAMILY_RUNTIME_CATALOG_PATH")
        with pytest.raises(ValueError, match="qualified_runtime_catalog_not_configured"):
            execute_family_policy_episode(scene_plan=scene, spec=spec, policy_client=client,
                packet_root=sdk.args["asset_root"], output_dir=tmp_path / "episode", episode_id="gap")
        assert not sdk.calls and not requests
        return
    if fault == "step":
        def broken_step(self, *args, **kwargs):
            raise RuntimeError("engine_actuator_failed")
        monkeypatch.setattr(sdk.module().Environment, "step_native_action", broken_step)
    if fault == "delivery_readback":
        step = sdk.module().Environment.step_native_action
        def undelivered(self, *args, **kwargs):
            step(self, *args, **kwargs)
            self.robot_action_applied = False
        monkeypatch.setattr(sdk.module().Environment, "step_native_action", undelivered)
    progress = {}
    episode = execute_family_policy_episode(scene_plan=scene, spec=spec, policy_client=client,
        packet_root=sdk.args["asset_root"], output_dir=tmp_path / "episode", episode_id="gap", progress=progress)
    assert episode["status"] == "invalid"
    assert sdk.module().instances[0].close_count == 1
    assert not episode["motion_evidence"]["actions_reached_robot"]
    assert {"action_bounds": "action_bounds_invalid", "step": "engine_actuator_failed",
            "delivery_readback": "native_action_delivery_unproven"}[fault] in episode["failure"]["reason"]
    assert progress["candidate_policy_queried"]


def _stage_family_worker(sdk, tmp_path, monkeypatch):
    from blueprint_pipeline.native_task_arena_policy_canary_bundle import _family_operator_inputs
    from blueprint_pipeline.native_task_arena_policy_canary_session import build_session_authority
    from tests.test_native_task_arena_policy_canary_lifecycle_rehearsal import _record, _write

    scene, spec = _inputs(sdk)
    scene["schema_version"] = "native_task_arena_scene_plan.v1"
    scene["plan_digest"] = canonical_digest(scene, digest_field="plan_digest")
    candidate = spec["candidate_id"]
    task = spec["task_success_contract"]
    runtime = tmp_path / "production-worker"
    packet = runtime / "native_task_packet"
    shutil.copytree(sdk.args["asset_root"], packet)
    _write(packet / "native_task_arena_scene_plan.v1.json", scene)
    _write(packet / "native_task_arena_packet_request.v1.json", {})
    packet_path = _write(packet / "native_task_arena_packet_receipt.v1.json", {"scope": "development_only"})
    source_path = _write(runtime / "runtime_source.json", {"scope": "development_only"})
    scenario = {"family": "canonical_anchor", "parameters": {}}
    cells = [{"cell_id": "cell1", "seed": 0, "cell_spec_digest": "sha256:" + "1" * 64,
        "family": "canonical_anchor", "resolved_scenario": scenario,
        "resolved_scenario_digest": canonical_digest(scenario), "control_diagnostic": {
            "mode": "native_family_prepolicy_required", "typed_gap": "native_site_qualification_pending",
            "policy_execution_blocked": True}}]
    activation = {"run_id": "production-family", "run_kind": "blueprint_beta_screening",
        "claim_ceiling": "diagnostic_policy_execution", "candidate_ids": [candidate],
        "campaign_unit_count": 1, "task_success_contract": task,
        "task_success_contract_digest": task["contract_digest"]}
    activation["activation_digest"] = canonical_digest(activation, digest_field="activation_digest")
    activation_path = _write(runtime / "activation.json", activation)
    digests = {"scene_revision_digest": "sha256:" + "2" * 64,
               "plan_digest": "sha256:" + "3" * 64, "configuration_digest": "sha256:" + "4" * 64,
               "task_success_contract_digest": task["contract_digest"]}
    static_digest = "sha256:" + "5" * 64
    admission = {"schema_version": "blueprint.beta_policy_screening_admission.v1",
        "candidate_id": candidate, "evaluation_purpose": "screening", "backend": "task_family_native.v1",
        "admission_digest": "sha256:" + "6" * 64, **digests,
        "pre_policy_site_qualification_required": True, "family_static_readiness_digest": static_digest}
    inputs = {"schema_version": "task_evaluation_policy_canary_runtime_inputs.v1",
        "run_id": activation["run_id"], "run_kind": activation["run_kind"],
        "claim_ceiling": activation["claim_ceiling"], "candidate_ids": [candidate],
        "task_success_contract": task, "activation_digest": activation["activation_digest"],
        **digests, "base_native_packet": _record(packet_path), "runtime_source": _record(source_path),
        "cells": cells, "pre_policy_site_qualification_required": True,
        "family_static_readiness_digest": static_digest, "beta_screening_admission": admission,
        "execution_authority": {"maximum_provider_allocations": 1, "retry_cap": 0,
            "single_warm_provider_session_required": True, "caller_surviving_watchdog_required": True,
            "billing_teardown_provider_zero_required": True}}
    inputs["runtime_inputs_digest"] = canonical_digest(inputs, digest_field="runtime_inputs_digest")
    inputs_path = _write(runtime / "runtime_inputs/policy_canary_runtime_inputs.json", inputs)
    authority = build_session_authority(activation_manifest=activation, activation_record=_record(activation_path),
        runtime_inputs=inputs, runtime_input_record=_record(inputs_path),
        resource_name="blueprint-native-task-policy-canary-hermetic-family", hard_cap_usd=1, hard_ttl_seconds=60)
    _write(runtime / "runtime_inputs/policy_canary_session_authority.json", authority)
    _write(runtime / f"runtime_inputs/policy_execution_spec.{candidate}.json", spec)
    monkeypatch.setenv("BLUEPRINT_TASK_FAMILY_RUNTIME_CATALOG_PATH", str(sdk.args["runtime_catalog_path"]))
    operator_files, operator_binding = _family_operator_inputs({candidate: spec})
    for relative, source in operator_files.items():
        target = runtime / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    manifest = {"schema_version": "native_task_arena_policy_canary_provider_bundle.v1",
        "execution_mode": "blueprint_beta_screening_session", "run_kind": activation["run_kind"],
        "claim_ceiling": activation["claim_ceiling"], "runtime_inputs_digest": inputs["runtime_inputs_digest"],
        "task_success_contract_digest": task["contract_digest"], "authority_digest": authority["authority_digest"],
        "execution_spec_digests": {candidate: spec["execution_spec_digest"]},
        "task_family_operator_inputs": operator_binding}
    manifest["input_digest"] = canonical_digest(manifest, digest_field="input_digest")
    _write(runtime / "adp_arena_provider_manifest.json", manifest)
    return runtime, spec


@pytest.mark.parametrize("controls_enabled", [True, False])
def test_actual_selected_cell_worker_runs_registered_factory_after_native_controls(
    installed_sdk, tmp_path, monkeypatch, controls_enabled,
):
    import blueprint_pipeline.native_task_arena_policy_canary_worker as worker
    from blueprint_pipeline import task_evaluation_control_stage_policy

    sdk = installed_sdk
    runtime, spec = _stage_family_worker(sdk, tmp_path, monkeypatch)
    monkeypatch.setattr(task_evaluation_control_stage_policy, "CONTROLS_PAUSED", not controls_enabled)
    requests = []

    def create_client(loaded_spec):
        # Site qualification must construct and close all three native controls
        # before the candidate transport is loaded or queried.
        assert len(sdk.calls) == 3
        assert all(engine.close_count == 1 for engine in sdk.module().instances)
        client, wire = _policy(loaded_spec)
        requests.append(wire)
        return client

    monkeypatch.setattr("blueprint_pipeline.team_candidate_policy_client.canary_policy_client", create_client)
    output = tmp_path / "selected-cell-output"
    status = worker._run_selected_cell(0, runtime_root=runtime, output_root=output)
    result = json.loads((output / worker.PROVIDER_RESULT_FILENAME).read_text())
    if controls_enabled:
        assert status == 0, result
        assert len(sdk.calls) == 4 and len(result["episodes"]) == 1
        assert all(engine.close_count == 1 for engine in sdk.module().instances)
        assert result["episodes"][0]["episode"]["score"]["task_succeeded"]
        assert result["preload_observation_gate"]["site_qualification"]["status"] == "passed"
        assert len([row for row in requests[0] if row["method"] == "POST"]) == spec["max_policy_queries"]
    else:
        assert status == 1 and result["status"] == "blocked"
        assert result["policy_loads"] == [] and not sdk.calls and not requests
        assert result["session_failure_type"] == "ValueError"


@pytest.mark.parametrize("fault,outcome", [("bounds", "failure"), ("wrong_output", "failure"),
    ("response", "failure"), ("inference", "failure"), ("identity", "failure"),
    ("runtime", "void"), ("media", "void"), ("local_request", "void"), ("sdk_spoof", "void"), ("preflight", "void")])
def test_selected_cell_preserves_native_policy_failure_cause_and_retained_evidence(
    installed_sdk, tmp_path, monkeypatch, fault, outcome,
):
    import blueprint_pipeline.native_task_arena_policy_canary_worker as worker
    from blueprint_pipeline.native_task_arena_policy_canary_session import _episode_evidence_valid
    from blueprint_pipeline.beta_policy_screening_result import candidate_result_summary
    from blueprint_pipeline.task_family_policy_https_client import TaskFamilyPolicyFailure
    from blueprint_pipeline.team_policy_episode_outcome import episode_outcome

    sdk = installed_sdk
    runtime, _spec_value = _stage_family_worker(sdk, tmp_path, monkeypatch)
    monkeypatch.setattr("blueprint_pipeline.task_evaluation_control_stage_policy.CONTROLS_PAUSED", False)
    requests = []

    def create_client(spec):
        assert len(sdk.calls) == 3
        client, wire = _policy(spec, fault=fault, actions=[[2]] if fault == "bounds" else
            [["not-an-action"]] if fault == "wrong_output" else None)
        requests.extend([wire])
        if fault in {"runtime", "sdk_spoof"}:
            def failed_step(self, *args, **kwargs):
                if fault == "sdk_spoof":
                    raise TaskFamilyPolicyFailure("task_family_policy_action_bounds_invalid", origin="returned_action_validation")
                raise RuntimeError("hermetic native actuator failed")
            monkeypatch.setattr(sdk.module().Environment, "step_native_action", failed_step)
        if fault == "media":
            def failed_write(*args, **kwargs):
                raise OSError("hermetic media disk full")
            monkeypatch.setattr("blueprint_pipeline.task_evaluation_task_family_native_adapter.persist_observation_frame", failed_write)
        if fault == "local_request":
            def failed_local_sink(*args, **kwargs):
                raise OSError("hermetic local request evidence disk full")
            client.bind_request_evidence_sink(failed_local_sink)
        return client

    monkeypatch.setattr("blueprint_pipeline.team_candidate_policy_client.canary_policy_client", create_client)
    output = tmp_path / "failure-output"
    worker._run_selected_cell(0, runtime_root=runtime, output_root=output)
    result = json.loads((output / worker.PROVIDER_RESULT_FILENAME).read_text())
    row = result["episodes"][0]
    assert episode_outcome(row)[0] == outcome
    assert row["episode_failure_cause"] == ("policy" if outcome == "failure" else "media" if fault == "media" else "infrastructure")
    assert _episode_evidence_valid(row), row
    # Closeout is a hermetic receipt fixture; retain the actual worker episode
    # unchanged through the production session validator and beta denominator.
    result.update(status="blocked", provider_allocations_observed=1,
        session_closeout={"provider_allocations_observed": 1, "teardown_completed": True, "provider_zero_confirmed": True})
    result["result_digest"] = canonical_digest(result, digest_field="result_digest")
    binding = result["beta_screening_admission"]
    staged_inputs = json.loads((runtime / "runtime_inputs/policy_canary_runtime_inputs.json").read_text())
    summary = candidate_result_summary(result, plan={"plan_digest": binding["plan_digest"],
        "scene_revision_digest": binding["scene_revision_digest"], "scene_inputs": {},
        "pre_policy_site_qualification_required": True,
        "family_static_readiness_digest": result["family_static_readiness_digest"]},
        candidate={"candidate_id": row["candidate_id"], "configuration_digest": binding["configuration_digest"],
            "admission_digest": binding["admission_digest"], "episode_count": 1,
            "scenario_cells": staged_inputs["cells"], "execution_spec": {"backend": "task_family_native.v1",
                "runtime_digest": row["runtime_identity_digest"],
                "task_success_contract_digest": result["task_success_contract_digest"]}})
    assert summary["episodes_failed"] == (outcome == "failure")
    assert summary["episodes_void"] == (outcome == "void")
    assert summary["episodes_succeeded"] == 0
    assert row["actions_reached_robot"] is False
    assert len(sdk.calls) == 4 and all(engine.close_count == 1 for engine in sdk.module().instances)
    assert "_native_policy_failure" not in json.dumps(result)
    if fault in {"media", "preflight"}:
        assert not row["first_observation_retained"] and not any(request["method"] == "POST" for request in requests[0])
        assert row["visual_evidence"]["media_gap"]["type"] == "before_first_observation"
    else:
        assert row["first_observation_retained"] and row["visual_evidence"]["status"] == "complete"
        assert row["visual_evidence"]["terminal_observation_invented"] is False
        assert row["evidence_artifacts"]["frame_manifest"] and row["evidence_artifacts"]["review_video"]
        assert row["episode"]["native_site_qualification"]["status"] == "passed"
    if outcome == "failure":
        assert row["policy_failure_code"].startswith("task_family_policy_")
    if fault == "bounds":
        assert row["candidate_action_shape_validated"] and row["candidate_action_finite_validated"]
        assert not row["candidate_action_bounds_validated"]
        assert row["candidate_policy_action_queries"][0]["actions"] == [[2.0]]
