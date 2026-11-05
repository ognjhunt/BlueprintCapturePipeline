"""Actual worker/transport paths with retained beta bundles and fake external I/O."""
from dataclasses import replace
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import zipfile

import pytest

from blueprint_pipeline import beta_policy_screening_launch as launch
from blueprint_pipeline import beta_screening_control_delivery as delivery
from blueprint_pipeline import beta_screening_runtime_control as control
from blueprint_pipeline import native_task_arena_policy_canary_worker as worker
from blueprint_pipeline import native_task_arena_policy_worker as policy_worker
from blueprint_pipeline import vast_provider_adapter as adapter
from blueprint_pipeline.decision_evidence_contracts import canonical_digest, cross_runtime_canonical_digest
from tests.test_beta_policy_screening_control_contracts import response
from tests.test_beta_policy_screening_launch import _plan
from tests.test_beta_screening_runtime_control import staged_control, install


def stage_beta_bundle(tmp_path, monkeypatch, *, hosted=False):
    if hosted:
        from tests.test_beta_policy_screening_hosted import hosted_plan
        plan, _specs, _settings = hosted_plan(tmp_path, monkeypatch)
    else:
        plan, _facts = _plan(tmp_path, monkeypatch)
    candidate = next(row for row in plan["candidates"] if row["status"] == "selected" and (hosted or row["candidate_id"] == "pi05_droid"))
    prepared = launch.prepare_candidate(plan=plan, candidate=candidate, attempt_dir=tmp_path / "attempt")
    authority = launch._read(Path(prepared["authority"]["path"]))
    bundle = launch._read(Path(prepared["bundle_receipt"]["path"]))
    with zipfile.ZipFile(bundle["bundle_path"]) as archive:
        archive.extractall(tmp_path / "gpu")
        names = set(archive.namelist())
        assert "provider_runtime/blueprint_pipeline/beta_screening_runtime_control.py" in names
        assert "provider_runtime/blueprint_pipeline/beta_policy_screening_control_contracts.py" in names
        assert not any(name.endswith("beta_screening_control_delivery.py") for name in names)
        entrypoint = archive.read("provider_runtime/run_adp_arena_provider_runtime.sh").decode()
        assert entrypoint.index("blueprint_pipeline.beta_screening_runtime_control") < entrypoint.index("native_task_runtime_source_provision")
    runtime = tmp_path / "gpu" / "provider_runtime"
    inputs = launch._read(runtime / "runtime_inputs" / "policy_canary_runtime_inputs.json")
    scope = {"campaign_id": plan["campaign_id"], "site_id": plan["site_id"], "plan_digest": plan["plan_digest"],
             "candidate_id": candidate["candidate_id"], "configuration_digest": candidate["configuration_digest"],
             "attempt_id": "screen-exact", "invocation_id": cross_runtime_canonical_digest({"invocation": 1}),
             "prepared_digest": prepared["prepared_digest"], "authority_digest": authority["authority_digest"],
             "resource_name": authority["resource_name"], "source": plan["source"]}
    monkeypatch.setattr(control.time, "time", lambda: 1000.0)
    install(runtime, authority, scope)
    output = runtime.parent / "runtime_output"
    output.mkdir(exist_ok=True)
    return runtime, output, inputs, authority, scope, prepared, bundle, candidate


@pytest.mark.parametrize("interrupted", [False, True])
def test_actual_parent_stops_before_next_cell_and_retains_historical_prefix(tmp_path, monkeypatch, interrupted):
    runtime, output, inputs, authority, scope, *_ = stage_beta_bundle(tmp_path, monkeypatch)
    assert not worker.controls_required(inputs["task_success_contract"])
    spawned = []

    def spawn(**kwargs):
        spawned.append(kwargs["index"])
        # A retained, deliberately incomplete child is diagnostic raw evidence;
        # it must survive the stop without becoming an accepted policy score.
        child = {"selected_cell_index": 0, "episodes": [], "fixture": "unaccepted_raw_prefix", "status": "blocked",
            "task_success_contract": inputs["task_success_contract"], "task_success_contract_digest": inputs["task_success_contract_digest"],
            "policy_loads": [], "candidate_policy_queried": False, "preload_observation_gate": None,
            "session_closeout": {"runtime_closed": True},
            "telemetry": {"message_count": 0, "channel_message_counts": {"observations": 0, "episode_envelopes": 0}},
            "session_failure_type": "fixture_camera_refusal", "session_failure_source_trace": [{"function": "prepolicy_observation_gate"}]}
        if interrupted:
            child["episodes"] = [{"candidate_id": scope["candidate_id"], "fixture": "interrupted_query"}]
            child["candidate_policy_queried"] = True
        child["result_digest"] = canonical_digest(child, digest_field="result_digest")
        (kwargs["child_root"] / worker.PROVIDER_RESULT_FILENAME).write_text(json.dumps(child))
        install(runtime, authority, scope, now=1001, revision=2, stopped=True)
        return 1
    assert worker._run_isolated_cell_processes(runtime_root=runtime, output_root=output, run_cell_process=spawn) == 1
    assert spawned == [0] and not (output / "cell_runs/01").exists()
    retained = json.loads((output / worker.PROVIDER_RESULT_FILENAME).read_text())
    assert retained["applicability"] == "historical_only"
    assert retained["execution_stopped_after_cell"] == 1 and retained["read_only_closeout_allowed"] is True
    if interrupted:
        partial = retained["retained_partial_cell_results"][0]
        assert (output / partial["relative_path"]).is_file()
        assert json.loads((output / partial["relative_path"]).read_text())["episodes"][0]["fixture"] == "interrupted_query"
        assert retained["candidate_policy_queried"] is True
        assert retained["policy_loads"] is None
    else:
        assert retained["isolated_cell_failures"][0]["raw_child_result"]["fixture"] == "unaccepted_raw_prefix"
    assert retained["episodes"] == []


def _completed_parent_fixture_child(inputs, index):
    """A parent-retention contract fixture, never engine or policy evidence."""
    child = {"schema_version": "native_task_arena_policy_canary_session_result.v1",
             "selected_cell_index": index, "status": "runtime_selected_cell_completed_pending_aggregation",
             "task_success_contract": inputs["task_success_contract"],
             "task_success_contract_digest": inputs["task_success_contract_digest"],
             "candidate_policy_queried": True,
             "episodes": [{**inputs["cells"][index], "candidate_id": candidate,
                           "candidate_policy_queried": True, "fixture": "retained_completed_prefix"}
                          for candidate in inputs["candidate_ids"]]}
    child["result_digest"] = canonical_digest(child, digest_field="result_digest")
    return child


@pytest.mark.parametrize("stop_kind", ["cancelled", "expired", "stale"])
@pytest.mark.parametrize("final_cell_queried", [False, True])
def test_final_isolated_child_stop_retains_queried_prefix_without_another_iteration(
    tmp_path, monkeypatch, stop_kind, final_cell_queried,
):
    runtime, output, inputs, authority, scope, *_ = stage_beta_bundle(tmp_path, monkeypatch)
    spawned, stop_receipts = [], []
    final = len(inputs["cells"]) - 1

    def spawn(**kwargs):
        index = kwargs["index"]
        spawned.append(index)
        child = _completed_parent_fixture_child(inputs, index)
        if index == final:
            if stop_kind == "cancelled":
                install(runtime, authority, scope, now=1001, revision=2, stopped=True)
            elif stop_kind == "expired":
                install(runtime, authority, scope, now=1001, deadline=1100, revision=2)
                monkeypatch.setattr(control.time, "time", lambda: 1100.0)
            else:
                monkeypatch.setattr(control.time, "time", lambda: 1600.0)
            guard = control.RuntimeControlGuard(runtime_root=runtime, inputs=inputs, authority=authority)
            with pytest.raises(control.RuntimeControlStopped) as stopped:
                guard.check("before_policy_query")
            stop_receipts.append(stopped.value.receipt)
            child.update(status="blocked", candidate_policy_queried=final_cell_queried,
                         episodes=[{"fixture": "interrupted_final_query"}] if final_cell_queried else [],
                         execution_control_stop=stopped.value.receipt, applicability="historical_only")
            child["result_digest"] = canonical_digest(child, digest_field="result_digest")
        (kwargs["child_root"] / worker.PROVIDER_RESULT_FILENAME).write_text(json.dumps(child))
        return 1 if index == final else 0

    assert worker._run_isolated_cell_processes(
        runtime_root=runtime, output_root=output, run_cell_process=spawn,
    ) == 1
    assert spawned == list(range(len(inputs["cells"])))
    retained = json.loads((output / worker.PROVIDER_RESULT_FILENAME).read_text())
    assert retained["status"] == "blocked" and retained["applicability"] == "historical_only"
    assert retained["candidate_policy_queried"] is True  # Earlier returned queries stay true.
    assert retained["new_processing_authorized"] is False and retained["read_only_closeout_allowed"] is True
    assert retained["episodes_per_policy"] == final
    assert len(retained["episodes"]) == final * len(inputs["candidate_ids"])
    assert [row["cell_id"] for row in retained["episodes"]] == [row["cell_id"] for row in inputs["cells"][:-1]]
    assert retained["learned_policy_rollout_count"] == len(retained["episodes"])
    assert retained["execution_stopped_after_cell"] == len(inputs["cells"])
    assert retained["planned_episodes_per_policy"] == len(inputs["cells"])
    assert retained["execution_control_stop"] == stop_receipts[0]
    assert retained["blockers"] == [stop_receipts[0]["reason"]]
    partial, = retained["retained_partial_cell_results"]
    path = output / partial["relative_path"]
    assert path == output / "cell_runs" / f"{final:02d}" / worker.PROVIDER_RESULT_FILENAME
    assert path.stat().st_size == partial["size_bytes"] and worker._sha256(path) == partial["sha256"]
    raw = json.loads(path.read_text())
    assert raw["result_digest"] == partial["result_digest"]
    assert raw["candidate_policy_queried"] is final_cell_queried
    assert retained["result_digest"] == canonical_digest(retained, digest_field="result_digest")


def test_final_isolated_child_without_stop_keeps_healthy_aggregation(tmp_path, monkeypatch):
    runtime, output, inputs, _authority, _scope, *_ = stage_beta_bundle(tmp_path, monkeypatch)
    spawned = []

    def spawn(**kwargs):
        spawned.append(kwargs["index"])
        child = _completed_parent_fixture_child(inputs, kwargs["index"])
        (kwargs["child_root"] / worker.PROVIDER_RESULT_FILENAME).write_text(json.dumps(child))
        return 0

    assert worker._run_isolated_cell_processes(runtime_root=runtime, output_root=output, run_cell_process=spawn) == 0
    retained = json.loads((output / worker.PROVIDER_RESULT_FILENAME).read_text())
    assert spawned == list(range(len(inputs["cells"])))
    assert retained["status"] == "runtime_completed_unqualified_pending_closeout"
    assert retained["candidate_policy_queried"] is True
    assert len(retained["episodes"]) == len(inputs["cells"]) * len(inputs["candidate_ids"])
    assert "execution_control_stop" not in retained and "retained_partial_cell_results" not in retained


def test_final_invalid_child_without_retained_stop_still_refuses_aggregation(tmp_path, monkeypatch):
    runtime, output, inputs, _authority, _scope, *_ = stage_beta_bundle(tmp_path, monkeypatch)

    def spawn(**kwargs):
        child = _completed_parent_fixture_child(inputs, kwargs["index"])
        if kwargs["index"] == len(inputs["cells"]) - 1:
            child["status"] = "blocked"
            child["result_digest"] = canonical_digest(child, digest_field="result_digest")
        (kwargs["child_root"] / worker.PROVIDER_RESULT_FILENAME).write_text(json.dumps(child))
        return 1

    with pytest.raises(RuntimeError, match="policy_canary_isolated_cell_result_invalid"):
        worker._run_isolated_cell_processes(runtime_root=runtime, output_root=output, run_cell_process=spawn)
    assert not (output / worker.PROVIDER_RESULT_FILENAME).exists()


def test_actual_selected_cell_refuses_before_engine_creation(tmp_path, monkeypatch):
    runtime, output, _inputs, authority, scope, *_ = stage_beta_bundle(tmp_path, monkeypatch)
    install(runtime, authority, scope, now=1001, revision=2, stopped=True)
    monkeypatch.setattr(worker, "isaac_cell_runtime", lambda: pytest.fail("engine must not be opened"))
    assert worker._run_selected_cell(0, runtime_root=runtime, output_root=output) == 1
    result = json.loads((output / worker.PROVIDER_RESULT_FILENAME).read_text())
    assert result["applicability"] == "historical_only" and result["candidate_policy_queried"] is False


@pytest.mark.parametrize("boundary", ["load", "query"])
def test_actual_selected_worker_guards_client_load_and_every_query(tmp_path, monkeypatch, boundary):
    from tests.test_native_task_arena_policy_canary_lifecycle_rehearsal import FakeIsaac, _rehearsal_runtime
    runtime, output, inputs, authority, scope, *_ = stage_beta_bundle(tmp_path, monkeypatch)
    loads, calls = [], []
    def client(spec, **kwargs):
        loads.append(spec["candidate_id"])
        return SimpleNamespace(infer=lambda observation: calls.append(observation) or {"actions": []})
    bound = replace(_rehearsal_runtime(FakeIsaac(output / worker.PROVIDER_RESULT_FILENAME)), policy_client=client)
    def drive_callbacks(**kwargs):
        if boundary == "load":
            install(runtime, authority, scope, now=1001, revision=2, stopped=True)
        policy = kwargs["load_policy"]({}, inputs["candidate_ids"][0])
        policy["client"].infer({"query": 1})
        install(runtime, authority, scope, now=1001, revision=2, stopped=True)
        policy["client"].infer({"query": 2})
        pytest.fail("a stopped query cannot be issued")
    monkeypatch.setattr(worker, "execute_paired_session", drive_callbacks)
    with pytest.raises(control.RuntimeControlStopped):
        worker._run_selected_cell(0, runtime_root=runtime, output_root=output, cell_runtime=bound)
    assert calls == ([] if boundary == "load" else [{"query": 1}])
    assert loads == ([] if boundary == "load" else [inputs["candidate_ids"][0]])


def test_policy_tracker_retains_earlier_query_truth_when_guard_refuses_next_inference(tmp_path, monkeypatch):
    runtime, inputs, authority, scope = staged_control(tmp_path, monkeypatch)
    install(runtime, authority, scope)
    guard = control.RuntimeControlGuard(runtime_root=runtime, inputs=inputs, authority=authority)
    queries = []
    tracker = policy_worker._PolicyQueryTracker(SimpleNamespace(infer=lambda observation: queries.append(observation) or "action"),
                                               before_query=lambda: guard.check("before_policy_query"))
    assert tracker.infer({"first": True}) == "action"
    install(runtime, authority, scope, now=1001, revision=2, stopped=True)
    with pytest.raises(control.RuntimeControlStopped):
        tracker.infer({"second": True})
    assert queries == [{"first": True}] and tracker.candidate_policy_queried is True


def test_fixed_ssh_push_uses_existing_pin_and_no_service_credential(tmp_path, monkeypatch):
    from blueprint_pipeline import gpu_render_providers as ssh
    from blueprint_pipeline import vast_provider_output_recovery as transport
    runtime, _inputs, authority, scope = staged_control(tmp_path, monkeypatch)
    value = response(scope=scope, deadline=3000)
    calls = []
    monkeypatch.setenv("PIPELINE_SYNC_TOKEN", "must-never-reach-gpu")
    monkeypatch.setattr(ssh, "_validated_vast_known_hosts_pin", lambda *a, **k: (tmp_path / "pinned_hosts", "fingerprint"))
    monkeypatch.setattr(transport, "_identity_file", lambda: tmp_path / "existing-private-identity")
    def send(command, **kwargs):
        calls.append((command, kwargs))
        assert command[-1] == delivery.REMOTE_COMMAND
        assert "must-never-reach-gpu" not in kwargs["input"].decode()
        parsed = json.loads(kwargs["input"])
        stored = control.install_runtime_control(runtime_root=runtime, delivery=parsed)
        return SimpleNamespace(returncode=0, stdout=json.dumps({"status": "delivered", "binding_digest": value["binding_digest"],
            "control_digest": value["control_digest"], "delivery_digest": stored["delivery_digest"]}).encode())
    monkeypatch.setattr(delivery.subprocess, "run", send)
    connection = {"ssh_host": "provider.example.com", "ssh_port": 22001, "instance_id": 41, "resource_name": authority["resource_name"]}
    assert delivery.push_runtime_control(connection=connection, attempt_dir=tmp_path, authority=authority, binding=scope, control=value)["status"] == "delivered"
    assert len(calls) == 1 and calls[0][1]["timeout"] == 20
    assert "StrictHostKeyChecking=yes" in calls[0][0]
    with pytest.raises(ValueError, match="owner_unproven"):
        delivery.push_runtime_control(connection={**connection, "resource_name": "unrelated-site"}, attempt_dir=tmp_path,
                                      authority=authority, binding=scope, control=value)
    assert len(calls) == 1


def test_provider_callback_requires_own_instance_and_pin_and_unsubscribes_once(tmp_path, monkeypatch):
    from blueprint_pipeline import gpu_render_providers as ssh
    calls, closed = [], []
    monkeypatch.setattr(ssh, "_validated_vast_known_hosts_pin", lambda *a, **k: None)
    wrapped, close = adapter._runtime_control_probe(probe=lambda connection: {"status": "observed", "milestones": []},
        ready=lambda connection: calls.append(connection) or (lambda: closed.append(True)),
        job_dir=tmp_path, instance_id=41, resource_name="owned-one")
    connection = {"observed": True, "instance_id": 41, "resource_name": "owned-one", "ssh_host": "provider.example.com", "ssh_port": 22001}
    wrapped(connection)
    assert not calls
    monkeypatch.setattr(ssh, "_validated_vast_known_hosts_pin", lambda *a, **k: (tmp_path / "known_hosts", "fingerprint"))
    wrapped({**connection, "instance_id": 42})
    wrapped({**connection, "resource_name": "other"})
    assert not calls
    wrapped(connection)
    wrapped(connection)
    assert len(calls) == 1
    close()
    close()
    assert closed == [True]


@pytest.mark.parametrize("hosted", [False, True])
def test_existing_beta_transport_requires_and_invokes_exact_monitor_hook(tmp_path, monkeypatch, hosted):
    from blueprint_pipeline import team_candidate_quick10_vast, native_task_arena_vast
    transport = team_candidate_quick10_vast if hosted else native_task_arena_vast
    runtime, _output, _inputs, authority, scope, prepared, bundle, candidate = stage_beta_bundle(tmp_path, monkeypatch, hosted=hosted)
    monkeypatch.setenv("BLUEPRINT_POLICY_CANARY_OUTPUT_DELIVERY", "download")
    required, subscribed, unsubscribed = [], [], []
    def require(**kwargs):
        required.append(kwargs)
    def subscribe(**kwargs):
        subscribed.append(kwargs)
        kwargs["sink"](scope, response(scope=scope, deadline=3000, stopped=True, revision=2))
        return lambda: unsubscribed.append(True)
    monkeypatch.setitem(sys.modules, "blueprint_pipeline.beta_policy_screening_control", SimpleNamespace(
        require_runtime_control_monitor=require, subscribe_runtime_control=subscribe))
    def push(**kwargs):
        stored = control.make_delivery(authority=kwargs["authority"], binding=kwargs["binding"], control=kwargs["control"])
        control.install_runtime_control(runtime_root=runtime, delivery=stored)
        return {"status": "delivered"}
    monkeypatch.setattr(delivery, "push_runtime_control", push)
    def allocate(**kwargs):
        unsubscribe = kwargs["runtime_control_ready"]({"resource_name": authority["resource_name"], "instance_id": 41,
            "ssh_host": "provider.example.com", "ssh_port": 22001, "control_pin_dir": str(tmp_path / "pin")})
        try:
            guard = control.RuntimeControlGuard(runtime_root=runtime, inputs=launch._read(runtime / "runtime_inputs/policy_canary_runtime_inputs.json"), authority=authority)
            with pytest.raises(control.RuntimeControlStopped):
                guard.check("before_cell_spawn")
            return {"status": "blocked"}
        finally:
            unsubscribe()
    monkeypatch.setattr(transport, "run_arena_native_control_vast", allocate)
    run = transport.run_team_candidate_session_vast if hosted else transport.run_native_task_arena_policy_canary_session_vast
    result = run(job_dir=tmp_path / "allocator", prepared_bundle=bundle,
        session_authority=authority, paid_resource_admission_grant=None, execute=True,
        **({"credential_file": Path(candidate["execution_spec"]["credential_file"])} if hosted else {}),
        hard_cap_usd=4.0, hard_ttl_seconds=authority["hard_ttl_seconds"])
    assert result["status"] == "blocked"
    assert required == [{"authority_digest": authority["authority_digest"], "resource_name": authority["resource_name"]}]
    assert len(subscribed) == len(unsubscribed) == 1


def test_team_execution_uses_no_blueprint_monitor_hook():
    assert delivery.runtime_control_ready_for_authority(authority={"run_kind": "team_candidate_evaluation"}, execute=True) is None


def test_interrupted_subscription_close_cannot_skip_owned_teardown(tmp_path, monkeypatch):
    from blueprint_pipeline import gpu_render_providers as ssh
    events = []
    def unsubscribe():
        events.append("unsubscribe")
        raise KeyboardInterrupt("control listener interrupted")
    monkeypatch.setattr(ssh, "_validated_vast_known_hosts_pin", lambda *a, **k: (tmp_path / "known_hosts", "fingerprint"))
    relay = delivery.RuntimeControlRelay(ready=lambda connection: unsubscribe,
        provider_bundle_kind="native_task_arena_policy_canary_session", resource_name="owned-one")
    probe = relay.wrap(lambda connection: {"status": "observed", "milestones": []}, job_dir=tmp_path, instance_id=41)
    probe({"observed": True, "instance_id": 41, "resource_name": "owned-one", "ssh_host": "provider.example.com", "ssh_port": 22001})
    relay.close()
    events.append("owned_teardown")
    relay.close()
    assert events == ["unsubscribe", "owned_teardown"]


def test_beta_authority_stop_cannot_be_ignored_by_legacy_local_signal_option(tmp_path, monkeypatch):
    from tests.test_vast_provider_adapter import _configure_live_gates, _paid_grant
    _configure_live_gates(tmp_path, monkeypatch)
    monkeypatch.setenv(adapter.VAST_LAUNCH_LOCK_FILE_ENV, str(tmp_path / "own-test-launch.lock"))
    monkeypatch.setenv(adapter.HF_TOKEN_FILE_ENV, str(tmp_path / "missing_hf_token"))
    monkeypatch.setenv("BLUEPRINT_VAST_IGNORE_LOCAL_SIGTERM_DURING_PROVIDER_RUN", "true")
    monkeypatch.setenv(adapter.VAST_EMPTY_OFFER_SEARCH_RETRY_ATTEMPTS_ENV, "0")
    monkeypatch.setattr(adapter.time, "sleep", lambda *_: None)
    captured = {}
    def capture(signum, handler):
        if callable(handler):
            captured[signum] = handler
        return None
    monkeypatch.setattr(adapter.signal, "signal", capture)
    def fake_api(**kwargs):
        if kwargs["method"] == "GET" and kwargs["path"] == "/instances/":
            return 200, {"instances": []}
        if kwargs["method"] == "POST":
            captured[adapter.signal.SIGTERM](adapter.signal.SIGTERM, None)
            pytest.fail("Beta authority stop must unwind provider execution")
        pytest.fail("unexpected provider API boundary")
    monkeypatch.setattr(adapter, "_api_json", fake_api)
    result = adapter.run_vast_provider_adapter(job_dir=tmp_path / "stopped", mode="live-startup-probe",
        paid_resource_admission_grant=_paid_grant(), allow_vast_api_call=True, allow_instance_launch=True,
        session_max_live_minutes=None, provider_bundle_kind="native_task_arena_policy_canary_session",
        instance_label_exact="blueprint-native-task-policy-canary-beta-exact", runtime_control_ready=lambda connection: lambda: None)
    assert result["reason"] == "vast_probe_interrupted"
    receipt = json.loads((tmp_path / "stopped/vast_signal_handling_manifest.json").read_text())
    assert receipt["status"] == "probe_interrupted_by_signal"


@pytest.mark.parametrize("hosted", [False, True])
def test_screening_create_intent_runs_only_inside_the_adapter_pre_create_hook(tmp_path, monkeypatch, hosted):
    from blueprint_pipeline import team_candidate_quick10_vast, native_task_arena_vast
    transport = team_candidate_quick10_vast if hosted else native_task_arena_vast
    _runtime, _output, _inputs, authority, _scope, _prepared, bundle, candidate = stage_beta_bundle(
        tmp_path, monkeypatch, hosted=hosted)
    monkeypatch.setenv("BLUEPRINT_POLICY_CANARY_OUTPUT_DELIVERY", "download")
    monkeypatch.setitem(sys.modules, "blueprint_pipeline.beta_policy_screening_control", SimpleNamespace(
        require_runtime_control_monitor=lambda **kwargs: None,
        subscribe_runtime_control=lambda **kwargs: (lambda: None)))
    created = []

    def allocate(**kwargs):
        # Consumption is done; staging and watchdog arming run before the adapter's hook.
        assert created == []
        consumed = kwargs["pre_provider_mutation_hook"]()
        assert consumed["status"] == "consumed" and created == ["intent"]
        return {"status": "blocked"}

    monkeypatch.setattr(transport, "run_arena_native_control_vast", allocate)
    run = transport.run_team_candidate_session_vast if hosted else transport.run_native_task_arena_policy_canary_session_vast
    result = run(job_dir=tmp_path / "allocator", prepared_bundle=bundle,
        session_authority=authority, paid_resource_admission_grant=None, execute=True,
        **({"credential_file": Path(candidate["execution_spec"]["credential_file"])} if hosted else {}),
        hard_cap_usd=4.0, hard_ttl_seconds=authority["hard_ttl_seconds"],
        pre_provider_create=lambda: created.append("intent"))
    assert result["status"] == "blocked" and created == ["intent"]


def test_beta_cell_that_cannot_open_isaac_seals_a_named_blocker_and_stops(tmp_path, monkeypatch):
    from blueprint_pipeline import policy_canary_cell_setup_failure as setup_failure
    from tests.test_native_task_arena_policy_canary_lifecycle_rehearsal import FakeIsaac, _rehearsal_runtime
    runtime, output, inputs, _authority, _scope, *_ = stage_beta_bundle(tmp_path, monkeypatch)
    spawned = []

    def launch(*_args, **_kwargs):
        raise RuntimeError("isaac_kit_failed_to_start")

    def spawn(**kwargs):
        spawned.append(kwargs["index"])
        bound = replace(_rehearsal_runtime(FakeIsaac(kwargs["child_root"] / worker.PROVIDER_RESULT_FILENAME)),
                        launch_isaac=launch)
        return worker._run_selected_cell(kwargs["index"], runtime_root=runtime, output_root=kwargs["child_root"],
                                         provider_output_root=output, cell_runtime=bound)

    assert worker._run_isolated_cell_processes(runtime_root=runtime, output_root=output, run_cell_process=spawn) == 1
    assert spawned == [0] and not (output / "cell_runs/01").exists()
    result = json.loads((output / worker.PROVIDER_RESULT_FILENAME).read_text())
    assert result["blockers"] == ["policy_canary_session_setup_failed:runtime_open"]
    assert result["run_kind"] == inputs["run_kind"] and result["candidate_ids"] == inputs["candidate_ids"]
    assert result["beta_screening_admission"] == inputs["beta_screening_admission"]
    assert result["planned_episodes_per_policy"] == len(inputs["cells"])
    assert setup_failure.sealed_setup_failure_blocker(result) == "policy_canary_session_setup_failed:runtime_open"


# Website beta screening, capture GPU runs of 2026-11-04/05: every cell reached the
# pre-policy runtime observation gate and stopped there with these exact blockers,
# while its frames were structurally usable. None protects privacy, rights or money.
REAL_GATE_BLOCKERS = (
    "droid_wrist_task_pixels_below_threshold",
    *(f"{role}:{target}:composition_support_background_occlusion_requires_review"
      for role in ("external", "overview") for target in ("task_object", "task_target_marker")),
    "policy_canary_task_semantic_visibility_failed",
)


def runtime_observation_gate(*, backend_digest, frame_structure_passed=True, blockers=REAL_GATE_BLOCKERS):
    """The sealed runtime camera gate in the shape the native worker writes it."""
    value = {"schema_version": "policy_canary_runtime_observation_integrity_gate.v1", "status": "blocked",
             "run_kind": "internal_policy_canary", "claim_ceiling": "diagnostic_policy_execution",
             "appearance_render_backend_receipt_digest": backend_digest,
             "wrist_camera_mount_selection_digest": "sha256:" + "4" * 64,
             "frame_structure_passed": frame_structure_passed, "target_semantic_visibility_passed": False,
             # Recorded, as on the real gate, but never in its blocker list.
             "visual_gate": {"frame_structure_passed": frame_structure_passed,
                             "policy_observation_integrity_blockers": [
                                 "native_task_appearance_reference_parity_missing",
                                 "native_task_human_visual_review_not_approved"]},
             "human_visual_review_status": "not_required_for_internal_diagnostic_policy_execution",
             "candidate_policy_loaded": False, "candidate_policy_queried": False,
             "official_ranking_permitted": False, "scene_promotion_permitted": False,
             "blockers": sorted(blockers), "policy_observation_integrity_passed": False, "gate_digest": ""}
    value["gate_digest"] = canonical_digest(value, digest_field="gate_digest")
    return value


def _beta_cell_runtime(child_root, *, frame_structure_passed=True, policy_client=None):
    """The rehearsal runtime with the real gate shape, the real renderer guard and a beta-bound client.

    Only Isaac, the policy server and RoboLab's outer loop are faked: the native
    episode runner (and its prestart readiness) and the post-gate guard are real.
    """
    from blueprint_pipeline.adp009d_policy_episode import run_policy_episode
    from blueprint_pipeline.native_task_rtx_streaming_guard import configure_post_gate_rtx_streaming_wait
    from blueprint_pipeline.openpi_droid_policy_runtime import OpenPIDroidPolicySpec, OpenPIWebsocketDroidPolicyClient
    from tests.test_native_task_arena_policy_canary_lifecycle_rehearsal import FakeIsaac, _OpenPIVendor, _rehearsal_runtime

    def gate(**kwargs):
        backend = worker.appearance_render_backend_from_plan(kwargs["plan"], packet_request=kwargs["packet_request"] or None)
        return runtime_observation_gate(backend_digest=backend["receipt_digest"],
                                        frame_structure_passed=frame_structure_passed)

    def client(spec, **_kwargs):
        policy_spec = OpenPIDroidPolicySpec(**spec["policy_spec"])

        class Vendor(_OpenPIVendor):
            def get_server_metadata(self):
                return {**super().get_server_metadata(),
                        "local_checkpoint_object_count": policy_spec.checkpoint_object_count,
                        "local_checkpoint_size_bytes": policy_spec.checkpoint_size_bytes}
        return OpenPIWebsocketDroidPolicyClient(spec=policy_spec, host=str(spec["policy_endpoint"]["host"]),
            port=int(spec["policy_endpoint"]["port"]), client_factory=lambda **_: Vendor(policy_spec))

    renderer = SimpleNamespace(_STREAMING_WAIT_TIMEOUT_S=30.0, _get_stage_streaming_busy=lambda: False)
    return replace(_rehearsal_runtime(FakeIsaac(child_root / worker.PROVIDER_RESULT_FILENAME)),
        prepolicy_camera_gate=gate, policy_client=policy_client or client,
        configure_post_gate_renderer=lambda **kw: configure_post_gate_rtx_streaming_wait(**kw, renderer_utils=renderer),
        # RoboLab's upstream loop only drives the same native episode events; it is not installed here.
        run_policy_episode=lambda **kw: run_policy_episode(**{k: v for k, v in kw.items() if k != "evaluation_backend"}))


def _run_beta_cell(runtime, output, index, **runtime_kwargs):
    child_root = output / "cell_runs" / f"{index:02d}"
    child_root.mkdir(parents=True, exist_ok=True)
    try:
        worker._run_selected_cell(index, runtime_root=runtime, output_root=child_root, provider_output_root=output,
                                  cell_runtime=_beta_cell_runtime(child_root, **runtime_kwargs))
    except SystemExit:  # The fake Isaac's close ends the interpreter, as Isaac does.
        pass
    return json.loads((child_root / worker.PROVIDER_RESULT_FILENAME).read_text()), child_root


@pytest.fixture
def _no_robolab_install(monkeypatch):
    from blueprint_pipeline import beta_robolab_runtime
    monkeypatch.setattr(beta_robolab_runtime, "preflight_robolab_runtime", lambda: {"status": "passed"})


@pytest.mark.usefixtures("_no_robolab_install")
def test_beta_cell_runs_past_a_usable_frame_gate_and_carries_its_blockers_as_flags(tmp_path, monkeypatch):
    runtime, output, inputs, *_ = stage_beta_bundle(tmp_path, monkeypatch)
    flags = sorted(REAL_GATE_BLOCKERS)
    child, child_root = _run_beta_cell(runtime, output, 0)

    assert child["status"] == "runtime_selected_cell_completed_pending_aggregation"
    assert child["preload_observation_gate"]["policy_observation_integrity_passed"] is False  # verdict kept
    assert child["observation_quality_flags"] == flags and child["observation_integrity_flagged"] is True
    assert child["candidate_policy_queried"] is True and child["policy_loads"] == [
        {"candidate_id": inputs["candidate_ids"][0], "loaded_once": True}]
    episode, = child["episodes"]
    assert episode["status"] == "completed" and episode["candidate_policy_queried"] is True
    quality = episode["episode"]["prestart_readiness"]["prepolicy_visual_quality"]
    assert quality["observation_integrity_flagged"] is True and quality["observation_quality_flags"] == flags
    # The measured integrity verdict is never rewritten into a pass.
    assert quality["policy_observation_integrity_passed"] is False
    assert quality["runtime_observation_gate"]["gate_digest"] == child["preload_observation_gate"]["gate_digest"]
    guard = json.loads((child_root / "prepolicy_observation_gate/post_gate_rtx_streaming_guard.v1.json").read_text())
    assert guard["status"] == "configured" and guard["observation_quality_flagged"] is True
    assert guard["observation_gate_digest"] == child["preload_observation_gate"]["gate_digest"]
    # The gate did not pass, so no cell milestone says it did.
    assert "observation_gate_passed" not in (child_root / "cell_progress.log").read_text()

    # The parent aggregates the run: the flags stay with the provider session result.
    def spawn(**kwargs):
        if kwargs["index"] == 0:
            (kwargs["child_root"] / worker.PROVIDER_RESULT_FILENAME).write_text(json.dumps(child))
        else:
            (kwargs["child_root"] / worker.PROVIDER_RESULT_FILENAME).write_text(
                json.dumps(_completed_parent_fixture_child(inputs, kwargs["index"])))
        return 0

    parent = tmp_path / "parent_output"
    assert worker._run_isolated_cell_processes(runtime_root=runtime, output_root=parent, run_cell_process=spawn) == 0
    retained = json.loads((parent / worker.PROVIDER_RESULT_FILENAME).read_text())
    assert retained["status"] == "runtime_completed_unqualified_pending_closeout"
    assert retained["observation_quality_flags"] == flags and retained["observation_integrity_flagged"] is True
    assert "cell_setup_failure" not in retained
    assert retained["result_digest"] == canonical_digest(retained, digest_field="result_digest")


@pytest.mark.usefixtures("_no_robolab_install")
def test_beta_cell_whose_frames_are_not_structurally_usable_still_stops(tmp_path, monkeypatch):
    from blueprint_pipeline import policy_canary_cell_setup_failure as setup_failure
    runtime, output, inputs, *_ = stage_beta_bundle(tmp_path, monkeypatch)

    def spawn(**kwargs):
        _run_beta_cell(runtime, output, kwargs["index"], frame_structure_passed=False)
        return 1

    assert worker._run_isolated_cell_processes(runtime_root=runtime, output_root=output, run_cell_process=spawn) == 1
    child = json.loads((output / "cell_runs/00" / worker.PROVIDER_RESULT_FILENAME).read_text())
    assert child["status"] == "blocked" and child["policy_loads"] == [] and child["episodes"] == []
    assert child["candidate_policy_queried"] is False
    assert "observation_quality_flags" not in child and "observation_integrity_flagged" not in child
    assert not (output / "cell_runs/00/prepolicy_observation_gate/post_gate_rtx_streaming_guard.v1.json").exists()
    result = json.loads((output / worker.PROVIDER_RESULT_FILENAME).read_text())
    assert result["blockers"] == ["policy_canary_session_setup_failed:observation_gate"]
    assert result["cell_setup_failure"]["observation_gate_verdict"] == "refused"
    assert setup_failure.sealed_setup_failure_blocker(result) == "policy_canary_session_setup_failed:observation_gate"


@pytest.mark.usefixtures("_no_robolab_install")
def test_a_flagged_gate_the_session_continued_past_is_not_a_setup_failure_of_the_gate(tmp_path, monkeypatch):
    from blueprint_pipeline import policy_canary_cell_setup_failure as setup_failure
    runtime, output, *_ = stage_beta_bundle(tmp_path, monkeypatch)

    def unreachable_server(*_args, **_kwargs):
        raise ConnectionError("policy server did not answer")

    def spawn(**kwargs):
        _run_beta_cell(runtime, output, kwargs["index"], policy_client=unreachable_server)
        return 1

    assert worker._run_isolated_cell_processes(runtime_root=runtime, output_root=output, run_cell_process=spawn) == 1
    result = json.loads((output / worker.PROVIDER_RESULT_FILENAME).read_text())
    failure = result["cell_setup_failure"]
    assert result["blockers"] == ["policy_canary_session_setup_failed:policy_load"]
    assert failure["stage"] == "policy_load" and failure["observation_gate_verdict"] == "flagged"
    assert failure["observation_quality_flags"] == sorted(REAL_GATE_BLOCKERS)
    assert setup_failure.sealed_setup_failure_blocker(result) == "policy_canary_session_setup_failed:policy_load"
