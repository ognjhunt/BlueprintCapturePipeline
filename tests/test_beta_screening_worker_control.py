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
    else:
        assert retained["isolated_cell_failures"][0]["raw_child_result"]["fixture"] == "unaccepted_raw_prefix"
    assert retained["episodes"] == []


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
