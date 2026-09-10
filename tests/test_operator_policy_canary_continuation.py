"""Existing-run handoff, stage ordering, owned teardown, and crash-safe custody."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from blueprint_pipeline import operator_policy_canary_continuation as coordinator
from blueprint_pipeline import operator_policy_canary_handoff as handoff
from blueprint_pipeline.decision_evidence_contracts import canonical_digest, cross_runtime_canonical_digest
from blueprint_pipeline.operator_policy_canary_terminal_delivery import file_record
from blueprint_pipeline.provider_output_range_ingestion import ingest_provider_output
from tests.test_provider_output_range_ingestion import _binding, store as store  # shared real HTTP fixture


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


@pytest.fixture
def fixture(tmp_path, store):  # noqa: F811 - imported shared pytest fixture
    root = tmp_path / "run-1"
    root.mkdir()
    binding, url = _binding(tmp_path, store)
    staging = Path(binding["staging_manifest"]["path"])
    manifest = json.loads(staging.read_text()) | {"bundle_sha256": "a" * 64,
        "output_key": "exact/run/output.zip", "bundle_key": "exact/run/bundle.zip",
        "object_store": {"key_prefix": "exact/run"}, "bundle_object_retained_for_reuse": False}
    write(staging, manifest)
    binding["staging_manifest"] = file_record(staging)
    binding["binding_digest"] = canonical_digest(binding, digest_field="binding_digest")
    name = "blueprint-native-task-policy-canary-" + "a" * 32
    old = write(tmp_path / "original/watchdog.json", {"schema_version": "groot_oscar_runpod_canary_watchdog.v1",
        "status": "armed", "provider": "vast", "independent_process": True, "deadline_epoch": 2000.,
        "pod_name_prefix": name, "resource_name_exact": name, "pid": 123})
    started = tmp_path / "original/started.txt"
    started.write_text("50518293\n")
    owner = {"pid": 123, "started_at_local": "fixture-before-launch", "hostname": "owner-mac",
        "state": "S", "command_sha256": "sha256:" + "e" * 64, "required_command_tokens_verified": True,
        "process_identity_digest": "sha256:" + "f" * 64}
    terminal = {"run_id": "run-1", "run_root": str(root), "allocator_result_path": str(root / "allocator_result.json"),
        "provider_zero_path": str(root / "post_teardown_global_provider_zero.json"),
        "native_result_path": str(root / "provider_output/native/runtime/result.json"),
        "official_billing_path": str(root / "billing.json"), "billing_audit_root": str(root / "billing-audit")}
    registration = {'schema_version': 'task_evaluation_operator_policy_canary_registration.v1',
        'run_id': 'run-1', 'run_kind': 'internal_policy_canary', 'claim_ceiling': 'diagnostic_policy_execution'}
    registration['registration_digest'] = cross_runtime_canonical_digest(registration, digest_field='registration_digest')
    terminal['records'] = {'registration': file_record(write(root / 'immutable-inputs/registration.json', registration))}
    values = {"session_authority": {"run_id": "run-1", "hard_cap_usd": 3., "resource_name": name, "retry_cap": 0},
        "operator_authorization": {"policy_cap_usd": 3.}, "bundle": {"implementation_commit": "c" * 40,
        "bundle_sha256": "sha256:" + "a" * 64, "retry_cap": 0},
        "runtime_inputs": {"runtime_inputs_digest": binding["runtime_inputs_digest"]}}
    watchroot = root / "watchdog"
    intent = coordinator.build_continuation_intent(terminal_delivery_intent=terminal, ingestion_binding=binding,
        instance_id=50518293, resource_name=name, deadline_epoch=2000., scientific_commit="c" * 40, hard_cap_usd=3.,
        original_watchdog=file_record(old), original_started_instance=file_record(started),
        local_allocator_identity=owner, local_allocator_marker="/original/allocator",
        control_plane_machine_id_sha256="sha256:" + "d" * 64, watchdog_root=watchroot,
        signed_get_url_file=url, billing_period_start_at="2026-09-10T00:00:00Z")
    write(watchroot / coordinator.watchdog.EVIDENCE_NAME,
        json.loads(old.read_text()) | {"pid": 999999, "watchdog_out_dir": str(watchroot)})
    (watchroot / coordinator.watchdog.VAST_STARTED_INSTANCE_ID_NAME).write_text("50518293\n")
    intent_path = write(root / 'continuation_intent.json', intent)
    def host_reader():
        return {'platform': 'Linux', 'machine_id_sha256': intent['control_plane_machine_id_sha256']}
    cloud_process = owner | {'pid': 999999, 'hostname': 'cloud', 'state': 'S'}
    handoff.record_watchdog_process(intent, intent_path, process_reader=lambda *args, **kwargs: cloud_process)
    events = []
    clock = [1000.]
    inventory = {"status": "observed", "api_confirmed": True, "live_resource_count": 1,
                 "resources": [{"instance_id": "50518293", "name": name}]}
    readiness = handoff.cloud_readiness(intent, inventory_reader=lambda _: inventory,
        inputs_validator=lambda _: values, clock=lambda: clock[0],
        host_reader=host_reader,
        process_reader=lambda pid, **kwargs: owner | {"pid": pid, "hostname": "cloud", "state": "S"})

    def commit():
        value = handoff.commit_owner_handoff(intent, readiness, process_reader=lambda *args, **kwargs: owner | {"state": "T"},
                                            clock=lambda: clock[0])
        write(handoff.metadata_root(intent) / "owner_handoff_commit.json", value)

    def collect(**kwargs):
        events.append("collect")
        return ingest_provider_output(**kwargs, opener=store.opener,
            disk_usage_provider=lambda path: SimpleNamespace(free=100 * 1024**3))

    def terminate(intent):
        events.append("teardown")
        return {"status": "provider_terminal", "provider_absence_confirmed": True,
            "resource_name_exact": name, "deadline_epoch": 2000., "completed_at": "2026-09-10T21:00:00Z",
            "recorded_vast_instance": {"instance_id": "50518293"},
            "recorded_vast_instance_teardown": {"instance_id": "50518293", "provider_absence_confirmed": True},
            "final_inventory": {"api_confirmed": True, "live_resource_count": 0},
            "terminations": [{"instance_id": "50518293", "status": "terminated"}]}

    def cleanup(intent):
        events.append("cleanup")
        return {"schema_version": "wam_provider_object_store_cleanup.v1", "status": "completed",
            "staging_manifest_sha256": binding["staging_manifest"]["sha256"].removeprefix("sha256:"),
            "exact_object_count": 2, "all_objects_absent": True, "all_ephemeral_objects_absent": True,
            "objects": [{"key_sha256": hashlib.sha256(key.encode()).hexdigest(), "absence": {"absence_confirmed": True}}
                        for key in (manifest["bundle_key"], manifest["output_key"])], "blockers": []}

    def zero():
        events.append("zero")
        return handoff.seal_value({"schema_version": "task_evaluation_policy_canary_vast_provider_zero.v1",
            "status": "provider_zero_confirmed", "api_confirmed": True, "provider_zero_verified": True,
            "live_instance_count": 0, "blockers": []})

    def finalize(intent, **kwargs):
        events.append("finalize")
        value = handoff.seal_value({"status": "completed", "all_required_phases_done": True}, "result_digest")
        write(root / "operator_terminal_delivery/completed.json", value)
        return value

    adapters = coordinator.ExistingRunContinuationAdapters(collector=collect, terminator=terminate, cleanup=cleanup,
        provider_zero_reader=zero, finalizer=finalize, delivery_adapters_factory=lambda _: None,
        inputs_validator=lambda _: values, clock=lambda: clock[0], host_reader=host_reader)
    return SimpleNamespace(root=root, intent=intent, values=values, adapters=adapters, commit=commit,
                           events=events, clock=clock, owner=owner, readiness=readiness,
                           cloud_process=cloud_process, inventory=inventory, intent_path=intent_path)


def test_no_collection_or_mutation_before_explicit_owner_handoff(fixture):
    result = coordinator.continue_existing_run(fixture.intent, adapters=fixture.adapters)
    assert result["status"] == "pending"
    assert result["blockers"] == ["continuation_explicit_owner_handoff_pending"]
    assert fixture.events == []


def test_readiness_verifies_exact_registration_alias_and_direct_operator_root(fixture):
    alias = fixture.root / 'website-operator-registration.json'
    source = Path(fixture.intent['terminal_delivery_intent']['records']['registration']['path'])
    assert alias.read_bytes() == source.read_bytes()
    assert fixture.readiness['operator_artifact_run_root_verified'] is True
    assert fixture.readiness['operator_registration_alias'] == file_record(alias)
    from blueprint_pipeline.live_pipeline_result_artifact_resolution import _registered_operator_run_root
    assert _registered_operator_run_root(activation_root=fixture.root.parent / 'run-1-activation', run_id='run-1') == fixture.root


def test_existing_run_stages_are_ordered_and_completed_resume_is_noop(fixture):
    fixture.commit()
    result = coordinator.continue_existing_run(fixture.intent, adapters=fixture.adapters)
    assert result["status"] == "completed", result
    assert fixture.events == ["collect", "teardown", "cleanup", "zero", "finalize"]
    assert coordinator.continue_existing_run(fixture.intent, adapters=fixture.adapters) == result
    assert len(fixture.events) == 5
    assert not list(fixture.root.rglob("*.zip"))
    # Exercise the actual official billing terminal adapter against the emitted layout.
    from blueprint_pipeline.vast_official_billing_extractor import _terminal_evidence
    evidence = _terminal_evidence(instance_id=50518293, terminal_result_path=fixture.root / "allocator_result.json")
    assert evidence["provider_absence_confirmed"] is True
    assert evidence["terminal_result"]["sha256"] == file_record(fixture.root / "allocator_result.json")["sha256"]


def test_not_ready_archive_cannot_cause_early_teardown(fixture):
    fixture.commit()
    adapter = replace(fixture.adapters, collector=lambda **kwargs: {"status": "not_ready"})
    result = coordinator.continue_existing_run(fixture.intent, adapters=adapter)
    assert result["status"] == "pending"
    assert fixture.events == []


def test_deadline_teardown_preserves_uncollected_output_object(fixture):
    fixture.commit()
    fixture.clock[0] = 2000.
    adapter = replace(fixture.adapters, collector=lambda **kwargs: {"status": "not_ready"})
    result = coordinator.continue_existing_run(fixture.intent, adapters=adapter)
    assert result["status"] == "pending"
    assert fixture.events == ["teardown"]
    receipt = json.loads((fixture.root / "vast_provider_run/vast_teardown_manifest.json").read_text())
    assert receipt["trigger"] == "original_hard_deadline"


@pytest.mark.parametrize("field,value", [("hard_cap_usd", 3.01), ("deadline_epoch", 2001.),
    ("instance_id", 50518294), ("scientific_commit", "b" * 40), ("new_provider_allocations_permitted", 1)])
def test_intent_cannot_rebind_original_budget_resource_deadline_or_code(fixture, field, value):
    intent = fixture.intent | {field: value}
    handoff.seal_value(intent, "intent_digest")
    with pytest.raises(handoff.ContinuationError):
        coordinator.continue_existing_run(intent, adapters=fixture.adapters)
    assert fixture.events == []


@pytest.mark.parametrize("crash_file", ["collected.json", "vast_teardown_manifest.json", "staged_object_cleanup.json",
    "post_teardown_global_provider_zero.json", "allocator_result.json"])
def test_crash_after_durable_phase_never_repeats_collection_or_closure(fixture, monkeypatch, crash_file):
    fixture.commit()
    real = coordinator._seal
    fired = []
    def crash(path, value):
        result = real(path, value)
        if Path(path).name == crash_file and not fired:
            fired.append(True)
            raise SystemExit("fixture crash after durable phase")
        return result
    monkeypatch.setattr(coordinator, "_seal", crash)
    with pytest.raises(SystemExit):
        coordinator.continue_existing_run(fixture.intent, adapters=fixture.adapters)
    result = coordinator.continue_existing_run(fixture.intent, adapters=fixture.adapters)
    assert result["status"] == "completed", result
    assert fixture.events == ["collect", "teardown", "cleanup", "zero", "finalize"]


def test_handoff_rejects_live_allocator_or_reused_pid(fixture):
    for observed in (fixture.owner, fixture.owner | {"state": "T", "process_identity_digest": "sha256:" + "0" * 64}):
        with pytest.raises(handoff.ContinuationError, match="pause_not_proven"):
            handoff.commit_owner_handoff(fixture.intent, fixture.readiness,
                process_reader=lambda *args, **kwargs: observed, clock=lambda: 1000.)


def test_exact_provider_wrapper_never_terminates_an_unowned_same_label_id():
    calls = []
    provider = coordinator._ExactExistingProvider(SimpleNamespace(terminate=lambda value: calls.append(value)), 50518293)
    with pytest.raises(handoff.ContinuationError, match="unowned_resource"):
        provider.terminate("50518294")
    assert calls == []


def test_job_lock_prevents_competing_collectors(fixture):
    import fcntl
    fixture.commit()
    with (handoff.metadata_root(fixture.intent) / ".lock").open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = coordinator.continue_existing_run(fixture.intent, adapters=fixture.adapters)
    assert result["blockers"] == ["continuation_already_running"]
    assert fixture.events == []


def test_secret_urls_and_exception_text_never_reach_coordinator_receipts(fixture):
    fixture.commit()
    def fail(**kwargs):
        raise RuntimeError("https://private/?token=SECRET_DO_NOT_RECORD")
    coordinator.continue_existing_run(fixture.intent, adapters=replace(fixture.adapters, collector=fail))
    for path in handoff.metadata_root(fixture.intent).rglob("*.json"):
        assert "SECRET_DO_NOT_RECORD" not in path.read_text()


def test_readiness_cannot_be_claimed_on_the_owners_machine(fixture):
    with pytest.raises(handoff.ContinuationError):
        handoff.cloud_readiness(fixture.intent, inventory_reader=lambda _: {},
            inputs_validator=lambda _: fixture.values, host_reader=lambda: {"platform": "Darwin"},
            process_reader=lambda *args, **kwargs: fixture.owner, clock=lambda: 1000.)


def test_collection_after_gpu_deadline_has_independent_cpu_budget(fixture):
    fixture.commit()
    fixture.clock[0] = fixture.intent['deadline_epoch'] + 100
    collect = fixture.adapters.collector
    def bounded_collection(**kwargs):
        assert fixture.events == ['teardown']
        assert kwargs['deadline_seconds'] == 3600
        return collect(**kwargs)
    result = coordinator.continue_existing_run(
        fixture.intent, adapters=replace(fixture.adapters, collector=bounded_collection))
    assert result['status'] == 'completed', result
    assert fixture.events == ['teardown', 'collect', 'cleanup', 'zero', 'finalize']


def test_every_continuation_tick_rejects_copied_handoff_on_another_host(fixture):
    fixture.commit()
    def wrong_host():
        return {'platform': 'Linux', 'machine_id_sha256': 'sha256:' + '0' * 64}
    with pytest.raises(handoff.ContinuationError, match='control_plane_host_mismatch'):
        coordinator.continue_existing_run(fixture.intent, adapters=replace(fixture.adapters, host_reader=wrong_host))
    assert fixture.events == []
    with pytest.raises(handoff.ContinuationError, match='control_plane_host_mismatch'):
        coordinator.run_existing_watchdog(fixture.intent, intent_path=fixture.intent_path, host_reader=wrong_host)
    assert fixture.events == []


def test_readiness_rejects_reused_watchdog_pid_and_checks_exact_intent_argument(fixture):
    observed = []
    def process_reader(pid, **kwargs):
        observed.append(kwargs['required_argument_pairs'])
        return fixture.cloud_process | {'process_identity_digest': 'sha256:' + '0' * 64,
                                         'started_at_local': 'different process birth'}
    with pytest.raises(handoff.ContinuationError, match='cloud_readiness_unproven'):
        handoff.cloud_readiness(fixture.intent, inventory_reader=lambda _: fixture.inventory,
            process_reader=process_reader, host_reader=fixture.adapters.host_reader,
            inputs_validator=lambda _: fixture.values, clock=lambda: 1000.)
    assert observed == [(('--intent', str(fixture.intent_path.resolve())),)]


@pytest.mark.parametrize('command', [
    'python -m module watchdog --intent /wrong/intent.json /exact/intent.json',
    'python -m module watchdog --intent /exact/intent.json --intent /wrong/intent.json',
])
def test_process_identity_rejects_wrong_or_duplicated_intent_option(command):
    def runner(argv, **kwargs):
        return SimpleNamespace(returncode=0, stdout={
            'lstart=': 'start', 'stat=': 'S', 'command=': command}[argv[-1]])
    with pytest.raises(handoff.ContinuationError, match='process_argument_mismatch'):
        handoff.process_identity(42, required_argument_pairs=(('--intent', '/exact/intent.json'),), runner=runner)
