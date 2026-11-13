"""Persistent intent reaches real factory output without allocating a provider."""
import json
import time
from pathlib import Path
import pytest

from blueprint_pipeline import task_evaluation_scene_progression as engine
from blueprint_pipeline import task_evaluation_scene_progression_state as state
from blueprint_pipeline import public_scene_host_input_intake
from blueprint_pipeline.decision_evidence_contracts import cross_runtime_canonical_digest
from tests.test_task_evaluation_public_scene_attempt_factory import context as context, write


def configuration(context, monkeypatch):
    args, _ = context
    root = args["intent_path"].parent.parent
    release = json.loads(args["release_binding_path"].read_text())
    monkeypatch.setattr(public_scene_host_input_intake, "_verified_checkout_head", lambda: release["source_commit"])
    bindings = root.parent / "bindings"
    binding = json.loads(args["source_binding_path"].read_text())
    write(bindings / (binding["binding_id"] + ".json"), binding)
    config = dict(schema_version=engine.CONFIG_SCHEMA, intent_root=str(root),
        public_source_binding_root=str(bindings), machinery_path=str(args["machinery_path"]),
        release_binding_path=str(args["release_binding_path"]),
        factory_output_root=str(root.parent / "progression-output"), trusted_clients=["blueprint-webapp"],
        submission_enabled=False)
    return write(root.parent / "progression-config.json", config, "config_digest")


def test_real_factory_is_idempotent_across_worker_restart(context, monkeypatch):
    config = configuration(context, monkeypatch)
    first = engine.process_scene_intents(config_path=config)
    assert first["results"][0]["phase"] == "publication_ready", first
    assert first["provider_allocation_performed"] is False
    second = engine.process_scene_intents(config_path=config)
    assert second == first
    directory = context[0]["intent_path"].parent
    intent = json.loads((directory / "intent.json").read_text())
    progress = state.load_progression(directory, intent)
    factory = json.loads(Path(progress["state"]["factory"]["path"]).read_text())
    assert factory["original_source_reinstalled"] is False
    assert factory["provider_mutation_performed"] is False


def test_missing_source_and_expired_owner_do_not_reserve(context, monkeypatch):
    config = configuration(context, monkeypatch)
    value = json.loads(config.read_text())
    value["public_source_binding_root"] = str(config.parent / "missing")
    write(config, value, "config_digest")
    before = list(context[0]["intent_path"].parent.joinpath("attempts").glob("*.json"))
    result = engine.process_scene_intents(config_path=config)
    assert result["results"][0]["status"] == "awaiting_source"
    result = engine.process_scene_intents(config_path=config, now=time.time() + 10000)
    assert result["results"][0]["phase"] == "authority"
    assert before == list(context[0]["intent_path"].parent.joinpath("attempts").glob("*.json"))


def test_projection_recovers_event_before_pointer_crash(context, monkeypatch):
    config = configuration(context, monkeypatch)
    engine.process_scene_intents(config_path=config)
    directory = context[0]["intent_path"].parent
    intent = json.loads((directory / "intent.json").read_text())
    expected = state.load_progression(directory, intent)
    (directory / "progression.json").unlink()
    assert state.load_progression(directory, intent) == expected


@pytest.mark.parametrize("context", [{"max_total_spend_usd": 40}], indirect=True)
def test_transport_timeout_reconciles_exact_local_queue_without_second_post(context, monkeypatch):
    from blueprint_pipeline.task_evaluation_launch_preparation_queue import (
        ensure_launch_preparation_queue_root, stage_launch_preparation_request,
    )
    config = configuration(context, monkeypatch)
    value = json.loads(config.read_text())
    queue = config.parent / "queue"
    ensure_launch_preparation_queue_root(queue)
    value.update(submission_enabled=True, preparation_queue_root=str(queue),
                 publication_lock_root=str(config.parent / "publication-locks"))
    write(config, value, "config_digest")
    calls = []
    def publish(**kwargs):
        return dict(status="published_and_read_back", source_commit=kwargs["expected_source_commit"],
            raw_source_uploaded=False, provider_allocated=False,
            manifest_sha256=engine.record(kwargs["manifest_path"])["sha256"])
    def post(*, request_path, config):
        request = json.loads(request_path.read_text())
        calls.append(cross_runtime_canonical_digest(request))
        stage_launch_preparation_request(value=request, queue_root=queue, submitted_by="test-webapp")
        raise OSError("response lost after queue acceptance")
    first = engine.process_scene_intents(config_path=config, publisher=publish, submitter=post)
    assert first["results"][0]["status"] == "blocked", first
    second = engine.process_scene_intents(config_path=config, publisher=publish, submitter=post)
    assert second["results"][0]["status"] == "running", second
    assert len(calls) == 1
    link = json.loads((context[0]["intent_path"].parent / "preparation-link.json").read_text())
    # R3: the preparation link is immutable + mode-independent -- the paid
    # construction reservation lives on the separate activation link, minted only
    # at the activation transition (not reached in this mid-preparation timeout).
    assert "scene_configuration_attempt" not in link


def test_retained_only_preparation_does_not_reserve_source_gpu_spend(context,monkeypatch):
    args,_=context
    machinery=json.loads(args['machinery_path'].read_text())
    binding=json.loads(args['source_binding_path'].read_text())
    machinery['retained_prefix_only_binding_ids']=[binding['binding_id']]
    write(args['machinery_path'],machinery,'machinery_digest')
    config=configuration(context,monkeypatch)
    directory=args['intent_path'].parent
    before=list((directory/'attempts').glob('*.json'))
    result=engine.process_scene_intents(config_path=config)
    assert list((directory/'attempts').glob('*.json'))==before
    preparations=list((directory/'preparation-attempts').glob('*.json'))
    assert len(preparations)==1
    value=json.loads(preparations[0].read_text())
    assert value['maximum_spend_usd']==0 and value['paid_authority_granted'] is False
    assert result['results'][0]['status']=='blocked'
    assert 'retained_only_complete_prefix_required' in str(result['results'][0]['blockers'])


def test_capacity_wait_does_not_start_factory_and_resumes_when_whole_chain_fits(context, monkeypatch):
    from blueprint_pipeline import control_plane_capacity_controller as capacity
    config = configuration(context, monkeypatch)
    value = json.loads(config.read_text())
    value['require_whole_chain_capacity'] = True
    write(config, value, 'config_digest')
    monkeypatch.setattr(capacity, 'whole_chain_admission', lambda *a, **kw:
        {'status': 'waiting_for_capacity', 'required_workspace_bytes': 10 * 1024**3})
    waiting = engine.process_scene_intents(config_path=config)
    assert waiting['results'][0]['phase'] == 'capacity'
    assert waiting['results'][0]['blockers'] == ['scene_whole_chain_capacity_insufficient']
    assert not Path(value['factory_output_root']).exists()
    monkeypatch.setattr(capacity, 'whole_chain_admission', lambda *a, **kw:
        {'status': 'admitted', 'required_workspace_bytes': 10 * 1024**3})
    resumed = engine.process_scene_intents(config_path=config)
    assert resumed['results'][0]['phase'] == 'publication_ready'
    # Existing attempts keep their per-stage reservations and can finish even
    # if another workload later reduces the room available for NEW chains.
    monkeypatch.setattr(capacity, 'whole_chain_admission', lambda *a, **kw:
        pytest.fail('an existing attempt must not be stopped by the new-chain gate'))
    assert engine.process_scene_intents(config_path=config) == resumed


def test_capacity_wait_records_shortfall_eta_and_clears_on_admission(context, monkeypatch):
    from blueprint_pipeline import control_plane_capacity_controller as capacity
    config = configuration(context, monkeypatch)
    value = json.loads(config.read_text())
    value['require_whole_chain_capacity'] = True
    write(config, value, 'config_digest')
    start = time.time()
    summary = config.parent / 'capacity-summary.json'
    summary.write_text(json.dumps({'observed_at_epoch': start,
        'reclaim_outlook': {'volume_growth': 'blocked', 'reclaimable_bytes': 8 * 1024**3,
                            'next_reclaim_epoch': start + 3600}}))
    monkeypatch.setenv('BLUEPRINT_CAPACITY_SUMMARY_PATH', str(summary))
    waiting_admission = {'status': 'waiting_for_capacity', 'required_workspace_bytes': 10 * 1024**3,
        'required_workspace_basis': 'measured_p95',
        'measurement': {'available_bytes': 4 * 1024**3},
        'devices': [{'device': 7, 'required_bytes': 10 * 1024**3,
                     'available_bytes': 4 * 1024**3, 'passed': False}]}
    monkeypatch.setattr(capacity, 'whole_chain_admission', lambda *a, **kw: waiting_admission)
    first = engine.process_scene_intents(config_path=config, now=start)['results'][0]
    directory = context[0]['intent_path'].parent
    intent = json.loads((directory / 'intent.json').read_text())
    wait = state.load_progression(directory, intent)['state']['capacity_wait']
    assert first['phase'] == 'capacity' and first['blockers'] == ['scene_whole_chain_capacity_insufficient']
    assert wait['since_epoch'] == start
    assert wait['required_bytes'] == 10 * 1024**3 and wait['available_bytes'] == 4 * 1024**3
    assert wait['shortfall_bytes'] == 6 * 1024**3 and wait['basis'] == 'measured_p95'
    assert wait['eta_epoch'] == start + 3600 and wait['eta_basis'] == 'reclaim_scheduled'
    assert wait['next_check_epoch'] == start + 600
    again = engine.process_scene_intents(config_path=config, now=start + 60)['results'][0]
    assert again['phase'] == 'capacity'
    assert state.load_progression(directory, intent)['state']['capacity_wait']['since_epoch'] == start
    monkeypatch.setattr(capacity, 'whole_chain_admission', lambda *a, **kw:
        {'status': 'admitted', 'required_workspace_bytes': 10 * 1024**3})
    admitted = engine.process_scene_intents(config_path=config, now=start + 120)['results'][0]
    assert admitted['phase'] == 'publication_ready'
    assert 'capacity_wait' not in state.load_progression(directory, intent)['state']


@pytest.mark.parametrize('status,outcome', [
    ('publication_ready', 'completed'),
    # A factory that stopped short measured a partial workspace.
    ('blocked', 'blocked'), ('needs_input', 'blocked'), ('awaiting_source', 'blocked'),
])
def test_preparation_storage_measures_the_attempt_directory(tmp_path, monkeypatch, status, outcome):
    from types import SimpleNamespace
    from blueprint_pipeline import control_plane_disk_budget as disk
    from blueprint_pipeline.task_evaluation_scene_preparation_attempts import (
        preparation_storage, settle_preparation_storage,
    )
    calls = []
    real = disk.reserve_control_plane_disk

    def recording(*args, **kwargs):
        calls.append(kwargs)
        return real(*args, **kwargs, disk_usage=lambda _path: SimpleNamespace(
            total=100 * disk.GIB, used=10 * disk.GIB, free=90 * disk.GIB))

    monkeypatch.setattr(disk, 'reserve_control_plane_disk', recording)
    output = tmp_path / 'factory' / 'intent-1' / 'attempt-1'
    output.mkdir(parents=True)
    config = {'preparation_worker': {'disk_reservation_root': str(tmp_path / 'ledger')}}
    binding = {'references': {'primary': {'size_bytes': 1000}, 'collision': {'size_bytes': 10}}}
    with preparation_storage(config, binding, output) as reservation:
        (output / 'materialized').mkdir()
        (output / 'materialized' / 'scene.bin').write_bytes(b's' * 20_000)
        settle_preparation_storage(reservation, {'status': status})
    assert calls[0]['target_root'] == output and calls[0]['workspace'] == output
    assert calls[0]['workload'] == 'scene_preparation_attempt'
    history = tmp_path / 'ledger' / 'history' / 'launch_preparation.jsonl'
    [sample] = [json.loads(line) for line in history.read_text().splitlines()]
    assert sample['workload'] == 'scene_preparation_attempt' and sample['outcome'] == outcome
    assert sample['observed_bytes'] >= 20_000


def test_preparation_storage_rerun_over_existing_factory_output_records_resumed(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from blueprint_pipeline import control_plane_disk_budget as disk
    from blueprint_pipeline.task_evaluation_scene_preparation_attempts import (
        preparation_storage, settle_preparation_storage,
    )
    real = disk.reserve_control_plane_disk
    monkeypatch.setattr(disk, 'reserve_control_plane_disk', lambda *args, **kwargs: real(
        *args, **kwargs, disk_usage=lambda _path: SimpleNamespace(
            total=100 * disk.GIB, used=10 * disk.GIB, free=90 * disk.GIB)))
    output = tmp_path / 'factory' / 'intent-1' / 'attempt-1'
    (output / 'materialized').mkdir(parents=True)  # an earlier pass of this attempt's factory
    (output / 'materialized' / 'scene.bin').write_bytes(b's' * 4096)
    config = {'preparation_worker': {'disk_reservation_root': str(tmp_path / 'ledger')}}
    with preparation_storage(config, {'references': {}}, output) as reservation:
        settle_preparation_storage(reservation, {'status': 'publication_ready'})
    history = tmp_path / 'ledger' / 'history' / 'launch_preparation.jsonl'
    [sample] = [json.loads(line) for line in history.read_text().splitlines()]
    assert (sample['outcome'], sample['fresh']) == ('resumed', False)


def test_invalid_publication_ready_factory_records_failed_footprint(context, monkeypatch):
    from types import SimpleNamespace
    from blueprint_pipeline import control_plane_disk_budget as disk

    config_path = configuration(context, monkeypatch)
    config = json.loads(config_path.read_text())
    ledger = config_path.parent / 'ledger'
    config['preparation_worker'] = {'disk_reservation_root': str(ledger)}
    write(config_path, config, 'config_digest')
    real_reserve = disk.reserve_control_plane_disk
    monkeypatch.setattr(disk, 'reserve_control_plane_disk', lambda *args, **kwargs: real_reserve(
        *args, **kwargs, disk_usage=lambda _path: SimpleNamespace(
            total=100 * disk.GIB, used=10 * disk.GIB, free=90 * disk.GIB)))
    real_source = engine._source

    def invalid_source(*args, **kwargs):
        resolved = real_source(*args, **kwargs)
        return engine.SourceResolution(
            resolved.status, resolved.binding_path, resolved.machinery_path,
            lambda **_kwargs: {'status': 'publication_ready'},
        )

    monkeypatch.setattr(engine, '_source', invalid_source)
    result = engine.process_scene_intents(config_path=config_path)
    assert result['results'][0]['blockers'] == ['scene_progression_factory_receipt_invalid']
    history = ledger / 'history' / 'launch_preparation.jsonl'
    [sample] = [json.loads(line) for line in history.read_text().splitlines()]
    assert sample['outcome'] == 'failed'


def test_hermetic_preparation_storage_has_nothing_to_settle():
    from blueprint_pipeline.task_evaluation_scene_preparation_attempts import (
        preparation_storage, settle_preparation_storage,
    )
    with preparation_storage({}, {}, None) as reservation:
        assert reservation is None
        settle_preparation_storage(reservation, {'status': 'blocked'})


@pytest.mark.parametrize("installed", [False, True])
def test_registered_terminal_adoption_does_not_restart_completed_scene_factory(context, monkeypatch, installed):
    from blueprint_pipeline import task_evaluation_controls_autoprovision as controls
    config_path = configuration(context, monkeypatch)
    engine.process_scene_intents(config_path=config_path)
    directory = context[0]['intent_path'].parent
    intent = json.loads((directory/'intent.json').read_text())
    previous = state.load_progression(directory, intent)
    state.advance(directory, intent, previous, status='running', phase='scene_configuration',
        state={**previous['state'], 'activation': {'path': 'retained-activation'}}, blockers=[], now=time.time())
    controls_config = write(config_path.parent/'controls-config.json', {'scene_root': str(directory.parent)})
    monkeypatch.setenv(controls.CONFIG_ENV, str(controls_config))
    monkeypatch.setattr(controls, '_registered_terminal_adoption', lambda **kw: {'status': 'installed_terminal_adoption', 'source_launch_id': 'completed-scene'} if installed else None)
    from blueprint_pipeline import task_evaluation_controls_terminal_adoption as adoption
    monkeypatch.setattr(adoption, 'terminal_adoption_source', lambda **kw: {'adoption': {'source_launch_id': 'completed-scene'}})
    monkeypatch.setattr(engine, '_source', lambda *a, **kw: pytest.fail('completed scene must not be reconstructed'))
    result = engine.process_scene_intents(config_path=config_path)
    assert result['results'][0]['phase'] == ('configured_controls' if installed else 'configured_controls_adoption')
    assert result['provider_allocation_performed'] is False


@pytest.mark.parametrize("registration_refused", [False, True])
def test_accepted_intent_registers_supervision_without_blocking_factory(context, monkeypatch, registration_refused):
    from blueprint_pipeline.agent_execution import supervision_producer
    config = configuration(context, monkeypatch)
    seen = []
    def register(**kwargs):
        assert not (config.parent / "progression-output").exists()
        seen.append(kwargs)
        if registration_refused:
            raise ValueError("fixture_optional_reasoning_refused")
    monkeypatch.setattr(supervision_producer, "register_run_supervision", register)
    result = engine.process_scene_intents(config_path=config)
    assert result["results"][0]["phase"] == "publication_ready"
    assert len(seen) == 1
    assert seen[0]["intent"]["intent_id"] == context[0]["intent_path"].parent.name
    assert seen[0]["directory"] == context[0]["intent_path"].parent
    assert seen[0]["source_commit"] == result["source_commit"]


def test_configured_owner_scope_does_not_touch_other_intents_or_shared_cursor(context, monkeypatch):
    config = configuration(context, monkeypatch)
    value = json.loads(config.read_text())
    selected = context[0]['intent_path'].parent.name
    other = Path(value['intent_root']) / 'scene-unrelated-owner'
    other.mkdir()
    marker = other / 'preserve.txt'
    marker.write_text('unrelated owner state')
    value['only_intent_id'] = selected
    write(config, value, 'config_digest')
    result = engine.process_scene_intents(config_path=config)
    assert [row['intent_id'] for row in result['results']] == [selected]
    assert list(other.iterdir()) == [marker]
    assert marker.read_text() == 'unrelated owner state'
    assert not (Path(value['intent_root']) / 'progression-cursor.json').exists()


def test_validation_progress_heartbeat_lands_in_scene_directory(context, monkeypatch):
    """A restart that sits in ``preparing/factory`` now says which step, for how long, and how many bytes."""
    from blueprint_pipeline import validation_progress as progress
    config = configuration(context, monkeypatch)
    engine.process_scene_intents(config_path=config)
    directory = context[0]["intent_path"].parent
    record = json.loads((directory / progress.FILENAME).read_text())
    assert record["schema_version"] == progress.SCHEMA
    assert record["intent_id"] == directory.name
    assert record["step"] in {"factory_start", "prefix_candidate", "prefix_phase", "adoption_publish",
                              "adoption_revalidate", "submission_inputs"}
    assert record["elapsed_seconds"] >= 0 and record["heartbeat_sequence"] >= 1
    assert set(record["digests"]) == {"files_hashed", "bytes_hashed", "hash_seconds", "cache_hits", "verdicts_computed", "verdicts_reused",
                                      "bytes_reused", "last_path"}
    assert not (directory / (progress.FILENAME + ".tmp")).exists()
    assert progress.heartbeat("outside") is None


def test_team_evaluation_does_not_rebuild_prepared_scene(context, monkeypatch):
    config = configuration(context, monkeypatch)
    args, _ = context
    path = args['intent_path']
    intent = json.loads(path.read_text())
    intent['request']['task']['evaluation_source'] = {
        'evaluation_run_id':'team-eval', 'source_launch_id':'delivered-scene',
        'source_profile_digest':'sha256:'+'a'*64,
        'configured_scene_revision_digest':'sha256:'+'b'*64}
    from blueprint_pipeline import task_evaluation_scene_intake as intake
    intent['request']['submission_id'] = 'team-eval'
    owner = intake.stage_scene_intent(value=intent['request'], queue_root=path.parent.parent,
        authenticated_client=intent['authenticated_issuer'], trusted_clients={intent['authenticated_issuer']},
        now=intent['accepted_at_epoch'])
    path = path.parent.parent / owner['intent_id'] / 'intent.json'
    before = list(path.parent.joinpath('attempts').glob('*.json'))
    result = engine.process_scene_intents(config_path=config, only_intent_id=owner['intent_id'])
    assert result['results'][0]['phase'] == 'selected_team_evaluation', result
    assert result['results'][0]['status'] == 'awaiting_execution'
    assert before == list(path.parent.joinpath('attempts').glob('*.json'))
    assert not Path(json.loads(config.read_text())['factory_output_root']).exists()


def test_catch_all_refusal_records_its_outcome_and_a_sealed_one_stops_new_execution(context, monkeypatch):
    """C-04: a refusal raised mid-pass gets the same outcome an emitted one would."""
    from blueprint_pipeline import task_evaluation_scene_intake as intake
    from blueprint_pipeline import task_evaluation_scene_outcome as outcome
    config = configuration(context, monkeypatch)
    directory = context[0]["intent_path"].parent
    intent = json.loads((directory / "intent.json").read_text())

    def refuse(code):
        def source(*args, **kwargs):
            raise ValueError(code)
        return source

    monkeypatch.setattr(engine, "_source", refuse("scene_intake_spend_cap_exhausted"))
    waiting = engine.process_scene_intents(config_path=config)["results"][0]
    assert (waiting["status"], waiting["blockers"]) == ("blocked", ["scene_intake_spend_cap_exhausted"])
    assert outcome.read_outcome(directory, intent)["kind"] == "awaiting_spend_approval"
    assert outcome.terminal_outcome(directory, intent) is None
    monkeypatch.setattr(engine, "_source", refuse("scene_progression_retry_cap_exhausted"))
    ended = engine.process_scene_intents(config_path=config)["results"][0]
    assert ended["blockers"] == ["scene_progression_retry_cap_exhausted"]
    sealed = outcome.terminal_outcome(directory, intent)
    assert (sealed["kind"], sealed["reason_code"]) == ("setup_repair", "retry_cap_exhausted")
    monkeypatch.setattr(engine, "_source", lambda *a, **kw: pytest.fail("a terminal outcome stops new execution"))
    assert engine.process_scene_intents(config_path=config)["results"][0] == ended
    status = intake.scene_intent_status(queue_root=directory.parent, intent_id=directory.name)
    assert status["status"] == "blocked" and status["outcome"] == sealed


def test_refusal_reasons_reach_the_run_result_but_not_the_recorded_blocker(context, monkeypatch):
    """The journal shows why a refusal held; progression keeps only its stable code."""
    config = configuration(context, monkeypatch)
    directory = context[0]["intent_path"].parent

    def refuse(*args, **kwargs):
        raise ValueError("scene_progression_recovery_provider_zero_required:"
                         "gpu_inventory_missing:vast;gpu_spend_guard_stale;/var/lib/blueprint/x.json;Free Text")

    monkeypatch.setattr(engine, "_source", refuse)
    row = engine.process_scene_intents(config_path=config)["results"][0]
    assert row["blockers"] == ["scene_progression_recovery_provider_zero_required"]
    assert row["blocker_detail"] == ["gpu_inventory_missing:vast", "gpu_spend_guard_stale"]
    recorded = json.loads((directory / "progression.json").read_text())
    assert recorded["blockers"] == ["scene_progression_recovery_provider_zero_required"]
    assert "blocker_detail" not in json.dumps(recorded)
    monkeypatch.setattr(engine, "_source", lambda *a, **kw: (_ for _ in ()).throw(
        ValueError("scene_progression_recovery_provider_zero_required")))
    assert "blocker_detail" not in engine.process_scene_intents(config_path=config)["results"][0]


def test_a_dependency_failure_records_its_type_and_code_location_but_not_its_message(context, monkeypatch):
    """2026-11-13: a FileNotFoundError parked a website job as
    ``scene_progression_dependency_unavailable`` and nothing said which call raised it."""
    config = configuration(context, monkeypatch)
    directory = context[0]["intent_path"].parent
    from blueprint_pipeline import task_evaluation_scene_attempt_evidence as evidence

    def source(*args, **kwargs):
        evidence.record(Path("/var/lib/blueprint/x/post_teardown_provider_zero_receipt.json"))

    monkeypatch.setattr(engine, "_source", source)
    row = engine.process_scene_intents(config_path=config)["results"][0]
    assert row["blockers"] == ["scene_progression_dependency_unavailable"]
    where = row["blocker_detail"][0]
    assert where.startswith("FileNotFoundError@blueprint_pipeline.")
    module, function, line = where.split("@", 1)[1].split(":")
    assert module == "blueprint_pipeline.validation_file_digests" and function == "sha256_file" and line.isdigit()
    recorded = json.loads((directory / "progression.json").read_text())
    assert recorded["state"][engine.DEPENDENCY_FAILURE_FIELD] == where
    assert recorded["blockers"] == ["scene_progression_dependency_unavailable"]
    assert "/var/lib" not in json.dumps(recorded) and "Errno" not in json.dumps(row)

    # The same failure again rewrites nothing; once the pass gets past it, the note is gone.
    events = sorted((directory / "progression-events").glob("*.json"))
    assert engine.process_scene_intents(config_path=config)["results"][0] == row
    assert sorted((directory / "progression-events").glob("*.json")) == events
    monkeypatch.undo()
    config = configuration(context, monkeypatch)
    later = engine.process_scene_intents(config_path=config)["results"][0]
    assert later["blockers"] != ["scene_progression_dependency_unavailable"]
    recorded = json.loads((directory / "progression.json").read_text())
    assert engine.DEPENDENCY_FAILURE_FIELD not in recorded["state"]


def test_a_recorded_dependency_failure_does_not_survive_a_later_refusal(context, monkeypatch):
    """Review probe: the handler rebuilds state from disk, so it drops the old location itself."""
    config = configuration(context, monkeypatch)
    directory = context[0]["intent_path"].parent

    def missing(*a, **k):
        Path("/nonexistent/rv/file.json").read_bytes()
    monkeypatch.setattr(engine, "_source", missing)
    engine.process_scene_intents(config_path=config)
    assert engine.DEPENDENCY_FAILURE_FIELD in json.loads((directory / "progression.json").read_text())["state"]

    def refused(*a, **k):
        raise ValueError("scene_progression_rv_refused:reason_one")
    monkeypatch.setattr(engine, "_source", refused)
    row = engine.process_scene_intents(config_path=config)["results"][0]
    assert row["blockers"] == ["scene_progression_rv_refused"] and row["blocker_detail"] == ["reason_one"]
    recorded = json.loads((directory / "progression.json").read_text())
    assert engine.DEPENDENCY_FAILURE_FIELD not in recorded["state"]


def test_any_exception_from_one_job_blocks_that_job_and_the_pass_goes_on(context, monkeypatch):
    """2026-11-13: a RuntimeError subclass from one job's activation ended the whole unit run."""
    from blueprint_pipeline import task_evaluation_scene_outcome as outcome
    config = configuration(context, monkeypatch)
    directory = context[0]["intent_path"].parent
    later = directory.parent / "scene-zzzz-later-job"
    later.mkdir()

    class Unforeseen(Exception):
        pass

    def unforeseen(*a, **k):
        raise Unforeseen("/var/lib/blueprint/private/path and site text")
    monkeypatch.setattr(engine, "_source", unforeseen)
    run = engine.process_scene_intents(config_path=config)
    first, second = run["results"]
    assert first["intent_id"] == directory.name and first["status"] == "blocked"
    assert first["blockers"] == [engine.UNEXPECTED_FAILURE_BLOCKER]
    [where] = first["blocker_detail"]
    assert where.startswith("Unforeseen@blueprint_pipeline.task_evaluation_scene_progression:_advance_intent:")
    assert "/var/lib" not in json.dumps(run) and "site text" not in json.dumps(run)
    # The job after it in the same pass is still visited.
    assert (second["intent_id"], second["status"], second["blockers"]) == (
        later.name, "blocked", ["scene_progression_state_invalid"])
    intent = json.loads((directory / "intent.json").read_text())
    decided = outcome.read_outcome(directory, intent)
    assert (decided["kind"], decided["terminal"], decided["reason_code"]) == ("setup_repair", False, "pipeline_fault")


def test_a_coded_pipeline_error_that_is_not_a_value_error_keeps_its_own_code(context, monkeypatch):
    from blueprint_pipeline.task_evaluation_scene_configuration_activation_identity import (
        SceneConfigurationActivationAutomationError,
    )
    config = configuration(context, monkeypatch)

    def refused(*a, **k):
        raise SceneConfigurationActivationAutomationError(
            "scene_configuration_activation_immutable_conflict:intent_registry_same_release_conflict")
    monkeypatch.setattr(engine, "_source", refused)
    row = engine.process_scene_intents(config_path=config)["results"][0]
    assert row["blockers"] == [engine.ACTIVATION_CONFLICT_BLOCKER]
    assert row["blocker_detail"][0] == "intent_registry_same_release_conflict"
    assert row["blocker_detail"][1].startswith("SceneConfigurationActivationAutomationError@blueprint_pipeline.")


@pytest.mark.parametrize("stop", [KeyboardInterrupt, SystemExit, MemoryError, RecursionError])
def test_an_interrupt_or_exit_still_stops_the_run(context, monkeypatch, stop):
    config = configuration(context, monkeypatch)

    def interrupted(*a, **k):
        raise stop()
    monkeypatch.setattr(engine, "_source", interrupted)
    with pytest.raises(stop):
        engine.process_scene_intents(config_path=config)


def test_a_pass_that_ends_normally_drops_the_earlier_notes(context, monkeypatch):
    """The start of every pass drops the last failure's location and refusal note."""
    from types import SimpleNamespace
    config = configuration(context, monkeypatch)
    directory = context[0]["intent_path"].parent

    def missing(*a, **k):
        Path("/nonexistent/rv/file.json").read_bytes()
    monkeypatch.setattr(engine, "_source", missing)
    engine.process_scene_intents(config_path=config)
    recorded = json.loads((directory / "progression.json").read_text())["state"]
    assert engine.DEPENDENCY_FAILURE_FIELD in recorded
    monkeypatch.setattr(engine, "_source", lambda *a, **k: SimpleNamespace(
        analysis_reference=None, status="awaiting_source", blockers=["source_not_yet_uploaded"]))
    row = engine.process_scene_intents(config_path=config)["results"][0]
    assert (row["status"], row["blockers"]) == ("awaiting_source", ["source_not_yet_uploaded"])
    recorded = json.loads((directory / "progression.json").read_text())["state"]
    assert engine.DEPENDENCY_FAILURE_FIELD not in recorded and engine.ACTIVATION_REFUSAL_FIELD not in recorded


@pytest.mark.parametrize("message, expected", [
    ("scene_progression_" + "a" * 200, "scene_progression_unexpected_failure"),
    ("scene_progression_caf\u00e9_refused", "scene_progression_unexpected_failure"),
    ("scene_progression_" + "a" * (120 - 18), "scene_progression_" + "a" * (120 - 18)),
])
def test_a_code_from_a_message_is_ascii_and_bounded(context, monkeypatch, message, expected):
    config = configuration(context, monkeypatch)

    def refused(*a, **k):
        raise ValueError(message + ":reason")
    monkeypatch.setattr(engine, "_source", refused)
    assert engine.process_scene_intents(config_path=config)["results"][0]["blockers"] == [expected]


def test_an_overlong_code_from_a_pipeline_error_falls_back_to_unexpected(context, monkeypatch):
    from blueprint_pipeline.task_evaluation_scene_configuration_activation_identity import (
        SceneConfigurationActivationAutomationError,
    )
    config = configuration(context, monkeypatch)

    def refused(*a, **k):
        raise SceneConfigurationActivationAutomationError("scene_configuration_" + "x" * 150)
    monkeypatch.setattr(engine, "_source", refused)
    row = engine.process_scene_intents(config_path=config)["results"][0]
    assert row["blockers"] == [engine.UNEXPECTED_FAILURE_BLOCKER]
    assert engine.MAX_BLOCKER_CODE_LENGTH == 120


def _parking(**changes):
    stored = {"attempt_id": "source-1", "attempt_commit": "c" * 40,
              engine.ACTIVATION_REFUSAL_FIELD: engine.SAME_RELEASE_CONFLICT_DETAIL}
    progress = {"status": "blocked", "blockers": [engine.ACTIVATION_CONFLICT_BLOCKER], "state": stored}
    state, attempt_id, commit = {}, "source-1", "c" * 40
    for key, value in changes.items():
        if key == "progress":
            progress.update(value)
        elif key == "stored":
            stored.update(value)
        elif key == "state":
            state = value
        elif key == "attempt_id":
            attempt_id = value
        elif key == "commit":
            commit = value
    return engine._parked_activation(progress, state, attempt_id=attempt_id, commit=commit)


@pytest.mark.parametrize("changes", [
    {"progress": {"blockers": ["scene_progression_dependency_unavailable"]}},
    {"progress": {"blockers": [engine.ACTIVATION_CONFLICT_BLOCKER, "another"]}},
    {"progress": {"status": "awaiting_execution"}},
    {"stored": {engine.ACTIVATION_REFUSAL_FIELD: None}},
    {"attempt_id": "source-2"},
    {"commit": "d" * 40},
    {"state": {"activation": {"path": "/x"}}},
])
def test_only_this_attempts_same_release_conflict_on_this_release_is_parked(changes):
    assert _parking() is True
    assert _parking(**changes) is False
