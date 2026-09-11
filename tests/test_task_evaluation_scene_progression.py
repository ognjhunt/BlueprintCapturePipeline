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
