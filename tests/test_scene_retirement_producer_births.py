# Covers (for impacted-test selection):
#   src/blueprint_pipeline/task_evaluation_scene_retirement_producer_births.py
#   src/blueprint_pipeline/task_evaluation_launch_activation_worker.py
#   src/blueprint_pipeline/task_evaluation_episode_compilation_worker.py
"""ADP-009D/day28: real child creation inherits exact authenticated preparation."""
import hashlib
import json
from pathlib import Path

import pytest

from tests.test_scene_retirement_normal_cache import owner_submission
from tests.test_scene_retirement_connected_acceptance import _sealed_file


def preparation(tmp_path, monkeypatch):
    from blueprint_pipeline import task_evaluation_scene_retirement_cache as cache
    from blueprint_pipeline.task_evaluation_launch_preparation_queue import stage_launch_preparation_request
    access, policy, request, _, queue, inputs, proofs = owner_submission(tmp_path, monkeypatch)
    cache.publish_preparation_storage_authority(queue_root=queue, request=request, now=101, **proofs)
    receipt = stage_launch_preparation_request(value=request, queue_root=queue, submitted_by='scene-progression')
    generation = cache.enroll_preparation_storage(queue_path=receipt['queue_path'], input_root=inputs, now=101)
    return access, policy, request, inputs / request['preparation_id'], generation


def test_authenticated_child_is_born_before_producer_payload(tmp_path, monkeypatch):
    from blueprint_pipeline.task_evaluation_scene_retirement_producer_births import enroll_preparation_child
    _, _, request, parent, prior = preparation(tmp_path, monkeypatch)
    target = parent.parent / 'actual-activation'
    result = enroll_preparation_child(target, preparation_root=parent, request=request, now=101)
    assert target.is_dir() and list(target.iterdir()) == []
    assert result['state'] == 'active'
    assert result['owner_raw_ref'] == prior['owner_raw_ref']
    assert result['birth_request_raw_ref'] == prior['birth_request_raw_ref']
    assert target.stat().st_ino == result['ino']


@pytest.mark.parametrize('change', ['owner', 'attempt', 'request', 'retired', 'inode'])
def test_child_cannot_borrow_changed_parent_or_authority(tmp_path, monkeypatch, change):
    from blueprint_pipeline.task_evaluation_scene_retirement_producer_births import enroll_preparation_child
    _, policy, request, parent, generation = preparation(tmp_path, monkeypatch)
    target = parent.parent / 'must-not-exist'
    key = hashlib.sha256(str(parent).encode()).hexdigest() + '.json'
    if change == 'owner':
        generation['owner_intent_id'] = 'foreign-owner'
    elif change == 'attempt':
        generation['birth_request_raw_ref'] = generation['owner_raw_ref']
    elif change == 'request':
        request = dict(request, run_id='foreign-run')
    elif change == 'retired':
        generation['state'] = 'retired'
    else:
        parent.rmdir()
        parent.mkdir()
        generation['ino'] = parent.stat().st_ino + 1
    _sealed_file(Path(policy['generation_store']) / key, generation, 'state_digest', mode=0o600)
    with pytest.raises(ValueError):
        enroll_preparation_child(target, preparation_root=parent, request=request, now=101)
    assert not target.exists()


def test_unregistered_preparation_never_adopts_child(tmp_path, monkeypatch):
    from blueprint_pipeline.task_evaluation_scene_retirement_producer_births import enroll_preparation_child
    _, policy, request, parent, _ = preparation(tmp_path, monkeypatch)
    (Path(policy['generation_store']) / (hashlib.sha256(str(parent).encode()).hexdigest() + '.json')).unlink()
    target = parent.parent / 'legacy-child'
    assert enroll_preparation_child(target, preparation_root=parent, request=request, now=101) is None
    assert not target.exists()


def test_verified_materialized_rows_must_belong_to_exact_preparation(tmp_path, monkeypatch):
    from blueprint_pipeline.task_evaluation_scene_retirement_producer_births import enroll_preparation_child
    _, _, request, parent, _ = preparation(tmp_path, monkeypatch)
    foreign = parent.parent / 'foreign-bytes'
    foreign.write_bytes(b'foreign')
    target = parent.parent / 'must-not-exist'
    with pytest.raises(ValueError):
        enroll_preparation_child(target, preparation_root=parent, request=request,
                                 verified_paths=[foreign], now=101)
    assert not target.exists()


def test_disabled_birth_preserves_original_producer_creation(tmp_path, monkeypatch):
    from blueprint_pipeline.task_evaluation_scene_retirement_producer_births import enroll_preparation_child
    monkeypatch.delenv('BLUEPRINT_SCENE_RETIREMENT_POLICY_FILE', raising=False)
    target = tmp_path / 'original-target'
    assert enroll_preparation_child(target, preparation_root=tmp_path / 'missing', request={}) is None
    assert not target.exists()


def test_actual_activation_native_join_births_from_normal_owned_preparation(tmp_path, monkeypatch):
    from blueprint_pipeline import task_evaluation_scene_retirement_cache as cache
    from blueprint_pipeline import task_evaluation_launch_preparation_worker as preparation_worker
    from blueprint_pipeline import task_evaluation_launch_activation_worker as activation_worker
    from blueprint_pipeline.task_evaluation_launch_preparation_contract import launch_preparation_request_digest
    from blueprint_pipeline.task_evaluation_launch_preparation_queue import stage_launch_preparation_request
    from tests.test_task_evaluation_launch_preparation_worker import fake_scene_render_inputs, SERVICE_ACCOUNT, fetcher
    _, policy, request, payloads, queue, inputs, proofs = owner_submission(tmp_path, monkeypatch, production=True)
    monkeypatch.setattr(cache.time, 'time', lambda: 101)
    cache.publish_preparation_storage_authority(queue_root=queue, request=request, now=101, **proofs)
    stage_launch_preparation_request(value=request, queue_root=queue, submitted_by='scene-progression')
    construction = tmp_path / 'construction'
    result = preparation_worker.process_launch_preparation_queue(queue_root=queue, input_root=inputs,
        allowed_uri_prefixes=['s3://blueprint-production-inputs/'], service_account=SERVICE_ACCOUNT,
        source_commit=request['expected_production_commit'], fetcher=fetcher(payloads),
        scene_render_input_materializer=fake_scene_render_inputs, construction_queue_root=construction)['results'][0]
    assert result['status'] == 'queued_for_production_scene_configuration', result
    activation = dict(preparation=dict(preparation_id=request['preparation_id'],
        request_digest=launch_preparation_request_digest(request), result_digest=result['result_digest']),
        team_namespace=request['team_namespace'], expected_production_commit=request['expected_production_commit'])
    target = inputs / 'activation-from-normal-parent'
    loaded, _, _, _ = activation_worker._load_verified_preparation(activation_request=activation,
        preparation_queue_root=queue, preparation_input_root=inputs,
        scene_construction_queue_root=construction, storage_birth_target=target)
    assert loaded == request and target.is_dir()
    key = hashlib.sha256(str(target).encode()).hexdigest() + '.json'
    generation = json.loads((Path(policy['generation_store']) / key).read_bytes())
    assert generation['owner_raw_ref'] == proofs['intent_raw_ref']
    assert generation['birth_request_raw_ref'] == proofs['attempt_raw_ref']


def test_actual_compilation_birth_precedes_compiler_payload(tmp_path, monkeypatch):
    from blueprint_pipeline import task_evaluation_scene_retirement_cache as cache
    from blueprint_pipeline import task_evaluation_episode_compilation_worker as worker
    from blueprint_pipeline.task_evaluation_launch_preparation_queue import stage_launch_preparation_request
    from tests.test_task_evaluation_episode_compilation_worker import _stage
    from tests.test_scene_retirement_connected_acceptance import _raw
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest
    _, policy, owner_request, _, queue, inputs, proofs = owner_submission(tmp_path, monkeypatch)
    compiler_fixture = tmp_path / 'compiler'
    compiler_fixture.mkdir()
    compiler_queue, old_inputs, envelope = _stage(compiler_fixture)
    request = envelope['request']
    request['expected_production_commit'] = owner_request['expected_production_commit']
    request['scene_intent_digest'] = owner_request['scene_intent_digest']
    request['task']['identity']['id'] = owner_request['task']['identity']['id']
    request_path = Path(proofs['submission_request_raw_ref']['path'])
    request_path.unlink()
    request_path.write_text(json.dumps(request))
    proofs['submission_request_raw_ref'] = _raw(request_path)
    factory_path = Path(proofs['factory_raw_ref']['path'])
    factory = json.loads(factory_path.read_bytes())
    factory_path.unlink()
    factory['submission_request'] = proofs['submission_request_raw_ref']
    _sealed_file(factory_path, factory, 'factory_digest')
    proofs['factory_raw_ref'] = _raw(factory_path)
    monkeypatch.setattr(cache.time, 'time', lambda: 101)
    cache.publish_preparation_storage_authority(queue_root=queue, request=request, now=101, **proofs)
    queued = stage_launch_preparation_request(value=request, queue_root=queue, submitted_by='scene-progression')
    cache.enroll_preparation_storage(queue_path=queued['queue_path'], input_root=inputs, now=101)
    parent = inputs / request['preparation_id']
    for row in envelope['materialized_references']:
        source = Path(row['materialized_path'])
        destination = parent / source.name
        source.rename(destination)
        row['materialized_path'] = str(destination)
    envelope['expected_production_commit'] = request['expected_production_commit']
    envelope['envelope_digest'] = canonical_digest(envelope, digest_field='envelope_digest')
    pending = compiler_queue / 'pending' / 'episode.json'
    pending.unlink()
    pending.write_text(json.dumps(envelope))
    outputs = inputs / 'compiled'
    observed = []
    def compile_episode(*, output_root, **kwargs):
        key = hashlib.sha256(str(output_root).encode()).hexdigest() + '.json'
        generation = json.loads((Path(policy['generation_store']) / key).read_bytes())
        assert generation['owner_raw_ref'] == proofs['intent_raw_ref']
        assert generation['birth_request_raw_ref'] == proofs['attempt_raw_ref']
        assert generation['ino'] == output_root.stat().st_ino
        observed.append(generation)
        raise RuntimeError('fixture stops before unrelated compiler output')
    run = worker.process_episode_compilation_queue(queue_root=compiler_queue, input_root=inputs,
        output_root=outputs, source_commit=request['expected_production_commit'], episode_compiler=compile_episode)
    assert len(observed) == 1, run
    assert run['results'][0]['status'] == 'blocked'
