# Covers (for impacted-test selection):
#   src/blueprint_pipeline/task_evaluation_scene_retirement_producer_births.py
#   src/blueprint_pipeline/task_evaluation_policy_canary_dispatcher.py
"""ADP-009D/day28: actual canary writes inherit exact current activation birth."""
import hashlib
import json
from pathlib import Path

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_evaluation_launch_preparation_contract import launch_preparation_request_digest
from tests.test_scene_retirement_connected_acceptance import _sealed_file
from tests.test_scene_retirement_producer_births import preparation
from tests.test_task_evaluation_policy_canary_dispatcher import _inputs


def owned_activation(tmp_path, monkeypatch):
    from blueprint_pipeline.task_evaluation_scene_retirement_producer_births import enroll_preparation_child
    from blueprint_pipeline import task_evaluation_scene_retirement_cache as cache
    _, policy, request, parent, prior = preparation(tmp_path, monkeypatch)
    monkeypatch.setattr(cache.time, 'time', lambda: 101)
    activation = parent.parent / 'activation-1'
    generation = enroll_preparation_child(activation, preparation_root=parent, request=request, now=101)
    result_path, setup_path, original_manifest = _inputs(tmp_path / 'native')
    for path in original_manifest.parent.iterdir():
        path.rename(activation / path.name)
    result = json.loads(result_path.read_bytes())
    runtime = activation / 'task_evaluation_policy_canary_runtime_inputs.v1.json'
    manifest = activation / original_manifest.name
    result.update(preparation_id=request['preparation_id'],
        request_digest=launch_preparation_request_digest(request),
        scene_intent_digest=request['scene_intent_digest'],
        policy_canary_runtime_inputs_path=str(runtime),
        policy_canary_runtime_inputs_sha256='sha256:' + hashlib.sha256(runtime.read_bytes()).hexdigest(),
        policy_canary_runtime_inputs_digest=json.loads(runtime.read_bytes())['runtime_inputs_digest'],
        policy_campaign_activation_sha256='sha256:' + hashlib.sha256(manifest.read_bytes()).hexdigest(),
        policy_campaign_activation_digest=json.loads(manifest.read_bytes())['activation_digest'],
        full_byte_activation_reference_readback_passed=True)
    result['result_digest'] = canonical_digest(result, digest_field='result_digest')
    result_path.write_text(json.dumps(result))
    output = parent.parent / 'canaries'
    output.mkdir()
    return policy, request, result, result_path, setup_path, activation, generation, output / 'activation-1'


def test_canary_birth_inherits_exact_current_activation_before_payload(tmp_path, monkeypatch):
    from blueprint_pipeline.task_evaluation_scene_retirement_producer_births import enroll_activation_child
    _, _, result, _, _, _, parent, output = owned_activation(tmp_path, monkeypatch)
    born = enroll_activation_child(output, activation_result=result, now=101)
    assert output.is_dir() and list(output.iterdir()) == []
    assert born['owner_raw_ref'] == parent['owner_raw_ref']
    assert born['birth_request_raw_ref'] == parent['birth_request_raw_ref']
    assert born['source_storage_authority_raw_ref'] == parent['source_storage_authority_raw_ref']
    assert born['ino'] == output.stat().st_ino


@pytest.mark.parametrize('field', ['activation_id', 'preparation_id', 'request_digest',
    'scene_intent_digest', 'source_commit', 'policy_campaign_activation_sha256',
    'policy_canary_runtime_inputs_sha256', 'policy_canary_runtime_inputs_digest'])
def test_resealed_foreign_native_join_never_births_canary(tmp_path, monkeypatch, field):
    from blueprint_pipeline.task_evaluation_scene_retirement_producer_births import enroll_activation_child
    _, _, result, _, _, _, _, output = owned_activation(tmp_path, monkeypatch)
    result[field] = 'sha256:' + 'f' * 64 if field.endswith(('digest', 'sha256')) else 'foreign'
    result['result_digest'] = canonical_digest(result, digest_field='result_digest')
    with pytest.raises(ValueError):
        enroll_activation_child(output, activation_result=result, now=101)
    assert not output.exists()


@pytest.mark.parametrize('change', ['retired', 'owner', 'inode', 'raw-authority'])
def test_current_parent_and_authority_are_reselected_before_canary_birth(tmp_path, monkeypatch, change):
    from blueprint_pipeline.task_evaluation_scene_retirement_producer_births import enroll_activation_child
    policy, _, result, _, _, activation, generation, output = owned_activation(tmp_path, monkeypatch)
    if change == 'raw-authority':
        path = Path(generation['source_storage_authority_raw_ref']['path'])
        path.unlink()
        path.write_text('{}')
    else:
        if change == 'retired':
            generation['state'] = 'retired'
        elif change == 'owner':
            generation['owner_intent_id'] = 'foreign'
        else:
            generation['ino'] += 1
        key = hashlib.sha256(str(activation).encode()).hexdigest() + '.json'
        _sealed_file(Path(policy['generation_store']) / key, generation, 'state_digest', mode=0o600)
    with pytest.raises(ValueError):
        enroll_activation_child(output, activation_result=result, now=101)
    assert not output.exists()


def test_legacy_activation_does_not_enroll_existing_canary(tmp_path, monkeypatch):
    from blueprint_pipeline.task_evaluation_scene_retirement_producer_births import enroll_activation_child
    policy, _, result, _, _, activation, _, output = owned_activation(tmp_path, monkeypatch)
    key = hashlib.sha256(str(activation).encode()).hexdigest() + '.json'
    (Path(policy['generation_store']) / key).unlink()
    output.mkdir()
    (output / 'legacy').write_bytes(b'keep')
    assert enroll_activation_child(output, activation_result=result, now=101) is None
    assert (output / 'legacy').read_bytes() == b'keep'
    assert not (Path(policy['generation_store']) / (hashlib.sha256(str(output).encode()).hexdigest() + '.json')).exists()


def test_actual_dispatcher_has_current_birth_before_first_progress_write(tmp_path, monkeypatch):
    from blueprint_pipeline import task_evaluation_policy_canary_dispatcher as dispatcher
    policy, _, _, result_path, setup_path, _, parent, output = owned_activation(tmp_path, monkeypatch)
    observed = []
    def first_write(root, **kwargs):
        key = hashlib.sha256(str(root).encode()).hexdigest() + '.json'
        current = json.loads((Path(policy['generation_store']) / key).read_bytes())
        assert current['owner_raw_ref'] == parent['owner_raw_ref']
        assert current['birth_request_raw_ref'] == parent['birth_request_raw_ref']
        assert current['ino'] == root.stat().st_ino
        observed.append(current)
        raise RuntimeError('fixture stops before provider work')
    monkeypatch.setattr(dispatcher, '_event_and_sync', first_write)
    with pytest.raises(RuntimeError, match='fixture stops'):
        dispatcher.dispatch_policy_canary_activation(activation_result_path=result_path,
            execution_setup_path=setup_path, output_root=output, implementation_commit='a' * 40,
            allocator_runner=lambda argv: pytest.fail('no provider execution'))
    assert len(observed) == 1


@pytest.mark.parametrize('mode', ['waiting', 'blocked'])
def test_actual_queue_births_before_wait_or_preprovider_block_receipt(tmp_path, monkeypatch, mode):
    from blueprint_pipeline import task_evaluation_policy_canary_dispatcher as dispatcher
    from tests.test_task_evaluation_policy_canary_dispatcher import _pending_canary, _record
    policy, _, _, result_path, setup_path, _, parent, output = owned_activation(tmp_path, monkeypatch)
    queue, setups = _pending_canary(tmp_path / 'queue-native')
    pending = queue / 'pending' / 'activation-1.json'
    envelope = json.loads(pending.read_bytes())
    envelope['activation_result'] = _record(result_path)
    envelope['envelope_digest'] = canonical_digest(envelope, digest_field='envelope_digest')
    pending.write_text(json.dumps(envelope))
    if mode == 'waiting':
        (setups / 'activation-1.json').unlink()
    else:
        setup = json.loads(setup_path.read_bytes())
        setup['source_commit'] = 'f' * 40
        setup['setup_digest'] = canonical_digest(setup, digest_field='setup_digest')
        (setups / 'activation-1.json').write_text(json.dumps(setup))
    run = dispatcher.process_policy_canary_dispatch_queue(dispatch_queue_root=queue,
        execution_setup_root=setups, dispatch_root=output.parent, implementation_commit='a' * 40,
        execute=False, blocked_sync_runner=lambda **kwargs: {'status': 'succeeded'})
    key = hashlib.sha256(str(output).encode()).hexdigest() + '.json'
    current = json.loads((Path(policy['generation_store']) / key).read_bytes())
    assert current['owner_raw_ref'] == parent['owner_raw_ref']
    assert current['ino'] == output.stat().st_ino
    filename = 'preprovider_waiting.json' if mode == 'waiting' else 'preprovider_blocked.json'
    assert (output / filename).is_file()
    assert run['results'][0]['provider_mutation_performed'] is False
