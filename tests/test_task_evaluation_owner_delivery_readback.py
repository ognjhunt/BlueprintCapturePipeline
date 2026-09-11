import copy
import json

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline import task_evaluation_scene_intake as intake
from blueprint_pipeline.task_evaluation_owner_delivery_readback import verify_owner_policy_delivery
from pathlib import Path
from tests.test_task_evaluation_scene_intake import attempt
from tests.test_task_evaluation_completed_scene_progression import _config
from tests import test_task_evaluation_completed_scene_source as completed_source_fixture
from blueprint_pipeline import task_evaluation_scene_progression as engine
from blueprint_pipeline import task_evaluation_scene_progression_state as progression_state


def _inputs(tmp_path, monkeypatch):
    original_submission = completed_source_fixture._submission
    def owned_submission(payload):
        value = original_submission(payload)
        value.update(customer_id='u1', organization_id='org1')
        return value
    monkeypatch.setattr(completed_source_fixture, '_submission', owned_submission)
    config, intent_id, queue, now = _config(tmp_path, monkeypatch, source_kind='mesh')
    result = engine.process_scene_intents(config_path=config, now=now)
    assert result['results'][0]['phase'] == 'publication_ready', result
    admitted = json.loads((queue / intent_id / 'intent.json').read_text())
    progression = progression_state.load_progression(queue / intent_id, admitted)
    factory = json.loads(Path(progression['state']['factory']['path']).read_text())
    prepared = json.loads(Path(factory['submission_request']['path']).read_text())
    assert prepared['team_namespace'].startswith('scene-')
    assert prepared['team_namespace'] != admitted['request']['owner']['organization_id']
    reserved = attempt(queue, admitted, now=now + 1)
    monkeypatch.delenv('BLUEPRINT_TASK_EVALUATION_CONTROLS_AUTOPROVISION_CONFIG', raising=False)
    binding = {key:reserved[key] for key in ('intent_digest','attempt_id','source_commit','runtime_digest','input_digest')}
    binding['intent_id'] = admitted['intent_id']
    setup = {'scene_attempt_binding':binding,'scene_intent_digest':admitted['intent_digest'],
             'capture_session_id':prepared['run_id'],'request_digest':'sha256:'+'1'*64}
    args = {'root':tmp_path/'dispatch','setup':setup,
        'runtime_inputs':{'configuration_digest':'sha256:'+'2'*64},
        'projection':{'run_id':'run-1','projection_digest':'sha256:'+'3'*64},
        'delivery':{'delivery_digest':'sha256:'+'4'*64,'artifacts':[
            {'artifact_id':'abcdef','digest':'sha256:'+'5'*64,'size_bytes':12}]},
        'publication':{'status':'succeeded'}}
    return args, queue, admitted


def _readback(**kwargs):
    authority = kwargs['owner_execution']
    result = {key:value for key,value in authority.items() if key != 'capture_session_id'}
    result.update(status='verified',result_delivery_digest=kwargs['result_delivery']['delivery_digest'],
        policy_canary_projection_digest=kwargs['policy_canary_result']['projection_digest'],
        inbox={'status':'verified','run_id':authority['run_id'],
               'projection_digest':kwargs['policy_canary_result']['projection_digest'],
               'owner_user_id':authority['owner_user_id'],'team_namespace':authority['team_namespace'],
               'source':'website_owner_run_index_readback'},
        artifacts=[{'artifact_id':r['artifact_id'],'sha256':r['digest'],'size_bytes':r['size_bytes'],
                    'verified':True,'http_status':200} for r in kwargs['result_delivery']['artifacts']])
    result['readback_digest'] = canonical_digest(result, digest_field='readback_digest')
    return result


def test_expired_or_revoked_execution_keeps_delivery_reachable_and_resume_reuses_proof(tmp_path, monkeypatch):
    args, queue, admitted = _inputs(tmp_path, monkeypatch)
    (queue/admitted['intent_id']/'revoked.json').write_text('{}')
    calls = []
    def reader(**kwargs):
        calls.append(kwargs)
        return _readback(**kwargs)
    result = verify_owner_policy_delivery(**args, readback_runner=reader)
    assert result['status'] == 'verified'
    assert calls[0]['owner_execution']['owner_user_id'] == 'u1'
    assert calls[0]['owner_execution']['team_namespace'].startswith('scene-')
    assert calls[0]['owner_execution']['team_namespace'] != 'org1'
    assert verify_owner_policy_delivery(**args, readback_runner=reader) == result
    assert len(calls) == 1
    assert not list((queue/admitted['intent_id']/'attempts').glob('a2*'))


@pytest.mark.parametrize('corruption', ['wrong_owner','missing_download','changed_byte_digest'])
def test_incomplete_owner_or_download_proof_never_seals(tmp_path, monkeypatch, corruption):
    args, _, _ = _inputs(tmp_path, monkeypatch)
    def reader(**kwargs):
        value = _readback(**kwargs)
        if corruption == 'wrong_owner':
            value['inbox']['owner_user_id'] = 'other'
        elif corruption == 'missing_download':
            value['artifacts'] = []
        else:
            value['artifacts'][0]['sha256'] = 'sha256:'+'f'*64
        value['readback_digest'] = canonical_digest(value, digest_field='readback_digest')
        return value
    assert verify_owner_policy_delivery(**args, readback_runner=reader)['status'] == 'pending'
    assert not (args['root']/'artifacts/result_delivery/owner_delivery_readback.json').exists()


def test_changed_owner_intent_prevents_even_readback_request(tmp_path, monkeypatch):
    args, queue, admitted = _inputs(tmp_path, monkeypatch)
    path = queue/admitted['intent_id']/'intent.json'
    original = json.loads(path.read_text())
    changed = copy.deepcopy(original)
    changed['request']['owner']['user_id'] = 'other'
    # Even a freshly hashed replacement cannot change this dispatch's sealed intent.
    changed['intent_digest'] = intake.canonical_digest(changed, digest_field='intent_digest')
    replacement = path.with_name('tampered.json')
    replacement.write_text(json.dumps(changed))
    replacement.replace(path)
    def forbidden(**_kwargs):
        pytest.fail('changed owner must fail before requesting private downloads')
    assert verify_owner_policy_delivery(**args, readback_runner=forbidden)['status'] == 'pending'


def _replace_json(path, value):
    replacement = path.with_name(path.name + '.replacement')
    replacement.write_text(json.dumps(value))
    replacement.replace(path)


def _change_retained_factory(queue, admitted, *, corruption):
    """Rehash derived fixtures so semantic bindings, not stale checksums, refuse them."""
    from blueprint_pipeline.task_evaluation_owner_delivery_readback import _record
    paths = sorted((queue / admitted['intent_id'] / 'progression-events').glob('*.json'))
    events = [json.loads(path.read_text()) for path in paths]
    state = events[-1]['state']
    factory_path = Path(state['factory']['path'])
    factory = json.loads(factory_path.read_text())
    attempt_path = Path(state['attempt']['path'])
    reserved = json.loads(attempt_path.read_text())
    binding_path = factory_path.parent / 'source_binding.json'
    binding = json.loads(binding_path.read_text())
    if corruption == 'team':
        task_path = Path(factory['task_request']['path'])
        task = json.loads(task_path.read_text())
        task['team_namespace'] = 'org1'
        _replace_json(task_path, task)
        factory['task_request'] = _record(task_path)
    else:
        binding['source_content_digest'] = 'sha256:' + 'f' * 64
        binding['binding_digest'] = canonical_digest(binding, digest_field='binding_digest')
        _replace_json(binding_path, binding)
        reserved['input_digest'] = binding['binding_digest']
        reserved['attempt_digest'] = intake.canonical_digest(reserved, digest_field='attempt_digest')
        _replace_json(attempt_path, reserved)
        factory['attempt_digest'] = reserved['attempt_digest']
    factory['factory_digest'] = canonical_digest(factory, digest_field='factory_digest')
    _replace_json(factory_path, factory)
    previous = None
    for path, event in zip(paths, events):
        if event.get('state', {}).get('factory'):
            event['state']['factory'] = _record(factory_path)
            event['state']['attempt'] = _record(attempt_path)
            event['state']['binding_digest'] = binding['binding_digest']
        event['previous_event_digest'] = previous
        event['event_digest'] = intake.canonical_digest(event, digest_field='event_digest')
        _replace_json(path, event)
        previous = event['event_digest']


@pytest.mark.parametrize('corruption', ['team', 'source'])
def test_rehashed_factory_cannot_change_owner_source_or_prepared_team(tmp_path, monkeypatch, corruption):
    args, queue, admitted = _inputs(tmp_path, monkeypatch)
    _change_retained_factory(queue, admitted, corruption=corruption)
    calls = []
    assert verify_owner_policy_delivery(**args, readback_runner=lambda **kw: calls.append(kw))['status'] == 'pending'
    assert calls == []


def test_another_scene_run_cannot_borrow_this_preparation_namespace(tmp_path, monkeypatch):
    args, _, _ = _inputs(tmp_path, monkeypatch)
    args['setup']['capture_session_id'] = 'another-scene-configuration'
    calls = []
    assert verify_owner_policy_delivery(**args, readback_runner=lambda **kw: calls.append(kw))['status'] == 'pending'
    assert calls == []


def test_v2_request_uses_factory_namespace_and_hashes_every_download_after_expiry(tmp_path, monkeypatch):
    from datetime import datetime, timezone
    from functools import partial
    from tests.test_task_evaluation_delivery_readback import fixture, Response
    from blueprint_pipeline.task_evaluation_delivery_readback import verify_website_delivery
    from blueprint_pipeline.task_evaluation_owner_delivery_readback import _owner_identity
    args, queue, admitted = _inputs(tmp_path, monkeypatch)
    expected = _owner_identity(args['setup'], args['runtime_inputs'], args['projection']['run_id'])
    assert expected['owner_user_id'] == 'u1'
    assert expected['team_namespace'].startswith('scene-') and expected['team_namespace'] != 'org1'
    (queue / admitted['intent_id'] / 'preparation-link.json').write_text('{"team_namespace":"untrusted-latest-pointer"}')
    (queue / admitted['intent_id'] / 'revoked.json').write_text('{}')
    expired = datetime.fromtimestamp(admitted['request']['execution']['expires_at_epoch'] + 1, timezone.utc)
    class ExpiredClock(datetime):
        @classmethod
        def now(cls, tz=None):
            return expired.astimezone(tz) if tz else expired.replace(tzinfo=None)
    monkeypatch.setattr(intake, 'datetime', ExpiredClock)
    http_args, state = fixture(tmp_path / 'http-fixture', count=25)
    args['delivery'] = http_args['result_delivery']
    args['projection']['projection_digest'] = http_args['policy_canary_result']['projection_digest']
    args['publication'] = {**http_args['publication'], 'request_digest': expected['request_digest'],
                           'configuration_digest': expected['configuration_digest']}
    def opener(call, *, timeout):
        response = http_args['opener'](call, timeout=timeout)
        if call.method != 'POST':
            return response
        body = json.loads(call.data)
        assert body['schema_version'] == 'task_evaluation_delivery_readback_request.v2'
        assert body['team_namespace'] == expected['team_namespace']
        assert body['owner_user_id'] == 'u1'
        assert body['capture_session_id'] == expected['capture_session_id']
        result = json.loads(response.read())
        result['inbox']['team_namespace'] = expected['team_namespace']
        return Response(json.dumps(result).encode())
    reader = partial(verify_website_delivery, endpoint_url=http_args['endpoint_url'],
                     token=http_args['token'], opener=opener, maximum_batches=1)
    assert verify_owner_policy_delivery(**args, readback_runner=reader)['status'] == 'pending'
    assert len(state['gets']) == 12
    assert verify_owner_policy_delivery(**args, readback_runner=reader)['status'] == 'pending'
    assert len(state['gets']) == 24
    result = verify_owner_policy_delivery(**args, readback_runner=reader)
    assert result['status'] == 'verified' and len(state['gets']) == 25
    receipt = json.loads(Path(result['receipt']['path']).read_text())
    assert len(receipt['artifacts']) == 25 and receipt['every_artifact_downloaded_and_hashed'] is True
    assert receipt['team_namespace'] == expected['team_namespace']
    assert 'PRIVATE-TICKET' not in json.dumps(receipt)
