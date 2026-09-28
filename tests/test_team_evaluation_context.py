"""Website receives exact owner context without granting execution or exposing credentials."""
import json
import pytest

from blueprint_pipeline import task_evaluation_team_run_context as context
from blueprint_pipeline import task_evaluation_scene_intake as intake
from blueprint_pipeline import task_evaluation_controls_autoprovision as worker
from tests.test_team_evaluation_controller import _case, _put


def fixture(tmp_path, monkeypatch):
    args, config, path, _ = _case(tmp_path, monkeypatch)
    config['robot_catalog_path']=str(_put(tmp_path/'catalog.json', args['catalog']))
    original=intake._read(path, 'intent_digest')
    return args, config, path, original


def test_context_is_exact_read_only_and_excludes_runtime_paths_and_secrets(tmp_path, monkeypatch):
    args, config, path, original=fixture(tmp_path, monkeypatch)
    before=path.read_bytes()
    result=context.evaluation_context(source_launch_id='source-launch', owner=original['request']['owner'], config=config)
    assert result['context_digest']==intake.canonical_digest(result, digest_field='context_digest')
    assert result['task']==original['request']['task']
    assert result['source']==original['request']['source']
    assert result['configurations'][0]['id']=='franka-droid'
    assert len(result['configurations'][0]['policy_candidates'])==2
    assert 'api_key' not in json.dumps(result) and 'proj_test' not in json.dumps(result)
    assert str(tmp_path) not in json.dumps(result['configurations'])
    assert result['provider_mutation_performed'] is False
    assert path.read_bytes()==before and not list(args['scene_root'].glob('scene-*/attempts/*.json'))


@pytest.mark.parametrize('mode', ['other_owner','revoked','path_escape','profile_changed'])
def test_private_source_cannot_be_read_without_exact_access(tmp_path, monkeypatch, mode):
    args, config, path, original=fixture(tmp_path, monkeypatch)
    owner=dict(original['request']['owner'])
    launch='source-launch'
    if mode=='other_owner':
        owner['user_id']='other'
    if mode=='revoked':
        _put(path.parent/'revoked.json', {})
    if mode=='path_escape':
        launch='../source-launch'
    if mode=='profile_changed':
        profile_path=tmp_path/'launches'/'source-launch'/'launch_profile.json'
        profile=json.loads(profile_path.read_text())
        profile['changed']=True
        _put(profile_path,profile)
    with pytest.raises(ValueError):
        context.evaluation_context(source_launch_id=launch, owner=owner, config=config)
    assert not list(args['scene_root'].glob('scene-*/attempts/*.json'))


def test_http_context_requires_signature_and_does_not_create_intent(tmp_path, monkeypatch):
    import hmac
    from datetime import datetime, timezone
    from fastapi.testclient import TestClient
    from blueprint_pipeline import live_pipeline_intake_service as service
    args, config, path, original=fixture(tmp_path, monkeypatch)
    monkeypatch.setenv(worker.CONFIG_ENV,str(_put(tmp_path/'config.json',config)))
    monkeypatch.setenv(service.INTAKE_TOKEN_ENV,'test-token')
    monkeypatch.delenv(service.INTAKE_CLIENT_SECRETS_ENV,raising=False)
    monkeypatch.setenv(service.INTAKE_NONCE_STORE_DIR_ENV,str(tmp_path/'nonces'))
    monkeypatch.setenv(service.INTAKE_WORK_DIR_ENV,str(tmp_path/'admission'))
    monkeypatch.setattr(service,'deployment_identity_payload',lambda:{})
    service._INTAKE_NONCE_CACHE.clear()
    client=TestClient(service.create_app())
    url='/api/live-pipeline/task-evaluation-team-context'
    payload=json.dumps({'source_launch_id':'source-launch','owner':original['request']['owner']})
    assert client.post(url,content=payload).status_code==401
    timestamp=datetime.now(timezone.utc).isoformat()
    nonce='context-test'
    signature=hmac.new(b'test-token',f'{timestamp}.webapp.{nonce}.{payload}'.encode(),'sha256').hexdigest()
    response=client.post(url,content=payload,headers={'content-type':'application/json',
        'x-blueprint-pipeline-client-id':'webapp','x-blueprint-pipeline-timestamp':timestamp,
        'x-blueprint-pipeline-nonce':nonce,'x-blueprint-pipeline-signature':'sha256='+signature})
    assert response.status_code==200, response.text
    assert response.json()['task']==original['request']['task']
    assert len(list(args['scene_root'].glob('scene-*/intent.json')))==1


def test_changed_robot_configuration_refuses_before_any_reservation(tmp_path, monkeypatch):
    from blueprint_pipeline import task_evaluation_controls_terminal_adoption as adoption
    from tests.test_task_evaluation_controls_autoprovision import COMMIT
    args, config, path, selected = _case(tmp_path, monkeypatch)
    owner = selected('eval-binding-change')
    intent_path = args['scene_root']/owner['intent_id']/'intent.json'
    intent = intake._read(intent_path, 'intent_digest')
    intent['request']['task']['robot_binding_digest'] = 'sha256:'+'9'*64
    intent['request']['submission_id'] = 'eval-binding-pin'
    intent['request']['task']['evaluation_source']['evaluation_run_id'] = 'eval-binding-pin'
    owner = intake.stage_scene_intent(value=intent['request'], queue_root=args['scene_root'],
        authenticated_client='webapp', trusted_clients={'webapp'}, now=args['now'])
    with pytest.raises(ValueError, match='team_evaluation_robot_binding_changed'):
        adoption.provision_terminal_controls_adoption(config=config, catalog=args['catalog'],
            intent_id=owner['intent_id'], expected_production_commit=COMMIT, now=args['now'])
    assert not list(args['scene_root'].glob('scene-*/attempts/*.json'))
