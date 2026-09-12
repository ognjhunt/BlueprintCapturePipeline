"""Missing robot choice is explicitly assigned, never inferred from one catalog row."""
import copy
import json
from pathlib import Path

import pytest

from blueprint_pipeline import task_evaluation_scene_robot_assignment as assignment
from blueprint_pipeline import task_evaluation_scene_intake as intake
from blueprint_pipeline import task_evaluation_controls_autoprovision as controls
from tests.test_task_evaluation_controls_autoprovision import setup


def authorize(tmp_path, intent, catalog, *, binding_id='franka-droid', name='authorization.json'):
    path = tmp_path / name
    path.write_text(json.dumps({'schema_version':assignment.AUTHORIZATION_SCHEMA, 'authorized':True,
        'intent_id':intent['intent_id'], 'intent_digest':intent['intent_digest'], 'owner':intent['request']['owner'],
        'authenticated_issuer':intent['authenticated_issuer'], 'robot_binding_id':binding_id,
        'catalog_binding_digest':assignment.catalog_binding_digest(catalog['bindings'][binding_id]),
        'authorization_basis':'fixture explicit robot authorization'}))
    path.chmod(0o640)
    return path


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    monkeypatch.setenv(intake.CLIENTS_ENV, 'webapp')
    kwargs = setup(tmp_path, robot_binding_id=None)
    directory = kwargs['link_path'].parent
    intent = intake._read(directory / 'intent.json', 'intent_digest')
    catalog = kwargs['catalog']
    catalog_path = tmp_path / 'catalog.json'
    catalog_path.write_text(json.dumps(catalog))
    auth = authorize(tmp_path, intent, catalog)
    args = dict(queue_root=kwargs['scene_root'], intent_id=intent['intent_id'], intent_digest=intent['intent_digest'],
        owner=intent['request']['owner'], authenticated_client='webapp', trusted_clients={'webapp'},
        robot_catalog_path=catalog_path, robot_binding_id='franka-droid', authorization_reference=auth,
        ack=assignment.ACK, now=kwargs['now'])
    return kwargs, directory, intent, args


def test_real_producer_retains_assignment_without_changing_task_or_old_holds(prepared):
    kwargs, directory, intent, args = prepared
    before = (directory / 'intent.json').read_bytes()
    old = intake.reserve_scene_attempt(queue_root=kwargs['scene_root'], intent_id=intent['intent_id'],
        attempt_id='old-source', source_commit=kwargs['expected_production_commit'], runtime_digest='sha256:'+'a'*64,
        input_digest='sha256:'+'b'*64, provider='vast', maximum_spend_usd=.1, now=kwargs['now'])
    old_path = directory / 'attempts/old-source.json'
    old_bytes = old_path.read_bytes()
    record = assignment.assign_scene_robot(**args)
    assert record['status'] == 'robot_assignment_recorded'
    assert assignment.assign_scene_robot(**args)['assignment_digest'] == record['assignment_digest']
    receipt = controls.provision_link(**kwargs)
    assert receipt['status'] == 'installed' and receipt['robot_assignment_digest'] == record['assignment_digest']
    assert controls.provision_link(**{**kwargs,'now':kwargs['now']+1}) == receipt
    retained = json.loads(next(kwargs['controls_root'].glob('*/*/autoprovision-inputs.json')).read_bytes())
    assert retained['robot_assignment_digest'] == record['assignment_digest']
    assert (directory / 'intent.json').read_bytes() == before
    assert old_path.read_bytes() == old_bytes
    assert intake._read(old_path, 'attempt_digest') == old
    assert 'robot_binding_id' not in intake._read(directory / 'intent.json','intent_digest')['request']['task']
    assert len(list((directory / 'attempts').glob('*.json'))) == 4


def test_no_implicit_single_robot_default_or_paid_reservation(prepared):
    kwargs, directory, _, _ = prepared
    with pytest.raises(ValueError, match='scene_robot_assignment_missing'):
        controls.provision_link(**kwargs)
    assert not list((directory / 'attempts').glob('*.json'))
    assert not kwargs['controls_root'].exists()


@pytest.mark.parametrize('change', [dict(owner={'user_id':'other','organization_id':'org1'}),
    dict(intent_digest='sha256:'+'0'*64), dict(authenticated_client='other',trusted_clients={'other'}),
    dict(authenticated_client='webapp',trusted_clients=set()), dict(ack=''), dict(robot_binding_id='absent')])
def test_wrong_owner_issuer_or_binding_never_issues(prepared, change):
    _, directory, _, args = prepared
    with pytest.raises(ValueError):
        assignment.assign_scene_robot(**{**args,**change})
    assert not (directory / assignment.FILENAME).exists()


@pytest.mark.parametrize('fault', ['digest','owner','robot','authorization_scope','authorization_bytes','authorization_public','record_symlink'])
def test_assignment_and_private_authority_tamper_reject(prepared, fault):
    kwargs, directory, intent, args = prepared
    assignment.assign_scene_robot(**args)
    path = directory / assignment.FILENAME
    if fault.startswith('authorization_'):
        path = Path(args['authorization_reference'])
        if fault == 'authorization_public':
            path.chmod(0o644)
        elif fault == 'authorization_bytes':
            path.write_text(path.read_text()+' ')
        else:
            value=json.loads(path.read_bytes())
            value['owner']={'user_id':'other'}
            path.write_text(json.dumps(value))
    elif fault == 'record_symlink':
        other=directory/'other.json'
        path.rename(other)
        path.symlink_to(other)
    else:
        value=json.loads(path.read_bytes())
        if fault=='digest':
            value['robot_binding_id']='changed'
        else:
            value['owner' if fault=='owner' else 'robot_binding_id'] = {'user_id':'other'} if fault=='owner' else 'absent'
            value=intake._seal(value,'assignment_digest')
        path.chmod(0o640)
        path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        assignment.read_scene_robot_assignment(directory=directory,intent=intent,catalog=kwargs['catalog'],now=args['now'])
    with pytest.raises(ValueError):
        controls.provision_link(**kwargs)
    assert not list((directory/'attempts').glob('*.json'))


@pytest.mark.parametrize('field', ['robot_asset_usd','embodiment_camera_template','runtime_digest',
                                   'phase_hard_cap_usd','openai_api_key_id','openai_project_id','project_spend_current_path'])
def test_resealed_catalog_drift_rejects_before_production(prepared, field):
    kwargs,directory,_,args=prepared
    assignment.assign_scene_robot(**args)
    catalog=copy.deepcopy(kwargs['catalog'])
    catalog['bindings']['franka-droid'][field] = .3 if field=='phase_hard_cap_usd' else 'changed'
    kwargs['catalog']=controls._seal(catalog,'catalog_digest')
    with pytest.raises(ValueError,match='record_invalid'):
        controls.provision_link(**kwargs)
    assert not list((directory/'attempts').glob('*.json'))


def test_release_rebinding_alone_preserves_choice(prepared):
    kwargs,directory,intent,args=prepared
    record=assignment.assign_scene_robot(**args)
    catalog=copy.deepcopy(kwargs['catalog'])
    catalog['bindings']['franka-droid']['expected_production_commit']='d'*40
    catalog=controls._seal(catalog,'catalog_digest')
    found=assignment.read_scene_robot_assignment(directory=directory,intent=intent,catalog=catalog,now=args['now'])
    assert found['assignment_digest']==record['assignment_digest']


def test_revocation_and_expiry_cannot_be_overridden(prepared):
    kwargs,directory,intent,args=prepared
    assignment.assign_scene_robot(**args)
    with pytest.raises(ValueError,match='owner_expired'):
        assignment.read_scene_robot_assignment(directory=directory,intent=intent,catalog=kwargs['catalog'],now=args['now']+7201)
    intake.revoke_scene_intent(queue_root=args['queue_root'],intent_id=intent['intent_id'],
        intent_digest=intent['intent_digest'],owner=intent['request']['owner'],now=args['now'])
    with pytest.raises(ValueError,match='owner_revoked'):
        assignment.assign_scene_robot(**args)
    with pytest.raises(ValueError,match='authority_revoked'):
        controls.provision_link(**kwargs)


def test_existing_explicit_robot_is_never_overridden(tmp_path):
    kwargs=setup(tmp_path)
    directory=kwargs['link_path'].parent
    intent=intake._read(directory/'intent.json','intent_digest')
    binding,row=assignment.resolve_controls_robot_binding(directory=directory,intent=intent,catalog=kwargs['catalog'],now=kwargs['now'])
    assert binding==kwargs['catalog']['bindings']['franka-droid'] and row is None
    auth=authorize(tmp_path,intent,kwargs['catalog'])
    catalog_path=tmp_path/'catalog.json'
    catalog_path.write_text(json.dumps(kwargs['catalog']))
    with pytest.raises(ValueError,match='explicit_choice_present'):
        assignment.assign_scene_robot(queue_root=kwargs['scene_root'],intent_id=intent['intent_id'],
            intent_digest=intent['intent_digest'],owner=intent['request']['owner'],authenticated_client='webapp',
            trusted_clients={'webapp'},robot_catalog_path=catalog_path,robot_binding_id='franka-droid',
            authorization_reference=auth,ack=assignment.ACK,now=kwargs['now'])
    intent['request']['task']['robot_binding_id']='missing-explicit-choice'
    with pytest.raises(ValueError,match='binding_missing'):
        assignment.resolve_controls_robot_binding(directory=directory,intent=intent,catalog=kwargs['catalog'])


def test_write_once_choice_cannot_be_replaced(prepared, tmp_path):
    kwargs,directory,intent,args=prepared
    first=assignment.assign_scene_robot(**args)
    before=(directory/assignment.FILENAME).read_bytes()
    catalog=copy.deepcopy(kwargs['catalog'])
    catalog['bindings']['other']=copy.deepcopy(catalog['bindings']['franka-droid'])
    catalog=controls._seal(catalog,'catalog_digest')
    Path(args['robot_catalog_path']).write_text(json.dumps(catalog))
    auth=authorize(tmp_path,intent,catalog,binding_id='other',name='other-authorization.json')
    with pytest.raises(ValueError,match='immutable_conflict'):
        assignment.assign_scene_robot(**{**args,'robot_binding_id':'other','authorization_reference':auth})
    assert (directory/assignment.FILENAME).read_bytes()==before
    assert first['provider_mutation_performed'] is first['original_task_modified'] is False


@pytest.mark.parametrize('clock', [False, 0, -1, float('nan'), float('inf')])
def test_invalid_clock_cannot_issue_or_reopen(prepared, clock):
    kwargs,directory,intent,args=prepared
    with pytest.raises(ValueError):
        assignment.assign_scene_robot(**{**args,'now':clock})
    assert not (directory/assignment.FILENAME).exists()
    assignment.assign_scene_robot(**args)
    with pytest.raises(ValueError):
        assignment.read_scene_robot_assignment(directory=directory,intent=intent,catalog=kwargs['catalog'],now=clock)


def test_authorization_symlink_never_issues(prepared, tmp_path):
    _,directory,_,args=prepared
    link=tmp_path/'linked-authorization.json'
    link.symlink_to(args['authorization_reference'])
    with pytest.raises(ValueError,match='path_unsafe'):
        assignment.assign_scene_robot(**{**args,'authorization_reference':link})
    assert not (directory/assignment.FILENAME).exists()


@pytest.mark.parametrize('case', ['valid', 'wrong_previous_bucket', 'robot_change', 'no_refresh', 'old_authority'])
def test_explicit_bucket_refresh_preserves_history_and_rejects_other_drift(prepared, tmp_path, case):
    kwargs, directory, intent, args = prepared
    old_catalog = copy.deepcopy(kwargs['catalog'])
    old_catalog['bindings']['franka-droid']['external_layer_bucket'] = 'legacy'
    old_catalog = controls._seal(old_catalog, 'catalog_digest')
    Path(args['robot_catalog_path']).write_text(json.dumps(old_catalog))
    args['authorization_reference'] = authorize(tmp_path, intent, old_catalog)
    old = assignment.assign_scene_robot(**args)
    before = (directory / assignment.FILENAME).read_bytes()
    original_intent = (directory / 'intent.json').read_bytes()
    current = copy.deepcopy(old_catalog)
    current['bindings']['franka-droid']['external_layer_bucket'] = 'artifacts'
    if case == 'robot_change':
        current['bindings']['franka-droid']['phase_hard_cap_usd'] = .37
    current = controls._seal(current, 'catalog_digest')
    Path(args['robot_catalog_path']).write_text(json.dumps(current))
    new_auth = authorize(tmp_path, intent, current, name='new-authorization.json')
    updated = {**args, 'authorization_reference': new_auth, 'previous_external_layer_bucket': 'legacy'}
    if case == 'wrong_previous_bucket':
        updated['previous_external_layer_bucket'] = 'unrelated'
    elif case == 'no_refresh':
        updated.pop('previous_external_layer_bucket')
    elif case == 'old_authority':
        updated['authorization_reference'] = args['authorization_reference']
    if case != 'valid':
        with pytest.raises(ValueError):
            assignment.assign_scene_robot(**updated)
        assert (directory / assignment.FILENAME).read_bytes() == before
        return
    result = assignment.assign_scene_robot(**updated)
    assert result['assignment_digest'] != old['assignment_digest']
    assert result['robot_binding_id'] == old['robot_binding_id']
    assert next((directory / 'robot-assignment-history').glob('*.json')).read_bytes() == before
    assert (directory / 'intent.json').read_bytes() == original_intent
    assert assignment.read_scene_robot_assignment(directory=directory, intent=intent,
        catalog=current, now=args['now'])['assignment_digest'] == result['assignment_digest']
    assert assignment.assign_scene_robot(**updated)['assignment_digest'] == result['assignment_digest']
