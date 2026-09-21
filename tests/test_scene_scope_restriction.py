"""ADP-009D: owner can stop automatic controls without cancelling scene creation."""
import json

import pytest

from blueprint_pipeline import task_evaluation_scene_scope_restriction as scope
from blueprint_pipeline import task_evaluation_scene_intake as intake
from blueprint_pipeline.task_evaluation_scene_execution_authority import bind_scene_attempt, scene_execution_authority_blockers
from tests.test_task_evaluation_scene_intake import stage, attempt


def restrict(root, intent, **changes):
    args = dict(queue_root=root, intent_id=intent['intent_id'], intent_digest=intent['intent_digest'],
        owner=intent['request']['owner'], authenticated_client='webapp', trusted_clients={'webapp'},
        authorization_reference='owner-request:scene-only', ack=scope.ACK, now=102)
    return scope.restrict_to_scene_preparation(**{**args, **changes})


def test_restriction_preserves_history_and_stops_only_new_controls(tmp_path, monkeypatch):
    staged = stage(tmp_path)
    directory = tmp_path / staged['intent_id']
    intent = intake._read(directory / 'intent.json', 'intent_digest')
    controls = attempt(tmp_path, intent, 'controls-1')
    scene = attempt(tmp_path, intent, 'scene-configuration-' + 'f' * 24)
    before = {p: p.read_bytes() for p in [directory / 'intent.json', *directory.glob('attempts/*.json')]}
    assert not scope.preparation_only(directory=directory, intent=intent)
    result = restrict(tmp_path, intent)
    assert result['historical_reservations_released'] is False
    assert scope.preparation_only(directory=directory, intent=intent)
    assert restrict(tmp_path, intent)['status'] == 'already_restricted'
    assert {p: p.read_bytes() for p in before} == before
    monkeypatch.setenv(intake.ROOT_ENV, str(tmp_path))
    monkeypatch.setenv(intake.CLIENTS_ENV, 'webapp')
    args = dict(provider='vast', maximum_spend_usd=2, source_commit=scene['source_commit'], now=103)
    assert scene_execution_authority_blockers(bind_scene_attempt(scene), **args) == []
    assert scene_execution_authority_blockers(
        {**bind_scene_attempt(scene), 'task_evaluation_run': {'run_mode': 'robot_evaluation'}}, **args
    ) == ['scene_execution_owner_scope_excludes_robot_controls']
    assert scene_execution_authority_blockers(bind_scene_attempt(controls), **args) == [
        'scene_execution_owner_scope_excludes_robot_controls']
    assert scene_execution_authority_blockers(bind_scene_attempt(controls), reopen_records=False, **args) == []
    with pytest.raises(ValueError, match='spend_cap_exhausted'):
        attempt(tmp_path, intent, 'scene-configuration-2')


@pytest.mark.parametrize('change', ['owner', 'issuer', 'digest'])
def test_scope_restriction_rejects_unbound_authority(tmp_path, change):
    staged = stage(tmp_path)
    directory = tmp_path / staged['intent_id']
    intent = intake._read(directory / 'intent.json', 'intent_digest')
    changes = {'owner': {'user_id': 'foreign'}, 'authenticated_client': 'foreign', 'intent_digest': 'sha256:' + 'f' * 64}
    key = {'owner': 'owner', 'issuer': 'authenticated_client', 'digest': 'intent_digest'}[change]
    with pytest.raises(ValueError):
        restrict(tmp_path, intent, **{key: changes[key]})
    assert not (directory / scope.FILE).exists()


def test_tampered_restriction_fails_closed(tmp_path):
    staged = stage(tmp_path)
    directory = tmp_path / staged['intent_id']
    intent = intake._read(directory / 'intent.json', 'intent_digest')
    restrict(tmp_path, intent)
    path = directory / scope.FILE
    value = json.loads(path.read_text())
    value['owner']['user_id'] = 'foreign'
    path.chmod(0o600)
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        scope.preparation_only(directory=directory, intent=intent)


def test_legacy_preparation_link_is_restricted_before_robot_provisioning(tmp_path):
    from tests.test_task_evaluation_controls_autoprovision import setup
    from blueprint_pipeline import task_evaluation_controls_autoprovision as controls
    kwargs = setup(tmp_path)
    directory = kwargs['link_path'].parent
    intent = intake._read(directory / 'intent.json', 'intent_digest')
    restrict(kwargs['scene_root'], intent)
    def forbidden(**unused):
        pytest.fail('restricted owner must not provision a robot')
    kwargs.update(provisioner=forbidden, installer=forbidden)
    assert controls.provision_link(**kwargs)['status'] == 'scene_preparation_only'
    config = tmp_path / 'controls.json'
    config.write_text(json.dumps({'scene_root': str(kwargs['scene_root']), 'trusted_clients': ['webapp'],
        'preparation_queue_root': str(kwargs['preparation_queue_root'])}))
    assert controls.process_config(config, expected_production_commit=kwargs['expected_production_commit'])[0]['status'] == 'scene_preparation_only'
    assert not list((directory / 'attempts').glob('*.json'))
