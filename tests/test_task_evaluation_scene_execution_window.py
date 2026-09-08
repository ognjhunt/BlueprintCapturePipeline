import json

import pytest

from blueprint_pipeline import task_evaluation_scene_intake as intake
from blueprint_pipeline import task_evaluation_scene_execution_authority as authority
from blueprint_pipeline import task_evaluation_scene_execution_window as window
from tests.test_task_evaluation_scene_intake import stage, attempt, request


def extend(root, owner_record, **overrides):
    args = dict(queue_root=root, intent_id=owner_record['intent_id'], intent_digest=owner_record['intent_digest'],
        owner=request()['owner'], authenticated_client='webapp', trusted_clients={'webapp'},
        expires_at_epoch=2000, authorization_reference='explicit-owner-time-only-approval', ack=window.ACK, now=1001)
    args.update(overrides)
    return window.extend_scene_execution_window(**args)


def test_explicit_extension_preserves_intent_attempts_spend_and_retry_caps(tmp_path, monkeypatch):
    owner = stage(tmp_path)
    reserved = attempt(tmp_path, owner)
    original = (tmp_path/owner['intent_id']/'intent.json').read_bytes()
    monkeypatch.setenv(intake.ROOT_ENV, str(tmp_path))
    monkeypatch.setenv(intake.CLIENTS_ENV, 'webapp')
    bound = {**authority.bind_scene_attempt(reserved), 'source_commit': reserved['source_commit']}
    assert authority.scene_execution_authority_blockers(bound, provider='vast', maximum_spend_usd=2, now=1100) == ['scene_execution_owner_expired']
    result = extend(tmp_path, owner)
    assert result['status'] == 'execution_window_extended'
    assert (tmp_path/owner['intent_id']/'intent.json').read_bytes() == original
    assert authority.scene_execution_authority_blockers(bound, provider='vast', maximum_spend_usd=2, now=1100) == []
    attempt(tmp_path, owner, 'a2', now=1100)
    with pytest.raises(intake.SceneIntakeError, match='attempt_cap_exhausted'):
        attempt(tmp_path, owner, 'a3', now=1101)
    assert extend(tmp_path, owner)['status'] == 'execution_window_already_covers_request'
    intent = intake._read(tmp_path/owner['intent_id']/'intent.json', 'intent_digest')
    assert window.effective_execution_expiry(tmp_path/owner['intent_id'], intent) == 2000
    assert intent['request']['execution']['max_total_spend_usd'] == 4
    assert intent['request']['execution']['max_retries'] == 0


@pytest.mark.parametrize('change', [dict(ack=''), dict(authenticated_client='untrusted'), dict(owner={'user_id':'other','organization_id':'org1'}), dict(expires_at_epoch=True), dict(expires_at_epoch=1001+8*86400)])
def test_extension_requires_explicit_same_owner_approval(tmp_path, change):
    owner = stage(tmp_path)
    with pytest.raises(ValueError):
        extend(tmp_path, owner, **change)
    assert not list((tmp_path/owner['intent_id']/window.DIRECTORY).glob('*.json'))


def test_revocation_cannot_be_undone_by_a_time_extension(tmp_path):
    owner = stage(tmp_path)
    intake.revoke_scene_intent(queue_root=tmp_path, intent_id=owner['intent_id'], intent_digest=owner['intent_digest'], owner=request()['owner'], now=103)
    with pytest.raises(ValueError, match='revocation_mismatch'):
        extend(tmp_path, owner)


def test_tampered_limits_in_window_fail_closed(tmp_path):
    owner = stage(tmp_path)
    result = extend(tmp_path, owner)
    path = tmp_path/owner['intent_id']/window.DIRECTORY/(result['extension_digest'].removeprefix('sha256:')+'.json')
    value = json.loads(path.read_text())
    value['unchanged_execution_bounds']['max_total_spend_usd'] = 100
    path.chmod(0o640)
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        attempt(tmp_path, owner, now=1100)
