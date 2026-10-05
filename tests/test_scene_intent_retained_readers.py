"""ADP-050/day-28: retained owner evidence cannot gain execution imports or authority."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

from blueprint_pipeline import task_evaluation_scene_execution_budget_evidence as budget
from blueprint_pipeline import task_evaluation_scene_execution_window_evidence as window
from blueprint_pipeline import task_evaluation_scene_intent_contracts as contracts
from blueprint_pipeline import task_evaluation_scene_owner_authority as owner
from tests.test_live_pipeline_import_isolation import HOT_LANE_MODULES, _transitive_local_modules


def _write(path, value, field):
    sealed = contracts._seal(value, field)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sealed))
    return sealed


def _reference(path):
    return {'path': str(path), 'sha256': 'sha256:' + hashlib.sha256(path.read_bytes()).hexdigest(),
            'size_bytes': path.stat().st_size}


@pytest.fixture
def retained_intent(tmp_path, monkeypatch):
    request = {'schema_version': contracts.REQUEST_SCHEMA, 'submission_id': 'upload-1',
        'owner': {'user_id': 'u1', 'organization_id': 'org1'},
        'source': {'kind': 'mesh', 'binding_id': 'mesh-1', 'content_digest': 'sha256:' + 'a' * 64},
        'task': {'task_id': 'pick-book', 'strategy': 'pick_and_place', 'subject': {'id': 'book'},
                 'support': {'id': 'table'}, 'destination': {'id': 'tray'}, 'success': {'inside': True}},
        'execution': {'max_total_spend_usd': 4, 'max_paid_attempts': 2, 'max_retries': 0,
            'expires_at_epoch': 1000, 'allowed_providers': ['vast', 'openai'],
            'claim_scope': 'development_only', 'policy_candidates': [
                {'id': candidate, 'artifact_digest': 'sha256:' + 'b' * 64}
                for candidate in contracts.SUPPORTED_POLICY_CANDIDATE_IDS]},
        'consent': {'accepted_by': 'u1', 'accepted_at_epoch': 99, 'rights_reference': 'rights-v1',
            'provider_terms_reference': 'terms-v1', 'private_processing_authorized': True,
            'provider_training_authorized': False, 'task_confirmed': True, 'spend_authorized': True}}
    directory = tmp_path / 'scene-retained'
    intent = _write(directory / 'intent.json', {
        'schema_version': contracts.INTENT_SCHEMA, 'intent_id': directory.name,
        'request': contracts.validate_request(request, now=100), 'authenticated_issuer': 'webapp',
        'accepted_at_epoch': 100, 'source_content_digest': request['source']['content_digest'],
        'task_content_digest': contracts.canonical_digest(request['task']),
        'provider_mutation_performed': False}, 'intent_digest')
    monkeypatch.setenv(contracts.ROOT_ENV, str(tmp_path))
    monkeypatch.setenv(contracts.CLIENTS_ENV, 'webapp')
    return directory, intent


def _budget_grant(directory, intent, **changes):
    execution = intent['request']['execution']
    original = {key: execution[key] for key in budget.LIMIT_FIELDS}
    record = {'schema_version': budget.SCHEMA, 'scope': 'cumulative_budget_and_attempts_only',
        'intent_id': intent['intent_id'], 'intent_digest': intent['intent_digest'],
        'owner': intent['request']['owner'], 'authenticated_issuer': intent['authenticated_issuer'],
        'authorization_reference': 'explicit-owner-grant', 'original_limits': original,
        'prior_limits': original, 'limits': {'max_total_spend_usd': 100, 'max_paid_attempts': 18},
        'unchanged_execution_bounds': {key: value for key, value in execution.items()
                                       if key not in budget.LIMIT_FIELDS},
        'sequence': 1, 'predecessor_digest': intent['intent_digest'], 'issued_at_epoch': 200,
        'provider_mutation_performed': False, 'historical_reservations_released': False,
        **changes}
    sealed = contracts._seal(record, 'extension_digest')
    path = directory / budget.DIRECTORY / (sealed['extension_digest'][7:] + '.json')
    _write(path, record, 'extension_digest')
    return sealed


def _window_grant(directory, intent, **changes):
    record = {'schema_version': window.SCHEMA, 'scope': 'execution_time_only',
        'intent_id': intent['intent_id'], 'intent_digest': intent['intent_digest'],
        'owner': intent['request']['owner'], 'authenticated_issuer': intent['authenticated_issuer'],
        'authorization_reference': 'explicit-owner-time-grant', 'issued_at_epoch': 1001,
        'expires_at_epoch': 2000, 'original_expires_at_epoch': 1000,
        'unchanged_execution_bounds': window._bounds(intent), 'provider_mutation_performed': False,
        **changes}
    sealed = contracts._seal(record, 'extension_digest')
    path = directory / window.DIRECTORY / (sealed['extension_digest'][7:] + '.json')
    _write(path, record, 'extension_digest')
    return sealed


def _bound_task(directory, intent):
    task = copy.deepcopy(intent['request']['task'])
    task['task_identity'] = {'id': task.pop('task_id')}
    task['scene_intent_authority'] = {
        'intent': _reference(directory / 'intent.json'), 'intent_digest': intent['intent_digest']}
    task['human_authority'] = {'accepted_by': 'u1',
        'accepted_on': datetime.fromtimestamp(99, timezone.utc).isoformat(),
        'authority_reference': 'scene-intent:' + intent['intent_digest']}
    return task


def test_pure_reader_static_closure_excludes_intake_mutation_and_execution():
    reachable = _transitive_local_modules((
        'task_evaluation_scene_intent_contracts', 'task_evaluation_scene_execution_budget_evidence',
        'task_evaluation_scene_execution_window_evidence', 'task_evaluation_scene_owner_authority'))
    assert 'decision_evidence_contracts' in reachable
    forbidden = HOT_LANE_MODULES | {'task_evaluation_scene_intake',
        'paid_attempt_authority',
        'task_evaluation_scene_execution_budget', 'task_evaluation_scene_execution_window',
        'task_evaluation_scene_recovery', 'source_calibration_finalization_reuse',
        'task_evaluation_stage_replay', 'task_evaluation_sam31_preparation_execution'}
    assert not reachable.intersection(forbidden)


def test_owner_reader_imports_in_isolated_process_without_mutation_modules():
    script = '''
import socket
import sys
def refuse_network(*args, **kwargs):
    raise AssertionError('retained reader attempted network access')
socket.socket.connect = refuse_network
socket.socket.connect_ex = refuse_network
import blueprint_pipeline.task_evaluation_scene_owner_authority
import blueprint_pipeline.task_evaluation_scene_execution_budget_evidence
import blueprint_pipeline.task_evaluation_scene_execution_window_evidence
assert 'blueprint_pipeline.paid_attempt_authority' not in sys.modules
assert 'blueprint_pipeline.task_evaluation_scene_intake' not in sys.modules
assert 'blueprint_pipeline.task_evaluation_scene_execution_budget' not in sys.modules
assert 'blueprint_pipeline.task_evaluation_scene_execution_window' not in sys.modules
assert 'blueprint_pipeline.gpu_render_providers' not in sys.modules
'''
    env = {**os.environ, 'PYTHONPATH': str(Path(__file__).parents[1] / 'src'),
           'PYTHONDONTWRITEBYTECODE': '1'}
    result = subprocess.run([sys.executable, '-c', script], env=env, capture_output=True,
                            text=True, timeout=15, check=False)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize('fault,code', [('revoked', 'scene_owner_authority_revoked'),
    ('expired', 'scene_owner_authority_expired'), ('issuer', 'scene_owner_issuer_not_trusted'),
    ('root', 'scene_owner_intent_path_invalid'), ('digest', 'record_digest_invalid')])
def test_retained_owner_refuses_invalid_authority(retained_intent, monkeypatch, fault, code):
    directory, intent = retained_intent
    now = 201
    if fault == 'revoked':
        (directory / 'revoked.json').write_text('{}')
    elif fault == 'expired':
        now = 1000
    elif fault == 'issuer':
        _write(directory / 'intent.json', {**intent, 'authenticated_issuer': 'forged'}, 'intent_digest')
    elif fault == 'root':
        monkeypatch.setenv(contracts.ROOT_ENV, str(directory.parent / 'wrong-root'))
    else:
        changed = {**intent, 'authenticated_issuer': 'forged'}
        (directory / 'intent.json').write_text(json.dumps(changed))
    before = {path: path.read_bytes() for path in directory.rglob('*') if path.is_file()}
    with pytest.raises(ValueError, match=code):
        owner.reopen_scene_intent(_reference(directory / 'intent.json'), now=now)
    assert before == {path: path.read_bytes() for path in directory.rglob('*') if path.is_file()}


@pytest.mark.parametrize('field,value', [('predecessor_digest', 'sha256:' + '0' * 64),
    ('sequence', 2), ('historical_reservations_released', True),
    ('authenticated_issuer', 'forged')])
def test_pure_budget_refuses_resealed_invalid_grants(retained_intent, field, value):
    directory, intent = retained_intent
    _budget_grant(directory, intent, **{field: value})
    with pytest.raises(ValueError, match='scene_execution_budget_(chain|record)_invalid'):
        budget.effective_execution_budget(directory, intent)
    with pytest.raises(ValueError, match='scene_execution_budget_(chain|record)_invalid'):
        owner.reopen_scene_intent(_reference(directory / 'intent.json'), now=201)


def test_pure_budget_refuses_changed_provider_bounds(retained_intent):
    directory, intent = retained_intent
    changed = {**intent['request']['execution'], 'allowed_providers': ['anthropic']}
    _budget_grant(directory, intent, unchanged_execution_bounds={
        key: value for key, value in changed.items() if key not in budget.LIMIT_FIELDS})
    with pytest.raises(ValueError, match='scene_execution_budget_record_invalid'):
        budget.effective_execution_budget(directory, intent)


def test_pure_owner_refuses_changed_task_and_provider_binding(retained_intent):
    directory, intent = retained_intent
    task = _bound_task(directory, intent)
    assert owner.validate_task_scene_owner(task, now=201) == intent
    task['subject']['id'] = 'another-object'
    with pytest.raises(ValueError, match='scene_owner_task_mismatch'):
        owner.validate_task_scene_owner(task, now=201)
    altered = copy.deepcopy(intent)
    altered['request']['execution']['allowed_providers'] = ['vast']
    intent = _write(directory / 'intent.json', altered, 'intent_digest')
    with pytest.raises(ValueError, match='scene_owner_review_not_authorized'):
        owner.validate_task_scene_owner(_bound_task(directory, intent), now=201)


def test_pure_read_preserves_original_holds_and_exact_attempt_grant(retained_intent):
    directory, intent = retained_intent
    original = _write(directory / 'attempts' / 'historical.json', {
        'schema_version': contracts.ATTEMPT_SCHEMA, 'intent_id': intent['intent_id'],
        'intent_digest': intent['intent_digest'], 'attempt_id': 'historical', 'provider': 'vast',
        'maximum_spend_usd': 4, 'reserved_at_epoch': 101}, 'attempt_digest')
    grant = _budget_grant(directory, intent)
    current = {**original, 'reserved_at_epoch': 201,
               budget.ATTEMPT_GRANT_FIELD: grant['extension_digest']}
    before = {path: path.read_bytes() for path in directory.rglob('*') if path.is_file()}
    budget.validate_attempt_execution_budget(directory, intent, original)
    budget.validate_attempt_execution_budget(directory, intent, current)
    assert budget.effective_execution_budget(directory, intent)['max_total_spend_usd'] == 100
    for changed in ({budget.ATTEMPT_GRANT_FIELD: 'sha256:' + '0' * 64},
                    {'intent_id': 'another'}, {'intent_digest': 'sha256:' + '0' * 64},
                    {'reserved_at_epoch': 199}):
        with pytest.raises(ValueError, match='scene_execution_budget_attempt_grant_invalid'):
            budget.validate_attempt_execution_budget(directory, intent, {**current, **changed})
    assert before == {path: path.read_bytes() for path in directory.rglob('*') if path.is_file()}
    assert budget.ATTEMPT_GRANT_FIELD not in original


def test_pure_window_preserves_bounds_and_owner_expiry(retained_intent):
    directory, intent = retained_intent
    ref = _reference(directory / 'intent.json')
    with pytest.raises(ValueError, match='scene_owner_authority_expired'):
        owner.reopen_scene_intent(ref, now=1100)
    _window_grant(directory, intent)
    before = {path: path.read_bytes() for path in directory.rglob('*') if path.is_file()}
    assert owner.reopen_scene_intent(ref, now=1100) == intent
    assert window.effective_execution_expiry(directory, intent) == 2000
    assert intent['request']['execution']['max_total_spend_usd'] == 4
    assert before == {path: path.read_bytes() for path in directory.rglob('*') if path.is_file()}


@pytest.mark.parametrize('field,value', [('owner', {'user_id': 'forged'}),
    ('intent_digest', 'sha256:' + '0' * 64), ('expires_at_epoch', True),
    ('unchanged_execution_bounds', {'max_total_spend_usd': 100})])
def test_pure_window_refuses_resealed_identity_or_bounds(retained_intent, field, value):
    directory, intent = retained_intent
    _window_grant(directory, intent, **{field: value})
    with pytest.raises(ValueError, match='scene_execution_window_extension_invalid'):
        window.effective_execution_expiry(directory, intent)


def test_original_unbounded_reader_retains_digest_and_leaf_symlink_checks(tmp_path):
    path = tmp_path / 'record.json'
    value = _write(path, {'payload': 'x' * (4 * 1024 * 1024 + 1)}, 'record_digest')
    assert contracts._read(path, 'record_digest') == value
    linked = tmp_path / 'linked.json'
    linked.symlink_to(path)
    with pytest.raises(contracts.SceneIntakeError, match='scene_intake_record_unsafe'):
        contracts._read(linked, 'record_digest')


def test_original_modules_reexport_the_same_readers():
    from blueprint_pipeline import task_evaluation_scene_execution_budget as original_budget
    from blueprint_pipeline import task_evaluation_scene_execution_window as original_window
    from blueprint_pipeline import task_evaluation_scene_intake as original_intake

    assert original_intake.SceneIntakeError is contracts.SceneIntakeError
    assert original_intake._read is contracts._read
    assert original_intake.validate_request is contracts.validate_request
    assert original_intake._number is contracts._number
    assert original_budget.effective_execution_budget is budget.effective_execution_budget
    assert original_budget.validate_attempt_execution_budget is budget.validate_attempt_execution_budget
    assert original_window.effective_execution_expiry is window.effective_execution_expiry
