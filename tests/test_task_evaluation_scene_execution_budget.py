"""ADP-009D: extend future capacity while retaining every old source hold."""
import copy
import hashlib
import json
from pathlib import Path

import pytest

from blueprint_pipeline import task_evaluation_scene_execution_budget as budget
from blueprint_pipeline import task_evaluation_scene_intake as intake
from blueprint_pipeline import task_evaluation_scene_execution_authority as authority
from tests.test_task_evaluation_scene_intake import stage, attempt, request


def extend(root, owner_record, **overrides):
    args = dict(queue_root=root, intent_id=owner_record['intent_id'], intent_digest=owner_record['intent_digest'],
        owner=request()['owner'], authenticated_client='webapp', trusted_clients={'webapp'},
        max_total_spend_usd=100, max_paid_attempts=18, authorization_reference='delegation:current-session',
        ack=budget.ACK, now=200)
    args.update(overrides)
    return budget.extend_scene_execution_budget(**args)


def loaded(root, owner):
    directory = root / owner['intent_id']
    return directory, intake._read(directory / 'intent.json', 'intent_digest')


def replace_record(result, change, *, reseal=True):
    path = Path(result['record_path'])
    value = json.loads(path.read_bytes())
    change(value)
    if reseal:
        value = intake._seal(value, 'extension_digest')
    path.chmod(0o640)
    path.unlink()
    new = path.with_name(value['extension_digest'][7:] + '.json')
    new.write_text(json.dumps(value))
    return new


def test_actual_eleven_holds_counted_with_100_dollar_18_attempt_extension(tmp_path):
    payload = request()
    payload['execution'].update(max_total_spend_usd=50, max_paid_attempts=12, max_retries=2)
    owner = stage(tmp_path, payload)
    directory, intent = loaded(tmp_path, owner)
    before = (directory / 'intent.json').read_bytes()
    old = [attempt(tmp_path, owner, f'old-{i}', cost=4.5) for i in range(11)]
    old_bytes = {p: p.read_bytes() for p in (directory / 'attempts').iterdir()}
    with pytest.raises(ValueError, match='spend_cap_exhausted'):
        attempt(tmp_path, owner, 'next', cost=4.5)
    grant = extend(tmp_path, owner)
    assert grant['limits'] == {'max_total_spend_usd': 100, 'max_paid_attempts': 18}
    assert grant['historical_reservations_released'] is False
    assert (directory / 'intent.json').read_bytes() == before
    assert all(p.read_bytes() == b for p, b in old_bytes.items())
    assert all(budget.ATTEMPT_GRANT_FIELD not in row for row in old)
    for i in range(7):
        row = attempt(tmp_path, owner, f'new-{i}', cost=4.5, now=201)
        assert row[budget.ATTEMPT_GRANT_FIELD] == grant['extension_digest']
        assert attempt(tmp_path, owner, f'new-{i}', cost=4.5, now=202) == row
    with pytest.raises(ValueError, match='attempt_cap_exhausted'):
        attempt(tmp_path, owner, 'nineteenth', cost=4.5, now=202)
    state = intake.scene_intent_status(queue_root=tmp_path, intent_id=owner['intent_id'], now=202)
    assert sum(r['maximum_spend_usd'] for r in state['attempts']) == 81
    assert state['effective_execution_budget']['max_total_spend_usd'] == 100
    assert state['effective_execution_budget']['max_paid_attempts'] == 18
    assert state['request_digest'] == intake.canonical_digest(payload)
    assert intent['request']['execution'] == payload['execution']


def test_original_exposure_and_per_action_limit_are_not_reset(tmp_path):
    owner = stage(tmp_path)
    attempt(tmp_path, owner)
    extend(tmp_path, owner, max_total_spend_usd=6, max_paid_attempts=4)
    attempt(tmp_path, owner, 'a2', cost=4, now=201)
    with pytest.raises(ValueError, match='spend_cap_exhausted'):
        attempt(tmp_path, owner, 'a3', cost=1, now=201)
    with pytest.raises(ValueError, match='attempt_spend_exceeds_original_limit'):
        attempt(tmp_path, owner, 'a3', cost=5, now=201)


def test_chain_is_monotonic_idempotent_and_old_grants_stay_valid(tmp_path):
    owner = stage(tmp_path)
    first = extend(tmp_path, owner, max_total_spend_usd=10, max_paid_attempts=4)
    row = attempt(tmp_path, owner, now=201)
    second = extend(tmp_path, owner, now=202)
    assert second['sequence'] == 2 and second['predecessor_digest'] == first['extension_digest']
    assert extend(tmp_path, owner, now=203)['status'] == 'execution_budget_already_covers_request'
    assert attempt(tmp_path, owner, now=204) == row
    assert len(list((tmp_path / owner['intent_id'] / budget.DIRECTORY).iterdir())) == 2
    directory, intent = loaded(tmp_path, owner)
    budget.validate_attempt_execution_budget(directory, intent, row)
    with pytest.raises(ValueError, match='not_monotonic'):
        extend(tmp_path, owner, max_total_spend_usd=101, max_paid_attempts=17, now=203)


@pytest.mark.parametrize('change', [dict(ack=''), dict(authenticated_client='untrusted'),
    dict(authenticated_client='other', trusted_clients={'other'}), dict(intent_digest='sha256:'+'0'*64),
    dict(owner={'user_id': 'other', 'organization_id': 'org1'}), dict(authorization_reference=''),
    dict(max_total_spend_usd=True), dict(max_total_spend_usd=float('nan')),
    dict(max_total_spend_usd=float('inf')), dict(max_total_spend_usd=1001),
    dict(max_paid_attempts=True), dict(max_paid_attempts=1.5), dict(max_paid_attempts=33),
    dict(max_paid_attempts=0), dict(now=1000), dict(now=99)])
def test_invalid_or_unapproved_extensions_do_not_write(tmp_path, change):
    owner = stage(tmp_path)
    with pytest.raises(ValueError):
        extend(tmp_path, owner, **change)
    assert not (tmp_path / owner['intent_id'] / budget.DIRECTORY).exists()


def test_revocation_is_not_overridden(tmp_path):
    owner = stage(tmp_path)
    extend(tmp_path, owner)
    intake.revoke_scene_intent(queue_root=tmp_path, intent_id=owner['intent_id'],
        intent_digest=owner['intent_digest'], owner=request()['owner'], now=201)
    with pytest.raises(ValueError, match='revocation'):
        extend(tmp_path, owner, max_total_spend_usd=200, now=202)
    with pytest.raises(ValueError, match='revoked'):
        attempt(tmp_path, owner, now=202)


@pytest.mark.parametrize('field,value', [('owner', {'user_id':'other'}),
    ('intent_digest', 'sha256:'+'0'*64), ('authenticated_issuer','other'),
    ('sequence', 2), ('predecessor_digest','sha256:'+'0'*64),
    ('historical_reservations_released', True), ('provider_mutation_performed', True)])
def test_resealed_wrong_identity_or_chain_refused(tmp_path, field, value):
    owner = stage(tmp_path)
    grant = extend(tmp_path, owner)
    replace_record(grant, lambda r: r.update({field:value}))
    with pytest.raises(ValueError):
        attempt(tmp_path, owner, now=201)


@pytest.mark.parametrize('field', ['max_retries', 'expires_at_epoch', 'allowed_providers',
                                    'policy_candidates', 'claim_scope'])
def test_other_execution_authority_cannot_change(tmp_path, field):
    owner = stage(tmp_path)
    grant = extend(tmp_path, owner)
    replace_record(grant, lambda r: r['unchanged_execution_bounds'].update({field: 'changed'}))
    with pytest.raises(ValueError, match='record_invalid'):
        attempt(tmp_path, owner, now=201)


def test_fork_or_removed_predecessor_refused(tmp_path):
    owner = stage(tmp_path)
    first = extend(tmp_path, owner, max_total_spend_usd=10, max_paid_attempts=4)
    second = extend(tmp_path, owner, now=201)
    replace_record(second, lambda r: r.update(sequence=1, predecessor_digest=owner['intent_digest']))
    directory, intent = loaded(tmp_path, owner)
    with pytest.raises(ValueError, match='chain_invalid'):
        budget.effective_execution_budget(directory, intent)
    Path(first['record_path']).unlink()
    with pytest.raises(ValueError, match='chain_invalid'):
        budget.effective_execution_budget(directory, intent)


@pytest.mark.parametrize('kind', ['raw_tamper', 'missing', 'wrong_name', 'symlink_record', 'symlink_store'])
def test_bound_grant_reopened_at_paid_admission_and_status(tmp_path, monkeypatch, kind):
    owner = stage(tmp_path)
    grant = extend(tmp_path, owner)
    row = attempt(tmp_path, owner, now=201)
    monkeypatch.setenv(intake.CLIENTS_ENV, 'webapp')
    bound = {**authority.bind_scene_attempt(row), 'source_commit':row['source_commit']}
    args = dict(provider='vast', maximum_spend_usd=2, queue_root=tmp_path, now=202)
    assert authority.scene_execution_authority_blockers(bound, **args) == []
    path = Path(grant['record_path'])
    if kind == 'raw_tamper':
        replace_record(grant, lambda r: r['limits'].update(max_total_spend_usd=999), reseal=False)
    elif kind == 'missing':
        path.unlink()
    elif kind == 'wrong_name':
        path.rename(path.with_name('wrong.json'))
    elif kind == 'symlink_record':
        target = tmp_path / 'external.json'
        path.rename(target)
        path.symlink_to(target)
    else:
        target = tmp_path / 'external'
        path.parent.rename(target)
        path.parent.symlink_to(target, target_is_directory=True)
    assert authority.scene_execution_authority_blockers(bound, **args) == ['scene_execution_owner_budget_extension_invalid']
    with pytest.raises(ValueError):
        intake.scene_intent_status(queue_root=tmp_path, intent_id=owner['intent_id'], now=202)
    with pytest.raises(ValueError):
        attempt(tmp_path, owner, now=202)


def test_owner_reader_checks_grants_without_changing_original_request(tmp_path, monkeypatch):
    from blueprint_pipeline.task_evaluation_scene_owner_authority import reopen_scene_intent
    owner = stage(tmp_path)
    monkeypatch.setenv(intake.ROOT_ENV, str(tmp_path))
    monkeypatch.setenv(intake.CLIENTS_ENV, 'webapp')
    path = tmp_path / owner['intent_id'] / 'intent.json'
    ref = {'path':str(path), 'sha256':'sha256:'+hashlib.sha256(path.read_bytes()).hexdigest(),
           'size_bytes':path.stat().st_size}
    before = copy.deepcopy(reopen_scene_intent(ref, now=201))
    grant = extend(tmp_path, owner)
    assert reopen_scene_intent(ref, now=201) == before
    replace_record(grant, lambda r: r.update(authenticated_issuer='other'))
    with pytest.raises(ValueError):
        reopen_scene_intent(ref, now=201)


def test_symlink_intent_directory_cannot_issue(tmp_path):
    owner = stage(tmp_path)
    directory = tmp_path / owner['intent_id']
    target = tmp_path / 'other'
    directory.rename(target)
    directory.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match='unsafe'):
        extend(tmp_path, owner)
    assert not (target / budget.DIRECTORY).exists()


def test_budget_does_not_extend_time_or_retry_authority(tmp_path):
    from blueprint_pipeline.task_evaluation_scene_execution_window import effective_execution_expiry
    owner = stage(tmp_path)
    extend(tmp_path, owner, max_total_spend_usd=1000, max_paid_attempts=32)
    directory, intent = loaded(tmp_path, owner)
    assert effective_execution_expiry(directory, intent) == 1000
    assert intent['request']['execution']['max_retries'] == 0
    with pytest.raises(ValueError, match='authority_expired'):
        attempt(tmp_path, owner, now=1000)
    assert not list((directory / 'attempts').glob('*.json'))


def test_review_owner_reopens_exact_attempt_grant_even_without_description_seed(tmp_path, monkeypatch):
    from blueprint_pipeline.task_evaluation_scene_owner_authority import validate_task_scene_owner
    owner = stage(tmp_path)
    grant = extend(tmp_path, owner)
    row = attempt(tmp_path, owner, now=201)
    monkeypatch.setenv(intake.ROOT_ENV, str(tmp_path))
    monkeypatch.setenv(intake.CLIENTS_ENV, 'webapp')
    directory = tmp_path / owner['intent_id']
    def ref(path):
        return {'path':str(path), 'sha256':'sha256:'+hashlib.sha256(path.read_bytes()).hexdigest(),
                'size_bytes':path.stat().st_size}
    task = {'scene_intent_authority': {'intent':ref(directory / 'intent.json'),
        'intent_digest':owner['intent_digest'], 'attempt':ref(directory / 'attempts' / (row['attempt_id']+'.json'))}}
    Path(grant['record_path']).unlink()
    with pytest.raises(ValueError, match='attempt_grant_invalid'):
        validate_task_scene_owner(task, now=202)


def test_future_grant_cannot_backdate_a_new_reservation(tmp_path):
    owner = stage(tmp_path)
    extend(tmp_path, owner)
    with pytest.raises(ValueError, match='attempt_grant_invalid'):
        attempt(tmp_path, owner, now=199)
    assert not list((tmp_path / owner['intent_id'] / 'attempts').glob('*.json'))
