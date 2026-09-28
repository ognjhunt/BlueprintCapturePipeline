# Covers (for impacted-test selection):
#   src/blueprint_pipeline/task_evaluation_scene_retirement_intent_receipt.py
"""Plan11 actual intent projection precedes payload changes and preserves history."""
import hashlib
import json
import os
import stat
from pathlib import Path

import pytest


def raw_ref(path):
    raw = path.read_bytes()
    return {'path': str(path), 'sha256': 'sha256:' + hashlib.sha256(raw).hexdigest(),
            'size_bytes': len(raw)}


def operation(tmp_path, monkeypatch, *, member_count=1, with_transport=False):
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest
    from blueprint_pipeline.task_evaluation_scene_retirement_journal import SceneJournal
    from blueprint_pipeline.task_evaluation_scene_retirement_mutation import inventory_digest
    from blueprint_pipeline.task_evaluation_scene_retirement_preservation import ActionAllowance, preserve_members
    from tests.test_scene_retirement_preservation import MemoryTransport
    from tests.test_scene_retirement_real_participants import access_fixture
    from tests.test_task_evaluation_scene_intake import stage

    base = tmp_path.resolve()
    monkeypatch.delenv('BLUEPRINT_SCENE_RETIREMENT_POLICY_FILE', raising=False)
    intake = base / 'intents'
    issued = stage(intake)
    intent_path = intake / issued['intent_id'] / 'intent.json'
    _, policy, member = access_fixture(base, monkeypatch)
    paths = [member]
    for index in range(1, member_count):
        sibling = member.parent / ('scene-' + str(index))
        sibling.mkdir()
        paths.append(sibling)
        policy['roots'].append({'root': str(sibling), 'storage_class': 'evidence', 'device': sibling.stat().st_dev})
    policy['reference_context'] = {'roots': {'intent_root': str(intake)}}
    policy['policy_digest'] = canonical_digest(policy, digest_field='policy_digest')
    Path(os.environ['BLUEPRINT_SCENE_RETIREMENT_POLICY_FILE']).write_text(json.dumps(policy))
    payload = member / 'proof.bin'
    for index, path in enumerate(paths):
        (path / 'proof.bin').write_bytes(b'actual-preserved-proof' if index == 0 else b'second-member-proof')
    allowance = ActionAllowance(expires_at=999, now=lambda: 200, monotonic=lambda: 0)
    transport = MemoryTransport(paths)
    preserved = preserve_members(paths, transport=transport,
                                 allowance=allowance, token='1' * 32)
    plan = base / 'plan.json'
    plan.write_text('{"action":"KEEP"}')
    plan.chmod(0o600)
    members = [{'canonical_path': str(path), 'class': 'host',
                'owner_intent_id': issued['intent_id'], 'owner_raw_ref': raw_ref(intent_path),
                'generation_id': str(index + 2) * 32, 'inventory_sha256': inventory_digest(preserved, index)}
               for index, path in enumerate(paths)]
    consent = {'intent_id': issued['intent_id'], 'intent_raw_ref': raw_ref(intent_path),
               'plan_raw_ref': raw_ref(plan), 'members': members}
    store = Path(policy['journal_store'])
    store.mkdir(mode=0o700)
    (store / 'retired').mkdir(mode=0o700)
    journal = SceneJournal.create(store, token='1' * 32, allowance=allowance,
        initial={'schema_version': 'scene_retirement_journal.v1', 'status': 'pending',
                 'intent_id': consent['intent_id'], 'intent_raw_ref': consent['intent_raw_ref'],
                 'plan_raw_ref': consent['plan_raw_ref'], 'members': members, 'preserved': preserved})
    result = (policy, consent, journal, preserved, allowance, payload)
    return (*result, transport) if with_transport else result


def test_pending_is_durable_in_actual_intent_before_any_source_mutation(tmp_path, monkeypatch):
    from blueprint_pipeline.task_evaluation_scene_retirement_intent_receipt import publish_pending_receipt
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest
    policy, consent, journal, preserved, allowance, payload = operation(tmp_path, monkeypatch)
    result = publish_pending_receipt(policy, consent, journal, preserved, allowance)
    path = Path(result['path'])
    assert path == Path(consent['intent_raw_ref']['path']).parent / 'scene-retired.v1.json'
    assert result == raw_ref(path)
    value = json.loads(path.read_bytes())
    assert value['schema_version'] == 'scene_lifecycle_retirement_receipt.v1'
    assert value['status'] == 'pending' and value['retiring_token'] == journal.token
    assert value['intent_id'] == consent['intent_id'] and value['intent_raw_ref'] == consent['intent_raw_ref']
    assert value['plan_raw_ref'] == consent['plan_raw_ref']
    assert value['journal_initial_raw_ref'] == journal.initial_ref
    assert value['archive'] == preserved['archive']
    assert value['planned_unique_allocated_bytes'] == preserved['unique_allocated_bytes']
    assert value['members'][0]['canonical_path'] == str(payload.parent)
    assert value['members'][0]['generation_id'] == '2' * 32
    assert value['members'][0]['inventory_sha256'] == consent['members'][0]['inventory_sha256']
    assert value['members'][0]['action'] == 'pending'
    assert value['receipt_digest'] == canonical_digest(value, digest_field='receipt_digest')
    assert stat.S_IMODE(path.stat().st_mode) == 0o644
    assert payload.read_bytes() == b'actual-preserved-proof'
    assert any(p.read_bytes() == path.read_bytes() for p in path.parent.glob('scene-retired.*.pending.json'))
    assert 'consent_raw_ref' not in value and 'principal_id' not in value


def test_terminal_updates_only_exact_pending_and_preserves_immutable_history(tmp_path, monkeypatch):
    from blueprint_pipeline.task_evaluation_scene_retirement_intent_receipt import publish_pending_receipt, publish_terminal_receipt
    from blueprint_pipeline.task_evaluation_scene_retirement_mutation import detach_and_remove
    policy, consent, journal, preserved, allowance, _ = operation(tmp_path, monkeypatch)
    pending = publish_pending_receipt(policy, consent, journal, preserved, allowance)
    original = Path(pending['path']).read_bytes()
    outcome = detach_and_remove(preserved, member_index=0, generation_id=consent['members'][0]['generation_id'],
                                journal=journal)
    snapshot = journal.retired_snapshot({'schema_version': 'scene_retirement_journal.v1',
        'intent_id': consent['intent_id'], 'members': consent['members'], 'outcomes': [outcome]})
    final = {'status': 'retired', 'intent_id': consent['intent_id'], 'token': journal.token,
             'members': [outcome],
             'retired_journal_raw_ref': snapshot}
    terminal = publish_terminal_receipt(policy, consent, pending, final, allowance)
    value = json.loads(Path(terminal['path']).read_bytes())
    assert value['status'] == 'retired'
    previous = value['pending_receipt_raw_ref']
    assert previous['path'] != pending['path']  # The current projection advances.
    assert previous['sha256'] == pending['sha256'] and previous['size_bytes'] == pending['size_bytes']
    assert raw_ref(Path(previous['path'])) == previous  # Immutable chain remains resolvable.
    assert value['retired_journal_raw_ref'] == snapshot and terminal == raw_ref(Path(terminal['path']))
    for key in ('logical_bytes', 'apparent_bytes', 'unique_allocated_bytes', 'removed_allocated_bytes',
                'removed_file_count', 'allocation_method', 'event_raw_ref'):
        assert value['members'][0][key] == outcome[key]
    assert any(p.read_bytes() == original for p in Path(pending['path']).parent.glob('scene-retired.*.pending.json'))
    assert any(p.read_bytes() == Path(terminal['path']).read_bytes()
               for p in Path(pending['path']).parent.glob('scene-retired.*.terminal.json'))


@pytest.mark.parametrize('drift', ['count', 'generation', 'event', 'snapshot'])
def test_terminal_measured_outcomes_must_match_durable_snapshot_and_event(tmp_path, monkeypatch, drift):
    from blueprint_pipeline.task_evaluation_scene_retirement_intent_receipt import publish_pending_receipt, publish_terminal_receipt
    from blueprint_pipeline.task_evaluation_scene_retirement_mutation import detach_and_remove
    policy, consent, journal, preserved, allowance, _ = operation(tmp_path, monkeypatch)
    pending = publish_pending_receipt(policy, consent, journal, preserved, allowance)
    original = Path(pending['path']).read_bytes()
    outcome = detach_and_remove(preserved, member_index=0, generation_id=consent['members'][0]['generation_id'],
                                journal=journal)
    recorded = dict(outcome)
    if drift == 'snapshot':
        recorded['removed_file_count'] += 1
    snapshot = journal.retired_snapshot({'schema_version': 'scene_retirement_journal.v1',
        'intent_id': consent['intent_id'], 'members': consent['members'], 'outcomes': [recorded]})
    claimed = dict(outcome)
    if drift == 'count':
        claimed['removed_allocated_bytes'] += 1
    elif drift == 'generation':
        claimed['generation_id'] = '4' * 32
    elif drift == 'event':
        claimed['event_raw_ref'] = journal.initial_ref
    final = {'status': 'retired', 'intent_id': consent['intent_id'], 'token': journal.token,
             'members': [claimed], 'retired_journal_raw_ref': snapshot}
    with pytest.raises(ValueError):
        publish_terminal_receipt(policy, consent, pending, final, allowance)
    assert Path(pending['path']).read_bytes() == original
    assert not list(Path(pending['path']).parent.glob('scene-retired.*.terminal.json'))


@pytest.mark.parametrize('drift', ['intent', 'installed-context', 'symlink-destination'])
def test_changed_authority_or_unsafe_destination_never_publishes(tmp_path, monkeypatch, drift):
    from blueprint_pipeline.task_evaluation_scene_retirement_intent_receipt import publish_pending_receipt
    policy, consent, journal, preserved, allowance, payload = operation(tmp_path, monkeypatch)
    projection = Path(consent['intent_raw_ref']['path']).parent / 'scene-retired.v1.json'
    if drift == 'intent':
        # Actual intake makes the original immutable. Substitute the named
        # entry as a foreign writer, rather than pretending it was writable.
        intent = Path(consent['intent_raw_ref']['path'])
        intent.unlink()
        intent.write_bytes(b'foreign-intent')
    elif drift == 'installed-context':
        policy['reference_context']['roots']['intent_root'] = str(tmp_path.resolve() / 'foreign')
    else:
        projection.symlink_to(payload)
    with pytest.raises((ValueError, OSError)):
        publish_pending_receipt(policy, consent, journal, preserved, allowance)
    assert payload.read_bytes() == b'actual-preserved-proof'
    if drift != 'symlink-destination':
        assert not projection.exists()


def test_oversized_projection_refuses_before_any_publication(tmp_path, monkeypatch):
    from blueprint_pipeline.task_evaluation_scene_retirement_intent_receipt import publish_pending_receipt
    policy, consent, journal, preserved, allowance, payload = operation(tmp_path, monkeypatch)
    preserved['archive']['uri'] += 'x' * 65536
    with pytest.raises(ValueError):
        publish_pending_receipt(policy, consent, journal, preserved, allowance)
    folder = Path(consent['intent_raw_ref']['path']).parent
    assert not list(folder.glob('scene-retired.*'))
    assert payload.read_bytes() == b'actual-preserved-proof'


def test_foreign_pending_bytes_are_never_replaced(tmp_path, monkeypatch):
    from blueprint_pipeline.task_evaluation_scene_retirement_intent_receipt import publish_pending_receipt, publish_terminal_receipt
    policy, consent, journal, preserved, allowance, payload = operation(tmp_path, monkeypatch)
    pending = publish_pending_receipt(policy, consent, journal, preserved, allowance)
    Path(pending['path']).write_bytes(b'foreign-projection')
    with pytest.raises(ValueError):
        publish_terminal_receipt(policy, consent, pending, {'status': 'retired',
            'token': journal.token, 'intent_id': consent['intent_id'], 'members': [],
            'retired_journal_raw_ref': journal.initial_ref}, allowance)
    assert Path(pending['path']).read_bytes() == b'foreign-projection'
    assert payload.read_bytes() == b'actual-preserved-proof'


def test_expired_action_has_no_receipt_or_payload_mutation(tmp_path, monkeypatch):
    from blueprint_pipeline.task_evaluation_scene_retirement_intent_receipt import publish_pending_receipt
    policy, consent, journal, preserved, allowance, payload = operation(tmp_path, monkeypatch)
    allowance.now = lambda: 999
    with pytest.raises(ValueError, match='scene_retirement_consent_expired'):
        publish_pending_receipt(policy, consent, journal, preserved, allowance)
    assert not list(Path(consent['intent_raw_ref']['path']).parent.glob('scene-retired.*'))
    assert payload.read_bytes() == b'actual-preserved-proof'


def partial_operation(tmp_path, monkeypatch):
    from blueprint_pipeline.task_evaluation_scene_retirement_intent_receipt import publish_pending_receipt
    from blueprint_pipeline.task_evaluation_scene_retirement_mutation import detach_and_remove
    policy, consent, journal, preserved, allowance, _ = operation(tmp_path, monkeypatch, member_count=2)
    pending = publish_pending_receipt(policy, consent, journal, preserved, allowance)
    removed = detach_and_remove(preserved, member_index=0,
                                generation_id=consent['members'][0]['generation_id'], journal=journal)
    progress = {'status': 'incomplete', 'token': journal.token, 'intent_id': consent['intent_id'],
                'members': [removed], 'last_event_raw_ref': journal.prior_ref}
    return policy, consent, journal, preserved, allowance, pending, progress


def test_partial_receipt_keeps_unremoved_member_and_replay_is_idempotent(tmp_path, monkeypatch):
    from blueprint_pipeline.task_evaluation_scene_retirement_intent_receipt import publish_progress_receipt
    policy, consent, journal, _, allowance, pending, progress = partial_operation(tmp_path, monkeypatch)
    current = publish_progress_receipt(policy, consent, pending, progress, allowance)
    value = json.loads(Path(current['path']).read_bytes())
    assert value['status'] == 'incomplete' and value['journal_sequence'] == journal.sequence
    assert value['last_event_raw_ref'] == journal.prior_ref
    assert value['members'][0]['outcome'] == 'removed'
    assert value['members'][0]['removed_allocated_bytes'] == progress['members'][0]['removed_allocated_bytes']
    assert value['members'][1]['action'] == 'pending' and 'removed_allocated_bytes' not in value['members'][1]
    assert (Path(consent['members'][1]['canonical_path']) / 'proof.bin').read_bytes() == b'second-member-proof'
    previous = value['prior_receipt_raw_ref']
    assert previous['path'] != current['path'] and raw_ref(Path(previous['path'])) == previous
    assert raw_ref(Path(value['pending_receipt_raw_ref']['path'])) == value['pending_receipt_raw_ref']
    before = sorted(Path(current['path']).parent.glob('scene-retired.*'))
    assert publish_progress_receipt(policy, consent, current, progress, allowance) == current
    assert before == sorted(Path(current['path']).parent.glob('scene-retired.*'))


def test_terminal_advances_partial_receipt_with_resolvable_history(tmp_path, monkeypatch):
    from blueprint_pipeline.task_evaluation_scene_retirement_intent_receipt import publish_progress_receipt, publish_terminal_receipt
    from blueprint_pipeline.task_evaluation_scene_retirement_mutation import detach_and_remove
    policy, consent, journal, preserved, allowance, pending, progress = partial_operation(tmp_path, monkeypatch)
    current = publish_progress_receipt(policy, consent, pending, progress, allowance)
    original_progress = Path(current['path']).read_bytes()
    second = detach_and_remove(preserved, member_index=1,
                               generation_id=consent['members'][1]['generation_id'], journal=journal)
    outcomes = progress['members'] + [second]
    snapshot = journal.retired_snapshot({'schema_version': 'scene_retirement_journal.v1',
        'intent_id': consent['intent_id'], 'members': consent['members'], 'outcomes': outcomes})
    final = {'status': 'retired', 'intent_id': consent['intent_id'], 'token': journal.token,
             'members': outcomes, 'retired_journal_raw_ref': snapshot}
    terminal = publish_terminal_receipt(policy, consent, current, final, allowance)
    value = json.loads(Path(terminal['path']).read_bytes())
    assert all(row['outcome'] == 'removed' for row in value['members'])
    assert value['journal_sequence'] == journal.sequence and value['last_event_raw_ref'] == journal.prior_ref
    for key in ('pending_receipt_raw_ref', 'prior_receipt_raw_ref'):
        assert raw_ref(Path(value[key]['path'])) == value[key]
    assert Path(value['prior_receipt_raw_ref']['path']).read_bytes() == original_progress


@pytest.mark.parametrize('drift', ['count', 'token', 'event', 'rollback', 'stale-current'])
def test_progress_refuses_drift_without_losing_recorded_outcomes(tmp_path, monkeypatch, drift):
    from blueprint_pipeline.task_evaluation_scene_retirement_intent_receipt import publish_progress_receipt
    policy, consent, _, _, allowance, pending, progress = partial_operation(tmp_path, monkeypatch)
    current = publish_progress_receipt(policy, consent, pending, progress, allowance)
    before = Path(current['path']).read_bytes()
    claimed = dict(progress, members=[dict(progress['members'][0])])
    if drift == 'count':
        claimed['members'][0]['removed_allocated_bytes'] += 1
    elif drift == 'token':
        claimed['token'] = '9' * 32
    elif drift == 'event':
        claimed['last_event_raw_ref'] = json.loads(before)['journal_initial_raw_ref']
    elif drift == 'rollback':
        claimed['members'] = []
    with pytest.raises((ValueError, OSError)):
        publish_progress_receipt(policy, consent, pending if drift == 'stale-current' else current, claimed, allowance)
    assert Path(current['path']).read_bytes() == before


def test_progress_recovers_after_immutable_history_before_projection_cas(tmp_path, monkeypatch):
    from blueprint_pipeline import task_evaluation_scene_retirement_intent_receipt as module
    policy, consent, _, _, allowance, pending, progress = partial_operation(tmp_path, monkeypatch)
    before = Path(pending['path']).read_bytes()
    publish = module._publish

    def interrupted(directory, name, raw, allowance, *, prior=None):
        if name == module.NAME:
            raise OSError('fixture crash before projection CAS')
        return publish(directory, name, raw, allowance, prior=prior)

    monkeypatch.setattr(module, '_publish', interrupted)
    with pytest.raises(OSError):
        module.publish_progress_receipt(policy, consent, pending, progress, allowance)
    assert Path(pending['path']).read_bytes() == before
    history = list(Path(pending['path']).parent.glob('scene-retired.*.incomplete.*.json'))
    assert len(history) == 1
    original = history[0].read_bytes()
    monkeypatch.setattr(module, '_publish', publish)
    current = module.publish_progress_receipt(policy, consent, pending, progress, allowance)
    assert json.loads(Path(current['path']).read_bytes())['status'] == 'incomplete'
    assert history[0].read_bytes() == original
    assert len(list(Path(pending['path']).parent.glob('scene-retired.*.incomplete.*.json'))) == 1


def restore_operation(tmp_path, monkeypatch):
    from blueprint_pipeline.task_evaluation_scene_retirement_intent_receipt import publish_pending_receipt, publish_terminal_receipt
    from blueprint_pipeline.task_evaluation_scene_retirement_mutation import detach_and_remove
    from blueprint_pipeline.task_evaluation_scene_retirement_journal import SceneJournal
    policy, retire_consent, journal, preserved, allowance, _, transport = operation(
        tmp_path, monkeypatch, member_count=2, with_transport=True)
    pending = publish_pending_receipt(policy, retire_consent, journal, preserved, allowance)
    outcomes = [detach_and_remove(preserved, member_index=index, generation_id=row['generation_id'], journal=journal)
                for index, row in enumerate(retire_consent['members'])]
    snapshot = journal.retired_snapshot({'schema_version': 'scene_retirement_journal.v1',
        'intent_id': retire_consent['intent_id'], 'members': retire_consent['members'], 'outcomes': outcomes})
    final = {'status': 'retired', 'intent_id': retire_consent['intent_id'], 'token': journal.token,
             'members': outcomes, 'retired_journal_raw_ref': snapshot}
    current = publish_terminal_receipt(policy, retire_consent, pending, final, allowance)
    consent = dict(retire_consent, plan_raw_ref=None, retired_journal_raw_ref=snapshot, action='restore')
    consent_path = tmp_path.resolve() / 'restore-consent.json'
    consent_path.write_text(json.dumps(consent))
    consent_path.chmod(0o600)
    restore = SceneJournal.create(Path(policy['journal_store']), token='8' * 32, allowance=allowance,
        initial={'schema_version': 'scene_restore_journal.v1', 'status': 'restoring',
                 'intent_id': consent['intent_id'], 'intent_raw_ref': consent['intent_raw_ref'],
                 'members': consent['members'], 'original_retirement_token': journal.token,
                 'retired_journal_raw_ref': snapshot, 'consent_raw_ref': raw_ref(consent_path)})
    progress = {'status': 'restoring', 'intent_id': consent['intent_id'], 'token': restore.token,
                'original_retirement_token': journal.token, 'restore_journal_initial_raw_ref': restore.initial_ref,
                'members': [], 'last_event_raw_ref': restore.prior_ref}
    transport.members = []  # Preserve/readback is complete; the roots are now absent.
    return policy, consent, restore, preserved, allowance, current, progress, transport


def restore_members(policy, consent, journal, preserved, transport):
    from blueprint_pipeline.task_evaluation_scene_retirement_restore import restore_preserved_members
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest
    outcomes = restore_preserved_members(preserved, transport=transport, journal=journal)
    for selected, outcome in zip(consent['members'], outcomes):
        # Hermetic enrollment metadata is bound to the actual restored inode;
        # this test does not claim production cohort/role admission.
        value = {'schema_version': 'scene_member_generation.v1', 'state': 'restored-active',
                 'canonical_path': selected['canonical_path'], 'generation_id': selected['generation_id'],
                 'owner_intent_id': selected['owner_intent_id'], 'owner_raw_ref': selected['owner_raw_ref'],
                 'retirement_token': '1' * 32}
        value.update(zip(('dev', 'ino', 'mode'), outcome['restore_identity']))
        value['state_digest'] = canonical_digest(value, digest_field='state_digest')
        key = hashlib.sha256(selected['canonical_path'].encode()).hexdigest() + '.json'
        target = Path(policy['generation_store']) / key
        target.write_text(json.dumps(value))
        target.chmod(0o600)
    return outcomes


def test_restore_receipt_starts_before_payload_and_finishes_from_real_restore(tmp_path, monkeypatch):
    from blueprint_pipeline.task_evaluation_scene_retirement_intent_receipt import publish_progress_receipt
    policy, consent, journal, preserved, allowance, retired, progress, transport = restore_operation(tmp_path, monkeypatch)
    current = publish_progress_receipt(policy, consent, retired, progress, allowance)
    value = json.loads(Path(current['path']).read_bytes())
    assert value['status'] == 'restoring' and value['restore_token'] == journal.token
    assert value['retiring_token'] == progress['original_retirement_token']
    assert value['restore_journal_initial_raw_ref'] == journal.initial_ref
    assert all(not Path(row['canonical_path']).exists() for row in consent['members'])
    assert 'consent_raw_ref' not in value
    assert raw_ref(Path(value['prior_receipt_raw_ref']['path'])) == value['prior_receipt_raw_ref']
    outcomes = restore_members(policy, consent, journal, preserved, transport)
    complete = dict(progress, status='restored', members=outcomes, last_event_raw_ref=journal.prior_ref)
    result = publish_progress_receipt(policy, consent, current, complete, allowance)
    value = json.loads(Path(result['path']).read_bytes())
    assert value['status'] == 'restored' and value['journal_sequence'] == journal.sequence
    assert all(row['outcome'] == 'restored' and row['action'] == 'restored' for row in value['members'])
    assert [row['restore_identity'] for row in value['members']] == [row['restore_identity'] for row in outcomes]
    for row in value['members']:
        assert raw_ref(Path(row['restore_event_raw_ref']['path'])) == row['restore_event_raw_ref']
    assert (Path(consent['members'][0]['canonical_path']) / 'proof.bin').read_bytes() == b'actual-preserved-proof'
    assert (Path(consent['members'][1]['canonical_path']) / 'proof.bin').read_bytes() == b'second-member-proof'
    assert publish_progress_receipt(policy, consent, result, complete, allowance) == result


@pytest.mark.parametrize('drift', ['token', 'snapshot', 'consent', 'outcome', 'generation'])
def test_restore_projection_refuses_wrong_proof_or_unfinished_generation(tmp_path, monkeypatch, drift):
    from blueprint_pipeline.task_evaluation_scene_retirement_intent_receipt import publish_progress_receipt
    policy, consent, journal, preserved, allowance, retired, progress, transport = restore_operation(tmp_path, monkeypatch)
    current = publish_progress_receipt(policy, consent, retired, progress, allowance)
    original = Path(current['path']).read_bytes()
    outcomes = restore_members(policy, consent, journal, preserved, transport)
    claimed = dict(progress, status='restored', members=[dict(row) for row in outcomes], last_event_raw_ref=journal.prior_ref)
    if drift == 'token':
        claimed['token'] = '9' * 32
    elif drift == 'snapshot':
        consent = dict(consent, retired_journal_raw_ref=journal.initial_ref)
    elif drift == 'consent':
        (tmp_path.resolve() / 'restore-consent.json').write_text('{"action":"foreign"}')
    elif drift == 'outcome':
        claimed['members'][0]['restore_identity'][1] += 1
    else:
        from blueprint_pipeline.decision_evidence_contracts import canonical_digest
        key = hashlib.sha256(consent['members'][0]['canonical_path'].encode()).hexdigest() + '.json'
        path = Path(policy['generation_store']) / key
        state = json.loads(path.read_bytes())
        state['state'] = 'restoring'
        state['state_digest'] = canonical_digest(state, digest_field='state_digest')
        path.write_text(json.dumps(state))
    with pytest.raises((ValueError, OSError)):
        publish_progress_receipt(policy, consent, current, claimed, allowance)
    assert Path(current['path']).read_bytes() == original


def test_restore_start_replays_history_after_crash_before_current_cas(tmp_path, monkeypatch):
    from blueprint_pipeline import task_evaluation_scene_retirement_intent_receipt as module
    policy, consent, _, _, allowance, current, progress, _ = restore_operation(tmp_path, monkeypatch)
    original = Path(current['path']).read_bytes()
    publish = module._publish

    def interrupted(directory, name, raw, allowance, *, prior=None):
        if name == module.NAME:
            raise OSError('fixture interrupted restore admission projection')
        return publish(directory, name, raw, allowance, prior=prior)

    monkeypatch.setattr(module, '_publish', interrupted)
    with pytest.raises(OSError):
        module.publish_progress_receipt(policy, consent, current, progress, allowance)
    assert Path(current['path']).read_bytes() == original
    monkeypatch.setattr(module, '_publish', publish)
    result = module.publish_progress_receipt(policy, consent, current, progress, allowance)
    assert json.loads(Path(result['path']).read_bytes())['status'] == 'restoring'
    assert all(not Path(row['canonical_path']).exists() for row in consent['members'])
