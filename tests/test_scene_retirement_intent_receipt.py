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


def operation(tmp_path, monkeypatch):
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest
    from blueprint_pipeline.task_evaluation_scene_retirement_journal import SceneJournal
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
    policy['reference_context'] = {'roots': {'intent_root': str(intake)}}
    policy['policy_digest'] = canonical_digest(policy, digest_field='policy_digest')
    Path(os.environ['BLUEPRINT_SCENE_RETIREMENT_POLICY_FILE']).write_text(json.dumps(policy))
    payload = member / 'proof.bin'
    payload.write_bytes(b'actual-preserved-proof')
    allowance = ActionAllowance(expires_at=999, now=lambda: 200, monotonic=lambda: 0)
    preserved = preserve_members([member], transport=MemoryTransport([member]),
                                 allowance=allowance, token='1' * 32)
    plan = base / 'plan.json'
    plan.write_text('{"action":"KEEP"}')
    plan.chmod(0o600)
    members = [{'canonical_path': str(member), 'class': 'host',
                'owner_intent_id': issued['intent_id'], 'owner_raw_ref': raw_ref(intent_path),
                'generation_id': '2' * 32, 'inventory_sha256': 'sha256:' + '3' * 64}]
    consent = {'intent_id': issued['intent_id'], 'intent_raw_ref': raw_ref(intent_path),
               'plan_raw_ref': raw_ref(plan), 'members': members}
    store = Path(policy['journal_store'])
    store.mkdir(mode=0o700)
    journal = SceneJournal.create(store, token='1' * 32, allowance=allowance,
        initial={'schema_version': 'scene_retirement_journal.v1', 'status': 'pending',
                 'intent_id': consent['intent_id'], 'intent_raw_ref': consent['intent_raw_ref'],
                 'plan_raw_ref': consent['plan_raw_ref'], 'members': members, 'preserved': preserved})
    return policy, consent, journal, preserved, allowance, payload


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
    assert value['members'][0]['inventory_sha256'] == 'sha256:' + '3' * 64
    assert value['members'][0]['action'] == 'pending'
    assert value['receipt_digest'] == canonical_digest(value, digest_field='receipt_digest')
    assert stat.S_IMODE(path.stat().st_mode) == 0o644
    assert payload.read_bytes() == b'actual-preserved-proof'
    assert any(p.read_bytes() == path.read_bytes() for p in path.parent.glob('scene-retired.*.pending.json'))
    assert 'consent_raw_ref' not in value and 'principal_id' not in value


def test_terminal_updates_only_exact_pending_and_preserves_immutable_history(tmp_path, monkeypatch):
    from blueprint_pipeline.task_evaluation_scene_retirement_intent_receipt import publish_pending_receipt, publish_terminal_receipt
    policy, consent, journal, preserved, allowance, _ = operation(tmp_path, monkeypatch)
    pending = publish_pending_receipt(policy, consent, journal, preserved, allowance)
    original = Path(pending['path']).read_bytes()
    snapshot = journal.retired_snapshot({'schema_version': 'scene_retirement_journal.v1',
        'intent_id': consent['intent_id'], 'members': consent['members']})
    final = {'status': 'retired', 'intent_id': consent['intent_id'], 'token': journal.token,
             'members': [{'canonical_path': consent['members'][0]['canonical_path'], 'outcome': 'removed'}],
             'retired_journal_raw_ref': snapshot}
    terminal = publish_terminal_receipt(policy, consent, pending, final, allowance)
    value = json.loads(Path(terminal['path']).read_bytes())
    assert value['status'] == 'retired' and value['pending_receipt_raw_ref'] == pending
    assert value['retired_journal_raw_ref'] == snapshot and terminal == raw_ref(Path(terminal['path']))
    assert any(p.read_bytes() == original for p in Path(pending['path']).parent.glob('scene-retired.*.pending.json'))
    assert any(p.read_bytes() == Path(terminal['path']).read_bytes()
               for p in Path(pending['path']).parent.glob('scene-retired.*.terminal.json'))


@pytest.mark.parametrize('drift', ['intent', 'installed-context', 'symlink-destination'])
def test_changed_authority_or_unsafe_destination_never_publishes(tmp_path, monkeypatch, drift):
    from blueprint_pipeline.task_evaluation_scene_retirement_intent_receipt import publish_pending_receipt
    policy, consent, journal, preserved, allowance, payload = operation(tmp_path, monkeypatch)
    projection = Path(consent['intent_raw_ref']['path']).parent / 'scene-retired.v1.json'
    if drift == 'intent':
        Path(consent['intent_raw_ref']['path']).write_bytes(b'foreign-intent')
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
