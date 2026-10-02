"""ADP-009D/day28: original capture birth survives real local restore receipt.

Hermetic capture delivery facts and actual archive/mutation/replay. This does
not supply completed CAD lineage or installed all-ten retirement acceptance.
"""
# Covers (for impacted-test selection):
#   src/blueprint_pipeline/task_evaluation_scene_retirement_intent_receipt.py
import json
from pathlib import Path

import pytest

from tests.test_capture_generation_birth import _fixture


def test_capture_birth_schema_and_selectors_survive_restore_and_receipt_replay(tmp_path, monkeypatch):
    from blueprint_pipeline import task_evaluation_scene_retirement as engine
    from blueprint_pipeline.task_evaluation_scene_retirement_generations import birth_capture_member
    from blueprint_pipeline.task_evaluation_scene_retirement_intent_receipt import _restored_generation
    from blueprint_pipeline.task_evaluation_scene_retirement_journal import SceneJournal
    from blueprint_pipeline.task_evaluation_scene_retirement_mutation import detach_and_remove
    from blueprint_pipeline.task_evaluation_scene_retirement_preservation import ActionAllowance, preserve_members
    from blueprint_pipeline.task_evaluation_scene_retirement_restore import restore_preserved_members
    from tests.test_scene_retirement_preservation import MemoryTransport
    access, policy, target, owner, selector, raw = _fixture(tmp_path, monkeypatch)
    born = birth_capture_member(target, observation=owner, membership_selector=selector, membership_raw=raw)
    payload = target / 'owned-development-only.bin'
    payload.write_bytes(b'actual-capture-workspace-local-roundtrip')
    selected = {key: born[key] for key in ('canonical_path', 'generation_id', 'capture_owner_user_id',
                                         'owner_observation_raw_ref', 'birth_delivery_raw_ref')}
    selected.update(request_id=owner['request_id'], source_membership_raw_ref=json.loads(
        Path(born['birth_delivery_raw_ref']['path']).read_bytes())['source_membership_raw_ref'])
    allowance = ActionAllowance(expires_at=200, now=lambda: 100, monotonic=lambda: 0)
    transport = MemoryTransport([target])
    preserved = preserve_members([target], transport=transport, allowance=allowance, token='1' * 32)
    store = Path(policy['journal_store'])
    store.mkdir(mode=0o700)
    journal = SceneJournal.create(store, token='1' * 32, allowance=allowance,
        initial={'status': 'pending', 'preservation': preserved})
    with access.exclusive_scene_access():
        retiring = engine._transition(policy, born, allowance=allowance, state='retiring',
            token=journal.token, journal_ref=journal.initial_ref)
        detach_and_remove(preserved, member_index=0, generation_id=born['generation_id'], journal=journal)
        retired = engine._transition(policy, retiring, allowance=allowance, state='retired',
            token=journal.token, journal_ref=journal.initial_ref)
        transport.members = []
        outcomes = restore_preserved_members(preserved, transport=transport, journal=journal)
        restored = engine._transition(policy, retired, allowance=allowance, state='restored-active',
            token=journal.token, journal_ref=journal.initial_ref, identity=outcomes[0]['restore_identity'])
        assert restored['schema_version'] == 'scene_capture_generation.v1'
        _restored_generation(policy, selected, outcomes[0], journal.token, allowance)
        # Re-selection after an interrupted receipt keeps the same original birth.
        _restored_generation(policy, selected, outcomes[0], journal.token, allowance)
        for change in ({'capture_owner_user_id': 'unselected-owner'},
                       {'owner_observation_raw_ref': selected['birth_delivery_raw_ref']},
                       {'birth_delivery_raw_ref': selected['owner_observation_raw_ref']}):
            with pytest.raises(ValueError, match='scene_retirement_generation_unavailable'):
                _restored_generation(policy, dict(selected, **change), outcomes[0], journal.token, allowance)
    assert payload.read_bytes() == b'actual-capture-workspace-local-roundtrip'
