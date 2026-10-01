"""ADP-009D/day28: restore must consult existing current native reader admission.

Portable public replay fixture isolates producer completion and current native
clearance. Refusal models prove control flow only, never Linux process absence.
"""
# Covers (for impacted-test selection):
#   src/blueprint_pipeline/task_evaluation_scene_retirement.py

import json
from pathlib import Path

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from tests.test_scene_retirement_public_resume import pending_action
from tests.test_scene_retirement_public_restore_resume import interrupted_public_restore


@pytest.mark.parametrize('phase', ['fresh', 'resumed', 'before_reopening'])
def test_current_native_unknown_readers_cannot_admit_or_reopen_restore(tmp_path, monkeypatch, phase):
    from blueprint_pipeline import task_evaluation_scene_retirement as engine
    from blueprint_pipeline import task_evaluation_scene_retirement_access as access
    from blueprint_pipeline import task_evaluation_scene_retirement_supervisor as native
    current_reader_gate = engine._current_readers
    if phase == 'resumed':
        engine, policy, scope, consent, retired, _, transport = interrupted_public_restore(
            tmp_path, monkeypatch, 'restore_directory_created')
    else:
        engine, policy, scope, retire_consent, _, transport = pending_action(tmp_path, monkeypatch)
        retired = engine.retire_scene(scope['plan_raw_ref']['path'], retire_consent, transport=transport,
                                      now=lambda: 201, monotonic=lambda: 1)
        assert retired['status'] == 'retired', retired
        value = dict(scope, consent_id='b'*32, action='restore', plan_raw_ref=None,
                     retired_journal_raw_ref=retired['retired_journal_raw_ref'])
        value['consent_digest'] = canonical_digest(value, digest_field='consent_digest')
        consent = tmp_path / 'restore-consent.json'
        consent.write_text(json.dumps(value))
        consent.chmod(0o600)
    monkeypatch.setattr(engine, '_current_readers', current_reader_gate)
    observed = []
    def unknown(policy, allowance):
        allowance.tick()
        observed.append(allowance)
        # Admit the early invocation only in the model that introduces an
        # unknown reader after actual restoration, immediately before reopen.
        if phase == 'before_reopening' and len(observed) == 1:
            return
        raise access.SceneRetirementAccessError('scene_retirement_reader_closure_unproven')
    monkeypatch.setattr(native, 'require_current_reader_closure', unknown)
    generations = Path(policy['generation_store'])
    before = {p.name: p.read_bytes() for p in generations.glob('*.json')}
    paths = [Path(row['canonical_path']) for row in scope['members']]
    existed = [p.exists() for p in paths]
    outcome = engine.restore_scene(retired['retired_journal_raw_ref']['path'], consent,
                                  transport=transport, now=lambda: 203, monotonic=lambda: 3)
    assert outcome['status'] != 'restored' and outcome['reason'] == 'scene_retirement_reader_closure_unproven'
    assert observed and all(value is observed[0] for value in observed)
    if phase != 'before_reopening':
        assert {p.name: p.read_bytes() for p in generations.glob('*.json')} == before
        assert [p.exists() for p in paths] == existed
    else:
        current = [json.loads(p.read_bytes()) for p in generations.glob('*.json') if p.name.count('.') == 1]
        assert current and all(value['state'] != 'restored-active' for value in current)
