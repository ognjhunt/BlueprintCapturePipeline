"""A root-proven active release is stable while remote main advances."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from blueprint_pipeline import active_deployed_release_admission as release
from blueprint_pipeline import paid_resource_allocator as allocator

COMMIT = 'a' * 40


def _write(path: Path, value: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return 'sha256:' + hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def deployed(tmp_path: Path, monkeypatch):
    root = tmp_path.resolve()
    releases, state, receipts = root/'releases', root/'state', root/'receipts'
    for path in (releases, state, receipts):
        path.mkdir()
    checkout = releases/COMMIT
    checkout.mkdir()
    active = root/'active'
    active.symlink_to(checkout, target_is_directory=True)
    source = root/'source'
    for name, value in [('ACTIVE_LINK', active), ('RELEASES_ROOT', releases), ('STATE_ROOT', state),
                        ('RECEIPTS_ROOT', receipts), ('SOURCE_CHECKOUT', source), ('TRUSTED_UID', os.getuid())]:
        monkeypatch.setattr(release, name, value)
    provenance = {'schema_version': 'blueprint.deploy_release_provenance.v1', 'status': 'iteration',
                  'git_sha': COMMIT, 'promotion_eligible': False,
                  'claim_boundary': {'canonical_full_lane_verified': False, 'promotion_eligible': False,
                                     'evidence_grade': 'development_only'}}
    provenance_path = state/COMMIT/'deploy-release-provenance.json'
    digest = _write(provenance_path, provenance)
    receipt = {'schema_version': 'control_plane_commit_deploy_receipt.v1', 'status': 'deployed',
               'source_commit': COMMIT, 'release_path': str(checkout),
               'intake_runtime': {'source_commit': COMMIT, 'commit_proven': True},
               'surfaces': [{'name': 'active_release', 'head': COMMIT, 'path': str(checkout)},
                            {'name': 'source_checkout', 'head': COMMIT, 'path': str(source)}],
               'release_provenance': {'git_sha': COMMIT, 'path': str(provenance_path), 'sha256': digest,
                                      'size_bytes': provenance_path.stat().st_size,
                                      'provenance_status': 'iteration', 'promotion_eligible': False,
                                      'canonical_full_lane_verified': False, 'run_id': None, 'run_url': None}}
    receipt_path = receipts/'iteration.json'
    _write(receipt_path, receipt)
    monkeypatch.setattr(allocator, 'ROOT', checkout)
    monkeypatch.setattr(allocator, 'CONTROL_PLANE_RELEASE_STATE_ROOT', state)
    monkeypatch.setattr(allocator, '_current_checkout_source_state', lambda: (COMMIT, True, True))
    def forbidden_remote_probe(*args):
        raise AssertionError('an active deployment must not depend on moving/unfetched remote main')
    for name in ['_current_origin_main_commit', '_current_remote_main_commit', '_commit_is_merged_into']:
        monkeypatch.setattr(allocator, name, forbidden_remote_probe)
    return locals()


def test_current_iteration_ignores_moving_unfetched_main_without_claim_upgrade(deployed):
    assert allocator._source_checkout_blockers(COMMIT) == ([], COMMIT)
    result = release.inspect_active_deployed_release(deployed['checkout'], COMMIT)
    assert result['evidence_grade'] == 'development_only'
    assert result['promotion_eligible'] is False
    assert allocator.release_promotion_eligible(COMMIT) is False


def test_promoted_active_release_keeps_full_lane_authority(deployed):
    value = deployed['provenance']
    value.update(status='verified', promotion_eligible=True, workflow_name='Full Test Lane',
                 workflow_path='.github/workflows/full-test-lane.yml', job_name='Full pytest lane on CPU runner',
                 run_id=42, collection={'test_count': 100, 'skipped_count': 0},
                 claim_boundary={'canonical_full_lane_verified': True})
    digest = _write(deployed['provenance_path'], value)
    receipt = deployed['receipt']
    receipt['release_provenance'].update(sha256=digest, size_bytes=deployed['provenance_path'].stat().st_size,
                                         provenance_status='verified', promotion_eligible=True,
                                         canonical_full_lane_verified=True, run_id=42)
    _write(deployed['receipt_path'], receipt)
    assert allocator._source_checkout_blockers(COMMIT) == ([], COMMIT)
    assert release.inspect_active_deployed_release(deployed['checkout'], COMMIT)['promotion_eligible'] is True


@pytest.mark.parametrize('defect', ['dirty', 'expected_commit', 'inactive', 'missing_receipt', 'changed_provenance',
                                  'missing_provenance', 'writable_receipt', 'untrusted_owner', 'intake_unproven',
                                  'source_mismatch', 'iteration_claim_upgrade', 'failed_deploy'])
def test_deployed_release_failures_never_fall_back_to_remote_main(deployed, monkeypatch, defect):
    if defect == 'dirty':
        monkeypatch.setattr(allocator, '_current_checkout_source_state', lambda: (COMMIT, False, True))
    elif defect == 'inactive':
        deployed['active'].unlink()
        deployed['active'].symlink_to(deployed['releases'] / ('b' * 40))
    elif defect == 'missing_receipt':
        deployed['receipt_path'].unlink()
    elif defect == 'changed_provenance':
        deployed['provenance_path'].write_text('{}')
    elif defect == 'missing_provenance':
        deployed['provenance_path'].unlink()
    elif defect == 'writable_receipt':
        deployed['receipt_path'].chmod(0o666)
    elif defect == 'untrusted_owner':
        monkeypatch.setattr(release, 'TRUSTED_UID', os.getuid() + 1)
    elif defect in ['intake_unproven', 'source_mismatch', 'iteration_claim_upgrade', 'failed_deploy']:
        receipt = deployed['receipt']
        if defect == 'intake_unproven':
            receipt['intake_runtime']['commit_proven'] = False
        elif defect == 'source_mismatch':
            receipt['surfaces'][1]['head'] = 'b' * 40
        elif defect == 'iteration_claim_upgrade':
            receipt['release_provenance']['promotion_eligible'] = True
        else:
            receipt['status'] = 'failed'
        _write(deployed['receipt_path'], receipt)
    blockers, observed = allocator._source_checkout_blockers('b' * 40 if defect == 'expected_commit' else COMMIT)
    assert blockers and observed == COMMIT


def test_arbitrary_checkout_cannot_borrow_active_deploy_receipt(deployed, monkeypatch):
    arbitrary = deployed['root']/'arbitrary'
    arbitrary.mkdir()
    monkeypatch.setattr(allocator, 'ROOT', arbitrary)
    monkeypatch.setattr(allocator, '_current_origin_main_commit', lambda: 'b' * 40)
    monkeypatch.setattr(allocator, '_current_remote_main_commit', lambda: 'b' * 40)
    monkeypatch.setattr(allocator, '_commit_is_merged_into', lambda *_: False)
    blockers, _ = allocator._source_checkout_blockers(COMMIT)
    assert 'gpu_canary_checkout_not_remote_main' in blockers
    assert release.inspect_active_deployed_release(arbitrary, COMMIT) is None
