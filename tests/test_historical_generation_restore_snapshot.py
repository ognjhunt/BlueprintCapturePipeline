"""ADP-009D/day28: exact restored observations; no execution authority."""
# Covers (for impacted-test selection):
#   src/blueprint_pipeline/control_plane_lane_historical_restore_snapshot.py
import copy
import stat

import pytest

from tests.test_historical_generation_authority import historical_installation  # noqa: F401
from tests.test_registered_experiment_issuer import installation  # noqa: F401
from tests.test_owner_target_version_publication import root_metadata  # noqa: F401

# ruff: noqa: F811


def observations(installed):
    from blueprint_pipeline.control_plane_lane_historical_generation import inventory_historical_generation
    original = inventory_historical_generation(installed[1], allowed_roots=(installed[1].parent,))
    # Metadata projections only. Actual kernel transitions are proved by the
    # connected disposable Linux worker, never by this validator fixture.
    private = copy.deepcopy(original)
    version = private['members'][0]['version']
    version[2:5] = [stat.S_IFDIR | 0o700, 0, 0]
    private['target_version'] = version.copy()
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest
    private['generation_digest'] = canonical_digest(private, digest_field='generation_digest')
    reopened = version.copy()
    reopened[2:5] = original['members'][0]['version'][2:5]
    reopened[8] += 1
    return original, private, reopened


def test_projects_only_exact_durable_root_access_transition(historical_installation):
    from blueprint_pipeline.control_plane_lane_historical_restore_snapshot import after_reopen
    original, private, reopened = observations(historical_installation)
    saved = copy.deepcopy(private)
    expected = after_reopen(original, private, reopened)
    assert private == saved
    assert expected['members'][0]['version'] == reopened
    assert expected['members'][1:] == private['members'][1:]
    assert expected['execution_authorized'] is False
    assert expected['target_version'] == reopened


def test_private_snapshot_validation_does_not_invent_an_access_transition(historical_installation):
    from blueprint_pipeline.control_plane_lane_historical_restore_snapshot import validate_private
    original, private, _ = observations(historical_installation)
    saved = copy.deepcopy(private)
    assert validate_private(original, private) is None
    assert private == saved and private['target_version'][3:5] == [0, 0]


@pytest.mark.parametrize('change', ['root_inode', 'root_owner', 'file_owner', 'file_digest'])
def test_private_snapshot_cannot_adopt_changed_generation_before_access(historical_installation, change):
    from blueprint_pipeline.control_plane_lane_historical_restore_snapshot import validate_private
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest
    original, private, _ = observations(historical_installation)
    if change == 'root_inode':
        private['members'][0]['version'][1] += 1
        private['target_version'] = private['members'][0]['version'].copy()
    elif change == 'root_owner':
        private['members'][0]['version'][3] += 1
        private['target_version'] = private['members'][0]['version'].copy()
    elif change == 'file_owner':
        private['members'][1]['version'][3] += 1
    else:
        private['members'][1]['sha256'] = 'sha256:' + 'f' * 64
    private['generation_digest'] = canonical_digest(private, digest_field='generation_digest')
    with pytest.raises(ValueError, match='restore_snapshot_changed'):
        validate_private(original, private)


def test_freshly_bound_parent_observation_is_preserved(historical_installation):
    from blueprint_pipeline.control_plane_lane_historical_restore_snapshot import after_reopen
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest
    original, private, reopened = observations(historical_installation)
    # A distinct restore approval binds this current parent separately. Never
    # derive the current parent version from the old offload authority.
    private['root_version'][8] += 1
    private['root_identity']['ctime_ns'] = private['root_version'][8]
    private['generation_digest'] = canonical_digest(private, digest_field='generation_digest')
    expected = after_reopen(original, private, reopened)
    assert expected['root_version'] == private['root_version']
    assert expected['root_identity'] == private['root_identity']


@pytest.mark.parametrize('change', ['bytes', 'member_missing', 'owner', 'root_inode', 'root_size', 'root_ctime'])
def test_changed_snapshot_or_unplanned_root_transition_refuses(historical_installation, change):
    from blueprint_pipeline.control_plane_lane_historical_restore_snapshot import after_reopen
    original, private, reopened = observations(historical_installation)
    if change == 'bytes':
        private['members'][1]['sha256'] = 'sha256:' + '0' * 64
    elif change == 'member_missing':
        private['members'].pop()
    elif change == 'owner':
        private['members'][1]['version'][3] += 1
    elif change == 'root_inode':
        reopened[1] += 1
    elif change == 'root_size':
        reopened[6] += 1
    else:
        reopened[8] = private['members'][0]['version'][8] - 1
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest
    private['generation_digest'] = canonical_digest(private, digest_field='generation_digest')
    with pytest.raises(ValueError, match='restore_snapshot_changed'):
        after_reopen(original, private, reopened)
