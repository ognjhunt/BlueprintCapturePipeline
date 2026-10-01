"""ADP-009D/day28: unfinished restore needs separate current DELETE rights.

These protected-policy tests prove authorization predicates only. They supply
no original restore, creation receipt, reader clearance or native execution.
"""
# Covers: src/blueprint_pipeline/control_plane_lane_historical_restore_reconciliation_authority.py
import json

import pytest

from tests.test_registered_experiment_issuer import installation  # noqa: F401
from tests.test_owner_target_version_publication import root_metadata  # noqa: F401
from tests.test_historical_generation_authority import historical_installation  # noqa: F401

# ruff: noqa: F811


def authorize(installed, *, principal='operator', owner='owner', expiry=1090):
    from blueprint_pipeline import control_plane_lane_historical_authority as authority
    from blueprint_pipeline.control_plane_lane_historical_restore_reconciliation_authority import _current_delete
    with authority._session(installed[0], authority._Operation(1030, lambda: 0)) as (files, config, _):
        return _current_delete(files, config, installed[0], principal, owner, 1030, expiry, 1100)


def test_current_restore_handling_never_implies_partial_file_delete(historical_installation):
    policy = historical_installation[2]
    value = json.loads(policy.read_bytes())
    value['principals'][0]['allowed_actions'].remove('delete')
    policy.write_text(json.dumps(value))
    before = {path.name: path.read_bytes() for path in historical_installation[3].iterdir()}
    with pytest.raises(ValueError):
        authorize(historical_installation)
    assert {path.name: path.read_bytes() for path in historical_installation[3].iterdir()} == before


@pytest.mark.parametrize('change', [dict(principal='unknown'), dict(owner='other'), dict(expiry=1030),
    dict(expiry=1101), dict(expiry=True), dict(expiry=float('nan'))])
def test_exact_owner_and_original_expiry_cannot_be_renewed(historical_installation, change):
    with pytest.raises(ValueError):
        authorize(historical_installation, **change)


def test_current_explicit_delete_returns_only_config_and_policy_selectors(historical_installation):
    from tests.test_historical_generation_authority import selector
    before = (historical_installation[1] / 'one.log').read_bytes()
    assert authorize(historical_installation) == (
        selector(historical_installation[0].read_bytes()), selector(historical_installation[2].read_bytes()))
    assert (historical_installation[1] / 'one.log').read_bytes() == before


def test_missing_reconciliation_approval_survives_authority_session(historical_installation):
    from blueprint_pipeline import control_plane_lane_historical_authority as authority
    from blueprint_pipeline.control_plane_lane_historical_restore_reconciliation_authority import select_reconciliation
    before = {path.name: path.read_bytes() for path in historical_installation[3].iterdir()}
    with pytest.raises(ValueError, match='historical_generation_restore_reconciliation_approval_missing'):
        with authority._session(historical_installation[0], authority._Operation(1030, lambda: 0)) as (files, config, store):
            # The missing protected decision must refuse before any selector,
            # packet, native fact or effect can be accepted.
            select_reconciliation(files, config, store, historical_installation[0],
                None, [], 'a' * 32, None, 1030)
    assert {path.name: path.read_bytes() for path in historical_installation[3].iterdir()} == before


def test_unreadable_reconciliation_stays_authority_io_refusal(historical_installation, monkeypatch):
    from blueprint_pipeline import control_plane_lane_historical_authority as authority
    from blueprint_pipeline.control_plane_lane_historical_restore_reconciliation_authority import select_reconciliation
    def denied(*args, **kwargs):
        raise PermissionError('unreadable protected decision')
    monkeypatch.setattr(authority._Store, 'read', denied)
    with pytest.raises(ValueError, match='historical_generation_authority_io_unavailable'):
        with authority._session(historical_installation[0], authority._Operation(1030, lambda: 0)) as (files, config, store):
            select_reconciliation(files, config, store, historical_installation[0],
                None, [], 'a' * 32, None, 1030)
