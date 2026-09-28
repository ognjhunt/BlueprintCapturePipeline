"""Finite protected-owner configuration and its optional single-budget seam."""
# Covers (for impacted-test selection):
#   deploy/operator-door/operator_door/config.py
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'deploy' / 'operator-door'))
from operator_door import config as door  # noqa: E402
from blueprint_pipeline.control_plane_reference_budget import ReferenceCollectionBudget  # noqa: E402


def test_disabled_default_and_derived_private_store():
    value = door.config_from_mapping({'state_root': '/state'})
    assert value.owner_census_decisions_enabled == 0
    assert value.lane_owner_policy_file == '/etc/blueprint-operator-door/lane-owner-policy.json'
    assert value.owner_consent_store == '/state/requests/owner-consents'


@pytest.mark.parametrize('value', [True, False, -1, 2, '1', None])
def test_enablement_is_only_exact_zero_or_one(value):
    with pytest.raises(door.DoorConfigError):
        door.config_from_mapping({'owner_census_decisions_enabled': value})


def test_policy_path_remains_absolute():
    with pytest.raises(door.DoorConfigError):
        door.config_from_mapping({'lane_owner_policy_file': 'relative'})


def test_budget_expires_after_existing_bounded_symlink_query(monkeypatch):
    budget = ReferenceCollectionBudget(monotonic=lambda: clock[0])
    clock = [0.0]
    def query(path):
        clock[0] = 6.0
        return False
    monkeypatch.setattr(Path, 'is_symlink', query)
    with pytest.raises(ValueError, match='reference_deadline_exceeded'):
        door.config_from_mapping({}, _work_budget=budget)


def test_none_mapping_does_not_import_or_touch_budget(monkeypatch):
    monkeypatch.setattr(ReferenceCollectionBudget, 'tick', lambda _: pytest.fail('None ticked'))
    assert door.config_from_mapping({'listen_port': 9001}).listen_port == 9001
