"""The real historical worker stays default-off and holds fresh authority."""
# Covers (for impacted-test selection):
#   deploy/operator-door/operator_door/config.py
#   src/blueprint_pipeline/control_plane_lane_historical_action.py
import json
from pathlib import Path

import pytest

from tests.test_historical_generation_authority import historical_installation, packet, decision  # noqa: F401
from tests.test_registered_experiment_issuer import installation, encoded  # noqa: F401
from tests.test_owner_target_version_publication import root_metadata  # noqa: F401


def test_historical_worker_is_default_off_before_journal_or_payload_mutation(historical_installation):  # noqa: F811
    from blueprint_pipeline.control_plane_lane_historical_action import run_historical_action
    installed = historical_installation
    approved = decision(installed, packet(installed))
    before = (installed[1] / 'one.log').read_bytes()
    with pytest.raises(ValueError, match='action_disabled'):
        run_historical_action(installed_config_path=installed[0], action_id=approved['action_id'],
                              now=1030, monotonic=lambda: 0)
    journals = installed[3].parent / 'historical-generation-journals'
    assert list(journals.iterdir()) == []
    assert (installed[1] / 'one.log').read_bytes() == before


@pytest.mark.parametrize('invalid', [0, 1, 'true', None])
def test_historical_enablement_requires_bool(monkeypatch, invalid):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / 'deploy/operator-door'))
    from operator_door.config import DoorConfigError, config_from_mapping
    with pytest.raises(DoorConfigError):
        config_from_mapping({'historical_generation_actions_enabled': invalid})


def test_historical_enablement_defaults_off_and_accepts_explicit_bool(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / 'deploy/operator-door'))
    from operator_door.config import config_from_mapping
    assert config_from_mapping({}).historical_generation_actions_enabled is False
    assert config_from_mapping({'historical_generation_actions_enabled': True}).historical_generation_actions_enabled is True


def test_enabled_worker_refuses_unproven_native_rights_and_preserves_bytes(historical_installation):  # noqa: F811
    from blueprint_pipeline.control_plane_lane_historical_action import run_historical_action
    installed = historical_installation
    settings = json.loads(installed[0].read_bytes())
    settings['historical_generation_actions_enabled'] = True
    installed[0].write_bytes(encoded(settings))
    approved = decision(installed, packet(installed))
    before = (installed[1] / 'one.log').read_bytes()
    with pytest.raises(ValueError, match='native_unavailable|unit_rights_unknown'):
        run_historical_action(installed_config_path=installed[0], action_id=approved['action_id'],
                              now=1030, monotonic=lambda: 0)
    assert (installed[1] / 'one.log').read_bytes() == before
    assert list((installed[3].parent / 'historical-generation-journals').iterdir()) == []
