"""ADP-009D/day28: past journal facts never renew mutation authority."""
# Covers: src/blueprint_pipeline/control_plane_lane_historical_receipts.py
# Covers: src/blueprint_pipeline/control_plane_lane_historical_gc.py

import pytest

from tests.test_historical_generation_authority import (
    decision, historical_installation, packet)  # noqa: F401
from tests.test_historical_generation_journal import journal_call
from tests.test_historical_generation_gc_dispatch import enable
from tests.test_registered_experiment_issuer import installation  # noqa: F401
from tests.test_owner_target_version_publication import root_metadata  # noqa: F401

# ruff: noqa: F811


def observe(installed, approved, *, now=1030, monotonic=lambda: 0):
    from blueprint_pipeline.control_plane_lane_historical_receipts import observe_historical_action
    return observe_historical_action(installed_config_path=installed[0], action_id=approved['action_id'],
                                     now=now, monotonic=monotonic)


def test_absent_journal_is_pending_without_creating_records(historical_installation):
    enable(historical_installation)
    approved = decision(historical_installation, packet(historical_installation))
    directory = historical_installation[3].parent / 'historical-generation-journals'
    before = (historical_installation[1] / 'one.log').read_bytes()
    assert observe(historical_installation, approved) is None
    assert list(directory.iterdir()) == []
    assert (historical_installation[1] / 'one.log').read_bytes() == before


def test_expired_incomplete_journal_can_be_read_but_never_resumed(historical_installation):
    enable(historical_installation)
    approved = decision(historical_installation, packet(historical_installation))
    journal_call(historical_installation, approved, lambda journal: journal.head)
    directory = historical_installation[3].parent / 'historical-generation-journals' / approved['action_id']
    before = {path.name: path.read_bytes() for path in directory.iterdir()}
    assert observe(historical_installation, approved, now=2000, monotonic=lambda: 20000) is None
    assert {path.name: path.read_bytes() for path in directory.iterdir()} == before
    assert (historical_installation[1] / 'one.log').is_file()


def test_completed_label_without_real_removal_transitions_is_not_a_receipt(historical_installation):
    enable(historical_installation)
    approved = decision(historical_installation, packet(historical_installation))
    # Deliberately false label: no fence or removal occurred. This negative
    # fixture must never stand in for a positive completed producer/action.
    def false_label(journal):
        journal.append('final', {'status': 'completed', 'action': 'delete'}, previous=journal.head['event_digest'])
    journal_call(historical_installation, approved, false_label)
    with pytest.raises(ValueError):
        observe(historical_installation, approved)
    assert (historical_installation[1] / 'one.log').read_bytes() == b'original owner diagnostics\n'


@pytest.mark.parametrize('change', ['mode', 'symlink', 'digest'])
def test_changed_original_journal_never_becomes_a_receipt(historical_installation, change):
    enable(historical_installation)
    approved = decision(historical_installation, packet(historical_installation))
    journal_call(historical_installation, approved, lambda journal: journal.head)
    path = historical_installation[3].parent / 'historical-generation-journals' / approved['action_id'] / 'e-00000.json'
    if change == 'mode':
        path.chmod(0o644)
    elif change == 'symlink':
        original = path.with_name('preserved')
        path.rename(original)
        path.symlink_to(original)
    else:
        path.write_bytes(path.read_bytes().replace(b'1030', b'1040'))
    with pytest.raises(ValueError):
        observe(historical_installation, approved)
    assert (historical_installation[1] / 'one.log').is_file()
