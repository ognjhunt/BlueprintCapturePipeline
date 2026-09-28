"""Actual full supported member domain; zero-sized payloads bound disk usage."""

# ruff: noqa: F811
# Covers (for impacted-test selection):
#   src/blueprint_pipeline/control_plane_lane_experiment_actions.py
#   src/blueprint_pipeline/control_plane_lane_experiment_work.py
#   src/blueprint_pipeline/control_plane_storage_gc.py
from pathlib import Path

import pytest

from tests.test_owner_target_version_publication import root_metadata  # noqa: F401
from tests.test_registered_experiment_issuer import installation, issue  # noqa: F401
from tests.test_registered_experiment_birth import birth
from tests.test_registered_experiment_offload import expired_completed_evidence  # noqa: F401
from tests.test_registered_experiment_retirement_flow import (  # noqa: F401
    retirement_installation,
    _issue_action,
    _gc,
    _current_entry,
)


@pytest.mark.slow
def test_actual_4096_member_expiry_reaches_gc_without_widening_native_budget(
    retirement_installation, monkeypatch
):
    from blueprint_pipeline import control_plane_lane_experiment_work as work
    from blueprint_pipeline import control_plane_lane_scratch as scratch

    setup = retirement_installation
    grant = issue(setup)
    born = birth(setup, grant)
    target = Path(born["path"])
    lease = (target / scratch.LEASE_FILE).read_bytes()
    inode = target.stat().st_ino
    for number in range(4096):
        (target / f"m-{number:04d}").write_bytes(b"")
    peaks = []
    original = work._ActionFiles.slot

    def observed(self):
        peaks.append(len(self.owned) + len(self.probe_owned))
        return original(self)

    monkeypatch.setattr(work._ActionFiles, "slot", observed)
    action = _issue_action(setup, grant)
    report = _gc(setup)
    outcome = next(
        value
        for value in report["registered_experiments"]["outcomes"]
        if value["action_id"] == action["action_id"]
    )
    assert outcome["decision"] == "retired", outcome
    assert len(list(target.iterdir())) == 2
    assert target.stat().st_ino == inode
    assert (target / scratch.LEASE_FILE).read_bytes() == lease
    assert _current_entry(setup, grant["intent_id"])["state"] == "retired"
    assert peaks and max(peaks) < 128


@pytest.mark.slow
def test_actual_4096_evidence_offload_restores_under_same_store_and_fd_limits(
    expired_completed_evidence, monkeypatch
):
    from blueprint_pipeline import control_plane_lane_experiment_archive as archive
    from blueprint_pipeline import control_plane_lane_experiment_retirement as root
    from blueprint_pipeline import control_plane_lane_experiment_work as work
    from blueprint_pipeline import control_plane_lane_scratch as scratch
    from tests.test_registered_experiment_offload import Cloud

    setup, target, born, _, intent_id = expired_completed_evidence
    control = {scratch.LEASE_FILE, '.registered-experiment.v1.json'}
    count = sum(path.relative_to(target).parts[0] not in control for path in target.rglob('*'))
    for number in range(4096 - count):
        (target / f'extra-{number:04d}').write_bytes(b'')
    before = {str(path.relative_to(target)): path.read_bytes()
              for path in target.rglob('*') if path.is_file() and path.name not in control}
    cloud = Cloud()
    monkeypatch.setattr(archive, '_client', lambda *args: (cloud, 'development-only'))
    peaks = []
    original = work._ActionFiles.slot

    def observed(self):
        peaks.append(len(self.owned) + len(self.probe_owned))
        return original(self)

    monkeypatch.setattr(work._ActionFiles, 'slot', observed)
    action = root.issue_experiment_action_intent(intent_id, principal='operator', owner='owner',
        action='offload', expires_at_epoch=3500, installed_config_path=setup[0], now=lambda: 2900)
    outcome = _gc(setup)['registered_experiments']['outcomes'][0]
    assert outcome['action_id'] == action['action_id']
    assert outcome['decision'] == 'retired', outcome
    selected = root.issue_experiment_restore_intent(intent_id, principal='operator', owner='owner',
        lease_ttl_seconds=600, expires_at_epoch=3400, installed_config_path=setup[0], now=lambda: 2901)
    restored = root.restore_registered_experiment(selected['action_id'],
        expected_restore_intent=selected['restore_intent'], installed_config_path=setup[0], now=lambda: 2901)
    assert restored['decision'] == 'restored', restored
    assert {str(path.relative_to(target)): path.read_bytes()
            for path in target.rglob('*') if path.is_file() and path.name not in control} == before
    assert _current_entry(setup, intent_id)['state'] == 'active'
    assert _current_entry(setup, intent_id)['generation'] == born['generation']
    assert peaks and max(peaks) < 128
