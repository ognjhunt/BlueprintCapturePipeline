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
