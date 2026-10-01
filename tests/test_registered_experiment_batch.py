"""ADP-009D/day28: finite registered batches preserve unsealed producer bytes.

These refusal tests do not prove positive retirement throughput; ordinary
producer completion and its durable no-future-writes seal remain required.
"""

# ruff: noqa: F811
# Covers (for impacted-test selection):
#   src/blueprint_pipeline/control_plane_lane_experiment_retirement.py
#   src/blueprint_pipeline/control_plane_lane_experiment_birth.py
import json
from pathlib import Path

import pytest

from tests.test_owner_target_version_publication import root_metadata  # noqa: F401
from tests.test_registered_experiment_birth import birth
from tests.test_registered_experiment_issuer import installation, issue  # noqa: F401
from tests.test_registered_experiment_retirement_flow import retirement_installation  # noqa: F401


def test_actual_sixteen_authentic_experiments_keep_independent_owner_generation(
    retirement_installation,
):
    from blueprint_pipeline import control_plane_lane_scratch as scratch

    installation = retirement_installation
    grants, targets = [], []
    for index in range(16):
        grant = issue(installation, reference_value=f"finite-batch-{index:02d}")
        born = birth(installation, grant)
        path = Path(born["path"])
        assert path.is_dir()
        lease = json.loads((path / scratch.LEASE_FILE).read_bytes())
        assert lease["owner"] == "owner"
        assert lease["run_ref"] == f"finite-batch-{index:02d}"
        assert lease["expires_at_epoch"] == 2800
        assert born["generation"] != grant["intent_id"]
        grants.append(grant)
        targets.append((path, path.stat().st_ino, born["generation"]))
    head = json.loads((installation[2].parents[1] / "experiment-authority/HEAD.json").read_bytes())
    record = json.loads(
        (installation[2].parents[1] / "experiment-authority" / head["record_name"]).read_bytes()
    )
    assert len(record["enrollments"]) == 16
    assert {entry["intent_id"] for entry in record["enrollments"]} == {
        grant["intent_id"] for grant in grants
    }
    assert len({entry["generation"] for entry in record["enrollments"]}) == 16
    assert all(path.stat().st_ino == inode for path, inode, _ in targets)


def test_actual_sixteen_expired_unsealed_experiments_are_kept_without_rebinding(
    retirement_installation, monkeypatch
):
    from blueprint_pipeline import control_plane_lane_experiment_work as work
    from tests.test_registered_experiment_retirement_flow import (
        _current_entry,
        _gc,
        _issue_action,
        _payload_snapshot,
    )

    setup = retirement_installation
    selected, peaks = [], []
    original_slot = work._ActionFiles.slot

    def observed(self):
        peaks.append(len(self.owned) + len(self.probe_owned))
        return original_slot(self)

    monkeypatch.setattr(work._ActionFiles, "slot", observed)
    for index in range(16):
        grant = issue(setup, reference_value=f"expired-batch-{index:02d}")
        born = birth(setup, grant)
        target = Path(born["path"])
        payload = f"bounded-result-{index:02d}".encode()
        (target / "result.bin").write_bytes(payload)
        entry = _current_entry(setup, grant["intent_id"])
        assert entry["state"] == "active" and entry["completion"] is None
        snapshot = _payload_snapshot(target)
        records = {path.relative_to(setup[2]): path.read_bytes()
                   for path in setup[2].rglob("*") if path.is_file()}
        with pytest.raises(ValueError, match="^experiment_producer_completion_missing$"):
            _issue_action(setup, grant)
        assert {path.relative_to(setup[2]): path.read_bytes()
                for path in setup[2].rglob("*") if path.is_file()} == records
        assert _current_entry(setup, grant["intent_id"]) == entry
        assert _payload_snapshot(target) == snapshot
        selected.append((grant, born, target, snapshot, entry))
    for _ in range(2):
        phase = _gc(setup)["registered_experiments"]
        assert phase["enabled"] is True and phase["outcomes"] == []
        for grant, born, target, snapshot, original_entry in selected:
            assert _payload_snapshot(target) == snapshot
            entry = _current_entry(setup, grant["intent_id"])
            assert entry == original_entry and entry["generation"] == born["generation"]
    assert not list(setup[2].glob("*.action.json"))
    assert peaks and max(peaks) < 128


def test_authentic_new_registration_is_attributed_by_existing_usage_survey(retirement_installation):
    from blueprint_pipeline.control_plane_disk_usage import survey_usage
    from tests.test_control_plane_disk_usage import _statvfs

    setup = retirement_installation
    grant = issue(setup)
    born = birth(setup, grant)
    target = Path(born["path"])
    (target / "result.bin").write_bytes(b"bounded attributed result")
    volume = Path(setup[1]["lane_scratch_work_root"]).parent
    survey = survey_usage(
        [str(volume)],
        aliases={str(volume): "/mnt/blueprint-work"},
        statvfs=_statvfs(),
        mountinfo=str(setup[0].parent / "missing-mountinfo"),
    )
    assert survey["status"] == "complete"
    assert any(
        row["owner"] == "lane:g1"
        and row["storage_class"] == "lane_scratch"
        and row["allocated_bytes"] > 0
        for row in survey["top_owners"]
    )
    assert not any(
        row["root"].endswith("/" + target.name) for row in survey["orphan_scratch_roots"]
    )
