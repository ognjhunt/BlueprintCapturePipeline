"""Exact current-generation ISSUE selection and conserved original clocks."""

# ruff: noqa: F811
# Covers (for impacted-test selection):
#   src/blueprint_pipeline/control_plane_lane_experiment_actions.py
#   src/blueprint_pipeline/control_plane_lane_experiment_work.py
import json
from pathlib import Path

import pytest

from tests.test_owner_target_version_publication import root_metadata  # noqa: F401
from tests.test_registered_experiment_issuer import installation, issue  # noqa: F401
from tests.test_registered_experiment_birth import birth
from tests.test_registered_experiment_retirement_flow import retirement_installation, _issue_action  # noqa: F401


def test_issue_selection_is_durable_before_scan_and_reuses_original_operation(
    retirement_installation, monkeypatch
):
    from blueprint_pipeline import control_plane_lane_experiment_actions as actions
    from blueprint_pipeline.control_plane_lane_owner_target_versions import OwnerTargetVersionError

    setup = retirement_installation
    grant = issue(setup)
    birth(setup, grant)
    scans = []

    def interrupted(*args, **kwargs):
        scans.append(True)
        raise OwnerTargetVersionError("injected_manifest_acquisition_failure")

    monkeypatch.setattr(actions, "_manifest", interrupted)
    for _ in range(2):
        with pytest.raises(ValueError, match="injected_manifest_acquisition_failure"):
            _issue_action(setup, grant)
        locators = list(setup[2].glob(grant["intent_id"] + ".issue-selection-*.json"))
        assert len(locators) == 1
        raw = locators[0].read_bytes()
        value = json.loads(raw)
        assert (
            value["intent_id"] == grant["intent_id"] and value["operation_id"] != grant["intent_id"]
        )
        if len(scans) == 1:
            first = raw
        else:
            assert raw == first
    assert len(scans) == 2
    assert not list(Path(setup[1]["lane_scratch_work_root"]).rglob("*.action.json"))


@pytest.mark.parametrize(
    "change", ["wall_rollback", "monotonic_rollback", "boot_changed", "expired_original"]
)
def test_durable_controller_cannot_gain_fresh_retry_deadline(monkeypatch, change):
    from blueprint_pipeline import control_plane_lane_experiment_work as work

    mono, epoch = [1000.0], [2000.0]
    boot = "12345678-1234-1234-1234-123456789abc"
    monkeypatch.setattr(work, "_controller_boot_id", lambda files: boot, raising=False)
    files = work._ActionFiles(monotonic=lambda: mono[0], now=lambda: epoch[0])
    original = dict(
        boot_id=boot,
        origin_monotonic=950.0,
        deadline_monotonic=1005.0,
        origin_epoch=1950.0,
        deadline_epoch=2005.0,
        last_monotonic=999.0,
        last_epoch=1999.0,
    )
    if change == "wall_rollback":
        epoch[0] = 1998.0
    elif change == "monotonic_rollback":
        mono[0] = 998.0
    elif change == "boot_changed":
        original["boot_id"] = "aaaaaaaa-1234-1234-1234-123456789abc"
    else:
        original["deadline_monotonic"] = 1000.0
    try:
        with pytest.raises(ValueError):
            files.bind_controller(original)
        assert files.failure is not None
    finally:
        files.finish()
        files.budget.close()


@pytest.mark.parametrize("state", ["unused", "used", "failed", "closed"])
def test_original_action_controller_initializes_once_before_any_reset_or_clock(state, tmp_path):
    from blueprint_pipeline import control_plane_lane_experiment_work as work
    import os

    clock = [1000.0]
    files = work._ActionFiles(monotonic=lambda: clock[0], now=lambda: 2000.0)
    try:
        if state == "used":
            directory = tmp_path / "target"
            directory.mkdir()
            files.parent(directory / "payload")
        elif state == "failed":
            clock[0] = 20000.0
            with pytest.raises(ValueError):
                files.check_long()
        elif state == "closed":
            files.budget.close()
        old_budget, old_owned = files.budget, dict(files.owned)
        old_origin, old_failure = files.controller_origin, files.failure

        def forbidden_clock():
            pytest.fail("reinitialization reached a new clock")

        with pytest.raises(ValueError, match="experiment_work_initialization_reused"):
            files.__init__(monotonic=forbidden_clock, now=forbidden_clock)
        assert files.budget is old_budget and files.owned == old_owned
        assert files.controller_origin == old_origin and files.failure == old_failure
        for fd in old_owned:
            os.fstat(fd)
    finally:
        files.finish()
        files.budget.close()


def test_owner_expiry_also_restricts_original_monotonic_time_when_wall_clock_stalls():
    from blueprint_pipeline import control_plane_lane_experiment_work as work

    clock = [1000.0]
    files = work._ActionFiles(monotonic=lambda: clock[0], now=lambda: 2000.0)
    try:
        files.bind_deadline(2060.0)
        clock[0] = 1060.0
        with pytest.raises(ValueError):
            files.check_long()
        assert files.failure is not None
    finally:
        files.finish()
        files.budget.close()


def test_interrupted_scan_permanently_consumes_two_passes_without_new_operation(
    retirement_installation, monkeypatch
):
    from blueprint_pipeline import control_plane_lane_experiment_actions as actions
    from blueprint_pipeline.control_plane_lane_owner_target_versions import OwnerTargetVersionError

    setup = retirement_installation
    grant = issue(setup)
    birth(setup, grant)
    calls = []

    def interrupted(*args, **kwargs):
        calls.append(True)
        raise OwnerTargetVersionError("injected_manifest_acquisition_failure")

    monkeypatch.setattr(actions, "_manifest", interrupted)
    for number in range(2):
        with pytest.raises(ValueError, match="injected_manifest_acquisition_failure"):
            _issue_action(setup, grant)
        records = list(setup[2].glob("operations/*/scan-issue-*-reserved.json"))
        assert len(records) == number + 1
        assert len({path.parent.name for path in records}) == 1
        record = json.loads(records[-1].read_bytes())
        assert record["reserved_member_operations"] == 4096
        assert record["reserved_batches"] == 256
    with pytest.raises(ValueError, match="experiment_scan_pass_exhausted"):
        _issue_action(setup, grant)
    assert len(calls) == 2
