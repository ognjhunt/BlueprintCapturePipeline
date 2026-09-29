"""ISSUE/scan mechanics after explicit test-only deletion eligibility bypass.

These tests do not establish producer completion or product cleanup authority.
"""

# ruff: noqa: F811
# Covers (for impacted-test selection):
#   src/blueprint_pipeline/control_plane_lane_experiment_actions.py
#   src/blueprint_pipeline/control_plane_lane_experiment_work.py
import json
from pathlib import Path

import pytest

from tests.test_owner_target_version_publication import root_metadata, protected_root_tmp_path  # noqa: F401
from tests.test_registered_experiment_issuer import installation, issue  # noqa: F401
from tests.test_registered_experiment_birth import birth
from tests.test_registered_experiment_retirement_flow import (  # noqa: F401
    retirement_installation, _issue_action, removal_engine_only,
)


def test_issue_selection_is_durable_before_scan_and_reuses_original_operation(
    retirement_installation, monkeypatch, removal_engine_only
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
    retirement_installation, monkeypatch, removal_engine_only
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


def test_gc_cannot_reset_original_issue_controller_after_boot_change(
    retirement_installation, monkeypatch, removal_engine_only
):
    from blueprint_pipeline import control_plane_lane_experiment_work as work
    from tests.test_registered_experiment_retirement_flow import _gc

    setup = retirement_installation
    grant = issue(setup)
    born = birth(setup, grant)
    payload = Path(born["path"]) / "payload"
    payload.write_bytes(b"retain across unknown operation boot")
    selected = _issue_action(setup, grant)
    monkeypatch.setattr(
        work, "_controller_boot_id", lambda files: "aaaaaaaa-1234-1234-1234-123456789abc"
    )
    report = _gc(setup)
    outcome = next(
        value
        for value in report["registered_experiments"]["outcomes"]
        if value["action_id"] == selected["action_id"]
    )
    assert outcome["decision"] == "kept", outcome
    assert payload.read_bytes() == b"retain across unknown operation boot"


def test_fixed_kernel_boot_record_accepts_zero_stat_size_with_bounded_original_read(
    protected_root_tmp_path, root_metadata, monkeypatch
):
    import os
    from types import SimpleNamespace
    from blueprint_pipeline import control_plane_lane_experiment_work as work

    path = protected_root_tmp_path / "kernel-boot-id"
    path.write_bytes(b"12345678-1234-1234-1234-123456789abc\n")
    path.chmod(0o444)
    identity = (path.stat().st_dev, path.stat().st_ino)
    original_stat, original_fstat = os.stat, os.fstat

    def proc_shape(info):
        if (info.st_dev, info.st_ino) != identity:
            return info
        values = {name: getattr(info, name) for name in dir(info) if name.startswith("st_")}
        values["st_size"] = 0  # Real procfs boot_id reports size zero, while read yields37bytes.
        return SimpleNamespace(**values)

    monkeypatch.setattr(
        os, "stat", lambda *args, **kwargs: proc_shape(original_stat(*args, **kwargs))
    )
    monkeypatch.setattr(
        os, "fstat", lambda *args, **kwargs: proc_shape(original_fstat(*args, **kwargs))
    )
    monkeypatch.setattr(work, "_BOOT_PATH", path)
    files = work._ActionFiles()
    try:
        assert work._controller_boot_id(files) == "12345678-1234-1234-1234-123456789abc"
        assert not any(
            (value.st_dev, value.st_ino) == identity
            for fd, value in files.acquired.items()
            if fd in files.owned
        )
    finally:
        files.finish()
        files.budget.close()


@pytest.mark.parametrize("kind", ["member_removed", "restore_directory", "restore_member"])
def test_all_row_event_kinds_refuse_oversized_payload_before_publication(
    protected_root_tmp_path, root_metadata, kind
):
    from blueprint_pipeline import control_plane_lane_experiment_actions as actions
    from blueprint_pipeline.control_plane_lane_experiment_work import _ActionFiles

    operation = protected_root_tmp_path / "operation"
    operation.mkdir(mode=0o700)
    action = dict(action_id="a" * 32, intent_id="b" * 32, generation="c" * 32)
    files = _ActionFiles()
    try:
        parent, _ = files.parent(operation / "e-00000.json", protected=True)
        with pytest.raises(ValueError, match="experiment_record_limit"):
            actions._event(files, parent, action, kind, dict(path="\\" * 2500), 0, None, 2000)
        assert not list(operation.iterdir())
    finally:
        files.finish()
        files.budget.close()


def test_escaped_row_size_keeps_payload(retirement_installation, removal_engine_only):
    from tests.test_registered_experiment_retirement_flow import _gc, _payload_snapshot

    setup = retirement_installation
    grant = issue(setup)
    born = birth(setup, grant)
    target = Path(born["path"])
    directory = target
    for number in range(15):
        directory = directory / ("\\" * 50 + f"{number:02d}")
        directory.mkdir()
    payload = directory / "payload"
    payload.write_bytes(b"known event envelope refusal must preserve this")
    before = _payload_snapshot(target)
    action = _issue_action(setup, grant)
    report = _gc(setup)
    outcome = next(
        value
        for value in report["registered_experiments"]["outcomes"]
        if value["action_id"] == action["action_id"]
    )
    assert outcome["decision"] == "kept", outcome
    assert _payload_snapshot(target) == before


@pytest.mark.parametrize("character", ["x", "\\"])
def test_manifest_admits_bounded_rows_before_accumulating_whole_inventory(
    retirement_installation, monkeypatch, character, removal_engine_only
):
    from blueprint_pipeline import control_plane_lane_experiment_actions as actions
    from blueprint_pipeline import control_plane_lane_experiment_work as work

    setup = retirement_installation
    grant = issue(setup)
    born = birth(setup, grant)
    target = Path(born["path"])
    names = {character * 100 + f"{number:02d}" for number in range(10)}
    for name in names:
        (target / name).write_bytes(b"tiny retained bytes")
    # Shrink only this new local output cap; native B and all global limits stay
    # unchanged. The real manifest cap is 1MiB, enforced before list growth.
    monkeypatch.setattr(actions, "_MANIFEST_LIMIT", 1200, raising=False)
    opened = []
    original = work._ActionFiles.open

    def observe(self, name, *args, **kwargs):
        if name in names:
            opened.append(name)
        return original(self, name, *args, **kwargs)

    monkeypatch.setattr(work._ActionFiles, "open", observe)
    with pytest.raises(ValueError, match="experiment_manifest_limit"):
        _issue_action(setup, grant)
    assert len(opened) < len(names)
    assert all((target / name).read_bytes() == b"tiny retained bytes" for name in names)
    assert not list(setup[2].glob("*.manifest.json"))
