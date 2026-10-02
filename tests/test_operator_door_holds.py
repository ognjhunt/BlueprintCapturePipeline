"""Durable owned timer holds survive a host reboot and expire on schedule."""

# Covers (for impacted-test selection):
#   deploy/operator-door/operator_door/holds.py

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deploy" / "operator-door"))

from operator_door import holds  # noqa: E402


def _record(root: Path, unit: str, *, expires_at_epoch: int, enabled_before: bool) -> Path:
    request_id = "20260927T000000Z-hold-0000abcd"
    path = root / f"{unit}.json"
    path.write_text(json.dumps({
        "schema": holds.SCHEMA, "unit": unit, "owner": "alice", "reason": "inspect",
        "requested_by": "cloud", "request_id": request_id,
        "created_at": holds.timestamp(1000), "expires_at": holds.timestamp(expires_at_epoch),
        "expires_at_epoch": expires_at_epoch, "enabled_before": enabled_before,
        "status": "active",
    }), encoding="utf-8")
    return path


def test_sweep_keeps_live_hold_disabled_and_releases_expired_hold(tmp_path, monkeypatch) -> None:
    root = tmp_path / "holds"
    root.mkdir()
    live = "blueprint-scene-progression.timer"
    expired = "blueprint-capture-progress.path"
    live_path = _record(root, live, expires_at_epoch=2000, enabled_before=True)
    expired_path = _record(root, expired, expires_at_epoch=1100, enabled_before=True)
    calls = []

    def systemctl(argv, **_kwargs):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(holds.subprocess, "run", systemctl)

    assert holds.sweep(root, now=1200) == 0
    assert ["systemctl", "disable", "--", live] in calls
    assert ["systemctl", "stop", "--", live] in calls
    assert ["systemctl", "enable", "--", expired] in calls
    assert ["systemctl", "--no-block", "start", "--", expired] in calls
    assert json.loads(live_path.read_text())["status"] == "active"
    assert not expired_path.exists()
    archived = list((root / "history").glob("*.json"))
    assert len(archived) == 1 and json.loads(archived[0].read_text())["status"] == "expired_released"


def test_expiry_does_not_enable_a_previously_disabled_unit(tmp_path, monkeypatch) -> None:
    root = tmp_path / "holds"
    root.mkdir()
    unit = "blueprint-scene-progression.timer"
    path = _record(root, unit, expires_at_epoch=1100, enabled_before=False)
    calls = []

    def systemctl(argv, **_kwargs):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(holds.subprocess, "run", systemctl)

    assert holds.expire(root, unit, json.loads(path.read_text())["request_id"], now=1200) == 0
    assert calls == [["systemctl", "--no-block", "start", "--", unit]]
    assert not path.exists()


def test_sweep_finishes_release_after_crash_between_guard_removal_and_start(tmp_path, monkeypatch) -> None:
    root = tmp_path / "holds"
    root.mkdir()
    unit = "blueprint-scene-progression.timer"
    path = _record(root, unit, expires_at_epoch=2000, enabled_before=True)
    record = json.loads(path.read_text())
    with holds.locked(root):
        holds.begin_release(root, unit, record, released_by="cloud", status="released")
    calls = []

    def systemctl(argv, **_kwargs):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(holds.subprocess, "run", systemctl)
    assert holds.sweep(root, now=1200) == 0
    assert not path.exists()
    assert calls == [
        ["systemctl", "enable", "--", unit],
        ["systemctl", "--no-block", "start", "--", unit],
    ]
    assert not list((root / "releasing").glob("*.json"))
    archived = list((root / "history").glob("*.json"))
    assert len(archived) == 1 and json.loads(archived[0].read_text())["status"] == "released"


def test_begin_release_rejects_stale_generation_without_a_pending_intent(tmp_path) -> None:
    root = tmp_path / "holds"
    root.mkdir()
    unit = "blueprint-scene-progression.timer"
    path = _record(root, unit, expires_at_epoch=2000, enabled_before=True)
    stale = json.loads(path.read_text())
    current = {**stale, "request_id": "20260927T000000Z-hold-0000abce"}
    holds.write(root, unit, current)

    with holds.locked(root), pytest.raises(holds.HoldError, match="generation_changed"):
        holds.begin_release(root, unit, stale, released_by="cloud", status="released")

    assert json.loads(path.read_text())["request_id"] == current["request_id"]
    assert not (root / "releasing" / f"{unit}.json").exists()


def test_sweep_refuses_invalid_pending_unit_without_running_systemctl(tmp_path, monkeypatch) -> None:
    root = tmp_path / "holds"
    pending = root / "releasing"
    pending.mkdir(parents=True)
    (pending / "blueprint-gpu-spend-guard.timer.json").write_text("{}")
    monkeypatch.setattr(holds.subprocess, "run", lambda *_args, **_kwargs: pytest.fail("systemctl called"))

    with pytest.raises(holds.HoldError, match="hold_record_invalid"):
        holds.sweep(root)


def test_sweep_finishes_failed_hold_without_starting_previously_inactive_unit(tmp_path, monkeypatch) -> None:
    root = tmp_path / "holds"
    root.mkdir()
    unit = "blueprint-scene-progression.timer"
    path = _record(root, unit, expires_at_epoch=2000, enabled_before=False)
    record = json.loads(path.read_text())
    with holds.locked(root):
        holds.begin_release(root, unit, record, released_by="runner", status="failed_released",
                            restart_on_release=False)
    monkeypatch.setattr(holds.subprocess, "run", lambda *_args, **_kwargs: pytest.fail("systemctl called"))

    assert holds.sweep(root, now=1200) == 0
    assert not path.exists()
    archived = list((root / "history").glob("*.json"))
    assert len(archived) == 1 and json.loads(archived[0].read_text())["status"] == "failed_released"


def test_explicit_dispatch_stop_survives_deadline_and_boot_until_release(tmp_path, monkeypatch) -> None:
    root = tmp_path / "holds"
    root.mkdir()
    unit = "blueprint-agent-run-dispatcher.timer"
    path = _record(root, unit, expires_at_epoch=1100, enabled_before=True)
    record = {**json.loads(path.read_text()), "require_explicit_release": True}
    holds.write(root, unit, record)
    calls = []

    def systemctl(argv, **_kwargs):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(holds.subprocess, "run", systemctl)
    assert holds.expire(root, unit, record["request_id"], now=1200) == 0
    assert calls == []
    assert holds.sweep(root, now=1200) == 0
    assert calls == [["systemctl", "disable", "--", unit], ["systemctl", "stop", "--", unit]]
    assert holds.read(root, unit) == record
    # Only the existing explicit-release operation restores its prior boot policy.
    with holds.locked(root):
        holds.begin_release(root, unit, record, released_by="owner", status="released")
        assert holds.finish_release(root, unit) == 0
    assert calls[-2:] == [["systemctl", "enable", "--", unit],
                          ["systemctl", "--no-block", "start", "--", unit]]
    assert not path.exists()


@pytest.mark.parametrize("policy", ["true", 1, None])
def test_invalid_retained_explicit_release_policy_fails_closed(tmp_path, policy) -> None:
    root = tmp_path / "holds"
    root.mkdir()
    unit = "blueprint-agent-run-dispatcher.timer"
    path = _record(root, unit, expires_at_epoch=1100, enabled_before=True)
    path.write_text(json.dumps({**json.loads(path.read_text()), "require_explicit_release": policy}))
    with pytest.raises(holds.HoldError, match="hold_record_invalid"):
        holds.expire(root, unit, "20260927T000000Z-hold-0000abcd", now=1200)
