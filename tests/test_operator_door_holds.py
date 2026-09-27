"""Durable owned timer holds survive a host reboot and expire on schedule."""

# Covers (for impacted-test selection):
#   deploy/operator-door/operator_door/holds.py

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

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
    assert json.loads(expired_path.read_text())["status"] == "expired_released"


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
