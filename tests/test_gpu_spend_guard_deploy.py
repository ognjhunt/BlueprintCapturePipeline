"""Structural validation for the scheduled GPU spend guard deployment (R055/R056).

``systemd-analyze verify`` is unavailable in the hermetic test lane, so the unit
files are parsed as the INI-like documents systemd expects and checked for the
required sections/keys, and the post-check script is exercised end to end so a
broken snapshot fails the unit.
"""

from __future__ import annotations

import configparser
import json
import subprocess
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SYSTEMD_DIR = REPO_ROOT / "deploy" / "systemd"
SERVICE = SYSTEMD_DIR / "blueprint-gpu-spend-guard.service"
TIMER = SYSTEMD_DIR / "blueprint-gpu-spend-guard.timer"
POSTCHECK = SYSTEMD_DIR / "blueprint-gpu-spend-guard-postchecks.sh"
INSTALLER = REPO_ROOT / "scripts" / "install_live_pipeline_control_plane.sh"


def _parse_unit(path: Path) -> configparser.RawConfigParser:
    # RawConfigParser(strict=False): systemd allows repeated keys (multiple
    # Environment= lines) and no %-interpolation is wanted.
    cp = configparser.RawConfigParser(strict=False)
    cp.read(path, encoding="utf-8")
    return cp


# --------------------------- unit files parse + shape ---------------------------


def test_service_unit_parses_and_has_required_shape() -> None:
    cp = _parse_unit(SERVICE)
    assert cp.has_section("Unit")
    assert cp.has_section("Service")
    assert cp.get("Service", "Type") == "oneshot"
    # A cost watchdog is useless without network to the providers.
    assert "network-online.target" in cp.get("Unit", "After")
    exec_start = cp.get("Service", "ExecStart")
    assert "scripts/gpu_spend_guard.py" in exec_start
    assert "--reap" in exec_start  # scheduled enforcement, not dry-run
    assert "--json-report" in exec_start  # durable teardown evidence
    assert "--orphan-booted-max-age-seconds" in exec_start  # R056 booted-orphan reap
    assert "blueprint-gpu-spend-guard-postchecks.sh" in cp.get("Service", "ExecStartPost")


def test_service_unit_reuses_existing_env_conventions() -> None:
    raw = SERVICE.read_text(encoding="utf-8")
    # Same EnvironmentFile as the control-plane unit; no new secret files invented.
    assert "EnvironmentFile=-/etc/blueprint/pipeline-control-plane.env" in raw
    assert "BLUEPRINT_PIPELINE_REPO=/opt/blueprint/BlueprintCapturePipeline" in raw
    assert "BLUEPRINT_GPU_ORPHAN_BOOTED_MAX_AGE_SECONDS=" in raw
    assert "BLUEPRINT_GPU_SPEND_GUARD_SNAPSHOT_PATH=" in raw
    # Hardening flags mirrored from the control-plane unit.
    assert "NoNewPrivileges=true" in raw
    assert "PrivateTmp=true" in raw


def test_timer_unit_parses_and_runs_on_a_short_interval() -> None:
    cp = _parse_unit(TIMER)
    assert cp.has_section("Timer")
    assert cp.get("Timer", "Unit") == "blueprint-gpu-spend-guard.service"
    assert cp.get("Timer", "Persistent") == "true"
    assert cp.get("Install", "WantedBy") == "timers.target"
    # Short cadence so runaway cost is caught quickly (<= 10 minutes).
    active = cp.get("Timer", "OnUnitActiveSec")
    assert active.endswith("min")
    assert int(active.removesuffix("min")) <= 10


def test_installer_wires_the_spend_guard_units() -> None:
    installer = INSTALLER.read_text(encoding="utf-8")
    assert "blueprint-gpu-spend-guard.service" in installer
    assert "blueprint-gpu-spend-guard.timer" in installer
    assert "systemctl enable --now blueprint-gpu-spend-guard.timer" in installer


def test_installer_dry_run_lists_spend_guard_units() -> None:
    proc = subprocess.run(
        ["bash", str(INSTALLER), "--dry-run"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    assert "blueprint-gpu-spend-guard.service" in proc.stdout
    assert "blueprint-gpu-spend-guard.timer" in proc.stdout


# --------------------------- post-check behavior ---------------------------


def _run_postcheck(snapshot: Path, *, max_age: str = "900") -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(POSTCHECK)],
        capture_output=True,
        text=True,
        timeout=60,
        env={
            "PATH": "/usr/bin:/bin:/usr/local/bin",
            "BLUEPRINT_PIPELINE_REPO": str(REPO_ROOT),
            "BLUEPRINT_GPU_SPEND_GUARD_SNAPSHOT_PATH": str(snapshot),
            "BLUEPRINT_GPU_SPEND_GUARD_SNAPSHOT_MAX_AGE_SECONDS": max_age,
        },
    )


def _valid_snapshot() -> dict:
    return {
        "schema_version": "gpu_spend_guard.v1",
        "reap_mode": True,
        "live_instance_count": 0,
        "total_burn_per_hour_usd": 0,
        "reap_results": [],
        "booted_orphan_reaping_enabled": True,
    }


def test_postcheck_passes_on_fresh_reap_snapshot(tmp_path: Path) -> None:
    snap = tmp_path / "snap.json"
    snap.write_text(json.dumps(_valid_snapshot()), encoding="utf-8")
    proc = _run_postcheck(snap)
    assert proc.returncode == 0, proc.stderr
    assert "snapshot ok" in proc.stdout


def test_postcheck_fails_when_snapshot_missing(tmp_path: Path) -> None:
    proc = _run_postcheck(tmp_path / "absent.json")
    assert proc.returncode == 1
    assert "not written" in proc.stderr


def test_postcheck_fails_on_dry_run_snapshot(tmp_path: Path) -> None:
    snap = tmp_path / "snap.json"
    payload = _valid_snapshot()
    payload["reap_mode"] = False
    snap.write_text(json.dumps(payload), encoding="utf-8")
    proc = _run_postcheck(snap)
    assert proc.returncode == 1
    assert "not produced in --reap mode" in proc.stderr


def test_postcheck_fails_on_wrong_schema(tmp_path: Path) -> None:
    snap = tmp_path / "snap.json"
    snap.write_text(json.dumps({"schema_version": "other", "reap_mode": True}), encoding="utf-8")
    proc = _run_postcheck(snap)
    assert proc.returncode == 1
    assert "schema_version" in proc.stderr


def test_postcheck_fails_on_stale_snapshot(tmp_path: Path) -> None:
    snap = tmp_path / "snap.json"
    snap.write_text(json.dumps(_valid_snapshot()), encoding="utf-8")
    # Backdate the file well beyond the freshness window.
    old = time.time() - 10_000
    import os

    os.utime(snap, (old, old))
    proc = _run_postcheck(snap, max_age="900")
    assert proc.returncode == 1
    assert "stale" in proc.stderr


def test_guard_reap_snapshot_passes_postcheck_end_to_end(tmp_path: Path, monkeypatch) -> None:
    # The real guard's --reap --json-report output must satisfy the post-check.
    import sys

    sys.path.insert(0, str(REPO_ROOT))
    from scripts import gpu_spend_guard as guard

    monkeypatch.setattr(guard, "_read_secret", lambda name, **_kw: None)
    snap = tmp_path / "snap.json"
    rc = guard.main(["--reap", "--orphan-booted-max-age-seconds", "21600",
                     "--json-report", str(snap)])
    assert rc == 0
    proc = _run_postcheck(snap)
    assert proc.returncode == 0, proc.stderr
