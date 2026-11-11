"""The host rebuild script: its dry run names every missing owner secret, invents none, and plans every step."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts/rebuild_control_plane_host.sh"
SECRETS = REPO / "deploy/host/rebuild-owner-secrets.txt"


def _names() -> list[str]:
    return [line for line in SECRETS.read_text().splitlines() if line and not line.startswith("#")]


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["bash", str(SCRIPT), "--dry-run", *args], capture_output=True, text=True, timeout=60)


def _supply(directory: Path, names: list[str]) -> None:
    for name in names:
        path = directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("owner-supplied\n")


def test_the_secret_list_is_every_secret_a_unit_or_the_installer_names():
    pattern = re.compile(r"/etc/blueprint/((?:provider-secrets|credentials|control-plane-backup)/[A-Za-z0-9_.-]+)"
                         r"|\$\{(PROVIDER_SECRETS_DIR|CREDENTIALS_DIR)\}/([A-Za-z0-9_.-]+)")
    named: set[str] = set()
    for path in [*sorted((REPO / "deploy/systemd").iterdir()), REPO / "scripts/install_live_pipeline_control_plane.sh"]:
        for match in pattern.finditer(path.read_text(encoding="utf-8")):
            if match[1]:
                named.add(match[1])
            else:
                named.add(("provider-secrets" if match[2] == "PROVIDER_SECRETS_DIR" else "credentials") + "/" + match[3])
    assert set(_names()) == named


def test_the_dry_run_names_a_missing_secret_and_refuses(tmp_path):
    secrets = tmp_path / "secrets"
    names = _names()
    _supply(secrets, [name for name in names if name != "provider-secrets/vast_api_key"])
    result = _run("--root", str(tmp_path / "host"), "--secrets-from", str(secrets),
                  "--work-volume-device", "/dev/disk/by-id/volume")
    assert result.returncode == 3
    assert "missing owner secret: provider-secrets/vast_api_key" in result.stderr
    assert result.stderr.count("missing owner secret:") == 1
    assert "plan:" not in result.stdout


def test_the_dry_run_with_every_secret_plans_every_step_and_changes_nothing(tmp_path):
    secrets = tmp_path / "secrets"
    _supply(secrets, _names())
    host = tmp_path / "host"
    result = _run("--root", str(host), "--secrets-from", str(secrets), "--work-volume-device", "/dev/disk/by-id/v")
    assert result.returncode == 0, result.stderr
    plans = [line for line in result.stdout.splitlines() if line.startswith("plan:")]
    for needle in ("bootstrap_control_plane_host.sh", "mount_work_volume.sh", "control_plane_state_backup restore",
                   "install_live_pipeline_control_plane.sh", "operator-door/install.sh", "systemctl enable --now"):
        assert any(needle in line for line in plans), needle
    assert "rebuild planned" in result.stdout
    assert not host.exists()


def test_a_waived_secret_is_named_never_defaulted(tmp_path):
    secrets = tmp_path / "secrets"
    _supply(secrets, [name for name in _names() if name != "provider-secrets/ngc_api_key"])
    result = _run("--root", str(tmp_path / "host"), "--secrets-from", str(secrets), "--work-volume-device", "/dev/x",
                  "--waive-secret", "provider-secrets/ngc_api_key")
    assert result.returncode == 0, result.stderr
    assert "waived by the owner: provider-secrets/ngc_api_key" in result.stderr
    assert not (tmp_path / "host/etc/blueprint/provider-secrets/ngc_api_key").exists()


def test_the_dry_run_enables_exactly_the_recorded_units(tmp_path):
    secrets = tmp_path / "secrets"
    _supply(secrets, _names())
    host = tmp_path / "host"
    report = host / "var/lib/blueprint/pipeline-control-plane/control-plane-backup"
    report.mkdir(parents=True)
    (report / "enabled_units.json").write_text(json.dumps({"units": ["blueprint-gpu-spend-guard.timer",
                                                                     "blueprint-gone.timer"]}))
    (host / "etc/systemd/system").mkdir(parents=True)
    (host / "etc/systemd/system/blueprint-gpu-spend-guard.timer").write_text("[Timer]\n")
    result = _run("--root", str(host), "--secrets-from", str(secrets), "--work-volume-device", "/dev/x")
    assert result.returncode == 0, result.stderr
    assert "plan: systemctl enable --now blueprint-gpu-spend-guard.timer" in result.stdout
    assert "not installed, not enabled: blueprint-gone.timer" in result.stderr


def test_a_rebuild_without_the_work_volume_refuses(tmp_path):
    secrets = tmp_path / "secrets"
    _supply(secrets, _names())
    result = _run("--root", str(tmp_path / "host"), "--secrets-from", str(secrets))
    assert result.returncode == 3 and "--work-volume-device is required" in result.stderr
