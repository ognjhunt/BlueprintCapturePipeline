"""The signed G1 team queue must survive a clean controller deployment."""

from __future__ import annotations

from pathlib import Path
import os
import subprocess

import scripts.deploy_control_plane_commit as deploy


ROOT = Path(__file__).resolve().parents[1]
SERVICE = "blueprint-native-g1-team-campaign-dispatcher.service"
TIMER = "blueprint-native-g1-team-campaign-dispatcher.timer"
SETTLEMENT_SERVICE = "blueprint-native-g1-team-campaign-settlement.service"
SETTLEMENT_TIMER = "blueprint-native-g1-team-campaign-settlement.timer"


def _text(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_g1_team_queue_has_hardened_once_only_controller_and_timer() -> None:
    service = _text(f"deploy/systemd/{SERVICE}")
    timer = _text(f"deploy/systemd/{TIMER}")
    intake = _text("deploy/systemd/blueprint-pipeline-intake.service")
    installer = _text("scripts/install_live_pipeline_control_plane.sh")
    assert "native_g1_team_campaign_dispatcher" in service
    assert "KillMode=process" in service
    assert "BLUEPRINT_VAST_WATCHDOG_CALLER_EXIT_SURVIVAL=systemd_dispatcher_kill_mode_process" in service
    assert "BLUEPRINT_NATIVE_G1_TEAM_CAMPAIGN_EXECUTE=true" in service
    assert 'ARGS+=(--execute)' in service
    assert "VAST_LAUNCH_LOCK_FILE=" in service
    assert "--registry-path" in service and "--queue-root" in service
    assert "OnUnitInactiveSec=2min" in timer
    assert SERVICE in timer
    assert "BLUEPRINT_NATIVE_G1_TEAM_CAMPAIGN_REGISTRY_PATH=" in intake
    assert "BLUEPRINT_NATIVE_G1_TEAM_CAMPAIGN_QUEUE_ROOT=" in intake
    assert SERVICE in deploy.DEFAULT_DEPLOYED_SYSTEMD_UNITS
    assert TIMER in deploy.DEFAULT_DEPLOYED_SYSTEMD_UNITS
    assert TIMER in deploy.DEFAULT_ALWAYS_ARM_TIMER_UNITS
    assert f"deploy/systemd/{SERVICE}" in installer
    assert f"deploy/systemd/{TIMER}" in installer
    assert f"systemctl enable --now {TIMER}" in installer


def test_g1_team_settlement_is_armed_for_posted_billing_and_private_review() -> None:
    service = _text(f"deploy/systemd/{SETTLEMENT_SERVICE}")
    timer = _text(f"deploy/systemd/{SETTLEMENT_TIMER}")
    installer = _text("scripts/install_live_pipeline_control_plane.sh")
    assert "native_g1_team_campaign_settlement" in service
    assert "--billing-audit-root" in service
    assert "--result-root" in service
    assert "--queue-root" in service
    assert "--execute" not in service
    assert "OnUnitInactiveSec=5min" in timer
    assert SETTLEMENT_SERVICE in timer
    assert SETTLEMENT_SERVICE in deploy.DEFAULT_DEPLOYED_SYSTEMD_UNITS
    assert SETTLEMENT_TIMER in deploy.DEFAULT_DEPLOYED_SYSTEMD_UNITS
    assert SETTLEMENT_TIMER in deploy.DEFAULT_ALWAYS_ARM_TIMER_UNITS
    assert f"deploy/systemd/{SETTLEMENT_SERVICE}" in installer
    assert f"deploy/systemd/{SETTLEMENT_TIMER}" in installer
    assert f"systemctl enable --now {SETTLEMENT_TIMER}" in installer


def test_installed_dispatch_command_reaches_selected_queue_without_builtin_queue(tmp_path):
    """Execute the actual oneshot shell against inert commands, never a provider."""
    service = _text(f"deploy/systemd/{SERVICE}")
    line = next(line for line in service.splitlines() if line.startswith("ExecStart="))
    shell = line.removeprefix("ExecStart=/bin/bash -lc '").removesuffix("'").replace("$$", "$")
    commands = tmp_path / "bin"
    commands.mkdir()
    git = commands / "git"
    git.write_text("#!/bin/bash\nprintf '%s\\n' '" + "a" * 40 + "'\n")
    git.chmod(0o700)
    python = commands / "python"
    python.write_text('#!/bin/bash\nprintf "%s\\n" "$@" > "$CAPTURE_ARGUMENTS"\n')
    python.chmod(0o700)
    registry = tmp_path / "registry.json"
    registry.write_text("{}")
    selected = tmp_path / "selected-queue"
    selected.mkdir()
    capture = tmp_path / "args.txt"
    environment = {
        **os.environ, "PATH": str(commands) + os.pathsep + os.environ["PATH"],
        "CAPTURE_ARGUMENTS": str(capture),
        "BLUEPRINT_TASK_EVALUATION_CONTROL_PLANE_REPO": str(tmp_path),
        "BLUEPRINT_TASK_EVALUATION_CONTROL_PLANE_PYTHON": str(python),
        "BLUEPRINT_NATIVE_G1_TEAM_CAMPAIGN_REGISTRY_PATH": str(registry),
        "BLUEPRINT_NATIVE_G1_TEAM_CAMPAIGN_QUEUE_ROOT": str(tmp_path / "absent-builtins"),
        "BLUEPRINT_NATIVE_G1_TEAM_CAMPAIGN_WORK_ROOT": str(tmp_path / "builtin-work"),
        "BLUEPRINT_NATIVE_G1_TEAM_CAMPAIGN_EXECUTE": "true",
        "BLUEPRINT_NATIVE_G1_TEAM_POLICY_QUEUE_ROOT": str(selected),
        "BLUEPRINT_NATIVE_G1_TEAM_POLICY_WORK_ROOT": str(tmp_path / "selected-work"),
        "BLUEPRINT_NATIVE_G1_TEAM_POLICY_APPROVAL_ROOT": str(tmp_path / "operator-approvals"),
        "BLUEPRINT_NATIVE_G1_TEAM_POLICY_CREDENTIAL_REGISTRY": str(tmp_path / "private/registry.json"),
        "BLUEPRINT_NATIVE_G1_TEAM_POLICY_SONIC_ASSET_DIR": str(tmp_path / "sonic"),
        "BLUEPRINT_NATIVE_G1_TEAM_POLICY_TRUSTED_CLIENT": "blueprint-webapp",
    }
    completed = subprocess.run(["bash", "-c", shell], env=environment, capture_output=True, timeout=10)
    assert completed.returncode == 0
    assert capture.is_file(), "selected-only queue was silently skipped by installed shell"
    arguments = capture.read_text().splitlines()
    assert arguments[arguments.index("--selected-queue-root") + 1] == str(selected)
    assert arguments[arguments.index("--selected-approval-root") + 1] == environment["BLUEPRINT_NATIVE_G1_TEAM_POLICY_APPROVAL_ROOT"]
    assert arguments[arguments.index("--selected-trusted-client") + 1] == "blueprint-webapp"
    assert "--execute" in arguments
