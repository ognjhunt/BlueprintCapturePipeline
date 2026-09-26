"""The signed G1 team queue must survive a clean controller deployment."""

from __future__ import annotations

from pathlib import Path

import scripts.deploy_control_plane_commit as deploy


ROOT = Path(__file__).resolve().parents[1]
SERVICE = "blueprint-native-g1-team-campaign-dispatcher.service"
TIMER = "blueprint-native-g1-team-campaign-dispatcher.timer"


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
    assert "BLUEPRINT_NATIVE_G1_TEAM_CAMPAIGN_EXECUTE=false" in service
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
