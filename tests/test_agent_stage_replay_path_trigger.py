"""The offline replay worker runs when a replay is requested, not every five seconds."""
from __future__ import annotations

import configparser
from pathlib import Path

import scripts.deploy_control_plane_commit as deploy
from blueprint_pipeline.agent_execution import stage_recovery

ROOT = Path(__file__).resolve().parents[1]
SYSTEMD = ROOT / "deploy" / "systemd"
WAKE = "/var/lib/blueprint/pipeline-control-plane/agent-execution/stage-replay.wake"


def unit(name: str) -> configparser.ConfigParser:
    parser = configparser.ConfigParser(strict=False, interpolation=None)
    parser.optionxform = str
    parser.read_string((SYSTEMD / name).read_text(encoding="utf-8"))
    return parser


def test_path_unit_watches_the_wake_file_the_agent_worker_writes() -> None:
    path = unit("blueprint-agent-stage-replay.path")
    assert path["Path"]["PathChanged"] == WAKE
    assert path["Path"]["Unit"] == "blueprint-agent-stage-replay.service"
    assert "PathExistsGlob" not in path["Path"] and "DirectoryNotEmpty" not in path["Path"], (
        "requests are never removed, so an existence trigger would loop like the old poll")
    assert path["Unit"]["ConditionPathExists"] == (
        "!/var/lib/blueprint-operator-door/requests/holds/blueprint-agent-stage-replay.path.json")
    agent = (SYSTEMD / "blueprint-agent-execution.service").read_text(encoding="utf-8")
    assert f"Environment={stage_recovery.WAKE_PATH_ENV}={WAKE}" in agent
    # The writer's sandbox can write the wake file; the replay service is untouched.
    assert "ReadWritePaths=/var/lib/blueprint/pipeline-control-plane/agent-execution\n" in agent
    assert WAKE.startswith("/var/lib/blueprint/pipeline-control-plane/agent-execution/")


def test_timer_is_only_a_fallback() -> None:
    timer = unit("blueprint-agent-stage-replay.timer")["Timer"]
    assert timer["OnUnitInactiveSec"] == "5min"
    assert timer["Unit"] == "blueprint-agent-stage-replay.service"


def test_deploy_and_volume_move_carry_the_path_unit_with_timer_authority() -> None:
    name = "blueprint-agent-stage-replay.path"
    assert name in deploy.DEFAULT_DEPLOYED_SYSTEMD_UNITS
    assert name in deploy.DEFAULT_ALWAYS_ARM_TIMER_UNITS
    assert name not in deploy.DEFAULT_ALWAYS_ARM_PATH_UNITS
    mover = (ROOT / "deploy" / "host" / "mount_work_volume.sh").read_text(encoding="utf-8")
    assert f"  {name}\n" in mover
