from __future__ import annotations

from pathlib import Path

import scripts.deploy_control_plane_commit as deploy


ROOT = Path(__file__).resolve().parents[1]


def text(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_progression_wakes_on_compilation_results_and_keeps_its_timer() -> None:
    """A compiled no-spend canary must not wait for the two-minute timer.

    Production measured the timer as the only non-event-driven hop between
    Website submission and the paid boundary: every other stage is woken by a
    ``.path`` unit.  The watcher targets the compilation worker's sealed
    results directory (never a pending queue it does not own) and the timer
    stays as the fallback for results that land during a running tick.
    """

    path = text(
        "deploy/systemd/blueprint-task-evaluation-configured-controls-progression.path"
    )
    timer = text(
        "deploy/systemd/blueprint-task-evaluation-configured-controls-progression.timer"
    )
    assert (
        "PathChanged=/var/lib/blueprint/pipeline-control-plane/"
        "task-evaluation-episode-compilations/results"
    ) in path
    assert "PathExistsGlob=" not in path, "persistent results would re-trigger forever"
    assert "Unit=blueprint-task-evaluation-configured-controls-progression.service" in path
    assert "pending" not in path
    assert "OnUnitInactiveSec=2min" in timer


def test_installer_and_deploy_carry_the_progression_path_with_timer_authority() -> None:
    installer = text("scripts/install_live_pipeline_control_plane.sh")
    for unit in (
        "blueprint-task-evaluation-configured-controls-progression.timer",
        "blueprint-task-evaluation-configured-controls-progression.path",
    ):
        assert f"deploy/systemd/{unit}" in installer
        assert f"${{SYSTEMD_DIR}}/{unit}" in installer
        assert f"systemctl enable --now {unit}" in installer
        assert unit in deploy.DEFAULT_DEPLOYED_SYSTEMD_UNITS
        assert unit in deploy.DEFAULT_ALWAYS_ARM_TIMER_UNITS
        assert unit not in deploy.DEFAULT_ALWAYS_ARM_PATH_UNITS
