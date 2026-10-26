# Covers (for impacted-test selection):
#   deploy/systemd/blueprint-team-scene-packet.path
#   deploy/systemd/blueprint-team-scene-packet.service
#   deploy/systemd/blueprint-task-evaluation-scene-progression.service
#   deploy/host/mount_work_volume.sh
"""The team scene half worker ships as a no-spend path and service pair that every deploy arms (beta L5)."""

from __future__ import annotations

from pathlib import Path

import scripts.deploy_control_plane_commit as deploy
from blueprint_pipeline import team_scene_packet_worker as worker
from blueprint_pipeline.control_plane_storage_roots import classify_path
from blueprint_pipeline.production_runtime_env_guard import CONTROL_PLANE_ENTRYPOINTS

ROOT = Path(__file__).resolve().parents[1]
PATH_UNIT, SERVICE = "blueprint-team-scene-packet.path", "blueprint-team-scene-packet.service"
QUEUE = "/var/lib/blueprint/pipeline-control-plane/team-scene-packets"


def _text(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def test_the_path_unit_wakes_a_hardened_worker_that_holds_no_provider_path() -> None:
    path = _text(f"deploy/systemd/{PATH_UNIT}")
    service = _text(f"deploy/systemd/{SERVICE}")
    assert f"PathExistsGlob={QUEUE}/pending/*.json" in path and f"Unit={SERVICE}" in path
    assert f"ConditionPathExists=!/var/lib/blueprint-operator-door/requests/holds/{PATH_UNIT}.json" in path
    assert "-m blueprint_pipeline.team_scene_packet_worker --max-messages 1" in service
    assert f"Environment={worker.QUEUE_ROOT_ENV}={QUEUE}" in service
    for control in ("User=blueprint", "NoNewPrivileges=true", "ProtectSystem=strict", "CapabilityBoundingSet=\n"):
        assert control in service
    for paid in ("paid_resource_allocator", "provider_adapter", "VAST_API_KEY", "vast_"):
        assert paid not in service
    module = _text("src/blueprint_pipeline/team_scene_packet_worker.py")
    assert "paid_resource_allocator" not in module and "vast" not in module.lower()
    # The deploy creates the watched queue (the unit's own sandbox paths) before it arms the watcher.
    sandbox = {entry[0] for entry in deploy._unit_sandbox_entries(service)}
    assert {QUEUE, f"{QUEUE}/pending"} <= sandbox
    assert classify_path(f"{QUEUE}/pending").storage_class == "work"


def test_scene_progression_enqueues_where_the_worker_reads() -> None:
    progression = _text("deploy/systemd/blueprint-task-evaluation-scene-progression.service")
    assert f"Environment={worker.QUEUE_ROOT_ENV}={QUEUE}\n" in progression
    assert "ReadWritePaths=/var/lib/blueprint/pipeline-control-plane " in progression


def test_every_deploy_installs_and_arms_the_pair_as_no_spend() -> None:
    assert {SERVICE, PATH_UNIT} <= set(deploy.DEFAULT_DEPLOYED_SYSTEMD_UNITS)
    assert PATH_UNIT in deploy.DEFAULT_ALWAYS_ARM_PATH_UNITS
    assert PATH_UNIT not in deploy.DEFAULT_ALWAYS_ARM_AUTHORITY_GATED_PATH_UNITS
    assert PATH_UNIT not in deploy.DEFAULT_ALWAYS_ARM_TIMER_UNITS
    assert "blueprint_pipeline.team_scene_packet_worker" in CONTROL_PLANE_ENTRYPOINTS
    mount = _text("deploy/host/mount_work_volume.sh")
    assert f"  {PATH_UNIT}\n" in mount and f"  {SERVICE}\n" in mount
