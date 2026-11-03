# Covers (for impacted-test selection):
#   deploy/systemd/blueprint-team-scene-packet-remote.service
#   deploy/systemd/blueprint-team-scene-packet-remote.path
#   deploy/systemd/blueprint-team-scene-packet-remote.timer
#   deploy/systemd/blueprint-team-scene-packet.path
#   deploy/systemd/blueprint-team-scene-packet.service
#   src/blueprint_pipeline/team_scene_packet_remote_condition.py
#   scripts/deploy_control_plane_commit.py
"""The team stage's paid unit, its ExecCondition and its deploy wiring, mirroring remote episode compilation."""

from __future__ import annotations

import json
from pathlib import Path

import scripts.deploy_control_plane_commit as deploy
from blueprint_pipeline import remote_cpu_job_contract as contract
from blueprint_pipeline import team_scene_packet_remote as lane
from blueprint_pipeline import team_scene_packet_remote_condition as condition
from blueprint_pipeline.production_runtime_env_guard import CONTROL_PLANE_ENTRYPOINTS
from tests.remote_team_scene_packet_support import team_config

ROOT = Path(__file__).resolve().parents[1]
SYSTEMD = ROOT / "deploy" / "systemd"
PAID, NO_SPEND = "blueprint-team-scene-packet-remote", "blueprint-team-scene-packet"
JOBS = "/var/lib/blueprint/pipeline-control-plane/remote-cpu-jobs"


def _unit(name: str) -> str:
    return (SYSTEMD / name).read_text(encoding="utf-8")


def _paths(text: str, key: str) -> list[str]:
    return [path.lstrip("-") for line in text.splitlines() if line.startswith(f"{key}=")
            for path in line.split("=", 1)[1].split()]


def test_the_conditions_constants_are_the_lanes_and_the_contracts() -> None:
    assert condition.EXECUTION_ENV == lane.EXECUTION_ENV and condition.STAGE == lane.STAGE
    assert condition.REMOTE_MODES == lane.REMOTE_MODES
    assert condition.JOB_PREFIX == f"rcj-{contract.STAGES[lane.STAGE]['abbreviation']}-"
    assert contract.job_id_for(lane.STAGE, "x-" + "0" * 64 + ".json").startswith(condition.JOB_PREFIX)
    assert set(condition.MARKER_DIRECTORIES) == {lane.MARKERS["authoritative"], lane.MARKERS["shadow"]}


def test_the_condition_runs_only_for_this_stages_work_or_a_remote_mode(tmp_path: Path) -> None:
    jobs, config = tmp_path / "jobs", tmp_path / "remote-cpu-workers.json"
    environ = {condition.CONFIG_ENV: str(config)}
    assert condition.should_run(jobs, environ) is False
    # The production config today names only episode compilation: auto stays host and the unit skips.
    episode_only = team_config()
    episode_only["stages"].pop("team_scene_packet")
    from tests.remote_cpu_allocator_fakes import seal

    config.write_text(json.dumps(seal(episode_only, "config_digest")), encoding="utf-8")
    config.chmod(0o640)
    assert condition.should_run(jobs, environ) is False
    (jobs / "live").mkdir(parents=True)
    (jobs / "live" / ("rcj-ec-" + "0" * 24)).write_text("", encoding="utf-8")
    assert condition.should_run(jobs, environ) is False  # another stage's live lease is that stage's unit's
    (jobs / "live" / ("rcj-tp-" + "0" * 24)).write_text("", encoding="utf-8")
    assert condition.should_run(jobs, environ) is True
    (jobs / "live" / ("rcj-tp-" + "0" * 24)).unlink()
    (jobs / "handoffs" / "team_scene_packet").mkdir(parents=True)
    (jobs / "handoffs" / "team_scene_packet" / ".partial.tmp").write_text("", encoding="utf-8")
    assert condition.should_run(jobs, environ) is False
    (jobs / "handoffs" / "team_scene_packet" / "row.json").write_text("{}", encoding="utf-8")
    assert condition.should_run(jobs, environ) is True
    (jobs / "handoffs" / "team_scene_packet" / "row.json").unlink()
    config.write_text(json.dumps(team_config()), encoding="utf-8")
    assert condition.should_run(jobs, environ) is True
    assert condition.should_run(jobs, {**environ, condition.EXECUTION_ENV: "host"}) is False
    assert condition.should_run(jobs, {condition.EXECUTION_ENV: "cloud_run_shadow"}) is True
    assert condition.should_run(jobs, {condition.EXECUTION_ENV: "cloudrun"}) is False


def test_the_paid_unit_holds_the_dispatcher_key_narrow_writes_and_a_read_only_authority() -> None:
    service = _unit(f"{PAID}.service")
    assert service.count("LoadCredential=remote-cpu-dispatcher:/etc/blueprint/credentials/remote-cpu-dispatcher.json") == 1
    writable = _paths(service, "ReadWritePaths")
    assert set(writable) == {JOBS, "/var/lib/blueprint/task-evaluation-inputs/team-scene-halves",
                             "/var/lib/blueprint/spend-authority/consumed",
                             "/var/lib/blueprint/spend-authority/remote-cpu-settled"}
    assert "/var/lib/blueprint/spend-authority/authorizations" in _paths(service, "ReadOnlyPaths")
    assert "-m blueprint_pipeline.team_scene_packet_remote_condition'" in service
    assert "-m blueprint_pipeline.team_scene_packet_collector run --source-commit" in service
    assert 'test -z "$$(git status --porcelain --untracked-files=no)"' in service
    for control in ("User=blueprint", "NoNewPrivileges=true", "ProtectSystem=strict", "CapabilityBoundingSet=\n",
                    "MemoryMax=2G", "UMask=0077"):
        assert control in service
    path, timer = _unit(f"{PAID}.path"), _unit(f"{PAID}.timer")
    assert f"PathChanged={JOBS}/handoffs/team_scene_packet\n" in path
    assert f"PathChanged={JOBS}/shadow/team_scene_packet\n" in path and "PathExistsGlob" not in path
    assert f"Unit={PAID}.service" in path and f"Unit={PAID}.service" in timer and "OnUnitInactiveSec=60s" in timer
    for unit in (path, timer):
        assert "ConditionPathExists=!/var/lib/blueprint-operator-door/requests/holds/" in unit


def test_the_no_spend_worker_holds_no_key_and_wakes_for_what_the_paid_unit_returns() -> None:
    service, path = _unit(f"{NO_SPEND}.service"), _unit(f"{NO_SPEND}.path")
    assert "LoadCredential" not in service and "remote-cpu-dispatcher" not in service
    assert f"Environment={lane.JOBS_ROOT_ENV}={JOBS}\n" in service
    assert f"Environment={lane.CONFIG_ENV}=/etc/blueprint/remote-cpu-workers.json\n" in service
    for kind in ("collected", "fallback"):
        assert f"PathExistsGlob={JOBS}/{kind}/team_scene_packet/*.json\n" in path
    # The paid unit creates executions only through the canonical subcommand, the episode collector's own runner.
    collector = (ROOT / "src/blueprint_pipeline/team_scene_packet_collector.py").read_text(encoding="utf-8")
    assert "allocate=shared.subprocess_allocate" in collector and "subprocess" not in collector.replace(
        "shared.subprocess_allocate", "")


def test_every_deploy_installs_and_arms_the_paid_trio_as_authority_gated() -> None:
    assert {f"{PAID}.service", f"{PAID}.timer", f"{PAID}.path"} <= set(deploy.DEFAULT_DEPLOYED_SYSTEMD_UNITS)
    assert f"{PAID}.path" in deploy.DEFAULT_ALWAYS_ARM_AUTHORITY_GATED_PATH_UNITS
    assert f"{PAID}.path" not in deploy.DEFAULT_ALWAYS_ARM_PATH_UNITS
    assert f"{PAID}.timer" in deploy.DEFAULT_ALWAYS_ARM_TIMER_UNITS
    directories = set(deploy.DEFAULT_EPISODE_COMPILATION_RUNTIME_DIRECTORIES)
    assert {f"{JOBS}/{kind}/team_scene_packet" for kind in ("handoffs", "shadow", "fallback", "collected",
                                                             "envelopes")} <= directories
    assert len(deploy.DEFAULT_EPISODE_COMPILATION_RUNTIME_DIRECTORIES) == len(directories)
    for module in ("team_scene_packet_collector", "team_scene_packet_remote_condition"):
        assert f"blueprint_pipeline.{module}" in CONTROL_PLANE_ENTRYPOINTS
    mount = (ROOT / "deploy/host/mount_work_volume.sh").read_text(encoding="utf-8")
    for unit in (f"{PAID}.path", f"{PAID}.timer", f"{PAID}.service"):
        assert f"  {unit}\n" in mount
