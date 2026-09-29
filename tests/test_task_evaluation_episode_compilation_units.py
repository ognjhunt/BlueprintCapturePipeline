# Covers (for impacted-test selection):
#   src/blueprint_pipeline/task_evaluation_episode_compilation_remote.py
#   src/blueprint_pipeline/task_evaluation_episode_compilation_worker.py
#   src/blueprint_pipeline/task_evaluation_episode_compilation_collector.py
#   src/blueprint_pipeline/task_evaluation_production_chain_preflight.py
#   deploy/systemd/blueprint-task-evaluation-episode-compilation.service
#   deploy/systemd/blueprint-task-evaluation-episode-compilation.path
#   deploy/systemd/blueprint-task-evaluation-episode-compilation-remote.service
#   deploy/systemd/blueprint-task-evaluation-episode-compilation-remote.timer
#   deploy/systemd/blueprint-task-evaluation-episode-compilation-remote.path
"""ADP-009D/day-28, plan 14 PR 4: the no-spend unit owns pending/ in every mode; the paid unit never compiles."""

from __future__ import annotations

import ast
import json
import shutil
from pathlib import Path

import pytest

from blueprint_pipeline import task_evaluation_episode_compilation_remote as remote
from blueprint_pipeline.task_evaluation_episode_compilation_worker import process_episode_compilation_queue
from tests.remote_cpu_allocator_fakes import IMAGE, remote_cpu_config
from tests.remote_episode_compilation_support import HOST_RECORD, Host, nurec_usdz, stage_compile, tree_snapshot

ROOT = Path(__file__).resolve().parents[1]
SYSTEMD = ROOT / "deploy" / "systemd"
NO_SPEND = "blueprint-task-evaluation-episode-compilation"
PAID = "blueprint-task-evaluation-episode-compilation-remote"
JOBS = "/var/lib/blueprint/pipeline-control-plane/remote-cpu-jobs"
AUTHORIZATIONS = "/var/lib/blueprint/spend-authority/authorizations"
CREDENTIAL = "LoadCredential=remote-cpu-dispatcher:/etc/blueprint/credentials/remote-cpu-dispatcher.json"


def _unit(name: str) -> str:
    return (SYSTEMD / name).read_text(encoding="utf-8")


def _paths(text: str, directive: str) -> list[str]:
    return [path for line in text.splitlines() if line.startswith(f"{directive}=")
            for path in line.split("=", 1)[1].split()]


def _run(host: Host, mode: str | None, compiler, *, config: dict | None = None, max_messages: int = 8) -> dict:
    return remote.run_no_spend_unit(
        queue_root=host.queue, input_root=host.inputs, output_root=host.outputs, source_commit="a" * 40,
        max_messages=max_messages, jobs_root=host.jobs, environ={} if mode is None else {remote.EXECUTION_ENV: mode},
        episode_compiler=compiler, filesystem_root=host.fs, cache_root=host.cache,
        config=config or remote_cpu_config(), host_environment=HOST_RECORD, now=lambda: 1_000.0)


def _stage_pair(host: Host) -> tuple[str, str]:
    """One row eligible to run remotely, and one whose NuRec appearance needs the host's transcoder."""

    _, eligible = stage_compile(host, label="eligible")
    _, transcode = stage_compile(host, label="transcode", appearance=nurec_usdz(16384),
                                 appearance_name="appearance.usdz")
    return eligible, transcode


@pytest.mark.parametrize("mode", ["host", "cloud_run_shadow", "cloud_run"])
def test_no_spend_unit_owns_pending_and_empties_it_in_every_mode(tmp_path: Path, monkeypatch, mode: str) -> None:
    from tests.remote_cpu_worker_stages import install_compile_stand_ins

    monkeypatch.setattr(remote, "MAX_INLINE_NUREC_BYTES", 8192)
    host = Host(tmp_path)
    host.record_worker_environment()
    for index in range(3):  # cloud_run routes a class only after three shadow passes
        remote.record_shadow_parity(host.jobs, {
            "closure_class": "not_applicable", "attempt_id": f"rcj-ec-{'0' * 24}-a1-{index:032x}",
            "queue_row": {"queue": remote.QUEUE, "name": f"row-{index}.json", "envelope_digest": "sha256:" + "1" * 64},
            "image": IMAGE, "host_environment_digest": HOST_RECORD["environment_digest"],
            "worker_environment_digest": HOST_RECORD["environment_digest"], "cpu_class": HOST_RECORD["cpu_class"],
            "parity": "passed", "mismatches": [], "compared_at_epoch": float(index)})
    eligible, transcode = _stage_pair(host)
    run = _run(host, mode, install_compile_stand_ins(monkeypatch.setattr))
    # Every run empties pending/, so the unit's PathExistsGlob can never loop.
    assert list((host.queue / "pending").iterdir()) == []
    # Rows only ever sit in the four states the identity check, leases, retention and the census know.
    assert sorted(path.name for path in host.queue.iterdir()) == ["blocked", "completed", "pending", "processing",
                                                                 "results"]
    handoff = remote.marker_path(host.jobs, "authoritative", eligible)
    shadow = remote.marker_path(host.jobs, "shadow", eligible)
    assert not any(path.is_relative_to(host.queue) for path in (handoff, shadow))
    assert (host.queue / "completed" / transcode).is_file()  # ineligible: compiled on the host in every mode
    assert not remote.marker_path(host.jobs, "authoritative", transcode).exists()
    if mode == "cloud_run":
        assert (host.queue / "processing" / eligible).is_file() and handoff.is_file() and not shadow.exists()
        assert not (host.queue / "results" / eligible).exists()
        assert run["host_decisions"] == {transcode: "remote_ineligible:particlefield_transcode_required"}
    else:
        assert (host.queue / "completed" / eligible).is_file() and not handoff.exists()
        assert shadow.is_file() == (mode == "cloud_run_shadow")
    assert remote.read_marker(handoff if mode == "cloud_run" else shadow) is not None or mode == "host"


def test_fallback_rows_are_compiled_by_the_no_spend_unit(tmp_path: Path, monkeypatch) -> None:
    from tests.remote_cpu_worker_stages import install_compile_stand_ins

    host = Host(tmp_path)
    host.record_worker_environment()
    envelope, name = stage_compile(host)
    claimed = host.claim(name)
    plan = remote.plan_remote_compilation(claimed, inputs=host.inputs, outputs=host.outputs, source_commit="a" * 40,
                                          config=remote_cpu_config(), jobs_root=host.jobs, filesystem_root=host.fs,
                                          cache_root=host.cache, host_environment=HOST_RECORD,
                                          require_shadow_gate=False)
    remote.write_handoff(host.jobs, plan, mode="authoritative", now=1.0)
    remote.write_fallback(host.jobs, plan.queue_row, reason="remote_cpu_receipt_missing", attempts=2, now=2.0)
    # The rollback run (host mode) still compiles fallback rows: they are already claimed, in processing/.
    run = _run(host, "host", install_compile_stand_ins(monkeypatch.setattr))
    assert run["fallback_results"][0]["status"] == "compiled_for_production_launch"
    assert (host.queue / "completed" / name).is_file()
    assert not remote.marker_path(host.jobs, "fallback", name).exists()


def test_paid_unit_has_no_exists_glob_and_never_compiles() -> None:
    path, timer, service = _unit(f"{PAID}.path"), _unit(f"{PAID}.timer"), _unit(f"{PAID}.service")
    # A timer and PathChanged= on the hand-off and shadow directories: a skipped run can never re-trigger itself.
    assert "PathExistsGlob" not in path
    assert set(_paths(path, "PathChanged")) == {f"{JOBS}/handoffs/episode_compilation", f"{JOBS}/shadow/episode_compilation"}
    assert f"Unit={PAID}.service" in path and f"Unit={PAID}.service" in timer
    assert "OnUnitInactiveSec=60s" in timer
    assert "-m blueprint_pipeline.task_evaluation_episode_compilation_collector run" in service
    assert "-m blueprint_pipeline.task_evaluation_episode_compilation_collector should-run" in service
    assert "task_evaluation_episode_compilation_worker" not in service
    # The collector reaches no compiler: it only dispatches, follows, promotes and lands.
    module = ROOT / "src/blueprint_pipeline/task_evaluation_episode_compilation_collector.py"
    called = {node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
              for node in ast.walk(ast.parse(module.read_text(encoding="utf-8"))) if isinstance(node, ast.Call)}
    assert not called & {"compile_claimed_envelope", "compile_native_arena_episode", "process_episode_compilation_queue",
                         "run_no_spend_unit", "run_episode_compilation_in_worker"}
    # The no-spend unit keeps pending/ and also wakes on fallback markers, which have no glob.
    no_spend = _unit(f"{NO_SPEND}.path")
    assert set(_paths(no_spend, "PathChanged")) == {
        "/var/lib/blueprint/pipeline-control-plane/task-evaluation-episode-compilations/pending",
        f"{JOBS}/fallback/episode_compilation"}
    assert _paths(no_spend, "PathExistsGlob") == [
        "/var/lib/blueprint/pipeline-control-plane/task-evaluation-episode-compilations/pending/*.json"]


def test_dispatcher_credential_is_loaded_only_by_the_paid_unit() -> None:
    holders = [unit.name for unit in sorted(SYSTEMD.iterdir()) if unit.is_file()
               and "remote-cpu-dispatcher" in unit.read_text(encoding="utf-8")]
    assert holders == [f"{PAID}.service"]
    service = _unit(f"{PAID}.service")
    assert service.count(CREDENTIAL) == 1 and service.count("LoadCredential") == 1
    assert "LoadCredential" not in _unit(f"{NO_SPEND}.service")
    # Loaded by systemd from a root-only file: never an environment value, never installed by the repo.
    assert "remote-cpu-dispatcher.json" not in _unit(f"{PAID}.service").replace(CREDENTIAL, "")
    for text in (ROOT / "scripts" / "install_live_pipeline_control_plane.sh",
                 SYSTEMD / "pipeline-control-plane.env.example"):
        assert "remote-cpu-dispatcher.json" not in text.read_text(encoding="utf-8")


def test_paid_unit_cannot_write_its_own_standing_authority() -> None:
    service = _unit(f"{PAID}.service")
    writable = _paths(service, "ReadWritePaths")
    assert writable and "/var/lib/blueprint" not in writable
    assert not any(AUTHORIZATIONS == path or AUTHORIZATIONS.startswith(path.rstrip("/") + "/") for path in writable)
    assert AUTHORIZATIONS in _paths(service, "ReadOnlyPaths")
    assert {"/var/lib/blueprint/spend-authority/consumed", "/var/lib/blueprint/spend-authority/remote-cpu-settled",
            JOBS, "/var/lib/blueprint/task-evaluation-inputs/compiled-episodes",
            "/var/lib/blueprint/pipeline-control-plane/task-evaluation-episode-compilations",
            "/var/lib/blueprint/pipeline-control-plane/disk-reservations",
            "/var/lib/blueprint/pipeline-control-plane/storage-pins"} == set(writable)


def test_host_mode_is_byte_identical_to_today(tmp_path: Path, monkeypatch) -> None:
    """The flag's default is today's path: the same results, rows, outputs and run record, byte for byte."""

    from tests.remote_cpu_worker_stages import install_compile_stand_ins

    compiler = install_compile_stand_ins(monkeypatch.setattr)
    root = tmp_path / "host"

    def observe(host: Host) -> dict:
        return {"queue": {path.relative_to(host.queue).as_posix(): path.read_bytes()
                          for path in sorted(host.queue.rglob("*")) if path.is_file()},
                "outputs": tree_snapshot(host.outputs)}

    host = Host(root)
    _stage_pair(host)
    today = process_episode_compilation_queue(queue_root=host.queue, input_root=host.inputs, output_root=host.outputs,
                                              source_commit="a" * 40, episode_compiler=compiler, max_messages=8)
    expected = observe(host)
    for mode in (None, "host", "no_such_mode"):
        shutil.rmtree(root)
        host = Host(root)
        _stage_pair(host)
        assert _run(host, mode, compiler) == today, mode
        assert observe(host) == expected, mode
        assert not (host.jobs / "handoffs").exists() and not (host.jobs / "shadow").exists()
    assert remote.execution_mode({remote.EXECUTION_ENV: "no_such_mode"}) == (
        "host", ["episode_compilation_execution_mode_invalid"])


def test_worker_entry_point_runs_the_no_spend_unit(tmp_path: Path, monkeypatch, capsys) -> None:
    """The unit's ExecStart is unchanged; its entry point now runs the no-spend unit, which in host mode
    prints exactly today's run record."""

    from blueprint_pipeline import task_evaluation_episode_compilation_worker as worker

    seen: dict = {}

    def recorded(**kwargs):
        seen.update(kwargs)
        return {"schema_version": "task_evaluation_episode_compilation_queue_run.v1", "status": "idle"}

    monkeypatch.setattr(remote, "run_no_spend_unit", recorded)
    monkeypatch.setenv(remote.JOBS_ROOT_ENV, str(tmp_path / "jobs"))
    assert worker.main(["--queue-root", str(tmp_path / "q"), "--input-root", str(tmp_path / "i"), "--output-root",
                        str(tmp_path / "o"), "--source-commit", "a" * 40, "--max-messages", "4"]) == 0
    assert json.loads(capsys.readouterr().out) == recorded()
    assert (seen["jobs_root"], seen["max_messages"]) == (str(tmp_path / "jobs"), 4)


def test_chain_preflight_reports_invalid_modes_and_remote_cpu_image_drift(tmp_path: Path) -> None:
    from blueprint_pipeline import task_evaluation_production_chain_preflight as preflight

    host = Host(tmp_path)
    config = tmp_path / "remote-cpu-workers.json"
    config.write_text(json.dumps(remote_cpu_config()), encoding="utf-8")
    config.chmod(0o640)
    environment = {remote.JOBS_ROOT_ENV: str(host.jobs), remote.CONFIG_ENV: str(config)}
    units = {preflight.EPISODE_COMPILATION_UNIT: {"effective_environment": {**environment,
                                                                            remote.EXECUTION_ENV: "cloudrun"}}}
    codes = {finding["code"]: finding for finding in preflight.remote_execution_checks(units)}
    assert codes["episode_compilation_execution_mode_invalid"]["severity"] == "warning"
    # No probe recorded for the configured image: every dispatch would be refused, so it is drift, as a warning.
    assert codes["remote_cpu_image_drift"]["severity"] == "warning"
    host.record_worker_environment()
    units[preflight.EPISODE_COMPILATION_UNIT]["effective_environment"][remote.EXECUTION_ENV] = "cloud_run_shadow"
    assert preflight.remote_execution_checks(units) == []
    remote.record_job_image(host.jobs, job_image=IMAGE.replace("d" * 64, "e" * 64), config_image=IMAGE, now=1.0)
    assert [finding["code"] for finding in preflight.remote_execution_checks(units)] == ["remote_cpu_image_drift"]
    assert preflight.PAID_EPISODE_COMPILATION_UNIT in preflight.CHAIN_UNITS
