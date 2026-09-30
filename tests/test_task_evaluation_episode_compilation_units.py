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
    # The config file the units name, under this host's own root and absent: an unset mode is auto, which this
    # host's missing config makes host, whatever the machine running the tests has at the real path.
    environ = {remote.CONFIG_ENV: str(host.local(remote.DEFAULT_CONFIG_PATH))}
    return remote.run_no_spend_unit(
        queue_root=host.queue, input_root=host.inputs, output_root=host.outputs, source_commit="a" * 40,
        max_messages=max_messages, jobs_root=host.jobs,
        environ=environ if mode is None else {**environ, remote.EXECUTION_ENV: mode},
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


def test_inline_nurec_rows_stay_on_the_host_in_shadow_mode_too(tmp_path: Path, monkeypatch) -> None:
    """Review I4: the inline NuRec conversion is not yet deterministic (usd-convert-gsplat writes a random temp
    PLY path into the layer comment), so its shadow comparison could only fail while spending: no marker."""

    from tests.remote_cpu_worker_stages import install_compile_stand_ins

    monkeypatch.setattr(remote, "MAX_INLINE_NUREC_BYTES", 8192)
    host = Host(tmp_path)
    host.record_worker_environment()
    _, name = stage_compile(host, label="inline", appearance=nurec_usdz(4096), appearance_name="appearance.usdz")
    run = _run(host, "cloud_run_shadow", install_compile_stand_ins(monkeypatch.setattr))
    assert run["host_decisions"] == {name: "remote_ineligible:inline_nurec_conversion_nondeterministic"}
    assert run["shadowed"] == [] and not remote.marker_path(host.jobs, "shadow", name).exists()
    assert (host.queue / "completed" / name).is_file()


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
    assert "-m blueprint_pipeline.task_evaluation_episode_compilation_remote_condition" in service
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


def test_cloud_run_before_census_support_runs_as_host_and_is_reported(tmp_path: Path, monkeypatch) -> None:
    """Plan 14 task 4.8: until the owner census accepts remote-output pointers, ``cloud_run`` runs as ``host``,
    the paid unit finds nothing to dispatch and the chain preflight says why."""

    from blueprint_pipeline import task_evaluation_production_chain_preflight as preflight
    from blueprint_pipeline import task_evaluation_scene_compilation_owner_outputs as census
    from tests.remote_cpu_worker_stages import install_compile_stand_ins

    requested = {remote.EXECUTION_ENV: "cloud_run"}
    assert remote.execution_mode(requested) == ("cloud_run", [])
    monkeypatch.delattr(census, "REMOTE_OUTPUT_POINTER_SCHEMAS")
    assert remote.execution_mode(requested) == (
        "host", ["episode_compilation_cloud_run_requires_census_pointer_support"])

    compiler = install_compile_stand_ins(monkeypatch.setattr)
    host = Host(tmp_path)
    host.record_worker_environment()
    for index in range(3):  # the row's class has its shadow passes: only the census gate keeps it home
        remote.record_shadow_parity(host.jobs, {
            "closure_class": "not_applicable", "attempt_id": f"rcj-ec-{'0' * 24}-a1-{index:032x}",
            "queue_row": {"queue": remote.QUEUE, "name": f"row-{index}.json", "envelope_digest": "sha256:" + "1" * 64},
            "image": IMAGE, "host_environment_digest": HOST_RECORD["environment_digest"],
            "worker_environment_digest": HOST_RECORD["environment_digest"], "cpu_class": HOST_RECORD["cpu_class"],
            "parity": "passed", "mismatches": [], "compared_at_epoch": float(index)})
    _, eligible = stage_compile(host, label="eligible")
    run = _run(host, "cloud_run", compiler)
    # Exactly the host run: the eligible row compiled here, and nothing was handed off.
    assert "mode" not in run and run["processed_count"] == 1
    assert (host.queue / "completed" / eligible).is_file()
    assert not remote.marker_path(host.jobs, "authoritative", eligible).exists()
    # The paid unit's condition reads only the requested mode, so it may start; its effective mode is host, so
    # it finds no hand-off and dispatches nothing.
    assert not remote.markers(host.jobs, "authoritative") and not remote.markers(host.jobs, "shadow")
    environment = {**requested, remote.JOBS_ROOT_ENV: str(host.jobs), remote.CONFIG_ENV: str(tmp_path / "absent.json")}
    findings = preflight.remote_execution_checks({preflight.EPISODE_COMPILATION_UNIT: {
        "effective_environment": environment}})
    assert [(finding["severity"], finding["code"]) for finding in findings] == [
        ("warning", "episode_compilation_cloud_run_requires_census_pointer_support")]


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


def test_a_skipped_exec_condition_is_not_a_failed_unit() -> None:
    """Review minor: in host mode the paid unit's ExecCondition skips every run, which systemd reports as
    ``Result=exec-condition``; that is the unit working as designed, never ``unit_failed_state``."""

    from blueprint_pipeline import task_evaluation_production_chain_preflight as preflight

    def unit(result: str, state: str = "inactive") -> dict:
        return {"properties": {"ActiveState": [state], "Result": [result], "LoadState": ["loaded"], "TriggeredBy": [""],
                               "ExecCondition": ["{ path=/usr/bin/python3 ; argv[]=python3 -m x should-run }"]}}

    assert [f["code"] for f in preflight.unit_health_checks({f"{PAID}.service": unit("exec-condition")})] == []
    assert [f["code"] for f in preflight.unit_health_checks({f"{PAID}.service": unit("exit-code", "failed")})] == [
        "unit_failed_state"]


def test_the_paid_units_exec_condition_imports_only_the_standard_library(tmp_path: Path) -> None:
    """Review minor: the ExecCondition runs every 60 s, so it answers from the filesystem with the standard
    library alone; importing the collector costs about 0.8 s of CPU and 75 MB on each check."""

    import sys

    from blueprint_pipeline import task_evaluation_episode_compilation_remote_condition as condition

    service = _unit(f"{PAID}.service")
    [line] = [line for line in service.splitlines() if line.startswith("ExecCondition=")]
    assert "-m blueprint_pipeline.task_evaluation_episode_compilation_remote_condition" in line
    tree = ast.parse(Path(condition.__file__).read_text(encoding="utf-8"))
    imported = {alias.name.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
    imported |= {node.module.split(".")[0] for node in ast.walk(tree)
                 if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module}
    assert not [node for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.level]
    assert imported <= set(sys.stdlib_module_names) | {"__future__"}, imported
    # Its constants are the remote module's own.
    assert (condition.EXECUTION_ENV, condition.JOBS_ROOT_ENV, condition.DEFAULT_JOBS_ROOT, condition.STAGE) == (
        remote.EXECUTION_ENV, remote.JOBS_ROOT_ENV, remote.DEFAULT_JOBS_ROOT, remote.STAGE)
    assert set(condition.REMOTE_MODES) == set(remote.MODES) - {"host"}
    assert condition.MARKER_DIRECTORIES == (remote.MARKERS["authoritative"], remote.MARKERS["shadow"])

    jobs = tmp_path / "jobs"
    unset = {condition.CONFIG_ENV: str(tmp_path / "absent.json")}  # auto, without a config: host
    assert condition.should_run(jobs, unset) is False  # host mode, nothing live: skipped cheaply
    for mode in ("cloud_run", "cloud_run_shadow"):
        assert condition.should_run(jobs, {remote.EXECUTION_ENV: mode}) is True
    assert condition.should_run(jobs, {remote.EXECUTION_ENV: "cloudrun"}) is False  # invalid runs as host
    handoffs = jobs / "handoffs" / "episode_compilation"
    handoffs.mkdir(parents=True)
    (handoffs / ".row.json.0123.tmp").write_text("{}", encoding="utf-8")
    assert condition.should_run(jobs, unset) is False  # a record still being written is not a hand-off
    for directory in (handoffs, jobs / "shadow" / "episode_compilation", jobs / "live"):
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "row.json").write_text("{}", encoding="utf-8")
        assert condition.should_run(jobs, unset) is True
        (directory / "row.json").unlink()


def _handed_back(host: Host, labels: list[str]) -> list[str]:
    names = []
    for label in labels:
        _, name = stage_compile(host, label=label)
        plan = remote.plan_remote_compilation(
            host.claim(name), inputs=host.inputs, outputs=host.outputs, source_commit="a" * 40,
            config=remote_cpu_config(), jobs_root=host.jobs, filesystem_root=host.fs, cache_root=host.cache,
            host_environment=HOST_RECORD, require_shadow_gate=False)
        remote.write_fallback(host.jobs, plan.queue_row, reason="remote_cpu_receipt_missing", attempts=2, now=1.0)
        names.append(name)
    return sorted(names)


def test_fallback_compiles_are_bounded_per_run_by_count_and_time(tmp_path: Path, monkeypatch) -> None:
    """Review minor: one run compiles at most ``max_messages`` handed-back rows, and starts none once its time
    budget (well inside the unit's TimeoutStartSec) is spent, though always at least one; the rest wait."""

    from tests.remote_cpu_worker_stages import install_compile_stand_ins

    compiler = install_compile_stand_ins(monkeypatch.setattr)
    host = Host(tmp_path / "count")
    host.record_worker_environment()
    names = _handed_back(host, ["back-0", "back-1", "back-2"])
    run = _run(host, "host", compiler, max_messages=2)
    assert [row["status"] for row in run["fallback_results"]] == ["compiled_for_production_launch"] * 2
    assert run["fallback_deferred"] == names[2:]
    assert (host.queue / "processing" / names[2]).is_file() and remote.marker_path(host.jobs, "fallback", names[2]).is_file()
    assert remote.FALLBACK_TIME_BUDGET_SECONDS < 15 * 60
    later = _run(host, "host", compiler, max_messages=2)
    assert len(later["fallback_results"]) == 1 and "fallback_deferred" not in later
    assert all((host.queue / "completed" / name).is_file() for name in names)

    monkeypatch.setattr(remote, "FALLBACK_TIME_BUDGET_SECONDS", 0)
    timed = Host(tmp_path / "time")
    timed.record_worker_environment()
    names = _handed_back(timed, ["late-0", "late-1"])
    run = _run(timed, "cloud_run", compiler, max_messages=8)
    assert len(run["fallback_results"]) == 1 and run["fallback_deferred"] == names[1:]


def test_the_no_spend_unit_has_a_timer_backstop_and_deploy_creates_its_marker_directories() -> None:
    """Review minor: on a host set up before PR 4 no deploy created the hand-off and fallback directories, and
    the no-spend unit had no timer, so a first fallback event (or a deferred handed-back row) could wait forever.
    Deploy now creates every marker directory for the service account, and a timer backstops the path unit."""

    import importlib.util

    timer = _unit(f"{NO_SPEND}.timer")
    assert f"Unit={NO_SPEND}.service" in timer and "OnUnitInactiveSec=" in timer and "OnBootSec=" in timer
    spec = importlib.util.spec_from_file_location("deploy_control_plane_commit",
                                                  ROOT / "scripts" / "deploy_control_plane_commit.py")
    deploy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(deploy)
    assert f"{NO_SPEND}.timer" in deploy.DEFAULT_DEPLOYED_SYSTEMD_UNITS
    assert f"{NO_SPEND}.timer" in deploy.DEFAULT_ALWAYS_ARM_TIMER_UNITS
    directories = set(deploy.DEFAULT_EPISODE_COMPILATION_RUNTIME_DIRECTORIES)
    for kind in ("handoffs", "shadow", "fallback", "recovery"):
        assert {f"{JOBS}/{kind}", f"{JOBS}/{kind}/episode_compilation"} <= directories, kind
    assert JOBS in directories
    installer = (ROOT / "scripts" / "install_live_pipeline_control_plane.sh").read_text(encoding="utf-8")
    assert f"systemctl enable --now {NO_SPEND}.timer" in installer
    assert f"deploy/systemd/{NO_SPEND}.timer" in installer


def test_the_fallback_time_budget_starts_before_the_queue_wait(tmp_path: Path, monkeypatch) -> None:
    """Review minor: the fallback budget started after the queue lock was taken, so a run that waited could
    start its last handed-back compile up to twelve minutes into TimeoutStartSec=15m.  It starts with the run."""

    from tests.remote_cpu_worker_stages import install_compile_stand_ins

    compiler = install_compile_stand_ins(monkeypatch.setattr)
    host = Host(tmp_path)
    host.record_worker_environment()
    names = _handed_back(host, ["waited-0", "waited-1"])
    ticks = iter([1_000.0])

    def clock() -> float:  # the run starts at 1000; by the time it holds the queue the whole budget is gone
        return next(ticks, 1_000.0 + remote.FALLBACK_TIME_BUDGET_SECONDS + 1)

    run = remote.run_no_spend_unit(
        queue_root=host.queue, input_root=host.inputs, output_root=host.outputs, source_commit="a" * 40,
        max_messages=8, jobs_root=host.jobs, environ={remote.EXECUTION_ENV: "host"}, episode_compiler=compiler,
        filesystem_root=host.fs, cache_root=host.cache, config=remote_cpu_config(), host_environment=HOST_RECORD,
        now=clock)
    assert len(run["fallback_results"]) == 1 and run["fallback_deferred"] == names[1:]


# ------------------------------------------------------------------ everything on by default (owner, 2026-09-30)


def _write_config(path: Path, value: object | None = None, *, mode: int = 0o640) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = remote_cpu_config() if value is None else value
    path.write_text(payload if isinstance(payload, str) else json.dumps(payload), encoding="utf-8")
    path.chmod(mode)
    return path


def _observe(host: Host) -> dict:
    """What consumers read (the queue, its results and the outputs) and everything under the jobs root."""

    def files(root: Path) -> dict:
        return {path.relative_to(root).as_posix(): path.read_bytes() for path in sorted(root.rglob("*"))
                if path.is_file()}

    return {"queue": files(host.queue), "outputs": tree_snapshot(host.outputs), "jobs": files(host.jobs)}


def _no_spend(host: Host, environ: dict, compiler, *labels: str) -> tuple[list[str], dict]:
    """Stage one row per label, then one run of the no-spend unit, which loads its own config."""

    names = [stage_compile(host, label=label)[1] for label in labels]
    run = remote.run_no_spend_unit(
        queue_root=host.queue, input_root=host.inputs, output_root=host.outputs, source_commit="a" * 40,
        max_messages=8, jobs_root=host.jobs, environ=environ, episode_compiler=compiler, filesystem_root=host.fs,
        cache_root=host.cache, host_environment=HOST_RECORD, now=lambda: 1_000.0)
    return sorted(names), run


def _passes(host: Host, klass: str, *, first: int, count: int = 3, parity: str = "passed") -> None:
    """Parity records as the paid unit seals them, compared at ``first``, ``first + 1``, ..."""

    for index in range(first, first + count):
        remote.record_shadow_parity(host.jobs, {
            "closure_class": klass, "attempt_id": f"rcj-ec-{'0' * 24}-a1-{index:032x}",
            "queue_row": {"queue": remote.QUEUE, "name": f"row-{index}.json", "envelope_digest": "sha256:" + "1" * 64},
            "image": IMAGE, "host_environment_digest": HOST_RECORD["environment_digest"],
            "worker_environment_digest": HOST_RECORD["environment_digest"], "cpu_class": HOST_RECORD["cpu_class"],
            "parity": parity, "mismatches": [], "compared_at_epoch": float(index)})


def test_unset_without_this_stages_config_is_todays_host_mode_byte_for_byte(tmp_path: Path, monkeypatch) -> None:
    """An unset flag is auto, and without a config the paid unit would load for this stage auto is exactly
    today's host mode: the same results, rows, outputs and run record, byte for byte; nothing new under the jobs
    root; the paid unit's condition skips it; and nothing reaches for the network.  An explicit ``host`` beside
    a usable config is the same."""

    import socket

    from blueprint_pipeline import task_evaluation_episode_compilation_remote_condition as condition
    from tests.remote_cpu_worker_stages import install_compile_stand_ins

    compiler = install_compile_stand_ins(monkeypatch.setattr)
    root, etc = tmp_path / "host", tmp_path / "etc"

    def fresh() -> Host:
        shutil.rmtree(root, ignore_errors=True)
        host = Host(root)
        host.record_worker_environment()  # this host's rows could run remotely, were auto to turn on
        _stage_pair(host)
        return host

    host = fresh()
    before = _observe(host)["jobs"]
    today = process_episode_compilation_queue(queue_root=host.queue, input_root=host.inputs, output_root=host.outputs,
                                              source_commit="a" * 40, episode_compiler=compiler, max_messages=8)
    expected = _observe(host)
    assert expected["jobs"] == before
    usable = remote_cpu_config()
    (etc / "directory.json").mkdir(parents=True)
    cases = {
        "absent": {remote.CONFIG_ENV: str(etc / "absent.json")},
        "directory": {remote.CONFIG_ENV: str(etc / "directory.json")},  # never a crash that stops host compiles
        "world_readable": {remote.CONFIG_ENV: str(_write_config(etc / "world-readable.json", usable, mode=0o644))},
        "edited_after_sealing": {remote.CONFIG_ENV: str(_write_config(etc / "edited.json",
                                                                      {**usable, "max_attempts": 1}))},
        "explicit_host": {remote.CONFIG_ENV: str(_write_config(etc / "usable.json", usable)),
                          remote.EXECUTION_ENV: "host"},
        # Review M1: configs no loader can finish reading are no config, never a crash that stops host compiles.
        "lone_surrogate": {remote.CONFIG_ENV: str(_write_config(etc / "surrogate.json",
                                                                '{"schema_version": "\\ud800"}'))},
        "deep_nesting": {remote.CONFIG_ENV: str(_write_config(etc / "nested.json", "[" * 50000 + "]" * 50000))},
    }

    def offline(*_args, **_kwargs):
        raise AssertionError("host mode reached for the network")

    monkeypatch.setattr(socket.socket, "connect", offline)
    for label, environ in cases.items():
        host = fresh()
        resolved = remote.resolve_execution_mode(environ)
        assert (resolved["effective"], resolved["reason"]) == (
            "host", "explicit" if label == "explicit_host" else "auto_without_config"), label
        assert condition.should_run(host.jobs, environ) is False, label  # no paid-unit start beyond the check
        run = remote.run_no_spend_unit(
            queue_root=host.queue, input_root=host.inputs, output_root=host.outputs, source_commit="a" * 40,
            max_messages=8, jobs_root=host.jobs, environ=environ, episode_compiler=compiler,
            filesystem_root=host.fs, cache_root=host.cache, host_environment=HOST_RECORD, now=lambda: 1_000.0)
        assert run == today, label
        assert _observe(host) == expected, label  # no marker, and nothing else under the jobs root either


def test_unset_with_this_stages_config_runs_cloud_run_which_proves_each_class_before_handing_it_off(
        tmp_path: Path, monkeypatch) -> None:
    """With this stage's config present an unset flag runs ``cloud_run``, and ``cloud_run`` progresses by itself.
    A row whose class lacks its three shadow passes compiles on the host, authoritatively, and gets a shadow
    marker, exactly as ``cloud_run_shadow`` does it: the same queue, results, outputs and marker bytes.  Once
    the class has its passes the next row is handed off; a failed comparison sends the class back to proving."""

    from tests.remote_cpu_worker_stages import install_compile_stand_ins

    compiler = install_compile_stand_ins(monkeypatch.setattr)
    root = tmp_path / "host"
    auto = {remote.CONFIG_ENV: str(_write_config(tmp_path / "etc" / "remote-cpu-workers.json"))}
    labels = ("proving-1", "proving-2")

    def fresh() -> Host:
        shutil.rmtree(root, ignore_errors=True)
        host = Host(root)
        host.record_worker_environment()
        return host

    host = fresh()
    for label in labels:
        stage_compile(host, label=label)
    process_episode_compilation_queue(queue_root=host.queue, input_root=host.inputs, output_root=host.outputs,
                                      source_commit="a" * 40, episode_compiler=compiler, max_messages=8)
    today = _observe(host)
    runs, observed = {}, {}
    for label, environ in (("cloud_run_shadow", {**auto, remote.EXECUTION_ENV: "cloud_run_shadow"}),
                           ("cloud_run", {**auto, remote.EXECUTION_ENV: "cloud_run"}), ("auto", auto)):
        host = fresh()
        names, runs[label] = _no_spend(host, environ, compiler, *labels)
        observed[label] = _observe(host)
        # The host compiled every row: consumers read exactly what today's host mode writes.
        assert (observed[label]["queue"], observed[label]["outputs"]) == (today["queue"], today["outputs"]), label
    # Each row has a shadow marker, byte for byte the one cloud_run_shadow writes, and nothing is handed off.
    assert observed["auto"]["jobs"] == observed["cloud_run"]["jobs"] == observed["cloud_run_shadow"]["jobs"]
    assert [path for path in observed["auto"]["jobs"] if not path.startswith("environment/")] == [
        f"shadow/episode_compilation/{name}" for name in names]
    assert runs["auto"]["execution_mode"] == {"requested": None, "effective": "cloud_run",
                                              "reason": "auto_with_config", "findings": []}
    assert runs["cloud_run"]["execution_mode"]["reason"] == "explicit"
    unproven = {name: "remote_ineligible:shadow_parity_unproven:not_applicable" for name in names}
    for label in ("auto", "cloud_run"):
        assert (runs[label]["mode"], runs[label]["handoffs"], runs[label]["shadowed"],
                runs[label]["host_decisions"]) == ("cloud_run", [], names, unproven), label
    assert (runs["cloud_run_shadow"]["shadowed"], runs["cloud_run_shadow"]["host_decisions"]) == (names, {})
    assert "handoffs" not in runs["cloud_run_shadow"]

    # The class earns its three passes (the paid unit records them: the collector's end-to-end test) ...
    _passes(host, "not_applicable", first=1)
    [proven], run = _no_spend(host, auto, compiler, "proven")
    # ... so its next row is handed off, stays claimed, and is not compiled here.
    assert (run["handoffs"], run["shadowed"], run["host_decisions"]) == ([proven], [], {})
    assert (host.queue / "processing" / proven).is_file() and not (host.queue / "results" / proven).exists()
    assert remote.read_marker(remote.marker_path(host.jobs, "authoritative", proven))["mode"] == "authoritative"
    assert not remote.marker_path(host.jobs, "shadow", proven).exists()
    # cloud_run_shadow never hands off, proven class or not.
    [shadowed], run = _no_spend(host, {**auto, remote.EXECUTION_ENV: "cloud_run_shadow"}, compiler, "still-shadowed")
    assert run["shadowed"] == [shadowed] and "handoffs" not in run
    assert (host.queue / "completed" / shadowed).is_file()
    # A later failed comparison sends the class back to proving: its next row is shadowed again, once the paid unit
    # has finished the shadows outstanding (three of them, the most a class may have).
    assert len(remote.markers(host.jobs, "shadow")) == remote.SHADOW_BACKLOG_LIMIT
    for path, _ in remote.markers(host.jobs, "shadow"):
        path.unlink()
    _passes(host, "not_applicable", first=4, count=1, parity="failed")
    [again], run = _no_spend(host, auto, compiler, "proving-again")
    assert (run["handoffs"], run["shadowed"]) == ([], [again])
    assert run["host_decisions"] == {again: "remote_ineligible:shadow_parity_unproven:not_applicable"}
    assert (host.queue / "completed" / again).is_file() and remote.marker_path(host.jobs, "shadow", again).is_file()


@pytest.mark.parametrize("flag", [None, "cloud_run"])
def test_inline_nurec_rows_stay_on_the_host_when_cloud_run_progresses_by_itself(tmp_path: Path, monkeypatch,
                                                                                  flag: str | None) -> None:
    """Review I4 holds in self-progressing ``cloud_run``, auto or explicit: an inline NuRec conversion is not
    deterministic, so its row compiles on the host with no marker at all, even with every class proven."""

    from tests.remote_cpu_worker_stages import install_compile_stand_ins

    monkeypatch.setattr(remote, "MAX_INLINE_NUREC_BYTES", 8192)
    host = Host(tmp_path / "host")
    host.record_worker_environment()
    environ = {remote.CONFIG_ENV: str(_write_config(tmp_path / "etc" / "remote-cpu-workers.json"))}
    if flag is not None:
        environ[remote.EXECUTION_ENV] = flag
    for offset, klass in enumerate(("absent_inline_only", "not_applicable", "shipped")):
        _passes(host, klass, first=10 * offset + 1)
    _, name = stage_compile(host, label="inline", appearance=nurec_usdz(4096), appearance_name="appearance.usdz")
    _, run = _no_spend(host, environ, install_compile_stand_ins(monkeypatch.setattr))
    assert run["mode"] == "cloud_run"
    assert run["host_decisions"] == {name: "remote_ineligible:inline_nurec_conversion_nondeterministic"}
    assert run["handoffs"] == run["shadowed"] == []
    assert not remote.markers(host.jobs, "authoritative") and not remote.markers(host.jobs, "shadow")
    assert (host.queue / "completed" / name).is_file()


def test_chain_preflight_reports_the_requested_and_effective_mode_and_why(tmp_path: Path, monkeypatch) -> None:
    """The chain preflight resolves the mode as the no-spend unit's own account would, reports it in its own
    section with the reason, and warns when a config is there that auto cannot use."""

    import os

    from blueprint_pipeline import task_evaluation_production_chain_preflight as preflight

    host = Host(tmp_path / "host")
    host.record_worker_environment()  # the probe matches the configured image: no drift, only the mode is asked
    config = _write_config(tmp_path / "etc" / "remote-cpu-workers.json")
    environment = {remote.JOBS_ROOT_ENV: str(host.jobs), remote.CONFIG_ENV: str(config)}
    unit = preflight.EPISODE_COMPILATION_UNIT
    mine, other = (os.getuid(), os.getgid()), (987654, 987653)  # no such account: it owns nothing here

    def units(extra: dict | None = None) -> dict:
        return {unit: {"effective_environment": {**environment, **(extra or {})}}}

    def codes(extra: dict | None = None, ids: tuple = mine) -> list:
        return [(finding["severity"], finding["code"]) for finding in preflight.remote_execution_checks(units(extra), ids)]

    def resolved(effective: str, reason: str, requested: str | None = None, findings: tuple = ()) -> dict:
        return {"requested": requested, "effective": effective, "reason": reason, "findings": list(findings),
                "config": str(config)}

    assert preflight.episode_compilation_execution(units(), mine) == resolved("cloud_run", "auto_with_config")
    assert codes() == []
    assert preflight.episode_compilation_execution(units({remote.EXECUTION_ENV: "host"}), mine) == resolved(
        "host", "explicit", "host")
    assert preflight.episode_compilation_execution(units({remote.EXECUTION_ENV: "cloudrun"}), mine) == resolved(
        "host", "explicit", "invalid", ("episode_compilation_execution_mode_invalid",))
    # What root can read but the unit's account cannot is no config for that unit: auto stays host, and says so.
    assert preflight.episode_compilation_execution(units(), other) == resolved("host", "auto_without_config")
    assert codes(ids=other) == [("warning", "episode_compilation_auto_config_unusable")]
    config.chmod(0o644)  # written with the default umask, which the paid unit refuses
    assert preflight.episode_compilation_execution(units(), mine) == resolved("host", "auto_without_config")
    assert codes() == [("warning", "episode_compilation_auto_config_unusable")]
    assert codes({remote.EXECUTION_ENV: "host"}) == []  # an explicit mode is the owner's own choice
    config.unlink()
    assert (preflight.episode_compilation_execution(units(), mine)["reason"], codes()) == ("auto_without_config", [])

    # A run writes the section into its report, resolved for the unit's account.
    _write_config(config)
    props = {"LoadState": ["loaded"], "User": ["blueprint"],
             "ExecStart": ["{ argv[]=python -m blueprint_pipeline.task_evaluation_episode_compilation_worker }"],
             "Environment": [" ".join(f"{name}={value}" for name, value in environment.items())]}
    monkeypatch.setattr(preflight.os, "geteuid", lambda: 0)
    monkeypatch.setattr(preflight, "CHAIN_UNITS", (unit,))
    monkeypatch.setattr(preflight, "unit_properties", lambda name: props)
    monkeypatch.setattr(preflight, "active_release", lambda: (None, "", []))
    monkeypatch.setattr(preflight, "_service_ids", lambda account: mine)
    monkeypatch.setattr(preflight, "interpreter_environment", lambda: {})
    for name in ("intent_checks", "binding_checks", "handoff_checks", "project_spend_checks",
                 "spend_refresh_sandbox_checks", "owner_scope_checks", "credential_file_checks",
                 "provider_credit_check", "disk_admission_check", "unit_health_checks", "intake_check"):
        monkeypatch.setattr(preflight, name, lambda *args, **kwargs: [])
    out = tmp_path / "preflight.json"
    assert preflight.run_chain(preflight.build_parser().parse_args(
        ["run", "--skip-sandbox", "--json-out", str(out)])) == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["episode_compilation_execution"] == resolved("cloud_run", "auto_with_config")
    assert (report["blocker_count"], report["warning_count"]) == (0, 0)


def test_cloud_run_shadows_only_what_the_host_compiled_and_at_most_three_per_class(tmp_path: Path,
                                                                                   monkeypatch) -> None:
    """Review I1: self-progression spends only where it can progress.  A row the host blocked is never shadowed
    in ``cloud_run`` (its comparison could only be inconclusive or failed), and a class never has more than
    three shadows outstanding, the passes it needs.  ``cloud_run_shadow`` keeps shadowing every eligible row."""

    from blueprint_pipeline.task_evaluation_native_arena_episode_compiler import (
        TaskEvaluationNativeArenaEpisodeCompilerError,
    )
    from tests.remote_cpu_worker_stages import install_compile_stand_ins

    compiler = install_compile_stand_ins(monkeypatch.setattr)
    auto = {remote.CONFIG_ENV: str(_write_config(tmp_path / "etc" / "remote-cpu-workers.json"))}

    def refuses(**_kwargs):
        raise TaskEvaluationNativeArenaEpisodeCompilerError("episode_compiler_destination_usd_format_unrecognized")

    for label, environ, shadows_everything in (
            ("cloud_run", auto, False), ("cloud_run_shadow", {**auto, remote.EXECUTION_ENV: "cloud_run_shadow"}, True)):
        host = Host(tmp_path / label)
        host.record_worker_environment()
        [blocked], run = _no_spend(host, environ, refuses, "refused")
        assert (host.queue / "blocked" / blocked).is_file(), label
        assert remote.marker_path(host.jobs, "shadow", blocked).is_file() is shadows_everything, label
        assert run.get("shadow_skipped", {}) == ({} if shadows_everything else {blocked: "host_compile_blocked"})
        names, run = _no_spend(host, environ, compiler, "row-1", "row-2", "row-3", "row-4", "row-5")
        assert all((host.queue / "completed" / name).is_file() for name in names), label
        if shadows_everything:
            assert run["shadowed"] == names and "shadow_skipped" not in run
            continue
        # The blocked row's marker was never written, so three of these five may be shadowed, and no more.
        assert run["shadowed"] == names[:3] and run["shadow_skipped"] == dict.fromkeys(names[3:],
                                                                                      "shadow_backlog_full")
        assert len(remote.markers(host.jobs, "shadow")) == 3


def test_readable_by_counts_supplementary_groups_and_can_require_every_parent_to_be_searchable(
        tmp_path: Path, monkeypatch) -> None:
    """Review M4: the kernel grants a file's group bits to any of the account's groups, not only its primary one,
    and nothing under a directory the account cannot search.  POSIX ACLs and a unit's own sandbox stay out of
    scope, as the preflight's docstring says."""

    import os
    import stat as mode
    import types

    from blueprint_pipeline import task_evaluation_production_chain_preflight as preflight

    folder = tmp_path / "etc"
    folder.mkdir()
    config = _write_config(folder / "remote-cpu-workers.json")
    group, account = config.stat().st_gid, (987654, 987653)  # no such account: it owns nothing here
    monkeypatch.setattr(preflight.pwd, "getpwuid", lambda uid: types.SimpleNamespace(pw_name="blueprint"))
    monkeypatch.setattr(preflight.os, "getgrouplist", lambda name, gid: [gid])
    assert preflight.readable_by(config, *account) is False
    monkeypatch.setattr(preflight.os, "getgrouplist", lambda name, gid: [gid, group])  # a supplementary member
    assert preflight.readable_by(config, *account) is True
    # The directories above this test's tree stand in for the host's own root-owned, searchable ones.
    real, above = os.stat, set(folder.parents)

    def system_above(path, *args, **kwargs):
        found = real(path, *args, **kwargs)
        if not isinstance(path, (str, os.PathLike)) or Path(path) not in above:
            return found
        return os.stat_result((mode.S_IFDIR | 0o755, found.st_ino, found.st_dev, found.st_nlink, 0, 0,
                               found.st_size, found.st_atime, found.st_mtime, found.st_ctime))

    monkeypatch.setattr(os, "stat", system_above)
    environment = {remote.CONFIG_ENV: str(config), remote.JOBS_ROOT_ENV: str(tmp_path / "jobs")}
    units = {preflight.EPISODE_COMPILATION_UNIT: {"effective_environment": environment}}
    folder.chmod(0o750)  # its group may search it
    assert preflight.readable_by(config, *account, traverse=True) is True
    assert preflight.episode_compilation_execution(units, account)["reason"] == "auto_with_config"
    folder.chmod(0o700)  # only its owner may
    assert preflight.readable_by(config, *account) is True  # the file's own bits alone
    assert preflight.readable_by(config, *account, traverse=True) is False
    # The preflight resolves auto mode for the unit's account through every parent.
    assert preflight.episode_compilation_execution(units, account)["reason"] == "auto_without_config"
