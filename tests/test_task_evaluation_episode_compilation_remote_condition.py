# Covers (for impacted-test selection):
#   src/blueprint_pipeline/task_evaluation_episode_compilation_remote_condition.py
#   src/blueprint_pipeline/task_evaluation_episode_compilation_remote.py
#   src/blueprint_pipeline/task_evaluation_episode_compilation_collector.py
#   src/blueprint_pipeline/remote_cpu_job_allocator.py
#   deploy/systemd/blueprint-task-evaluation-episode-compilation.service
#   deploy/systemd/blueprint-task-evaluation-episode-compilation-remote.service
"""Owner decision 2026-09-30, everything on by default: an unset ``BLUEPRINT_EPISODE_COMPILATION_EXECUTION`` is
auto.  It runs ``cloud_run`` once this stage's remote-CPU config is there, exactly as the paid unit loads it, and
``host`` otherwise; the paid unit's ExecCondition answers the same question with the standard library alone."""

from __future__ import annotations

import json
from pathlib import Path

from blueprint_pipeline import remote_cpu_job_allocator as allocator
from blueprint_pipeline import remote_cpu_job_contract as contract
from blueprint_pipeline import task_evaluation_episode_compilation_remote as remote
from blueprint_pipeline import task_evaluation_episode_compilation_remote_condition as condition
from tests.remote_cpu_allocator_fakes import remote_cpu_config, seal

ROOT = Path(__file__).resolve().parents[1]
SYSTEMD = ROOT / "deploy" / "systemd"
AUTO_ON = {"requested": None, "effective": "cloud_run", "reason": "auto_with_config", "findings": []}
AUTO_OFF = {"requested": None, "effective": "host", "reason": "auto_without_config", "findings": []}


def _write(path: Path, value: object, *, mode: int = 0o640) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value if isinstance(value, str) else json.dumps(value), encoding="utf-8")
    path.chmod(mode)
    return path


def _config_states(root: Path) -> dict[str, tuple[Path, bool]]:
    """Every state the config file can be in, and whether the paid unit would load it for this stage."""

    usable = remote_cpu_config()
    stage = usable["stages"]["episode_compilation"]
    states = {
        "usable": (_write(root / "usable.json", usable), True),
        "owner_only": (_write(root / "owner-only.json", usable, mode=0o600), True),
        "absent": (root / "absent.json", False),
        "world_readable": (_write(root / "world-readable.json", usable, mode=0o644), False),
        "group_writable": (_write(root / "group-writable.json", usable, mode=0o660), False),
        "not_json": (_write(root / "not-json.json", "{"), False),
        "not_an_object": (_write(root / "list.json", "[]"), False),
        "edited_after_sealing": (_write(root / "edited.json", {**usable, "max_attempts": 1}), False),
        "other_schema": (_write(root / "other-schema.json", seal(
            {**usable, "schema_version": "remote_cpu_workers_config.v0"}, "config_digest")), False),
        "other_stage_only": (_write(root / "other-stage.json", seal(
            {**usable, "stages": {"cpu_prestage": stage}}, "config_digest")), False),
        # Larger than the allocator reads: it refuses the file, so auto must not turn on for it either.
        "oversize": (_write(root / "oversize.json", json.dumps(usable) + " " * allocator._MAX_RECORD_BYTES), False),
    }
    (root / "directory.json").mkdir()
    states["directory"] = (root / "directory.json", False)
    (root / "symlink.json").symlink_to(root / "usable.json")
    states["symlink"] = (root / "symlink.json", False)
    return states


def test_auto_turns_on_only_for_a_config_the_paid_unit_would_load_for_this_stage(tmp_path: Path) -> None:
    """The no-spend unit's loader is the paid unit's (``remote_cpu_job_allocator.load_remote_cpu_config``), which
    the remote module may not import: the worker's stage child imports it from the release.  They agree on
    every state the file can be in, and so does the ExecCondition's standard-library reading."""

    for label, (path, usable) in _config_states(tmp_path).items():
        # The allocator's reader raises on a directory (it wraps the descriptor before checking it).  The no-spend
        # side reads one as no config: an unset mode reads this path on every run and must never stop host compiles.
        if label != "directory":
            _, blockers = allocator.load_remote_cpu_config(path)
            assert (remote.load_config(path) is not None) == (not blockers), label
        environ = {remote.CONFIG_ENV: str(path)}
        assert remote.remote_configured(environ) is usable, label
        assert remote.resolve_execution_mode(environ) == (AUTO_ON if usable else AUTO_OFF), label
        assert condition.configured(path) is usable, label
    # One file, one name, one bound, one schema: the units' environment, both loaders and the condition agree.
    assert (condition.CONFIG_ENV, condition.DEFAULT_CONFIG_PATH) == (remote.CONFIG_ENV, remote.DEFAULT_CONFIG_PATH) == (
        allocator.CONFIG_ENV, allocator.DEFAULT_CONFIG_PATH) == (
        "BLUEPRINT_REMOTE_CPU_WORKERS_CONFIG", "/etc/blueprint/remote-cpu-workers.json")
    assert condition.CONFIG_MAX_BYTES == remote.CONFIG_MAX_BYTES == allocator._MAX_RECORD_BYTES
    assert condition.CONFIG_SCHEMA_VERSION == contract.CONFIG_SCHEMA_VERSION
    for unit in ("blueprint-task-evaluation-episode-compilation.service",
                 "blueprint-task-evaluation-episode-compilation-remote.service"):
        text = (SYSTEMD / unit).read_text(encoding="utf-8")
        assert f"Environment={remote.CONFIG_ENV}={remote.DEFAULT_CONFIG_PATH}\n" in text, unit


def test_the_exec_condition_runs_the_paid_unit_when_the_effective_mode_is_remote_or_to_drain(
        tmp_path: Path, monkeypatch) -> None:
    jobs = tmp_path / "jobs"
    usable = _write(tmp_path / "etc" / "usable.json", remote_cpu_config())
    with_config, without = {condition.CONFIG_ENV: str(usable)}, {condition.CONFIG_ENV: str(tmp_path / "absent.json")}
    flag = condition.EXECUTION_ENV
    # Unset or empty is auto: without this stage's config it is today's host mode, skipped; with it, it runs.
    for unset in ({}, {flag: ""}, {flag: " "}):
        assert condition.should_run(jobs, {**without, **unset}) is False
        assert condition.should_run(jobs, {**with_config, **unset}) is True
    # A set value keeps its name: host skips even beside a config, a remote mode runs even without one, and an
    # invalid value runs as host.
    assert condition.should_run(jobs, {**with_config, flag: "host"}) is False
    for mode in ("cloud_run", "cloud_run_shadow"):
        assert condition.should_run(jobs, {**without, flag: mode}) is True
    assert condition.should_run(jobs, {**with_config, flag: "cloudrun"}) is False
    # The condition's answer is the no-spend unit's effective mode (the census gate aside, which the run applies).
    cases = [{**without}, {**with_config}, {**with_config, flag: "host"}, {**without, flag: "cloud_run"},
             {**without, flag: "cloud_run_shadow"}, {**with_config, flag: "cloudrun"}]
    for environ in cases:
        assert condition.should_run(jobs, environ) is (remote.resolve_execution_mode(environ)["effective"] != "host")
    # Draining is unchanged: a live lease or a waiting marker runs it in every mode, auto without config included.
    for directory in (jobs / "live", jobs / "handoffs" / "episode_compilation", jobs / "shadow" / "episode_compilation"):
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "row.json").write_text("{}", encoding="utf-8")
        for environ in (without, {**with_config, flag: "host"}, {flag: "cloudrun"}):
            assert condition.should_run(jobs, environ) is True, (directory, environ)
        (directory / "row.json").unlink()
    # The unit's own entry point reads its environment: exit 0 runs the unit, 1 skips it.
    monkeypatch.delenv(flag, raising=False)
    monkeypatch.setenv(condition.CONFIG_ENV, str(usable))
    assert condition.main([str(jobs)]) == 0
    monkeypatch.setenv(condition.CONFIG_ENV, str(tmp_path / "absent.json"))
    assert condition.main([str(jobs)]) == 1


def test_a_sealed_config_the_paid_unit_refuses_starts_it_only_to_drain_without_a_provider(
        tmp_path: Path, monkeypatch) -> None:
    """The condition checks what the standard library can: the file, its JSON, schema, seal and stage.  A sealed
    config that fails a deeper check (here a region outside the US) still starts the unit.  The run resolves auto
    to host, loads the config as the allocator does, finds it unusable, and drains without ever connecting: no
    network, no spend, and a summary that says why."""

    from blueprint_pipeline import task_evaluation_episode_compilation_collector as collector
    from blueprint_pipeline.task_evaluation_episode_compilation_worker import OUTPUT_ROOT_ENV, QUEUE_ROOT_ENV

    config = _write(tmp_path / "etc" / "remote-cpu-workers.json", remote_cpu_config(region="europe-west1"))
    assert condition.configured(config) is True and remote.load_config(config) is None
    jobs = tmp_path / "jobs"
    jobs.mkdir()
    assert condition.should_run(jobs, {condition.CONFIG_ENV: str(config)}) is True

    def never(*_args, **_kwargs):
        raise AssertionError("the paid unit connected with a config it cannot use")

    monkeypatch.delenv(remote.EXECUTION_ENV, raising=False)
    monkeypatch.setenv(remote.CONFIG_ENV, str(config))
    monkeypatch.setenv(QUEUE_ROOT_ENV, str(tmp_path / "queue"))
    monkeypatch.setenv(OUTPUT_ROOT_ENV, str(tmp_path / "outputs"))
    monkeypatch.setattr(allocator, "_connect", never)
    assert collector.main(["run", "--source-commit", "a" * 40, "--jobs-root", str(jobs)]) == 0
    summary = json.loads((jobs / "summary.json").read_text(encoding="utf-8"))
    assert (summary["mode"], summary["execution_mode"], summary["rows"]) == ("host", AUTO_OFF, {})
    assert summary["blockers"] == ["remote_cpu_config_region_not_us"]
