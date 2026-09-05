"""The chain look-ahead finds every plumbing blocker at once, under each unit's own sandbox."""

from __future__ import annotations

import json
import os
import stat
from collections import namedtuple
from pathlib import Path

from blueprint_pipeline import task_evaluation_production_chain_preflight as preflight

GIB = 1024**3
Usage = namedtuple("Usage", "total used free")


def _package(tmp_path: Path) -> Path:
    src = tmp_path / "release" / "src"
    package = src / "blueprint_pipeline"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "worker.py").write_text(
        'import os\nfrom blueprint_pipeline.queue import enqueue\n'
        'QUEUE = "/var/lib/blueprint/pipeline-control-plane/task-evaluation-launches"\n'
        'def tick():\n    from . import ledger\n    root = os.environ.get("BLUEPRINT_SPEND_AUTHORITY_ROOT")\n',
        encoding="utf-8",
    )
    (package / "queue.py").write_text(
        'CHILD = "/var/lib/blueprint/pipeline-control-plane/sam31-preparation-executions/"\n'
        'def enqueue():\n    return CHILD\n',
        encoding="utf-8",
    )
    (package / "ledger.py").write_text(
        'import os\nHOME_ROOT = os.environ.get("BLUEPRINT_LEDGER_ROOT") or "/var/lib/blueprint/spend-authority"\n'
        'SCHEMA = "BLUEPRINT_NOT_AN_ENV_NAME"\n',
        encoding="utf-8",
    )
    (package / "unrelated.py").write_text('X = "/etc/blueprint/unrelated"\n', encoding="utf-8")
    return src


def test_import_closure_follows_absolute_relative_and_lazy_imports_only(tmp_path: Path) -> None:
    src = _package(tmp_path)

    closure = preflight.import_closure(src, "blueprint_pipeline.worker")

    assert set(closure) == {
        "blueprint_pipeline",
        "blueprint_pipeline.worker",
        "blueprint_pipeline.queue",
        "blueprint_pipeline.queue.enqueue",
        "blueprint_pipeline.ledger",
    } - {"blueprint_pipeline.queue.enqueue"}
    assert "blueprint_pipeline.unrelated" not in closure
    sources = {name: path.read_text(encoding="utf-8") for name, path in closure.items()}
    roots = preflight.literal_roots(sources)
    assert roots == {
        "/var/lib/blueprint/pipeline-control-plane/sam31-preparation-executions": ["blueprint_pipeline.queue"],
        "/var/lib/blueprint/pipeline-control-plane/task-evaluation-launches": ["blueprint_pipeline.worker"],
        "/var/lib/blueprint/spend-authority": ["blueprint_pipeline.ledger"],
    }
    names = preflight.environment_names_read(sources)
    assert set(names) == {"BLUEPRINT_SPEND_AUTHORITY_ROOT", "BLUEPRINT_LEDGER_ROOT"}


def test_path_probe_names_the_errno_and_the_missing_ancestor(tmp_path: Path) -> None:
    writable = tmp_path / "writable"
    writable.mkdir()
    assert preflight.path_probe(str(writable))["status"] == "writable"
    locked = tmp_path / "locked"
    locked.mkdir()
    locked.chmod(0o500)
    try:
        verdict = preflight.path_probe(str(locked))
    finally:
        locked.chmod(0o700)
    if os.getuid() != 0:
        assert verdict["status"] == "permission_denied" and verdict["errno"] == 13
    missing = preflight.path_probe(str(tmp_path / "absent" / "deeper"))
    assert missing["status"] == "missing_creatable" and missing["nearest_ancestor"] == str(tmp_path)
    text = tmp_path / "config.json"
    text.write_text("{}", encoding="utf-8")
    assert preflight.path_probe(str(text))["status"] == "readable"


def test_severity_follows_storage_class_and_the_unit_declared_intent() -> None:
    read_only = {"status": "read_only"}
    assert preflight._severity_for_directory(read_only, storage_class="work", in_rw=False) == "blocker"
    assert preflight._severity_for_directory(read_only, storage_class="work", in_rw=False, declared_ro=True) == "info"
    # ReadOnlyPaths inside a ReadWritePaths tree is still the operator's intent.
    assert preflight._severity_for_directory(read_only, storage_class="cache", in_rw=True, declared_ro=True) == "info"
    assert preflight._severity_for_directory(read_only, storage_class="container", in_rw=False) == "info"
    # Root-owned container and release trees are never written; naming them is not a blocker.
    assert preflight._severity_for_directory({"status": "permission_denied"}, storage_class="container", in_rw=True) == "info"
    assert preflight._severity_for_directory({"status": "permission_denied"}, storage_class="release", in_rw=True) == "info"
    assert preflight._severity_for_directory(read_only, storage_class=None, in_rw=True) == "blocker"
    assert preflight._severity_for_directory(read_only, storage_class=None, in_rw=False) == "warning"
    missing = {"status": "missing_not_creatable:read_only"}
    assert preflight._severity_for_directory(missing, storage_class="ledger", in_rw=False) == "warning"
    assert preflight._severity_for_directory({"status": "writable"}, storage_class="work", in_rw=True) == "ok"
    assert preflight._severity_for_directory({"status": "unreadable"}, storage_class=None, in_rw=False) == "blocker"


def test_sandbox_replay_uses_only_the_directives_the_unit_configured(tmp_path: Path) -> None:
    env_file = tmp_path / "unit.env"
    env_file.write_text("BLUEPRINT_FROM_FILE=/var/lib/blueprint/from-file\nQUOTED='v w'\n", encoding="utf-8")
    props = {
        "User": ["blueprint"],
        "Group": ["blueprint"],
        "ProtectSystem": ["strict"],
        "ProtectHome": ["yes"],
        "PrivateTmp": ["yes"],
        "UMask": ["0077"],
        "ReadWritePaths": ["/var/lib/blueprint/a -/var/lib/blueprint/b"],
        "ReadOnlyPaths": ["/etc/blueprint/profiles"],
        "CapabilityBoundingSet": ["cap_chown cap_dac_override"],
        "SystemCallFilter": ["~@mount @privileged"],
        "Environment": ['BLUEPRINT_A=1 "BLUEPRINT_JSON=[\\"s3://x/\\"]"'],
        "EnvironmentFiles": [f"{env_file} (ignore_errors=yes)"],
    }
    directives = {"User", "ProtectSystem", "ProtectHome", "PrivateTmp", "UMask", "ReadWritePaths", "ReadOnlyPaths", "SystemCallFilter", "Environment", "EnvironmentFile"}

    command = preflight.systemd_run_command(
        unit="blueprint-x.service",
        props=props,
        directives=directives,
        python="/usr/bin/python3",
        script=tmp_path / "probe.py",
        module="blueprint_pipeline.worker",
        release=tmp_path / "release",
        active_sha="a" * 40,
    )

    joined = " ".join(command)
    assert command[0] == "systemd-run" and "--wait" in command and "--pipe" in command
    assert "-p User=blueprint" in joined and "-p ProtectSystem=strict" in joined
    assert "-p ReadWritePaths=/var/lib/blueprint/a -/var/lib/blueprint/b" in joined
    assert "-p SystemCallFilter=~@mount @privileged" in joined
    # Not configured by the unit, so not replayed: Group and CapabilityBoundingSet.
    assert "Group=" not in joined and "CapabilityBoundingSet" not in joined
    assert f"-p EnvironmentFile=-{env_file}" in joined
    assert "--setenv=BLUEPRINT_A=1" in command and '--setenv=BLUEPRINT_JSON=["s3://x/"]' in command
    assert command[-6:] == ["--active-sha", "a" * 40, "--read-write-paths", "/var/lib/blueprint/a -/var/lib/blueprint/b", "--read-only-paths", "/etc/blueprint/profiles"]
    effective = preflight.effective_environment(props)
    assert effective["BLUEPRINT_FROM_FILE"] == "/var/lib/blueprint/from-file"
    assert effective["QUOTED"] == "v w" and effective["BLUEPRINT_A"] == "1"


def test_readable_by_follows_discretionary_access_and_root_bypasses_it(tmp_path: Path) -> None:
    secret = tmp_path / "secret"
    secret.write_text("x", encoding="utf-8")
    secret.chmod(0o000)
    try:
        assert preflight.readable_by(secret, 0, 0) is True
        if os.getuid() != 0:
            assert preflight.readable_by(secret, os.getuid(), os.getgid()) is False
        secret.chmod(0o400)
        assert preflight.readable_by(secret, os.getuid(), os.getgid()) is True
    finally:
        secret.chmod(0o600)
    assert preflight.readable_by(tmp_path / "absent", 0, 0) is False


def test_disk_admission_projection_reports_every_refused_role(tmp_path: Path, monkeypatch) -> None:
    total = 154 * GIB
    monkeypatch.setattr(preflight.shutil, "disk_usage", lambda _p: Usage(total, total - int(9.67 * GIB), int(9.67 * GIB)))
    monkeypatch.setattr(preflight, "DISK_RESERVATION_ROOT", tmp_path / "reservations")

    [finding] = preflight.disk_admission_check({})

    assert finding["severity"] == "blocker" and finding["code"] == "disk_admission_projection"
    assert finding["refused_roles"] == ["episode_compilation", "launch_activation", "launch_dispatch", "launch_preparation", "policy_canary_dispatch"]
    assert finding["free_needed_for_one_role_gib"] == 10.0 and finding["free_needed_for_whole_chain_gib"] == 18.0
    monkeypatch.setattr(preflight.shutil, "disk_usage", lambda _p: Usage(total, 0, 60 * GIB))
    [healthy] = preflight.disk_admission_check({})
    assert healthy["severity"] == "info" and "refused_roles" not in healthy


def test_handoff_checks_name_each_missing_piece_of_the_canary_chain(tmp_path: Path) -> None:
    secret = tmp_path / "submit_secret"
    secret.write_text("s", encoding="utf-8")
    progression = "blueprint-task-evaluation-configured-controls-progression.service"
    dispatcher = "blueprint-task-evaluation-launch-dispatcher.service"
    units = {
        progression: {"effective_environment": {
            "BLUEPRINT_POLICY_CANARY_NOTIFICATION_EMAIL": "",
            "BLUEPRINT_TASK_EVALUATION_WEBAPP_SUBMISSION_SECRET_FILE": str(secret),
            "BLUEPRINT_TASK_EVALUATION_WEBAPP_SUBMISSION_URL": "https://tryblueprint.io/api/internal/x",
            "BLUEPRINT_TASK_EVALUATION_LAUNCH_PROFILE_CATALOG": "/var/lib/blueprint/pipeline-control-plane/catalog.json",
        }},
        "blueprint-pipeline-intake.service": {"effective_environment": {
            "BLUEPRINT_TASK_EVALUATION_LAUNCH_PUBLIC_CATALOG_PATH": "/var/lib/blueprint/pipeline-control-plane/other.json",
        }},
        dispatcher: {"effective_environment": {"BLUEPRINT_TASK_EVALUATION_LAUNCH_EXECUTE": "0"}},
    }

    codes = sorted(f["code"] for f in preflight.handoff_checks(units, (os.getuid(), os.getgid())))

    assert codes == [
        "launch_dispatcher_execute_gate_closed",
        "launch_dispatcher_execute_id_unset",
        "launch_profile_catalog_path_disagrees_with_intake",
        "policy_canary_notification_email_unset",
    ]
    units[progression]["effective_environment"]["BLUEPRINT_POLICY_CANARY_NOTIFICATION_EMAIL"] = "owner@example.com"
    units["blueprint-pipeline-intake.service"]["effective_environment"]["BLUEPRINT_TASK_EVALUATION_LAUNCH_PUBLIC_CATALOG_PATH"] = "/var/lib/blueprint/pipeline-control-plane/catalog.json"
    units[dispatcher]["effective_environment"] = {"BLUEPRINT_TASK_EVALUATION_LAUNCH_EXECUTE": "1", "BLUEPRINT_TASK_EVALUATION_LAUNCH_EXECUTE_ID": "gate-1"}
    assert preflight.handoff_checks(units, (os.getuid(), os.getgid())) == []


def test_probe_reports_roots_the_closure_names_with_their_writability(tmp_path: Path, monkeypatch) -> None:
    _package(tmp_path)
    monkeypatch.setenv("BLUEPRINT_SPEND_AUTHORITY_ROOT", str(tmp_path / "spend"))
    (tmp_path / "spend").mkdir()
    monkeypatch.setenv("BLUEPRINT_INPUT_FILE", str(tmp_path / "input.json"))
    (tmp_path / "input.json").write_text("{}", encoding="utf-8")
    monkeypatch.delenv("BLUEPRINT_LEDGER_ROOT", raising=False)
    args = preflight.argparse.Namespace(
        unit="blueprint-x.service",
        module="blueprint_pipeline.worker",
        release=str(tmp_path / "release"),
        active_sha="a" * 40,
        read_write_paths=str(tmp_path / "spend"),
        read_only_paths="",
    )

    report = preflight.run_probe(args)

    paths = report["paths"]
    assert paths[str(tmp_path / "spend")]["status"] == "writable"
    assert paths[str(tmp_path / "input.json")]["status"] == "readable"
    assert "/var/lib/blueprint/pipeline-control-plane/sam31-preparation-executions" in paths
    assert "BLUEPRINT_LEDGER_ROOT" in report["environment_names_read_but_unset"]
    assert "BLUEPRINT_SPEND_AUTHORITY_ROOT" not in report["environment_names_read_but_unset"]
    assert report["closure_size"] == 4
    assert json.dumps(report, default=str)


def test_intent_checks_require_an_activation_intent_at_the_active_release(tmp_path: Path, monkeypatch) -> None:
    activation = tmp_path / "activation-intents"
    controls = tmp_path / "controls-intents"
    activation.mkdir()
    controls.mkdir()
    monkeypatch.setattr(preflight, "ACTIVATION_INTENT_ROOT", activation)
    monkeypatch.setattr(preflight, "CONTROLS_INTENT_ROOT", controls)
    (activation / "old.json").write_text(json.dumps({"expected_production_commit": "b" * 40}), encoding="utf-8")
    ids = (os.getuid(), os.getgid())

    codes = [f["code"] for f in preflight.intent_checks("a" * 40, ids)]

    assert "activation_intent_bound_to_other_release" in codes
    assert "activation_intent_missing_for_active_release" in codes
    (activation / "live.json").write_text(json.dumps({"expected_production_commit": "a" * 40}), encoding="utf-8")
    codes = [f["code"] for f in preflight.intent_checks("a" * 40, ids)]
    assert "activation_intent_missing_for_active_release" not in codes
    st = (activation / "live.json").stat()
    assert stat.S_ISREG(st.st_mode)
