"""The chain look-ahead finds every plumbing blocker at once, under each unit's own sandbox."""

from __future__ import annotations

import json
import os
import shlex
import stat
import textwrap
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


def test_declaring_read_only_does_not_hide_an_unreadable_directory(tmp_path, monkeypatch):
    def denied(_):
        raise PermissionError(13, "not readable")
    monkeypatch.setattr(preflight.os, "scandir", denied)
    verdict = preflight.path_probe(str(tmp_path))
    assert verdict["status"] == "unreadable"
    assert preflight._severity_for_directory(verdict, storage_class="ledger", in_rw=False, declared_ro=True) == "blocker"


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
    assert command[-4:] == ["--active-sha", "a" * 40, "--read-write-paths=/var/lib/blueprint/a -/var/lib/blueprint/b", "--read-only-paths=/etc/blueprint/profiles"]
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

    findings = preflight.handoff_checks(units, (os.getuid(), os.getgid()))
    codes = sorted(f["code"] for f in findings if f["severity"] == "blocker")

    assert codes == [
        "launch_dispatcher_execution_hold_present",
        "launch_profile_catalog_path_disagrees_with_intake",
        "policy_canary_notification_email_unset",
    ]
    # The empty per-launch id is the hands-off state (standing authorization admits): recorded, not blocking.
    assert [f["code"] for f in findings if f["severity"] == "info"] == ["launch_dispatcher_execute_id_unset"]
    units[progression]["effective_environment"]["BLUEPRINT_POLICY_CANARY_NOTIFICATION_EMAIL"] = "owner@example.com"
    units["blueprint-pipeline-intake.service"]["effective_environment"]["BLUEPRINT_TASK_EVALUATION_LAUNCH_PUBLIC_CATALOG_PATH"] = "/var/lib/blueprint/pipeline-control-plane/catalog.json"
    units[dispatcher]["effective_environment"] = {"BLUEPRINT_TASK_EVALUATION_LAUNCH_EXECUTE": "1"}
    assert [f["code"] for f in preflight.handoff_checks(units, (os.getuid(), os.getgid())) if f["severity"] == "blocker"] == []


def test_execution_holds_are_named_with_their_source(tmp_path: Path) -> None:
    """2026-09-05: the construction launch was held by LAUNCH_EXECUTE=false in a per-scene
    EnvironmentFile and the canary dispatcher by an ExecCondition=/usr/bin/false drop-in;
    both are operator holds that stall a hands-off run without queue evidence."""

    hold = tmp_path / "scene.env"
    hold.write_text("BLUEPRINT_TASK_EVALUATION_LAUNCH_EXECUTE=false\n", encoding="utf-8")
    dispatcher = "blueprint-task-evaluation-launch-dispatcher.service"
    units = {dispatcher: {
        "effective_environment": {"BLUEPRINT_TASK_EVALUATION_LAUNCH_EXECUTE": "false"},
        "properties": {"EnvironmentFiles": [f"{hold} (ignore_errors=yes)"]},
    }}

    [finding] = [f for f in preflight.handoff_checks(units, (os.getuid(), os.getgid())) if f["code"] == "launch_dispatcher_execution_hold_present"]
    assert finding["held_by"] == [str(hold)]

    canary = "blueprint-task-evaluation-policy-canary-dispatcher.service"
    health = preflight.unit_health_checks({canary: {"properties": {
        "ActiveState": ["inactive"], "Result": ["exec-condition"], "LoadState": ["loaded"], "TriggeredBy": [""],
        "ExecCondition": ["{ path=/usr/bin/false ; argv[]=/usr/bin/false ; ignore_errors=no }"],
        "DropInPaths": ["/etc/systemd/system/blueprint-task-evaluation-policy-canary-dispatcher.service.d/99-no-dispatch-hold.conf"],
    }}})
    [hold_finding] = [f for f in health if f["code"] == "unit_execution_hold_present"]
    assert hold_finding["severity"] == "blocker"
    assert hold_finding["drop_ins"] == ["/etc/systemd/system/blueprint-task-evaluation-policy-canary-dispatcher.service.d/99-no-dispatch-hold.conf"]


def test_probe_imports_from_the_release_src_root_not_the_package_directory(tmp_path: Path, monkeypatch) -> None:
    """Run as a script, the probe's own directory led sys.path; a bare-name import in
    the closure then resolved a package module as top-level and its relative imports
    failed.  Six production units reported a false entry_module_import_failed."""
    import sys as _sys

    _package(tmp_path)
    package_dir = str(Path(preflight.__file__).resolve().parent)
    monkeypatch.setattr(_sys, "path", [package_dir, *_sys.path])
    args = preflight.argparse.Namespace(
        unit="blueprint-x.service",
        module="blueprint_pipeline.worker",
        release=str(tmp_path / "release"),
        active_sha="a" * 40,
        read_write_paths="",
        read_only_paths="",
    )
    preflight.run_probe(args)
    assert package_dir not in _sys.path
    assert _sys.path[0] == str(tmp_path / "release" / "src")


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


def test_sam31_provider_profile_from_another_release_is_a_blocker_before_submission(tmp_path: Path) -> None:
    """Scene 841757's hardware profile referenced scene 840920's provider profile from
    three weeks earlier; the launch-packet validator refused it after a paid render."""
    stack = tmp_path / "worker-stack.json"
    stack.write_text(json.dumps({"source_commit_sha": "b" * 40}), encoding="utf-8")
    provider = tmp_path / "sam31_provider_profile.v1.json"
    provider.write_text(
        json.dumps({"source_commit_sha": "b" * 40, "worker_stack_manifest": {"path": str(stack)}}), encoding="utf-8"
    )
    hardware = tmp_path / "sam31_preparation_profile.v1.json"
    hardware.write_text(
        json.dumps({"artifact_references": {"sam31_provider_profile": {"path": str(provider)}}}), encoding="utf-8"
    )

    findings = preflight._sam31_provider_profile_findings(hardware, "sam.service", "a" * 40)

    [finding] = findings
    assert finding["severity"] == "blocker"
    assert finding["code"] == "sam31_provider_profile_bound_to_other_release"
    # #1669: only the provider profile itself must carry the active release; the worker stack
    # manifest and runtime image build receipt need a valid commit, not the active one.
    assert finding["bound"] == {"provider_profile": "b" * 12}
    assert finding["records"] == {"provider_profile": "b" * 12, "worker_stack_manifest": "b" * 12}

    provider.write_text(
        json.dumps({"source_commit_sha": "a" * 40, "worker_stack_manifest": {"path": str(stack)}}), encoding="utf-8"
    )
    [current] = preflight._sam31_provider_profile_findings(hardware, "sam.service", "a" * 40)
    assert current["code"] == "sam31_provider_profile_bound_to_active_release"
    assert current["severity"] != "blocker"
    stack.write_text(json.dumps({"source_commit_sha": "a" * 40}), encoding="utf-8")
    [current] = preflight._sam31_provider_profile_findings(hardware, "sam.service", "a" * 40)
    assert current["code"] == "sam31_provider_profile_bound_to_active_release"
    hardware.write_text(json.dumps({"artifact_references": {}}), encoding="utf-8")
    [missing] = preflight._sam31_provider_profile_findings(hardware, "sam.service", "a" * 40)
    assert missing["code"] == "sam31_provider_profile_reference_missing"


def test_history_records_one_bounded_line_per_run(tmp_path: Path) -> None:
    report = {"generated_at": "2026-09-05T17:00:00Z", "active_sha": "a" * 40, "warning_count": 3}
    blockers = [{"code": "disk_admission_projection"}, {"code": "sam31_provider_profile_bound_to_other_release"}, {"code": "disk_admission_projection"}]
    row = preflight.append_history(tmp_path / "preflight" / "history.jsonl", report, blockers=blockers)
    preflight.append_history(tmp_path / "preflight" / "history.jsonl", report, blockers=[])
    lines = (tmp_path / "preflight" / "history.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert row["blocker_count"] == 3
    assert row["blocker_codes"] == ["disk_admission_projection", "sam31_provider_profile_bound_to_other_release"]
    assert json.loads(lines[1])["blocker_count"] == 0


def test_installation_receipt_from_another_release_is_provenance_not_a_blocker(tmp_path: Path) -> None:
    """Installed sources are bound by content identity (#1653); before that change the
    per-release re-installation was the first production blocker of the day."""
    import hashlib

    receipt = tmp_path / "public_scene_host_input_installation_receipt.v1.json"
    receipt.write_text(json.dumps({"source_commit_sha": "b" * 40}), encoding="utf-8")
    intake = tmp_path / "publisher_intake.v1.json"
    intake.write_bytes(b'{"scene_id": "841757"}')
    digest = "sha256:" + hashlib.sha256(intake.read_bytes()).hexdigest()
    binding = json.dumps([{"installation_receipt_path": str(receipt), "publisher_intake_path": str(intake), "publisher_intake_sha256": digest}])
    units = {
        "blueprint-task-evaluation-launch-preparation.service": {
            "effective_environment": {"BLUEPRINT_TASK_EVALUATION_INSTALLED_SOURCE_BINDINGS_JSON": binding, "BLUEPRINT_TASK_EVALUATION_SAM31_PREPARATION_PROFILE_FILE": ""},
            "environment_files": [],
        },
        "blueprint-task-evaluation-sam31-preparation-execution.service": {"effective_environment": {}, "environment_files": []},
    }

    findings = preflight.binding_checks(units, "a" * 40, (os.getuid(), os.getgid()))

    codes = {f["code"]: f["severity"] for f in findings}
    assert codes["installation_receipt_installed_by_other_release"] == "info"
    assert "installation_receipt_bound_to_other_release" not in codes
    assert codes["sam31_profile_unbound"] == "blocker"
    receipt.write_text(json.dumps({"source_commit_sha": "nope"}), encoding="utf-8")
    codes = {f["code"]: f["severity"] for f in preflight.binding_checks(units, "a" * 40, (os.getuid(), os.getgid()))}
    assert codes["installation_receipt_commit_invalid"] == "blocker"


def test_probe_replays_the_paid_admission_identity_gate_from_the_release(tmp_path: Path) -> None:
    """Submission #8 (2026-09-05): every static check was clean and the paid
    allocator refused the release at admission because main had moved after the
    deploy.  The probe now runs that gate from the release tree, so the refusal
    is a pre-submission finding, and it reports the evidence grade the release
    will stamp on its runs."""
    src = _package(tmp_path)
    package = src / "blueprint_pipeline"
    (package / "paid_resource_allocator.py").write_text(
        textwrap.dedent(
            """
            def _source_checkout_blockers(expected, *, allow_pushed_branch_diagnostic=False):
                return (["gpu_canary_checkout_not_remote_main"], expected)


            def _control_plane_checkout_blockers():
                return ([], {"release_promotion_eligible": False, "evidence_grade_ceiling": "development_only"})
            """
        ),
        encoding="utf-8",
    )
    # The SAM child reaches the allocator by module name on an argv, not by import.
    (package / "admitter.py").write_text(
        'ARGV = ["python", "-m", "blueprint_pipeline.paid_resource_allocator", "gpu-canary"]\n', encoding="utf-8"
    )
    args = preflight.argparse.Namespace(
        unit="blueprint-task-evaluation-sam31-preparation-execution.service",
        module="blueprint_pipeline.admitter",
        release=str(tmp_path / "release"),
        active_sha="a" * 40,
        read_write_paths="",
        read_only_paths="",
    )

    report = preflight.run_probe(args)

    findings = {(row["code"], row.get("blocker")) for row in report["findings"]}
    assert ("paid_admission_checkout_identity_refused", "gpu_canary_checkout_not_remote_main") in findings
    assert ("release_evidence_grade_development_only", None) in findings
    assert report["paid_admission_identity"]["observed_commit"] == "a" * 40
    assert not any(row["code"] == "paid_admission_identity_probe_failed" for row in report["findings"])


def test_preflight_refuses_a_paid_unit_without_the_credit_guard_and_reads_the_credit_once(tmp_path: Path) -> None:
    """A paid unit that leaves BLUEPRINT_VAST_CREDIT_GUARD_ENABLED off would let an attempt start on credit
    Vast tears down mid-run; the preflight also reads the credit once ($0) so a thin account is named before
    a submission rather than at the guard."""

    from blueprint_pipeline import provider_credit_admission as credit

    key_file = tmp_path / "vast_api_key"
    key_file.write_text("k\n")
    dispatcher = "blueprint-task-evaluation-launch-dispatcher.service"
    canary = "blueprint-task-evaluation-policy-canary-dispatcher.service"
    units = {
        dispatcher: {"effective_environment": {"VAST_API_KEY_FILE": str(key_file), credit.ENABLED_ENV: "true"}},
        canary: {"effective_environment": {"VAST_API_KEY_FILE": str(key_file)}},
    }
    seen: list = []

    def observer(*, api_key, credit_usd=4.67):
        seen.append(api_key)
        return credit.observe_vast_credit(api_key=api_key, request=lambda **_kw: (200, {"credit": credit_usd}))

    findings = preflight.provider_credit_check(units, observer=observer)

    assert seen == ["k"]
    assert findings[0] == preflight._finding(
        "blocker", "provider_credit_guard_disabled", unit=canary, env=credit.ENABLED_ENV, value=None
    )
    assert findings[1]["severity"] == "warning" and findings[1]["code"] == "provider_credit_low"
    assert findings[1]["credit_usd"] == 4.67 and findings[1]["warning_usd"] == 5.0

    units[canary]["effective_environment"][credit.ENABLED_ENV] = "true"
    [fine] = preflight.provider_credit_check(units, observer=lambda *, api_key: observer(api_key=api_key, credit_usd=20.6))
    assert fine["severity"] == "info" and fine["code"] == "provider_credit_available" and fine["credit_usd"] == 20.6
    [exhausted] = preflight.provider_credit_check(units, observer=lambda *, api_key: observer(api_key=api_key, credit_usd=0.2))
    assert exhausted["severity"] == "blocker" and exhausted["code"] == "provider_credit_exhausted"
    [unknown] = preflight.provider_credit_check(
        units, observer=lambda *, api_key: credit.observe_vast_credit(api_key=api_key, request=lambda **_kw: (503, {}))
    )
    assert unknown["severity"] == "warning" and unknown["code"] == "provider_credit_unverifiable"
    assert unknown["blockers"] == ["provider_credit_unverifiable"] and unknown["http_status"] == 503
    assert preflight.provider_credit_check({"other.service": {"effective_environment": {}}}, observer=observer) == [
        preflight._finding("warning", "provider_credit_key_file_unset")
    ]


def _probe_argv(command: list[str]) -> list[str]:
    """The argv the probe subcommand actually receives, from ``probe`` onward.

    ``systemd_run_command`` appends ``-- <python> <script> probe ...`` so the
    real subparser sees everything after ``python`` and ``script``.
    """

    tail = command[command.index("--") + 1 :]
    assert tail[2] == "probe", tail[:3]
    return tail[2:]


def test_sandbox_probe_paths_survive_the_real_parser_even_with_leading_dash(tmp_path: Path) -> None:
    """systemd path directives may start with ``-`` (optional path).  Serialised
    as a separate ``--read-only-paths`` ``-/x`` pair, argparse read the value as
    another option and failed the probe with ``expected one argument`` — exactly
    what blocked the live intake and gpu-spend-guard units.  The ``--opt=value``
    form keeps every value one argv element through the real parser."""

    parser = preflight.build_parser()
    # (read_only, read_write, read_only tokens once the probe shlex-splits it, read_write tokens)
    cases = {
        "empty": ("", "", [], []),
        "leading_dash": (
            "-/var/lib/blueprint/pipeline-control-plane/release-retention",
            "/var/lib/blueprint",
            ["-/var/lib/blueprint/pipeline-control-plane/release-retention"],
            ["/var/lib/blueprint"],
        ),
        "leading_dash_and_second_path": (
            "-/var/lib/blueprint/a /var/lib/blueprint/b",
            "-/etc/blueprint/x /etc/blueprint/y",
            ["-/var/lib/blueprint/a", "/var/lib/blueprint/b"],
            ["-/etc/blueprint/x", "/etc/blueprint/y"],
        ),
        "space_containing": (
            '"/var/lib/blue print"',
            "/var/lib/blueprint",
            ["/var/lib/blue print"],
            ["/var/lib/blueprint"],
        ),
        "multiple_plain": (
            "/var/lib/blueprint/a /var/lib/blueprint/b",
            "/etc/blueprint/profiles",
            ["/var/lib/blueprint/a", "/var/lib/blueprint/b"],
            ["/etc/blueprint/profiles"],
        ),
    }
    for label, (read_only, read_write, ro_tokens, rw_tokens) in cases.items():
        props = {"ReadOnlyPaths": [read_only], "ReadWritePaths": [read_write]}
        command = preflight.systemd_run_command(
            unit="blueprint-x.service",
            props=props,
            directives=set(),
            python="/usr/bin/python3",
            script=tmp_path / "probe.py",
            module="blueprint_pipeline.worker",
            release=tmp_path / "release",
            active_sha="a" * 40,
        )
        probe_argv = _probe_argv(command)
        # Each path is exactly one argv element, so a leading dash or an embedded
        # space never spills into another token or trips the parser.
        assert f"--read-only-paths={read_only}" in probe_argv, label
        assert f"--read-write-paths={read_write}" in probe_argv, label
        namespace = parser.parse_args(probe_argv)  # the real argparse; must not raise SystemExit
        assert namespace.read_only_paths == read_only, label
        assert namespace.read_write_paths == read_write, label
        # The probe tokenises with shlex; multi-path, optional-path and space
        # semantics must survive the round trip unchanged.
        assert shlex.split(namespace.read_only_paths or "") == ro_tokens, label
        assert shlex.split(namespace.read_write_paths or "") == rw_tokens, label


def test_git_safe_directory_value_env_names_pairs_key_and_value() -> None:
    environ = {
        "GIT_CONFIG_COUNT": "2",
        "GIT_CONFIG_KEY_0": "safe.directory",
        "GIT_CONFIG_VALUE_0": "/opt/blueprint/task-evaluation-control-plane-releases/" + "c" * 40,
        "GIT_CONFIG_KEY_1": "http.sslVerify",
        "GIT_CONFIG_VALUE_1": "false",
        "GIT_CONFIG_KEY_2": " safe.directory ",
        "GIT_CONFIG_VALUE_2": "/opt/blueprint/task-evaluation-control-plane-releases/" + "d" * 40,
        "PATH": "/usr/bin",
    }
    assert preflight._git_safe_directory_value_env_names(environ) == {"GIT_CONFIG_VALUE_0", "GIT_CONFIG_VALUE_2"}
    assert preflight._git_safe_directory_value_env_names({}) == set()


def test_probe_reclassifies_safe_directory_release_waiver_but_keeps_real_binding(tmp_path: Path, monkeypatch) -> None:
    """An old-release path reached through a git ``safe.directory`` env waiver is
    a trust declaration, not a runtime binding, so it must be an ``info`` finding.
    A genuine code-literal binding to another release stays a blocker."""

    src = tmp_path / "release" / "src"
    package = src / "blueprint_pipeline"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "worker.py").write_text(
        "from blueprint_pipeline import binding\ndef tick():\n    return binding.OLD_RELEASE\n",
        encoding="utf-8",
    )
    real_binding_sha = "b" * 40
    (package / "binding.py").write_text(
        f'OLD_RELEASE = "/opt/blueprint/task-evaluation-control-plane-releases/{real_binding_sha}/src"\n',
        encoding="utf-8",
    )
    waiver_sha = "c" * 40
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", "safe.directory")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", f"/opt/blueprint/task-evaluation-control-plane-releases/{waiver_sha}")

    args = preflight.argparse.Namespace(
        unit="blueprint-x.service",
        module="blueprint_pipeline.worker",
        release=str(tmp_path / "release"),
        active_sha="a" * 40,
        read_write_paths="",
        read_only_paths="",
    )
    report = preflight.run_probe(args)
    findings = report["findings"]

    # The real binding (named as a code literal) is still a blocker.
    binding_blockers = [
        f for f in findings if f["code"] == "path_bound_to_other_release" and f["bound_sha"] == real_binding_sha
    ]
    assert len(binding_blockers) == 1
    assert binding_blockers[0]["severity"] == "blocker"
    assert binding_blockers[0]["source"] == "code_literal"

    # The safe.directory waiver is preserved but reclassified to info.
    waiver = [f for f in findings if f["code"] == "git_safe_directory_names_other_release"]
    assert len(waiver) == 1
    assert waiver[0]["severity"] == "info"
    assert waiver[0]["bound_sha"] == waiver_sha
    assert waiver[0]["source"] == "env:GIT_CONFIG_VALUE_0"

    # The waiver is never counted as a binding blocker.
    assert not any(
        f["code"] == "path_bound_to_other_release" and f["bound_sha"] == waiver_sha for f in findings
    )
