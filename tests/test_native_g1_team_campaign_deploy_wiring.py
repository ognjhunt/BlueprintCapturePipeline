"""The signed G1 team queue must survive a clean controller deployment."""

from __future__ import annotations

from pathlib import Path
import os
import subprocess



ROOT = Path(__file__).resolve().parents[1]
SERVICE = "blueprint-native-g1-team-campaign-dispatcher.service"
TIMER = "blueprint-native-g1-team-campaign-dispatcher.timer"
SETTLEMENT_SERVICE = "blueprint-native-g1-team-campaign-settlement.service"
SETTLEMENT_TIMER = "blueprint-native-g1-team-campaign-settlement.timer"


def _text(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_installed_dispatch_command_reaches_selected_queue_without_builtin_queue(tmp_path):
    """Execute the actual oneshot shell against inert commands, never a provider."""
    service = _text(f"deploy/systemd/{SERVICE}")
    line = next(line for line in service.splitlines() if line.startswith("ExecStart="))
    shell = line.removeprefix("ExecStart=/bin/bash -lc '").removesuffix("'").replace("$$", "$")
    commands = tmp_path / "bin"
    commands.mkdir()
    git = commands / "git"
    git.write_text("#!/bin/bash\nprintf '%s\\n' '" + "a" * 40 + "'\n")
    git.chmod(0o700)
    python = commands / "python"
    python.write_text('#!/bin/bash\nprintf "%s\\n" "$@" > "$CAPTURE_ARGUMENTS"\n')
    python.chmod(0o700)
    registry = tmp_path / "registry.json"
    registry.write_text("{}")
    selected = tmp_path / "selected-queue"
    selected.mkdir()
    capture = tmp_path / "args.txt"
    environment = {
        **os.environ, "PATH": str(commands) + os.pathsep + os.environ["PATH"],
        "CAPTURE_ARGUMENTS": str(capture),
        "BLUEPRINT_TASK_EVALUATION_CONTROL_PLANE_REPO": str(tmp_path),
        "BLUEPRINT_TASK_EVALUATION_CONTROL_PLANE_PYTHON": str(python),
        "BLUEPRINT_NATIVE_G1_TEAM_CAMPAIGN_REGISTRY_PATH": str(registry),
        "BLUEPRINT_NATIVE_G1_TEAM_CAMPAIGN_QUEUE_ROOT": str(tmp_path / "absent-builtins"),
        "BLUEPRINT_NATIVE_G1_TEAM_CAMPAIGN_WORK_ROOT": str(tmp_path / "builtin-work"),
        "BLUEPRINT_NATIVE_G1_TEAM_CAMPAIGN_EXECUTE": "true",
        "BLUEPRINT_NATIVE_G1_TEAM_POLICY_QUEUE_ROOT": str(selected),
        "BLUEPRINT_NATIVE_G1_TEAM_POLICY_WORK_ROOT": str(tmp_path / "selected-work"),
        "BLUEPRINT_NATIVE_G1_TEAM_POLICY_APPROVAL_ROOT": str(tmp_path / "operator-approvals"),
        "BLUEPRINT_NATIVE_G1_TEAM_POLICY_CREDENTIAL_REGISTRY": str(tmp_path / "private/registry.json"),
        "BLUEPRINT_NATIVE_G1_TEAM_POLICY_SONIC_ASSET_DIR": str(tmp_path / "sonic"),
        "BLUEPRINT_NATIVE_G1_TEAM_POLICY_TRUSTED_CLIENT": "blueprint-webapp",
    }
    completed = subprocess.run(["bash", "-c", shell], env=environment, capture_output=True, timeout=10)
    assert completed.returncode == 0
    assert capture.is_file(), "selected-only queue was silently skipped by installed shell"
    arguments = capture.read_text().splitlines()
    assert arguments[arguments.index("--selected-queue-root") + 1] == str(selected)
    assert arguments[arguments.index("--selected-approval-root") + 1] == environment["BLUEPRINT_NATIVE_G1_TEAM_POLICY_APPROVAL_ROOT"]
    assert arguments[arguments.index("--selected-trusted-client") + 1] == "blueprint-webapp"
    assert "--execute" in arguments


def test_installed_settlement_command_reaches_selected_queue_without_builtin_queue(tmp_path):
    service = _text(f"deploy/systemd/{SETTLEMENT_SERVICE}")
    line = next(line for line in service.splitlines() if line.startswith("ExecStart="))
    shell = line.removeprefix("ExecStart=/bin/bash -lc '").removesuffix("'").replace("$$", "$")
    executable = tmp_path / "python"
    capture = tmp_path / "args.txt"
    executable.write_text('#!/bin/bash\nprintf "%s\\n" "$@" > "$CAPTURE_ARGUMENTS"\n')
    executable.chmod(0o700)
    selected = tmp_path / "selected-queue"
    selected.mkdir()
    environment = {
        **os.environ, "CAPTURE_ARGUMENTS": str(capture),
        "BLUEPRINT_TASK_EVALUATION_CONTROL_PLANE_REPO": str(tmp_path),
        "BLUEPRINT_TASK_EVALUATION_CONTROL_PLANE_PYTHON": str(executable),
        "BLUEPRINT_NATIVE_G1_TEAM_CAMPAIGN_QUEUE_ROOT": str(tmp_path / "absent-builtins"),
        "BLUEPRINT_NATIVE_G1_TEAM_CAMPAIGN_WORK_ROOT": str(tmp_path / "builtin-work"),
        "BLUEPRINT_NATIVE_G1_TEAM_CAMPAIGN_RESULT_ROOT": str(tmp_path / "results"),
        "BLUEPRINT_NATIVE_G1_TEAM_POLICY_QUEUE_ROOT": str(selected),
        "BLUEPRINT_NATIVE_G1_TEAM_POLICY_WORK_ROOT": str(tmp_path / "selected-work"),
        "BLUEPRINT_PROVIDER_BILLING_AUDIT_ROOT": str(tmp_path / "audit"),
        "PIPELINE_SYNC_WEBAPP_URL": "https://tryblueprint.io",
    }
    completed = subprocess.run(["bash", "-c", shell], env=environment, capture_output=True, timeout=10)
    assert completed.returncode == 0, completed.stderr
    assert capture.is_file(), "selected-only settlement silently skipped"
    arguments = capture.read_text().splitlines()
    assert arguments[arguments.index("--selected-queue-root") + 1] == str(selected)
    assert arguments[arguments.index("--selected-work-root") + 1] == str(tmp_path / "selected-work")
    assert "--execute" not in arguments


def test_settlement_cli_keeps_builtin_delivery_visible_while_selected_waits(tmp_path, monkeypatch, capsys):
    from blueprint_pipeline import native_g1_team_campaign_settlement as worker
    builtin, selected = tmp_path / "builtin", tmp_path / "selected"
    builtin.mkdir()
    selected.mkdir()
    calls = []
    def selected_settle(**kwargs):
        calls.append("selected")
        return {"status": "awaiting_settlement_evidence"}
    def builtin_settle(**kwargs):
        calls.append("builtin")
        return {"status": "delivered_owner_only", "run_id": "builtin-one"}
    monkeypatch.setattr("blueprint_pipeline.native_g1_team_policy_settlement.settle_pending_g1_team_policies", selected_settle)
    monkeypatch.setattr(worker, "settle_pending_g1_team_campaigns", builtin_settle)
    assert worker.main([
        "--queue-root", str(builtin), "--work-root", str(tmp_path / "work"),
        "--selected-queue-root", str(selected), "--selected-work-root", str(tmp_path / "selected-work"),
        "--billing-audit-root", str(tmp_path / "audit"), "--result-root", str(tmp_path / "results"),
    ]) == 0
    assert calls == ["selected", "builtin"]
    assert '"run_id": "builtin-one"' in capsys.readouterr().out
