from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest


def _load_gate_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "run_external_alpha_launch_gate.py"
    spec = importlib.util.spec_from_file_location("run_external_alpha_launch_gate", module_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_resolves_duplicate_simulator_names_to_unique_udid(monkeypatch) -> None:
    gate = _load_gate_module()
    simctl_payload = {
        "devices": {
            "com.apple.CoreSimulator.SimRuntime.iOS-26-0": [
                {
                    "name": "iPhone 17 Pro",
                    "udid": "UDID-IOS-26",
                    "isAvailable": True,
                }
            ],
            "com.apple.CoreSimulator.SimRuntime.iOS-26-1": [
                {
                    "name": "iPhone 17 Pro",
                    "udid": "UDID-IOS-261",
                    "isAvailable": True,
                }
            ],
        }
    }

    monkeypatch.setattr(gate.subprocess, "check_output", lambda *_args, **_kwargs: json.dumps(simctl_payload))

    destination = gate._resolve_simulator_destination(
        preferred_name="iPhone 17 Pro",
        preferred_os=None,
        preferred_udid=None,
    )

    assert destination == "platform=iOS Simulator,id=UDID-IOS-261"


def test_explicit_simulator_udid_overrides_name_and_os(monkeypatch) -> None:
    gate = _load_gate_module()
    simctl_payload = {
        "devices": {
            "com.apple.CoreSimulator.SimRuntime.iOS-26-0": [
                {
                    "name": "iPhone 16",
                    "udid": "EXPLICIT-UDID",
                    "isAvailable": True,
                }
            ],
            "com.apple.CoreSimulator.SimRuntime.iOS-26-1": [
                {
                    "name": "iPhone 17 Pro",
                    "udid": "OTHER-UDID",
                    "isAvailable": True,
                }
            ],
        }
    }

    monkeypatch.setattr(gate.subprocess, "check_output", lambda *_args, **_kwargs: json.dumps(simctl_payload))

    destination = gate._resolve_simulator_destination(
        preferred_name="iPhone 17 Pro",
        preferred_os="26.1",
        preferred_udid="EXPLICIT-UDID",
    )

    assert destination == "platform=iOS Simulator,id=EXPLICIT-UDID"


def test_android_skip_reason_when_sdk_is_missing(monkeypatch, tmp_path: Path) -> None:
    gate = _load_gate_module()
    monkeypatch.delenv("ANDROID_HOME", raising=False)
    monkeypatch.delenv("ANDROID_SDK_ROOT", raising=False)
    monkeypatch.setattr(gate, "_android_sdk_root_from_env_or_common_paths", lambda: None)
    android_dir = tmp_path / "android"
    android_dir.mkdir()
    (android_dir / "gradlew").write_text("#!/usr/bin/env bash\n", encoding="utf-8")

    reason = gate._android_skip_reason(android_dir)

    assert reason == "ANDROID_HOME or ANDROID_SDK_ROOT is not configured in this shell."


def test_android_sdk_common_path_detection_forwards_env(monkeypatch, tmp_path: Path) -> None:
    gate = _load_gate_module()
    sdk_root = tmp_path / "Library" / "Android" / "sdk"
    platform_tools = sdk_root / "platform-tools"
    platform_tools.mkdir(parents=True)
    (platform_tools / "adb").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    monkeypatch.setattr(gate.Path, "home", staticmethod(lambda: tmp_path))
    monkeypatch.delenv("ANDROID_HOME", raising=False)
    monkeypatch.delenv("ANDROID_SDK_ROOT", raising=False)

    resolved = gate._android_sdk_root_from_env_or_common_paths()
    env = gate._android_subprocess_env(resolved)

    assert resolved == sdk_root
    assert env is not None
    assert env["ANDROID_HOME"] == str(sdk_root)
    assert env["ANDROID_SDK_ROOT"] == str(sdk_root)


def test_run_forwards_timeout_to_subprocess(monkeypatch, tmp_path: Path) -> None:
    gate = _load_gate_module()
    captured: dict[str, object] = {}

    def fake_run(cmd, *, cwd, check, env, timeout):  # type: ignore[no-untyped-def]
        captured["cmd"] = cmd
        captured["cwd"] = cwd
        captured["check"] = check
        captured["env"] = env
        captured["timeout"] = timeout

    monkeypatch.setattr(gate.subprocess, "run", fake_run)

    gate._run(["xcodebuild", "test"], cwd=tmp_path, timeout_seconds=123)

    assert captured == {
        "cmd": ["xcodebuild", "test"],
        "cwd": tmp_path,
        "check": True,
        "env": None,
        "timeout": 123,
    }


def test_run_turns_timeout_into_actionable_failure(monkeypatch, tmp_path: Path) -> None:
    gate = _load_gate_module()

    def fake_run(cmd, *, cwd, check, env, timeout):  # type: ignore[no-untyped-def]
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=timeout)

    monkeypatch.setattr(gate.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="Command timed out after 3 seconds: xcodebuild test"):
        gate._run(["xcodebuild", "test"], cwd=tmp_path, timeout_seconds=3)


def test_pipeline_gate_uses_current_python_interpreter() -> None:
    gate = _load_gate_module()

    command = gate._pipeline_pytest_command()

    assert command[:3] == [sys.executable, "-m", "pytest"]
    assert "tests/test_alpha_readiness.py" in command
    assert "tests/test_webapp_sync.py" in command


def test_main_writes_failure_artifacts_when_capture_repo_is_missing(tmp_path: Path) -> None:
    gate = _load_gate_module()
    json_out = tmp_path / "external_alpha_launch_gate.json"
    markdown_out = tmp_path / "external_alpha_launch_gate.md"

    exit_code = gate.main(
        [
            "--capture-repo",
            str(tmp_path / "missing-capture"),
            "--pipeline-repo",
            str(tmp_path),
            "--skip-ios",
            "--skip-android",
            "--skip-pipeline",
            "--skip-spend-guard",
            "--json-out",
            str(json_out),
            "--markdown-out",
            str(markdown_out),
        ]
    )

    assert exit_code == 1
    report = json.loads(json_out.read_text(encoding="utf-8"))
    assert report["overall_status"] == "automation_failed"
    assert report["error_type"] == "RuntimeError"
    assert "Capture cloud extract-frames directory is missing" in report["error"]
    assert markdown_out.is_file()
    assert "Overall status: `automation_failed`" in markdown_out.read_text(encoding="utf-8")


def test_main_writes_manual_required_android_artifact(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    gate = _load_gate_module()
    monkeypatch.delenv("ANDROID_HOME", raising=False)
    monkeypatch.delenv("ANDROID_SDK_ROOT", raising=False)
    monkeypatch.setattr(gate, "_android_sdk_root_from_env_or_common_paths", lambda: None)
    json_out = tmp_path / "external_alpha_launch_gate.json"
    markdown_out = tmp_path / "external_alpha_launch_gate.md"

    exit_code = gate.main(
        [
            "--capture-repo",
            str(tmp_path),
            "--pipeline-repo",
            str(tmp_path),
            "--skip-capture-cloud",
            "--skip-ios",
            "--skip-pipeline",
            "--skip-spend-guard",
            "--json-out",
            str(json_out),
            "--markdown-out",
            str(markdown_out),
        ]
    )

    assert exit_code == 0
    report = json.loads(json_out.read_text(encoding="utf-8"))
    assert report["overall_status"] == "passed_manual_required"
    android = next(
        check for check in report["checks"] if check["id"] == "android_capture_contract_tests"
    )
    assert android["status"] == "manual_required"
    assert "ANDROID_HOME or ANDROID_SDK_ROOT" in android["reason"]
    markdown = markdown_out.read_text(encoding="utf-8")
    assert "Overall status: `passed_manual_required`" in markdown
    assert "Manual-required rows are intentionally not counted as proof" in markdown


def test_main_blocks_when_spend_guard_snapshots_are_missing(tmp_path: Path) -> None:
    gate = _load_gate_module()
    json_out = tmp_path / "external_alpha_launch_gate.json"
    markdown_out = tmp_path / "external_alpha_launch_gate.md"

    exit_code = gate.main(
        [
            "--capture-repo",
            str(tmp_path),
            "--pipeline-repo",
            str(tmp_path),
            "--skip-capture-cloud",
            "--skip-ios",
            "--skip-android",
            "--skip-pipeline",
            "--json-out",
            str(json_out),
            "--markdown-out",
            str(markdown_out),
        ]
    )

    assert exit_code == 1
    report = json.loads(json_out.read_text(encoding="utf-8"))
    assert report["overall_status"] == "automation_failed"
    pre = next(
        check for check in report["checks"] if check["id"] == "gpu_spend_guard_pre_snapshot"
    )
    post = next(
        check for check in report["checks"] if check["id"] == "gpu_spend_guard_post_snapshot"
    )
    assert pre["status"] == "failed"
    assert post["status"] == "failed"
    assert pre["blockers"] == ["spend_guard_snapshot_path_missing"]
    assert post["blockers"] == ["spend_guard_snapshot_path_missing"]


def test_spend_guard_snapshot_check_requires_fresh_passed_fleet_budget(
    tmp_path: Path,
) -> None:
    gate = _load_gate_module()
    now = datetime(2026, 7, 9, 12, 0, tzinfo=timezone.utc)
    snapshot = tmp_path / "gpu_spend_guard.json"
    snapshot.write_text(
        json.dumps(
            {
                "schema_version": "gpu_spend_guard.v1",
                "generated_at": "2026-07-09T11:59:00+00:00",
                "reap_mode": True,
                "live_instance_count": 0,
                "total_burn_per_hour_usd": 0.0,
                "reap_candidate_ids": [],
                "fleet_budget": {
                    "schema_version": "gpu_fleet_budget_guard.v1",
                    "status": "passed",
                    "blockers": [],
                },
            }
        ),
        encoding="utf-8",
    )

    check = gate._spend_guard_snapshot_check(
        path=snapshot,
        snapshot_label="pre",
        max_age_seconds=300,
        now=now,
    )
    assert check["status"] == "passed"

    payload = json.loads(snapshot.read_text(encoding="utf-8"))
    payload["fleet_budget"]["status"] = "blocked"
    payload["fleet_budget"]["blockers"] = ["fleet_burn_rate_limit_exceeded"]
    snapshot.write_text(json.dumps(payload), encoding="utf-8")

    blocked = gate._spend_guard_snapshot_check(
        path=snapshot,
        snapshot_label="pre",
        max_age_seconds=300,
        now=now,
    )
    assert blocked["status"] == "failed"
    assert "spend_guard_snapshot_fleet_budget_not_passed" in blocked["blockers"]
