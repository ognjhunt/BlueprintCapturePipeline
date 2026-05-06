from __future__ import annotations

import importlib.util
import json
from pathlib import Path


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
    android_dir = tmp_path / "android"
    android_dir.mkdir()
    (android_dir / "gradlew").write_text("#!/usr/bin/env bash\n", encoding="utf-8")

    reason = gate._android_skip_reason(android_dir)

    assert reason == "ANDROID_HOME or ANDROID_SDK_ROOT is not configured in this shell."
