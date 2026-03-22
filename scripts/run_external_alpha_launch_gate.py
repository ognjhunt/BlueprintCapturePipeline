#!/usr/bin/env python3
"""Cross-repo external alpha launch gate for BlueprintCapture + BlueprintCapturePipeline."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_capture_repo() -> Path:
    return _repo_root().parent / "BlueprintCapture"


def _desktop_capture_repo() -> Path:
    return Path.home() / "Desktop" / "BlueprintCapture"


def _run(cmd: Iterable[str], *, cwd: Path) -> None:
    printable = " ".join(cmd)
    print(f"[external-alpha-gate] cwd={cwd}")
    print(f"[external-alpha-gate] $ {printable}")
    subprocess.run(list(cmd), cwd=cwd, check=True)


def _resolve_simulator_name(preferred_name: str, preferred_os: str | None) -> str:
    raw = subprocess.check_output(
        ["xcrun", "simctl", "list", "devices", "available", "-j"],
        text=True,
    )
    payload = json.loads(raw)
    devices = payload.get("devices", {})
    candidates: list[tuple[str, str]] = []
    for runtime, runtime_devices in devices.items():
        if not runtime.startswith("com.apple.CoreSimulator.SimRuntime.iOS-"):
            continue
        os_name = runtime.removeprefix("com.apple.CoreSimulator.SimRuntime.iOS-").replace("-", ".")
        for device in runtime_devices or []:
            if not device.get("isAvailable", True):
                continue
            name = str(device.get("name") or "").strip()
            if not name:
                continue
            candidates.append((name, os_name))
    for name, os_name in candidates:
        if name == preferred_name and (preferred_os is None or os_name == preferred_os):
            return name
    for name, _os_name in candidates:
        if "iPhone" in name:
            return name
    raise RuntimeError("No available iPhone simulator found for the external alpha launch gate.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-repo", type=Path, default=_default_capture_repo())
    parser.add_argument("--pipeline-repo", type=Path, default=_repo_root())
    parser.add_argument("--skip-ios", action="store_true")
    parser.add_argument("--skip-android", action="store_true")
    parser.add_argument("--skip-capture-cloud", action="store_true")
    parser.add_argument("--skip-pipeline", action="store_true")
    args = parser.parse_args()

    capture_repo = args.capture_repo.resolve()
    pipeline_repo = args.pipeline_repo.resolve()
    desktop_repo = _desktop_capture_repo()

    print(f"[external-alpha-gate] canonical capture repo: {capture_repo}")
    print(f"[external-alpha-gate] canonical pipeline repo: {pipeline_repo}")
    if desktop_repo.exists() and desktop_repo.resolve() != capture_repo:
        print(
            f"[external-alpha-gate] note: {desktop_repo} exists but is treated as stale; "
            f"all checks run against {capture_repo}"
        )

    if not args.skip_capture_cloud:
        _run(["npm", "test"], cwd=capture_repo / "cloud" / "extract-frames")

    if not args.skip_ios:
        simulator_name = _resolve_simulator_name(
            preferred_name=os.getenv("BLUEPRINT_IOS_SIMULATOR_NAME", "iPhone 17 Pro"),
            preferred_os=os.getenv("BLUEPRINT_IOS_SIMULATOR_OS"),
        )
        _run(
            [
                "xcodebuild",
                "test",
                "-project",
                "BlueprintCapture.xcodeproj",
                "-scheme",
                "BlueprintCapture",
                "-destination",
                f"platform=iOS Simulator,name={simulator_name}",
                "-derivedDataPath",
                "build/DerivedData",
                "-only-testing:BlueprintCaptureTests/CaptureBundleAndInferenceTests",
                "-only-testing:BlueprintCaptureTests/PipelineContractTests",
                "-only-testing:BlueprintCaptureTests/RuntimeConfigTests",
            ],
            cwd=capture_repo,
        )

    if not args.skip_android:
        _run(["./gradlew", "assembleDebug"], cwd=capture_repo / "android")

    if not args.skip_pipeline:
        _run(
            [
                "pytest",
                "tests/test_alpha_readiness.py",
                "tests/test_qualification_alpha.py",
                "tests/test_site_world_packaging.py",
                "tests/test_storage_trigger.py",
                "tests/test_webapp_sync.py",
                "tests/test_world_model_candidate_parity.py",
            ],
            cwd=pipeline_repo,
        )

    print("[external-alpha-gate] passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
