#!/usr/bin/env python3
"""Cross-repo external alpha launch gate for BlueprintCapture + BlueprintCapturePipeline."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from blueprint_pipeline.safe_env import contract_test_env, load_env_files  # noqa: E402


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_capture_repo() -> Path:
    return _repo_root().parent / "BlueprintCapture"


def _desktop_capture_repo() -> Path:
    return Path.home() / "Desktop" / "BlueprintCapture"


DEFAULT_IOS_TEST_TIMEOUT_SECONDS = 900


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _run(
    cmd: Iterable[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
    timeout_seconds: int | None = None,
) -> None:
    command = list(cmd)
    printable = " ".join(command)
    print(f"[external-alpha-gate] cwd={cwd}")
    print(f"[external-alpha-gate] $ {printable}")
    if timeout_seconds is not None:
        print(f"[external-alpha-gate] timeout_seconds={timeout_seconds}")
    try:
        subprocess.run(command, cwd=cwd, check=True, env=env, timeout=timeout_seconds)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"Command timed out after {timeout_seconds} seconds: {printable}"
        ) from exc


def _ensure_extract_frames_dependencies(extract_frames_dir: Path) -> None:
    if (extract_frames_dir / "node_modules" / ".bin" / "tsc").exists():
        return
    _run(["npm", "ci"], cwd=extract_frames_dir)


def _resolve_swift_packages(capture_repo: Path, derived_data_path: Path) -> None:
    cmd = [
        "xcodebuild",
        "-resolvePackageDependencies",
        "-project",
        "BlueprintCapture.xcodeproj",
        "-scheme",
        "BlueprintCapture",
        "-derivedDataPath",
        str(derived_data_path),
    ]
    try:
        _run(cmd, cwd=capture_repo)
    except subprocess.CalledProcessError:
        source_packages_dir = derived_data_path / "SourcePackages"
        print(f"[external-alpha-gate] repairing stale Swift package state at {source_packages_dir}")
        shutil.rmtree(source_packages_dir, ignore_errors=True)
        _run(cmd, cwd=capture_repo)


def _pipeline_pytest_command() -> list[str]:
    return [
        sys.executable,
        "-m",
        "pytest",
        "tests/test_alpha_readiness.py",
        "tests/test_qualification_alpha.py",
        "tests/test_site_world_packaging.py",
        "tests/test_storage_trigger.py",
        "tests/test_webapp_sync.py",
        "tests/test_world_model_candidate_parity.py",
    ]


def _android_skip_reason(android_dir: Path) -> str | None:
    if not (os.getenv("ANDROID_HOME") or os.getenv("ANDROID_SDK_ROOT")):
        return "ANDROID_HOME or ANDROID_SDK_ROOT is not configured in this shell."
    if not (android_dir / "gradlew").is_file():
        return "Android Gradle wrapper is missing."
    return None


def _parse_os_version(os_name: str) -> tuple[int, ...]:
    parts: list[int] = []
    for part in os_name.split("."):
        try:
            parts.append(int(part))
        except ValueError:
            parts.append(0)
    return tuple(parts)


def _available_ios_simulators() -> list[dict[str, str]]:
    raw = subprocess.check_output(
        ["xcrun", "simctl", "list", "devices", "available", "-j"],
        text=True,
    )
    payload = json.loads(raw)
    devices = payload.get("devices", {})
    candidates: list[dict[str, str]] = []
    for runtime, runtime_devices in devices.items():
        if not runtime.startswith("com.apple.CoreSimulator.SimRuntime.iOS-"):
            continue
        os_name = runtime.removeprefix("com.apple.CoreSimulator.SimRuntime.iOS-").replace("-", ".")
        for device in runtime_devices or []:
            if not device.get("isAvailable", True):
                continue
            name = str(device.get("name") or "").strip()
            udid = str(device.get("udid") or "").strip()
            if not name or not udid:
                continue
            candidates.append({"name": name, "os": os_name, "udid": udid})
    return candidates


def _simulator_description(candidates: list[dict[str, str]]) -> str:
    if not candidates:
        return "none"
    return ", ".join(
        f"{candidate['name']} iOS {candidate['os']} ({candidate['udid']})"
        for candidate in sorted(candidates, key=lambda item: (item["name"], _parse_os_version(item["os"]), item["udid"]))
    )


def _newest_simulator(candidates: list[dict[str, str]]) -> dict[str, str]:
    return max(candidates, key=lambda item: (_parse_os_version(item["os"]), item["name"], item["udid"]))


def _resolve_simulator_destination(
    *,
    preferred_name: str,
    preferred_os: str | None,
    preferred_udid: str | None,
) -> str:
    candidates = _available_ios_simulators()
    if preferred_udid:
        for candidate in candidates:
            if candidate["udid"] == preferred_udid:
                return f"platform=iOS Simulator,id={candidate['udid']}"
        raise RuntimeError(
            "Configured iOS simulator UDID was not found among available iOS simulators: "
            f"{preferred_udid}. Available: {_simulator_description(candidates)}"
        )

    named_candidates = [
        candidate
        for candidate in candidates
        if candidate["name"] == preferred_name
        and (preferred_os is None or candidate["os"] == preferred_os)
    ]
    if named_candidates:
        return f"platform=iOS Simulator,id={_newest_simulator(named_candidates)['udid']}"

    if preferred_os:
        raise RuntimeError(
            "No available iOS simulator matched "
            f"name={preferred_name!r} and os={preferred_os!r}. "
            f"Available: {_simulator_description(candidates)}"
        )

    iphone_candidates = [candidate for candidate in candidates if "iPhone" in candidate["name"]]
    if iphone_candidates:
        return f"platform=iOS Simulator,id={_newest_simulator(iphone_candidates)['udid']}"
    raise RuntimeError("No available iPhone simulator found for the external alpha launch gate.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-repo", type=Path, default=_default_capture_repo())
    parser.add_argument("--pipeline-repo", type=Path, default=_repo_root())
    parser.add_argument("--skip-ios", action="store_true")
    parser.add_argument("--skip-android", action="store_true")
    parser.add_argument("--skip-capture-cloud", action="store_true")
    parser.add_argument("--skip-pipeline", action="store_true")
    parser.add_argument("--require-android", action="store_true")
    parser.add_argument("--ios-simulator-udid", default=os.getenv("BLUEPRINT_IOS_SIMULATOR_UDID"))
    parser.add_argument("--ios-simulator-name", default=os.getenv("BLUEPRINT_IOS_SIMULATOR_NAME", "iPhone 17 Pro"))
    parser.add_argument("--ios-simulator-os", default=os.getenv("BLUEPRINT_IOS_SIMULATOR_OS"))
    parser.add_argument(
        "--ios-test-timeout-seconds",
        type=_positive_int,
        default=_positive_int(
            os.getenv("BLUEPRINT_IOS_TEST_TIMEOUT_SECONDS", str(DEFAULT_IOS_TEST_TIMEOUT_SECONDS))
        ),
        help=(
            "Hard timeout for the targeted iOS simulator test leg. "
            "Defaults to BLUEPRINT_IOS_TEST_TIMEOUT_SECONDS or 900 seconds."
        ),
    )
    args = parser.parse_args()

    capture_repo = args.capture_repo.resolve()
    pipeline_repo = args.pipeline_repo.resolve()
    desktop_repo = _desktop_capture_repo()
    load_env_files([pipeline_repo, capture_repo])

    print(f"[external-alpha-gate] canonical capture repo: {capture_repo}")
    print(f"[external-alpha-gate] canonical pipeline repo: {pipeline_repo}")
    if desktop_repo.exists() and desktop_repo.resolve() != capture_repo:
        print(
            f"[external-alpha-gate] note: {desktop_repo} exists but is treated as stale; "
            f"all checks run against {capture_repo}"
        )

    if not args.skip_capture_cloud:
        _ensure_extract_frames_dependencies(capture_repo / "cloud" / "extract-frames")
        _run(["npm", "test"], cwd=capture_repo / "cloud" / "extract-frames")

    if not args.skip_ios:
        derived_data_path = capture_repo / "build" / "DerivedData"
        _resolve_swift_packages(capture_repo, derived_data_path)
        simulator_destination = _resolve_simulator_destination(
            preferred_name=args.ios_simulator_name,
            preferred_os=args.ios_simulator_os,
            preferred_udid=args.ios_simulator_udid,
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
                simulator_destination,
                "-derivedDataPath",
                str(derived_data_path),
                "-only-testing:BlueprintCaptureTests/CaptureBundleAndInferenceTests",
                "-only-testing:BlueprintCaptureTests/PipelineContractTests",
                "-only-testing:BlueprintCaptureTests/RuntimeConfigTests",
            ],
            cwd=capture_repo,
            timeout_seconds=args.ios_test_timeout_seconds,
        )

    if not args.skip_android:
        android_dir = capture_repo / "android"
        android_skip_reason = _android_skip_reason(android_dir)
        if android_skip_reason and not args.require_android:
            print(f"[external-alpha-gate] android manual_required: {android_skip_reason}")
        elif android_skip_reason:
            raise RuntimeError(android_skip_reason)
        else:
            _run(["./gradlew", "testDebugUnitTest", "assembleDebug"], cwd=android_dir)

    if not args.skip_pipeline:
        _run(
            _pipeline_pytest_command(),
            cwd=pipeline_repo,
            env=contract_test_env(),
        )

    print("[external-alpha-gate] passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
