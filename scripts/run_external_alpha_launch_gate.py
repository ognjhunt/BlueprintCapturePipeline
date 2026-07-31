#!/usr/bin/env python3
"""Cross-repo external alpha launch gate for BlueprintCapture + BlueprintCapturePipeline."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from blueprint_pipeline.safe_env import contract_test_env, load_env_files  # noqa: E402
from blueprint_pipeline.artifact_storage import default_evidence_root  # noqa: E402
from blueprint_pipeline.source_metadata import git_source_metadata  # noqa: E402
from scripts.validate_capture_truth_backup_policy import validate_backup_policy  # noqa: E402
from scripts.validate_beta_capacity_storage import validate_files as validate_beta_capacity_storage_files  # noqa: E402


def _repo_root() -> Path:
    return REPO_ROOT


def _default_capture_repo() -> Path:
    return _repo_root().parent / "BlueprintCapture"


def _default_webapp_repo() -> Path:
    return _repo_root().parent / "Blueprint-WebApp"


def _desktop_capture_repo() -> Path:
    return Path.home() / "Desktop" / "BlueprintCapture"


DEFAULT_IOS_TEST_TIMEOUT_SECONDS = 900
ANDROID_SDK_MISSING_REASON = "ANDROID_HOME or ANDROID_SDK_ROOT is not configured in this shell."
SPEND_GUARD_SCHEMA_VERSION = "gpu_spend_guard.v1"
DEFAULT_SPEND_GUARD_SNAPSHOT_MAX_AGE_SECONDS = 15 * 60


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
    if not cwd.is_dir():
        raise RuntimeError(f"Required working directory is missing: {cwd}")
    try:
        subprocess.run(command, cwd=cwd, check=True, env=env, timeout=timeout_seconds)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"Command timed out after {timeout_seconds} seconds: {printable}"
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"Command failed with exit code {exc.returncode}: {printable}"
        ) from exc
    except FileNotFoundError as exc:
        raise RuntimeError(f"Command could not start: {printable}: {exc}") from exc


def _ensure_extract_frames_dependencies(extract_frames_dir: Path) -> None:
    if not extract_frames_dir.is_dir():
        raise RuntimeError(
            "Capture cloud extract-frames directory is missing: "
            f"{extract_frames_dir}. Pass --capture-repo or --skip-capture-cloud "
            "if this checkout is intentionally unavailable."
        )
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


def _android_sdk_root_from_env_or_common_paths() -> Path | None:
    for key in ("ANDROID_HOME", "ANDROID_SDK_ROOT"):
        value = str(os.getenv(key) or "").strip()
        if value:
            path = Path(value).expanduser()
            if _looks_like_android_sdk(path):
                return path
    for path in (
        Path.home() / "Library" / "Android" / "sdk",
        Path.home() / "Android" / "Sdk",
        Path("/opt/android-sdk"),
        Path("/usr/local/share/android-sdk"),
        Path("/opt/homebrew/share/android-sdk"),
    ):
        if _looks_like_android_sdk(path):
            return path
    return None


def _looks_like_android_sdk(path: Path) -> bool:
    return path.is_dir() and (
        (path / "platform-tools" / "adb").is_file()
        or (path / "platform-tools" / "adb.exe").is_file()
        or (path / "platforms").is_dir()
        or (path / "cmdline-tools").is_dir()
    )


def _android_subprocess_env(sdk_root: Path | None) -> dict[str, str] | None:
    if sdk_root is None:
        return None
    env = os.environ.copy()
    env.setdefault("ANDROID_HOME", str(sdk_root))
    env.setdefault("ANDROID_SDK_ROOT", str(sdk_root))
    return env


def _android_skip_reason(android_dir: Path) -> str | None:
    if _android_sdk_root_from_env_or_common_paths() is None:
        return ANDROID_SDK_MISSING_REASON
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


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_iso(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _record_check(
    report: dict[str, Any],
    *,
    check_id: str,
    status: str,
    **extra: Any,
) -> None:
    row = {"id": check_id, "status": status}
    row.update(extra)
    report["checks"].append(row)


def _spend_guard_snapshot_check(
    *,
    path: Path | None,
    snapshot_label: str,
    max_age_seconds: int,
    now: datetime | None = None,
) -> dict[str, Any]:
    blockers: list[str] = []
    payload: dict[str, Any] = {}
    if path is None:
        blockers.append("spend_guard_snapshot_path_missing")
    elif not path.is_file():
        blockers.append("spend_guard_snapshot_file_missing")
    else:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(raw, Mapping):
                payload = dict(raw)
            else:
                blockers.append("spend_guard_snapshot_not_json_object")
        except (OSError, json.JSONDecodeError):
            blockers.append("spend_guard_snapshot_parse_failed")

    generated_at = _parse_iso(payload.get("generated_at")) if payload else None
    age_seconds: float | None = None
    if payload:
        if payload.get("schema_version") != SPEND_GUARD_SCHEMA_VERSION:
            blockers.append("spend_guard_snapshot_schema_invalid")
        if generated_at is None:
            blockers.append("spend_guard_snapshot_generated_at_invalid")
        else:
            age_seconds = ((now or datetime.now(timezone.utc)) - generated_at).total_seconds()
            if age_seconds < 0 or age_seconds > max_age_seconds:
                blockers.append("spend_guard_snapshot_stale")
        if payload.get("reap_mode") is not True:
            blockers.append("spend_guard_snapshot_reap_mode_not_true")
        if payload.get("reap_candidate_ids"):
            blockers.append("spend_guard_snapshot_has_reap_candidates")
        fleet_budget = payload.get("fleet_budget")
        if not isinstance(fleet_budget, Mapping):
            blockers.append("spend_guard_snapshot_fleet_budget_missing")
        elif fleet_budget.get("status") != "passed":
            blockers.append("spend_guard_snapshot_fleet_budget_not_passed")
        failed_reaps = [
            item
            for item in payload.get("reap_results") or []
            if isinstance(item, Mapping)
            and item.get("status") not in {None, "terminated", "deleted", "stopped"}
        ]
        if failed_reaps:
            blockers.append("spend_guard_snapshot_has_failed_reap_results")

    return {
        "id": f"gpu_spend_guard_{snapshot_label}_snapshot",
        "status": "passed" if not blockers else "failed",
        "path": str(path) if path else None,
        "blockers": sorted(set(blockers)),
        "generated_at": payload.get("generated_at"),
        "age_seconds": age_seconds,
        "live_instance_count": payload.get("live_instance_count"),
        "total_burn_per_hour_usd": payload.get("total_burn_per_hour_usd"),
        "fleet_budget": payload.get("fleet_budget"),
        "reap_candidate_ids": payload.get("reap_candidate_ids") or [],
        "max_age_seconds": max_age_seconds,
    }


def _write_report(
    *,
    pipeline_repo: Path,
    report: dict[str, Any],
    json_out: str | Path | None,
    markdown_out: str | Path | None,
) -> tuple[Path, Path]:
    json_path = (
        Path(json_out).expanduser()
        if json_out
        else default_evidence_root() / "external_alpha_launch_gate.json"
    )
    markdown_path = (
        Path(markdown_out).expanduser()
        if markdown_out
        else default_evidence_root() / "external_alpha_launch_gate.md"
    )
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    markdown_path.write_text(_render_markdown(report), encoding="utf-8")
    return json_path, markdown_path


def _render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# External Alpha Launch Gate",
        "",
        f"Overall status: `{report.get('overall_status')}`",
        "",
        "This is an automated cross-repo contract gate, not live deployment, Android device, payment, or legal proof.",
        "",
        "## Checks",
    ]
    for check in report.get("checks") or []:
        if not isinstance(check, Mapping):
            continue
        detail = check.get("detail") or check.get("reason") or check.get("error") or ""
        suffix = f" - {detail}" if detail else ""
        lines.append(f"- `{check.get('id')}`: `{check.get('status')}`{suffix}")
    if report.get("overall_status") == "automation_failed":
        lines.extend(
            [
                "",
                "## Failure",
                "",
                f"- `{report.get('error_type')}`: {report.get('error')}",
            ]
        )
    if any(
        isinstance(check, Mapping) and check.get("status") == "manual_required"
        for check in report.get("checks") or []
    ):
        lines.extend(
            [
                "",
                "## Manual Required",
                "",
                "Manual-required rows are intentionally not counted as proof. They must be replaced by executed evidence before making that launch claim.",
            ]
        )
    lines.append("")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-repo", type=Path, default=_default_capture_repo())
    parser.add_argument("--webapp-repo", type=Path, default=_default_webapp_repo())
    parser.add_argument("--pipeline-repo", type=Path, default=_repo_root())
    parser.add_argument("--skip-storage-rules-parity", action="store_true")
    parser.add_argument("--skip-storage-lifecycle", action="store_true")
    parser.add_argument("--skip-backup-drill", action="store_true")
    parser.add_argument("--backup-drill-artifact", type=Path)
    parser.add_argument("--skip-ios", action="store_true")
    parser.add_argument("--skip-android", action="store_true")
    parser.add_argument("--skip-capture-cloud", action="store_true")
    parser.add_argument("--skip-pipeline", action="store_true")
    parser.add_argument("--skip-spend-guard", action="store_true")
    parser.add_argument("--spend-guard-pre-snapshot", type=Path)
    parser.add_argument("--spend-guard-post-snapshot", type=Path)
    parser.add_argument(
        "--spend-guard-max-age-seconds",
        type=_positive_int,
        default=DEFAULT_SPEND_GUARD_SNAPSHOT_MAX_AGE_SECONDS,
    )
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
    parser.add_argument("--json-out")
    parser.add_argument("--markdown-out")
    args = parser.parse_args(argv)

    capture_repo = args.capture_repo.resolve()
    webapp_repo = args.webapp_repo.resolve()
    pipeline_repo = args.pipeline_repo.resolve()
    desktop_repo = _desktop_capture_repo()
    load_env_files([pipeline_repo, capture_repo])
    report: dict[str, Any] = {
        "schema_version": "external_alpha_launch_gate.v1",
        "generated_at": _utc_now_iso(),
        "pipeline_source": git_source_metadata(
            pipeline_repo,
            repo_name="BlueprintCapturePipeline",
        ),
        "capture_repo": str(capture_repo),
        "webapp_repo": str(webapp_repo),
        "pipeline_repo": str(pipeline_repo),
        "checks": [],
    }

    print(f"[external-alpha-gate] canonical capture repo: {capture_repo}")
    print(f"[external-alpha-gate] canonical webapp repo: {webapp_repo}")
    print(f"[external-alpha-gate] canonical pipeline repo: {pipeline_repo}")
    if desktop_repo.exists() and desktop_repo.resolve() != capture_repo:
        print(
            f"[external-alpha-gate] note: {desktop_repo} exists but is treated as stale; "
            f"all checks run against {capture_repo}"
        )

    try:
        if args.skip_storage_rules_parity:
            _record_check(
                report,
                check_id="storage_rules_cross_repo_parity",
                status="skipped",
                detail=(
                    "Storage-rules parity skipped; this is not proof that WebApp "
                    "and Capture deploy the same Firebase Storage ruleset."
                ),
            )
        else:
            _run(
                [
                    "bash",
                    "scripts/check-storage-rules-parity.sh",
                    "--ios-rules",
                    str(capture_repo / "storage.rules"),
                ],
                cwd=webapp_repo,
            )
            _record_check(
                report,
                check_id="storage_rules_cross_repo_parity",
                status="passed",
                webapp_rules=str(webapp_repo / "storage.rules"),
                capture_rules=str(capture_repo / "storage.rules"),
            )

        if args.skip_storage_lifecycle:
            _record_check(
                report,
                check_id="primary_capture_bucket_lifecycle_contract",
                status="skipped",
                detail=(
                    "Primary capture bucket lifecycle validation skipped; this is not "
                    "proof that raw, temporary, hosted, or buyer-delivery storage "
                    "retention is bounded."
                ),
            )
        else:
            lifecycle_result = validate_beta_capacity_storage_files(pipeline_repo)
            _record_check(
                report,
                check_id="primary_capture_bucket_lifecycle_contract",
                status="passed",
                lifecycle_path=lifecycle_result.get("lifecycle_path"),
                target_concurrent_uploaders=lifecycle_result.get("target_concurrent_uploaders"),
                external_users=lifecycle_result.get("external_users"),
                modeled_captures_per_month=lifecycle_result.get("modeled_captures_per_month"),
            )

        if args.skip_backup_drill:
            _record_check(
                report,
                check_id="capture_truth_backup_restore_drill",
                status="skipped",
                detail=(
                    "Capture-truth backup/restore drill skipped; this is not proof "
                    "that Firestore or primary-bucket restore works."
                ),
            )
        else:
            backup_result = validate_backup_policy(
                pipeline_repo,
                args.backup_drill_artifact.resolve() if args.backup_drill_artifact else None,
                require_restore_drill=True,
            )
            _record_check(
                report,
                check_id="capture_truth_backup_restore_drill",
                status="passed",
                script=backup_result.get("script"),
                runbook=backup_result.get("runbook"),
                restore_drill_artifact=backup_result.get("restore_drill_artifact"),
                source_project_id=backup_result.get("source_project_id"),
                restore_project_id=backup_result.get("restore_project_id"),
            )

        if args.skip_spend_guard:
            _record_check(
                report,
                check_id="gpu_spend_guard_snapshots",
                status="skipped",
                detail=(
                    "Spend-guard snapshot proof skipped; this is not production "
                    "provider-spend readiness evidence."
                ),
            )
        else:
            failed_spend_checks: list[dict[str, Any]] = []
            for label, path in (
                ("pre", args.spend_guard_pre_snapshot),
                ("post", args.spend_guard_post_snapshot),
            ):
                check = _spend_guard_snapshot_check(
                    path=path.resolve() if path else None,
                    snapshot_label=label,
                    max_age_seconds=args.spend_guard_max_age_seconds,
                )
                check_id = str(check.pop("id"))
                status = str(check.pop("status"))
                _record_check(report, check_id=check_id, status=status, **check)
                if status != "passed":
                    failed_spend_checks.append({"id": check_id, **check})
            if failed_spend_checks:
                summary = "; ".join(
                    f"{check['id']}:{','.join(check.get('blockers') or [])}"
                    for check in failed_spend_checks
                )
                raise RuntimeError(f"GPU spend guard snapshot gate failed: {summary}")

        if args.skip_capture_cloud:
            _record_check(report, check_id="capture_cloud_extract_frames", status="skipped")
        else:
            _ensure_extract_frames_dependencies(capture_repo / "cloud" / "extract-frames")
            _run(["npm", "test"], cwd=capture_repo / "cloud" / "extract-frames")
            _record_check(report, check_id="capture_cloud_extract_frames", status="passed")

        if args.skip_ios:
            _record_check(report, check_id="ios_capture_contract_tests", status="skipped")
        else:
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
            _record_check(
                report,
                check_id="ios_capture_contract_tests",
                status="passed",
                simulator_destination=simulator_destination,
                timeout_seconds=args.ios_test_timeout_seconds,
            )

        if args.skip_android:
            _record_check(report, check_id="android_capture_contract_tests", status="skipped")
        else:
            android_dir = capture_repo / "android"
            android_skip_reason = _android_skip_reason(android_dir)
            if android_skip_reason and not args.require_android:
                print(f"[external-alpha-gate] android manual_required: {android_skip_reason}")
                _record_check(
                    report,
                    check_id="android_capture_contract_tests",
                    status="manual_required",
                    reason=android_skip_reason,
                )
            elif android_skip_reason:
                _record_check(
                    report,
                    check_id="android_capture_contract_tests",
                    status="failed",
                    reason=android_skip_reason,
                )
                raise RuntimeError(android_skip_reason)
            else:
                android_sdk_root = _android_sdk_root_from_env_or_common_paths()
                _run(
                    ["./gradlew", "testDebugUnitTest", "assembleDebug"],
                    cwd=android_dir,
                    env=_android_subprocess_env(android_sdk_root),
                )
                _record_check(
                    report,
                    check_id="android_capture_contract_tests",
                    status="passed",
                    android_sdk_root=str(android_sdk_root) if android_sdk_root else None,
                    android_sdk_auto_detected=not (
                        os.getenv("ANDROID_HOME") or os.getenv("ANDROID_SDK_ROOT")
                    ),
                )

        if args.skip_pipeline:
            _record_check(report, check_id="pipeline_alpha_contract_tests", status="skipped")
        else:
            _run(
                _pipeline_pytest_command(),
                cwd=pipeline_repo,
                env=contract_test_env(),
            )
            _record_check(report, check_id="pipeline_alpha_contract_tests", status="passed")
    except Exception as exc:
        report["overall_status"] = "automation_failed"
        report["error_type"] = type(exc).__name__
        report["error"] = str(exc)
        json_path, markdown_path = _write_report(
            pipeline_repo=pipeline_repo,
            report=report,
            json_out=args.json_out,
            markdown_out=args.markdown_out,
        )
        print(f"[external-alpha-gate] failed: {exc}", file=sys.stderr)
        print(f"[external-alpha-gate] json={json_path}")
        print(f"[external-alpha-gate] markdown={markdown_path}")
        return 1

    manual_required = any(check["status"] == "manual_required" for check in report["checks"])
    report["overall_status"] = "passed_manual_required" if manual_required else "passed"
    json_path, markdown_path = _write_report(
        pipeline_repo=pipeline_repo,
        report=report,
        json_out=args.json_out,
        markdown_out=args.markdown_out,
    )
    print(f"[external-alpha-gate] overall_status={report['overall_status']}")
    print(f"[external-alpha-gate] json={json_path}")
    print(f"[external-alpha-gate] markdown={markdown_path}")
    print("[external-alpha-gate] passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
