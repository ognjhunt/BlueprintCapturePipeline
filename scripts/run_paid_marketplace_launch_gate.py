#!/usr/bin/env python3
"""Run the paid marketplace beta launch gate across Blueprint repos."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from blueprint_pipeline.safe_env import contract_test_env, load_env_files  # noqa: E402
from blueprint_pipeline.source_metadata import git_source_metadata  # noqa: E402


@dataclass
class CommandSpec:
    id: str
    label: str
    repo: str
    cwd: Path
    command: list[str]
    blocking: bool = True
    source_tags: tuple[str, ...] = ()
    runs_when: str = "always"
    timeout_seconds: int | None = None
    preflight_failure: str | None = None


@dataclass
class CommandResult:
    id: str
    label: str
    repo: str
    command: list[str]
    cwd: str
    status: str
    blocking: bool
    source_tags: tuple[str, ...]
    exit_code: int | None = None
    stdout_tail: str = ""
    stderr_tail: str = ""
    skip_reason: str | None = None
    evidence_class: str | None = None
    evidence_note: str | None = None


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def tail_text(text: str, limit: int = 80) -> str:
    lines = [line for line in text.strip().splitlines() if line.strip()]
    return "\n".join(lines[-limit:])


DEFAULT_IOS_SIMULATOR_NAME = "iPhone 17 Pro"
DEFAULT_IOS_TEST_TIMEOUT_SECONDS = 900
ANDROID_SDK_MISSING_REASON = "ANDROID_HOME or ANDROID_SDK_ROOT is not configured in this shell."


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def parse_os_version(os_name: str) -> tuple[int, ...]:
    parts: list[int] = []
    for part in os_name.split("."):
        try:
            parts.append(int(part))
        except ValueError:
            parts.append(0)
    return tuple(parts)


def available_ios_simulators() -> list[dict[str, str]]:
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


def simulator_description(candidates: list[dict[str, str]]) -> str:
    if not candidates:
        return "none"
    return ", ".join(
        f"{candidate['name']} iOS {candidate['os']} ({candidate['udid']})"
        for candidate in sorted(
            candidates,
            key=lambda item: (item["name"], parse_os_version(item["os"]), item["udid"]),
        )
    )


def newest_simulator(candidates: list[dict[str, str]]) -> dict[str, str]:
    return max(candidates, key=lambda item: (parse_os_version(item["os"]), item["name"], item["udid"]))


def resolve_ios_simulator_destination(
    *,
    preferred_name: str,
    preferred_os: str | None,
    preferred_udid: str | None,
) -> str:
    candidates = available_ios_simulators()
    if preferred_udid:
        for candidate in candidates:
            if candidate["udid"] == preferred_udid:
                return f"platform=iOS Simulator,id={candidate['udid']}"
        raise RuntimeError(
            "Configured iOS simulator UDID was not found among available iOS simulators: "
            f"{preferred_udid}. Available: {simulator_description(candidates)}"
        )

    named_candidates = [
        candidate
        for candidate in candidates
        if candidate["name"] == preferred_name
        and (preferred_os is None or candidate["os"] == preferred_os)
    ]
    if named_candidates:
        return f"platform=iOS Simulator,id={newest_simulator(named_candidates)['udid']}"

    if preferred_os:
        raise RuntimeError(
            "No available iOS simulator matched "
            f"name={preferred_name!r} and os={preferred_os!r}. "
            f"Available: {simulator_description(candidates)}"
        )

    iphone_candidates = [candidate for candidate in candidates if "iPhone" in candidate["name"]]
    if iphone_candidates:
        return f"platform=iOS Simulator,id={newest_simulator(iphone_candidates)['udid']}"
    raise RuntimeError("No available iPhone simulator found for the paid marketplace launch gate.")


def _timeout_text(value: bytes | str | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _looks_like_android_sdk(path: Path) -> bool:
    return path.is_dir() and (
        (path / "platform-tools" / "adb").is_file()
        or (path / "platform-tools" / "adb.exe").is_file()
        or (path / "platforms").is_dir()
        or (path / "cmdline-tools").is_dir()
    )


def android_sdk_root_from_env_or_common_paths() -> Path | None:
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


def _command_env(spec: CommandSpec) -> dict[str, str]:
    env = contract_test_env()
    if spec.id == "android_bundle_contracts":
        sdk_root = android_sdk_root_from_env_or_common_paths()
        if sdk_root is not None:
            env.setdefault("ANDROID_HOME", str(sdk_root))
            env.setdefault("ANDROID_SDK_ROOT", str(sdk_root))
    return env


def run_command(spec: CommandSpec) -> CommandResult:
    if not spec.cwd.is_dir():
        return unavailable_result(
            spec,
            f"required_working_directory_missing: {spec.cwd}",
        )
    if spec.preflight_failure:
        return unavailable_result(spec, spec.preflight_failure)
    try:
        completed = subprocess.run(
            spec.command,
            cwd=spec.cwd,
            capture_output=True,
            text=True,
            env=_command_env(spec),
            timeout=spec.timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        return CommandResult(
            id=spec.id,
            label=spec.label,
            repo=spec.repo,
            command=spec.command,
            cwd=str(spec.cwd),
            status="failed",
            blocking=spec.blocking,
            source_tags=spec.source_tags,
            stdout_tail=tail_text(_timeout_text(exc.stdout)),
            stderr_tail=tail_text(_timeout_text(exc.stderr)),
            skip_reason=f"command_timed_out_after_{spec.timeout_seconds}_seconds",
            evidence_class="launch_gate_command_timeout",
            evidence_note="A required launch-gate command exceeded its configured timeout.",
        )
    except FileNotFoundError as exc:
        return unavailable_result(
            spec,
            f"command_or_working_directory_missing: {exc}",
        )
    status = "passed" if completed.returncode == 0 else "failed"
    return CommandResult(
        id=spec.id,
        label=spec.label,
        repo=spec.repo,
        command=spec.command,
        cwd=str(spec.cwd),
        status=status,
        blocking=spec.blocking,
        source_tags=spec.source_tags,
        exit_code=completed.returncode,
        stdout_tail=tail_text(completed.stdout),
        stderr_tail=tail_text(completed.stderr),
    )


def skip_evidence_class(spec: CommandSpec, reason: str) -> tuple[str, str] | tuple[None, None]:
    if spec.id == "android_bundle_contracts" and "ANDROID_HOME" in reason:
        return (
            "operator_toolchain_required",
            "Android SDK/Gradle unit evidence is unrun because this shell lacks ANDROID_HOME/ANDROID_SDK_ROOT; this is not product readiness or live-device proof.",
        )
    if spec.id == "ios_launch_contracts" and "xcodebuild" in reason:
        return (
            "operator_toolchain_required",
            "iOS simulator unit evidence is unrun because this shell lacks xcodebuild; this is not real-device proof.",
        )
    return (None, None)


def skipped_result(spec: CommandSpec, reason: str) -> CommandResult:
    evidence_class, evidence_note = skip_evidence_class(spec, reason)
    return CommandResult(
        id=spec.id,
        label=spec.label,
        repo=spec.repo,
        command=spec.command,
        cwd=str(spec.cwd),
        status="manual_required",
        blocking=spec.blocking,
        source_tags=spec.source_tags,
        skip_reason=reason,
        evidence_class=evidence_class,
        evidence_note=evidence_note,
    )


def unavailable_result(spec: CommandSpec, reason: str) -> CommandResult:
    return CommandResult(
        id=spec.id,
        label=spec.label,
        repo=spec.repo,
        command=spec.command,
        cwd=str(spec.cwd),
        status="failed",
        blocking=spec.blocking,
        source_tags=spec.source_tags,
        skip_reason=reason,
        evidence_class="required_repo_or_command_unavailable",
        evidence_note=(
            "A required launch-gate command could not start. The gate wrote this "
            "failure artifact instead of raising an unhandled traceback."
        ),
    )


def resolve_repo(root: Path, explicit: str | None, sibling_name: str) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()
    return (root.parent / sibling_name).resolve()


def default_specs(
    *,
    pipeline_repo: Path,
    capture_repo: Path,
    webapp_repo: Path,
    run_ios_tests: bool,
    ios_simulator_destination: str | None = None,
    ios_preflight_failure: str | None = None,
    ios_test_timeout_seconds: int = DEFAULT_IOS_TEST_TIMEOUT_SECONDS,
) -> list[CommandSpec]:
    specs = [
        CommandSpec(
            id="webapp_request_sync_contracts",
            label="WebApp request, publication, inventory, and sync contracts",
            repo="Blueprint-WebApp",
            cwd=webapp_repo,
            command=[
                "npx",
                "vitest",
                "run",
                "server/tests/inbound-request.test.ts",
                "server/tests/admin-capture-job-publication.test.ts",
                "server/tests/marketplace-live-inventory.test.ts",
                "server/tests/pipeline-routes.test.ts",
                "server/tests/creator-mobile-parity.test.ts",
                "server/tests/stripe-native-parity.test.ts",
                "server/tests/stripe-treasury-funding.test.ts",
            ],
            source_tags=("iphone", "glasses", "android"),
        ),
        CommandSpec(
            id="webapp_creator_payout_contracts",
            label="WebApp creator payout-state transition contract",
            repo="Blueprint-WebApp",
            cwd=webapp_repo,
            command=[
                "npx",
                "vitest",
                "run",
                "server/tests/creator-payout-launch-gate.test.ts",
            ],
            source_tags=("iphone", "glasses", "android"),
        ),
        CommandSpec(
            id="webapp_marketplace_fulfillment_contracts",
            label="WebApp marketplace fulfillment checkout contract",
            repo="Blueprint-WebApp",
            cwd=webapp_repo,
            command=[
                "npx",
                "vitest",
                "run",
                "server/tests/marketplace-checkout-fulfillment.test.ts",
            ],
            source_tags=("iphone", "glasses", "android"),
        ),
        CommandSpec(
            id="webapp_buyer_artifact_access_contracts",
            label="WebApp buyer artifact access and signed delivery contracts",
            repo="Blueprint-WebApp",
            cwd=webapp_repo,
            command=[
                "npx",
                "vitest",
                "run",
                "server/tests/marketplace-entitlements.test.ts",
                "server/tests/pipeline-routes.test.ts",
                "server/tests/firebase-storage-config.test.ts",
            ],
            source_tags=("iphone", "glasses", "android"),
        ),
        CommandSpec(
            id="capture_bridge_contracts",
            label="Capture cloud bridge source contracts",
            repo="BlueprintCapture",
            cwd=capture_repo / "cloud" / "extract-frames",
            command=["npm", "test"],
            source_tags=("iphone", "glasses", "android"),
        ),
        CommandSpec(
            id="pipeline_launch_gate",
            label="Pipeline source-specific launch gate and sync artifacts",
            repo="BlueprintCapturePipeline",
            cwd=pipeline_repo,
            command=[
                "pytest",
                "tests/test_webapp_sync.py",
                "tests/test_qualification_alpha.py",
                "tests/test_alpha_readiness.py",
                "tests/test_run_e2e.py",
            ],
            source_tags=("iphone", "glasses", "android"),
        ),
        CommandSpec(
            id="android_bundle_contracts",
            label="Android bundle contract",
            repo="BlueprintCapture",
            cwd=capture_repo / "android",
            command=[
                "./gradlew",
                "app:testDebugUnitTest",
                "--tests",
                "app.blueprint.capture.data.capture.AndroidCaptureBundleBuilderTest",
                "--tests",
                "app.blueprint.capture.data.config.LocalConfigTest",
            ],
            source_tags=("android",),
        ),
    ]

    if run_ios_tests:
        derived_data_path = capture_repo / "build" / "DerivedDataPaidMarketplaceGate"
        specs.append(
            CommandSpec(
                id="ios_launch_contracts",
                label="iPhone and glasses Capture unit contracts",
                repo="BlueprintCapture",
                cwd=capture_repo,
                command=[
                    "xcodebuild",
                    "test",
                    "-project",
                    "BlueprintCapture.xcodeproj",
                    "-scheme",
                    "BlueprintCapture",
                    "-destination",
                    ios_simulator_destination or "platform=iOS Simulator,name=unresolved",
                    "-derivedDataPath",
                    str(derived_data_path),
                    "-only-testing:BlueprintCaptureTests/PipelineContractTests",
                    "-only-testing:BlueprintCaptureTests/ScanHomeAndUploadTests",
                ],
                source_tags=("iphone", "glasses"),
                timeout_seconds=ios_test_timeout_seconds,
                preflight_failure=ios_preflight_failure,
            )
        )

    return specs


def should_skip(spec: CommandSpec) -> str | None:
    if spec.id == "android_bundle_contracts":
        if android_sdk_root_from_env_or_common_paths() is None:
            return ANDROID_SDK_MISSING_REASON
        if not spec.cwd.joinpath("gradlew").is_file():
            return "Android Gradle wrapper is missing."
    if spec.id == "ios_launch_contracts" and shutil.which("xcodebuild") is None:
        return "xcodebuild is not available in this shell."
    return None


def summarize_sources(results: Sequence[CommandResult]) -> list[dict[str, str]]:
    by_id = {result.id: result for result in results}
    webapp_ids = (
        "webapp_request_sync_contracts",
        "webapp_creator_payout_contracts",
        "webapp_marketplace_fulfillment_contracts",
        "webapp_buyer_artifact_access_contracts",
    )
    webapp_ok = all(
        by_id.get(check_id, CommandResult("", "", "", [], "", "failed", True, ())).status == "passed"
        for check_id in webapp_ids
    )
    capture_ok = by_id.get("capture_bridge_contracts", CommandResult("", "", "", [], "", "failed", True, ())).status == "passed"
    pipeline_ok = by_id.get("pipeline_launch_gate", CommandResult("", "", "", [], "", "failed", True, ())).status == "passed"
    android_result = by_id.get(
        "android_bundle_contracts",
        CommandResult("", "", "", [], "", "manual_required", True, ()),
    )
    android_ok = android_result.status
    ios_ok = by_id.get("ios_launch_contracts", CommandResult("", "", "", [], "", "manual_required", True, ())).status

    sources = []
    iphone_status = (
        "external_beta_contract_ready"
        if webapp_ok and capture_ok and pipeline_ok
        else "blocked"
    )
    if ios_ok == "manual_required":
        iphone_status = f"{iphone_status}_manual_device_confirmation_required"
    sources.append(
        {
            "source": "iPhone",
            "status": iphone_status,
            "automated_claim": "External beta contract-ready path only when request, bridge, and pipeline suites all pass.",
        }
    )

    glasses_status = (
        "internal_only_contract_ready"
        if webapp_ok and capture_ok and pipeline_ok
        else "blocked"
    )
    if ios_ok == "manual_required":
        glasses_status = f"{glasses_status}_manual_device_confirmation_required"
    sources.append(
        {
            "source": "glasses",
            "status": glasses_status,
            "automated_claim": "Internal-only contract-ready; external site-faithful claims remain blocked.",
        }
    )

    if android_ok == "passed" and webapp_ok and capture_ok and pipeline_ok:
        android_status = "internal_only_contract_ready"
    elif android_ok == "manual_required" and webapp_ok and capture_ok and pipeline_ok:
        if android_result.evidence_class == "operator_toolchain_required":
            android_status = "internal_only_contract_ready_operator_toolchain_evidence_required"
        else:
            android_status = "internal_only_contract_ready_manual_bundle_confirmation_required"
    else:
        android_status = "blocked"
    android_claim = "Internal-only contract-ready; external site-faithful claims remain blocked."
    if android_result.evidence_class == "operator_toolchain_required":
        android_claim = (
            "Internal-only contract-ready from WebApp, Capture bridge, and Pipeline suites; "
            "Android SDK unit evidence is an operator/toolchain requirement in this shell, "
            "not product readiness or live-device proof."
        )
    sources.append(
        {
            "source": "Android",
            "status": android_status,
            "automated_claim": android_claim,
        }
    )
    return sources


def manual_checks() -> list[dict[str, str]]:
    return [
        {
            "id": "legal_consent_posture_signoff",
            "category": "legal_ehs",
            "status": "manual_signoff_required",
            "required_evidence": "Signature over the current capture consent, rights, redaction, and delivery posture.",
            "not_proven_by_automation": "Repository contracts do not prove legal/EHS approval for the current external-user consent posture.",
        },
        {
            "id": "operator_dpa_data_processing_terms",
            "category": "legal_privacy_ops",
            "status": "manual_signoff_required",
            "required_evidence": "Operator DPA or equivalent data-processing terms covering retention policy, subprocessors, and access-audit terms.",
            "not_proven_by_automation": "Repository contracts do not prove signed operator data-processing terms or privacy-ops approval.",
        },
        {
            "id": "paperclip_ops_relay_secret_rotation",
            "category": "ops_security",
            "status": "manual_security_evidence_required",
            "required_evidence": "Cloud Secret Manager version or equivalent rotation record, plus redeploy evidence for the Paperclip ops relay secret.",
            "not_proven_by_automation": "Repository contracts do not prove the production Paperclip ops relay secret was rotated and redeployed from a secret manager.",
        },
        {
            "id": "iphone_real_device_claim_flow",
            "category": "real_device_capture",
            "status": "manual_live_evidence_required",
            "required_evidence": "Screen recording showing discovery, reservation, upload completion, and the same capture_job_id on iPhone.",
            "not_proven_by_automation": "Automated WebApp, Capture bridge, and Pipeline contracts do not prove a real iPhone completed the paid capture workflow.",
        },
        {
            "id": "glasses_real_device_claim_flow",
            "category": "real_device_capture",
            "status": "manual_live_evidence_required",
            "required_evidence": "Screen recording showing discovery, reservation, upload completion, and the same capture_job_id on glasses.",
            "not_proven_by_automation": "Automated bridge and pipeline fixtures do not prove glasses capture is externally site-faithful or ready for public paid launch.",
        },
        {
            "id": "android_real_device_claim_flow",
            "category": "real_device_capture",
            "status": "manual_live_evidence_required",
            "required_evidence": "Screen recording showing discovery, reservation, upload completion, and the same capture_job_id on Android.",
            "not_proven_by_automation": "The Android SDK/unit-test gap is operator-toolchain evidence; it is separate from real Android device proof.",
        },
        {
            "id": "buyer_payment_settlement",
            "category": "live_payment",
            "status": "manual_live_evidence_required",
            "required_evidence": "Stripe checkout or payment-intent evidence for a live marketplace purchase.",
            "not_proven_by_automation": "Checkout metadata and mocked webhook contract tests do not prove live Stripe money movement.",
        },
        {
            "id": "capturer_payout_settlement",
            "category": "live_payout",
            "status": "manual_live_evidence_required",
            "required_evidence": "Live Stripe connected account state, live payout evidence, webhook reconciliation, and matching creator capture ledger entry for the approved capture.",
            "not_proven_by_automation": "Creator payout-state transitions in tests do not prove live Stripe payout settlement.",
        },
        {
            "id": "stripe_connected_account_live_readiness",
            "category": "live_payout",
            "status": "manual_live_evidence_required",
            "required_evidence": "Backend /v1/stripe/account response showing provider_state_checked=true, provider_mode=live, live_provider_ready=true, payouts_enabled=true, and no blocking requirements.",
            "not_proven_by_automation": "Backend route shape, publishable keys, and mocked Stripe fixtures do not prove live Connect readiness.",
        },
        {
            "id": "payout_exception_monitor_live",
            "category": "ops_monitoring",
            "status": "manual_live_evidence_required",
            "required_evidence": "Live monitor or query evidence for payout.failed, payout.canceled, disbursement_failed, and overdue finance_review records.",
            "not_proven_by_automation": "Repo tests do not prove the live payout exception monitor is configured, running, or watched.",
        },
        {
            "id": "identity_kyc_provider_decision",
            "category": "identity_kyc",
            "status": "manual_decision_required",
            "required_evidence": "Document whether Stripe Connect onboarding alone is the near-term KYC path or whether Persona/Stripe Identity is being added, with required env/account IDs.",
            "not_proven_by_automation": "No automated contract chooses or proves a live identity/KYC provider decision.",
        },
        {
            "id": "background_check_provider_decision",
            "category": "identity_kyc",
            "status": "manual_decision_required",
            "required_evidence": "Document that no Checkr/background-check provider is integrated yet, or provide provider account/env proof before making screening claims.",
            "not_proven_by_automation": "No automated contract proves background-check provider readiness.",
        },
        {
            "id": "human_finance_review_owner",
            "category": "finance_ops",
            "status": "manual_owner_required",
            "required_evidence": "Named human finance owner and review queue/route for payout exceptions before any live payout execution flag is enabled.",
            "not_proven_by_automation": "Automation cannot substitute for a named finance owner and live review route.",
        },
        {
            "id": "buyer_artifact_access",
            "category": "buyer_access",
            "status": "manual_live_evidence_required",
            "required_evidence": "Live authenticated buyer session proving artifact or fulfillment access after purchase.",
            "not_proven_by_automation": "Route/storage-rule contracts do not prove that a live buyer fetched a real purchased artifact after payment.",
        },
    ]


def build_claims(results: Sequence[CommandResult]) -> dict[str, list[str]]:
    blocking_failed = [result for result in results if result.blocking and result.status == "failed"]
    if blocking_failed:
        return {
            "justified": [],
            "not_justified": [
                "Do not claim the paid marketplace beta gate passes while any blocking automated contract suite is failing.",
            ],
        }
    return {
        "justified": [
            "Inbound request intake, marketplace publication, pipeline sync, checkout fulfillment metadata, buyer artifact signed-URL access, and creator payout transitions are covered at contract level.",
            "Qualification and readiness records remain enforced support artifacts, and privacy-safe buyer media plus launchable export packaging are required before buyer-facing readiness is declared.",
            "iPhone is externally marketable only at contract level; glasses and Android remain internal-only for site-faithful launch claims.",
            "Repo-safe payout claim guardrails distinguish mocked contract coverage from live Stripe/provider readiness.",
        ],
        "not_justified": [
            "Do not claim live buyer payments or live capturer payouts are proven until the operator checklist is completed.",
            "Do not claim Stripe, identity/KYC, background-check, instant-pay, or payout-timing readiness from backend URL, publishable key, or mocked tests.",
            "Do not claim real-device production discovery and claim UX is proven until the operator checklist is completed.",
            "Do not market glasses or Android as externally site-faithful world-model paths yet.",
        ],
    }


def evidence_boundary(results: Sequence[CommandResult]) -> dict[str, object]:
    operator_toolchain = [
        {
            "id": result.id,
            "label": result.label,
            "repo": result.repo,
            "reason": result.skip_reason,
            "note": result.evidence_note,
        }
        for result in results
        if result.evidence_class == "operator_toolchain_required"
    ]
    return {
        "automated_proof_scope": (
            "This run proves repository contract behavior only. It does not prove live buyer "
            "payments, capturer payouts, identity/KYC, background checks, instant-pay, "
            "real-device capture flows, or a live post-purchase buyer artifact fetch."
        ),
        "manual_live_evidence_scope": (
            "Manual and live evidence requirements below remain open until operator artifacts "
            "from real devices, Stripe/live provider state, buyer access, and finance ownership "
            "are attached."
        ),
        "operator_toolchain_evidence": operator_toolchain,
    }


def closeout_summary(report: dict) -> dict[str, object]:
    automated_passed = [
        result["label"]
        for result in report["automated_checks"]
        if result["status"] == "passed"
    ]
    automated_failed = [
        {
            "id": result.get("id", result["label"]),
            "label": result["label"],
            "status": result["status"],
            "skip_reason": result.get("skip_reason"),
        }
        for result in report["automated_checks"]
        if result.get("blocking", True) and result["status"] == "failed"
    ]
    manual_required = [
        item["id"]
        for item in report["manual_checks"]
        if item["status"].startswith("manual_")
    ]
    if report.get("overall_status") == "automation_failed":
        readout = (
            "Automated repository contracts did not pass. This is a failing "
            "launch-gate closeout, not manual-ops-ready proof."
        )
    else:
        readout = (
            "Automated repository contracts passed, with any listed toolchain "
            "items marked as operator-required in this shell. This is a "
            "manual-ops closeout, not Operational Launch Ready proof."
        )
    return {
        "operator_readout": readout,
        "automated_contracts_failed": automated_failed,
        "automated_contracts_prove": automated_passed,
        "automated_contracts_do_not_prove": [
            "live Stripe buyer payment completion",
            "live Stripe Connect payout settlement",
            "identity/KYC or background-check provider readiness",
            "real-device discovery, reservation, upload, or capture_job_id continuity",
            "live authenticated buyer artifact fetch after purchase",
            "human finance ownership or live payout exception monitoring",
        ],
        "remaining_manual_evidence_ids": manual_required,
    }


def render_markdown(report: dict) -> str:
    lines = [
        "# Paid Marketplace Beta Launch Gate",
        "",
        f"Generated: {report['generated_at']}",
        "",
        f"Overall status: `{report['overall_status']}`",
        "",
        "## Operator Closeout",
        "",
    ]
    closeout = report.get("closeout_summary") or {}
    if closeout.get("operator_readout"):
        lines.append(str(closeout["operator_readout"]))
        lines.append("")
    if closeout.get("automated_contracts_prove"):
        lines.append("Automated contracts prove:")
        for item in closeout["automated_contracts_prove"]:
            lines.append(f"- {item}")
        lines.append("")
    if closeout.get("automated_contracts_failed"):
        lines.append("Automated contracts failed:")
        for item in closeout["automated_contracts_failed"]:
            label = item.get("label") or item.get("id")
            lines.append(f"- {label}: `{item.get('status')}`")
            if item.get("skip_reason"):
                lines.append(f"  Reason: {item['skip_reason']}")
        lines.append("")
    if closeout.get("automated_contracts_do_not_prove"):
        lines.append("Automated contracts do not prove:")
        for item in closeout["automated_contracts_do_not_prove"]:
            lines.append(f"- {item}")
        lines.append("")
    if closeout.get("remaining_manual_evidence_ids"):
        lines.append("Remaining manual/live evidence ids:")
        for item in closeout["remaining_manual_evidence_ids"]:
            lines.append(f"- `{item}`")
        lines.append("")
    lines.extend([
        "",
        "## Automated Checks",
        "",
    ])
    for result in report["automated_checks"]:
        status = result["status"]
        lines.append(f"- {result['label']}: `{status}`")
        if result.get("skip_reason"):
            lines.append(f"  Reason: {result['skip_reason']}")
        if result.get("evidence_class"):
            lines.append(f"  Evidence class: `{result['evidence_class']}`")
        if result.get("evidence_note"):
            lines.append(f"  Note: {result['evidence_note']}")
    boundary = report.get("evidence_boundary") or {}
    if boundary:
        lines.extend(["", "## Evidence Boundary", ""])
        if boundary.get("automated_proof_scope"):
            lines.append(f"- Automated proof: {boundary['automated_proof_scope']}")
        if boundary.get("manual_live_evidence_scope"):
            lines.append(f"- Manual/live evidence: {boundary['manual_live_evidence_scope']}")
        for item in boundary.get("operator_toolchain_evidence") or []:
            lines.append(f"- Operator toolchain: {item['label']}: {item['note']}")
    lines.extend(["", "## Source Status", ""])
    for source in report["source_status"]:
        lines.append(f"- {source['source']}: `{source['status']}`")
        lines.append(f"  {source['automated_claim']}")
    lines.extend(["", "## Manual Checks", ""])
    for item in report["manual_checks"]:
        lines.append(f"- {item['id']}: `{item['status']}` / `{item['category']}`")
        lines.append(f"  Required evidence: {item['required_evidence']}")
        lines.append(f"  Not proven by automation: {item['not_proven_by_automation']}")
    lines.extend(["", "## Truthful Claims", ""])
    for claim in report["launch_claims"]["justified"]:
        lines.append(f"- Justified: {claim}")
    for claim in report["launch_claims"]["not_justified"]:
        lines.append(f"- Not justified: {claim}")
    return "\n".join(lines) + "\n"


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the paid marketplace beta launch gate")
    parser.add_argument("--capture-repo")
    parser.add_argument("--pipeline-repo")
    parser.add_argument("--webapp-repo")
    parser.add_argument("--json-out")
    parser.add_argument("--markdown-out")
    parser.add_argument("--run-ios-tests", action="store_true")
    parser.add_argument("--ios-simulator-udid", default=os.getenv("BLUEPRINT_IOS_SIMULATOR_UDID"))
    parser.add_argument(
        "--ios-simulator-name",
        default=os.getenv("BLUEPRINT_IOS_SIMULATOR_NAME", DEFAULT_IOS_SIMULATOR_NAME),
    )
    parser.add_argument("--ios-simulator-os", default=os.getenv("BLUEPRINT_IOS_SIMULATOR_OS"))
    parser.add_argument(
        "--ios-test-timeout-seconds",
        type=positive_int,
        default=DEFAULT_IOS_TEST_TIMEOUT_SECONDS,
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    pipeline_repo = resolve_repo(Path(__file__).resolve().parents[1], args.pipeline_repo, "BlueprintCapturePipeline")
    capture_repo = resolve_repo(pipeline_repo, args.capture_repo, "BlueprintCapture")
    webapp_repo = resolve_repo(pipeline_repo, args.webapp_repo, "Blueprint-WebApp")
    load_env_files([pipeline_repo, capture_repo, webapp_repo])

    ios_simulator_destination: str | None = None
    ios_preflight_failure: str | None = None
    if args.run_ios_tests and shutil.which("xcodebuild") is not None:
        try:
            ios_simulator_destination = resolve_ios_simulator_destination(
                preferred_name=args.ios_simulator_name,
                preferred_os=args.ios_simulator_os,
                preferred_udid=args.ios_simulator_udid,
            )
        except (FileNotFoundError, json.JSONDecodeError, RuntimeError, subprocess.CalledProcessError) as exc:
            ios_preflight_failure = f"ios_simulator_destination_unavailable: {exc}"

    specs = default_specs(
        pipeline_repo=pipeline_repo,
        capture_repo=capture_repo,
        webapp_repo=webapp_repo,
        run_ios_tests=bool(args.run_ios_tests),
        ios_simulator_destination=ios_simulator_destination,
        ios_preflight_failure=ios_preflight_failure,
        ios_test_timeout_seconds=args.ios_test_timeout_seconds,
    )

    results: list[CommandResult] = []
    for spec in specs:
        reason = should_skip(spec)
        results.append(skipped_result(spec, reason) if reason else run_command(spec))

    report = {
        "schema_version": "v1",
        "generated_at": utc_now_iso(),
        "pipeline_source": git_source_metadata(
            pipeline_repo,
            repo_name="BlueprintCapturePipeline",
        ),
        "repos": {
          "BlueprintCapture": str(capture_repo),
          "BlueprintCapturePipeline": str(pipeline_repo),
          "Blueprint-WebApp": str(webapp_repo),
        },
        "automated_checks": [asdict(result) for result in results],
        "evidence_boundary": evidence_boundary(results),
        "source_status": summarize_sources(results),
        "manual_checks": manual_checks(),
        "launch_claims": build_claims(results),
    }

    blocking_failed = [
        result for result in results if result.blocking and result.status == "failed"
    ]
    report["overall_status"] = (
        "automation_failed"
        if blocking_failed
        else "automated_contracts_passed_manual_ops_required"
    )
    report["closeout_summary"] = closeout_summary(report)

    json_out = Path(args.json_out).expanduser() if args.json_out else pipeline_repo / "output" / "paid_marketplace_launch_gate.json"
    markdown_out = Path(args.markdown_out).expanduser() if args.markdown_out else pipeline_repo / "output" / "paid_marketplace_launch_gate.md"
    json_out.parent.mkdir(parents=True, exist_ok=True)
    markdown_out.parent.mkdir(parents=True, exist_ok=True)

    json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    markdown_out.write_text(render_markdown(report), encoding="utf-8")

    print(f"[launch-gate] overall_status={report['overall_status']}")
    print(f"[launch-gate] json={json_out}")
    print(f"[launch-gate] markdown={markdown_out}")

    return 1 if blocking_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
