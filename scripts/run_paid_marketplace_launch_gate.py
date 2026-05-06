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


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def tail_text(text: str, limit: int = 80) -> str:
    lines = [line for line in text.strip().splitlines() if line.strip()]
    return "\n".join(lines[-limit:])


def run_command(spec: CommandSpec) -> CommandResult:
    completed = subprocess.run(
        spec.command,
        cwd=spec.cwd,
        capture_output=True,
        text=True,
        env=contract_test_env(),
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


def skipped_result(spec: CommandSpec, reason: str) -> CommandResult:
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
            ],
            source_tags=("android",),
        ),
    ]

    if run_ios_tests:
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
                    "platform=iOS Simulator,name=iPhone 16",
                    "-only-testing:BlueprintCaptureTests/PipelineContractTests",
                    "-only-testing:BlueprintCaptureTests/ScanHomeAndUploadTests",
                ],
                source_tags=("iphone", "glasses"),
            )
        )

    return specs


def should_skip(spec: CommandSpec) -> str | None:
    if spec.id == "android_bundle_contracts":
        if not (os.getenv("ANDROID_HOME") or os.getenv("ANDROID_SDK_ROOT")):
            return "ANDROID_HOME or ANDROID_SDK_ROOT is not configured in this shell."
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
    )
    webapp_ok = all(
        by_id.get(check_id, CommandResult("", "", "", [], "", "failed", True, ())).status == "passed"
        for check_id in webapp_ids
    )
    capture_ok = by_id.get("capture_bridge_contracts", CommandResult("", "", "", [], "", "failed", True, ())).status == "passed"
    pipeline_ok = by_id.get("pipeline_launch_gate", CommandResult("", "", "", [], "", "failed", True, ())).status == "passed"
    android_ok = by_id.get("android_bundle_contracts", CommandResult("", "", "", [], "", "manual_required", True, ())).status
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
        android_status = "internal_only_contract_ready_manual_bundle_confirmation_required"
    else:
        android_status = "blocked"
    sources.append(
        {
            "source": "Android",
            "status": android_status,
            "automated_claim": "Internal-only contract-ready; external site-faithful claims remain blocked.",
        }
    )
    return sources


def manual_checks() -> list[dict[str, str]]:
    return [
        {
            "id": "iphone_real_device_claim_flow",
            "required_evidence": "Screen recording showing discovery, reservation, upload completion, and the same capture_job_id on iPhone.",
        },
        {
            "id": "glasses_real_device_claim_flow",
            "required_evidence": "Screen recording showing discovery, reservation, upload completion, and the same capture_job_id on glasses.",
        },
        {
            "id": "android_real_device_claim_flow",
            "required_evidence": "Screen recording showing discovery, reservation, upload completion, and the same capture_job_id on Android.",
        },
        {
            "id": "buyer_payment_settlement",
            "required_evidence": "Stripe checkout or payment-intent evidence for a live marketplace purchase.",
        },
        {
            "id": "capturer_payout_settlement",
            "required_evidence": "Stripe payout evidence and matching creator capture ledger entry for the approved capture.",
        },
        {
            "id": "buyer_artifact_access",
            "required_evidence": "Authenticated buyer session proving artifact or fulfillment access after purchase.",
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
            "Inbound request intake, marketplace publication, pipeline sync, checkout fulfillment metadata, and creator payout transitions are covered at contract level.",
            "Qualification remains authoritative and privacy-safe buyer media plus launchable export packaging are required before buyer-facing readiness is declared.",
            "iPhone is externally marketable only at contract level; glasses and Android remain internal-only for site-faithful launch claims.",
        ],
        "not_justified": [
            "Do not claim live buyer payments or live capturer payouts are proven until the operator checklist is completed.",
            "Do not claim real-device production discovery and claim UX is proven until the operator checklist is completed.",
            "Do not market glasses or Android as externally site-faithful world-model paths yet.",
        ],
    }


def render_markdown(report: dict) -> str:
    lines = [
        "# Paid Marketplace Beta Launch Gate",
        "",
        f"Generated: {report['generated_at']}",
        "",
        "## Automated Checks",
        "",
    ]
    for result in report["automated_checks"]:
        status = result["status"]
        lines.append(f"- {result['label']}: `{status}`")
        if result.get("skip_reason"):
            lines.append(f"  Reason: {result['skip_reason']}")
    lines.extend(["", "## Source Status", ""])
    for source in report["source_status"]:
        lines.append(f"- {source['source']}: `{source['status']}`")
        lines.append(f"  {source['automated_claim']}")
    lines.extend(["", "## Manual Checks", ""])
    for item in report["manual_checks"]:
        lines.append(f"- {item['id']}: {item['required_evidence']}")
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
    args = parser.parse_args(list(argv) if argv is not None else None)

    pipeline_repo = resolve_repo(Path(__file__).resolve().parents[1], args.pipeline_repo, "BlueprintCapturePipeline")
    capture_repo = resolve_repo(pipeline_repo, args.capture_repo, "BlueprintCapture")
    webapp_repo = resolve_repo(pipeline_repo, args.webapp_repo, "Blueprint-WebApp")
    load_env_files([pipeline_repo, capture_repo, webapp_repo])

    specs = default_specs(
        pipeline_repo=pipeline_repo,
        capture_repo=capture_repo,
        webapp_repo=webapp_repo,
        run_ios_tests=bool(args.run_ios_tests),
    )

    results: list[CommandResult] = []
    for spec in specs:
        reason = should_skip(spec)
        results.append(skipped_result(spec, reason) if reason else run_command(spec))

    report = {
        "schema_version": "v1",
        "generated_at": utc_now_iso(),
        "repos": {
          "BlueprintCapture": str(capture_repo),
          "BlueprintCapturePipeline": str(pipeline_repo),
          "Blueprint-WebApp": str(webapp_repo),
        },
        "automated_checks": [asdict(result) for result in results],
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
