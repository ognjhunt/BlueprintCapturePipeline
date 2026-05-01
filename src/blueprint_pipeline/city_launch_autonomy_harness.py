"""Autonomous city-launch harness state, work packets, and proof synthesis."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CAPTURE_REPO = ROOT.parent / "BlueprintCapture"
DEFAULT_WEBAPP_REPO = ROOT.parent / "Blueprint-WebApp"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def set_nested(payload: dict[str, Any], dotted_path: str, value: Any) -> None:
    current = payload
    parts = dotted_path.split(".")
    for key in parts[:-1]:
        nested = current.get(key)
        if not isinstance(nested, dict):
            nested = {}
            current[key] = nested
        current = nested
    current[parts[-1]] = value


def get_nested(payload: Mapping[str, Any], dotted_path: str) -> Any:
    current: Any = payload
    for key in dotted_path.split("."):
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


@dataclass(frozen=True)
class CommandSpec:
    id: str
    command: list[str]
    cwd: str
    proof_on_pass: tuple[str, ...] = ()
    run_by_default: bool = False


@dataclass(frozen=True)
class WorkPacket:
    lane_id: str
    repo: str
    owner_agent: str
    description: str
    allowed_paths: tuple[str, ...]
    commands: tuple[CommandSpec, ...] = ()
    expected_artifacts: tuple[str, ...] = ()
    proof_fields: tuple[str, ...] = ()
    blocking_conditions: tuple[str, ...] = ()
    requires_live_service: bool = False
    requires_physical_hardware: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "lane_id": self.lane_id,
            "repo": self.repo,
            "owner_agent": self.owner_agent,
            "description": self.description,
            "allowed_paths": list(self.allowed_paths),
            "commands": [
                {
                    "id": command.id,
                    "command": command.command,
                    "cwd": command.cwd,
                    "proof_on_pass": list(command.proof_on_pass),
                    "run_by_default": command.run_by_default,
                }
                for command in self.commands
            ],
            "expected_artifacts": list(self.expected_artifacts),
            "proof_fields": list(self.proof_fields),
            "blocking_conditions": list(self.blocking_conditions),
            "requires_live_service": self.requires_live_service,
            "requires_physical_hardware": self.requires_physical_hardware,
        }


REQUIRED_LAUNCH_PROOF_FIELDS: tuple[tuple[str, Any, str, str, bool], ...] = (
    ("evidence_generated_at", "non_empty", "proof", "evidence timestamp missing", False),
    ("release.config_validated_by_archive_script", True, "ios", "release config not verified", False),
    ("city.backend_supported", True, "city_backend", "city is not proven supported", True),
    ("city.live_approved_job_count", "number>=1", "city_backend", "no live approved jobs", True),
    ("city.live_capture_target_count", "number>=1", "city_backend", "no live capture targets", True),
    ("city.mock_fallback_disabled", True, "city_backend", "mock fallback not proven disabled", False),
    ("city.internal_test_space_disabled", True, "city_backend", "internal test space not proven disabled", False),
    ("capture.real_device_capture_uploaded", True, "ios", "real-device iPhone upload missing", True),
    ("capture.capture_submissions_document_exists", True, "ios", "capture submission document missing", True),
    ("capture.raw_upload_complete_exists", True, "ios", "raw upload completion marker missing", True),
    ("pipeline.capture_descriptor_exists", True, "pipeline", "capture descriptor missing", False),
    ("pipeline.qa_report_exists", True, "pipeline", "pipeline QA report missing", False),
    ("pipeline.pipeline_handoff_exists", True, "pipeline", "pipeline handoff missing", False),
    ("pipeline.pubsub_handoff_succeeded", True, "pipeline", "Pub/Sub handoff missing", True),
    ("pipeline.pipeline_processed_capture", True, "pipeline", "pipeline processed capture missing", True),
    ("privacy_provider.final_walkthrough_uri", "non_empty", "privacy_provider", "privacy-safe final walkthrough missing", False),
    ("privacy_provider.worldlabs_input_uri", "non_empty", "privacy_provider", "World Labs input proof missing", False),
    ("privacy_provider.raw_bypass_disabled", True, "privacy_provider", "raw World Labs bypass not proven disabled", False),
    ("retrieval.dense_index_exists", True, "pipeline", "dense retrieval export missing", False),
    ("retrieval.site_reference_manifest_exists", True, "pipeline", "site reference memory missing", False),
    ("hosted_session.runtime_url", "non_empty", "runtime", "runtime URL missing", True),
    ("hosted_session.webapp_listing_id", "non_empty", "runtime_webapp", "WebApp listing/attachment ID missing", True),
    ("hosted_session.buyer_access_checked", True, "runtime_webapp", "buyer access check missing", True),
    ("meta_glasses.physical_device_smoke_passed", True, "meta_glasses", "physical Meta glasses smoke missing", True),
    ("meta_glasses.video_first_positioning_confirmed", True, "meta_glasses", "glasses video-first claim boundary missing", False),
    ("meta_glasses.native_geometry_not_marketed", True, "meta_glasses", "native-geometry marketing guardrail missing", False),
    ("open_capture.review_gated", True, "marketing", "open capture review gate missing", False),
    ("open_capture.payout_cents", 0, "payments", "open-capture payout must be zero", False),
    ("open_capture.paid_anywhere_claim_disabled", True, "marketing", "paid-anywhere claim not disabled", False),
    ("payouts.backend_configured", True, "payments", "payout backend not configured", True),
    ("payouts.stripe_state_checked", True, "payments", "Stripe state not checked", True),
    ("payouts.marketing_claims_require_stripe_ready", True, "payments", "Stripe claim guardrail missing", False),
    ("ops.launch_owner", "non_empty", "ops", "launch owner missing", False),
    ("ops.failed_upload_monitor", True, "ops", "failed upload monitor missing", True),
    ("ops.submission_registration_monitor", True, "ops", "submission registration monitor missing", True),
    ("ops.push_device_sync_monitor", True, "ops", "push/device sync monitor missing", True),
    ("ops.bridge_pipeline_monitor", True, "ops", "bridge pipeline monitor missing", True),
    ("ops.payout_exception_monitor", True, "ops", "payout exception monitor missing", True),
    ("ops.session_events_queryable", True, "ops", "session events query missing", True),
    ("ops.cloud_logging_handoff_alert", True, "ops", "cloud logging handoff alert missing", True),
)


def default_proof(city_slug: str, budget_cents: int, capture_paths: Sequence[str]) -> dict[str, Any]:
    proof: dict[str, Any] = {
        "schema_version": "city-launch-proof.v1",
        "launch_proof_status": "incomplete",
        "contract_only": False,
        "city_slug": city_slug,
        "budget_cents": budget_cents,
        "capture_paths": list(capture_paths),
        "evidence_generated_at": utc_now_iso(),
        "release": {},
        "city": {"live_approved_job_count": 0, "live_capture_target_count": 0},
        "capture": {},
        "pipeline": {},
        "meta_glasses": {},
        "open_capture": {"payout_cents": 0},
        "payouts": {},
        "ops": {},
        "privacy_provider": {},
        "hosted_session": {},
        "retrieval": {},
        "harness": {"generated_at": utc_now_iso(), "proof_mode": "incomplete_until_live_evidence"},
    }
    for field_name, expected, *_rest in REQUIRED_LAUNCH_PROOF_FIELDS:
        if get_nested(proof, field_name) is None:
            if expected == "non_empty":
                set_nested(proof, field_name, "")
            elif expected == "number>=1":
                set_nested(proof, field_name, 0)
            else:
                set_nested(proof, field_name, False if expected is True else expected)
    proof["evidence_generated_at"] = utc_now_iso()
    proof["city_slug"] = city_slug
    return proof


def build_work_packets(
    *,
    pipeline_repo: Path,
    capture_repo: Path,
    webapp_repo: Path,
    capture_paths: Sequence[str],
) -> list[WorkPacket]:
    packets = [
        WorkPacket(
            lane_id="ios_compile_and_real_device",
            repo="BlueprintCapture",
            owner_agent="mobile-ios-agent",
            description="Fix and verify iOS launch build, then collect real-device iPhone proof.",
            allowed_paths=(
                str(capture_repo / "BlueprintCapture"),
                str(capture_repo / "BlueprintCaptureTests"),
                str(capture_repo / "scripts"),
                str(capture_repo / "ops/launch-readiness"),
            ),
            commands=(
                CommandSpec(
                    id="ios_targeted_launch_tests",
                    command=[
                        "xcodebuild",
                        "test",
                        "-project",
                        "BlueprintCapture.xcodeproj",
                        "-scheme",
                        "BlueprintCapture",
                        "-destination",
                        "platform=iOS Simulator,name=iPhone 17 Pro",
                        "-parallel-testing-enabled",
                        "NO",
                        "-only-testing:BlueprintCaptureTests/CaptureBundleAndInferenceTests/finalizerAndExportProducePipelineReadyBundle",
                        "-only-testing:BlueprintCaptureTests/LaunchCityGateTests",
                    ],
                    cwd=str(capture_repo),
                    proof_on_pass=("harness.ios_targeted_launch_tests_passed",),
                ),
            ),
            expected_artifacts=("ops/launch-readiness/<city-slug>.launch-proof.json",),
            proof_fields=(
                "release.config_validated_by_archive_script",
                "capture.real_device_capture_uploaded",
                "capture.capture_submissions_document_exists",
                "capture.raw_upload_complete_exists",
            ),
            blocking_conditions=("iOS tests fail", "real iPhone upload is unavailable"),
            requires_physical_hardware=True,
        ),
        WorkPacket(
            lane_id="site_identity_dense_export",
            repo="BlueprintCapture + BlueprintCapturePipeline",
            owner_agent="capture-contract-agent",
            description="Guarantee site identity, topology, revisit anchors, and dense export for marketed capture paths.",
            allowed_paths=(
                str(capture_repo / "BlueprintCapture"),
                str(capture_repo / "cloud/extract-frames/src"),
                str(pipeline_repo / "src/blueprint_pipeline/retrieval_index_stage.py"),
                str(pipeline_repo / "tests"),
            ),
            commands=(
                CommandSpec(
                    id="pipeline_world_model_contracts",
                    command=[
                        "pytest",
                        "tests/test_world_model_candidate_parity.py",
                        "tests/test_retrieval_index_geometry_source.py",
                    ],
                    cwd=str(pipeline_repo),
                    proof_on_pass=("harness.site_identity_dense_export_tests_passed",),
                    run_by_default=True,
                ),
            ),
            proof_fields=("pipeline.capture_descriptor_exists", "pipeline.qa_report_exists"),
            blocking_conditions=("site_id missing", "world_model_candidate false", "dense export missing"),
        ),
        WorkPacket(
            lane_id="pipeline_readiness_contracts",
            repo="BlueprintCapturePipeline",
            owner_agent="pipeline-readiness-agent",
            description="Run source-specific launch gate contracts and keep non-iPhone paths internal unless live proof exists.",
            allowed_paths=(str(pipeline_repo / "src/blueprint_pipeline"), str(pipeline_repo / "tests")),
            commands=(
                CommandSpec(
                    id="pipeline_launch_gate_tests",
                    command=[
                        "pytest",
                        "tests/test_alpha_readiness.py",
                        "tests/test_run_e2e.py",
                        "tests/test_webapp_sync.py",
                        "tests/test_site_world_packaging.py",
                    ],
                    cwd=str(pipeline_repo),
                    proof_on_pass=("harness.pipeline_launch_gate_tests_passed",),
                    run_by_default=True,
                ),
            ),
            proof_fields=(
                "pipeline.capture_descriptor_exists",
                "pipeline.pipeline_handoff_exists",
                "pipeline.pipeline_processed_capture",
            ),
            blocking_conditions=("Pipeline tests fail", "readiness overclaims external market readiness"),
        ),
        WorkPacket(
            lane_id="city_backend_routes",
            repo="Blueprint-WebApp + BlueprintCapture",
            owner_agent="city-backend-agent",
            description="Prove city support, approved jobs, capture targets, and launch route truth.",
            allowed_paths=(str(webapp_repo / "server"), str(capture_repo / "scripts/validate_launch_readiness.py")),
            proof_fields=(
                "city.backend_supported",
                "city.live_approved_job_count",
                "city.live_capture_target_count",
                "city.mock_fallback_disabled",
                "city.internal_test_space_disabled",
            ),
            blocking_conditions=("auth token missing", "city is not live-supported", "feed returns no capture jobs"),
            requires_live_service=True,
        ),
        WorkPacket(
            lane_id="privacy_safe_provider",
            repo="BlueprintCapturePipeline",
            owner_agent="privacy-provider-agent",
            description="Prove only privacy-safe video reaches World Labs or video-to-world providers.",
            allowed_paths=(
                str(pipeline_repo / "src/blueprint_pipeline/privacy_processing.py"),
                str(pipeline_repo / "src/blueprint_pipeline/provider_preview.py"),
                str(pipeline_repo / "tests"),
            ),
            proof_fields=(
                "privacy_provider.final_walkthrough_uri",
                "privacy_provider.worldlabs_input_uri",
                "privacy_provider.raw_bypass_disabled",
            ),
            blocking_conditions=("privacy runner unavailable", "raw provider bypass enabled", "World Labs input is not privacy-safe"),
            requires_live_service=True,
        ),
        WorkPacket(
            lane_id="runtime_webapp_buyer_access",
            repo="BlueprintCapturePipeline + Blueprint-WebApp",
            owner_agent="runtime-webapp-agent",
            description="Prove hosted runtime, WebApp sync, and authenticated buyer artifact access.",
            allowed_paths=(str(pipeline_repo / "src/blueprint_pipeline"), str(webapp_repo / "server")),
            proof_fields=("hosted_session.runtime_url", "hosted_session.webapp_listing_id", "hosted_session.buyer_access_checked"),
            blocking_conditions=("runtime URL missing", "WebApp sync failed", "buyer route inaccessible"),
            requires_live_service=True,
        ),
        WorkPacket(
            lane_id="payments_payouts_marketing",
            repo="Blueprint-WebApp + BlueprintCapture",
            owner_agent="marketplace-payments-agent",
            description="Prove Stripe state and keep capturer marketing claims inside proven payout truth.",
            allowed_paths=(str(webapp_repo / "server"), str(capture_repo / "BlueprintCapture")),
            proof_fields=(
                "payouts.backend_configured",
                "payouts.stripe_state_checked",
                "payouts.marketing_claims_require_stripe_ready",
                "open_capture.review_gated",
                "open_capture.paid_anywhere_claim_disabled",
            ),
            blocking_conditions=("Stripe route unavailable", "payout state unverified", "marketing copy overclaims payout readiness"),
            requires_live_service=True,
        ),
        WorkPacket(
            lane_id="ops_monitors_recovery",
            repo="BlueprintCapture + BlueprintCapturePipeline + Blueprint-WebApp",
            owner_agent="ops-reliability-agent",
            description="Prove launch monitors and recovery paths for uploads, handoffs, payouts, and alerts.",
            allowed_paths=(str(capture_repo), str(pipeline_repo), str(webapp_repo)),
            proof_fields=(
                "ops.failed_upload_monitor",
                "ops.submission_registration_monitor",
                "ops.push_device_sync_monitor",
                "ops.bridge_pipeline_monitor",
                "ops.payout_exception_monitor",
                "ops.session_events_queryable",
                "ops.cloud_logging_handoff_alert",
            ),
            blocking_conditions=("monitor query missing", "alert route missing", "retry path unverified"),
            requires_live_service=True,
        ),
    ]
    if "meta_glasses" in set(capture_paths):
        packets.append(
            WorkPacket(
                lane_id="meta_glasses_physical_pilot",
                repo="BlueprintCapture",
                owner_agent="wearables-capture-agent",
                description="Prove physical Meta glasses connection, upload, video-first guardrail, and internal-only pilot status.",
                allowed_paths=(str(capture_repo / "BlueprintCapture"), str(capture_repo / "BlueprintCaptureTests")),
                proof_fields=(
                    "meta_glasses.physical_device_smoke_passed",
                    "meta_glasses.video_first_positioning_confirmed",
                    "meta_glasses.native_geometry_not_marketed",
                ),
                blocking_conditions=("physical Meta glasses unavailable", "upload smoke missing", "native geometry marketed externally"),
                requires_physical_hardware=True,
            )
        )
    return packets


def proof_field_satisfied(proof: Mapping[str, Any], field_name: str, expected: Any) -> bool:
    value = get_nested(proof, field_name)
    if expected == "non_empty":
        return isinstance(value, str) and bool(value.strip())
    if expected == "number>=1":
        return isinstance(value, (int, float)) and value >= 1
    return value == expected


def build_blockers(proof: Mapping[str, Any]) -> list[dict[str, Any]]:
    blockers: list[dict[str, Any]] = []
    for field_name, expected, lane_id, message, human_required in REQUIRED_LAUNCH_PROOF_FIELDS:
        if proof_field_satisfied(proof, field_name, expected):
            continue
        blockers.append(
            {
                "id": f"missing_{field_name.replace('.', '_')}",
                "lane_id": lane_id,
                "severity": "blocker",
                "message": message,
                "proof_field": field_name,
                "expected": expected,
                "actual": get_nested(proof, field_name),
                "human_required": bool(human_required),
                "created_at": utc_now_iso(),
            }
        )
    return blockers


def determine_status(proof: Mapping[str, Any], blockers: Sequence[Mapping[str, Any]]) -> str:
    if not blockers:
        return "ready_to_market_iphone_city_beta"
    repo_lanes = {"ios", "pipeline", "proof"}
    if any(str(blocker.get("lane_id")) in repo_lanes and not blocker.get("human_required") for blocker in blockers):
        return "blocked_repo_or_contract_failure"
    meta_ready = (
        get_nested(proof, "meta_glasses.physical_device_smoke_passed") is True
        and get_nested(proof, "meta_glasses.video_first_positioning_confirmed") is True
        and get_nested(proof, "meta_glasses.native_geometry_not_marketed") is True
    )
    if meta_ready:
        return "ready_for_internal_glasses_pilot"
    return "blocked_external_dependency"


def apply_lane_results(proof: dict[str, Any], results_dir: Path) -> list[dict[str, Any]]:
    applied: list[dict[str, Any]] = []
    if not results_dir.is_dir():
        return applied
    for path in sorted(results_dir.glob("*.json")):
        result = read_json(path)
        evidence = result.get("evidence") if isinstance(result.get("evidence"), Mapping) else {}
        for field_name, value in evidence.items():
            set_nested(proof, str(field_name), value)
        applied.append({"path": str(path), "lane_id": result.get("lane_id"), "status": result.get("status")})
    return applied


def command_allowed(command: CommandSpec, *, args: argparse.Namespace) -> bool:
    if command.id == "ios_targeted_launch_tests":
        return bool(args.include_ios_tests)
    if command.id == "pipeline_launch_gate_tests":
        return bool(args.include_pipeline_tests or command.run_by_default)
    if command.id == "pipeline_world_model_contracts":
        return bool(args.include_pipeline_tests or command.run_by_default)
    return bool(command.run_by_default)


def run_command(command: CommandSpec) -> dict[str, Any]:
    completed = subprocess.run(
        command.command,
        cwd=command.cwd,
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )
    return {
        "id": command.id,
        "status": "passed" if completed.returncode == 0 else "failed",
        "exit_code": completed.returncode,
        "command": command.command,
        "cwd": command.cwd,
        "stdout_tail": "\n".join(completed.stdout.strip().splitlines()[-80:]),
        "stderr_tail": "\n".join(completed.stderr.strip().splitlines()[-80:]),
        "proof_on_pass": list(command.proof_on_pass),
    }


def _optional_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def _write_lane_result(path: Path, *, lane_id: str, status: str, evidence: Mapping[str, Any], blockers: Sequence[str]) -> dict[str, Any]:
    result = {
        "schema_version": "lane-result.v1",
        "lane_id": lane_id,
        "status": status,
        "generated_at": utc_now_iso(),
        "evidence": dict(evidence),
        "blockers": list(blockers),
    }
    write_json(path, result)
    return result


def collect_capture_root_evidence(*, capture_root: Path, lane_results_root: Path) -> list[dict[str, Any]]:
    pipeline_root = capture_root / "pipeline"
    descriptor = _optional_json(capture_root / "capture_descriptor.json")
    metadata = descriptor.get("metadata") if isinstance(descriptor.get("metadata"), Mapping) else {}
    privacy_manifest = _optional_json(pipeline_root / "privacy_processing_manifest.json")
    worldlabs_audit = _optional_json(pipeline_root / "worldlabs_input_audit.json")
    provider_run = _optional_json(pipeline_root / "provider_run_manifest.json")
    registration = _optional_json(pipeline_root / "evaluation_prep" / "site_world_registration.json")
    health = _optional_json(pipeline_root / "evaluation_prep" / "site_world_health.json")
    webapp_sync = _optional_json(pipeline_root / "webapp_sync_result.json")

    final_walkthrough = capture_root / "privacy" / "final_walkthrough.mov"
    if not final_walkthrough.is_file():
        final_walkthrough = capture_root / "privacy" / "final_walkthrough.mp4"
    dense_index = pipeline_root / "world_model_export" / "dense_index.jsonl"
    site_reference = next(capture_root.glob("sites/*/reference_memory/site_reference_manifest.json"), None)

    results: list[dict[str, Any]] = []
    pipeline_evidence = {
        "pipeline.capture_descriptor_exists": (capture_root / "capture_descriptor.json").is_file(),
        "pipeline.qa_report_exists": (capture_root / "qa_report.json").is_file(),
        "pipeline.pipeline_handoff_exists": (pipeline_root / "opportunity_handoff.json").is_file(),
        "pipeline.pipeline_processed_capture": (pipeline_root / ".qualification_pipeline_complete").is_file(),
    }
    results.append(
        _write_lane_result(
            lane_results_root / "pipeline.capture-root-evidence.json",
            lane_id="pipeline",
            status="succeeded" if all(pipeline_evidence.values()) else "blocked",
            evidence=pipeline_evidence,
            blockers=[key for key, value in pipeline_evidence.items() if not value],
        )
    )

    worldlabs_input_uri = str(
        worldlabs_audit.get("output_video_uri")
        or metadata.get("worldlabs_input_video_uri")
        or provider_run.get("selected_video_uri")
        or ""
    ).strip()
    final_walkthrough_uri = str(privacy_manifest.get("final_walkthrough_uri") or "").strip()
    if not final_walkthrough_uri and final_walkthrough.is_file():
        final_walkthrough_uri = str(final_walkthrough)
    privacy_evidence = {
        "privacy_provider.final_walkthrough_uri": final_walkthrough_uri,
        "privacy_provider.worldlabs_input_uri": worldlabs_input_uri,
        "privacy_provider.raw_bypass_disabled": (
            bool(worldlabs_audit.get("privacy_safe_input"))
            and not bool(worldlabs_audit.get("raw_video_bypass_used"))
        ),
    }
    results.append(
        _write_lane_result(
            lane_results_root / "privacy_safe_provider.capture-root-evidence.json",
            lane_id="privacy_safe_provider",
            status="succeeded" if all(bool(value) for value in privacy_evidence.values()) else "blocked",
            evidence=privacy_evidence,
            blockers=[key for key, value in privacy_evidence.items() if not value],
        )
    )

    retrieval_evidence = {
        "retrieval.dense_index_exists": dense_index.is_file(),
        "retrieval.site_reference_manifest_exists": bool(site_reference and site_reference.is_file()),
    }
    results.append(
        _write_lane_result(
            lane_results_root / "site_identity_dense_export.capture-root-evidence.json",
            lane_id="site_identity_dense_export",
            status="succeeded" if all(retrieval_evidence.values()) else "blocked",
            evidence=retrieval_evidence,
            blockers=[key for key, value in retrieval_evidence.items() if not value],
        )
    )

    response_ids = webapp_sync.get("webapp_response_ids") if isinstance(webapp_sync.get("webapp_response_ids"), Mapping) else {}
    buyer_access = webapp_sync.get("buyer_access_check") if isinstance(webapp_sync.get("buyer_access_check"), Mapping) else {}
    runtime_url = str(registration.get("runtime_base_url") or health.get("runtime_base_url") or "").strip()
    webapp_listing_id = str(response_ids.get("listing_id") or response_ids.get("attachment_id") or response_ids.get("pipeline_attachment_id") or "").strip()
    hosted_evidence = {
        "hosted_session.runtime_url": runtime_url,
        "hosted_session.webapp_listing_id": webapp_listing_id,
        "hosted_session.buyer_access_checked": bool(buyer_access.get("buyer_access_checked")),
    }
    results.append(
        _write_lane_result(
            lane_results_root / "runtime_webapp_buyer_access.capture-root-evidence.json",
            lane_id="runtime_webapp_buyer_access",
            status="succeeded" if all(bool(value) for value in hosted_evidence.values()) else "blocked",
            evidence=hosted_evidence,
            blockers=[key for key, value in hosted_evidence.items() if not value],
        )
    )
    return results


def execute_local_packets(
    *,
    packets: Sequence[WorkPacket],
    proof: dict[str, Any],
    run_root: Path,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    command_results: list[dict[str, Any]] = []
    if not args.execute_local:
        results_dir = run_root / "lane-results"
        for packet in packets:
            _write_lane_result(
                results_dir / f"{packet.lane_id}.not-executed.json",
                lane_id=packet.lane_id,
                status="blocked",
                evidence={},
                blockers=["local_execution_disabled"],
            )
        return command_results
    results_dir = run_root / "command-results"
    results_dir.mkdir(parents=True, exist_ok=True)
    lane_results_dir = run_root / "lane-results"
    for packet in packets:
        packet_blockers: list[str] = []
        packet_evidence: dict[str, Any] = {
            "harness.local_execution.command_count": len(packet.commands),
            "harness.local_execution.executed_command_count": 0,
            "harness.local_execution.passed_command_count": 0,
        }
        for command in packet.commands:
            if not command_allowed(command, args=args):
                continue
            result = run_command(command)
            result["lane_id"] = packet.lane_id
            command_results.append(result)
            write_json(results_dir / f"{packet.lane_id}.{command.id}.json", result)
            packet_evidence["harness.local_execution.executed_command_count"] = (
                int(packet_evidence["harness.local_execution.executed_command_count"]) + 1
            )
            if result["status"] == "passed":
                packet_evidence["harness.local_execution.passed_command_count"] = (
                    int(packet_evidence["harness.local_execution.passed_command_count"]) + 1
                )
                for proof_field in command.proof_on_pass:
                    set_nested(proof, proof_field, True)
                    packet_evidence[proof_field] = True
            else:
                packet_blockers.append(f"{command.id}:exit_{result['exit_code']}")
        if int(packet_evidence["harness.local_execution.executed_command_count"]) == 0:
            packet_blockers.append("no_commands_executed")
        _write_lane_result(
            lane_results_dir / f"{packet.lane_id}.local-execution.json",
            lane_id=packet.lane_id,
            status="passed" if not packet_blockers else "blocked",
            evidence=packet_evidence,
            blockers=packet_blockers,
        )
    return command_results


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(dict(row), sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def run_harness(args: argparse.Namespace) -> dict[str, Any]:
    pipeline_repo = Path(args.pipeline_repo).expanduser().resolve()
    capture_repo = Path(args.capture_repo).expanduser().resolve()
    webapp_repo = Path(args.webapp_repo).expanduser().resolve()
    capture_paths = tuple(args.capture_path or ("iphone",))
    run_id = args.run_id or f"{utc_now_iso().replace(':', '').replace('+', 'Z')}-{uuid.uuid4().hex[:8]}"
    output_root = Path(args.output_root).expanduser().resolve()
    run_root = output_root / args.city_slug / run_id
    work_packet_root = run_root / "work-packets"
    lane_results_root = run_root / "lane-results"

    proof_path = run_root / "proof.launch-proof.json"
    if args.resume and proof_path.is_file():
        proof = read_json(proof_path)
    else:
        proof = default_proof(args.city_slug, args.budget_cents, capture_paths)

    proof["city_slug"] = args.city_slug
    proof["budget_cents"] = args.budget_cents
    proof["capture_paths"] = list(capture_paths)
    proof["harness"]["run_id"] = run_id
    proof["harness"]["run_root"] = str(run_root)

    packets = build_work_packets(
        pipeline_repo=pipeline_repo,
        capture_repo=capture_repo,
        webapp_repo=webapp_repo,
        capture_paths=capture_paths,
    )
    run_root.mkdir(parents=True, exist_ok=True)
    for packet in packets:
        write_json(work_packet_root / f"{packet.lane_id}.json", packet.to_dict())

    applied_lane_results = apply_lane_results(proof, lane_results_root)
    command_results = execute_local_packets(packets=packets, proof=proof, run_root=run_root, args=args)
    applied_lane_results.extend(apply_lane_results(proof, lane_results_root))
    capture_root_arg = str(getattr(args, "capture_root", "") or "").strip()
    capture_root_evidence: list[dict[str, Any]] = []
    if capture_root_arg:
        capture_root_evidence = collect_capture_root_evidence(
            capture_root=Path(capture_root_arg).expanduser().resolve(),
            lane_results_root=lane_results_root,
        )
        applied_lane_results.extend(apply_lane_results(proof, lane_results_root))
    elif not args.execute_local:
        applied_lane_results.extend(apply_lane_results(proof, lane_results_root))
    blockers = build_blockers(proof)
    status = determine_status(proof, blockers)
    proof["launch_proof_status"] = status
    proof["harness"]["applied_lane_results"] = applied_lane_results
    proof["harness"]["command_result_count"] = len(command_results)
    proof["harness"]["capture_root_evidence_result_count"] = len(capture_root_evidence)
    proof["harness"]["work_packet_count"] = len(packets)
    proof["harness"]["updated_at"] = utc_now_iso()

    manifest = {
        "schema_version": "city-launch-harness-run.v1",
        "run_id": run_id,
        "city_slug": args.city_slug,
        "budget_cents": args.budget_cents,
        "capture_paths": list(capture_paths),
        "status": status,
        "created_or_updated_at": utc_now_iso(),
        "repos": {
            "pipeline": str(pipeline_repo),
            "capture": str(capture_repo),
            "webapp": str(webapp_repo),
        },
        "artifacts": {
            "proof": str(proof_path),
            "blockers": str(run_root / "blockers.jsonl"),
            "work_packets": str(work_packet_root),
            "lane_results": str(lane_results_root),
        },
        "first_blocker": blockers[0] if blockers else None,
    }
    summary = {
        "run_id": run_id,
        "status": status,
        "first_blocker": blockers[0] if blockers else None,
        "blocker_count": len(blockers),
        "work_packet_count": len(packets),
        "proof_path": str(proof_path),
    }

    write_json(proof_path, proof)
    write_json(run_root / "manifest.json", manifest)
    write_json(run_root / "summary.json", summary)
    write_jsonl(run_root / "blockers.jsonl", blockers)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Blueprint autonomous city-launch harness.")
    parser.add_argument("--city-slug", required=True)
    parser.add_argument("--budget-cents", type=int, required=True)
    parser.add_argument("--capture-path", action="append", choices=("iphone", "meta_glasses", "android"))
    parser.add_argument("--run-id")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--execute-local", action="store_true")
    parser.add_argument("--include-ios-tests", action="store_true")
    parser.add_argument("--include-pipeline-tests", action="store_true")
    parser.add_argument("--pipeline-repo", default=str(ROOT))
    parser.add_argument("--capture-repo", default=str(DEFAULT_CAPTURE_REPO))
    parser.add_argument("--webapp-repo", default=str(DEFAULT_WEBAPP_REPO))
    parser.add_argument("--output-root", default=str(ROOT / "ops/city-launch-runs"))
    parser.add_argument("--capture-root", help="Optional real capture root to scan for Pipeline lane evidence.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_harness(args)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if not str(summary["status"]).startswith("blocked_repo") else 2


if __name__ == "__main__":
    raise SystemExit(main())
