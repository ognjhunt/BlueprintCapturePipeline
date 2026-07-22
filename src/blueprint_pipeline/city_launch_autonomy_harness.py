"""Autonomous city-launch harness state, work packets, and proof synthesis."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from blueprint_pipeline.city_launch_evidence_policy import (
    RUN_SCHEMA_VERSION,
    build_artifact_inventory,
    default_output_root,
    evidence_policy,
    prepare_evidence_root,
    prepare_run_root,
)
from blueprint_pipeline.common import write_json as atomic_write_json
from blueprint_pipeline.common import write_text as atomic_write_text
from blueprint_pipeline.local_capture import resolve_local_capture_context
from blueprint_pipeline.core.safe_env import contract_test_env, load_env_files

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CAPTURE_REPO = ROOT.parent / "BlueprintCapture"
DEFAULT_WEBAPP_REPO = ROOT.parent / "Blueprint-WebApp"
WEBAPP_STATUS_TIMEOUT_SECONDS = 30


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    atomic_write_json(path, dict(payload))


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
    ("payouts.provider_name", "non_empty", "payments", "payout provider not named", False),
    ("payouts.provider_state_checked", True, "payments", "payout provider state not checked", True),
    ("payouts.live_provider_ready", True, "payments", "live payout provider readiness not proven", True),
    ("payouts.contract_readiness_not_live_readiness", True, "payments", "mock contract readiness not separated from live provider readiness", False),
    ("payouts.live_payout_execution_human_gate", True, "payments", "live payout execution human gate missing", False),
    ("payouts.identity_kyc_state_documented", True, "payments", "identity/KYC state not documented", False),
    ("payouts.background_check_state_documented", True, "payments", "background-check state not documented", False),
    ("payouts.marketing_claims_require_stripe_ready", True, "payments", "Stripe claim guardrail missing", False),
    ("ops.launch_owner", "non_empty", "ops", "launch owner missing", False),
    ("ops.failed_upload_monitor", True, "ops", "failed upload monitor missing", True),
    ("ops.submission_registration_monitor", True, "ops", "submission registration monitor missing", True),
    ("ops.push_device_sync_monitor", True, "ops", "push/device sync monitor missing", True),
    ("ops.bridge_pipeline_monitor", True, "ops", "bridge pipeline monitor missing", True),
    ("ops.payout_exception_monitor_repo_contract", True, "ops", "repo payout exception monitor contract missing", False),
    ("ops.human_finance_review_gate", True, "ops", "human finance review gate missing", False),
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
        "open_capture": {
            "payout_cents": 0,
            "review_gated": True,
            "paid_anywhere_claim_disabled": True,
        },
        "payouts": {
            "provider_name": "stripe",
            "contract_readiness_not_live_readiness": True,
            "live_payout_execution_human_gate": True,
            "identity_kyc_state_documented": True,
            "background_check_state_documented": True,
            "marketing_claims_require_stripe_ready": True,
        },
        "ops": {
            "payout_exception_monitor_repo_contract": True,
            "human_finance_review_gate": True,
        },
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
                    id="release_config_validation",
                    command=[
                        "./scripts/archive_external_alpha.sh",
                        "--validate-config-only",
                    ],
                    cwd=str(capture_repo),
                    proof_on_pass=("release.config_validated_by_archive_script",),
                    run_by_default=True,
                ),
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
                "payouts.provider_name",
                "payouts.provider_state_checked",
                "payouts.live_provider_ready",
                "payouts.contract_readiness_not_live_readiness",
                "payouts.live_payout_execution_human_gate",
                "payouts.identity_kyc_state_documented",
                "payouts.background_check_state_documented",
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
                "ops.payout_exception_monitor_repo_contract",
                "ops.human_finance_review_gate",
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
        diagnostic = ""
        if field_name == "release.config_validated_by_archive_script":
            validation = get_nested(proof, "harness.release_config_validation")
            if isinstance(validation, Mapping):
                diagnostic = str(validation.get("stderr_tail") or validation.get("stdout_tail") or "").strip()
        blocker_message = f"{message}: {diagnostic}" if diagnostic else message
        blockers.append(
            {
                "id": f"missing_{field_name.replace('.', '_')}",
                "lane_id": lane_id,
                "severity": "blocker",
                "message": blocker_message,
                "proof_field": field_name,
                "expected": expected,
                "actual": get_nested(proof, field_name),
                "human_required": bool(human_required),
                "created_at": utc_now_iso(),
                **({"diagnostic": diagnostic} if diagnostic else {}),
            }
        )
    return blockers


def _capture_root_retrieval_input_missing(lane_results: Sequence[Mapping[str, Any]]) -> bool:
    input_shape_blockers = {
        "capture_descriptor_missing_site_id_or_metadata_site_identity_site_id",
        "privacy_safe_walkthrough_missing_for_retrieval_index",
    }
    for result in lane_results:
        if result.get("lane_id") != "site_identity_dense_export":
            continue
        blockers = result.get("blockers") if isinstance(result.get("blockers"), list) else ()
        if any(str(blocker) in input_shape_blockers for blocker in blockers):
            return True
    return False


def _capture_paths_from_proof(proof: Mapping[str, Any]) -> set[str]:
    capture_paths = proof.get("capture_paths")
    if not isinstance(capture_paths, list):
        return set()
    return {str(capture_path).strip() for capture_path in capture_paths if str(capture_path).strip()}


def determine_status(
    proof: Mapping[str, Any],
    blockers: Sequence[Mapping[str, Any]],
    *,
    capture_root_evidence: Sequence[Mapping[str, Any]] = (),
) -> str:
    if not blockers:
        return "ready_to_market_iphone_city_beta"
    capture_paths = _capture_paths_from_proof(proof)
    includes_iphone = "iphone" in capture_paths
    repo_lanes = {"ios", "pipeline", "proof"}
    capture_root_input_missing = _capture_root_retrieval_input_missing(capture_root_evidence)
    for blocker in blockers:
        lane_id = str(blocker.get("lane_id"))
        if lane_id not in repo_lanes or blocker.get("human_required"):
            continue
        field_name = str(blocker.get("proof_field") or "")
        if not includes_iphone and field_name.startswith(("release.", "capture.")):
            continue
        if capture_root_input_missing and field_name.startswith("retrieval."):
            continue
        return "blocked_repo_or_contract_failure"
    meta_ready = (
        "meta_glasses" in capture_paths
        and not includes_iphone
        and get_nested(proof, "meta_glasses.physical_device_smoke_passed") is True
        and get_nested(proof, "meta_glasses.video_first_positioning_confirmed") is True
        and get_nested(proof, "meta_glasses.native_geometry_not_marketed") is True
    )
    if meta_ready:
        required_downstream_prefixes = (
            "pipeline.",
            "privacy_provider.",
            "retrieval.",
            "hosted_session.",
            "meta_glasses.",
        )
        downstream_blockers = [
            blocker
            for blocker in blockers
            if str(blocker.get("proof_field") or "").startswith(required_downstream_prefixes)
        ]
        if not downstream_blockers:
            return "ready_for_internal_glasses_pilot"
    return "blocked_external_dependency"


def _work_packet_for_proof_field(field_name: str) -> str:
    if field_name.startswith(("release.", "capture.")):
        return "ios_compile_and_real_device"
    if field_name.startswith("city."):
        return "city_backend_routes"
    if field_name.startswith("privacy_provider."):
        return "privacy_safe_provider"
    if field_name.startswith("retrieval."):
        return "site_identity_dense_export"
    if field_name.startswith("hosted_session."):
        return "runtime_webapp_buyer_access"
    if field_name.startswith(("payouts.", "open_capture.")):
        return "payments_payouts_marketing"
    if field_name.startswith("ops."):
        return "ops_monitors_recovery"
    if field_name.startswith("meta_glasses."):
        return "meta_glasses_physical_pilot"
    if field_name.startswith("pipeline."):
        return "pipeline_readiness_contracts"
    return "proof"


def _dependency_class(blocker: Mapping[str, Any]) -> str:
    field_name = str(blocker.get("proof_field") or "")
    lane_id = str(blocker.get("lane_id") or "")
    if field_name.startswith("city."):
        return "live_city_backend"
    if field_name.startswith("capture."):
        return "real_device_iphone_capture"
    if field_name == "pipeline.pubsub_handoff_succeeded":
        return "live_pipeline_handoff"
    if field_name.startswith("privacy_provider."):
        return "privacy_safe_provider_artifact"
    if field_name.startswith("retrieval."):
        return "capture_root_or_retrieval_export"
    if field_name.startswith("hosted_session."):
        return "hosted_buyer_access"
    if field_name.startswith("payouts."):
        return "payment_or_payout_ops"
    if field_name.startswith("ops."):
        return "launch_ops_monitoring"
    if field_name.startswith("meta_glasses."):
        return "internal_hardware_evidence"
    if blocker.get("human_required") is True:
        return "human_required_external_evidence"
    if lane_id in {"ios", "pipeline", "proof"}:
        return "repo_or_contract_gap"
    return "launch_policy_gap"


def _expected_evidence_for_blocker(blocker: Mapping[str, Any]) -> str:
    field_name = str(blocker.get("proof_field") or "")
    if field_name == "city.backend_supported":
        return "WebApp public launch status returns ok=true, currentCity.citySlug matches, isSupported=true, status=live, and sourceStatus is available."
    if field_name in {"city.live_approved_job_count", "city.live_capture_target_count"}:
        return "Live city activation ledgers contain at least one approved job and one capture target for the requested city."
    if field_name.startswith("capture."):
        return "Real iPhone device upload evidence, capture submission document, and raw upload completion marker for the same capture/job id."
    if field_name == "pipeline.pubsub_handoff_succeeded":
        return "A real Pub/Sub or equivalent pipeline handoff result from the uploaded capture, not a local placeholder."
    if field_name.startswith("privacy_provider."):
        return "Privacy-safe walkthrough and provider input audit proving raw video bypass is disabled."
    if field_name.startswith("retrieval."):
        return "Capture root with site_id, privacy-safe walkthrough, world_model_export/dense_index.jsonl, and sites/<site_id>/reference_memory/site_reference_manifest.json."
    if field_name.startswith("hosted_session."):
        return "Hosted runtime URL, WebApp listing/attachment id, and authenticated buyer access check against a real configured route."
    if field_name in {"payouts.provider_name", "payouts.contract_readiness_not_live_readiness"}:
        return "Launch proof names the current payout provider and separates mocked contract readiness from live provider readiness."
    if field_name in {"payouts.live_payout_execution_human_gate", "payouts.identity_kyc_state_documented", "payouts.background_check_state_documented"}:
        return "Repo-safe payment/KYC/background posture is documented without claiming live payout or screening readiness."
    if field_name in {"payouts.stripe_state_checked", "payouts.provider_state_checked", "payouts.live_provider_ready"}:
        return "Live backend Stripe account check returns provider_state_checked=true, provider_mode=live, live_provider_ready=true, and no blocking requirements."
    if field_name == "payouts.backend_configured":
        return "Live backend has configured Stripe secrets/routes; this is necessary but not sufficient for payout readiness."
    if field_name.startswith("payouts."):
        return "Stripe/payment backend state and payout guardrails checked for the marketplace flow without converting mocks into live proof."
    if field_name in {"ops.payout_exception_monitor_repo_contract", "ops.human_finance_review_gate"}:
        return "Repo evidence shows payout exception monitoring and human finance review are fail-closed contracts."
    if field_name.startswith("ops."):
        return "Launch owner plus failed-upload, submission, device sync, pipeline, payout, session-event, and cloud-log monitors."
    if field_name.startswith("meta_glasses."):
        return "Physical Meta glasses smoke evidence and explicit internal-only marketing guardrails."
    return str(blocker.get("message") or "").strip()


def _lane_result_names_for_proof_field(field_name: str) -> tuple[str, ...]:
    if field_name.startswith("city."):
        return ("city_backend.webapp-status-route.json",)
    if field_name.startswith("privacy_provider."):
        return ("privacy_safe_provider.capture-root-evidence.json",)
    if field_name.startswith("retrieval."):
        return ("site_identity_dense_export.capture-root-evidence.json",)
    if field_name.startswith("hosted_session."):
        return ("runtime_webapp_buyer_access.capture-root-evidence.json",)
    if field_name.startswith("pipeline."):
        return ("pipeline.capture-root-evidence.json",)
    return ()


def build_launch_gap_report(
    *,
    city_slug: str,
    status: str,
    blockers: Sequence[Mapping[str, Any]],
    packets: Sequence[WorkPacket],
    capture_root: str,
    run_root: Path,
    work_packet_root: Path,
    lane_results_root: Path,
) -> dict[str, Any]:
    packet_ids = {packet.lane_id for packet in packets}
    gaps: list[dict[str, Any]] = []
    for blocker in blockers:
        field_name = str(blocker.get("proof_field") or "")
        packet_id = _work_packet_for_proof_field(field_name)
        gap: dict[str, Any] = {
            "id": blocker.get("id"),
            "proof_field": field_name,
            "lane_id": blocker.get("lane_id"),
            "dependency_class": _dependency_class(blocker),
            "message": blocker.get("message"),
            "actual": blocker.get("actual"),
            "expected": blocker.get("expected"),
            "human_required": blocker.get("human_required"),
            "expected_evidence": _expected_evidence_for_blocker(blocker),
        }
        if packet_id in packet_ids:
            gap["work_packet_path"] = str(work_packet_root / f"{packet_id}.json")
        lane_result_paths = [
            str(candidate)
            for name in _lane_result_names_for_proof_field(field_name)
            if (candidate := lane_results_root / name).is_file()
        ]
        if lane_result_paths:
            gap["lane_result_paths"] = lane_result_paths
        gaps.append(gap)
    external_classes = {
        "live_city_backend",
        "real_device_iphone_capture",
        "live_pipeline_handoff",
        "privacy_safe_provider_artifact",
        "hosted_buyer_access",
        "payment_or_payout_ops",
        "launch_ops_monitoring",
        "internal_hardware_evidence",
        "human_required_external_evidence",
    }
    repo_gap_classes = {"repo_or_contract_gap", "capture_root_or_retrieval_export"}
    capture_root_shape = [
        "capture_descriptor.json with site_id or metadata.site_identity.site_id",
        "qa_report.json",
        "pipeline/opportunity_handoff.json",
        "pipeline/.qualification_pipeline_complete",
        "privacy/final_walkthrough.mov or privacy/final_walkthrough.mp4 or pipeline/privacy_processing_manifest.json final_walkthrough_uri",
        "world_model_export/dense_index.jsonl",
        "sites/<site_id>/reference_memory/site_reference_manifest.json",
        "pipeline/webapp_sync_result.json with status=succeeded, no placeholder flags, and buyer_access_check.buyer_accessible=true",
    ]
    return {
        "schema_version": "city-launch-gap-report.v1",
        "city_slug": city_slug,
        "status": status,
        "generated_at": utc_now_iso(),
        "run_root": str(run_root),
        "capture_root": capture_root,
        "blocker_count": len(blockers),
        "first_blocker": dict(blockers[0]) if blockers else None,
        "repo_or_contract_gaps": [gap for gap in gaps if gap["dependency_class"] in repo_gap_classes],
        "external_dependencies": [gap for gap in gaps if gap["dependency_class"] in external_classes],
        "launch_policy_gaps": [
            gap
            for gap in gaps
            if gap["dependency_class"] not in external_classes and gap["dependency_class"] not in repo_gap_classes
        ],
        "expected_capture_root_shape": capture_root_shape,
        "rerun_command": [
            "python",
            "scripts/run_autonomous_city_launch_harness.py",
            "--city-slug",
            city_slug,
            "--budget-cents",
            "<budget-cents>",
            "--capture-path",
            "iphone",
            "--execute-local",
            "--include-pipeline-tests",
            "--include-webapp-city-status",
            "--capture-root",
            "<capture-root>",
        ],
    }


def apply_lane_results(proof: dict[str, Any], results_dir: Path) -> list[dict[str, Any]]:
    applied: list[dict[str, Any]] = []
    if not results_dir.is_dir():
        return applied
    for path in sorted(results_dir.glob("*.json")):
        if path.name.endswith(".not-executed.json"):
            continue
        result = read_json(path)
        status = str(result.get("status") or "").strip().lower()
        if status == "contract_only" or result.get("contract_only") is True:
            continue
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
    command_env = contract_test_env() if command.command and command.command[0] == "pytest" else os.environ.copy()
    completed = subprocess.run(
        command.command,
        cwd=command.cwd,
        capture_output=True,
        text=True,
        env=command_env,
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


def _webapp_sync_has_placeholder(sync_result: Mapping[str, Any]) -> bool:
    placeholder_flags = (
        "placeholder_fallback_allowed",
        "placeholder_sync_used",
        "placeholder_request_created",
        "placeholder_request",
        "placeholder",
    )
    return any(bool(sync_result.get(flag)) for flag in placeholder_flags)


def _lane_result_name_for_proof(path: Path) -> str:
    return f"cross_repo_proof.{path.name}"


def _capture_root_site_id(descriptor: Mapping[str, Any]) -> str:
    metadata = descriptor.get("metadata") if isinstance(descriptor.get("metadata"), Mapping) else {}
    site_identity = metadata.get("site_identity") if isinstance(metadata.get("site_identity"), Mapping) else {}
    return str(site_identity.get("site_id") or descriptor.get("site_id") or "").strip()


def _first_existing_path(paths: Iterable[Path]) -> Path | None:
    for path in paths:
        if path.is_file():
            return path
    return None


def _capture_root_site_reference_candidates(*, capture_root: Path, descriptor: Mapping[str, Any]) -> list[Path]:
    candidates: list[Path] = []
    site_id = _capture_root_site_id(descriptor)
    if site_id:
        try:
            ctx = resolve_local_capture_context(capture_root)
            candidates.append(ctx.storage_root / ctx.bucket / "sites" / site_id / "reference_memory" / "site_reference_manifest.json")
        except Exception:
            pass
    candidates.extend(capture_root.glob("sites/*/reference_memory/site_reference_manifest.json"))
    return candidates


def _has_privacy_safe_walkthrough(*, capture_root: Path, descriptor: Mapping[str, Any], privacy_manifest: Mapping[str, Any]) -> bool:
    if (capture_root / "privacy" / "final_walkthrough.mov").is_file():
        return True
    if (capture_root / "privacy" / "final_walkthrough.mp4").is_file():
        return True
    return bool(
        str(privacy_manifest.get("final_walkthrough_uri") or "").strip()
        or str(descriptor.get("world_model_video_uri") or "").strip()
        or str(descriptor.get("privacy_processed_video_uri") or "").strip()
    )


def _city_query_from_slug(city_slug: str) -> tuple[str, str | None]:
    parts = [part for part in city_slug.strip().split("-") if part]
    if len(parts) >= 2 and len(parts[-1]) == 2:
        city = " ".join(part.capitalize() for part in parts[:-1])
        return city, parts[-1].upper()
    return " ".join(part.capitalize() for part in parts), None


def _webapp_origin_from_configured_url(value: str) -> str:
    parsed = urllib.parse.urlparse(value.strip())
    if parsed.scheme and parsed.netloc:
        return urllib.parse.urlunparse((parsed.scheme, parsed.netloc, "", "", "", ""))
    return value.strip().rstrip("/")


def _webapp_launch_status_url(*, base_url: str, city_slug: str) -> str:
    origin = _webapp_origin_from_configured_url(base_url)
    city, state_code = _city_query_from_slug(city_slug)
    query = {"city": city}
    if state_code:
        query["state_code"] = state_code
    return urllib.parse.urljoin(origin.rstrip("/") + "/", "api/public/launch/status") + "?" + urllib.parse.urlencode(query)


def collect_webapp_city_status_evidence(
    *,
    city_slug: str,
    lane_results_root: Path,
) -> list[dict[str, Any]]:
    base_url = str(os.environ.get("PIPELINE_SYNC_WEBAPP_URL") or "").strip()
    blockers: list[str] = []
    evidence: dict[str, Any] = {}
    diagnostics: dict[str, Any] = {}
    if not base_url:
        blockers.append("PIPELINE_SYNC_WEBAPP_URL_missing")
    else:
        status_url = _webapp_launch_status_url(base_url=base_url, city_slug=city_slug)
        diagnostics["status_url"] = status_url
        try:
            with urllib.request.urlopen(status_url, timeout=WEBAPP_STATUS_TIMEOUT_SECONDS) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            blockers.append(f"webapp_status_route_http_{exc.code}")
            try:
                error_body = exc.read().decode("utf-8", errors="replace")
                error_payload = json.loads(error_body)
                if isinstance(error_payload, Mapping):
                    error_message = str(error_payload.get("error") or error_payload.get("message") or "").strip()
                    if error_message:
                        blockers.append(f"webapp_status_route_error:{error_message[:500]}")
            except Exception:
                pass
            payload = {}
        except Exception as exc:
            blockers.append(f"webapp_status_route_unreachable:{type(exc).__name__}")
            payload = {}
        if isinstance(payload, Mapping):
            current_city = payload.get("currentCity") if isinstance(payload.get("currentCity"), Mapping) else {}
            supported_cities = payload.get("supportedCities") if isinstance(payload.get("supportedCities"), list) else []
            supported_slugs = {
                str(city.get("citySlug") or "").strip()
                for city in supported_cities
                if isinstance(city, Mapping)
            }
            source_status = payload.get("sourceStatus") if isinstance(payload.get("sourceStatus"), Mapping) else {}
            source_unavailable = any(
                str(source_status.get(key) or "").strip() == "unavailable"
                for key in (
                    "cityLaunchActivations",
                    "cityLaunchProspects",
                    "cityLaunchCandidateSignals",
                )
            )
            route_supported = (
                payload.get("ok") is True
                and str(current_city.get("citySlug") or "").strip() == city_slug
                and current_city.get("isSupported") is True
                and current_city.get("status") == "live"
                and city_slug in supported_slugs
                and not source_unavailable
            )
            evidence["city.backend_supported"] = route_supported
            diagnostics["current_city"] = {
                "citySlug": current_city.get("citySlug"),
                "isSupported": current_city.get("isSupported"),
                "status": current_city.get("status"),
            }
            diagnostics["supported_city_count"] = len(supported_slugs)
            if source_status:
                diagnostics["source_status"] = {
                    "cityLaunchActivations": source_status.get("cityLaunchActivations"),
                    "cityLaunchProspects": source_status.get("cityLaunchProspects"),
                    "cityLaunchCandidateSignals": source_status.get("cityLaunchCandidateSignals"),
                    "warnings": source_status.get("warnings"),
                }
            if source_unavailable:
                blockers.append("webapp_status_route_source_unavailable")
            if not route_supported and not blockers:
                blockers.append("webapp_status_route_city_not_live_supported")
        elif not blockers:
            blockers.append("webapp_status_route_invalid_json")
    lane_blockers = list(blockers)
    if lane_blockers:
        lane_blockers.extend(f"{key}={value}" for key, value in diagnostics.items())
    return [
        _write_lane_result(
            lane_results_root / "city_backend.webapp-status-route.json",
            lane_id="city_backend",
            status="succeeded" if evidence.get("city.backend_supported") is True and not blockers else "blocked",
            evidence=evidence,
            blockers=lane_blockers,
        )
    ]


def collect_cross_repo_proof_evidence(
    *,
    city_slug: str,
    proof_files: Sequence[Path],
    lane_results_root: Path,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for proof_file in proof_files:
        blockers: list[str] = []
        evidence: dict[str, Any] = {}
        if not proof_file.is_file():
            blockers.append(f"proof_file_missing:{proof_file}")
        else:
            payload = _optional_json(proof_file)
            proof_city_slug = str(payload.get("city_slug") or "").strip()
            if payload.get("contract_only") is True:
                blockers.append("contract_only_proof_rejected")
            if proof_city_slug and proof_city_slug != city_slug:
                blockers.append(f"city_slug_mismatch:{proof_city_slug}")
            if not blockers:
                for field_name, *_rest in REQUIRED_LAUNCH_PROOF_FIELDS:
                    value = get_nested(payload, field_name)
                    if value is not None:
                        evidence[field_name] = value
                if not evidence:
                    blockers.append("proof_file_contains_no_required_evidence")
        results.append(
            _write_lane_result(
                lane_results_root / _lane_result_name_for_proof(proof_file),
                lane_id="cross_repo_proof",
                status="succeeded" if evidence and not blockers else "blocked",
                evidence=evidence if not blockers else {},
                blockers=blockers,
            )
        )
    return results


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
    dense_index_candidates = [
        capture_root / "world_model_export" / "dense_index.jsonl",
        pipeline_root / "world_model_export" / "dense_index.jsonl",
    ]
    dense_index = _first_existing_path(dense_index_candidates)
    site_reference_candidates = _capture_root_site_reference_candidates(
        capture_root=capture_root,
        descriptor=descriptor,
    )
    site_reference = _first_existing_path(site_reference_candidates)

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
        "retrieval.dense_index_exists": bool(dense_index and dense_index.is_file()),
        "retrieval.site_reference_manifest_exists": bool(site_reference and site_reference.is_file()),
    }
    retrieval_blockers = [key for key, value in retrieval_evidence.items() if not value]
    if not _capture_root_site_id(descriptor):
        retrieval_blockers.append("capture_descriptor_missing_site_id_or_metadata_site_identity_site_id")
    if not _has_privacy_safe_walkthrough(
        capture_root=capture_root,
        descriptor=descriptor,
        privacy_manifest=privacy_manifest,
    ):
        retrieval_blockers.append("privacy_safe_walkthrough_missing_for_retrieval_index")
    results.append(
        _write_lane_result(
            lane_results_root / "site_identity_dense_export.capture-root-evidence.json",
            lane_id="site_identity_dense_export",
            status="succeeded" if not retrieval_blockers else "blocked",
            evidence=retrieval_evidence,
            blockers=retrieval_blockers,
        )
    )

    response_ids = webapp_sync.get("webapp_response_ids") if isinstance(webapp_sync.get("webapp_response_ids"), Mapping) else {}
    buyer_access = webapp_sync.get("buyer_access_check") if isinstance(webapp_sync.get("buyer_access_check"), Mapping) else {}
    webapp_sync_valid = (
        webapp_sync.get("status") == "succeeded"
        and not _webapp_sync_has_placeholder(webapp_sync)
    )
    runtime_url = str(registration.get("runtime_base_url") or health.get("runtime_base_url") or "").strip()
    webapp_listing_id = (
        str(response_ids.get("listing_id") or response_ids.get("attachment_id") or response_ids.get("pipeline_attachment_id") or "").strip()
        if webapp_sync_valid
        else ""
    )
    hosted_evidence = {
        "hosted_session.runtime_url": runtime_url,
        "hosted_session.webapp_listing_id": webapp_listing_id,
        "hosted_session.buyer_access_checked": (
            webapp_sync_valid
            and bool(buyer_access.get("buyer_access_checked"))
            and bool(buyer_access.get("buyer_accessible"))
        ),
    }
    hosted_blockers = [key for key, value in hosted_evidence.items() if not value]
    if webapp_sync and not webapp_sync_valid:
        hosted_blockers.append("webapp_sync_placeholder_or_not_succeeded")
    results.append(
        _write_lane_result(
            lane_results_root / "runtime_webapp_buyer_access.capture-root-evidence.json",
            lane_id="runtime_webapp_buyer_access",
            status="succeeded" if all(bool(value) for value in hosted_evidence.values()) else "blocked",
            evidence=hosted_evidence,
            blockers=hosted_blockers,
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
            result_path = results_dir / f"{packet.lane_id}.{command.id}.json"
            result["result_path"] = str(result_path)
            command_results.append(result)
            packet_evidence["harness.local_execution.executed_command_count"] = (
                int(packet_evidence["harness.local_execution.executed_command_count"]) + 1
            )
            write_json(result_path, result)
            if command.id == "release_config_validation":
                proof.setdefault("harness", {})["release_config_validation"] = {
                    "status": result["status"],
                    "exit_code": result["exit_code"],
                    "stdout_tail": result["stdout_tail"],
                    "stderr_tail": result["stderr_tail"],
                    "command_result_path": str(result_path),
                }
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
    atomic_write_text(
        path,
        "".join(json.dumps(dict(row), sort_keys=True) + "\n" for row in rows),
    )


def run_harness(args: argparse.Namespace) -> dict[str, Any]:
    pipeline_repo = Path(args.pipeline_repo).expanduser().resolve()
    capture_repo = Path(args.capture_repo).expanduser().resolve()
    webapp_repo = Path(args.webapp_repo).expanduser().resolve()
    env_summary = load_env_files([pipeline_repo, capture_repo, webapp_repo])
    capture_paths = tuple(args.capture_path or ("iphone",))
    run_id = args.run_id or f"{utc_now_iso().replace(':', '').replace('+', 'Z')}-{uuid.uuid4().hex[:8]}"
    output_root = prepare_evidence_root(Path(args.output_root), source_root=ROOT)
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
    proof["harness"]["env_files_loaded"] = env_summary

    packets = build_work_packets(
        pipeline_repo=pipeline_repo,
        capture_repo=capture_repo,
        webapp_repo=webapp_repo,
        capture_paths=capture_paths,
    )
    prepare_run_root(run_root)
    for packet in packets:
        write_json(work_packet_root / f"{packet.lane_id}.json", packet.to_dict())

    applied_lane_results = apply_lane_results(proof, lane_results_root)
    command_results = execute_local_packets(packets=packets, proof=proof, run_root=run_root, args=args)
    webapp_city_status_results = (
        collect_webapp_city_status_evidence(
            city_slug=args.city_slug,
            lane_results_root=lane_results_root,
        )
        if getattr(args, "include_webapp_city_status", False)
        else []
    )
    explicit_proof_files = [
        Path(path).expanduser().resolve()
        for path in (getattr(args, "proof_file", None) or [])
        if str(path).strip()
    ]
    auto_proof_files = [
        capture_repo / "ops" / "launch-readiness" / f"{args.city_slug}.launch-proof.json",
        webapp_repo / "ops" / "launch-readiness" / f"{args.city_slug}.launch-proof.json",
    ]
    proof_files = list(dict.fromkeys([*explicit_proof_files, *[path for path in auto_proof_files if path.is_file()]]))
    cross_repo_proof_results = collect_cross_repo_proof_evidence(
        city_slug=args.city_slug,
        proof_files=proof_files,
        lane_results_root=lane_results_root,
    )
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
    elif cross_repo_proof_results or webapp_city_status_results:
        applied_lane_results.extend(apply_lane_results(proof, lane_results_root))
    elif args.execute_local:
        applied_lane_results.extend(apply_lane_results(proof, lane_results_root))
    blockers = build_blockers(proof)
    status = determine_status(proof, blockers, capture_root_evidence=capture_root_evidence)
    gap_report_path = run_root / "launch-gap-report.json"
    gap_report = build_launch_gap_report(
        city_slug=args.city_slug,
        status=status,
        blockers=blockers,
        packets=packets,
        capture_root=capture_root_arg,
        run_root=run_root,
        work_packet_root=work_packet_root,
        lane_results_root=lane_results_root,
    )
    proof["launch_proof_status"] = status
    proof["harness"]["applied_lane_results"] = applied_lane_results
    proof["harness"]["command_result_count"] = len(command_results)
    proof["harness"]["capture_root_evidence_result_count"] = len(capture_root_evidence)
    proof["harness"]["cross_repo_proof_result_count"] = len(cross_repo_proof_results)
    proof["harness"]["webapp_city_status_result_count"] = len(webapp_city_status_results)
    proof["harness"]["work_packet_count"] = len(packets)
    proof["harness"]["launch_gap_report"] = str(gap_report_path)
    proof["harness"]["updated_at"] = utc_now_iso()

    created_or_updated_at = utc_now_iso()
    manifest = {
        "schema_version": RUN_SCHEMA_VERSION,
        "run_id": run_id,
        "city_slug": args.city_slug,
        "budget_cents": args.budget_cents,
        "capture_paths": list(capture_paths),
        "status": status,
        "created_or_updated_at": created_or_updated_at,
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
            "launch_gap_report": str(gap_report_path),
        },
        "env_files_loaded": env_summary,
        "first_blocker": blockers[0] if blockers else None,
        "evidence_policy": evidence_policy(
            created_at=datetime.fromisoformat(created_or_updated_at.replace("Z", "+00:00"))
        ),
    }
    summary = {
        "run_id": run_id,
        "status": status,
        "first_blocker": blockers[0] if blockers else None,
        "blocker_count": len(blockers),
        "work_packet_count": len(packets),
        "proof_path": str(proof_path),
        "launch_gap_report": str(gap_report_path),
    }

    write_json(proof_path, proof)
    write_json(run_root / "summary.json", summary)
    write_jsonl(run_root / "blockers.jsonl", blockers)
    write_json(gap_report_path, gap_report)
    manifest["artifact_inventory"] = build_artifact_inventory(run_root)
    write_json(run_root / "manifest.json", manifest)
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
    parser.add_argument(
        "--output-root",
        default=str(default_output_root()),
        help=(
            "Private evidence root outside the source checkout. Defaults to "
            "BLUEPRINT_CITY_LAUNCH_OUTPUT_ROOT or the user state directory."
        ),
    )
    parser.add_argument("--capture-root", help="Optional real capture root to scan for Pipeline lane evidence.")
    parser.add_argument("--proof-file", action="append", help="Optional real cross-repo launch proof JSON to merge.")
    parser.add_argument(
        "--include-webapp-city-status",
        action="store_true",
        help="Check the configured WebApp public launch-status route for city backend support.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_harness(args)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if not str(summary["status"]).startswith("blocked_repo") else 2


if __name__ == "__main__":  # pragma: no cover - exercised through main().
    raise SystemExit(main())
