#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


EXPECTED_MAX_UPLOAD_BYTES = 20 * 1024 * 1024 * 1024
EXPECTED_MAX_DURATION_SECONDS = 45 * 60
EXPECTED_INLINE_EXTRACT_BYTES = 1_500_000_000
EXPECTED_TARGET_CONCURRENCY = 25
EXPECTED_QUEUE_DEPTH_ALERT_THRESHOLD = 50
EXPECTED_QUEUE_DEPTH_ALERT_DURATION_SECONDS = 300
EXPECTED_FIRESTORE_CREATED_AT_SHARD_FIELD = "createdAtShard"
EXPECTED_FIRESTORE_LATENCY_METRIC = "serviceruntime.googleapis.com/api/request_latencies"
EXPECTED_FIRESTORE_P99_ALERT_THRESHOLD_SECONDS = 0.25
EXPECTED_FIRESTORE_P99_ALERT_DURATION_SECONDS = 300
EXPECTED_GPU_RUNNER_MAX_INSTANCES = {
    "sam3": 3,
    "vip": 2,
    "deepprivacy2": 2,
    "video_to_world": 2,
}
EXPECTED_LARGE_VIDEO_INGEST_TOPIC = "blueprint-large-video-ingest"
EXPECTED_COHORT_HARD_STOP_USD = 5000
EXPECTED_COHORT_REVIEW_THRESHOLD_USD = 2500
EXPECTED_MODELED_CAPTURES_PER_MONTH = 300
EXPECTED_LOCAL_ROBOT_EVAL_JOBS_PER_MONTH = 75
EXPECTED_LOCAL_ROBOT_EVAL_JOBS_REVIEW_GIB = 25
EXPECTED_LOCAL_ROBOT_EVAL_JOBS_HARD_STOP_GIB = 50
EXPECTED_BILLING_BUDGET_THRESHOLDS = [0.5, 0.8, 1.0]
EXPECTED_BETA_DATA_RETENTION_POLICY_FILE = "docs/beta_data_retention_policy_2026-07-09.json"
EXPECTED_BETA_DATA_RETENTION_POLICY_DOC = "docs/BETA_DATA_RETENTION_POLICY_2026-07-09.md"
EXPECTED_BETA_DATA_RETENTION_POLICY_SCHEMA = "blueprint.beta_data_retention_policy.v1"
EXPECTED_BETA_DATA_RETENTION_STATUS = "declared_validator_enforced_operator_signoff_required"
EXPECTED_OPERATOR_DPA_EVIDENCE_ID = "operator_dpa_data_processing_terms"
EXPECTED_STORAGE_DATA_CLASSES = {
    "raw_capture_truth": {
        "prefixes": ["scenes/", "targets/"],
        "nearline_after_days": 30,
        "coldline_after_days": 90,
        "delete_after_days": 180,
    },
    "temporary_processing": {
        "prefixes": ["tmp/", "staging/", "debug/"],
        "delete_after_days": 14,
    },
    "buyer_eval_hosted_artifacts": {
        "prefixes": [
            "buyer_delivery/",
            "marketplace/",
            "hosted_sessions/",
            "robot_eval_jobs/",
        ],
        "delete_after_days": 365,
        "contract_specific_hold_may_override": True,
    },
}


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise AssertionError(f"{path} must contain a JSON object")
    return payload


def _find_rule(lifecycle: dict[str, Any], action_type: str, age: int, prefixes: set[str]) -> dict[str, Any]:
    for rule in lifecycle.get("rule", []):
        if not isinstance(rule, dict):
            continue
        action = rule.get("action") if isinstance(rule.get("action"), dict) else {}
        condition = rule.get("condition") if isinstance(rule.get("condition"), dict) else {}
        if action.get("type") != action_type:
            continue
        if condition.get("age") != age:
            continue
        if prefixes.issubset(set(condition.get("matchesPrefix") or [])):
            return rule
    raise AssertionError(f"missing lifecycle rule action={action_type} age={age} prefixes={sorted(prefixes)}")


def _require_text(path: Path, required: list[str]) -> None:
    text = path.read_text(encoding="utf-8")
    for needle in required:
        if needle not in text:
            raise AssertionError(f"{path} is missing required text: {needle}")


def _forbid_text(path: Path, forbidden: list[str]) -> None:
    text = path.read_text(encoding="utf-8")
    for needle in forbidden:
        if needle in text:
            raise AssertionError(f"{path} contains forbidden text: {needle}")


def _validate_capture_swift_constants(path: Path) -> None:
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    if "betaMaxFileSizeBytes" not in text or "20 * 1024 * 1024 * 1024" not in text:
        raise AssertionError(f"{path} does not define the 20 GiB beta upload size cap")
    if "betaMaxDurationSeconds" not in text or "45 * 60" not in text:
        raise AssertionError(f"{path} does not define the 45 minute beta duration cap")
    if not re.search(r"maxFileSizeBytes:\s*Int64", text):
        raise AssertionError(f"{path} must keep an explicit maxFileSizeBytes policy field")
    if not re.search(r"maxDurationSeconds:\s*Double", text):
        raise AssertionError(f"{path} must keep an explicit maxDurationSeconds policy field")


def validate_files(repo_root: Path, capture_swift_policy: Path | None = None) -> dict[str, Any]:
    model_path = repo_root / "docs" / "beta_capacity_cost_storage_model_2026-07-08.json"
    retention_policy_path = repo_root / EXPECTED_BETA_DATA_RETENTION_POLICY_FILE
    retention_policy_doc_path = repo_root / EXPECTED_BETA_DATA_RETENTION_POLICY_DOC
    lifecycle_path = repo_root / "deploy" / "storage" / "primary-capture-bucket-lifecycle.json"
    doc_path = repo_root / "docs" / "BETA_CAPACITY_COST_STORAGE_MODEL_2026-07-08.md"
    terraform_path = repo_root / "deploy" / "terraform" / "main.tf"
    terraform_vars_example_path = repo_root / "deploy" / "terraform" / "terraform.tfvars.example"
    spend_admission_doc_path = repo_root / "docs" / "PAID_SPEND_ADMISSION_LOCK.md"
    spend_guard_service_path = (
        repo_root / "deploy" / "systemd" / "blueprint-gpu-spend-guard.service"
    )

    model = _load_json(model_path)
    retention_policy = _load_json(retention_policy_path)
    lifecycle = _load_json(lifecycle_path)

    if model.get("schema_version") != "blueprint.beta_capacity_cost_storage_model.v1":
        raise AssertionError("unexpected capacity model schema_version")
    if model.get("beta_target", {}).get("external_users") != 100:
        raise AssertionError("capacity model must target 100 external users")
    if model.get("beta_target", {}).get("modeled_captures_per_month") != EXPECTED_MODELED_CAPTURES_PER_MONTH:
        raise AssertionError("capacity model must explicitly model 300 captures/month")
    if model.get("beta_target", {}).get("target_concurrent_uploaders") != EXPECTED_TARGET_CONCURRENCY:
        raise AssertionError("capacity model must target 25 concurrent uploaders")

    limits = model.get("per_capture_limits", {})
    if limits.get("max_upload_payload_bytes") != EXPECTED_MAX_UPLOAD_BYTES:
        raise AssertionError("capacity model max_upload_payload_bytes must be 20 GiB")
    if limits.get("max_duration_seconds") != EXPECTED_MAX_DURATION_SECONDS:
        raise AssertionError("capacity model max_duration_seconds must be 45 minutes")
    if limits.get("inline_extract_frames_max_video_bytes") != EXPECTED_INLINE_EXTRACT_BYTES:
        raise AssertionError("capacity model inline extract limit must match extractFrames default")

    budget_guardrails = model.get("budget_guardrails", {})
    if budget_guardrails.get("cohort_hard_stop_threshold_usd") != EXPECTED_COHORT_HARD_STOP_USD:
        raise AssertionError("capacity model must pin the cohort provider spend hard-stop threshold")
    if budget_guardrails.get("requires_paid_spend_admission_lock") is not True:
        raise AssertionError("capacity model must require the paid spend admission lock")
    if budget_guardrails.get("paid_spend_admission_lock_schema") != (
        "blueprint.paid_spend_admission_lock.v1"
    ):
        raise AssertionError("capacity model must pin the admission lock schema")
    if budget_guardrails.get("threshold_equality_blocks_new_paid_work") is not True:
        raise AssertionError("capacity model must block admission at exactly $5000")
    if budget_guardrails.get("requires_current_provider_billing_reconciliation") is not True:
        raise AssertionError("capacity model must require current billing reconciliation")
    if budget_guardrails.get("requires_page_event_and_controlled_drain") is not True:
        raise AssertionError("capacity model must require page and controlled drain evidence")
    if budget_guardrails.get("maximum_override_duration_seconds") != 14400:
        raise AssertionError("capacity model must cap paid spend overrides at four hours")
    if budget_guardrails.get("requires_gcp_billing_budget") is not True:
        raise AssertionError("capacity model must require a GCP billing budget")
    if budget_guardrails.get("gcp_billing_budget_resource") != "google_billing_budget.gpu_fleet_beta":
        raise AssertionError("capacity model must name the Terraform billing budget resource")
    if budget_guardrails.get("gcp_billing_budget_thresholds") != EXPECTED_BILLING_BUDGET_THRESHOLDS:
        raise AssertionError("capacity model must pin GCP billing budget thresholds")

    cost_model = model.get("cost_per_capture_model", {})
    if cost_model.get("schema_version") != "blueprint.beta_cost_per_capture_model.v1":
        raise AssertionError("capacity model must include a cost-per-capture model")
    if cost_model.get("modeled_captures_per_month") != EXPECTED_MODELED_CAPTURES_PER_MONTH:
        raise AssertionError("cost model must use the modeled monthly capture count")
    if cost_model.get("budget_cap_usd_per_100_user_month") != EXPECTED_COHORT_HARD_STOP_USD:
        raise AssertionError("cost model must carry the 100-user hard-stop budget")
    if cost_model.get("provider_spend_review_threshold_usd_per_100_user_month") != EXPECTED_COHORT_REVIEW_THRESHOLD_USD:
        raise AssertionError("cost model must carry the provider spend review threshold")
    if cost_model.get("budget_cap_usd_per_capture") != round(
        EXPECTED_COHORT_HARD_STOP_USD / EXPECTED_MODELED_CAPTURES_PER_MONTH,
        2,
    ):
        raise AssertionError("cost model must calculate the per-capture budget cap")
    if cost_model.get("provider_spend_review_threshold_usd_per_capture") != round(
        EXPECTED_COHORT_REVIEW_THRESHOLD_USD / EXPECTED_MODELED_CAPTURES_PER_MONTH,
        2,
    ):
        raise AssertionError("cost model must calculate the per-capture review threshold")
    if cost_model.get("storage_gib_per_capture_p50") != 4.2:
        raise AssertionError("cost model must pin p50 storage GiB per capture")
    if cost_model.get("storage_gib_per_capture_p95") != 28:
        raise AssertionError("cost model must pin p95 storage GiB per capture")
    if cost_model.get("gpu_seconds_per_capture_estimate") != 1200:
        raise AssertionError("cost model must pin GPU seconds per capture")
    if cost_model.get("estimated_usd_per_capture_p50") != 3.56:
        raise AssertionError("cost model must pin the p50 estimated cost per capture")

    runtime_capacity = model.get("runtime_capacity", {})
    if runtime_capacity.get("target_concurrent_jobs") != EXPECTED_TARGET_CONCURRENCY:
        raise AssertionError("runtime capacity must target 25 concurrent jobs")
    if runtime_capacity.get("task_queue_max_concurrent_dispatches") != EXPECTED_TARGET_CONCURRENCY:
        raise AssertionError("task queue capacity must target 25 concurrent dispatches")
    if runtime_capacity.get("task_queue_depth_alert_threshold") != EXPECTED_QUEUE_DEPTH_ALERT_THRESHOLD:
        raise AssertionError("runtime capacity must pin the queue-depth beta alert threshold")
    if runtime_capacity.get("task_queue_depth_alert_duration_seconds") != EXPECTED_QUEUE_DEPTH_ALERT_DURATION_SECONDS:
        raise AssertionError("runtime capacity must pin the queue-depth beta alert duration")
    if runtime_capacity.get("gpu_runner_max_instances") != EXPECTED_GPU_RUNNER_MAX_INSTANCES:
        raise AssertionError("runtime capacity must pin per-service GPU runner max instances")
    if runtime_capacity.get("gpu_runner_total_max_instances") != sum(EXPECTED_GPU_RUNNER_MAX_INSTANCES.values()):
        raise AssertionError("runtime capacity must pin total GPU runner max instances")
    if runtime_capacity.get("large_video_ingest_topic") != EXPECTED_LARGE_VIDEO_INGEST_TOPIC:
        raise AssertionError("runtime capacity must name the large-video ingest topic")
    hotspot_policy = runtime_capacity.get("firestore_created_at_hotspot_policy", {})
    if hotspot_policy.get("schema_version") != "blueprint.firestore_created_at_hotspot_policy.v2":
        raise AssertionError("runtime capacity must include the v2 Firestore createdAt hotspot policy")
    if hotspot_policy.get("collection") != "creatorCaptures":
        raise AssertionError(
            "Firestore hotspot policy must cover the creatorCaptures collection "
            "(the literal `captures` collection was a v1 phantom nothing writes)"
        )
    if hotspot_policy.get("index_manifest") != "Blueprint-WebApp/firestore.indexes.json":
        raise AssertionError("Firestore hotspot policy must name the webapp-owned index manifest")
    if hotspot_policy.get("shard_field") != EXPECTED_FIRESTORE_CREATED_AT_SHARD_FIELD:
        raise AssertionError("Firestore hotspot policy must pin the createdAt shard field")
    if hotspot_policy.get("shard_count") != 16:
        raise AssertionError("Firestore hotspot policy must pin the canonical 16-way shard count")
    if hotspot_policy.get("scale_up_sharded_indexes") != [
        "creatorCaptures: creator_id ASC, createdAtShard ASC, created_at DESC",
        "creatorCaptures: status ASC, createdAtShard ASC, created_at ASC",
    ]:
        raise AssertionError("Firestore hotspot policy must pin the creatorCaptures sharded composites")
    if hotspot_policy.get("legacy_indexes_retained_for_current_readers") != [
        "creatorCaptures: creator_id ASC, created_at DESC",
    ]:
        raise AssertionError("Firestore hotspot policy must keep current-reader legacy indexes explicit")
    if hotspot_policy.get("beta_without_sharding_max_write_rate_per_second") != 500:
        raise AssertionError("Firestore hotspot policy must pin the unsharded sequential-index write-rate limit")
    if hotspot_policy.get("monitoring_alert_policy") != "google_monitoring_alert_policy.firestore_request_latency":
        raise AssertionError("Firestore hotspot policy must name the latency alert policy")
    if hotspot_policy.get("latency_metric") != EXPECTED_FIRESTORE_LATENCY_METRIC:
        raise AssertionError("Firestore hotspot policy must pin the Firestore latency metric")
    if hotspot_policy.get("p99_alert_threshold_seconds") != EXPECTED_FIRESTORE_P99_ALERT_THRESHOLD_SECONDS:
        raise AssertionError("Firestore hotspot policy must pin the p99 latency alert threshold")
    if hotspot_policy.get("p99_alert_duration_seconds") != EXPECTED_FIRESTORE_P99_ALERT_DURATION_SECONDS:
        raise AssertionError("Firestore hotspot policy must pin the p99 latency alert duration")
    if hotspot_policy.get("soak_report_observation_field") != "firestore_latency_observation":
        raise AssertionError("Firestore hotspot policy must name the soak report observation field")

    lifecycle_ref = model.get("storage_lifecycle", {})
    if lifecycle_ref.get("policy_file") != "deploy/storage/primary-capture-bucket-lifecycle.json":
        raise AssertionError("capacity model must point to the primary capture bucket lifecycle file")
    if lifecycle_ref.get("apply_script") != "scripts/apply_primary_capture_bucket_lifecycle.sh":
        raise AssertionError("capacity model must point to the lifecycle apply script")
    if lifecycle_ref.get("enforcement_layer") != "gcs_bucket_lifecycle":
        raise AssertionError("capacity model must identify GCS bucket lifecycle enforcement")
    if lifecycle_ref.get("package_retention_artifact") != "arena_delivery_retention_policy.v1":
        raise AssertionError("capacity model must name the package retention artifact schema")
    beta_retention_ref = lifecycle_ref.get("beta_data_retention_policy", {})
    if beta_retention_ref.get("policy_file") != EXPECTED_BETA_DATA_RETENTION_POLICY_FILE:
        raise AssertionError("capacity model must point to the beta data retention policy artifact")
    if beta_retention_ref.get("policy_doc") != EXPECTED_BETA_DATA_RETENTION_POLICY_DOC:
        raise AssertionError("capacity model must point to the beta data retention policy doc")
    if beta_retention_ref.get("schema_version") != EXPECTED_BETA_DATA_RETENTION_POLICY_SCHEMA:
        raise AssertionError("capacity model must pin the beta data retention policy schema")
    if beta_retention_ref.get("operator_signoff_evidence_id") != EXPECTED_OPERATOR_DPA_EVIDENCE_ID:
        raise AssertionError("capacity model must tie data retention to the operator DPA evidence id")
    if beta_retention_ref.get("launch_readiness_artifact_id") != "beta_data_retention_policy_json":
        raise AssertionError("capacity model must name the launch readiness artifact id for data retention")
    if lifecycle_ref.get("data_classes") != EXPECTED_STORAGE_DATA_CLASSES:
        raise AssertionError("capacity model must pin per-data-class retention policy")
    local_jobs = lifecycle_ref.get("local_robot_eval_jobs_retention", {})
    if local_jobs.get("schema_version") != "blueprint.local_robot_eval_jobs_retention.v1":
        raise AssertionError("capacity model must include local robot_eval_jobs retention schema")
    if local_jobs.get("root") != "robot_eval_jobs/":
        raise AssertionError("local robot_eval_jobs retention must point at repo-root robot_eval_jobs/")
    if local_jobs.get("retention_tool") != "scripts/manage_output_artifact_retention.py":
        raise AssertionError("local robot_eval_jobs retention must use the shared retention tool")
    if local_jobs.get("delete_after_days") != 30:
        raise AssertionError("local robot_eval_jobs retention must delete after 30 days by default")
    if local_jobs.get("review_threshold_gib") != EXPECTED_LOCAL_ROBOT_EVAL_JOBS_REVIEW_GIB:
        raise AssertionError("local robot_eval_jobs retention must pin the 25 GiB review threshold")
    if local_jobs.get("hard_stop_gib") != EXPECTED_LOCAL_ROBOT_EVAL_JOBS_HARD_STOP_GIB:
        raise AssertionError("local robot_eval_jobs retention must pin the 50 GiB hard stop")
    if local_jobs.get("modeled_captures_per_month") != EXPECTED_MODELED_CAPTURES_PER_MONTH:
        raise AssertionError("local robot_eval_jobs retention must use the 300 capture/month model")
    if local_jobs.get("planned_robot_eval_jobs_per_month") != EXPECTED_LOCAL_ROBOT_EVAL_JOBS_PER_MONTH:
        raise AssertionError("local robot_eval_jobs retention must model 75 robot eval jobs/month")
    if "not launch proof" not in local_jobs.get("canonical_handoff_rule", ""):
        raise AssertionError("local robot_eval_jobs retention must keep launch-proof boundary explicit")

    _find_rule(lifecycle, "SetStorageClass", 30, {"scenes/", "targets/"})
    _find_rule(lifecycle, "SetStorageClass", 90, {"scenes/", "targets/"})
    _find_rule(lifecycle, "Delete", 180, {"scenes/", "targets/"})
    _find_rule(lifecycle, "Delete", 14, {"tmp/", "staging/", "debug/"})
    _find_rule(lifecycle, "Delete", 365, {"buyer_delivery/", "marketplace/", "hosted_sessions/", "robot_eval_jobs/"})

    if retention_policy.get("schema_version") != EXPECTED_BETA_DATA_RETENTION_POLICY_SCHEMA:
        raise AssertionError("unexpected beta data retention policy schema_version")
    if retention_policy.get("status") != EXPECTED_BETA_DATA_RETENTION_STATUS:
        raise AssertionError("beta data retention policy must be validator-enforced with operator signoff required")
    if retention_policy.get("effective_for", {}).get("external_users") != 100:
        raise AssertionError("beta data retention policy must target 100 external users")
    if retention_policy.get("effective_for", {}).get("modeled_captures_per_month") != EXPECTED_MODELED_CAPTURES_PER_MONTH:
        raise AssertionError("beta data retention policy must use the beta capture model")
    policy_sources = retention_policy.get("source_of_truth", {})
    if policy_sources.get("capacity_model") != "docs/beta_capacity_cost_storage_model_2026-07-08.json":
        raise AssertionError("beta data retention policy must cite the beta capacity model")
    if policy_sources.get("gcs_lifecycle_policy") != "deploy/storage/primary-capture-bucket-lifecycle.json":
        raise AssertionError("beta data retention policy must cite the GCS lifecycle policy")
    if policy_sources.get("local_output_retention_runbook") != "docs/runbooks/output-artifact-retention.md":
        raise AssertionError("beta data retention policy must cite the local output retention runbook")
    policy_data_classes = retention_policy.get("data_classes", {})
    for class_name, expected in EXPECTED_STORAGE_DATA_CLASSES.items():
        observed = policy_data_classes.get(class_name)
        if not isinstance(observed, dict):
            raise AssertionError(f"beta data retention policy missing data class {class_name}")
        for key, value in expected.items():
            if observed.get(key) != value:
                raise AssertionError(f"beta data retention policy {class_name}.{key} mismatch")
    local_output = policy_data_classes.get("local_output_snapshots", {})
    if local_output.get("retention_tool") != "scripts/manage_output_artifact_retention.py":
        raise AssertionError("beta data retention policy must name the local output retention tool")
    if local_output.get("canonical_launch_evidence_days") != 365:
        raise AssertionError("beta data retention policy must keep canonical launch evidence for 365 days")
    if local_output.get("provider_runtime_or_paid_run_days") != 30:
        raise AssertionError("beta data retention policy must keep provider/runtime local snapshots for 30 days")
    if local_output.get("local_preflight_or_dry_run_days") != 14:
        raise AssertionError("beta data retention policy must keep local preflight snapshots for 14 days")
    local_robot_eval = policy_data_classes.get("local_robot_eval_jobs_cache", {})
    if local_robot_eval.get("root") != "robot_eval_jobs/":
        raise AssertionError("beta data retention policy must cover repo-root robot_eval_jobs cache")
    if local_robot_eval.get("delete_after_days") != 30:
        raise AssertionError("beta data retention policy must delete local robot_eval_jobs cache after 30 days")
    if local_robot_eval.get("review_threshold_gib") != EXPECTED_LOCAL_ROBOT_EVAL_JOBS_REVIEW_GIB:
        raise AssertionError("beta data retention policy must pin local robot_eval_jobs review threshold")
    if local_robot_eval.get("hard_stop_gib") != EXPECTED_LOCAL_ROBOT_EVAL_JOBS_HARD_STOP_GIB:
        raise AssertionError("beta data retention policy must pin local robot_eval_jobs hard stop")
    support_ops = retention_policy.get("support_operations", {})
    if support_ops.get("default_support_evidence_window_days") != 90:
        raise AssertionError("beta data retention policy must pin the support evidence window")
    if support_ops.get("incident_response_runbook") != "docs/runbooks/beta-ops-incident-response.md":
        raise AssertionError("beta data retention policy must point support ops at the beta incident runbook")
    operator_signoff = retention_policy.get("operator_signoff", {})
    if operator_signoff.get("required_evidence_id") != EXPECTED_OPERATOR_DPA_EVIDENCE_ID:
        raise AssertionError("beta data retention policy must require the operator DPA evidence id")
    if operator_signoff.get("status") != "manual_signoff_required":
        raise AssertionError("beta data retention policy must keep operator signoff manual-required")
    verification = retention_policy.get("verification", {})
    if verification.get("validator") != "scripts/validate_beta_capacity_storage.py":
        raise AssertionError("beta data retention policy must name the capacity/storage validator")
    if verification.get("launch_readiness_artifact_id") != "beta_data_retention_policy_json":
        raise AssertionError("beta data retention policy must name the launch readiness artifact id")
    claim_boundary = retention_policy.get("claim_boundary", {})
    if claim_boundary.get("not_signed_dpa_or_access_audit_terms") is not True:
        raise AssertionError("beta data retention policy must not claim signed DPA/access-audit proof")
    if claim_boundary.get("not_live_bucket_apply_proof") is not True:
        raise AssertionError("beta data retention policy must not claim live lifecycle apply proof")
    if claim_boundary.get("not_user_deletion_execution_proof") is not True:
        raise AssertionError("beta data retention policy must not claim user deletion execution proof")

    _require_text(
        doc_path,
        [
            "Max capture upload payload: 20 GiB",
            "Max capture duration: 45 minutes",
            "max_concurrent_jobs",
            "Cloud Tasks queue-depth alerting trips above 50 queued tasks for 5 minutes",
            "SAM3=3, VIP=2, DeepPrivacy2=2,",
            "Firestore CreatedAt Hotspot Guard",
            EXPECTED_FIRESTORE_CREATED_AT_SHARD_FIELD,
            "google_monitoring_alert_policy.firestore_request_latency",
            EXPECTED_FIRESTORE_LATENCY_METRIC,
            "firestore_latency_observation",
            "--require-firestore-latency",
            EXPECTED_LARGE_VIDEO_INGEST_TOPIC,
            "Cost Per Capture",
            "blueprint.beta_cost_per_capture_model.v1",
            "p50 planning estimate",
            "gpu_fleet_budget_guard.v1",
            "blueprint.paid_spend_admission_lock.v1",
            "google_billing_budget.gpu_fleet_beta",
            "billing_account_id",
            "Per-data-class policy",
            EXPECTED_BETA_DATA_RETENTION_POLICY_FILE,
            "beta_data_retention_policy_json",
            "arena_delivery_retention_policy.v1",
            "primary_capture_bucket_lifecycle_apply_proof_missing",
            "scripts/apply_primary_capture_bucket_lifecycle.sh",
            "Local `robot_eval_jobs/` Cache",
            "Planned robot-eval jobs/month",
            "Local review threshold",
            "python scripts/manage_output_artifact_retention.py \\",
            "--output-root robot_eval_jobs",
            "not launch proof",
            "scripts/run_beta_intake_soak_test.py --dry-run",
            "--duration-seconds 900",
            "--concurrency 25",
        ],
    )

    _require_text(
        spend_admission_doc_path,
        [
            "`$5,000`",
            "`$5,000.00` stops new paid work",
            "blueprint.provider_billing_export.v1",
            "maximum override validity interval is four hours",
            "page_event.delivery_status",
            "API-confirmed teardown proof",
        ],
    )
    _require_text(
        spend_guard_service_path,
        [
            "BLUEPRINT_GPU_FLEET_MAX_TOTAL_SPEND_USD=5000.0",
            "--require-billing-reconciliation",
            "--admission-lock-report",
            "blueprint_pipeline.live_pipeline_manifest_alert",
        ],
    )

    _require_text(
        retention_policy_doc_path,
        [
            "blueprint.beta_data_retention_policy.v1",
            "declared_validator_enforced_operator_signoff_required",
            "Raw capture truth",
            "Temporary processing",
            "Buyer/eval/hosted artifacts",
            "Local output snapshots",
            "Local `robot_eval_jobs/` cache",
            "operator_dpa_data_processing_terms",
            "not signed DPA",
            "not live bucket apply proof",
            "not user-deletion execution proof",
            "python scripts/validate_beta_capacity_storage.py",
        ],
    )

    _require_text(
        terraform_path,
        [
            'variable "max_concurrent_jobs"',
            "default     = 25",
            "var.max_concurrent_jobs >= 25",
            "max_concurrent_dispatches = var.max_concurrent_jobs",
            'variable "pipeline_queue_depth_alert_threshold"',
            "default     = 50",
            'variable "pipeline_queue_depth_alert_duration"',
            'default     = "300s"',
            "threshold_value = var.pipeline_queue_depth_alert_threshold",
            "duration        = var.pipeline_queue_depth_alert_duration",
            'resource "google_monitoring_alert_policy" "firestore_request_latency"',
            EXPECTED_FIRESTORE_LATENCY_METRIC,
            "| condition val() > 0.25 's'",
            "local.privacy_runner_max_instances",
            'resource "google_pubsub_topic" "large_video_ingest"',
            f'name   = "{EXPECTED_LARGE_VIDEO_INGEST_TOPIC}"',
            'variable "billing_account_id"',
            'resource "google_billing_budget" "gpu_fleet_beta"',
            '"billingbudgets.googleapis.com"',
            'units         = tostring(var.gpu_fleet_billing_budget_usd)',
        ],
    )
    # SCALE2-07: the four `captures_*` Firestore indexes targeted a collection
    # nothing writes; the real creatorCaptures composites are declared in
    # Blueprint-WebApp/firestore.indexes.json. Keep the phantoms from returning.
    _forbid_text(
        terraform_path,
        [
            'resource "google_firestore_index" "captures_status"',
            'resource "google_firestore_index" "captures_user"',
            'resource "google_firestore_index" "captures_status_created_at_shard"',
            'resource "google_firestore_index" "captures_user_created_at_shard"',
        ],
    )
    _require_text(
        terraform_vars_example_path,
        [
            "max_concurrent_jobs = 25",
            "pipeline_queue_depth_alert_threshold = 50",
            'pipeline_queue_depth_alert_duration  = "300s"',
            "billing_account_id",
            f"gpu_fleet_billing_budget_usd          = {EXPECTED_COHORT_HARD_STOP_USD}",
        ],
    )

    if capture_swift_policy is not None:
        _validate_capture_swift_constants(capture_swift_policy)

    return {
        "status": "passed",
        "model_path": str(model_path),
        "retention_policy_path": str(retention_policy_path),
        "lifecycle_path": str(lifecycle_path),
        "external_users": model["beta_target"]["external_users"],
        "modeled_captures_per_month": model["beta_target"]["modeled_captures_per_month"],
        "target_concurrent_uploaders": model["beta_target"]["target_concurrent_uploaders"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate beta capacity, cost, and storage lifecycle artifacts.")
    parser.add_argument("--repo-root", default=Path(__file__).resolve().parents[1])
    parser.add_argument("--capture-swift-policy")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    capture_swift_policy = Path(args.capture_swift_policy).resolve() if args.capture_swift_policy else None
    result = validate_files(repo_root, capture_swift_policy)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
