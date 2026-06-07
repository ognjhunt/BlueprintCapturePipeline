"""Isaac Lab-Arena result ingest and data-package support lane.

This module consumes local Arena rollout artifacts after an owner/runner has
produced them. It does not run Isaac Lab-Arena by default. External vision,
storage, Agents SDK, and Codex SDK actions are gated by both CLI flags and
environment variables, and the deterministic manifests remain the proof source.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shlex
import shutil
import subprocess
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from .agent_operator_runtime import LIVE_AGENTS_SDK_ENV, LIVE_CODEX_SDK_ENV
from .common import ensure_dir, read_json_any, utc_now_iso, write_json, write_text
from .local_capture import resolve_local_capture_context


ARENA_EVAL_SCHEDULE_SCHEMA_VERSION = "arena_eval_schedule.v1"
ARENA_RESULT_INGEST_LEDGER_SCHEMA_VERSION = "arena_result_ingest_ledger.v1"
ARENA_METRICS_SCHEMA_VERSION = "arena_eval_metrics.v1"
NORMALIZED_ATTEMPT_TRACE_SCHEMA_VERSION = "arena_normalized_attempt_trace.v1"
FAILURE_LABELS_SCHEMA_VERSION = "arena_failure_labels.v1"
POLICY_ADAPTER_MANIFEST_SCHEMA_VERSION = "arena_policy_adapter_manifest.v1"
CLIPS_MANIFEST_SCHEMA_VERSION = "arena_rollout_clips_manifest.v1"
VISION_LABELS_SCHEMA_VERSION = "arena_rollout_vision_labels.v1"
REVIEW_RESOLUTION_LEDGER_SCHEMA_VERSION = "arena_review_resolution_ledger.v1"
CUSTOMER_HANDOFF_REPORT_SCHEMA_VERSION = "arena_customer_handoff_report.v1"
DELIVERY_MANIFEST_SCHEMA_VERSION = "arena_delivery_manifest.v1"
RERUN_PLAN_SCHEMA_VERSION = "arena_rerun_plan.v1"
LIVE_OPERATOR_LEDGER_SCHEMA_VERSION = "arena_live_operator_ledger.v1"

DEFAULT_SCENARIO_COUNT = 500
DEFAULT_SHARD_SIZE = 50
DEFAULT_NUM_ENVS = 16
DEFAULT_TIMEOUT_SECONDS = 3600
DEFAULT_RETRY_BUDGET = 2

POLICY_MODALITY_ORDER = (
    "policy_api_endpoint",
    "docker_container",
    "recorded_action_trace",
    "high_level_skill_trace",
    "teleop_demo",
    "sim_controller_plugin",
)

SECRET_KEY_MARKERS = ("token", "secret", "password", "api_key", "apikey", "authorization")

CLAIM_BOUNDARY: Dict[str, Any] = {
    "artifact_purpose": "arena_eval_ingest_and_post_training_package_support",
    "repo_local_only_by_default": True,
    "real_arena_execution_not_performed_by_this_module": True,
    "live_provider_calls_performed": False,
    "remote_asset_downloads_performed": False,
    "storage_upload_performed": False,
    "agents_sdk_operator_performed": False,
    "codex_sdk_operator_performed": False,
    "vision_model_labeling_performed": False,
    "simulator_execution_proven": False,
    "robot_policy_execution_proven": False,
    "robot_readiness_proven": False,
    "physics_contact_validated": False,
    "safety_validated": False,
    "public_claim_upgrade_allowed": False,
    "proof_upgrade_requires": [
        "owner-system Arena execution logs and artifact manifests",
        "accepted policy/action traces for the exact robot-team submission",
        "human-accepted or owner-proof-backed review labels",
        "contact/physics validation logs when contact claims are made",
        "rights/privacy clearance for the exact package and delivery scope",
    ],
}


def _mapping(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _string(value: Any) -> str:
    return str(value or "").strip()


def _string_list(value: Any) -> List[str]:
    if value is None:
        values: Iterable[Any] = []
    elif isinstance(value, str):
        values = [value]
    elif isinstance(value, Iterable):
        values = value
    else:
        values = [value]
    out: List[str] = []
    seen: set[str] = set()
    for item in values:
        text = _string(item)
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return out


def _number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return _string(value).lower() in {"1", "true", "yes", "on", "passed", "success"}


def _env_truthy(name: str) -> bool:
    return _string(os.getenv(name)).lower() in {"1", "true", "yes", "on"}


def _relative_to(base_dir: Path, target: Path) -> str:
    return os.path.relpath(target.resolve(), start=base_dir.resolve()).replace("\\", "/")


def _read_optional_mapping(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    payload = read_json_any(path)
    return dict(payload) if isinstance(payload, Mapping) else {}


def _read_optional_json(path: Path) -> Any:
    if not path.is_file():
        return None
    return read_json_any(path)


def _sha_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha_payload(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return sha256(encoded.encode("utf-8")).hexdigest()


def _artifact_ref(base_dir: Path, path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {
            "path": _relative_to(base_dir, path),
            "exists": False,
            "size_bytes": 0,
            "sha256": None,
        }
    return {
        "path": _relative_to(base_dir, path),
        "exists": True,
        "size_bytes": path.stat().st_size,
        "sha256": _sha_file(path),
    }


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    ensure_dir(path.parent)
    content = "\n".join(json.dumps(dict(row), sort_keys=True) for row in rows)
    if content:
        content += "\n"
    write_text(path, content)


def _cards(payload: Mapping[str, Any]) -> List[Dict[str, Any]]:
    values = payload.get("cards")
    if isinstance(values, list):
        return [dict(item) for item in values if isinstance(item, Mapping)]
    if isinstance(payload.get("scenarios"), list):
        return [dict(item) for item in payload["scenarios"] if isinstance(item, Mapping)]
    return []


def _load_scenario_cards(pipeline_dir: Path) -> List[Dict[str, Any]]:
    payload = _read_optional_mapping(pipeline_dir / "robot_eval_dataset" / "scenario_cards.json")
    cards = _cards(payload)
    if cards:
        return cards
    return [
        {
            "scenario_id": "arena_placeholder_scenario",
            "task_id": "review_required_task",
            "robot_profile_id": "review_required_robot_profile",
            "status": "placeholder_until_robot_eval_dataset_cards_exist",
        }
    ]


def _scenario_id(card: Mapping[str, Any], index: int) -> str:
    return _string(card.get("scenario_id") or card.get("id") or f"scenario_{index + 1:03d}")


def _build_arena_schedule(
    *,
    pipeline_dir: Path,
    output_dir: Path,
    generated_at: str,
    scenario_count: int,
    shard_size: int,
    num_envs: int,
    timeout_seconds: int,
    retry_budget: int,
    cost_budget_usd: float | None,
) -> Dict[str, Any]:
    scenario_cards = _load_scenario_cards(pipeline_dir)
    total = max(1, int(scenario_count or DEFAULT_SCENARIO_COUNT))
    shard = max(1, int(shard_size or DEFAULT_SHARD_SIZE))
    runs: List[Dict[str, Any]] = []
    for index in range(total):
        card = scenario_cards[index % len(scenario_cards)]
        base_id = _scenario_id(card, index)
        shard_index = index // shard
        runs.append(
            {
                "scenario_run_id": f"{base_id}__arena_run_{index + 1:04d}",
                "scenario_id": base_id,
                "task_id": _string(card.get("task_id")) or None,
                "robot_profile_id": _string(card.get("robot_profile_id")) or None,
                "source_card_index": index % len(scenario_cards),
                "global_index": index,
                "shard_id": f"arena_shard_{shard_index + 1:04d}",
                "rerun_policy": {
                    "max_retries": retry_budget,
                    "retry_on": [
                        "failed",
                        "flaky",
                        "ambiguous",
                        "missing_artifact",
                        "timeout",
                        "review_required",
                    ],
                },
            }
        )
    shard_dir = output_dir / "arena_eval_shards"
    ensure_dir(shard_dir)
    shard_manifests: List[Dict[str, Any]] = []
    for start in range(0, len(runs), shard):
        shard_runs = runs[start : start + shard]
        shard_id = shard_runs[0]["shard_id"]
        manifest = {
            "schema_version": "arena_eval_shard_manifest.v1",
            "generated_at": generated_at,
            "shard_id": shard_id,
            "status": "planned",
            "num_envs": num_envs,
            "timeout_seconds": timeout_seconds,
            "scenario_run_count": len(shard_runs),
            "scenario_runs": shard_runs,
            "claim_boundary": dict(CLAIM_BOUNDARY),
        }
        path = shard_dir / f"{shard_id}.json"
        write_json(path, manifest)
        shard_manifests.append(
            {
                "shard_id": shard_id,
                "path": _relative_to(output_dir, path),
                "scenario_run_count": len(shard_runs),
            }
        )

    estimate_per_scenario = _number(os.getenv("BLUEPRINT_ARENA_COST_PER_SCENARIO_USD"), 0.0)
    estimated_cost = round(estimate_per_scenario * len(runs), 4)
    cost_status = (
        "within_budget"
        if cost_budget_usd is None or estimated_cost <= cost_budget_usd
        else "blocked_cost_budget_exceeded"
    )
    schedule = {
        "schema_version": ARENA_EVAL_SCHEDULE_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": "planned" if cost_status == "within_budget" else cost_status,
        "backend": "isaac_lab_arena",
        "scenario_count": len(runs),
        "base_scenario_card_count": len(scenario_cards),
        "num_envs": num_envs,
        "shard_size": shard,
        "shard_count": len(shard_manifests),
        "timeout_seconds": timeout_seconds,
        "retry_budget": retry_budget,
        "cost_budget_usd": cost_budget_usd,
        "scenario_runs": runs,
        "shards": shard_manifests,
        "deterministic_scheduler": True,
        "simulator_execution_performed": False,
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }
    retry_queue = {
        "schema_version": "arena_eval_retry_queue.v1",
        "generated_at": generated_at,
        "status": "empty_until_results_ingested",
        "retry_budget": retry_budget,
        "queued": [],
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }
    cost_ledger = {
        "schema_version": "arena_eval_cost_ledger.v1",
        "generated_at": generated_at,
        "status": cost_status,
        "scenario_count": len(runs),
        "estimated_cost_usd": estimated_cost,
        "cost_budget_usd": cost_budget_usd,
        "actual_cost_usd": 0.0,
        "spend_performed": False,
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }
    resume = {
        "schema_version": "arena_eval_resume_manifest.v1",
        "generated_at": generated_at,
        "status": "ready",
        "completed_scenario_run_ids": [],
        "pending_scenario_run_ids": [run["scenario_run_id"] for run in runs],
        "failed_scenario_run_ids": [],
        "next_shard_id": shard_manifests[0]["shard_id"] if shard_manifests else None,
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }
    write_json(output_dir / "arena_eval_schedule.json", schedule)
    write_json(output_dir / "arena_eval_retry_queue.json", retry_queue)
    write_json(output_dir / "arena_eval_cost_ledger.json", cost_ledger)
    write_json(output_dir / "arena_eval_resume_manifest.json", resume)
    return schedule


def _redact(value: Any) -> Any:
    if isinstance(value, Mapping):
        out: Dict[str, Any] = {}
        for key, child in value.items():
            key_text = str(key)
            if any(marker in key_text.lower() for marker in SECRET_KEY_MARKERS):
                out[key_text] = "<redacted>"
            else:
                out[key_text] = _redact(child)
        return out
    if isinstance(value, list):
        return [_redact(item) for item in value]
    return value


def _modality_payload(policy_package: Mapping[str, Any], modality: str) -> Dict[str, Any]:
    camel = {
        "policy_api_endpoint": "policyApiEndpoint",
        "docker_container": "dockerContainer",
        "recorded_action_trace": "recordedActionTrace",
        "high_level_skill_trace": "highLevelSkillTrace",
        "teleop_demo": "teleopDemo",
        "sim_controller_plugin": "simControllerPlugin",
    }[modality]
    return _mapping(policy_package.get(modality) or policy_package.get(camel))


def _policy_adapter_status(modality: str, payload: Mapping[str, Any]) -> tuple[str, List[str]]:
    missing: List[str] = []
    if not payload:
        return "blocked_missing_reference", [f"policy_package.{modality}"]
    if modality == "policy_api_endpoint":
        if not _string(payload.get("endpoint_url") or payload.get("endpointUrl") or payload.get("url")):
            missing.append("endpoint_url")
    elif modality == "docker_container":
        if not _string(payload.get("image_ref") or payload.get("imageRef")):
            missing.append("image_ref")
        if not _string(payload.get("digest")).startswith("sha256:"):
            missing.append("digest")
    elif modality == "recorded_action_trace":
        if not _string(payload.get("trace_manifest_uri") or payload.get("traceManifestUri")):
            missing.append("trace_manifest_uri")
    elif modality == "high_level_skill_trace":
        sequence = payload.get("ordered_skill_sequence") or payload.get("orderedSkillSequence")
        if not isinstance(sequence, list) or not sequence:
            missing.append("ordered_skill_sequence")
    elif modality == "teleop_demo":
        if not _string(payload.get("demo_artifact_uri") or payload.get("demoArtifactUri")):
            missing.append("demo_artifact_uri")
    elif modality == "sim_controller_plugin":
        if not _string(payload.get("plugin_uri") or payload.get("pluginUri")):
            missing.append("plugin_uri")
    return ("blocked_missing_fields", missing) if missing else ("launch_ready_review_required", [])


def _build_policy_adapter_manifest(
    *,
    job_request: Mapping[str, Any],
    output_dir: Path,
    generated_at: str,
) -> Dict[str, Any]:
    policy_package = _mapping(job_request.get("policy_package") or job_request.get("policyPackage"))
    adapters: Dict[str, Dict[str, Any]] = {}
    blocked_reasons: List[str] = []
    for modality in POLICY_MODALITY_ORDER:
        payload = _modality_payload(policy_package, modality)
        status, missing = _policy_adapter_status(modality, payload)
        if missing:
            blocked_reasons.extend(f"{modality}.{field}" for field in missing)
        adapters[modality] = {
            "status": status,
            "missing_inputs": missing,
            "reference": _redact(payload),
            "interface_contract": {
                "modality": modality,
                "input": "Arena observation/action contract or timestamp-aligned trace",
                "output": "Action, skill, trace, or plugin command stream",
                "proof_boundary": "reference_or_launch_template_only_until_owner_run_logs_exist",
            },
            "command_templates": _policy_command_templates(modality, payload),
            "launch_proof": {
                "status": "not_run",
                "execution_performed": False,
                "robot_policy_execution_proven": False,
            },
        }
    manifest = {
        "schema_version": POLICY_ADAPTER_MANIFEST_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": "blocked" if blocked_reasons else "ready_for_owner_launch_review",
        "backend": "isaac_lab_arena",
        "adapters": adapters,
        "blocked_reasons": blocked_reasons,
        "secrets_redacted": True,
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }
    write_json(output_dir / "policy_adapter_manifest.json", manifest)
    return manifest


def _policy_command_templates(modality: str, payload: Mapping[str, Any]) -> List[str]:
    if modality == "policy_api_endpoint":
        endpoint = _string(payload.get("endpoint_url") or payload.get("endpointUrl") or payload.get("url"))
        return [f"arena-policy-api-adapter --endpoint {endpoint or '<url>'} --timeout 30"]
    if modality == "docker_container":
        image = _string(payload.get("image_ref") or payload.get("imageRef") or "<image>")
        return [f"docker run --rm {image} blueprint-arena-policy-adapter"]
    if modality == "recorded_action_trace":
        trace = _string(payload.get("trace_manifest_uri") or payload.get("traceManifestUri") or "<trace>")
        return [f"arena-replay-action-trace --trace-manifest {trace}"]
    if modality == "high_level_skill_trace":
        return ["arena-skill-trace-adapter --skill-sequence <skills.json>"]
    if modality == "teleop_demo":
        demo = _string(payload.get("demo_artifact_uri") or payload.get("demoArtifactUri") or "<demo>")
        return [f"arena-teleop-demo-adapter --demo {demo}"]
    return ["arena-controller-plugin-adapter --plugin <plugin-uri>"]


def _candidate_json_files(results_dir: Path) -> List[Path]:
    if not results_dir.is_dir():
        return []
    ignored = {
        "review_resolutions.json",
        "accepted_failure_labels.json",
        "arena_eval_schedule.json",
    }
    files = []
    for path in sorted(results_dir.rglob("*.json")):
        if path.name in ignored:
            continue
        files.append(path)
    return files


def _extract_episode_records(results_dir: Path) -> tuple[List[Dict[str, Any]], List[str]]:
    if not results_dir.is_dir():
        return [], ["arena_results_dir_missing"]
    records: List[Dict[str, Any]] = []
    parsed_files: List[str] = []
    for path in _candidate_json_files(results_dir):
        payload = _read_optional_json(path)
        if payload is None:
            continue
        if isinstance(payload, Mapping):
            candidates: List[Any] = []
            for key in ("episodes", "attempts", "rollouts", "results"):
                value = payload.get(key)
                if isinstance(value, list):
                    candidates.extend(value)
            if _looks_like_episode(payload):
                candidates.append(payload)
            for item in candidates:
                if isinstance(item, Mapping):
                    record = dict(item)
                    record.setdefault("_source_json", str(path))
                    records.append(record)
            if candidates:
                parsed_files.append(str(path))
        elif isinstance(payload, list):
            for item in payload:
                if isinstance(item, Mapping):
                    record = dict(item)
                    record.setdefault("_source_json", str(path))
                    records.append(record)
            if payload:
                parsed_files.append(str(path))
    blockers = [] if records else ["missing_episode_records"]
    return records, blockers


def _looks_like_episode(payload: Mapping[str, Any]) -> bool:
    keys = {
        "episode_id",
        "episodeId",
        "scenario_id",
        "scenarioId",
        "scenario_run_id",
        "success",
        "status",
        "metrics",
    }
    return bool(keys.intersection(payload.keys()))


def _resolve_result_path(results_dir: Path, value: Any) -> str | None:
    text = _string(value)
    if not text:
        return None
    path = Path(text)
    if not path.is_absolute():
        path = results_dir / path
    return str(path)


def _normalize_attempts(
    *,
    records: Sequence[Mapping[str, Any]],
    results_dir: Path,
    generated_at: str,
) -> Dict[str, Any]:
    attempts: List[Dict[str, Any]] = []
    for index, record in enumerate(records):
        status = _string(record.get("status") or record.get("result") or "completed").lower()
        success_value = record.get("success")
        if success_value is None:
            success = status in {"success", "succeeded", "passed", "completed"}
        else:
            success = _boolish(success_value)
        metrics = _mapping(record.get("metrics"))
        failure_reason = _string(
            record.get("failure_reason")
            or record.get("failureReason")
            or record.get("reason")
            or record.get("error")
        )
        if not failure_reason and not success:
            failure_reason = "threshold_miss_or_failed_status"
        episode_id = _string(record.get("episode_id") or record.get("episodeId")) or f"episode_{index + 1:04d}"
        scenario_id = _string(record.get("scenario_id") or record.get("scenarioId")) or episode_id
        scenario_run_id = (
            _string(record.get("scenario_run_id") or record.get("scenarioRunId"))
            or f"{scenario_id}__arena_result_{index + 1:04d}"
        )
        attempts.append(
            {
                "attempt_id": f"arena_attempt_{index + 1:04d}",
                "episode_id": episode_id,
                "scenario_id": scenario_id,
                "scenario_run_id": scenario_run_id,
                "task_id": _string(record.get("task_id") or record.get("taskId")) or None,
                "shard_id": _string(record.get("shard_id") or record.get("shardId")) or None,
                "status": status,
                "success": bool(success),
                "failure_reason": failure_reason or None,
                "metrics": metrics,
                "start_time_seconds": _number(
                    record.get("start_time_seconds") or record.get("startTimeSeconds"), 0.0
                ),
                "end_time_seconds": _number(
                    record.get("end_time_seconds") or record.get("endTimeSeconds"), 0.0
                ),
                "video_path": _resolve_result_path(
                    results_dir,
                    record.get("video_path") or record.get("videoPath") or record.get("video"),
                ),
                "log_path": _resolve_result_path(
                    results_dir,
                    record.get("log_path") or record.get("logPath") or record.get("log"),
                ),
                "stdout_path": _resolve_result_path(results_dir, record.get("stdout_path")),
                "stderr_path": _resolve_result_path(results_dir, record.get("stderr_path")),
                "artifact_manifest_path": _resolve_result_path(
                    results_dir,
                    record.get("artifact_manifest_path") or record.get("artifactManifestPath"),
                ),
                "source_json": _string(record.get("_source_json")) or None,
                "review_required": not bool(success),
            }
        )
    status = "completed" if attempts else "blocked_missing_arena_results"
    manifest = {
        "schema_version": NORMALIZED_ATTEMPT_TRACE_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": status,
        "backend": "isaac_lab_arena",
        "attempt_count": len(attempts),
        "attempts": attempts,
        "result_ingested": bool(attempts),
        "simulator_execution_proven": False,
        "robot_policy_execution_proven": False,
        "robot_readiness_proven": False,
        "public_claim_upgrade_allowed": False,
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }
    return manifest


def _aggregate_metrics(attempt_trace: Mapping[str, Any], generated_at: str) -> Dict[str, Any]:
    attempts = [
        _mapping(item)
        for item in attempt_trace.get("attempts", []) or []
        if isinstance(item, Mapping)
    ]
    metric_values: Dict[str, List[float]] = {}
    for attempt in attempts:
        for key, value in _mapping(attempt.get("metrics")).items():
            if isinstance(value, (int, float)):
                metric_values.setdefault(key, []).append(float(value))
    metrics = {
        key: {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "mean": sum(values) / len(values),
        }
        for key, values in sorted(metric_values.items())
        if values
    }
    failures = [attempt for attempt in attempts if not bool(attempt.get("success"))]
    return {
        "schema_version": ARENA_METRICS_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": "completed" if attempts else "blocked_missing_arena_results",
        "attempt_count": len(attempts),
        "success_count": len(attempts) - len(failures),
        "failure_count": len(failures),
        "success_rate": (len(attempts) - len(failures)) / len(attempts) if attempts else 0.0,
        "metrics": metrics,
        "proof_boundary": dict(CLAIM_BOUNDARY),
    }


def _build_failure_labels(attempt_trace: Mapping[str, Any], generated_at: str) -> Dict[str, Any]:
    labels: List[Dict[str, Any]] = []
    for attempt in attempt_trace.get("attempts", []) or []:
        if not isinstance(attempt, Mapping) or bool(attempt.get("success")):
            continue
        reason = _string(attempt.get("failure_reason")) or "failed_or_ambiguous_attempt"
        categories = _failure_categories(reason, _mapping(attempt.get("metrics")))
        labels.append(
            {
                "label_id": f"label_{_string(attempt.get('attempt_id'))}",
                "attempt_id": attempt.get("attempt_id"),
                "episode_id": attempt.get("episode_id"),
                "scenario_id": attempt.get("scenario_id"),
                "scenario_run_id": attempt.get("scenario_run_id"),
                "status": "review_required",
                "failure_reason": reason,
                "failure_categories": categories,
                "threshold_miss": "threshold_miss" in categories,
                "contact_review_required": "contact" in categories,
                "occlusion_review_required": "occlusion" in categories,
                "proof_effect": "none_until_review_accepted_or_owner_proof_supplied",
            }
        )
    return {
        "schema_version": FAILURE_LABELS_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": "review_required" if labels else "no_failures_labeled",
        "label_count": len(labels),
        "labels": labels,
        "robot_readiness_proven": False,
        "public_claim_upgrade_allowed": False,
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }


def _failure_categories(reason: str, metrics: Mapping[str, Any]) -> List[str]:
    text = reason.lower()
    categories: List[str] = []
    for key in ("timeout", "collision", "contact", "occlusion", "threshold", "missing_artifact"):
        if key in text:
            categories.append("threshold_miss" if key == "threshold" else key)
    if any(_number(value, 0.0) < 0 for value in metrics.values()):
        categories.append("metric_out_of_bounds")
    return categories or ["failure_evidence"]


def _checksum_manifest(results_dir: Path, output_dir: Path, generated_at: str) -> Dict[str, Any]:
    artifacts: List[Dict[str, Any]] = []
    if results_dir.is_dir():
        for path in sorted(item for item in results_dir.rglob("*") if item.is_file()):
            artifacts.append(
                {
                    "path": _relative_to(results_dir, path),
                    "size_bytes": path.stat().st_size,
                    "sha256": _sha_file(path),
                }
            )
    manifest = {
        "schema_version": "arena_artifact_checksums.v1",
        "generated_at": generated_at,
        "status": "completed" if artifacts else "blocked_missing_arena_results",
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }
    write_json(output_dir / "arena_artifact_checksums.json", manifest)
    return manifest


def _copy_clip_source(video_path: Path, clip_path: Path) -> bool:
    if not video_path.is_file():
        return False
    ensure_dir(clip_path.parent)
    shutil.copy2(video_path, clip_path)
    return True


def _build_clips_manifest(
    *,
    attempt_trace: Mapping[str, Any],
    output_dir: Path,
    generated_at: str,
) -> Dict[str, Any]:
    clips_dir = output_dir / "clips"
    clips: List[Dict[str, Any]] = []
    for attempt in attempt_trace.get("attempts", []) or []:
        if not isinstance(attempt, Mapping):
            continue
        video_text = _string(attempt.get("video_path"))
        source = Path(video_text) if video_text else Path()
        clip_name = f"{_string(attempt.get('attempt_id')) or 'attempt'}.clip{source.suffix or '.bin'}"
        clip_path = clips_dir / clip_name
        copied = _copy_clip_source(source, clip_path) if video_text else False
        status = "degraded_full_source_copy" if copied else "blocked_missing_video"
        clips.append(
            {
                "clip_id": f"clip_{_string(attempt.get('attempt_id'))}",
                "attempt_id": attempt.get("attempt_id"),
                "scenario_id": attempt.get("scenario_id"),
                "source_video_path": video_text or None,
                "clip_path": _relative_to(output_dir, clip_path) if copied else None,
                "start_time_seconds": attempt.get("start_time_seconds"),
                "end_time_seconds": attempt.get("end_time_seconds"),
                "status": status,
                "extraction_method": (
                    "source_video_copied_without_timestamp_cut"
                    if copied
                    else "not_extracted_missing_video_or_optional_media_tool"
                ),
                "proof_boundary": "clip_artifact_only_not_contact_or_policy_proof",
            }
        )
    manifest = {
        "schema_version": CLIPS_MANIFEST_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": "completed_with_degraded_clips" if clips else "blocked_missing_attempts",
        "clip_count": len(clips),
        "clips": clips,
        "keyframes": {
            "status": "blocked_optional_dependency",
            "reason": "opencv_or_moviepy_not_required_for_default_fixture_path",
        },
        "contact_sheets": {
            "status": "blocked_optional_dependency",
            "reason": "opencv_or_moviepy_not_required_for_default_fixture_path",
        },
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }
    write_json(output_dir / "clips_manifest.json", manifest)
    return manifest


def _run_optional_command(command_text: str, timeout_seconds: int, cwd: Path) -> Dict[str, Any]:
    command = shlex.split(command_text)
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except FileNotFoundError:
        return {
            "status": "blocked",
            "reason": "missing_command_dependency",
            "command": command,
            "stdout": "",
            "stderr": "",
            "exit_code": None,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "status": "failed",
            "reason": "timeout",
            "command": command,
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or "",
            "exit_code": None,
        }
    return {
        "status": "completed" if completed.returncode == 0 else "failed",
        "reason": None if completed.returncode == 0 else f"exit_code:{completed.returncode}",
        "command": command,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "exit_code": completed.returncode,
    }


def _build_vision_labels(
    *,
    failure_labels: Mapping[str, Any],
    output_dir: Path,
    allow_vision_labeling: bool,
    vision_labeling_command: str | None,
    timeout_seconds: int,
    generated_at: str,
) -> Dict[str, Any]:
    labels: List[Dict[str, Any]] = []
    for label in failure_labels.get("labels", []) or []:
        if not isinstance(label, Mapping):
            continue
        labels.append(
            {
                "vision_label_id": f"vision_{_string(label.get('label_id'))}",
                "source_failure_label_id": label.get("label_id"),
                "attempt_id": label.get("attempt_id"),
                "status": "review_required",
                "masks": [],
                "object_state": "review_required",
                "contact": "review_required",
                "occlusion": "review_required",
                "threshold_miss": bool(label.get("threshold_miss")),
                "failure_evidence": label.get("failure_categories") or [],
                "label_source": "deterministic_fallback",
                "proof_effect": "none_until_human_review_or_owner_proof",
            }
        )
    env_allowed = _env_truthy("BLUEPRINT_ALLOW_ROLLOUT_VISION_LABELING")
    command_text = _string(vision_labeling_command or os.getenv("BLUEPRINT_ROLLOUT_VISION_LABELING_COMMAND"))
    blockers: List[str] = []
    command_result: Dict[str, Any] | None = None
    if allow_vision_labeling or command_text:
        if not env_allowed:
            blockers.append("missing_env_BLUEPRINT_ALLOW_ROLLOUT_VISION_LABELING")
        if not allow_vision_labeling:
            blockers.append("missing_cli_allow_rollout_vision_labeling")
        if not command_text:
            blockers.append("missing_vision_labeling_command")
        if not blockers:
            command_result = _run_optional_command(command_text, timeout_seconds, output_dir)
            if command_result["status"] != "completed":
                blockers.append(f"vision_labeling_command_{command_result['status']}")
    status = "completed_review_required" if not blockers else "blocked_review_required"
    if not labels:
        status = "no_failure_labels"
    manifest = {
        "schema_version": VISION_LABELS_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": status,
        "blockers": blockers,
        "label_count": len(labels),
        "labels": labels,
        "command_result": command_result,
        "vision_model_labeling_performed": bool(command_result and command_result["status"] == "completed"),
        "human_review_required": bool(labels),
        "claim_boundary": {
            **dict(CLAIM_BOUNDARY),
            "vision_model_labeling_performed": bool(
                command_result and command_result["status"] == "completed"
            ),
        },
    }
    write_json(output_dir / "rollout_vision_labels.json", manifest)
    return manifest


def _build_review_resolution(
    *,
    results_dir: Path,
    failure_labels: Mapping[str, Any],
    vision_labels: Mapping[str, Any],
    output_dir: Path,
    generated_at: str,
) -> Dict[str, Any]:
    resolution_payload = _read_optional_mapping(results_dir / "review_resolutions.json")
    raw_resolutions = resolution_payload.get("resolutions")
    if not isinstance(raw_resolutions, list):
        raw_resolutions = []
    by_label = {
        _string(item.get("label_id") or item.get("source_failure_label_id")): _mapping(item)
        for item in raw_resolutions
        if isinstance(item, Mapping)
    }
    ledger_entries: List[Dict[str, Any]] = []
    accepted: List[Dict[str, Any]] = []
    source_labels = [
        _mapping(item)
        for item in failure_labels.get("labels", []) or []
        if isinstance(item, Mapping)
    ]
    for label in source_labels:
        label_id = _string(label.get("label_id"))
        resolution = by_label.get(label_id, {})
        decision = _string(resolution.get("decision")).lower() or "pending"
        entry = {
            "label_id": label_id,
            "decision": decision,
            "reviewer": _string(resolution.get("reviewer")) or None,
            "evidence_uri": _string(resolution.get("evidence_uri") or resolution.get("evidenceUri")) or None,
            "proof_effect": "none" if decision != "accepted" else "accepted_label_only",
            "claim_upgrade_allowed": False,
        }
        ledger_entries.append(entry)
        if decision == "accepted":
            accepted_label = dict(label)
            accepted_label["review_status"] = "accepted"
            accepted_label["reviewer"] = entry["reviewer"]
            accepted.append(accepted_label)
    status = (
        "accepted_labels_ready"
        if accepted
        else "review_required"
        if source_labels or vision_labels.get("label_count")
        else "no_review_required"
    )
    ledger = {
        "schema_version": REVIEW_RESOLUTION_LEDGER_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": status,
        "resolution_source_path": (
            _relative_to(output_dir, results_dir / "review_resolutions.json")
            if (results_dir / "review_resolutions.json").is_file()
            else None
        ),
        "entry_count": len(ledger_entries),
        "entries": ledger_entries,
        "human_acceptance_required_for_claim_upgrade": True,
        "public_claim_upgrade_allowed": False,
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }
    accepted_manifest = {
        "schema_version": "arena_accepted_failure_labels.v1",
        "generated_at": generated_at,
        "status": "accepted" if accepted else "empty",
        "label_count": len(accepted),
        "labels": accepted,
        "public_claim_upgrade_allowed": False,
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }
    write_json(output_dir / "review_resolution_ledger.json", ledger)
    write_json(output_dir / "accepted_failure_labels.json", accepted_manifest)
    return ledger


def _build_prediction_outcome_artifacts(
    *,
    attempt_trace: Mapping[str, Any],
    output_dir: Path,
    generated_at: str,
) -> None:
    attempts = [
        _mapping(item)
        for item in attempt_trace.get("attempts", []) or []
        if isinstance(item, Mapping)
    ]
    records = [
        {
            "scenario_id": attempt.get("scenario_id"),
            "scenario_run_id": attempt.get("scenario_run_id"),
            "predicted_status": "needs_actual_outcome",
            "actual_status": "passed" if attempt.get("success") else "failed",
            "calibration_delta": None,
            "source": "arena_result_ingest",
        }
        for attempt in attempts
    ]
    failures = [attempt for attempt in attempts if not bool(attempt.get("success"))]
    write_json(
        output_dir / "prediction_outcome_ledger.json",
        {
            "schema_version": "arena_prediction_outcome_ledger.v1",
            "generated_at": generated_at,
            "status": "completed" if attempts else "not_available",
            "record_count": len(records),
            "records": records,
            "claim_boundary": dict(CLAIM_BOUNDARY),
        },
    )
    write_json(
        output_dir / "calibration_report.json",
        {
            "schema_version": "arena_calibration_report.v1",
            "generated_at": generated_at,
            "status": "review_required" if attempts else "not_available",
            "record_count": len(records),
            "records": records,
            "public_claim_upgrade_allowed": False,
            "claim_boundary": dict(CLAIM_BOUNDARY),
        },
    )
    write_json(
        output_dir / "breakage_library.json",
        {
            "schema_version": "arena_breakage_library.v1",
            "generated_at": generated_at,
            "status": "review_required" if failures else "no_breakages_recorded",
            "record_count": len(failures),
            "records": [
                {
                    "scenario_id": attempt.get("scenario_id"),
                    "scenario_run_id": attempt.get("scenario_run_id"),
                    "failure_reason": attempt.get("failure_reason"),
                    "review_required": True,
                }
                for attempt in failures
            ],
            "claim_boundary": dict(CLAIM_BOUNDARY),
        },
    )


def _build_rerun_plan(
    *,
    attempt_trace: Mapping[str, Any],
    review_ledger: Mapping[str, Any],
    output_dir: Path,
    generated_at: str,
    retry_budget: int,
    cost_budget_usd: float | None,
) -> Dict[str, Any]:
    review_required_ids = {
        _string(entry.get("label_id"))
        for entry in review_ledger.get("entries", []) or []
        if isinstance(entry, Mapping) and _string(entry.get("decision")) in {"pending", ""}
    }
    queue: List[Dict[str, Any]] = []
    lineage: List[Dict[str, Any]] = []
    for attempt in attempt_trace.get("attempts", []) or []:
        if not isinstance(attempt, Mapping):
            continue
        reasons: List[str] = []
        status = _string(attempt.get("status")).lower()
        if status in {"timeout", "timed_out"}:
            reasons.append("timeout")
        if not bool(attempt.get("success")):
            reasons.append("failed")
        if not _string(attempt.get("video_path")):
            reasons.append("missing_artifact")
        if any(_string(label_id).endswith(_string(attempt.get("attempt_id"))) for label_id in review_required_ids):
            reasons.append("review_required")
        if reasons:
            queue.append(
                {
                    "scenario_run_id": attempt.get("scenario_run_id"),
                    "scenario_id": attempt.get("scenario_id"),
                    "attempt_id": attempt.get("attempt_id"),
                    "eligible": len(queue) < retry_budget,
                    "rerun_reasons": sorted(set(reasons)),
                    "max_additional_retries": max(0, retry_budget),
                }
            )
            lineage.append(
                {
                    "source_attempt_id": attempt.get("attempt_id"),
                    "scenario_run_id": attempt.get("scenario_run_id"),
                    "rerun_reasons": sorted(set(reasons)),
                    "lineage_status": "queued" if len(queue) <= retry_budget else "budget_exhausted",
                }
            )
    status = "reruns_queued" if any(item["eligible"] for item in queue) else "no_eligible_reruns"
    if cost_budget_usd is not None and cost_budget_usd <= 0 and queue:
        status = "blocked_cost_budget_exhausted"
        for item in queue:
            item["eligible"] = False
    plan = {
        "schema_version": RERUN_PLAN_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": status,
        "retry_budget": retry_budget,
        "cost_budget_usd": cost_budget_usd,
        "queued_count": len(queue),
        "eligible_count": sum(1 for item in queue if item["eligible"]),
        "queue": queue,
        "stop_conditions": {
            "retry_budget_exhausted": sum(1 for item in queue if item["eligible"]) >= retry_budget,
            "cost_budget_exhausted": status == "blocked_cost_budget_exhausted",
        },
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }
    write_json(output_dir / "arena_rerun_plan.json", plan)
    write_json(
        output_dir / "arena_rerun_lineage.json",
        {
            "schema_version": "arena_rerun_lineage.v1",
            "generated_at": generated_at,
            "status": "completed",
            "lineage": lineage,
            "claim_boundary": dict(CLAIM_BOUNDARY),
        },
    )
    write_json(
        output_dir / "arena_eval_retry_queue.json",
        {
            "schema_version": "arena_eval_retry_queue.v1",
            "generated_at": generated_at,
            "status": status,
            "queued": queue,
            "claim_boundary": dict(CLAIM_BOUNDARY),
        },
    )
    return plan


def _build_customer_handoff_report(
    *,
    output_dir: Path,
    attempt_trace: Mapping[str, Any],
    metrics: Mapping[str, Any],
    failure_labels: Mapping[str, Any],
    clips: Mapping[str, Any],
    review_ledger: Mapping[str, Any],
    rerun_plan: Mapping[str, Any],
    generated_at: str,
) -> Dict[str, Any]:
    report = {
        "schema_version": CUSTOMER_HANDOFF_REPORT_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": "ready_review_required",
        "buyer_summary": {
            "backend": "isaac_lab_arena",
            "attempt_count": attempt_trace.get("attempt_count", 0),
            "success_rate": metrics.get("success_rate", 0.0),
            "failure_label_count": failure_labels.get("label_count", 0),
            "clip_count": clips.get("clip_count", 0),
            "review_status": review_ledger.get("status"),
            "rerun_status": rerun_plan.get("status"),
        },
        "known_limits": [
            "Arena artifacts are ingested from local result files; this module did not run Isaac Lab-Arena.",
            "Failure and vision labels remain review-required unless accepted by a human or owner proof.",
            "Robot readiness, safety, and contact validation remain false.",
        ],
        "package_inventory": {
            "normalized_attempt_trace": "normalized_attempt_trace.json",
            "failure_labels": "failure_labels.json",
            "clips_manifest": "clips_manifest.json",
            "metrics": "arena_eval_metrics.json",
            "review_resolution_ledger": "review_resolution_ledger.json",
            "rerun_plan": "arena_rerun_plan.json",
        },
        "export_instructions": [
            "Review accepted_failure_labels.json before using labels for training.",
            "Use post_training_data_package_export_manifest.json for package checksums and archive paths.",
            "Use delivery_manifest.json for local bundle or gated signed-access status.",
        ],
        "proof_boundary": dict(CLAIM_BOUNDARY),
    }
    write_json(output_dir / "customer_handoff_report.json", report)
    markdown = _customer_report_markdown(report)
    write_text(output_dir / "customer_handoff_report.md", markdown)
    return report


def _customer_report_markdown(report: Mapping[str, Any]) -> str:
    summary = _mapping(report.get("buyer_summary"))
    limits = "\n".join(f"- {item}" for item in report.get("known_limits", []) or [])
    inventory = "\n".join(
        f"- `{key}`: `{value}`"
        for key, value in _mapping(report.get("package_inventory")).items()
    )
    return (
        "# Arena Evaluation Handoff Report\n\n"
        f"- Backend: `{summary.get('backend')}`\n"
        f"- Attempts: `{summary.get('attempt_count')}`\n"
        f"- Success rate: `{summary.get('success_rate')}`\n"
        f"- Failure labels: `{summary.get('failure_label_count')}`\n"
        f"- Clips: `{summary.get('clip_count')}`\n"
        f"- Review status: `{summary.get('review_status')}`\n"
        f"- Rerun status: `{summary.get('rerun_status')}`\n\n"
        "## Known Limits\n\n"
        f"{limits}\n\n"
        "## Package Inventory\n\n"
        f"{inventory}\n\n"
        "## Proof Boundary\n\n"
        "This report does not prove robot readiness, safety validation, policy success, "
        "or public claim upgrade eligibility.\n"
    )


def _build_delivery_artifacts(
    *,
    output_dir: Path,
    allow_delivery_upload: bool,
    delivery_command: str | None,
    timeout_seconds: int,
    generated_at: str,
) -> Dict[str, Any]:
    bundle_dir = output_dir / "delivery_bundle"
    ensure_dir(bundle_dir)
    bundle_files: List[Dict[str, Any]] = []
    for name in (
        "customer_handoff_report.md",
        "customer_handoff_report.json",
        "post_training_data_package_export_manifest.json",
        "package_index.json",
        "archive_manifest.json",
    ):
        source = output_dir / name
        if source.is_file():
            target = bundle_dir / name
            shutil.copy2(source, target)
            bundle_files.append(_artifact_ref(output_dir, target))
    entitlement = {
        "schema_version": "arena_delivery_entitlement_check.v1",
        "generated_at": generated_at,
        "status": "review_required",
        "entitlement_verified": False,
        "reason": "local_bundle_created_but_customer_entitlement_requires_owner_system_check",
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }
    retention = {
        "schema_version": "arena_delivery_retention_policy.v1",
        "generated_at": generated_at,
        "status": "draft_review_required",
        "default_retention_days": 30,
        "requires_contract_confirmation": True,
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }
    total_bytes = sum(item.get("size_bytes", 0) or 0 for item in bundle_files)
    egress = {
        "schema_version": "arena_delivery_egress_estimate.v1",
        "generated_at": generated_at,
        "status": "estimated",
        "bytes": total_bytes,
        "estimated_gib": round(total_bytes / (1024**3), 6),
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }
    command_text = _string(delivery_command or os.getenv("BLUEPRINT_PACKAGE_DELIVERY_UPLOAD_COMMAND"))
    upload_result = None
    blockers: List[str] = []
    if allow_delivery_upload or command_text:
        if not _env_truthy("BLUEPRINT_ALLOW_PACKAGE_DELIVERY_UPLOAD"):
            blockers.append("missing_env_BLUEPRINT_ALLOW_PACKAGE_DELIVERY_UPLOAD")
        if not allow_delivery_upload:
            blockers.append("missing_cli_allow_delivery_upload")
        if not command_text:
            blockers.append("missing_delivery_upload_command")
        if not blockers:
            upload_result = _run_optional_command(command_text, timeout_seconds, output_dir)
            if upload_result["status"] != "completed":
                blockers.append(f"delivery_upload_{upload_result['status']}")
    signed_access = {
        "schema_version": "arena_signed_access_manifest.v1",
        "generated_at": generated_at,
        "status": "blocked" if blockers or not upload_result else "signed_access_ready",
        "blockers": blockers or ["upload_not_requested"],
        "signed_urls": [],
        "upload_result": upload_result,
        "storage_upload_performed": bool(upload_result and upload_result["status"] == "completed"),
        "claim_boundary": {
            **dict(CLAIM_BOUNDARY),
            "storage_upload_performed": bool(upload_result and upload_result["status"] == "completed"),
        },
    }
    delivery = {
        "schema_version": DELIVERY_MANIFEST_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": "local_delivery_bundle_ready",
        "local_bundle_dir": _relative_to(output_dir, bundle_dir),
        "bundle_files": bundle_files,
        "entitlement_check": "entitlement_check.json",
        "retention_policy": "retention_policy.json",
        "egress_estimate": "egress_estimate.json",
        "signed_access_manifest": "signed_access_manifest.json",
        "storage_upload_performed": signed_access["storage_upload_performed"],
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }
    write_json(output_dir / "entitlement_check.json", entitlement)
    write_json(output_dir / "retention_policy.json", retention)
    write_json(output_dir / "egress_estimate.json", egress)
    write_json(output_dir / "signed_access_manifest.json", signed_access)
    write_json(output_dir / "delivery_manifest.json", delivery)
    return delivery


def _module_available(candidates: Sequence[str]) -> str | None:
    return next((candidate for candidate in candidates if importlib.util.find_spec(candidate)), None)


def _build_live_operator_ledger(
    *,
    output_dir: Path,
    rerun_plan: Mapping[str, Any],
    allow_live_agents_sdk: bool,
    allow_live_codex_sdk: bool,
    operator_mode: str,
    timeout_seconds: int,
    generated_at: str,
) -> Dict[str, Any]:
    decisions: List[Dict[str, Any]] = []
    blockers: List[str] = []
    agents_performed = False
    codex_performed = False
    if operator_mode == "fake":
        if not _env_truthy("BLUEPRINT_ALLOW_FAKE_LIVE_OPERATORS"):
            blockers.append("missing_env_BLUEPRINT_ALLOW_FAKE_LIVE_OPERATORS")
        else:
            decisions.append(
                {
                    "operator": "fake_agents_sdk_eval_director",
                    "decision": "queue_eligible_reruns",
                    "command_chosen": "blueprint-ingest-arena-results --resume-from arena_rerun_plan.json",
                    "tool_call_summary": {
                        "rerun_status": rerun_plan.get("status"),
                        "eligible_count": rerun_plan.get("eligible_count", 0),
                    },
                    "proof_effect": "none",
                }
            )
            decisions.append(
                {
                    "operator": "fake_codex_sdk_code_maintainer",
                    "decision": "no_code_patch_required",
                    "command_chosen": "pytest focused arena/result package tests",
                    "tool_call_summary": {"code_changes_performed": False},
                    "proof_effect": "none",
                }
            )
            agents_performed = True
            codex_performed = True
    elif operator_mode == "agents-sdk":
        agents_blockers = _live_agents_blockers(allow_live_agents_sdk)
        if agents_blockers:
            blockers.extend(agents_blockers)
        else:
            decisions.append(_run_agents_sdk_operator(output_dir, timeout_seconds))
            agents_performed = True
        codex_blockers = _live_codex_blockers(allow_live_codex_sdk)
        if codex_blockers:
            blockers.extend(codex_blockers)
        else:
            decisions.append(_run_codex_sdk_operator(output_dir, timeout_seconds))
            codex_performed = True
    else:
        blockers.append("operator_mode_not_requested")
    status = "completed" if decisions and not blockers else "blocked" if blockers else "not_requested"
    ledger = {
        "schema_version": LIVE_OPERATOR_LEDGER_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": status,
        "operator_mode": operator_mode,
        "blockers": blockers,
        "decisions": decisions,
        "refusals": [
            {
                "operator": "all",
                "refusal": "proof_boolean_upgrade_without_deterministic_artifact",
                "proof_effect": "none",
            }
        ],
        "agents_sdk_operator_performed": agents_performed,
        "codex_sdk_operator_performed": codex_performed,
        "live_provider_calls_performed": operator_mode == "agents-sdk" and agents_performed,
        "public_claim_upgrade_allowed": False,
        "claim_boundary": {
            **dict(CLAIM_BOUNDARY),
            "agents_sdk_operator_performed": agents_performed,
            "codex_sdk_operator_performed": codex_performed,
            "live_provider_calls_performed": operator_mode == "agents-sdk" and agents_performed,
        },
    }
    write_json(output_dir / "live_operator_ledger.json", ledger)
    return ledger


def _live_agents_blockers(allow_live_agents_sdk: bool) -> List[str]:
    blockers: List[str] = []
    if not _env_truthy(LIVE_AGENTS_SDK_ENV):
        blockers.append(f"missing_env_{LIVE_AGENTS_SDK_ENV}")
    if not allow_live_agents_sdk:
        blockers.append("missing_cli_allow_live_agents_sdk")
    if not _string(os.getenv("OPENAI_API_KEY")):
        blockers.append("missing_openai_api_key")
    if not _module_available(("agents", "openai_agents")):
        blockers.append("missing_openai_agents_sdk")
    return blockers


def _live_codex_blockers(allow_live_codex_sdk: bool) -> List[str]:
    blockers: List[str] = []
    if not _env_truthy(LIVE_CODEX_SDK_ENV):
        blockers.append(f"missing_env_{LIVE_CODEX_SDK_ENV}")
    if not allow_live_codex_sdk:
        blockers.append("missing_cli_allow_live_codex_sdk")
    if not _module_available(("openai_codex",)):
        blockers.append("missing_openai_codex_sdk")
    return blockers


def _run_agents_sdk_operator(output_dir: Path, timeout_seconds: int) -> Dict[str, Any]:
    del timeout_seconds
    try:
        import asyncio
        from agents import Agent, Runner

        async def _run() -> Any:
            agent = Agent(
                name="Blueprint Arena Eval Director",
                instructions=(
                    "Inspect the listed deterministic manifests and return concise next actions. "
                    "Do not claim proof booleans are true."
                ),
            )
            return await Runner.run(
                agent,
                (
                    "Review arena_rerun_plan.json, review_resolution_ledger.json, and "
                    "delivery_manifest.json. Return only next deterministic commands and blockers."
                ),
            )

        result = asyncio.run(_run())
        final_output = _string(getattr(result, "final_output", ""))
        return {
            "operator": "agents_sdk_eval_director",
            "decision": "live_agent_completed",
            "command_chosen": "inspect_manifests_and_route_next_action",
            "tool_call_summary": {"final_output": final_output[:2000]},
            "proof_effect": "none",
        }
    except Exception as exc:  # pragma: no cover - gated external path
        return {
            "operator": "agents_sdk_eval_director",
            "decision": "live_agent_failed",
            "command_chosen": None,
            "tool_call_summary": {"error": f"{type(exc).__name__}: {exc}"},
            "proof_effect": "none",
            "workspace": str(output_dir),
        }


def _run_codex_sdk_operator(output_dir: Path, timeout_seconds: int) -> Dict[str, Any]:
    del timeout_seconds
    try:
        from openai_codex import Codex, Sandbox

        with Codex() as codex:
            thread = codex.thread_start(model="gpt-5.4", sandbox=Sandbox.workspace_write)
            result = thread.run(
                "Review the Arena package artifacts for code-maintenance issues. "
                "Do not mutate proof booleans."
            )
        return {
            "operator": "codex_sdk_code_maintainer",
            "decision": "live_codex_completed",
            "command_chosen": "diagnose_code_maintenance_issues",
            "tool_call_summary": {"final_response": _string(result.final_response)[:2000]},
            "proof_effect": "none",
            "workspace": str(output_dir),
        }
    except Exception as exc:  # pragma: no cover - gated external path
        return {
            "operator": "codex_sdk_code_maintainer",
            "decision": "live_codex_failed",
            "command_chosen": None,
            "tool_call_summary": {"error": f"{type(exc).__name__}: {exc}"},
            "proof_effect": "none",
            "workspace": str(output_dir),
        }


def _build_ingest_ledger(
    *,
    output_dir: Path,
    results_dir: Path,
    records: Sequence[Mapping[str, Any]],
    parsed_blockers: Sequence[str],
    checksum_manifest: Mapping[str, Any],
    generated_at: str,
) -> Dict[str, Any]:
    stdout_paths = sorted(path for path in results_dir.rglob("stdout*") if path.is_file()) if results_dir.is_dir() else []
    stderr_paths = sorted(path for path in results_dir.rglob("stderr*") if path.is_file()) if results_dir.is_dir() else []
    ledger = {
        "schema_version": ARENA_RESULT_INGEST_LEDGER_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": "completed" if records else "blocked_missing_arena_results",
        "results_dir": str(results_dir),
        "parsed_episode_count": len(records),
        "blockers": list(parsed_blockers),
        "stdout_artifacts": [_artifact_ref(output_dir, path) for path in stdout_paths],
        "stderr_artifacts": [_artifact_ref(output_dir, path) for path in stderr_paths],
        "artifact_checksum_manifest": "arena_artifact_checksums.json",
        "artifact_count": checksum_manifest.get("artifact_count", 0),
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }
    write_json(output_dir / "arena_result_ingest_ledger.json", ledger)
    return ledger


def _write_result_artifacts(
    *,
    output_dir: Path,
    attempt_trace: Mapping[str, Any],
    failure_labels: Mapping[str, Any],
    metrics: Mapping[str, Any],
) -> None:
    write_json(output_dir / "normalized_attempt_trace.json", attempt_trace)
    write_json(output_dir / "failure_labels.json", failure_labels)
    write_json(output_dir / "arena_eval_metrics.json", metrics)
    _write_jsonl(
        output_dir / "normalized_attempt_trace.jsonl",
        [
            _mapping(item)
            for item in attempt_trace.get("attempts", []) or []
            if isinstance(item, Mapping)
        ],
    )
    _write_jsonl(
        output_dir / "failure_labels.jsonl",
        [
            _mapping(item)
            for item in failure_labels.get("labels", []) or []
            if isinstance(item, Mapping)
        ],
    )


def build_arena_result_ingest(
    *,
    capture_root: str | Path,
    job_dir: str | Path | None = None,
    arena_results_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    job_request: Mapping[str, Any] | None = None,
    scenario_count: int = DEFAULT_SCENARIO_COUNT,
    shard_size: int = DEFAULT_SHARD_SIZE,
    num_envs: int = DEFAULT_NUM_ENVS,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    retry_budget: int = DEFAULT_RETRY_BUDGET,
    cost_budget_usd: float | None = None,
    allow_rollout_vision_labeling: bool = False,
    vision_labeling_command: str | None = None,
    allow_delivery_upload: bool = False,
    delivery_command: str | None = None,
    operator_mode: str = "none",
    allow_live_agents_sdk: bool = False,
    allow_live_codex_sdk: bool = False,
) -> Dict[str, Any]:
    context = resolve_local_capture_context(capture_root)
    pipeline_dir = context.pipeline_root
    generated_at = utc_now_iso()
    resolved_job_dir = Path(job_dir).resolve() if job_dir else None
    resolved_output_dir = (
        Path(output_dir).resolve()
        if output_dir
        else resolved_job_dir
        if resolved_job_dir
        else pipeline_dir / "arena_eval_package"
    )
    ensure_dir(resolved_output_dir)
    results_dir = (
        Path(arena_results_dir).resolve()
        if arena_results_dir
        else resolved_output_dir / "arena_results"
    )
    request = dict(job_request or {})

    schedule = _build_arena_schedule(
        pipeline_dir=pipeline_dir,
        output_dir=resolved_output_dir,
        generated_at=generated_at,
        scenario_count=scenario_count,
        shard_size=shard_size,
        num_envs=num_envs,
        timeout_seconds=timeout_seconds,
        retry_budget=retry_budget,
        cost_budget_usd=cost_budget_usd,
    )
    policy_manifest = _build_policy_adapter_manifest(
        job_request=request,
        output_dir=resolved_output_dir,
        generated_at=generated_at,
    )
    records, parsed_blockers = _extract_episode_records(results_dir)
    attempt_trace = _normalize_attempts(
        records=records,
        results_dir=results_dir,
        generated_at=generated_at,
    )
    metrics = _aggregate_metrics(attempt_trace, generated_at)
    failure_labels = _build_failure_labels(attempt_trace, generated_at)
    _write_result_artifacts(
        output_dir=resolved_output_dir,
        attempt_trace=attempt_trace,
        failure_labels=failure_labels,
        metrics=metrics,
    )
    checksums = _checksum_manifest(results_dir, resolved_output_dir, generated_at)
    ingest_ledger = _build_ingest_ledger(
        output_dir=resolved_output_dir,
        results_dir=results_dir,
        records=records,
        parsed_blockers=parsed_blockers,
        checksum_manifest=checksums,
        generated_at=generated_at,
    )
    clips = _build_clips_manifest(
        attempt_trace=attempt_trace,
        output_dir=resolved_output_dir,
        generated_at=generated_at,
    )
    vision_labels = _build_vision_labels(
        failure_labels=failure_labels,
        output_dir=resolved_output_dir,
        allow_vision_labeling=allow_rollout_vision_labeling,
        vision_labeling_command=vision_labeling_command,
        timeout_seconds=timeout_seconds,
        generated_at=generated_at,
    )
    review_ledger = _build_review_resolution(
        results_dir=results_dir,
        failure_labels=failure_labels,
        vision_labels=vision_labels,
        output_dir=resolved_output_dir,
        generated_at=generated_at,
    )
    _build_prediction_outcome_artifacts(
        attempt_trace=attempt_trace,
        output_dir=resolved_output_dir,
        generated_at=generated_at,
    )
    rerun_plan = _build_rerun_plan(
        attempt_trace=attempt_trace,
        review_ledger=review_ledger,
        output_dir=resolved_output_dir,
        generated_at=generated_at,
        retry_budget=retry_budget,
        cost_budget_usd=cost_budget_usd,
    )
    handoff_report = _build_customer_handoff_report(
        output_dir=resolved_output_dir,
        attempt_trace=attempt_trace,
        metrics=metrics,
        failure_labels=failure_labels,
        clips=clips,
        review_ledger=review_ledger,
        rerun_plan=rerun_plan,
        generated_at=generated_at,
    )
    from .post_training_data_package import build_post_training_data_package_export

    package_export = build_post_training_data_package_export(
        capture_root=context.capture_root,
        job_dir=resolved_output_dir,
        output_dir=resolved_output_dir,
    )
    delivery = _build_delivery_artifacts(
        output_dir=resolved_output_dir,
        allow_delivery_upload=allow_delivery_upload,
        delivery_command=delivery_command,
        timeout_seconds=timeout_seconds,
        generated_at=generated_at,
    )
    operators = _build_live_operator_ledger(
        output_dir=resolved_output_dir,
        rerun_plan=rerun_plan,
        allow_live_agents_sdk=allow_live_agents_sdk,
        allow_live_codex_sdk=allow_live_codex_sdk,
        operator_mode=operator_mode,
        timeout_seconds=timeout_seconds,
        generated_at=generated_at,
    )
    run_manifest = {
        "schema_version": "arena_result_ingest_run_manifest.v1",
        "generated_at": generated_at,
        "status": attempt_trace.get("status"),
        "capture_root": str(context.capture_root),
        "output_dir": str(resolved_output_dir),
        "arena_results_dir": str(results_dir),
        "scenario_count": schedule.get("scenario_count"),
        "attempt_count": attempt_trace.get("attempt_count"),
        "failure_label_count": failure_labels.get("label_count"),
        "clip_count": clips.get("clip_count"),
        "policy_adapter_status": policy_manifest.get("status"),
        "ingest_ledger_status": ingest_ledger.get("status"),
        "post_training_data_package_export_status": package_export.get("status"),
        "customer_handoff_status": handoff_report.get("status"),
        "delivery_status": delivery.get("status"),
        "operator_status": operators.get("status"),
        "deterministic_fingerprint": _sha_payload(
            {
                "schedule": schedule,
                "attempt_trace": attempt_trace,
                "failure_labels": failure_labels,
                "metrics": metrics,
                "rerun_plan": rerun_plan,
            }
        ),
        "simulator_execution_proven": False,
        "robot_policy_execution_proven": False,
        "robot_readiness_proven": False,
        "public_claim_upgrade_allowed": False,
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }
    write_json(resolved_output_dir / "arena_result_ingest_run_manifest.json", run_manifest)
    return {
        "schema_version": "arena_result_ingest_result.v1",
        "status": run_manifest["status"],
        "capture_root": str(context.capture_root),
        "output_dir": str(resolved_output_dir),
        "manifest_path": str((resolved_output_dir / "arena_result_ingest_run_manifest.json").resolve()),
        "attempt_trace": attempt_trace,
        "failure_labels": failure_labels,
        "metrics": metrics,
        "run_manifest": run_manifest,
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Ingest proof-bounded Isaac Lab-Arena rollout results and package support artifacts"
    )
    parser.add_argument("--capture-root", required=True)
    parser.add_argument("--job-dir")
    parser.add_argument("--arena-results-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--job-request")
    parser.add_argument("--scenario-count", type=int, default=DEFAULT_SCENARIO_COUNT)
    parser.add_argument("--shard-size", type=int, default=DEFAULT_SHARD_SIZE)
    parser.add_argument("--num-envs", type=int, default=DEFAULT_NUM_ENVS)
    parser.add_argument("--timeout-seconds", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--retry-budget", type=int, default=DEFAULT_RETRY_BUDGET)
    parser.add_argument("--cost-budget-usd", type=float, default=None)
    parser.add_argument("--allow-rollout-vision-labeling", action="store_true")
    parser.add_argument("--vision-labeling-command")
    parser.add_argument("--allow-delivery-upload", action="store_true")
    parser.add_argument("--delivery-command")
    parser.add_argument(
        "--operator-mode",
        choices=("none", "fake", "agents-sdk"),
        default="none",
        help="Fake is local-only; agents-sdk requires explicit env and CLI gates.",
    )
    parser.add_argument("--allow-live-agents-sdk", action="store_true")
    parser.add_argument("--allow-live-codex-sdk", action="store_true")
    args = parser.parse_args(argv)
    job_request = _read_optional_mapping(Path(args.job_request)) if args.job_request else {}
    result = build_arena_result_ingest(
        capture_root=args.capture_root,
        job_dir=args.job_dir,
        arena_results_dir=args.arena_results_dir,
        output_dir=args.output_dir,
        job_request=job_request,
        scenario_count=args.scenario_count,
        shard_size=args.shard_size,
        num_envs=args.num_envs,
        timeout_seconds=args.timeout_seconds,
        retry_budget=args.retry_budget,
        cost_budget_usd=args.cost_budget_usd,
        allow_rollout_vision_labeling=args.allow_rollout_vision_labeling,
        vision_labeling_command=args.vision_labeling_command,
        allow_delivery_upload=args.allow_delivery_upload,
        delivery_command=args.delivery_command,
        operator_mode=args.operator_mode,
        allow_live_agents_sdk=args.allow_live_agents_sdk,
        allow_live_codex_sdk=args.allow_live_codex_sdk,
    )
    print(f"[arena-result-ingest] manifest={result['manifest_path']}")
    print(f"[arena-result-ingest] status={result['status']}")
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
