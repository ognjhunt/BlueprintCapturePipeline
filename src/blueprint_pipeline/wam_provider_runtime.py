"""Provider-runtime support artifacts for WAM evaluation jobs."""

from __future__ import annotations

import os
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from .common import read_json_any
from .wam_eval_substrate import build_wam_eval_claim_boundary


WAM_PROVIDER_RUNTIME_PACKAGE_SCHEMA_VERSION = "wam_provider_runtime_package.v1"
WAM_PROVIDER_EXECUTION_MANIFEST_SCHEMA_VERSION = "wam_provider_execution_manifest.v1"
WAM_PROVIDER_COST_LEDGER_SCHEMA_VERSION = "wam_provider_cost_control_ledger.v1"
WAM_PROVIDER_ARTIFACT_UPLOAD_PROOF_SCHEMA_VERSION = "wam_provider_artifact_upload_proof.v1"
WAM_POLICY_INTERFACE_BINDING_SCHEMA_VERSION = "wam_policy_interface_binding.v1"
WAM_VISION_REVIEW_QUEUE_SCHEMA_VERSION = "wam_vision_success_review_queue.v1"
WAM_REAL_WORLD_ANCHOR_SCHEMA_VERSION = "wam_real_world_validation_anchor_manifest.v1"
WAM_CUSTOMER_VALIDATION_ENVELOPE_SCHEMA_VERSION = "wam_customer_validation_envelope.v1"
WAM_PRODUCTION_OPS_SCHEMA_VERSION = "wam_production_ops_manifest.v1"
WAM_CLASSICAL_SIM_CROSS_CHECK_SCHEMA_VERSION = "wam_classical_sim_cross_check_plan.v1"

WAM_PROVIDER_COMMAND_ENV_BY_SUBSTRATE = {
    "cosmos3_wam": "BLUEPRINT_COSMOS3_WAM_PROVIDER_COMMAND",
    "oscar_wam": "BLUEPRINT_OSCAR_WAM_PROVIDER_COMMAND",
}

WAM_PROVIDER_AUTH_ENV_BY_SUBSTRATE = {
    "cosmos3_wam": (
        "BLUEPRINT_COSMOS3_WAM_API_KEY",
        "COSMOS_API_KEY",
        "NVIDIA_API_KEY",
    ),
    "oscar_wam": (
        "BLUEPRINT_OSCAR_WAM_API_KEY",
        "OSCAR_WAM_API_KEY",
    ),
}


def _mapping(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _string(value: Any) -> str:
    return str(value or "").strip()


def _string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_string(item) for item in value if _string(item)]
    return []


def _number(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_id(value: Any, *, fallback: str = "item") -> str:
    text = _string(value) or fallback
    cleaned = "".join(char.lower() if char.isalnum() else "_" for char in text)
    return "_".join(part for part in cleaned.split("_") if part) or fallback


def _read_optional_mapping(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    payload = read_json_any(path)
    return dict(payload) if isinstance(payload, Mapping) else {}


def _claim_boundary(*, substrate: str, generated_at: str) -> Dict[str, Any]:
    return build_wam_eval_claim_boundary(substrate=substrate, generated_at=generated_at)


def env_truthy(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "y", "on"}


def redact(value: Any) -> Any:
    if isinstance(value, Mapping):
        redacted: Dict[str, Any] = {}
        for key, child in value.items():
            key_text = str(key)
            if any(marker in key_text.lower() for marker in ("token", "secret", "password", "key")):
                redacted[key_text] = "<redacted>"
            else:
                redacted[key_text] = redact(child)
        return redacted
    if isinstance(value, list):
        return [redact(item) for item in value]
    return value


def substrate_provider_command(substrate: str, explicit_command: str | None) -> str:
    return _string(
        explicit_command
        or os.getenv(WAM_PROVIDER_COMMAND_ENV_BY_SUBSTRATE.get(substrate, ""))
        or os.getenv("BLUEPRINT_WAM_PROVIDER_COMMAND")
    )


def provider_auth_status(substrate: str) -> Dict[str, Any]:
    env_names = WAM_PROVIDER_AUTH_ENV_BY_SUBSTRATE.get(substrate, ())
    present = [name for name in env_names if os.getenv(name)]
    return {
        "required_env_any_of": list(env_names),
        "present_env_names": present,
        "auth_available": bool(present),
        "secrets_redacted": True,
    }


def policy_interface_binding(
    *,
    job_id: str,
    substrate: str,
    request: Mapping[str, Any],
    policy_manifest: Mapping[str, Any],
    policies: Sequence[Mapping[str, Any]],
    generated_at: str,
) -> Dict[str, Any]:
    policy_package = _mapping(request.get("policy_package") or request.get("policyPackage"))
    robot = _mapping(request.get("robot") or request.get("robotProfile") or request.get("robot_profile"))
    action_interface = (
        _string(policy_package.get("action_interface") or policy_package.get("actionInterface"))
        or _string(request.get("action_interface") or request.get("actionInterface"))
        or _string(robot.get("action_interface") or robot.get("actionInterface"))
    )
    hardware_id = (
        _string(robot.get("hardware_id") or robot.get("hardwareId"))
        or _string(robot.get("robot_id") or robot.get("robotId"))
        or _string(robot.get("robot_model") or robot.get("robotModel"))
    )
    policy_rows = []
    for policy in policies:
        policy_id = _string(policy.get("policy_id")) or "policy"
        policy_rows.append(
            {
                "policy_id": policy_id,
                "checkpoint_id": _string(
                    policy.get("checkpoint_id")
                    or policy.get("checkpointId")
                    or policy.get("checkpoint")
                )
                or None,
                "source": _string(policy.get("source")) or "job_request_or_manifest",
                "capabilities": _string_list(policy.get("capabilities")),
                "reference": redact(policy),
            }
        )
    return {
        "schema_version": WAM_POLICY_INTERFACE_BINDING_SCHEMA_VERSION,
        "generated_at": generated_at,
        "job_id": job_id,
        "evaluation_substrate": substrate,
        "status": "ready" if policy_rows else "blocked_missing_policy_candidates",
        "policy_count": len(policy_rows),
        "policies": policy_rows,
        "hardware_id": hardware_id or None,
        "robot_model": _string(robot.get("robot_model") or robot.get("robotModel")) or None,
        "action_interface": action_interface or None,
        "policy_package_manifest_path": "policy_package_manifest.json",
        "policy_manifest_selected_modalities": _string_list(
            policy_manifest.get("selected_modalities")
        ),
        "requirements": {
            "exact_policy_or_checkpoint_ids_required": True,
            "action_schema_validation_required": True,
            "scenario_eval_run_id_join_required": True,
            "secrets_must_not_be_written_to_artifacts": True,
        },
        "claim_boundary": {
            "binding_is_interface_contract_not_policy_execution_proof": True,
            "robot_policy_execution_proven": False,
            "robot_readiness_proven": False,
        },
    }


def provider_runtime_package(
    *,
    capture_root: Path,
    job_dir: Path,
    job_id: str,
    substrate: str,
    request: Mapping[str, Any],
    scenario_eval_run_count: int,
    policies: Sequence[Mapping[str, Any]],
    generated_at: str,
    artifact_output_uri: str | None,
    budget_usd: float | None,
) -> Dict[str, Any]:
    return {
        "schema_version": WAM_PROVIDER_RUNTIME_PACKAGE_SCHEMA_VERSION,
        "generated_at": generated_at,
        "job_id": job_id,
        "evaluation_substrate": substrate,
        "capture_root": str(capture_root),
        "job_dir": str(job_dir),
        "inputs": {
            "job_request": "job_request.json",
            "scenario_eval_matrix": "scenario_eval_matrix.json",
            "policy_package_manifest": "policy_package_manifest.json",
            "policy_interface_binding": "wam_policy_interface_binding.json",
        },
        "output_contract": {
            "schema_version": "wam_provider_output_contract.v1",
            "accepted_top_level_rollout_keys": ["rollouts", "wam_rollout_results.rollouts"],
            "required_rollout_fields": [
                "rollout_id",
                "policy_id",
                "scenario_eval_run_id",
                "predicted_success",
                "uncertainty_score",
            ],
            "optional_fields": [
                "generated_video_uri",
                "failure_mode_ids",
                "ood_flags",
                "metrics",
                "artifact_paths",
            ],
        },
        "scenario_eval_run_count": scenario_eval_run_count,
        "policy_count": len(policies),
        "policy_ids": [_string(policy.get("policy_id")) for policy in policies],
        "artifact_output_uri": artifact_output_uri or None,
        "budget_usd": budget_usd,
        "auth_status": provider_auth_status(substrate),
        "request_summary": {
            "operation": _string(request.get("operation")) or "evaluate_only",
            "evaluation_substrate": substrate,
        },
        "claim_boundary": _claim_boundary(substrate=substrate, generated_at=generated_at),
    }


def normalize_provider_rollouts(
    *,
    payload: Any,
    substrate: str,
    generated_at: str,
) -> list[Dict[str, Any]]:
    source = _mapping(payload)
    raw = source.get("rollouts")
    if not isinstance(raw, list):
        raw = _mapping(source.get("wam_rollout_results")).get("rollouts")
    if not isinstance(raw, list):
        raw = []
    rollouts: list[Dict[str, Any]] = []
    for index, item in enumerate(raw, start=1):
        if not isinstance(item, Mapping):
            continue
        rollout = dict(item)
        policy_id = _string(rollout.get("policy_id") or rollout.get("policyId")) or "policy"
        run_id = _string(
            rollout.get("scenario_eval_run_id") or rollout.get("scenarioEvalRunId")
        ) or f"scenario_eval_run_{index:04d}"
        rollout.setdefault("rollout_id", f"wam_{_safe_id(policy_id)}_{_safe_id(run_id)}")
        rollout.setdefault("attempt_id", f"{rollout['rollout_id']}_attempt")
        rollout["policy_id"] = policy_id
        rollout["scenario_eval_run_id"] = run_id
        rollout["evaluation_substrate"] = substrate
        rollout["simulator_engine"] = substrate
        rollout.setdefault("generated_at", generated_at)
        rollout["predicted_success"] = bool(
            rollout.get("predicted_success")
            if "predicted_success" in rollout
            else rollout.get("success")
        )
        rollout["uncertainty_score"] = _number(rollout.get("uncertainty_score"), 0.35)
        rollout["failure_mode_ids"] = _string_list(rollout.get("failure_mode_ids"))
        rollout["ood_flags"] = _string_list(rollout.get("ood_flags"))
        rollout["metrics"] = {
            **_mapping(rollout.get("metrics")),
            "world_model_uncertainty": _number(rollout.get("uncertainty_score"), 0.35),
        }
        rollout["claim_boundary"] = {
            **_mapping(rollout.get("claim_boundary")),
            "model_derived_support_artifact": True,
            "raw_capture_evidence": False,
            "robot_readiness_proven": False,
            "public_claim_upgrade_allowed": False,
        }
        rollouts.append(rollout)
    return rollouts


def run_provider_command(
    *,
    command_text: str,
    runtime_package_path: Path,
    output_path: Path,
    substrate: str,
    artifact_output_uri: str | None,
    timeout_seconds: int,
) -> tuple[str, Any, Dict[str, Any]]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stdout_log = output_path.parent / "wam_provider.stdout.log"
    stderr_log = output_path.parent / "wam_provider.stderr.log"
    try:
        command = shlex.split(command_text)
    except ValueError as exc:
        stdout_log.write_text("", encoding="utf-8")
        stderr_log.write_text(str(exc), encoding="utf-8")
        return (
            "blocked",
            {},
            {
                "returncode": None,
                "duration_seconds": 0.0,
                "stdout_log": "wam_provider/wam_provider.stdout.log",
                "stderr_log": "wam_provider/wam_provider.stderr.log",
                "output_path": "wam_provider/wam_provider_output.json",
                "blockers": [f"wam_provider_command_parse_failed:{type(exc).__name__}"],
            },
        )
    env = {
        **os.environ,
        "BLUEPRINT_WAM_PROVIDER_INPUT": str(runtime_package_path),
        "BLUEPRINT_WAM_PROVIDER_OUTPUT": str(output_path),
        "BLUEPRINT_WAM_PROVIDER_SUBSTRATE": substrate,
    }
    if artifact_output_uri:
        env["BLUEPRINT_WAM_PROVIDER_ARTIFACT_OUTPUT_URI"] = artifact_output_uri
    started = time.monotonic()
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            env=env,
        )
    except subprocess.TimeoutExpired as exc:
        duration = round(time.monotonic() - started, 6)
        stdout = (
            exc.stdout.decode("utf-8", errors="replace")
            if isinstance(exc.stdout, bytes)
            else (exc.stdout or "")
        )
        stderr = (
            exc.stderr.decode("utf-8", errors="replace")
            if isinstance(exc.stderr, bytes)
            else (exc.stderr or "")
        )
        stdout_log.write_text(stdout, encoding="utf-8")
        stderr_log.write_text(stderr, encoding="utf-8")
        return (
            "blocked",
            {},
            {
                "returncode": None,
                "duration_seconds": duration,
                "stdout_log": "wam_provider/wam_provider.stdout.log",
                "stderr_log": "wam_provider/wam_provider.stderr.log",
                "output_path": "wam_provider/wam_provider_output.json",
                "blockers": ["wam_provider_command_timeout"],
            },
        )
    except OSError as exc:
        duration = round(time.monotonic() - started, 6)
        stdout_log.write_text("", encoding="utf-8")
        stderr_log.write_text(str(exc), encoding="utf-8")
        return (
            "blocked",
            {},
            {
                "returncode": None,
                "duration_seconds": duration,
                "stdout_log": "wam_provider/wam_provider.stdout.log",
                "stderr_log": "wam_provider/wam_provider.stderr.log",
                "output_path": "wam_provider/wam_provider_output.json",
                "blockers": [f"wam_provider_command_launch_failed:{type(exc).__name__}"],
            },
        )
    duration = round(time.monotonic() - started, 6)
    stdout_log.write_text(completed.stdout or "", encoding="utf-8")
    stderr_log.write_text(completed.stderr or "", encoding="utf-8")
    payload: Any = {}
    blockers: list[str] = []
    if output_path.is_file():
        try:
            payload = read_json_any(output_path)
        except Exception as exc:
            blockers.append(f"wam_provider_output_json_invalid:{type(exc).__name__}")
    detail = {
        "returncode": completed.returncode,
        "duration_seconds": duration,
        "stdout_log": "wam_provider/wam_provider.stdout.log",
        "stderr_log": "wam_provider/wam_provider.stderr.log",
        "output_path": "wam_provider/wam_provider_output.json",
    }
    if blockers:
        detail["blockers"] = blockers
    status = (
        "completed"
        if completed.returncode == 0 and output_path.is_file() and not blockers
        else "blocked"
    )
    return status, payload, detail


def vision_review_queue(
    *,
    substrate: str,
    labels: Mapping[str, Any],
    generated_at: str,
) -> Dict[str, Any]:
    label_rows = [dict(item) for item in labels.get("labels", []) or [] if isinstance(item, Mapping)]
    review_items = [
        {
            "review_item_id": f"wam_vision_review_{index:04d}",
            "label_id": label.get("label_id"),
            "rollout_id": label.get("rollout_id"),
            "attempt_id": label.get("attempt_id"),
            "scenario_eval_run_id": label.get("scenario_eval_run_id"),
            "policy_id": label.get("policy_id"),
            "reason": "low_confidence_or_ood"
            if label.get("human_review_required")
            else "spot_check",
            "confidence": label.get("confidence"),
            "uncertainty_score": label.get("uncertainty_score"),
            "ood_flags": _string_list(label.get("ood_flags")),
            "required_resolution": "human_or_live_vlm_review_accept_reject_or_relabel",
        }
        for index, label in enumerate(label_rows, start=1)
        if bool(label.get("human_review_required"))
    ]
    return {
        "schema_version": WAM_VISION_REVIEW_QUEUE_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": "review_required" if review_items else "no_review_required",
        "evaluation_substrate": substrate,
        "label_count": len(label_rows),
        "review_item_count": len(review_items),
        "confidence_threshold": 0.5,
        "live_vlm_or_human_review_performed": False,
        "review_items": review_items,
        "audit_log": [
            {
                "event": "review_queue_built",
                "source": "vision_success_labels.json",
                "live_vlm_or_human_review_performed": False,
            }
        ],
        "claim_boundary": {
            **_claim_boundary(substrate=substrate, generated_at=generated_at),
            "review_queue_is_not_review_completion": True,
        },
    }


def real_world_anchor_manifest(
    *,
    job_dir: Path,
    substrate: str,
    scorecard: Mapping[str, Any],
    generated_at: str,
) -> Dict[str, Any]:
    ledger = _read_optional_mapping(job_dir / "deployment_outcome_ledger.json")
    raw_records = ledger.get("records") or ledger.get("outcomes") or []
    records = (
        [dict(item) for item in raw_records if isinstance(item, Mapping)]
        if isinstance(raw_records, list)
        else []
    )
    usable = []
    missing = []
    for record in records:
        run_id = _string(record.get("scenario_eval_run_id") or record.get("scenarioEvalRunId"))
        policy_id = _string(record.get("policy_id") or record.get("policyId"))
        hardware_id = _string(record.get("hardware_id") or record.get("hardwareId"))
        owner_evidence = _string(
            record.get("owner_evidence_uri")
            or record.get("ownerEvidenceUri")
            or record.get("operator_attestation")
            or record.get("operatorAttestation")
        )
        missing_fields = [
            field
            for field, value in {
                "scenario_eval_run_id": run_id,
                "policy_id": policy_id,
                "hardware_id": hardware_id,
                "owner_evidence_or_operator_attestation": owner_evidence,
            }.items()
            if not value
        ]
        if missing_fields:
            missing.append(
                {
                    "record_id": _string(record.get("record_id") or record.get("id")) or None,
                    "missing_fields": missing_fields,
                }
            )
        else:
            usable.append(
                {
                    "scenario_eval_run_id": run_id,
                    "policy_id": policy_id,
                    "hardware_id": hardware_id,
                    "actual_success": bool(
                        record.get("actual_success")
                        if "actual_success" in record
                        else record.get("success")
                    ),
                    "owner_evidence_or_operator_attestation": owner_evidence,
                }
            )
    return {
        "schema_version": WAM_REAL_WORLD_ANCHOR_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": "ready_for_srcc_computation" if usable and not missing else "requires_real_world_rollout_anchors",
        "evaluation_substrate": substrate,
        "top_policy_id": scorecard.get("top_policy_id"),
        "deployment_outcome_ledger_path": "deployment_outcome_ledger.json"
        if (job_dir / "deployment_outcome_ledger.json").is_file()
        else None,
        "usable_anchor_count": len(usable),
        "missing_or_incomplete_anchor_count": len(missing),
        "anchors": usable,
        "missing_anchor_requirements": missing
        or [
            {
                "missing_fields": [
                    "paired_real_world_rollout_outcomes",
                    "scenario_eval_run_id",
                    "policy_id",
                    "hardware_id",
                    "owner_evidence_or_operator_attestation",
                ]
            }
        ],
        "customer_specific_srcc_claimed": False,
        "claim_boundary": _claim_boundary(substrate=substrate, generated_at=generated_at),
    }


def customer_validation_envelope(
    *,
    job_id: str,
    substrate: str,
    request: Mapping[str, Any],
    scorecard: Mapping[str, Any],
    anchor_manifest: Mapping[str, Any],
    generated_at: str,
) -> Dict[str, Any]:
    robot = _mapping(request.get("robot") or request.get("robotProfile") or request.get("robot_profile"))
    task = _mapping(request.get("task") or request.get("task_request") or request.get("taskRequest"))
    hardware_id = (
        _string(robot.get("hardware_id") or robot.get("hardwareId"))
        or _string(robot.get("robot_model") or robot.get("robotModel"))
    )
    task_family = (
        _string(task.get("task_family") or task.get("taskFamily"))
        or _string(task.get("task_id") or task.get("taskId"))
    )
    return {
        "schema_version": WAM_CUSTOMER_VALIDATION_ENVELOPE_SCHEMA_VERSION,
        "generated_at": generated_at,
        "job_id": job_id,
        "status": "validation_envelope_draft",
        "evaluation_substrate": substrate,
        "hardware_id": hardware_id or None,
        "task_family": task_family or None,
        "top_policy_id": scorecard.get("top_policy_id"),
        "validity_scope": {
            "site_specific": True,
            "hardware_specific": bool(hardware_id),
            "policy_specific": True,
            "task_family_specific": bool(task_family),
        },
        "srcc_claim_status": "not_claimed",
        "anchor_status": anchor_manifest.get("status"),
        "required_before_customer_correlation_claim": [
            "paired_real_world_rollouts",
            "exact_scenario_eval_run_id_join",
            "exact_policy_or_checkpoint_id_join",
            "exact_hardware_id_join",
            "owner_evidence_or_operator_attestation",
        ],
        "claim_boundary": {
            **_claim_boundary(substrate=substrate, generated_at=generated_at),
            "validation_envelope_is_not_universal_correlation_claim": True,
        },
    }


def production_ops_manifest(
    *,
    job_id: str,
    substrate: str,
    request: Mapping[str, Any],
    provider_execution: Mapping[str, Any],
    generated_at: str,
    artifact_output_uri: str | None,
    budget_usd: float | None,
) -> Dict[str, Any]:
    return {
        "schema_version": WAM_PRODUCTION_OPS_SCHEMA_VERSION,
        "generated_at": generated_at,
        "job_id": job_id,
        "evaluation_substrate": substrate,
        "status": "ready_for_local_or_owner_provider_run"
        if provider_execution.get("status") == "completed"
        else "blocked_or_fixture_only",
        "queueing": {
            "queue_contract": "robot_eval_job_request.v1",
            "job_status_projection_required": True,
            "webapp_status_projection_artifact": "webapp_robot_eval_status_projection.json",
        },
        "billing": {
            "billing_meter": "wam_policy_eval_episode",
            "scenario_count": None,
            "budget_usd": budget_usd,
            "provider_spend_limit_required": True,
        },
        "artifact_retention": {
            "generated_rollout_retention_policy_required": True,
            "raw_capture_retention_policy_reused": True,
            "artifact_output_uri": artifact_output_uri or None,
        },
        "secure_policy_handling": {
            "policy_references_redacted": True,
            "secret_values_written_to_artifacts": False,
            "customer_policy_source_access_required": False,
        },
        "provider_execution_status": provider_execution.get("status"),
        "provider_blockers": _string_list(provider_execution.get("blockers")),
        "requested_outputs": _string_list(request.get("requested_outputs") or request.get("requestedOutputs")),
        "claim_boundary": _claim_boundary(substrate=substrate, generated_at=generated_at),
    }


def classical_sim_cross_check_plan(
    *,
    job_id: str,
    substrate: str,
    request: Mapping[str, Any],
    scorecard: Mapping[str, Any],
    generated_at: str,
) -> Dict[str, Any]:
    requested = _string_list(
        request.get("classical_sim_cross_checks")
        or request.get("classicalSimCrossChecks")
        or request.get("simulator_engines")
        or request.get("simulatorEngines")
    )
    normalized = []
    for item in requested:
        text = item.lower().replace("-", "_")
        if text in {"mujoco", "classical_sim_mujoco"}:
            normalized.append("classical_sim_mujoco")
        elif text in {"isaac", "isaac_sim", "isaac_lab_arena", "classical_sim_isaac"}:
            normalized.append("classical_sim_isaac")
    if not normalized:
        normalized = ["classical_sim_mujoco"]
    return {
        "schema_version": WAM_CLASSICAL_SIM_CROSS_CHECK_SCHEMA_VERSION,
        "generated_at": generated_at,
        "job_id": job_id,
        "status": "planned_optional_cross_check",
        "primary_evaluation_substrate": substrate,
        "recommended_cross_checks": sorted(set(normalized)),
        "top_policy_id": scorecard.get("top_policy_id"),
        "purpose": [
            "stricter_physics_or_contact_sanity_check",
            "wam_vs_classical_disagreement_analysis",
            "safety_contact_review_input_not_safety_approval",
        ],
        "promotion_effect": "none_without_owner_execution_evidence_and_review",
        "claim_boundary": _claim_boundary(substrate=substrate, generated_at=generated_at),
    }


def provider_execution_manifest(
    *,
    substrate: str,
    generated_at: str,
    status: str,
    command_used: bool,
    detail: Mapping[str, Any] | None = None,
    blockers: Sequence[str] = (),
    attempt_count: int = 0,
    max_retries: int = 0,
) -> Dict[str, Any]:
    return {
        "schema_version": WAM_PROVIDER_EXECUTION_MANIFEST_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": status,
        "evaluation_substrate": substrate,
        "provider_command_used": command_used,
        "attempt_count": attempt_count,
        "max_retries": max_retries,
        "blockers": list(blockers),
        "detail": dict(detail or {}),
        "claim_boundary": {
            **_claim_boundary(substrate=substrate, generated_at=generated_at),
            "live_provider_calls_performed": status == "completed" and command_used,
        },
    }


def provider_cost_ledger(
    *,
    substrate: str,
    generated_at: str,
    budget_usd: float | None,
    status: str,
    duration_seconds: float | None = None,
) -> Dict[str, Any]:
    return {
        "schema_version": WAM_PROVIDER_COST_LEDGER_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": "within_budget" if status == "completed" else "blocked_or_not_run",
        "evaluation_substrate": substrate,
        "budget_usd": budget_usd,
        "estimated_spend_usd": None,
        "duration_seconds": duration_seconds,
        "hard_spend_limit_enforced_by_provider": False,
        "claim_boundary": _claim_boundary(substrate=substrate, generated_at=generated_at),
    }


def provider_artifact_upload_proof(
    *,
    substrate: str,
    generated_at: str,
    artifact_output_uri: str | None,
    provider_payload: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    evidence = _mapping((provider_payload or {}).get("artifact_upload_evidence"))
    complete = bool(
        evidence.get("artifact_upload_evidence_complete")
        or evidence.get("upload_complete")
        or evidence.get("provider_writable_output_verified")
    )
    return {
        "schema_version": WAM_PROVIDER_ARTIFACT_UPLOAD_PROOF_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": "upload_proven" if complete else "not_proven",
        "evaluation_substrate": substrate,
        "artifact_output_uri": artifact_output_uri or None,
        "artifact_upload_evidence_complete": complete,
        "evidence": redact(evidence),
        "claim_boundary": _claim_boundary(substrate=substrate, generated_at=generated_at),
    }
