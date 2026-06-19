"""Policy Improvement Run offer manifest builder.

This module binds Blueprint's existing Task Evaluation Run, Post-Training Data
Package, and policy-autoresearch artifacts into one customer-facing offer
contract. It is intentionally model-agnostic and source-code-optional: robot
teams can start with a black-box policy API/container, a configurable adapter
surface, or full source/training access.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from .common import ensure_dir, read_json_any, utc_now_iso, write_json, write_text
from .local_capture import resolve_local_capture_context


POLICY_IMPROVEMENT_RUN_SCHEMA_VERSION = "policy_improvement_run_offer.v2"
DEFAULT_OUTPUT_DIR_NAME = "policy_improvement_run"

ACCESS_LEVELS = {
    "black_box": {
        "label": "Black-box policy API or container",
        "source_code_required": False,
        "allowed_work": [
            "baseline evaluation",
            "failure diagnosis",
            "wrapper or action-interface tuning",
            "scenario and curriculum generation",
            "sealed regression evaluation",
        ],
        "typical_artifact": "evidence report plus wrapper/config recommendations",
    },
    "config_adapter": {
        "label": "Config, adapter, or task-head access",
        "source_code_required": False,
        "allowed_work": [
            "baseline evaluation",
            "failure diagnosis",
            "adapter or task-head post-training",
            "distilled skill candidate generation",
            "sealed regression evaluation",
        ],
        "typical_artifact": "versioned adapter, task head, or distilled skill package",
    },
    "source_training": {
        "label": "Source and training access",
        "source_code_required": True,
        "allowed_work": [
            "baseline evaluation",
            "failure diagnosis",
            "training recipe changes",
            "policy architecture or reward-development changes outside sealed scoring",
            "complete policy candidate generation",
            "sealed regression evaluation",
        ],
        "typical_artifact": "adapter, task head, distilled skill, or complete policy candidate",
    },
}

IMPROVEMENT_TARGETS = ("adapter", "task_head", "distilled_skill", "complete_policy")

CLAIM_BOUNDARY: dict[str, Any] = {
    "artifact_purpose": "policy_improvement_run_offer",
    "extends_task_evaluation_run": True,
    "extends_post_training_data_package": True,
    "model_backend_agnostic": True,
    "customer_supplied_policy_friendly": True,
    "source_code_required_by_default": False,
    "development_scenarios_may_be_used_for_training": True,
    "sealed_audit_scenarios_must_not_be_used_for_training": True,
    "sim_heldout_success_is_not_deployment_approval": True,
    "simulator_execution_proven": False,
    "robot_policy_execution_proven": False,
    "real_world_outcome_proven": False,
    "robot_readiness_proven": False,
    "safety_validation_proven": False,
    "public_claim_upgrade_allowed": False,
}


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _string(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None


def _safe_id(value: Any, *, fallback: str = "item") -> str:
    text = _string(value) or fallback
    cleaned = "".join(character.lower() if character.isalnum() else "_" for character in text)
    return "_".join(part for part in cleaned.split("_") if part) or fallback


def _read_optional_mapping(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    payload = read_json_any(path)
    return dict(payload) if isinstance(payload, Mapping) else {}


def _relative_to(base_dir: Path, target: Path) -> str:
    return os.path.relpath(target.resolve(), start=base_dir.resolve()).replace("\\", "/")


def _include_if_file(
    included: dict[str, str],
    *,
    key: str,
    base_dir: Path,
    path: Path,
) -> None:
    if path.is_file():
        included[key] = _relative_to(base_dir, path)


def _first_string(*values: Any) -> str:
    for value in values:
        text = _string(value)
        if text:
            return text
    return ""


def _split_counts(matrix: Mapping[str, Any]) -> dict[str, int]:
    counts = {
        "development": 0,
        "validation": 0,
        "heldout": 0,
        "sealed_audit": 0,
        "unknown": 0,
    }
    runs = matrix.get("runs")
    if not isinstance(runs, list):
        return counts
    for run in runs:
        if not isinstance(run, Mapping):
            counts["unknown"] += 1
            continue
        split = _safe_id(
            run.get("split")
            or run.get("scenario_split")
            or run.get("scenarioSplit")
            or run.get("eval_split")
            or run.get("evalSplit"),
            fallback="unknown",
        )
        if split in {"train", "training", "dev", "development", "autoresearch"}:
            counts["development"] += 1
        elif split in {"val", "validation"}:
            counts["validation"] += 1
        elif split in {"heldout", "holdout", "test"}:
            counts["heldout"] += 1
        elif split in {"sealed", "sealed_audit", "audit"}:
            counts["sealed_audit"] += 1
        else:
            counts["unknown"] += 1
    return counts


def _failure_mode_summary(labels: Mapping[str, Any]) -> dict[str, Any]:
    rows: list[Mapping[str, Any]] = []
    for key in ("labels", "failure_labels", "accepted_failure_labels"):
        value = labels.get(key)
        if isinstance(value, list):
            rows.extend(item for item in value if isinstance(item, Mapping))
    if not rows:
        value = labels.get("failures")
        if isinstance(value, list):
            rows.extend(item for item in value if isinstance(item, Mapping))

    counts: dict[str, int] = {}
    for row in rows:
        mode = _first_string(
            row.get("failure_mode_id"),
            row.get("failure_mode"),
            row.get("label"),
            row.get("reason"),
            "unlabeled_failure",
        )
        counts[mode] = counts.get(mode, 0) + 1
    return {
        "label_count": int(labels.get("label_count") or len(rows)),
        "dominant_failure_modes": [
            {"failure_mode": mode, "count": count}
            for mode, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        ],
    }


def _scorecard_summary(evaluation_result: Mapping[str, Any]) -> dict[str, Any]:
    scorecard = _mapping(evaluation_result.get("standard_policy_scorecard"))
    cycle_time = _mapping(scorecard.get("cycle_time"))
    return {
        "status": evaluation_result.get("status") or "missing",
        "success_rate": _float(scorecard.get("success_rate")),
        "cycle_time_mean_seconds": _float(cycle_time.get("mean_seconds")),
        "cycle_time_sample_count": cycle_time.get("sample_count"),
        "intervention_rate": _float(scorecard.get("intervention_rate")),
        "sim_vs_real_calibration_score": _float(
            scorecard.get("sim_vs_real_calibration_score")
        ),
        "required_scenario_eval_run_ids": scorecard.get("required_scenario_eval_run_ids")
        or [],
        "completed_scenario_eval_run_ids": scorecard.get("completed_scenario_eval_run_ids")
        or [],
    }


def _policy_autoresearch_summary(
    *,
    report: Mapping[str, Any],
    candidate: Mapping[str, Any],
    heldout: Mapping[str, Any],
) -> dict[str, Any]:
    baseline = _float(report.get("baseline_heldout_success_rate"))
    best = _float(report.get("best_heldout_success_rate"))
    delta = None if baseline is None or best is None else round(best - baseline, 6)
    return {
        "status": report.get("status") or candidate.get("status") or "missing",
        "candidate_status": candidate.get("status") or "missing",
        "baseline_heldout_success_rate": baseline,
        "best_heldout_success_rate": best,
        "heldout_success_rate_delta": delta,
        "target_success_reached": report.get("target_success_reached") is True,
        "safety_contact_gate_passed": heldout.get("safety_contact_gate_passed") is True,
        "frozen_verifier_sha256": (
            report.get("frozen_verifier_sha256")
            or candidate.get("frozen_verifier_sha256")
            or heldout.get("frozen_verifier_sha256")
        ),
        "promoted_artifact_kind": _first_string(
            candidate.get("promoted_artifact_kind"),
            candidate.get("artifact_kind"),
            "policy_candidate_package",
        ),
        "simulator_execution_proven": candidate.get("simulator_execution_proven") is True,
        "robot_readiness_proven": candidate.get("robot_readiness_proven") is True,
        "public_claim_upgrade_allowed": candidate.get("public_claim_upgrade_allowed") is True,
    }


def _customer_inputs(
    *,
    job_request: Mapping[str, Any],
    policy_package: Mapping[str, Any],
    customer_policy_ref: str | None,
    embodiment: str | None,
    action_interface: str | None,
    target_task: str | None,
    success_threshold: float | None,
    cycle_time_threshold_seconds: float | None,
) -> dict[str, dict[str, Any]]:
    request_policy = _mapping(job_request.get("policy_package") or job_request.get("policy"))
    request_robot = _mapping(job_request.get("robot") or job_request.get("robot_profile"))
    request_task = _mapping(job_request.get("task") or job_request.get("target_task"))
    request_thresholds = _mapping(
        job_request.get("thresholds")
        or job_request.get("success_thresholds")
        or job_request.get("task_thresholds")
    )
    package_policy_id = _first_string(
        policy_package.get("policy_id"),
        policy_package.get("policyId"),
        policy_package.get("name"),
    )
    base_policy = _first_string(
        customer_policy_ref,
        request_policy.get("policy_id"),
        request_policy.get("policy_uri"),
        request_policy.get("policy_api_endpoint"),
        package_policy_id,
    )
    robot_embodiment = _first_string(
        embodiment,
        request_robot.get("embodiment"),
        request_robot.get("robot_model"),
        request_robot.get("robot_profile_id"),
        job_request.get("robot_profile_id"),
    )
    interface = _first_string(
        action_interface,
        request_policy.get("action_interface"),
        request_policy.get("action_space"),
        policy_package.get("action_interface"),
        policy_package.get("action_space"),
    )
    task = _first_string(
        target_task,
        request_task.get("task_id"),
        request_task.get("task_statement"),
        job_request.get("task_id"),
        job_request.get("target_task"),
    )
    success = (
        success_threshold
        if success_threshold is not None
        else _float(
            request_thresholds.get("success_rate")
            or request_thresholds.get("target_success_rate")
            or job_request.get("target_success_rate")
        )
    )
    cycle_time = (
        cycle_time_threshold_seconds
        if cycle_time_threshold_seconds is not None
        else _float(
            request_thresholds.get("cycle_time_seconds")
            or request_thresholds.get("max_cycle_time_seconds")
            or job_request.get("cycle_time_threshold_seconds")
        )
    )
    return {
        "base_policy_or_model": {"status": "present" if base_policy else "missing", "value": base_policy},
        "robot_embodiment": {
            "status": "present" if robot_embodiment else "missing",
            "value": robot_embodiment,
        },
        "action_interface": {"status": "present" if interface else "missing", "value": interface},
        "target_task": {"status": "present" if task else "missing", "value": task},
        "success_threshold": {"status": "present" if success is not None else "missing", "value": success},
        "cycle_time_threshold_seconds": {
            "status": "present" if cycle_time is not None else "missing",
            "value": cycle_time,
        },
    }



def _readiness_ladder(
    *,
    customer_inputs: Mapping[str, Mapping[str, Any]],
    included_artifacts: Mapping[str, str],
    split_counts: Mapping[str, int],
    post_training_package: Mapping[str, Any],
    policy_summary: Mapping[str, Any],
) -> list[dict[str, Any]]:
    checks = [
        (
            "customer_policy_intake",
            all(value.get("status") == "present" for value in customer_inputs.values()),
            [
                f"missing_customer_input_{key}"
                for key, value in customer_inputs.items()
                if value.get("status") != "present"
            ],
        ),
        (
            "task_evaluation_run_contract",
            "scenario_eval_matrix" in included_artifacts
            and int(split_counts.get("heldout") or 0)
            + int(split_counts.get("sealed_audit") or 0)
            > 0,
            [
                blocker
                for blocker, failed in (
                    ("missing_scenario_eval_matrix", "scenario_eval_matrix" not in included_artifacts),
                    (
                        "missing_heldout_or_sealed_audit_split",
                        int(split_counts.get("heldout") or 0)
                        + int(split_counts.get("sealed_audit") or 0)
                        == 0,
                    ),
                )
                if failed
            ],
        ),
        (
            "baseline_evaluation",
            "normalized_attempt_trace" in included_artifacts
            and "evaluation_result" in included_artifacts,
            [
                blocker
                for blocker, failed in (
                    ("baseline_normalized_attempt_trace_missing", "normalized_attempt_trace" not in included_artifacts),
                    ("evaluation_result_missing", "evaluation_result" not in included_artifacts),
                )
                if failed
            ],
        ),
        (
            "failure_diagnosis",
            "failure_labels" in included_artifacts,
            [] if "failure_labels" in included_artifacts else ["failure_labels_missing"],
        ),
        (
            "post_training_data_package",
            post_training_package.get("status") == "export_ready_review_required",
            []
            if post_training_package.get("status") == "export_ready_review_required"
            else ["post_training_data_package_export_not_ready"],
        ),
        (
            "policy_autoresearch",
            "policy_autoresearch_report" in included_artifacts,
            []
            if "policy_autoresearch_report" in included_artifacts
            else ["policy_autoresearch_report_missing"],
        ),
        (
            "candidate_promotion",
            policy_summary.get("candidate_status") == "promoted_sim_only_policy_candidate"
            and policy_summary.get("safety_contact_gate_passed") is True,
            [
                blocker
                for blocker, failed in (
                    (
                        "promoted_policy_candidate_missing",
                        policy_summary.get("candidate_status")
                        != "promoted_sim_only_policy_candidate",
                    ),
                    (
                        "policy_candidate_safety_contact_gate_not_passed",
                        policy_summary.get("safety_contact_gate_passed") is not True,
                    ),
                )
                if failed
            ],
        ),
        (
            "customer_review_package",
            "policy_candidate_package" in included_artifacts
            and "heldout_eval_result" in included_artifacts,
            [
                blocker
                for blocker, failed in (
                    ("policy_candidate_package_missing", "policy_candidate_package" not in included_artifacts),
                    ("heldout_eval_result_missing", "heldout_eval_result" not in included_artifacts),
                )
                if failed
            ],
        ),
    ]
    return [
        {
            "stage": stage,
            "status": "ready" if ready else "blocked",
            "blockers": blockers,
        }
        for stage, ready, blockers in checks
    ]


def _webapp_summary_projection(manifest: Mapping[str, Any]) -> dict[str, Any]:
    policy_summary = _mapping(manifest.get("policy_autoresearch_summary"))
    baseline = _mapping(manifest.get("baseline_evaluation_summary"))
    customer_inputs = _mapping(manifest.get("customer_inputs"))
    return {
        "schema_version": "policy_improvement_run_webapp_summary.v1",
        "product_family": "policy_improvement_run",
        "scene_id": manifest.get("scene_id"),
        "capture_id": manifest.get("capture_id"),
        "status": manifest.get("status"),
        "blockers": manifest.get("blockers") or [],
        "customer_input_status": {
            key: _mapping(value).get("status") for key, value in customer_inputs.items()
        },
        "baseline_success_rate": baseline.get("success_rate"),
        "candidate_success_rate": policy_summary.get("best_heldout_success_rate"),
        "heldout_success_rate_delta": policy_summary.get("heldout_success_rate_delta"),
        "safety_contact_gate_passed": policy_summary.get("safety_contact_gate_passed"),
        "readiness_ladder": manifest.get("readiness_ladder") or [],
        "artifact_uris": {
            "manifest": manifest.get("manifest_path"),
            "brief": manifest.get("brief_path"),
        },
        "safe_for_firestore": True,
        "dense_or_secret_payloads_included": False,
        "claim_boundary": manifest.get("claim_boundary") or {},
    }

def _status_and_blockers(
    *,
    customer_inputs: Mapping[str, Mapping[str, Any]],
    included_artifacts: Mapping[str, str],
    split_counts: Mapping[str, int],
    policy_summary: Mapping[str, Any],
    post_training_package: Mapping[str, Any],
) -> tuple[str, list[str]]:
    blockers = [
        f"missing_customer_input_{key}"
        for key, value in customer_inputs.items()
        if value.get("status") != "present"
    ]
    if "scenario_eval_matrix" not in included_artifacts:
        blockers.append("missing_scenario_eval_matrix")
    if int(split_counts.get("heldout") or 0) + int(split_counts.get("sealed_audit") or 0) == 0:
        blockers.append("missing_heldout_or_sealed_audit_split")

    if blockers:
        return "blocked_missing_policy_improvement_inputs", blockers

    if "normalized_attempt_trace" not in included_artifacts:
        return "ready_for_baseline_evaluation", ["baseline_normalized_attempt_trace_missing"]
    if "failure_labels" not in included_artifacts:
        return "ready_for_failure_diagnosis", ["failure_labels_missing"]
    if post_training_package.get("status") != "export_ready_review_required":
        return "ready_for_post_training_data_package", [
            "post_training_data_package_export_not_ready"
        ]
    if "policy_autoresearch_report" not in included_artifacts:
        return "ready_for_policy_autoresearch", ["policy_autoresearch_report_missing"]
    if policy_summary.get("candidate_status") != "promoted_sim_only_policy_candidate":
        return "completed_no_promoted_candidate", ["promoted_policy_candidate_missing"]
    if policy_summary.get("safety_contact_gate_passed") is not True:
        return "blocked_candidate_failed_safety_contact_gate", [
            "policy_candidate_safety_contact_gate_not_passed"
        ]
    return "improvement_candidate_ready_for_customer_review", []


def _markdown_brief(manifest: Mapping[str, Any]) -> str:
    product = _mapping(manifest.get("product"))
    access = _mapping(manifest.get("access_model"))
    summary = _mapping(manifest.get("policy_autoresearch_summary"))
    boundary = _mapping(manifest.get("claim_boundary"))
    lines = [
        "# Policy Improvement Run",
        "",
        f"Status: `{manifest.get('status')}`",
        "",
        "## Offer",
        "",
        _string(product.get("one_line")),
        "",
        "## Access",
        "",
        f"- Level: `{access.get('access_level')}`",
        f"- Source code required: `{access.get('source_code_required')}`",
        "",
        "## Evidence",
        "",
        f"- Baseline heldout success: `{summary.get('baseline_heldout_success_rate')}`",
        f"- Best heldout success: `{summary.get('best_heldout_success_rate')}`",
        f"- Heldout delta: `{summary.get('heldout_success_rate_delta')}`",
        f"- Safety/contact gate passed: `{summary.get('safety_contact_gate_passed')}`",
        "",
        "## Boundary",
        "",
        f"- Sim heldout success is deployment approval: `{not boundary.get('sim_heldout_success_is_not_deployment_approval', True)}`",
        f"- Robot readiness proven: `{boundary.get('robot_readiness_proven')}`",
        f"- Public claim upgrade allowed: `{boundary.get('public_claim_upgrade_allowed')}`",
    ]
    blockers = manifest.get("blockers")
    if isinstance(blockers, list) and blockers:
        lines.extend(["", "## Blockers", ""])
        lines.extend(f"- `{blocker}`" for blocker in blockers)
    return "\n".join(lines) + "\n"


def build_policy_improvement_run_offer(
    *,
    capture_root: str | Path,
    job_dir: str | Path,
    output_dir: str | Path | None = None,
    access_level: str = "black_box",
    customer_policy_ref: str | None = None,
    embodiment: str | None = None,
    action_interface: str | None = None,
    target_task: str | None = None,
    success_threshold: float | None = None,
    cycle_time_threshold_seconds: float | None = None,
    improvement_targets: Sequence[str] | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    if access_level not in ACCESS_LEVELS:
        raise ValueError(f"Unsupported access level: {access_level}")
    targets = tuple(improvement_targets or ("adapter", "task_head", "distilled_skill"))
    unknown_targets = sorted(set(targets) - set(IMPROVEMENT_TARGETS))
    if unknown_targets:
        raise ValueError(f"Unsupported improvement target(s): {', '.join(unknown_targets)}")

    context = resolve_local_capture_context(capture_root)
    resolved_job_dir = Path(job_dir).resolve()
    resolved_output_dir = (
        Path(output_dir).resolve()
        if output_dir
        else resolved_job_dir / DEFAULT_OUTPUT_DIR_NAME
    )
    ensure_dir(resolved_output_dir)
    generated = generated_at or utc_now_iso()

    included_artifacts: dict[str, str] = {}
    job_artifacts = {
        "job_request": "job_request.json",
        "scenario_eval_matrix": "scenario_eval_matrix.json",
        "normalized_attempt_trace": "normalized_attempt_trace.json",
        "failure_labels": "failure_labels.json",
        "policy_package_manifest": "policy_package_manifest.json",
        "policy_execution_manifest": "policy_execution_manifest.json",
        "policy_execution_trace": "policy_execution_trace.json",
        "training_request": "training_request.json",
        "training_result": "training_result.json",
        "evaluation_result": "evaluation_result.json",
        "post_training_data_package_export_manifest": (
            "post_training_data_package_export_manifest.json"
        ),
        "policy_autoresearch_report": "policy_autoresearch/policy_autoresearch_report.json",
        "policy_candidate_package": "policy_autoresearch/policy_candidate_package.json",
        "heldout_eval_result": "policy_autoresearch/heldout_eval_result.json",
        "agent_idea_tree": "policy_autoresearch/agent_idea_tree.json",
        "followup_real_world_validation_request": (
            "policy_autoresearch/followup_real_world_validation_request.json"
        ),
        "budget_ledger": "policy_autoresearch/budget_ledger.json",
        "live_eval_closure_manifest": "live_eval_closure_manifest.json",
        "proof_boundary": "proof_boundary.json",
    }
    for key, relative_path in job_artifacts.items():
        _include_if_file(
            included_artifacts,
            key=key,
            base_dir=resolved_output_dir,
            path=resolved_job_dir / relative_path,
        )

    pipeline_artifacts = {
        "site_card": "robot_eval_dataset/site_card.json",
        "task_cards": "robot_eval_dataset/task_cards.json",
        "scenario_cards": "robot_eval_dataset/scenario_cards.json",
        "eval_cards": "robot_eval_dataset/eval_cards.json",
        "rights_packet": "robot_eval_dataset/rights_packet.json",
        "proof_boundaries": "robot_eval_dataset/proof_boundaries.json",
    }
    for key, relative_path in pipeline_artifacts.items():
        _include_if_file(
            included_artifacts,
            key=key,
            base_dir=resolved_output_dir,
            path=context.pipeline_root / relative_path,
        )

    job_request = _read_optional_mapping(resolved_job_dir / "job_request.json")
    scenario_matrix = _read_optional_mapping(resolved_job_dir / "scenario_eval_matrix.json")
    labels = _read_optional_mapping(resolved_job_dir / "failure_labels.json")
    policy_package = _read_optional_mapping(resolved_job_dir / "policy_package_manifest.json")
    post_training_package = _read_optional_mapping(
        resolved_job_dir / "post_training_data_package_export_manifest.json"
    )
    policy_report = _read_optional_mapping(
        resolved_job_dir / "policy_autoresearch" / "policy_autoresearch_report.json"
    )
    candidate_package = _read_optional_mapping(
        resolved_job_dir / "policy_autoresearch" / "policy_candidate_package.json"
    )
    heldout_result = _read_optional_mapping(
        resolved_job_dir / "policy_autoresearch" / "heldout_eval_result.json"
    )
    evaluation_result = _read_optional_mapping(resolved_job_dir / "evaluation_result.json")

    customer_input_summary = _customer_inputs(
        job_request=job_request,
        policy_package=policy_package,
        customer_policy_ref=customer_policy_ref,
        embodiment=embodiment,
        action_interface=action_interface,
        target_task=target_task,
        success_threshold=success_threshold,
        cycle_time_threshold_seconds=cycle_time_threshold_seconds,
    )
    split_counts = _split_counts(scenario_matrix)
    policy_summary = _policy_autoresearch_summary(
        report=policy_report,
        candidate=candidate_package,
        heldout=heldout_result,
    )
    status, blockers = _status_and_blockers(
        customer_inputs=customer_input_summary,
        included_artifacts=included_artifacts,
        split_counts=split_counts,
        policy_summary=policy_summary,
        post_training_package=post_training_package,
    )
    readiness_ladder = _readiness_ladder(
        customer_inputs=customer_input_summary,
        included_artifacts=included_artifacts,
        split_counts=split_counts,
        post_training_package=post_training_package,
        policy_summary=policy_summary,
    )

    access = dict(ACCESS_LEVELS[access_level])
    access["access_level"] = access_level
    boundary = dict(CLAIM_BOUNDARY)
    boundary["simulator_execution_proven"] = bool(
        candidate_package.get("simulator_execution_proven") is True
    )
    boundary["robot_policy_execution_proven"] = bool(
        candidate_package.get("robot_policy_execution_proven") is True
    )
    boundary["robot_readiness_proven"] = bool(
        candidate_package.get("robot_readiness_proven") is True
    )
    boundary["public_claim_upgrade_allowed"] = bool(
        candidate_package.get("public_claim_upgrade_allowed") is True
    )

    manifest: dict[str, Any] = {
        "schema_version": POLICY_IMPROVEMENT_RUN_SCHEMA_VERSION,
        "generated_at": generated,
        "scene_id": context.scene_id,
        "capture_id": context.capture_id,
        "job_dir": str(resolved_job_dir),
        "product": {
            "name": "Policy Improvement Run",
            "product_family": "task_eval_to_policy_improvement",
            "one_line": (
                "Evaluate a customer-supplied robot policy against a real-site "
                "Task Evaluation Run, turn failures into twin/cousin curricula, "
                "post-train a bounded candidate, and report sealed before/after evidence."
            ),
            "primary_sale": "policy lift toward a pilot gate",
            "not_sold_as": [
                "foundation-model ownership",
                "deployment approval",
                "physical safety certification",
                "guaranteed site production readiness",
            ],
            "extends": ["Task Evaluation Run", "Post-Training Data Package"],
        },
        "status": status,
        "blockers": blockers,
        "customer_inputs": customer_input_summary,
        "access_model": access,
        "improvement_targets": list(targets),
        "workflow": [
            "baseline_evaluation",
            "dominant_failure_mode_diagnosis",
            "digital_twin_and_cousin_scenario_generation",
            "curriculum_build",
            "candidate_post_training_or_policy_lift",
            "sealed_scenario_evaluation",
            "improved_artifact_and_evidence_report",
        ],
        "task_evaluation_run_parity": {
            "baseline_eval_result_required": True,
            "scenario_eval_matrix_required": True,
            "normalized_attempt_trace_required": True,
            "standard_policy_scorecard_projected": True,
            "customer_handoff_ready_only_after_review": True,
            "webapp_projection_safe": True,
        },
        "baseline_evaluation_summary": _scorecard_summary(evaluation_result),
        "scenario_split_policy": {
            "development_visible_to_autoresearch": True,
            "validation_limited_feedback": True,
            "heldout_or_sealed_required_for_promotion": True,
            "sealed_audit_inaccessible_to_training": True,
            "split_counts": split_counts,
        },
        "failure_mode_summary": _failure_mode_summary(labels),
        "post_training_package_summary": {
            "status": post_training_package.get("status") or "missing",
            "package_type": post_training_package.get("package_type"),
            "export_policy": post_training_package.get("export_policy") or {},
            "manifest_counts": post_training_package.get("manifest_counts") or {},
        },
        "policy_autoresearch_summary": policy_summary,
        "readiness_ladder": readiness_ladder,
        "deliverables": [
            "baseline_eval_report",
            "failure_mode_report",
            "twin_and_cousin_scenario_curriculum",
            "post_training_data_package",
            "policy_candidate_package",
            "sealed_heldout_eval_result",
            "customer_evidence_report",
            "real_world_validation_followup_request",
        ],
        "commercial_tiers": [
            {
                "tier": "baseline_eval_and_failure_diagnosis",
                "typical_when": "team needs a credible before score and failure taxonomy",
            },
            {
                "tier": "policy_lift_sprint",
                "typical_when": "team has a borderline policy and needs a better candidate",
            },
            {
                "tier": "pilot_gate_package",
                "typical_when": "team needs before/after evidence for a site pilot decision",
            },
        ],
        "included_artifacts": included_artifacts,
        "manifest_path": "policy_improvement_run_offer.json",
        "brief_path": "policy_improvement_run_offer.md",
        "claim_boundary": boundary,
    }
    manifest["webapp_summary_projection"] = _webapp_summary_projection(manifest)

    write_json(resolved_output_dir / "policy_improvement_run_offer.json", manifest)
    write_json(
        resolved_output_dir / "policy_improvement_run_webapp_summary.json",
        manifest["webapp_summary_projection"],
    )
    write_text(resolved_output_dir / "policy_improvement_run_offer.md", _markdown_brief(manifest))
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a Policy Improvement Run offer manifest")
    parser.add_argument("--capture-root", required=True)
    parser.add_argument("--job-dir", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument(
        "--access-level",
        choices=sorted(ACCESS_LEVELS),
        default="black_box",
    )
    parser.add_argument("--customer-policy-ref")
    parser.add_argument("--embodiment")
    parser.add_argument("--action-interface")
    parser.add_argument("--target-task")
    parser.add_argument("--success-threshold", type=float)
    parser.add_argument("--cycle-time-threshold-seconds", type=float)
    parser.add_argument(
        "--improvement-target",
        action="append",
        choices=IMPROVEMENT_TARGETS,
        dest="improvement_targets",
    )
    args = parser.parse_args(argv)
    result = build_policy_improvement_run_offer(
        capture_root=args.capture_root,
        job_dir=args.job_dir,
        output_dir=args.output_dir,
        access_level=args.access_level,
        customer_policy_ref=args.customer_policy_ref,
        embodiment=args.embodiment,
        action_interface=args.action_interface,
        target_task=args.target_task,
        success_threshold=args.success_threshold,
        cycle_time_threshold_seconds=args.cycle_time_threshold_seconds,
        improvement_targets=args.improvement_targets,
    )
    manifest_dir = Path(args.output_dir or Path(args.job_dir) / DEFAULT_OUTPUT_DIR_NAME)
    print(f"[policy-improvement-run] manifest={manifest_dir / 'policy_improvement_run_offer.json'}")
    print(f"[policy-improvement-run] status={result['status']}")
    return 0 if not str(result["status"]).startswith("blocked") else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
