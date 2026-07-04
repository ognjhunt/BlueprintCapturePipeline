"""Buyer-facing Task Evaluation Run report composer.

This is the single deliverable artifact a robot team receives for a Task
Evaluation Run. It composes, fail-closed:

- the 7-layer success-claim ledger from ``success_claim_contracts`` (the only
  path through which any task-success language may reach a buyer),
- a per task x scenario scorecard where every rate carries its trial count and
  a Wilson 95% binomial interval (a bare success percentage is never emitted),
- provider/runtime execution health, strictly separated from task-success
  proof (provider success is infrastructure health, never task success),
- the rights/privacy delivery gate verdict,
- an explicit safety-claim boundary: evaluation results are evidence inputs to
  the operator's ISO 10218-2 / ANSI R15.06 / ISO 3691-4 risk assessment of the
  integrated application. They are never "safe", "compliant", or
  "deployment ready" claims.

The report never contains a top-level bare success boolean. All success
language is scoped to the ledger's ``highest_truthful_claim``.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from typing import Any, Dict, List, Mapping, Sequence

from .success_claim_contracts import (
    CLAIM_LADDER,
    LEDGER_SCHEMA_VERSION,
    build_success_claim_ledger,
    coerce_strict_success,
)
from .common import utc_now_iso

TASK_EVAL_RUN_REPORT_SCHEMA_VERSION = "task_eval_run_buyer_report.v1"
SCORECARD_SCHEMA_VERSION = "task_eval_run_scorecard.v1"

# TRI-style published protocol treats ~20 trials per condition as the floor for
# a decision-grade comparison; below that the scorecard row is flagged.
RECOMMENDED_MIN_TRIALS_PER_CONDITION = 20

BINOMIAL_CI_METHOD = "wilson_score_95"
_WILSON_Z = 1.959963984540054  # two-sided 95%

# Provider payloads describe infrastructure health only. Any of these keys in a
# provider payload is an attempted task-success claim and is refused.
_PROVIDER_OVERCLAIM_KEYS = frozenset(
    {
        "task_success",
        "task_success_proven",
        "policy_task_success",
        "success",
        "success_rate",
        "deployment_ready",
        "safety_validated",
    }
)

SAFETY_CLAIM_BOUNDARY = {
    "results_are_evidence_inputs_only": True,
    "safety_or_compliance_claimed": False,
    "deployment_readiness_claimed": False,
    "statement": (
        "Task Evaluation Run results are evidence inputs to the site operator's "
        "risk assessment of the integrated application (ISO 10218-1/-2:2025, "
        "ANSI/A3 R15.06, ISO 3691-4 / R15.08 for mobile platforms). They are "
        "not a safety validation, compliance determination, or deployment "
        "approval, which attach to a specific robot, task, cell, and site and "
        "must be made by the integrator/operator."
    ),
}


def _mapping(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _string(value: Any) -> str:
    return str(value).strip() if isinstance(value, str) else ""


def _wilson_interval(successes: int, trials: int) -> Dict[str, float] | None:
    if trials <= 0:
        return None
    z = _WILSON_Z
    p_hat = successes / trials
    denom = 1.0 + (z * z) / trials
    center = (p_hat + (z * z) / (2 * trials)) / denom
    margin = (
        z
        * math.sqrt((p_hat * (1.0 - p_hat) + (z * z) / (4 * trials)) / trials)
        / denom
    )
    return {
        "point": round(p_hat, 6),
        "lower_95": round(max(0.0, center - margin), 6),
        "upper_95": round(min(1.0, center + margin), 6),
    }


def build_task_eval_scorecard(
    *,
    attempts: Sequence[Mapping[str, Any]],
    evidence_level: str,
    recommended_min_trials: int = RECOMMENDED_MIN_TRIALS_PER_CONDITION,
) -> Dict[str, Any]:
    """Group attempts by (task_id, scenario_id) and report interval-bounded rates.

    Success labels are read strictly (bool or 0/1 only); anything else makes the
    attempt invalid and blocks the scorecard rather than silently coercing.
    """
    conditions: "OrderedDict[tuple[str, str], Dict[str, int]]" = OrderedDict()
    blockers: List[str] = []
    invalid_attempts: List[str] = []
    for index, attempt in enumerate(attempts):
        row = _mapping(attempt)
        attempt_id = _string(row.get("attempt_id")) or f"attempt_{index + 1:04d}"
        success = coerce_strict_success(row.get("success"))
        if success is None:
            invalid_attempts.append(attempt_id)
            continue
        key = (
            _string(row.get("task_id")) or "unspecified_task",
            _string(row.get("scenario_id")) or "unspecified_scenario",
        )
        bucket = conditions.setdefault(key, {"trials": 0, "successes": 0})
        bucket["trials"] += 1
        bucket["successes"] += int(success)
    if invalid_attempts:
        blockers.append(
            "attempts_with_non_boolean_success_label:" + ",".join(invalid_attempts)
        )
    if not conditions and not invalid_attempts:
        blockers.append("no_attempts_available_for_scorecard")

    rows: List[Dict[str, Any]] = []
    for (task_id, scenario_id), bucket in conditions.items():
        interval = _wilson_interval(bucket["successes"], bucket["trials"])
        rows.append(
            {
                "task_id": task_id,
                "scenario_id": scenario_id,
                "trials": bucket["trials"],
                "successes": bucket["successes"],
                "success_rate": interval,
                "below_recommended_trials": bucket["trials"] < recommended_min_trials,
            }
        )
    return {
        "schema_version": SCORECARD_SCHEMA_VERSION,
        "status": "completed" if rows and not blockers else "blocked",
        "evidence_level": evidence_level,
        "success_definition": (
            "Rates count attempt-level success labels at evidence level "
            f"'{evidence_level}' on the success-claim ladder. They are not "
            "physical-robot, contact, or deployment proof unless that ladder "
            "level says so."
        ),
        "binomial_ci_method": BINOMIAL_CI_METHOD,
        "recommended_min_trials_per_condition": recommended_min_trials,
        "conditions": rows,
        "invalid_attempt_ids": invalid_attempts,
        "blockers": blockers,
    }


def _sanitize_provider_execution(
    provider_execution: Mapping[str, Any] | None,
) -> tuple[Dict[str, Any], List[str]]:
    payload = _mapping(provider_execution)
    blockers: List[str] = []
    refused = sorted(key for key in payload if key in _PROVIDER_OVERCLAIM_KEYS)
    for key in refused:
        payload.pop(key, None)
        blockers.append(f"provider_payload_attempted_task_success_claim:{key}")
    payload["reader_boundary"] = (
        "Provider/runtime status (exit codes, GPU hours, cost) is "
        "infrastructure health only. Task success is proven exclusively by "
        "the success_claim_ledger in this report."
    )
    payload["refused_task_success_keys"] = refused
    return payload, blockers


def build_task_eval_run_report(
    *,
    job_id: str | None,
    scene_id: str | None = None,
    capture_id: str | None = None,
    attempt_trace: Mapping[str, Any] | None,
    task_metadata: Mapping[str, Any] | None = None,
    success_claim_ledger: Mapping[str, Any] | None = None,
    success_claim_layers: Mapping[str, Any] | None = None,
    provider_execution: Mapping[str, Any] | None = None,
    policy_binding: Mapping[str, Any] | None = None,
    rights_privacy_gate: Mapping[str, Any] | None = None,
    generated_at: str | None = None,
) -> Dict[str, Any]:
    """Compose the buyer deliverable for one Task Evaluation Run, fail-closed.

    Either a prebuilt ``success_claim_ledger`` or the raw ``success_claim_layers``
    (keyword inputs for ``build_success_claim_ledger``) must be provided; when
    both are absent the report is blocked and carries claim level ``no_claim``.
    """
    blockers: List[str] = []
    trace = _mapping(attempt_trace)
    attempts = [row for row in trace.get("attempts") or [] if isinstance(row, Mapping)]
    if not attempts:
        blockers.append("attempt_trace_missing_or_empty")

    ledger = _mapping(success_claim_ledger)
    if not ledger and success_claim_layers is not None:
        ledger = build_success_claim_ledger(
            task_metadata=task_metadata,
            **{
                key: value
                for key, value in _mapping(success_claim_layers).items()
                if key
                in {
                    "media_validity",
                    "review_task_success",
                    "task_success_contract",
                    "simulator_execution",
                    "policy_action_execution",
                    "contact_state_change",
                    "physical_readiness",
                }
            },
        )
    if not ledger:
        blockers.append("success_claim_ledger_missing")
        evidence_level = "no_claim"
    else:
        if ledger.get("schema_version") != LEDGER_SCHEMA_VERSION:
            blockers.append("success_claim_ledger_schema_unrecognized")
        evidence_level = _string(ledger.get("highest_truthful_claim")) or "no_claim"
        if evidence_level not in CLAIM_LADDER:
            blockers.append("success_claim_ledger_claim_level_unrecognized")
            evidence_level = "no_claim"

    rights_gate = _mapping(rights_privacy_gate)
    if not rights_gate:
        blockers.append("rights_privacy_gate_missing")
    else:
        gate_status = _string(rights_gate.get("status")).lower()
        gate_cleared = rights_gate.get("cleared") is True or gate_status in {
            "cleared",
            "pass",
            "passed",
        }
        if not gate_cleared:
            blockers.append("rights_privacy_gate_not_cleared")

    provider_payload, provider_blockers = _sanitize_provider_execution(
        provider_execution
    )
    blockers.extend(provider_blockers)

    scorecard = build_task_eval_scorecard(
        attempts=attempts, evidence_level=evidence_level
    )
    blockers.extend(
        f"scorecard:{blocker}" for blocker in scorecard.get("blockers") or []
    )

    status = "ready_review_required" if not blockers else "blocked"
    return {
        "schema_version": TASK_EVAL_RUN_REPORT_SCHEMA_VERSION,
        "generated_at": generated_at or utc_now_iso(),
        "status": status,
        "job_id": _string(job_id) or None,
        "scene_id": _string(scene_id) or None,
        "capture_id": _string(capture_id) or None,
        "success_claim_ledger": ledger or None,
        "evidence_level": evidence_level,
        "scorecard": scorecard,
        "policy_binding": _mapping(policy_binding) or None,
        "provider_execution": provider_payload,
        "rights_privacy_gate": rights_gate or None,
        "blockers": sorted(set(blockers)),
        "claim_boundary": {
            "task_success_language_scoped_to_ledger": True,
            "bare_success_booleans_forbidden": True,
            "provider_runtime_success_is_not_task_success": True,
            "generated_or_rendered_media_is_not_physical_proof": True,
            "safety": dict(SAFETY_CLAIM_BOUNDARY),
        },
    }
