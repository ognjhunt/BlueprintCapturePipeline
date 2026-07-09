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
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from .buyer_claim_ceiling import build_buyer_claim_ceiling
from .success_claim_contracts import (
    CLAIM_LADDER,
    LEDGER_SCHEMA_VERSION,
    build_success_claim_ledger,
    coerce_strict_success,
)
from .common import utc_now_iso
from .wam_score_claim_gate import summarize_wam_evaluation_for_report

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
_PROVIDER_OVERCLAIM_KEY_TOKENS = frozenset(
    "".join(ch for ch in key.lower() if ch.isalnum())
    for key in _PROVIDER_OVERCLAIM_KEYS
) | {
    "safetodeploy",
    "safefordeployment",
    "deploymentapproved",
    "deploymentapproval",
    "compliancevalidated",
    "complianceapproved",
    "safetycompliant",
}
_POLICY_BINDING_SECRET_KEY_TOKENS = frozenset(
    {
        "apikey",
        "apitoken",
        "authorization",
        "bearertoken",
        "clientsecret",
        "credential",
        "credentials",
        "privatekey",
        "refreshtoken",
        "sessiontoken",
        "secret",
        "secretaccesskey",
        "token",
        "accesskey",
        "accesskeyid",
    }
)
_POLICY_BINDING_SECRET_KEY_MARKERS = frozenset(
    {
        "accesskey",
        "accesskeyid",
        "apikey",
        "apitoken",
        "authorization",
        "bearertoken",
        "clientsecret",
        "privatekey",
        "refreshtoken",
        "secretaccesskey",
        "sessiontoken",
    }
)
_POLICY_BINDING_SECRET_KEY_SUFFIXES = frozenset(
    {
        "credential",
        "credentials",
        "password",
        "secret",
        "token",
    }
)
_REDACTED_VALUES = frozenset({"", "<redacted>", "redacted", "***", "****"})

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


def _string_list(value: Any) -> List[str]:
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [text for text in (_string(item) for item in value) if text]
    return []


def _provider_key_token(key: Any) -> str:
    return "".join(ch for ch in str(key).lower() if ch.isalnum())


def _policy_binding_key_is_secret(key: Any) -> bool:
    token = _provider_key_token(key)
    if token in _POLICY_BINDING_SECRET_KEY_TOKENS:
        return True
    if any(marker in token for marker in _POLICY_BINDING_SECRET_KEY_MARKERS):
        return True
    return any(token.endswith(suffix) for suffix in _POLICY_BINDING_SECRET_KEY_SUFFIXES)


def _explicit_true(value: Any) -> bool:
    if value is True:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {
            "1",
            "true",
            "yes",
            "y",
            "revoked",
            "takedown_required",
            "blocked_consent_revoked_takedown_required",
        }
    return False


def _is_already_redacted(value: Any) -> bool:
    return isinstance(value, str) and value.strip().lower() in _REDACTED_VALUES


def _sanitize_policy_binding(
    policy_binding: Mapping[str, Any] | None,
) -> tuple[Dict[str, Any] | None, List[str]]:
    if not policy_binding:
        return None, []

    redacted_paths: List[str] = []
    leaked_paths: List[str] = []

    def _sanitize(value: Any, path: str) -> Any:
        if isinstance(value, Mapping):
            sanitized: Dict[str, Any] = {}
            for key, child in value.items():
                key_text = str(key)
                child_path = f"{path}.{key_text}" if path else key_text
                if _policy_binding_key_is_secret(key_text):
                    redacted_paths.append(child_path)
                    if child not in (None, "", []) and not _is_already_redacted(child):
                        leaked_paths.append(child_path)
                    sanitized[key_text] = "<redacted>"
                    continue
                sanitized[key_text] = _sanitize(child, child_path)
            return sanitized
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return [
                _sanitize(child, f"{path}[{index}]")
                for index, child in enumerate(value)
            ]
        return value

    sanitized = _sanitize(_mapping(policy_binding), "")
    if redacted_paths:
        sanitized["secret_values_redacted"] = True
        sanitized["redacted_secret_paths"] = sorted(set(redacted_paths))
        sanitized["reader_boundary"] = (
            "Policy bindings may identify policy/checkpoint interfaces, but "
            "customer-visible reports never carry provider tokens, API keys, "
            "credentials, or bearer secrets."
        )
    blockers = [
        f"policy_binding_secret_value_redacted:{path}"
        for path in sorted(set(leaked_paths))
    ]
    return sanitized, blockers


def _rights_privacy_gate_blockers(rights_gate: Mapping[str, Any]) -> List[str]:
    gate_status = _string(rights_gate.get("status")).lower()
    gate_cleared = rights_gate.get("cleared") is True or gate_status in {
        "cleared",
        "pass",
        "passed",
    }
    blockers: List[str] = []
    if not gate_cleared:
        blockers.append("rights_privacy_gate_not_cleared")
    gate_blockers = _string_list(rights_gate.get("blockers"))
    gate_blockers.extend(_string_list(rights_gate.get("blocking_reasons")))
    blockers.extend(
        f"rights_privacy_gate_blocker:{blocker}" for blocker in gate_blockers
    )

    revocation_takedown = _mapping(rights_gate.get("revocation_takedown"))
    if (
        _explicit_true(rights_gate.get("consent_revoked"))
        or _explicit_true(rights_gate.get("takedown_open"))
        or _explicit_true(rights_gate.get("delivery_blocked"))
        or _explicit_true(revocation_takedown.get("consent_revoked"))
        or _explicit_true(revocation_takedown.get("takedown_open"))
        or _string(revocation_takedown.get("status")).lower() == "takedown_required"
        or gate_status
        in {
            "blocked_consent_revoked_takedown_required",
            "blocked_open_consent_takedown",
            "takedown_required",
            "revoked",
        }
    ):
        blockers.append("rights_privacy_gate_consent_revoked_takedown_required")
    return sorted(set(blockers))


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

    # Numeric success rates are published only when the evidence level carries a
    # task-success claim (review_task_success and above). Below that (no_claim,
    # media_valid) the per-attempt labels are not task-success measurements, so
    # emitting rates + "completed" would let a reader anchor on numbers the ladder
    # never earned — keep the factual trial/success counts, withhold the interval.
    rates_published = (
        evidence_level in CLAIM_LADDER
        and CLAIM_LADDER.index(evidence_level)
        >= CLAIM_LADDER.index("review_task_success")
    )
    rows: List[Dict[str, Any]] = []
    for (task_id, scenario_id), bucket in conditions.items():
        interval = _wilson_interval(bucket["successes"], bucket["trials"])
        rows.append(
            {
                "task_id": task_id,
                "scenario_id": scenario_id,
                "trials": bucket["trials"],
                "successes": bucket["successes"],
                "success_rate": interval if rates_published else None,
                "below_recommended_trials": bucket["trials"] < recommended_min_trials,
            }
        )
    if not rows or blockers:
        status = "blocked"
    elif rates_published:
        status = "completed"
    else:
        status = "rates_withheld_insufficient_evidence"
    return {
        "schema_version": SCORECARD_SCHEMA_VERSION,
        "status": status,
        "rates_published": rates_published,
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
    def _sanitize(value: Any, path: str) -> tuple[Any, List[str]]:
        if isinstance(value, Mapping):
            sanitized: Dict[str, Any] = {}
            refused_paths: List[str] = []
            for key, child in value.items():
                key_text = str(key)
                child_path = f"{path}.{key_text}" if path else key_text
                if _provider_key_token(key_text) in _PROVIDER_OVERCLAIM_KEY_TOKENS:
                    refused_paths.append(child_path)
                    continue
                sanitized_child, child_refused = _sanitize(child, child_path)
                sanitized[key_text] = sanitized_child
                refused_paths.extend(child_refused)
            return sanitized, refused_paths
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            sanitized_items: List[Any] = []
            refused_paths = []
            for index, child in enumerate(value):
                child_path = f"{path}[{index}]"
                sanitized_child, child_refused = _sanitize(child, child_path)
                sanitized_items.append(sanitized_child)
                refused_paths.extend(child_refused)
            return sanitized_items, refused_paths
        return value, []

    payload, refused = _sanitize(_mapping(provider_execution), "")
    blockers: List[str] = []
    refused = sorted(refused)
    for key in refused:
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
    wam_evaluation: Mapping[str, Any] | None = None,
    buyer_claim_proof_boundary: Mapping[str, Any] | None = None,
    live_closure: Mapping[str, Any] | None = None,
    buyer_claim_copy: Any = None,
    capture_root: str | Path | None = None,
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
        blockers.extend(_rights_privacy_gate_blockers(rights_gate))
    # TOCTOU: re-read consent LIVE at report emit so a revocation that landed
    # after the rights gate was computed still blocks the buyer report. Fail-closed:
    # a live read can only ADD the revocation blocker, never clear an inherited one.
    if capture_root is not None:
        from .consent_takedown import read_consent_state

        if read_consent_state(capture_root).get("state") == "revoked":
            blockers.append("rights_privacy_gate_consent_revoked_takedown_required")

    policy_binding_payload, policy_binding_blockers = _sanitize_policy_binding(
        policy_binding
    )
    blockers.extend(policy_binding_blockers)

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

    wam_section, wam_blockers = summarize_wam_evaluation_for_report(wam_evaluation)
    blockers.extend(wam_blockers)
    buyer_claim_ceiling = build_buyer_claim_ceiling(
        success_claim_ledger=ledger,
        proof_boundary=buyer_claim_proof_boundary,
        live_closure=live_closure,
        buyer_copy_inputs=buyer_claim_copy,
    )
    blockers.extend(
        f"buyer_claim_ceiling:{blocker}"
        for blocker in buyer_claim_ceiling.get("blockers", [])
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
        "policy_binding": policy_binding_payload,
        "wam_evaluation": wam_section,
        "provider_execution": provider_payload,
        "rights_privacy_gate": rights_gate or None,
        "blockers": sorted(set(blockers)),
        "claim_boundary": {
            "task_success_language_scoped_to_ledger": True,
            "bare_success_booleans_forbidden": True,
            "provider_runtime_success_is_not_task_success": True,
            "generated_or_rendered_media_is_not_physical_proof": True,
            "buyer_claim_ceiling": buyer_claim_ceiling,
            "safety": dict(SAFETY_CLAIM_BOUNDARY),
        },
    }
