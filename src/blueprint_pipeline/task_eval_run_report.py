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

from .success_claim_contracts import (
    CLAIM_LADDER,
    LEDGER_SCHEMA_VERSION,
    build_success_claim_ledger,
    coerce_strict_success,
)
from .common import utc_now_iso
from .site_taxonomy import INDUSTRIAL_CATEGORIES, resolve_site_type
from .wam_score_claim_gate import summarize_wam_evaluation_for_report

TASK_EVAL_RUN_REPORT_SCHEMA_VERSION = "task_eval_run_buyer_report.v1"
SCORECARD_SCHEMA_VERSION = "task_eval_run_scorecard.v1"

# Controlled vocabulary for where an attempt's success label came from. This is
# how a buyer tells a VLM judgment over GENERATED rollout video apart from
# simulator physics or a recorded real-world trace. The WAM generated-video VLM
# labelers (wam_generated_video_success_label_gemini / _openai) stamp each label
# with ``success_label_provenance = generated_video_vlm``; that value is threaded
# into the attempt rows this scorecard reads.
GENERATED_VIDEO_VLM_PROVENANCE = "generated_video_vlm"
SIMULATOR_PHYSICS_PROVENANCE = "simulator_physics"
RECORDED_TRACE_PROVENANCE = "recorded_trace"
UNKNOWN_PROVENANCE = "unknown"
SUCCESS_LABEL_PROVENANCE_VOCABULARY: tuple[str, ...] = (
    GENERATED_VIDEO_VLM_PROVENANCE,
    SIMULATOR_PHYSICS_PROVENANCE,
    RECORDED_TRACE_PROVENANCE,
    UNKNOWN_PROVENANCE,
)
# Only these provenances establish a success_rate as physics or captured truth.
_PHYSICS_OR_CAPTURED_TRUTH_PROVENANCES = frozenset(
    {SIMULATOR_PHYSICS_PROVENANCE, RECORDED_TRACE_PROVENANCE}
)
# Provenances that must NEVER be presented as an unqualified physics/real-world
# success claim (generated video is a VLM judgment; unknown is unestablished).
_NON_TRUTH_PROVENANCES = frozenset(
    {GENERATED_VIDEO_VLM_PROVENANCE, UNKNOWN_PROVENANCE}
)
# The disclosure string a generated-video-VLM success_rate must always carry.
SUCCESS_RATE_GENERATED_VIDEO_VLM_CLAIM_BOUNDARY = (
    "success_rate_from_generated_video_vlm_is_not_physics_or_captured_truth"
)
SUCCESS_RATE_UNKNOWN_PROVENANCE_CLAIM_BOUNDARY = (
    "success_rate_provenance_unknown_is_not_established_physics_or_captured_truth"
)


def _normalize_success_label_provenance(value: Any) -> str:
    """Map an attempt's declared provenance onto the controlled vocabulary.

    Anything absent or outside the vocabulary is treated conservatively as
    ``unknown`` — provenance is never fabricated.
    """
    text = _string(value)
    return text if text in SUCCESS_LABEL_PROVENANCE_VOCABULARY else UNKNOWN_PROVENANCE


def _collapse_row_provenance(distinct: Sequence[str]) -> str:
    """Collapse a condition's distinct provenances to one vocabulary value.

    A single source is reported as-is. Mixed sources within one condition are
    never presented as a single trusted provenance: a generated-video-VLM
    contributor wins the disclosure, otherwise the row is marked ``unknown``.
    """
    ordered = list(dict.fromkeys(distinct))
    if len(ordered) == 1:
        return ordered[0]
    if GENERATED_VIDEO_VLM_PROVENANCE in ordered:
        return GENERATED_VIDEO_VLM_PROVENANCE
    return UNKNOWN_PROVENANCE


def _row_provenance_claim_boundary(distinct: Sequence[str]) -> str | None:
    """The disclosure string a row's success_rate must carry, if any."""
    present = set(distinct)
    if GENERATED_VIDEO_VLM_PROVENANCE in present:
        return SUCCESS_RATE_GENERATED_VIDEO_VLM_CLAIM_BOUNDARY
    if UNKNOWN_PROVENANCE in present:
        return SUCCESS_RATE_UNKNOWN_PROVENANCE_CLAIM_BOUNDARY
    return None


# ---------------------------------------------------------------------------
# R066: industrial-assembly success-metric semantics.
#
# The generic scorecard carries navigation/transfer/kitchen-ish metrics but no
# industrial-assembly success semantics. These four metrics are surfaced,
# additively and per-condition, whenever a condition belongs to an industrial
# assembly/insertion/precision-placement task family OR the run's site resolves
# (via site_taxonomy) to an industrial category. They are DECLARED spec
# tolerances or MEASURED attempt outcomes threaded in from inputs — never
# fabricated. A metric with no supplied value is surfaced explicitly as unset
# with provenance "unset", mirroring the site_extent capture-truth contract.
# ---------------------------------------------------------------------------

# Substring tokens (short, human-authored task text) marking a condition as an
# industrial assembly/insertion/precision-placement family. Matched against a
# condition's declared task_family / task_category, consistent with the
# site_taxonomy synonym approach.
INDUSTRIAL_ASSEMBLY_TASK_FAMILY_TOKENS: tuple[str, ...] = (
    "assembly",
    "insertion",
    "insert",
    "peg_in_hole",
    "peg-in-hole",
    "peg in hole",
    "press_fit",
    "press-fit",
    "press fit",
    "fastening",
    "fasten",
    "screw",
    "bolt",
    "kitting",
    "part_placement",
    "part-placement",
    "part placement",
    "precision_placement",
    "precision placement",
    "machine_tending",
    "machine tending",
)

INDUSTRIAL_METRIC_UNSET_PROVENANCE = "unset"
INDUSTRIAL_METRIC_PROVENANCE_UNSPECIFIED = "declared_provenance_unspecified"
INDUSTRIAL_METRIC_STATUS_UNSET = "needs_measurement_or_operator_input"
INDUSTRIAL_METRIC_STATUS_PRESENT = "declared_or_measured_present"

# field_id -> (unit, kind). kind distinguishes a declared spec tolerance (a task
# input) from a measured attempt outcome; neither is fabricated.
_INDUSTRIAL_SUCCESS_METRIC_FIELDS: tuple[tuple[str, str, str], ...] = (
    ("placement_accuracy_m", "meters", "measured_outcome"),
    ("insertion_tolerance_m", "meters", "declared_tolerance"),
    ("force_torque_within_envelope", "boolean", "measured_outcome"),
    ("dimensional_tolerance", "meters", "declared_tolerance"),
)
_INDUSTRIAL_METRIC_FIELD_IDS: tuple[str, ...] = tuple(
    field for field, _unit, _kind in _INDUSTRIAL_SUCCESS_METRIC_FIELDS
)

INDUSTRIAL_SUCCESS_METRIC_CLAIM_BOUNDARY = (
    "industrial_success_metrics_are_declared_or_measured_capture_inputs_"
    "not_fabricated_or_verified_success_proof"
)


def _industrial_task_family_match(*texts: Any) -> str | None:
    """Return the industrial-assembly token a task family text matches, if any."""
    for text in texts:
        lowered = _string(text).lower()
        if not lowered:
            continue
        for token in INDUSTRIAL_ASSEMBLY_TASK_FAMILY_TOKENS:
            if token in lowered:
                return token
    return None


def _resolve_site_industrial(task_metadata: Mapping[str, Any]) -> tuple[bool, str]:
    """Resolve whether the run's site is an industrial category (site_taxonomy).

    Prefers an explicit canonical ``site_category``; otherwise resolves free-text
    ``site_type`` through the shared taxonomy. Never guesses beyond the taxonomy.
    """
    md = _mapping(task_metadata)
    explicit_category = _string(md.get("site_category"))
    if explicit_category:
        return explicit_category in INDUSTRIAL_CATEGORIES, explicit_category
    site_type = _string(md.get("site_type")) or _string(md.get("target_site_type"))
    if site_type:
        resolution = resolve_site_type(site_type)
        return resolution.is_industrial, resolution.category
    return False, ""


def _coerce_metric_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and not isinstance(value, bool) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"true", "1", "yes", "y", "within_envelope", "pass", "passed"}:
            return True
        if token in {"false", "0", "no", "n", "out_of_envelope", "fail", "failed"}:
            return False
    return None


def _coerce_metric_value(unit: str, value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    if unit == "boolean":
        return _coerce_metric_bool(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _industrial_metric_observation(unit: str, raw: Any) -> tuple[Any, str] | None:
    """Coerce one declared/measured metric input into a (value, provenance) pair.

    ``raw`` may be a bare scalar or a mapping ``{"value": ..., "provenance": ...}``.
    ``unit`` selects the coercion (``"boolean"`` vs a numeric metric). Returns None
    when no usable value is present (never fabricated).
    """
    if isinstance(raw, Mapping):
        value = _coerce_metric_value(unit, raw.get("value"))
        provenance = _string(raw.get("provenance"))
    else:
        value = _coerce_metric_value(unit, raw)
        provenance = ""
    if value is None:
        return None
    return value, provenance or INDUSTRIAL_METRIC_PROVENANCE_UNSPECIFIED


def _industrial_metric_entry(
    field: str, unit: str, kind: str, observations: Sequence[tuple[Any, str]]
) -> Dict[str, Any]:
    """Build one industrial success-metric entry; unset when no value supplied."""
    if not observations:
        return {
            "field": field,
            "unit": unit,
            "kind": kind,
            "value": None,
            "observed_values": [],
            "provenance": INDUSTRIAL_METRIC_UNSET_PROVENANCE,
            "provenances": [INDUSTRIAL_METRIC_UNSET_PROVENANCE],
            "status": INDUSTRIAL_METRIC_STATUS_UNSET,
            "claim_boundary": INDUSTRIAL_SUCCESS_METRIC_CLAIM_BOUNDARY,
        }
    distinct_values: List[Any] = list(
        dict.fromkeys(value for value, _prov in observations)
    )
    provenances = sorted({prov for _value, prov in observations})
    return {
        "field": field,
        "unit": unit,
        "kind": kind,
        "value": distinct_values[0] if len(distinct_values) == 1 else None,
        "observed_values": distinct_values,
        "provenance": provenances[0] if len(provenances) == 1 else "mixed",
        "provenances": provenances,
        "status": INDUSTRIAL_METRIC_STATUS_PRESENT,
        "claim_boundary": INDUSTRIAL_SUCCESS_METRIC_CLAIM_BOUNDARY,
    }


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
    task_metadata: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Group attempts by (task_id, scenario_id) and report interval-bounded rates.

    Success labels are read strictly (bool or 0/1 only); anything else makes the
    attempt invalid and blocks the scorecard rather than silently coercing.

    ``task_metadata`` is an optional, read-only source of run-level context: the
    declared task family / site type (used to decide whether industrial-assembly
    success-metric semantics apply, R066) and any run-level declared industrial
    metric tolerances. It never affects success rates.
    """
    conditions: "OrderedDict[tuple[str, str], Dict[str, Any]]" = OrderedDict()
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
        bucket = conditions.setdefault(
            key,
            {
                "trials": 0,
                "successes": 0,
                "provenances": set(),
                "task_families": set(),
                "industrial_metric_observations": {
                    field: [] for field in _INDUSTRIAL_METRIC_FIELD_IDS
                },
            },
        )
        bucket["trials"] += 1
        bucket["successes"] += int(success)
        bucket["provenances"].add(
            _normalize_success_label_provenance(row.get("success_label_provenance"))
        )
        family_text = _string(row.get("task_family")) or _string(row.get("task_category"))
        if family_text:
            bucket["task_families"].add(family_text)
        attempt_metrics = _mapping(row.get("industrial_metrics"))
        for field, unit, _kind in _INDUSTRIAL_SUCCESS_METRIC_FIELDS:
            if field in attempt_metrics:
                observation = _industrial_metric_observation(
                    unit, attempt_metrics.get(field)
                )
                if observation is not None:
                    bucket["industrial_metric_observations"][field].append(observation)
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
    # Run-level context for industrial-assembly success-metric semantics (R066).
    # Read-only: this never influences the success rates above.
    metadata = _mapping(task_metadata)
    metadata_family = _string(metadata.get("task_family")) or _string(
        metadata.get("task_category")
    )
    metadata_metrics = _mapping(metadata.get("industrial_metrics"))
    metadata_metric_observations: Dict[str, tuple[Any, str] | None] = {}
    for field, unit, _kind in _INDUSTRIAL_SUCCESS_METRIC_FIELDS:
        metadata_metric_observations[field] = (
            _industrial_metric_observation(unit, metadata_metrics.get(field))
            if field in metadata_metrics
            else None
        )
    site_is_industrial, site_category = _resolve_site_industrial(metadata)

    rows: List[Dict[str, Any]] = []
    observed_provenances: set[str] = set()
    industrial_condition_count = 0
    for (task_id, scenario_id), bucket in conditions.items():
        interval = _wilson_interval(bucket["successes"], bucket["trials"])
        distinct = sorted(bucket["provenances"])
        observed_provenances.update(distinct)
        # A success_rate is physics or captured truth ONLY when every contributing
        # label came from a simulator-physics or recorded real trace. A
        # generated-video-VLM or unknown contributor keeps the rate qualified.
        rate_is_truth = bool(distinct) and set(distinct).issubset(
            _PHYSICS_OR_CAPTURED_TRUTH_PROVENANCES
        )
        row_out = {
            "task_id": task_id,
            "scenario_id": scenario_id,
            "trials": bucket["trials"],
            "successes": bucket["successes"],
            "success_rate": interval if rates_published else None,
            "below_recommended_trials": bucket["trials"] < recommended_min_trials,
            "success_label_provenance": _collapse_row_provenance(distinct),
            "success_label_provenances": distinct,
            "success_rate_is_physics_or_captured_truth": rate_is_truth,
            "success_rate_claim_boundary": _row_provenance_claim_boundary(distinct),
        }
        # R066: additively surface industrial-assembly success-metric semantics
        # only for industrial-assembly task families or industrial sites. Any
        # other condition (kitchen/home/generic) is left byte-for-byte unchanged.
        declared_families = sorted(bucket["task_families"])
        family_token = _industrial_task_family_match(*declared_families, metadata_family)
        if family_token or site_is_industrial:
            metrics_block: Dict[str, Any] = {}
            any_metric_present = False
            for field, unit, kind in _INDUSTRIAL_SUCCESS_METRIC_FIELDS:
                observations = list(bucket["industrial_metric_observations"][field])
                metadata_observation = metadata_metric_observations[field]
                if metadata_observation is not None:
                    observations.append(metadata_observation)
                entry = _industrial_metric_entry(field, unit, kind, observations)
                if entry["status"] == INDUSTRIAL_METRIC_STATUS_PRESENT:
                    any_metric_present = True
                metrics_block[field] = entry
            if family_token and site_is_industrial:
                surfaced_reason = "task_family_and_industrial_site_category"
            elif family_token:
                surfaced_reason = "task_family"
            else:
                surfaced_reason = "industrial_site_category"
            row_out["industrial_success_metrics"] = {
                "surfaced": True,
                "surfaced_reason": surfaced_reason,
                "task_family_matched": family_token,
                "declared_task_families": declared_families,
                "site_category": site_category or None,
                "is_industrial_site": site_is_industrial,
                "metrics": metrics_block,
                "any_metric_declared_or_measured": any_metric_present,
                "all_metrics_unset": not any_metric_present,
                "claim_boundary": INDUSTRIAL_SUCCESS_METRIC_CLAIM_BOUNDARY,
            }
            industrial_condition_count += 1
        rows.append(row_out)
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
        # R066: always-present declaration of the industrial-assembly metric
        # vocabulary so the semantics are documented even when no condition is
        # industrial. Per-condition values live in each row's
        # ``industrial_success_metrics`` block (declared/measured or explicitly
        # unset — never fabricated).
        "industrial_success_metric_semantics": {
            "fields": [
                {"field": field, "unit": unit, "kind": kind}
                for field, unit, kind in _INDUSTRIAL_SUCCESS_METRIC_FIELDS
            ],
            "surfaced_condition_count": industrial_condition_count,
            "any_condition_industrial_assembly": industrial_condition_count > 0,
            "site_category": site_category or None,
            "is_industrial_site": site_is_industrial,
            "keyed_off": (
                "industrial_task_family_or_site_taxonomy_industrial_category"
            ),
            "claim_boundary": INDUSTRIAL_SUCCESS_METRIC_CLAIM_BOUNDARY,
        },
        "invalid_attempt_ids": invalid_attempts,
        "success_label_provenance_vocabulary": list(
            SUCCESS_LABEL_PROVENANCE_VOCABULARY
        ),
        "observed_success_label_provenances": sorted(observed_provenances),
        "success_label_provenance_boundary": {
            "any_success_rate_from_generated_video_vlm": (
                GENERATED_VIDEO_VLM_PROVENANCE in observed_provenances
            ),
            "any_success_rate_provenance_unknown": (
                UNKNOWN_PROVENANCE in observed_provenances
            ),
            "all_success_rates_are_physics_or_captured_truth": bool(
                observed_provenances
            )
            and observed_provenances.issubset(_PHYSICS_OR_CAPTURED_TRUTH_PROVENANCES),
            "generated_video_vlm_success_rate_is_not_physics_or_captured_truth": True,
            "unknown_provenance_success_rate_is_not_established_truth": True,
            "generated_video_vlm_boundary": (
                SUCCESS_RATE_GENERATED_VIDEO_VLM_CLAIM_BOUNDARY
            ),
        },
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
        attempts=attempts,
        evidence_level=evidence_level,
        task_metadata=task_metadata,
    )
    blockers.extend(
        f"scorecard:{blocker}" for blocker in scorecard.get("blockers") or []
    )
    scorecard_provenance_boundary = _mapping(
        scorecard.get("success_label_provenance_boundary")
    )

    wam_section, wam_blockers = summarize_wam_evaluation_for_report(wam_evaluation)
    blockers.extend(wam_blockers)

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
            "success_rate_provenance_disclosed": True,
            "generated_video_vlm_success_rate_is_not_physics_or_captured_truth": True,
            "all_success_rates_are_physics_or_captured_truth": bool(
                scorecard_provenance_boundary.get(
                    "all_success_rates_are_physics_or_captured_truth"
                )
            ),
            "any_success_rate_from_generated_video_vlm": bool(
                scorecard_provenance_boundary.get(
                    "any_success_rate_from_generated_video_vlm"
                )
            ),
            "safety": dict(SAFETY_CLAIM_BOUNDARY),
        },
    }
