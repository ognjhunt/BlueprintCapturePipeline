"""Buyer-facing claim ceiling and overclaim copy checks."""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, Mapping, Sequence

from .success_claim_contracts import CLAIM_LADDER


BUYER_CLAIM_CEILING_SCHEMA_VERSION = "buyer_facing_claim_ceiling.v1"

_LIVE_SIMULATOR_ASSERTION_PATTERNS = (
    re.compile(
        r"\blive\s+(?:mujoco|isaac|simulator|simulation)\s+"
        r"(?:execution|run|rollout|episode)s?\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:mujoco|isaac|simulator|simulation)\s+"
        r"(?:execution|run|rollout|episode)s?\s+"
        r"(?:was\s+|is\s+|are\s+)?(?:live|verified|proven)\b",
        re.IGNORECASE,
    ),
)

_LIVE_POLICY_ASSERTION_PATTERNS = (
    re.compile(
        r"\blive\s+(?:robot\s+)?policy\s+"
        r"(?:execution|run|rollout|episode)s?\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:robot\s+)?policy\s+(?:execution|run|rollout|episode)s?\s+"
        r"(?:was\s+|is\s+|are\s+)?(?:live|verified|proven|completed)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bpolicy\s+executed\s+(?:in|inside|through)\s+(?:the\s+)?"
        r"(?:mujoco|isaac|simulator|simulation)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bexecuted\s+(?:the\s+)?(?:customer\s+|claimed\s+|robot\s+)?policy\b",
        re.IGNORECASE,
    ),
)


def _mapping(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _string(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _strict_true(value: Any) -> bool:
    return value is True


def _highest_truthful_claim(
    success_claim_ledger: Mapping[str, Any] | None,
    live_closure: Mapping[str, Any] | None,
) -> str:
    ledger = _mapping(success_claim_ledger)
    closure = _mapping(live_closure)
    candidates = (
        ledger.get("highest_truthful_claim"),
        _mapping(closure.get("success_claim_ledger")).get("highest_truthful_claim"),
        closure.get("highest_truthful_claim"),
    )
    for candidate in candidates:
        claim = _string(candidate)
        if claim in CLAIM_LADDER:
            return claim
    return "no_claim"


def _claim_rank(claim: str) -> int:
    try:
        return CLAIM_LADDER.index(claim)
    except ValueError:
        return 0


def _copy_inputs(value: Any, *, prefix: str = "copy") -> list[dict[str, str]]:
    if isinstance(value, str):
        text = value.strip()
        return [{"source": prefix, "text": text}] if text else []
    if isinstance(value, Mapping):
        rows: list[dict[str, str]] = []
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            rows.extend(_copy_inputs(child, prefix=child_prefix))
        return rows
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
        rows = []
        for index, child in enumerate(value):
            rows.extend(_copy_inputs(child, prefix=f"{prefix}[{index}]"))
        return rows
    return []


def _asserted_sources(
    copy_rows: Iterable[Mapping[str, str]],
    patterns: Sequence[re.Pattern[str]],
) -> list[str]:
    sources: list[str] = []
    for row in copy_rows:
        text = _string(row.get("text"))
        if text and any(pattern.search(text) for pattern in patterns):
            sources.append(_string(row.get("source")) or "copy")
    return sorted(set(sources))


def build_buyer_claim_ceiling(
    *,
    success_claim_ledger: Mapping[str, Any] | None = None,
    proof_boundary: Mapping[str, Any] | None = None,
    live_closure: Mapping[str, Any] | None = None,
    buyer_copy_inputs: Any = None,
) -> Dict[str, Any]:
    """Pin buyer language to the success ledger and live execution gates."""

    boundary = _mapping(proof_boundary)
    closure = _mapping(live_closure)
    closure_boundary = _mapping(closure.get("proof_boundary"))
    highest_truthful_claim = _highest_truthful_claim(success_claim_ledger, closure)
    live_simulator_execution_proven = bool(
        _strict_true(boundary.get("live_simulator_execution_proven"))
        or _strict_true(closure_boundary.get("live_simulator_execution_proven"))
        or _strict_true(closure_boundary.get("simulator_execution_proven"))
    )
    live_policy_execution_proven = bool(
        _strict_true(boundary.get("live_policy_execution_proven"))
        or _strict_true(closure_boundary.get("live_policy_execution_proven"))
        or _strict_true(closure_boundary.get("robot_policy_execution_proven"))
    )
    task_claim_rank = _claim_rank(highest_truthful_claim)
    simulator_task_success_claim_allowed = task_claim_rank >= _claim_rank(
        "simulator_task_success"
    )
    policy_task_success_claim_allowed = task_claim_rank >= _claim_rank(
        "policy_task_success"
    )
    physical_deployment_claim_allowed = task_claim_rank >= _claim_rank(
        "physical_deployment_ready"
    )

    copy_rows = _copy_inputs(buyer_copy_inputs)
    live_simulator_sources = _asserted_sources(
        copy_rows, _LIVE_SIMULATOR_ASSERTION_PATTERNS
    )
    live_policy_sources = _asserted_sources(copy_rows, _LIVE_POLICY_ASSERTION_PATTERNS)

    blockers: list[str] = []
    if live_simulator_sources and not live_simulator_execution_proven:
        blockers.append("buyer_copy_claims_live_simulator_execution_without_live_gate")
    if live_policy_sources and not live_policy_execution_proven:
        blockers.append("buyer_copy_claims_live_policy_execution_without_live_gate")

    allowed_claims = ["review_grade_evaluation_package"]
    if simulator_task_success_claim_allowed:
        allowed_claims.append("simulator_task_success")
    if policy_task_success_claim_allowed:
        allowed_claims.append("policy_task_success")
    if live_simulator_execution_proven:
        allowed_claims.append("live_simulator_execution")
    if live_policy_execution_proven:
        allowed_claims.append("live_policy_execution")
    if physical_deployment_claim_allowed:
        allowed_claims.append("physical_deployment_ready")

    forbidden_claims = []
    if not live_simulator_execution_proven:
        forbidden_claims.append("live_simulator_execution")
    if not live_policy_execution_proven:
        forbidden_claims.append("live_policy_execution")
    if not physical_deployment_claim_allowed:
        forbidden_claims.append("physical_deployment_ready")

    return {
        "schema_version": BUYER_CLAIM_CEILING_SCHEMA_VERSION,
        "status": "passed" if not blockers else "blocked",
        "highest_truthful_claim": highest_truthful_claim,
        "buyer_facing_claim_ceiling_pinned_to_highest_truthful_claim": True,
        "live_simulator_execution_claim_allowed": live_simulator_execution_proven,
        "live_policy_execution_claim_allowed": live_policy_execution_proven,
        "simulator_task_success_claim_allowed": simulator_task_success_claim_allowed,
        "policy_task_success_claim_allowed": policy_task_success_claim_allowed,
        "physical_deployment_claim_allowed": physical_deployment_claim_allowed,
        "allowed_buyer_claims": allowed_claims,
        "forbidden_buyer_claims": forbidden_claims,
        "asserted_claim_sources": {
            "live_simulator_execution": live_simulator_sources,
            "live_policy_execution": live_policy_sources,
        },
        "blockers": sorted(set(blockers)),
    }
