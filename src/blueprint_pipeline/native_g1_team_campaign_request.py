"""Admit a robot team's G1 choices against one trusted retained task packet.

This contract is the no-spend boundary for a future signed WebApp submission.
It does not stage a provider bundle, grant model rights, or allocate a GPU.
"""

from __future__ import annotations

import re
import time
from collections.abc import Mapping
from typing import Any

from .decision_evidence_contracts import cross_runtime_canonical_digest
from .native_g1_development_pair import PAIR_ORDER
from .task_evaluation_g1_catalog import G1_PRESET_ID
from .task_evaluation_packet_planning_setup import (
    validate_packet_planning_setup,
    validate_packet_policy_handoff,
)


SCHEMA = "native_g1_team_campaign_request.v1"
FIELDS = frozenset({
    "schema_version", "run_id", "owner", "scene_id", "task_id",
    "source_packet_receipt_digest", "robot_preset_id", "book_handoff",
    "movement_handoff", "authorization", "claim_ceiling",
    "public_redistribution_authorized", "request_digest",
})
AUTHORIZATION_FIELDS = frozenset({
    "maximum_cost_usd", "hard_ttl_seconds", "expires_at_epoch", "retry_cap",
})


def validate_g1_team_campaign_request(
    value: Mapping[str, Any] | Any,
    *,
    trusted_setup: Mapping[str, Any],
    authenticated_owner: Mapping[str, str],
    now_epoch: float | None = None,
) -> dict[str, Any]:
    """Bind both objective pairs and spend bounds to the server's packet view."""

    setup = validate_packet_planning_setup(trusted_setup)
    if not isinstance(value, Mapping) or set(value) != FIELDS:
        raise ValueError("g1_team_campaign_request_shape_invalid")
    request = dict(value)
    owner = request.get("owner")
    if (
        request.get("schema_version") != SCHEMA
        or not isinstance(request.get("run_id"), str)
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,191}", request["run_id"]) is None
        or not isinstance(owner, dict)
        or set(owner) != {"user_id", "organization_id"}
        or any(not isinstance(owner[key], str) or not owner[key] for key in owner)
        or owner != dict(authenticated_owner)
        or request.get("scene_id") != setup["scene_id"]
        or request.get("task_id") != setup["task_id"]
        or request.get("source_packet_receipt_digest")
        != setup["source_packet_receipt_digest"]
        or request.get("robot_preset_id") != G1_PRESET_ID
        or request.get("claim_ceiling") != "development_only"
        or request.get("public_redistribution_authorized") is not False
        or request.get("request_digest")
        != cross_runtime_canonical_digest(request, digest_field="request_digest")
    ):
        raise ValueError("g1_team_campaign_request_binding_invalid")
    authorization = request.get("authorization")
    moment = time.time() if now_epoch is None else now_epoch
    if not isinstance(authorization, dict) or set(authorization) != AUTHORIZATION_FIELDS:
        raise ValueError("g1_team_campaign_authorization_invalid")
    cap = authorization.get("maximum_cost_usd")
    ttl = authorization.get("hard_ttl_seconds")
    expiry = authorization.get("expires_at_epoch")
    if (
        isinstance(cap, bool) or not isinstance(cap, (int, float)) or not 0 < cap <= 12
        or isinstance(ttl, bool) or not isinstance(ttl, int) or not 1800 <= ttl <= 14400
        or isinstance(expiry, bool) or not isinstance(expiry, (int, float))
        or not moment < expiry <= moment + 7 * 86400
        or type(authorization.get("retry_cap")) is not int
        or authorization["retry_cap"] != 0
    ):
        raise ValueError("g1_team_campaign_authorization_invalid")
    book = validate_packet_policy_handoff(request["book_handoff"])
    movement = validate_packet_policy_handoff(request["movement_handoff"])
    if (
        book["setup"] != setup or movement["setup"] != setup
        or book["choice"]["objective_id"] != "task_success"
        or movement["choice"]["objective_id"] != "g1_navigation_goal"
        or [*book["choice"]["policy_candidate_ids"],
            *movement["choice"]["policy_candidate_ids"]] != list(PAIR_ORDER)
    ):
        raise ValueError("g1_team_campaign_policy_or_packet_mismatch")
    return request


__all__ = ["SCHEMA", "validate_g1_team_campaign_request"]
