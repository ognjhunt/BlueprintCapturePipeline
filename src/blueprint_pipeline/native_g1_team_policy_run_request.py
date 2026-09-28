"""Admit a team's selected G1 policy profile for one development episode.

This signed WebApp request is only an intent. Operator approval, policy bytes,
site disclosure, paid admission and provider teardown are checked separately
before execution. A registered profile alone never launches a GPU.
"""

from __future__ import annotations

import math
import re
import time
from collections.abc import Mapping
from typing import Any

from .decision_evidence_contracts import cross_runtime_canonical_digest
from .native_g1_team_policy_approval import OBJECTIVES
from .task_evaluation_g1_catalog import G1_PRESET_ID
from .task_evaluation_packet_planning_setup import validate_packet_planning_setup
from .team_policy_delivery_profile import validate_team_policy_delivery_profile


SCHEMA = "native_g1_team_policy_run_request.v1"
_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,191}\Z")
_FIELDS = frozenset({
    "schema_version", "run_id", "owner", "scene_id", "task_id",
    "source_packet_receipt_digest", "source_setup_digest", "robot_preset_id",
    "objective_id", "policy_profile", "authorization",
    "site_observation_exchange_authorized", "claim_ceiling",
    "public_redistribution_authorized", "request_digest",
})
_AUTH_FIELDS = frozenset({
    "maximum_cost_usd", "hard_ttl_seconds", "expires_at_epoch", "retry_cap",
})


def validate_g1_team_policy_run_request(
    value: Mapping[str, Any] | Any,
    *,
    trusted_setup: Mapping[str, Any],
    authenticated_owner: Mapping[str, str],
    now_epoch: float | None = None,
) -> dict[str, Any]:
    """Bind one owner, retained packet, robot interface and delivery profile."""

    setup = validate_packet_planning_setup(trusted_setup)
    if not isinstance(value, Mapping) or set(value) != _FIELDS:
        raise ValueError("g1_team_policy_request_shape_invalid")
    request = dict(value)
    profile = validate_team_policy_delivery_profile(
        request.get("policy_profile"),
        trusted_setup=setup,
        authenticated_owner=authenticated_owner,
    )
    run_id = request.get("run_id")
    if (
        request.get("schema_version") != SCHEMA
        or not isinstance(run_id, str)
        or _RUN_ID.fullmatch(run_id) is None
        or request.get("owner") != dict(authenticated_owner)
        or request.get("scene_id") != setup["scene_id"]
        or request.get("task_id") != setup["task_id"]
        or request.get("source_packet_receipt_digest")
        != setup["source_packet_receipt_digest"]
        or request.get("source_setup_digest") != setup["setup_digest"]
        or request.get("robot_preset_id") != G1_PRESET_ID
        or request.get("robot_preset_id") != profile["robot_preset_id"]
        or request.get("objective_id") not in OBJECTIVES
        or request.get("site_observation_exchange_authorized") is not True
        or request.get("claim_ceiling") != "development_only"
        or request.get("public_redistribution_authorized") is not False
        or request.get("request_digest")
        != cross_runtime_canonical_digest(request, digest_field="request_digest")
    ):
        raise ValueError("g1_team_policy_request_binding_invalid")
    authorization = request.get("authorization")
    now = time.time() if now_epoch is None else now_epoch
    if not isinstance(authorization, dict) or set(authorization) != _AUTH_FIELDS:
        raise ValueError("g1_team_policy_request_authorization_invalid")
    cap = authorization.get("maximum_cost_usd")
    ttl = authorization.get("hard_ttl_seconds")
    expiry = authorization.get("expires_at_epoch")
    if (
        type(cap) not in (int, float) or not math.isfinite(cap) or not 0 < cap <= 12
        or abs(cap * 100 - round(cap * 100)) > 1e-7
        or type(ttl) is not int or not 1800 <= ttl <= 14400
        or type(expiry) not in (int, float) or not math.isfinite(expiry)
        or not now < expiry <= now + 7 * 86400
        or type(authorization.get("retry_cap")) is not int
        or authorization["retry_cap"] != 0
    ):
        raise ValueError("g1_team_policy_request_authorization_invalid")
    return request


__all__ = ["SCHEMA", "validate_g1_team_policy_run_request"]
