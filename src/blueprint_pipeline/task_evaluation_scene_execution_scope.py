"""Separate site-scene preparation authority from robot-team evaluation authority.

ADP-009B/day-21: preparing a task scene never selects an embodiment or policy.
Legacy evaluation requests retain their frozen-candidate admission unchanged.
"""
from collections.abc import Mapping


def scene_preparation_only(request):
    execution = request.get("execution") if isinstance(request, Mapping) else None
    return isinstance(execution, Mapping) and execution.get("purpose") == "scene_preparation"


def validate_execution(value, *, now):
    from .task_evaluation_scene_intake import _require, _number, _identifier, _DIGEST, SUPPORTED_POLICY_CANDIDATE_IDS
    execution = value.get("execution")
    _require(isinstance(execution, Mapping) and set(execution) - {"purpose"} == {
        "max_total_spend_usd", "max_paid_attempts", "max_retries", "expires_at_epoch",
        "allowed_providers", "policy_candidates", "claim_scope"}, "execution_invalid")
    _require(_number(execution["max_total_spend_usd"])
             and 0 < execution["max_total_spend_usd"] <= 1000, "spend_invalid")
    _require(type(execution["max_paid_attempts"]) is int and 1 <= execution["max_paid_attempts"] <= 32
             and type(execution["max_retries"]) is int and 0 <= execution["max_retries"] <= 3,
             "attempt_bounds_invalid")
    _require(_number(execution["expires_at_epoch"])
             and now < execution["expires_at_epoch"] <= now + 7 * 86400, "authority_expiry_invalid")
    providers = execution["allowed_providers"]
    _require(isinstance(providers, list) and bool(providers)
             and all(isinstance(p, str) and p in {"vast", "runpod", "openai"} for p in providers)
             and len(providers) == len(set(providers)), "providers_invalid")
    _require("purpose" not in execution or execution["purpose"] == "scene_preparation", "execution_purpose_invalid")
    _require(execution["claim_scope"] == "development_only", "claim_scope_invalid")
    policies = execution["policy_candidates"]
    if scene_preparation_only(value):
        _require(policies == [] and "robot_binding_id" not in value["task"], "scene_preparation_must_not_select_robot_or_policy")
        return
    _require(isinstance(policies, list) and len(policies) == 2, "two_policies_required")
    for policy in policies:
        _require(isinstance(policy, Mapping) and set(policy) == {"id", "artifact_digest"}
                 and _identifier(policy["id"]) and isinstance(policy["artifact_digest"], str)
                 and _DIGEST.fullmatch(policy["artifact_digest"]) is not None, "policy_identity_invalid")
    _require(policies[0]["id"] != policies[1]["id"], "two_distinct_policies_required")
    _require([policy["id"] for policy in policies] == list(SUPPORTED_POLICY_CANDIDATE_IDS),
             "policy_candidates_unsupported")
