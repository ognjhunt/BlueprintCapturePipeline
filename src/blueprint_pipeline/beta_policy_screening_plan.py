"""Freeze a bounded, data-selected beta evaluation campaign for one site job.

The catalog describes candidates; trusted runtime capabilities and fresh cost
quotes determine what can be scheduled. This pure planner performs no launch.
It reuses the existing bounded scenario draws under a separate screening seed
domain and never consumes the private scored matrix. The completed campaign is
the job's evaluation result; only spend beyond the site allowance needs the
owner's approval, and no plan can exceed it.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from copy import deepcopy
from decimal import ROUND_CEILING, Decimal, InvalidOperation
from pathlib import PurePosixPath
from typing import Any
from urllib.parse import urlsplit

from .beta_policy_candidate_catalog import CANDIDATE_ID_PATTERN, DIGEST_PATTERN, ID_PATTERN, validate_catalog
from .decision_evidence_contracts import canonical_digest
from .team_evaluation_plan import scored_cells
from .task_family_policy_execution import FAMILY_BACKEND, family_policy_execution_binding
from .beta_policy_screening_g1 import BACKEND as G1_BACKEND, CANDIDATES as G1_CANDIDATES, frozen_cells as g1_cells

PLAN_SCHEMA_VERSION = "beta_policy_screening_plan.v1"
EXECUTION_SCHEMA_VERSION = "blueprint.beta_policy_screening_execution.v1"
QUOTE_SCHEMA_VERSION = "beta_policy_screening_cost_quote.v1"
# Runtime ceiling shared with the native session: beta runs never become Arm
# Decision Proof qualified or physical evidence. It does not limit the job's
# use of its own result, which is published as a simulation estimate.
CLAIM_CEILING = "diagnostic_policy_execution"
SITE_BUDGET_USD = Decimal("15.00")
# A runnable candidate the budget cannot fund yet. Plans built before the
# waitlist list such rows as ``deferred`` with the same reason and no
# execution fields; those stay valid and are never funded.
WAITLISTED = "waitlisted"
WAITLIST_BLOCKER = "site_screening_budget_insufficient"
DEFAULT_EPISODES = 10
_DIGEST = re.compile(DIGEST_PATTERN)
_ID = re.compile(ID_PATTERN)
_CANDIDATE_ID = re.compile(CANDIDATE_ID_PATTERN)
_MONEY = re.compile(r"^(?:0|[1-9][0-9]{0,8})\.[0-9]{2}$")
SOURCE_FIELDS = ("scene_intent_id", "scene_intent_digest", "capture_id", "task_id",
                 "task_context_digest", "scene_revision_digest")
_RECEIPT_KEYS = {"path", "sha256", "size_bytes"}
_HOSTED_COST_FIELDS = {"provider_hard_cap_usd", "hosted_inference_maximum_usd", "hosted_inference_payment",
                       "hosted_max_cost_per_request_usd", "hosted_http_request_limit", "credential_origin"}
_UNIT_PRICE = re.compile(r"^(?:0|[1-9][0-9]{0,8})(?:\.[0-9]{1,12})?$")
_ROOT_POSE_PARAMETERS = {"object_start_x_delta_m", "object_start_y_delta_m", "object_yaw_delta_degrees"}


class BetaPolicyScreeningPlanError(ValueError):
    """Stable refusal for malformed or contradictory screening plans."""


def money(value: Any) -> Decimal:
    if not isinstance(value, str) or not _MONEY.fullmatch(value):
        raise BetaPolicyScreeningPlanError("beta_screening_money_invalid")
    try:
        result = Decimal(value)
    except InvalidOperation as exc:
        raise BetaPolicyScreeningPlanError("beta_screening_money_invalid") from exc
    return result


def _digest(value: Any, code: str) -> str:
    if not isinstance(value, str) or not _DIGEST.fullmatch(value):
        raise BetaPolicyScreeningPlanError(code)
    return value


def _id(value: Any, code: str) -> str:
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise BetaPolicyScreeningPlanError(code)
    return value


def _json(value: Any) -> dict[str, Any]:
    try:
        result = json.loads(json.dumps(value, allow_nan=False))
    except (ValueError, TypeError) as exc:
        raise BetaPolicyScreeningPlanError("beta_screening_not_json") from exc
    if not isinstance(result, dict):
        raise BetaPolicyScreeningPlanError("beta_screening_not_mapping")
    return result


def _receipt(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _RECEIPT_KEYS:
        raise BetaPolicyScreeningPlanError("beta_screening_artifact_reference_invalid")
    path = value.get("path")
    if not isinstance(path, str) or not PurePosixPath(path).is_absolute() or ".." in PurePosixPath(path).parts:
        raise BetaPolicyScreeningPlanError("beta_screening_artifact_reference_invalid")
    _digest(value.get("sha256"), "beta_screening_artifact_reference_invalid")
    if type(value.get("size_bytes")) is not int or value["size_bytes"] <= 0:
        raise BetaPolicyScreeningPlanError("beta_screening_artifact_reference_invalid")
    return dict(value)


def validate_execution_descriptor(value: Mapping[str, Any], *, candidate_id: str,
                                  configuration_digest: str) -> dict[str, Any]:
    """Check the non-executable launch envelope; the backend verifies its files.

    Only immutable artifact references and resource bounds are accepted here.
    A customer cannot supply an import target, command or alternate launcher.
    """
    result = _json(value)
    required = {"schema_version", "candidate_id", "policy_execution_spec", "checkpoint_inventory",
                "configuration_digest", "runtime_digest", "backend", "max_hourly_rate_usd",
                "hard_ttl_seconds", "implementation_commit", "task_success_contract_digest"}
    if (set(result) - required - {"machine_avoidlist_path", "credential_file", "team_robot_model", "minimum_gpu_memory_gb"} - _HOSTED_COST_FIELDS
            or required - set(result)
            or result["schema_version"] != EXECUTION_SCHEMA_VERSION
            or result["backend"] not in {"robolab.v1", FAMILY_BACKEND, G1_BACKEND}
            or result["candidate_id"] != candidate_id
            or result["configuration_digest"] != configuration_digest):
        raise BetaPolicyScreeningPlanError("beta_screening_execution_binding_invalid")
    _digest(result["runtime_digest"], "beta_screening_execution_runtime_invalid")
    _digest(result["task_success_contract_digest"], "beta_screening_execution_task_invalid")
    _receipt(result["policy_execution_spec"])
    if result["checkpoint_inventory"] is not None:
        _receipt(result["checkpoint_inventory"])
    if "team_robot_model" in result:
        _receipt(result["team_robot_model"])
    if "minimum_gpu_memory_gb" in result:
        memory = result["minimum_gpu_memory_gb"]
        if type(memory) not in (int, float) or not 0 < memory <= 1024:
            raise BetaPolicyScreeningPlanError("beta_screening_gpu_memory_bound_invalid")
    if money(result["max_hourly_rate_usd"]) <= 0:
        raise BetaPolicyScreeningPlanError("beta_screening_execution_price_invalid")
    if type(result["hard_ttl_seconds"]) is not int or not 1 <= result["hard_ttl_seconds"] <= 86400:
        raise BetaPolicyScreeningPlanError("beta_screening_execution_ttl_invalid")
    if not re.fullmatch(r"[0-9a-f]{40}", str(result["implementation_commit"])):
        raise BetaPolicyScreeningPlanError("beta_screening_implementation_commit_invalid")
    for key in ("machine_avoidlist_path", "credential_file"):
        if key not in result:
            continue
        path = result[key]
        if not isinstance(path, str) or not PurePosixPath(path).is_absolute() or ".." in PurePosixPath(path).parts:
            raise BetaPolicyScreeningPlanError("beta_screening_" + key + "_invalid")
    if "credential_file" in result or set(result) & _HOSTED_COST_FIELDS:
        if not _HOSTED_COST_FIELDS <= set(result):
            raise BetaPolicyScreeningPlanError("beta_screening_hosted_cost_binding_missing")
        origin = result["credential_origin"]
        try:
            if not isinstance(origin, str) or any(ord(char) < 33 or ord(char) == 127 for char in origin):
                raise ValueError("invalid origin")
            parsed = urlsplit(origin)
            if (parsed.scheme != "https" or not parsed.hostname or parsed.username is not None
                    or parsed.password is not None or parsed.path not in {"", "/"}
                    or parsed.query or parsed.fragment or parsed.port == 0):
                raise ValueError("invalid origin")
        except ValueError as exc:
            raise BetaPolicyScreeningPlanError("beta_screening_hosted_origin_invalid") from exc
        provider = money(result["provider_hard_cap_usd"])
        inference = money(result["hosted_inference_maximum_usd"])
        payer = result["hosted_inference_payment"]
        price_text = result["hosted_max_cost_per_request_usd"]
        limit = result["hosted_http_request_limit"]
        if (not isinstance(price_text, str) or not _UNIT_PRICE.fullmatch(price_text)
                or type(limit) is not int or not 1 <= limit <= 10_000_000):
            raise BetaPolicyScreeningPlanError("beta_screening_hosted_request_bound_invalid")
        price = Decimal(price_text)
        api_bound = (price * Decimal(limit)).quantize(Decimal("0.01"), rounding=ROUND_CEILING)
        if (payer not in {"blueprint", "caller_paid", "free"}
                or (payer == "blueprint" and (inference <= 0 or price <= 0))
                or (payer in {"caller_paid", "free"} and (inference != 0 or price_text != "0"))
                or inference < api_bound
                or provider < _compute_cost_floor(result)):
            raise BetaPolicyScreeningPlanError("beta_screening_hosted_cost_binding_invalid")
    return result


def _compute_cost_floor(spec: Mapping[str, Any]) -> Decimal:
    return (money(spec["max_hourly_rate_usd"]) * Decimal(spec["hard_ttl_seconds"]) / Decimal(3600)).quantize(
        Decimal("0.01"), rounding=ROUND_CEILING)


def execution_cost_floor(spec: Mapping[str, Any]) -> Decimal:
    """Minimum reservation for an already validated local or hosted descriptor."""
    if _HOSTED_COST_FIELDS <= set(spec):
        return money(spec["provider_hard_cap_usd"]) + money(spec["hosted_inference_maximum_usd"])
    return _compute_cost_floor(spec)


def validate_cost_quote(value: Mapping[str, Any], *, configuration_digest: str,
                        scene_revision_digest: str, episode_count: int,
                        now: dt.datetime | None = None) -> dict[str, Any]:
    """Validate a bound quote; pass now before planning/reserving, omit for history."""
    result = _json(value)
    fields = {"schema_version", "quote_id", "price_source", "valid_until", "configuration_digest",
              "scene_revision_digest", "episode_count", "maximum_cost_usd", "quote_digest"}
    if (set(result) != fields or result["schema_version"] != QUOTE_SCHEMA_VERSION
            or result["configuration_digest"] != configuration_digest
            or result["scene_revision_digest"] != scene_revision_digest
            or type(result["episode_count"]) is not int or result["episode_count"] != episode_count):
        raise BetaPolicyScreeningPlanError("beta_screening_cost_quote_binding_invalid")
    _id(result["quote_id"], "beta_screening_cost_quote_id_invalid")
    if not isinstance(result["price_source"], str) or not result["price_source"].strip():
        raise BetaPolicyScreeningPlanError("beta_screening_cost_quote_source_missing")
    if money(result["maximum_cost_usd"]) <= 0:
        raise BetaPolicyScreeningPlanError("beta_screening_cost_quote_price_invalid")
    try:
        expiry = dt.datetime.fromisoformat(str(result["valid_until"]).replace("Z", "+00:00"))
        if expiry.tzinfo is None or (now is not None and now.tzinfo is None):
            raise ValueError("timezone required")
    except (ValueError, TypeError) as exc:
        raise BetaPolicyScreeningPlanError("beta_screening_cost_quote_expiry_invalid") from exc
    if now is not None and expiry <= now:
        raise BetaPolicyScreeningPlanError("beta_screening_cost_quote_expired")
    if result["quote_digest"] != canonical_digest(result, digest_field="quote_digest"):
        raise BetaPolicyScreeningPlanError("beta_screening_cost_quote_digest_mismatch")
    return result


def _native_scene_plan(scene: Mapping[str, Any]) -> Mapping[str, Any]:
    plan = scene.get("native_scene_plan")
    if scene.get("evaluation_backend") == FAMILY_BACKEND:
        if (not isinstance(plan, Mapping) or plan.get("task_kind") != scene.get("task_kind")
                or plan.get("plan_digest") != canonical_digest(plan, digest_field="plan_digest")):
            raise BetaPolicyScreeningPlanError("beta_screening_native_family_scene_plan_invalid")
        try:
            family_policy_execution_binding(plan)
        except (ValueError, KeyError, TypeError) as exc:
            raise BetaPolicyScreeningPlanError("beta_screening_native_family_scene_plan_invalid") from exc
        if (plan.get("task_spec") != scene.get("task_spec")
                or plan["plan_digest"] != scene.get("scene_revision_digest")
                or plan["native_runtime_contract"]["task_spec"]["task_success_contract"]["scope"] != {
                    "scene_id": scene.get("scene_id"), "task_id": scene.get("task_id"),
                    "task_context_digest": scene.get("task_context_digest")}
                or plan["native_runtime_contract"]["task_spec"]["task_success_contract"]["contract_digest"]
                != scene.get("task_success_contract_digest")):
            raise BetaPolicyScreeningPlanError("beta_screening_native_family_scene_plan_invalid")
        return plan
    if (not isinstance(plan, Mapping) or plan.get("task_kind") != scene.get("task_kind")
            or plan.get("plan_digest") != canonical_digest(plan, digest_field="plan_digest")
            or not isinstance(plan.get("objects"), list)
            or any(not isinstance(row, Mapping) for row in plan["objects"])
            or sum(row.get("task_subject") is True for row in plan["objects"]) != 1):
        raise BetaPolicyScreeningPlanError("beta_screening_native_scene_plan_invalid")
    return plan


def screening_cells(scene: Mapping[str, Any], count: int = DEFAULT_EPISODES) -> list[dict[str, Any]]:
    if type(count) is not int or not 1 <= count <= DEFAULT_EPISODES:
        raise BetaPolicyScreeningPlanError("beta_screening_episode_count_invalid")
    from .policy_canary_scene_resolution import constrain_diagnostic_scenario

    native_scene = _native_scene_plan(scene)
    # Domain separation is independent of catalog order and candidate count.
    # This public screening salt is never the private scored-200 salt.
    seed_material = "beta_policy_screening.v1:" + scene["scene_revision_digest"] + ":" + scene["task_context_digest"]
    salt = hashlib.sha256(seed_material.encode()).hexdigest()
    if scene.get("evaluation_backend") == FAMILY_BACKEND:
        # These are matched, frozen-reset repetitions. Do not advertise the
        # rigid lane's placement/lighting perturbations for engines that have
        # not qualified their application to this family's native assets.
        cells = []
        for ordinal in range(count):
            scenario = {"family": "canonical_anchor", "ordinal": ordinal, "parameters": {}}
            digest = canonical_digest(scenario)
            seed = int(hashlib.sha256(f"{salt}:{ordinal}".encode()).hexdigest()[:8], 16) % 2**31
            cells.append({"cell_id": f"screening.{scene['scene_id']}.{ordinal:03d}", "seed": seed,
                "family": "canonical_anchor", "partition": "screening", "resolved_scenario": scenario,
                "cell_spec_digest": digest, "resolved_scenario_digest": digest})
        return cells
    draws = scored_cells(scene_revision_digest=scene["scene_revision_digest"], scene_id=scene["scene_id"],
                         salt=salt, task_spec=scene["task_spec"], task_kind=scene["task_kind"],
                         nominal_dynamic_friction=scene.get("nominal_dynamic_friction"), count=max(10, count))
    cells = []
    for ordinal, draw in enumerate(draws[:count]):
        row = deepcopy(draw)
        row["cell_id"] = f"screening.{scene['scene_id']}.{ordinal:03d}"
        row["partition"] = "screening"
        if row["family"] == "held_out_composition":
            row["family"] = "composed_stress"
            row["resolved_scenario"]["family"] = "composed_stress"
        # Installed task subjects cannot be moved at their roots. Preserve the
        # scientific draw as an explicit unexercised-dimension receipt rather
        # than claiming that a repeated anchor tested placement robustness.
        row["resolved_scenario"] = constrain_diagnostic_scenario(native_scene, row["resolved_scenario"])
        row["cell_spec_digest"] = canonical_digest(row["resolved_scenario"])
        row["resolved_scenario_digest"] = row["cell_spec_digest"]
        cells.append(row)
    return cells


def _scene(value: Mapping[str, Any]) -> dict[str, Any]:
    scene = _json(value)
    for key in ("site_id", "scene_id", "capture_id", "task_id", "scene_intent_id", "task_kind"):
        _id(scene.get(key), "beta_screening_scene_" + key + "_missing")
    family = scene.get("evaluation_backend") == FAMILY_BACKEND
    for key in ("scene_revision_digest", "scene_intent_digest", "task_context_digest",
                "task_success_contract_digest"):
        _digest(scene.get(key), "beta_screening_scene_" + key + "_missing")
    if family:
        _digest(scene.get("family_static_readiness_digest"), "beta_screening_family_readiness_missing")
        if (scene.get("pre_policy_site_qualification_required") is not True
                or scene.get("scenario_scope") != "native_frozen_reset_seeds"
                or "robot_free_probe_digest" in scene or "robot_free_probe_status" in scene):
            raise BetaPolicyScreeningPlanError("beta_screening_family_native_qualification_required")
    else:
        _digest(scene.get("robot_free_probe_digest"), "beta_screening_scene_robot_free_probe_digest_missing")
        if scene.get("robot_free_probe_status") != "passed":
            raise BetaPolicyScreeningPlanError("beta_screening_robot_free_probe_not_passed")
    if not isinstance(scene.get("task_spec"), dict) or scene["task_spec"].get("task_kind") != scene["task_kind"]:
        raise BetaPolicyScreeningPlanError("beta_screening_task_spec_binding_invalid")
    _receipt(scene.get("scene_inputs"))
    _native_scene_plan(scene)
    # Reference to the existing sealed scene half; the launch backend reuses
    # read_team_scene_inputs to verify its contents and constituent artifacts.
    return scene


def _validate_coverage_gaps(scenario: Mapping[str, Any]) -> None:
    if "runtime_coverage_gaps" not in scenario:
        return
    gaps = scenario["runtime_coverage_gaps"]
    if not isinstance(gaps, list) or len(gaps) != 1:
        raise BetaPolicyScreeningPlanError("beta_screening_coverage_gap_invalid")
    gap = gaps[0]
    if (not isinstance(gap, Mapping) or set(gap) != {"family", "reason", "parameters_not_exercised",
            "fallback", "qualified_variation_claim_permitted"}
            or gap["family"] != scenario["family"] or gap["reason"] != "task_subject_root_fixed_in_scene"
            or gap["fallback"] != "canonical_task_root" or gap["qualified_variation_claim_permitted"] is not False):
        raise BetaPolicyScreeningPlanError("beta_screening_coverage_gap_invalid")
    omitted = gap["parameters_not_exercised"]
    if (not isinstance(omitted, Mapping) or not omitted or set(omitted) - _ROOT_POSE_PARAMETERS
            or set(omitted) & set(scenario["parameters"])
            or any(isinstance(number, bool) or not isinstance(number, (int, float)) or number == 0
                   for number in omitted.values())):
        raise BetaPolicyScreeningPlanError("beta_screening_coverage_gap_invalid")


def _candidate_blockers(candidate: Mapping[str, Any], scene: Mapping[str, Any],
                        binding: Mapping[str, Any] | None, credentials: set[str]) -> list[str]:
    blockers = list(candidate["eligibility"]["blockers"])
    if not candidate["eligibility"]["enabled"]:
        blockers.append("candidate_disabled")
    if candidate["eligibility"]["commercial_use"] not in {"allowed", "allowed_with_conditions"}:
        blockers.append("commercial_use_not_admitted")
    pinned_runtime = (candidate["policy"]["digest_kind"] == "pinned_runtime_file_inventory"
                      and (candidate.get("interface") or {}).get("policy_interface_id") in {"openwam_piper_eef20.v1", "openwam_franka_eef10.v1", "molmoact2_yam_joint14.v1", "pi05_trlc_joint14.v1", "molmoact2_droid_joint8.v1", "cosmos_nano_droid.v1", "humanoidarena_semantic_v3"})
    pinned_runtime = pinned_runtime or (candidate["policy"]["kind"] == "scripted"
        and candidate["policy"]["digest_kind"] == "source_code_inventory"
        and (candidate.get("interface") or {}).get("policy_interface_id") == "blueprint_scripted_pick_place.v1")
    if candidate["policy"]["digest_kind"] not in {"downloaded_snapshot_inventory", "provider_metadata"} and not pinned_runtime:
        blockers.append("policy_identity_not_verified")
    if not candidate["policy"]["digest"] or not candidate["policy"]["revision"]:
        blockers.append("policy_identity_not_pinned")
    if scene["task_kind"] not in candidate["supported_task_kinds"]:
        blockers.append("task_kind_not_supported_by_candidate" if candidate["supported_task_kinds"]
                        else "task_kind_support_unverified")
    for credential in candidate["eligibility"]["requires_credentials"]:
        if credential not in credentials:
            blockers.append("credential_unavailable:" + credential)
    if not isinstance(binding, Mapping):
        blockers.append("runtime_candidate_binding_missing")
        return sorted(set(blockers))
    try:
        spec = validate_execution_descriptor(binding.get("execution_spec", {}),
                                             candidate_id=candidate["candidate_id"],
                                             configuration_digest=candidate["configuration_digest"])
        if (binding.get("configuration_digest") != candidate["configuration_digest"]
                or binding.get("runtime_digest") != spec["runtime_digest"]
                or binding.get("backend") != spec["backend"]
                or binding.get("admission_digest") != canonical_digest(spec)
                or spec["task_success_contract_digest"] != scene["task_success_contract_digest"]
                or (spec["backend"] == FAMILY_BACKEND) is not (scene.get("evaluation_backend") == FAMILY_BACKEND)
                or (candidate["runtime"] or {}).get("backend") != spec["backend"]):
            blockers.append("runtime_candidate_binding_mismatch")
    except BetaPolicyScreeningPlanError as exc:
        blockers.append(str(exc))
    return sorted(set(blockers))


def build_screening_plan(catalog: Mapping[str, Any], scene: Mapping[str, Any], *, campaign_id: str,
                         candidate_ids: Sequence[str] | None = None, budget_usd: str = "15.00",
                         episodes_per_candidate: int = DEFAULT_EPISODES,
                         capabilities: Mapping[str, Any] | None = None,
                         cost_quotes: Mapping[str, Any] | None = None,
                         now: dt.datetime | None = None,
                         priority: Sequence[str] | None = None) -> dict[str, Any]:
    """Every requested candidate gets a status and reason; the budget decides how many run.

    ``priority`` (the job's intake ranking) sets the order the budget is spent
    in; unranked candidates follow by identifier, learned policies before
    scripted baselines, which are reference only and never chosen (a site's few
    funded slots must not all go to them). Capabilities may name measured
    ``candidate_infeasible`` refusals, which stay listed as ``infeasible``.
    Runnable candidates the budget cannot fund yet are ``waitlisted``, in the
    same order, for money a selected candidate releases before spend.
    """
    catalog = validate_catalog(catalog)
    scene = _scene(scene)
    _id(campaign_id, "beta_screening_campaign_id_invalid")
    budget = money(budget_usd)
    if budget <= 0 or budget > SITE_BUDGET_USD:
        raise BetaPolicyScreeningPlanError("beta_screening_site_budget_exceeded")
    capabilities = _json(capabilities or {})
    bindings = capabilities.get("candidate_bindings") or {}
    if not isinstance(bindings, dict):
        raise BetaPolicyScreeningPlanError("beta_screening_capabilities_invalid")
    failures = capabilities.get("candidate_blockers") or {}
    infeasible = capabilities.get("candidate_infeasible") or {}
    for value, reason_required in ((failures, False), (infeasible, True)):
        if (not isinstance(value, dict) or any(not isinstance(reasons, list) or (reason_required and not reasons)
                or any(not isinstance(reason, str) or not re.fullmatch(r"[a-z][a-z0-9_]{1,127}", reason)
                       for reason in reasons) for reasons in value.values())):
            raise BetaPolicyScreeningPlanError("beta_screening_capabilities_invalid")
    available = capabilities.get("credentials_available", [])
    if not isinstance(available, list) or any(not isinstance(name, str) for name in available):
        raise BetaPolicyScreeningPlanError("beta_screening_capabilities_invalid")
    credentials = set(available)
    cost_quotes = _json(cost_quotes or {})
    current = now or dt.datetime.now(dt.timezone.utc)
    by_id = {row["candidate_id"]: row for row in catalog["candidates"]}
    if isinstance(candidate_ids, (str, bytes)):
        raise BetaPolicyScreeningPlanError("beta_screening_candidate_selection_invalid")
    requested = list(candidate_ids) if candidate_ids is not None else sorted(by_id)
    if (not requested or any(not isinstance(row, str) or not _CANDIDATE_ID.fullmatch(row) for row in requested)
            or len(set(requested)) != len(requested)):
        raise BetaPolicyScreeningPlanError("beta_screening_candidate_selection_invalid")
    if set(requested) - set(by_id):
        raise BetaPolicyScreeningPlanError("beta_screening_candidate_not_in_catalog")
    if isinstance(priority, (str, bytes)) or any(not isinstance(row, str) for row in priority or ()):
        raise BetaPolicyScreeningPlanError("beta_screening_candidate_priority_invalid")
    ranked = [row for row in dict.fromkeys(priority or ()) if row in set(requested)]
    cells = screening_cells(scene, episodes_per_candidate)
    remaining = budget
    candidates = []
    unranked = sorted(set(requested) - set(ranked),
                      key=lambda row: (by_id[row]["policy"]["kind"] == "scripted", row))
    for candidate_id in ranked + unranked:
        candidate = by_id[candidate_id]
        row: dict[str, Any] = {"candidate_id": candidate_id,
                               "display_name": candidate["display_name"],
                               "configuration_digest": candidate["configuration_digest"],
                               "status": "unavailable", "blockers": []}
        if candidate["policy"]["kind"] == "scripted":
            row["execution_evidence_kind"] = "scripted_simulator_state_baseline"
        elif candidate_id in G1_CANDIDATES:
            row["execution_evidence_kind"] = "learned_policy"
        if candidate_id in infeasible:
            # Measured on this scene (station reach); listed with its reason, never quoted.
            row.update(status="infeasible", blockers=sorted(set(infeasible[candidate_id])))
            candidates.append(row)
            continue
        binding = bindings.get(candidate_id)
        row["blockers"] = _candidate_blockers(candidate, scene, binding, credentials)
        row["blockers"] = sorted(set(row["blockers"] + failures.get(candidate_id, [])))
        if row["blockers"]:
            candidates.append(row)
            continue
        try:
            quote = validate_cost_quote(cost_quotes.get(candidate_id, {}),
                                         configuration_digest=candidate["configuration_digest"],
                                         scene_revision_digest=scene["scene_revision_digest"],
                                         episode_count=episodes_per_candidate, now=current)
        except BetaPolicyScreeningPlanError as exc:
            row.update(status="deferred", blockers=[str(exc)])
            candidates.append(row)
            continue
        maximum = money(quote["maximum_cost_usd"])
        spec = binding["execution_spec"]
        if maximum < execution_cost_floor(spec):
            row.update(status="deferred", blockers=["cost_quote_below_resource_bound"])
            candidates.append(row)
            continue
        frozen = {"admission_digest": binding["admission_digest"],
                  "execution_spec": deepcopy(binding["execution_spec"]), "cost_quote": quote,
                  "max_cost_usd": quote["maximum_cost_usd"], "episode_count": episodes_per_candidate,
                  "scenario_cells": g1_cells(cells) if spec["backend"] == G1_BACKEND else deepcopy(cells)}
        if maximum > budget:
            # More than this plan could ever fund, even with nothing else selected.
            row.update(status="deferred", blockers=[WAITLIST_BLOCKER])
            candidates.append(row)
            continue
        if maximum > remaining:
            # Runnable but unfunded for now. It is frozen like a selected row,
            # so the coordinator can fund it from money a selected candidate
            # releases before spend, within the same site and plan budget.
            row.update(status=WAITLISTED, blockers=[WAITLIST_BLOCKER], **frozen)
            candidates.append(row)
            continue
        row.update(status="selected", **frozen)
        remaining -= maximum
        candidates.append(row)
    plan = {
        "schema_version": PLAN_SCHEMA_VERSION, "campaign_id": campaign_id, "site_id": scene["site_id"],
        "scene_id": scene["scene_id"], "capture_id": scene["capture_id"], "task_id": scene["task_id"],
        "scene_revision_digest": scene["scene_revision_digest"],
        "task_success_contract_digest": scene["task_success_contract_digest"],
        **({"evaluation_backend": FAMILY_BACKEND, "pre_policy_site_qualification_required": True,
            "family_static_readiness_digest": scene["family_static_readiness_digest"],
            "scenario_scope": "native_frozen_reset_seeds"} if scene.get("evaluation_backend") == FAMILY_BACKEND
           else {"robot_free_probe_digest": scene["robot_free_probe_digest"]}),
        "source": {field: scene[field] for field in SOURCE_FIELDS},
        "scene_inputs": deepcopy(scene["scene_inputs"]), "catalog_digest": catalog["catalog_digest"],
        "scenario_plan_digest": canonical_digest({"cells": cells}), "scenario_cells": cells,
        "budget_usd": budget_usd, "maximum_planned_cost_usd": f"{budget - remaining:.2f}",
        "purpose": "screening", "claim_ceiling": CLAIM_CEILING,
        "candidates": candidates, "plan_digest": "",
    }
    plan["plan_digest"] = canonical_digest(plan, digest_field="plan_digest")
    return validate_screening_plan(plan)


def validate_screening_plan(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate retained plans without expiring their historical cost evidence."""
    plan = _json(value)
    family = plan.get("evaluation_backend") == FAMILY_BACKEND
    required = {"schema_version", "campaign_id", "site_id", "scene_id", "capture_id", "task_id",
                "scene_revision_digest", "task_success_contract_digest", "source",
                "scene_inputs", "catalog_digest", "scenario_plan_digest", "scenario_cells", "budget_usd",
                "maximum_planned_cost_usd", "purpose", "claim_ceiling", "candidates", "plan_digest"}
    required |= ({"evaluation_backend", "pre_policy_site_qualification_required",
                  "family_static_readiness_digest", "scenario_scope"} if family else {"robot_free_probe_digest"})
    # Plans enqueued before the single-tier change carry a digest-bound
    # ``recommendation_eligible: false``; they stay valid and publish alike.
    if (set(plan) - {"recommendation_eligible"} != required or plan["schema_version"] != PLAN_SCHEMA_VERSION
            or plan["purpose"] != "screening" or plan["claim_ceiling"] != CLAIM_CEILING
            or plan.get("recommendation_eligible", False) is not False):
        raise BetaPolicyScreeningPlanError("beta_screening_plan_schema_invalid")
    for key in ("campaign_id", "site_id", "scene_id", "capture_id", "task_id"):
        _id(plan[key], "beta_screening_plan_identity_invalid")
    for key in ("scene_revision_digest", "task_success_contract_digest", "catalog_digest"):
        _digest(plan[key], "beta_screening_plan_digest_invalid")
    if family:
        _digest(plan["family_static_readiness_digest"], "beta_screening_family_readiness_missing")
        if (plan["pre_policy_site_qualification_required"] is not True
                or plan["scenario_scope"] != "native_frozen_reset_seeds"):
            raise BetaPolicyScreeningPlanError("beta_screening_family_native_qualification_required")
    else:
        _digest(plan["robot_free_probe_digest"], "beta_screening_plan_digest_invalid")
    source = plan["source"]
    if not isinstance(source, dict) or set(source) != set(SOURCE_FIELDS):
        raise BetaPolicyScreeningPlanError("beta_screening_source_invalid")
    for key in SOURCE_FIELDS:
        (_digest if key.endswith("digest") else _id)(source[key], "beta_screening_source_invalid")
    if any(source[key] != plan[key] for key in ("capture_id", "task_id", "scene_revision_digest")):
        raise BetaPolicyScreeningPlanError("beta_screening_source_mismatch")
    budget = money(plan["budget_usd"])
    if not 0 < budget <= SITE_BUDGET_USD:
        raise BetaPolicyScreeningPlanError("beta_screening_site_budget_exceeded")
    cells = plan["scenario_cells"]
    _receipt(plan["scene_inputs"])
    if not isinstance(cells, list) or not 1 <= len(cells) <= DEFAULT_EPISODES:
        raise BetaPolicyScreeningPlanError("beta_screening_cells_invalid")
    ids, seeds = [], []
    cell_fields = {"cell_id", "seed", "family", "partition", "resolved_scenario", "cell_spec_digest",
                   "resolved_scenario_digest"}
    for ordinal, cell in enumerate(cells):
        if not isinstance(cell, dict) or not isinstance(cell.get("resolved_scenario"), dict):
            raise BetaPolicyScreeningPlanError("beta_screening_cell_binding_invalid")
        if (not isinstance(cell, dict) or cell.get("partition") != "screening"
                or set(cell) != cell_fields
                or not isinstance(cell.get("cell_id"), str) or not cell["cell_id"].startswith("screening.")
                or type(cell.get("seed")) is not int or not 0 <= cell["seed"] < 2**31
                or cell.get("cell_spec_digest") != canonical_digest(cell.get("resolved_scenario", {}))
                or cell.get("resolved_scenario_digest") != cell["cell_spec_digest"]):
            raise BetaPolicyScreeningPlanError("beta_screening_cell_binding_invalid")
        scenario = cell["resolved_scenario"]
        if (not isinstance(scenario, dict) or set(scenario) - {"family", "ordinal", "parameters", "runtime_coverage_gaps"}
                or not {"family", "ordinal", "parameters"} <= set(scenario)
                or type(scenario["ordinal"]) is not int or scenario["ordinal"] != ordinal
                or scenario["family"] != cell["family"]
                or cell["family"] not in {"canonical_anchor", "placement_approach", "illumination",
                                          "camera_sensor", "bounded_physics", "pairwise_stress", "composed_stress"}
                or not isinstance(scenario["parameters"], dict)
                or set(scenario["parameters"]) - {"object_start_x_delta_m", "object_start_y_delta_m",
                    "object_yaw_delta_degrees", "task_light_intensity_scale", "external_camera_x_delta_m", "dynamic_friction"}
                or any(isinstance(number, bool) or not isinstance(number, (int, float))
                       for number in scenario["parameters"].values())):
            raise BetaPolicyScreeningPlanError("beta_screening_scenario_parameters_invalid")
        _validate_coverage_gaps(scenario)
        if family and (cell["family"] != "canonical_anchor" or scenario["parameters"]
                       or "runtime_coverage_gaps" in scenario):
            raise BetaPolicyScreeningPlanError("beta_screening_family_scenario_not_qualified")
        ids.append(cell["cell_id"])
        seeds.append(cell["seed"])
    if (len(set(ids)) != len(ids) or len(set(seeds)) != len(seeds)
            or canonical_digest({"cells": cells}) != plan["scenario_plan_digest"]):
        raise BetaPolicyScreeningPlanError("beta_screening_scenario_plan_invalid")
    rows = plan["candidates"]
    if not isinstance(rows, list) or not 1 <= len(rows) <= 256:
        raise BetaPolicyScreeningPlanError("beta_screening_candidates_invalid")
    candidates, committed = [], Decimal("0.00")
    for row in rows:
        if not isinstance(row, dict) or not {"candidate_id", "display_name", "configuration_digest", "status", "blockers"} <= set(row):
            raise BetaPolicyScreeningPlanError("beta_screening_candidate_invalid")
        candidates.append(_id(row["candidate_id"], "beta_screening_candidate_invalid"))
        evidence_fields = {"execution_evidence_kind"} if "execution_evidence_kind" in row else set()
        if evidence_fields and row["execution_evidence_kind"] != "scripted_simulator_state_baseline" and not (
                row["candidate_id"] in G1_CANDIDATES and row["execution_evidence_kind"] == "learned_policy"):
            raise BetaPolicyScreeningPlanError("beta_screening_evidence_kind_invalid")
        if not _CANDIDATE_ID.fullmatch(row["candidate_id"]):
            raise BetaPolicyScreeningPlanError("beta_screening_candidate_invalid")
        _digest(row["configuration_digest"], "beta_screening_configuration_invalid")
        if (not isinstance(row["display_name"], str) or not 1 <= len(row["display_name"]) <= 200
                or row["display_name"] != row["display_name"].strip()):
            raise BetaPolicyScreeningPlanError("beta_screening_candidate_name_invalid")
        if (not isinstance(row["blockers"], list) or len(row["blockers"]) > 100
                or any(not isinstance(reason, str) or not 1 <= len(reason) <= 300 for reason in row["blockers"])):
            raise BetaPolicyScreeningPlanError("beta_screening_candidate_blockers_invalid")
        if row["status"] in {"selected", WAITLISTED}:
            if set(row) - evidence_fields != {"candidate_id", "display_name", "configuration_digest", "status", "blockers", "admission_digest",
                            "execution_spec", "cost_quote", "max_cost_usd", "episode_count", "scenario_cells"}:
                raise BetaPolicyScreeningPlanError("beta_screening_selected_fields_invalid")
            expected_cells = g1_cells(cells) if row.get("execution_spec", {}).get("backend") == G1_BACKEND else cells
            if (row["blockers"] != ([] if row["status"] == "selected" else [WAITLIST_BLOCKER])
                    or row.get("episode_count") != len(cells) or row.get("scenario_cells") != expected_cells):
                raise BetaPolicyScreeningPlanError("beta_screening_selected_cells_invalid")
            _digest(row["configuration_digest"], "beta_screening_configuration_invalid")
            spec = validate_execution_descriptor(row.get("execution_spec", {}), candidate_id=row["candidate_id"],
                                                 configuration_digest=row["configuration_digest"])
            if row.get("admission_digest") != canonical_digest(spec):
                raise BetaPolicyScreeningPlanError("beta_screening_admission_binding_invalid")
            if spec["task_success_contract_digest"] != plan["task_success_contract_digest"]:
                raise BetaPolicyScreeningPlanError("beta_screening_execution_task_mismatch")
            if (spec["backend"] == FAMILY_BACKEND) is not family:
                raise BetaPolicyScreeningPlanError("beta_screening_execution_backend_mismatch")
            quote = validate_cost_quote(row.get("cost_quote", {}), configuration_digest=row["configuration_digest"],
                                         scene_revision_digest=plan["scene_revision_digest"], episode_count=len(cells))
            if row.get("max_cost_usd") != quote["maximum_cost_usd"]:
                raise BetaPolicyScreeningPlanError("beta_screening_cost_mismatch")
            if money(row["max_cost_usd"]) < execution_cost_floor(spec):
                raise BetaPolicyScreeningPlanError("beta_screening_cost_below_resource_bound")
            if row["status"] == "selected":
                committed += money(row["max_cost_usd"])
            elif money(row["max_cost_usd"]) > budget:
                # A waitlisted row could never be funded within this plan.
                raise BetaPolicyScreeningPlanError("beta_screening_plan_budget_invalid")
        elif row["status"] not in {"unavailable", "deferred", "infeasible"} or not row["blockers"]:
            raise BetaPolicyScreeningPlanError("beta_screening_candidate_status_invalid")
        elif set(row) - evidence_fields != {"candidate_id", "display_name", "configuration_digest", "status", "blockers"}:
            raise BetaPolicyScreeningPlanError("beta_screening_unselected_execution_forbidden")
    if len(set(candidates)) != len(candidates):
        raise BetaPolicyScreeningPlanError("beta_screening_duplicate_candidate")
    if committed > budget or money(plan["maximum_planned_cost_usd"]) != committed:
        raise BetaPolicyScreeningPlanError("beta_screening_plan_budget_invalid")
    if plan["plan_digest"] != canonical_digest(plan, digest_field="plan_digest"):
        raise BetaPolicyScreeningPlanError("beta_screening_plan_digest_mismatch")
    return plan


__all__ = ["PLAN_SCHEMA_VERSION", "QUOTE_SCHEMA_VERSION", "EXECUTION_SCHEMA_VERSION", "CLAIM_CEILING",
           "SITE_BUDGET_USD", "DEFAULT_EPISODES", "WAITLISTED", "WAITLIST_BLOCKER", "BetaPolicyScreeningPlanError", "build_screening_plan",
           "execution_cost_floor", "money", "screening_cells", "validate_cost_quote", "validate_execution_descriptor",
           "validate_screening_plan"]
