"""Site-local unspent plan generations over the existing screening ledger.

Quote/availability refresh is not new paid retry authority. Original plans and
their effect histories remain immutable; completed results are adopted only for
identical scientific bindings, and uncertain/paid attempts stay with their
original campaign. The existing shared site ledger owns every commitment.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import hashlib
import os
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest

REFRESH_RETRY_SECONDS = 300
_PRICE_FIELDS = {"max_hourly_rate_usd", "provider_hard_cap_usd", "hosted_inference_maximum_usd",
                 "hosted_max_cost_per_request_usd"}


def active_campaign_id(root: Path, base_campaign_id: str) -> str:
    from .beta_policy_screening_coordinator import _ledger, _lock
    with _lock(root):
        return _ledger(root).get("refreshes", {}).get(base_campaign_id, {}).get("active_campaign_id", base_campaign_id)


def _same_bytes(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _same_bytes(item) for key, item in value.items()
                if not (key == "path" and "sha256" in value and "size_bytes" in value)}
    if isinstance(value, list):
        return [_same_bytes(item) for item in value]
    return value


def _science(descriptor: Mapping[str, Any]) -> dict[str, Any]:
    # Quotes may change and an existing secret may move within its private
    # delivery channel. Origin, payer, request limit, TTL, every runtime byte,
    # robot/policy/interface and implementation identity must stay identical.
    return _same_bytes({key: value for key, value in descriptor.items()
                        if key not in _PRICE_FIELDS | {"credential_file"}})


def retain_planning_bindings(root: Path, campaign_id: str, *, capabilities: Mapping[str, Any],
                             expectations: Mapping[str, Any] | None = None) -> None:
    """Retain pinned unavailable runtimes without granting execution capability."""
    from .beta_policy_screening_coordinator import _lock, _read, _write
    from .beta_policy_screening_plan import validate_execution_descriptor
    descriptors = {key: row["execution_spec"] for key, row in
                   capabilities.get("candidate_bindings", {}).items()}
    descriptors.update(expectations or {})
    candidates = {}
    for key, descriptor in descriptors.items():
        validated = validate_execution_descriptor(descriptor, candidate_id=key,
            configuration_digest=descriptor["configuration_digest"])
        candidates[key] = _science(validated)
    value = {"schema_version": "blueprint.beta_screening_planning_bindings.v1",
             "campaign_id": campaign_id, "candidates": candidates}
    value["bindings_digest"] = canonical_digest(value, digest_field="bindings_digest")
    with _lock(root):
        directory = root / "planning_bindings"
        directory.mkdir(mode=0o700, exist_ok=True)
        path = directory / (campaign_id + ".json")
        if path.exists():
            if _read(path) != value:
                raise ValueError("beta_screening_refresh_original_runtime_changed")
        else:
            _write(path, value)


def _planning_bindings(root: Path, campaign_id: str) -> Mapping[str, Any]:
    from .beta_policy_screening_coordinator import _read
    path = root / "planning_bindings" / (campaign_id + ".json")
    if not path.exists():
        return {}
    value = _read(path)
    if (value.get("schema_version") != "blueprint.beta_screening_planning_bindings.v1"
            or value.get("campaign_id") != campaign_id
            or value.get("bindings_digest") != canonical_digest(value, digest_field="bindings_digest")
            or not isinstance(value.get("candidates"), dict)):
        raise ValueError("beta_screening_refresh_original_runtime_invalid")
    return value["candidates"]


def reopen_refresh_source(plan: Mapping[str, Any]) -> None:
    """Reopen current rights AND the original canonical execution window."""
    from .beta_policy_screening_worker import require_current_source
    from .task_evaluation_scene_owner_authority import reopen_scene_intent
    require_current_source(plan["source"], site_id=plan["site_id"])
    root = Path(os.environ.get("BLUEPRINT_TASK_EVALUATION_SCENE_INTAKE_ROOT", ""))
    if not root.is_absolute():
        raise ValueError("scene_owner_intake_root_missing")
    path = root / plan["source"]["scene_intent_id"] / "intent.json"
    if any(item.is_symlink() for item in (path, *path.parents)):
        raise ValueError("scene_owner_intent_path_invalid")
    raw = path.read_bytes()
    intent = reopen_scene_intent({"path": str(path), "sha256": "sha256:" + hashlib.sha256(raw).hexdigest(),
                                  "size_bytes": len(raw)})
    if (intent["intent_id"] != plan["source"]["scene_intent_id"]
            or intent["intent_digest"] != plan["source"]["scene_intent_digest"]
            or intent["request"]["submission_id"] != plan["capture_id"]
            or intent["request"]["task"]["task_id"] != plan["task_id"]):
        raise ValueError("beta_screening_refresh_source_intent_changed")


def _pre_effect(root: Path, row: Mapping[str, Any]) -> bool:
    from .beta_policy_screening_coordinator import _capacity_refusal_proof, _unstarted_refusal_proof
    for attempt in [*row.get("attempt_history", []), *([row["attempt"]] if row.get("attempt") else [])]:
        if (attempt.get("provider_allocation_observed") is not False
                or attempt.get("reservation_status") == "committed"):
            return False
        if attempt.get("reservation_status") == "released" and attempt.get("no_policy_api_calls_proven") is True:
            observation = {**attempt, "no_policy_api_calls_proven": True}
            if attempt.get("terminal_reason") == "vast_paid_launch_lock_busy":
                observation.update(retryable_capacity_refusal=True, blockers=["vast_paid_launch_lock_busy"])
                if _capacity_refusal_proof(root, attempt, observation):
                    continue
            elif attempt.get("terminal_reason") == "allocator_fenced_before_create":
                observation["retryable_unstarted_refusal"] = True
                if _unstarted_refusal_proof(root, attempt, observation):
                    continue
        if attempt.get("launch_intent_at"):
            return False
        directory = root / "attempts" / str(attempt["attempt_id"])
        if any((directory / name).exists() for name in (
                "allocator_launch_intent.json", "allocator_process.json", "allocator_result.json",
                "invocation/manifest.json", "invocation/started.json", "invocation/provider_create_intent.json")):
            return False
    return True


def _origins(root: Path, plan: Mapping[str, Any], state: Mapping[str, Any], ledger: Mapping[str, Any],
             *, expectations=None) -> dict[str, dict[str, Any]]:
    """Resolve only our retained no-retry wrappers to their actual operation."""
    from .beta_policy_screening_coordinator import _plan
    plans = {plan["campaign_id"]: plan}
    result = {}
    if state["plan_digest"] != plan["plan_digest"]:
        raise ValueError("beta_screening_refresh_origin_plan_changed")
    for configured in plan["candidates"]:
        key, campaign_id = configured["candidate_id"], plan["campaign_id"]
        row = state["candidates"][key]
        seen = {campaign_id}
        while (row["status"] == "blocked" and row.get("blockers") == ["beta_screening_prior_attempt_not_retry_authority"]
               and row.get("prior_campaign_id")):
            campaign_id = row["prior_campaign_id"]
            if campaign_id in seen:
                raise ValueError("beta_screening_refresh_origin_cycle")
            seen.add(campaign_id)
            origin = ledger["campaigns"].get(campaign_id)
            if origin is None or origin["site_id"] != plan["site_id"]:
                raise ValueError("beta_screening_refresh_origin_scope_changed")
            if campaign_id not in plans:
                plans[campaign_id] = _plan(root, campaign_id, origin["plan_digest"])
                _unchanged(plans[campaign_id], plan, expectations=expectations)
            row = origin["candidates"][key]
        result[key] = {"campaign_id": campaign_id, "plan": plans[campaign_id], "row": row}
    return result


def _eligible(root: Path, plan: Mapping[str, Any], state: Mapping[str, Any], *, origins=None) -> set[str]:
    from .beta_policy_screening_coordinator import _quote_current
    if origins is None:
        from .beta_policy_screening_coordinator import _ledger
        base_id = state.get("refresh", {}).get("base_campaign_id", plan["campaign_id"])
        origins = _origins(root, plan, state, _ledger(root), expectations=_planning_bindings(root, base_id))
    result = set()
    for key, origin in origins.items():
        configured = next(row for row in origin["plan"]["candidates"] if row["candidate_id"] == key)
        row = origin["row"]
        if row["status"] in {"completed", "running", "launching", "reconciliation_required"} or not _pre_effect(root, row):
            continue
        if configured["status"] == "selected" and not _quote_current(configured) and row["status"] in {
                "pending", "preparing", "deferred"}:
            if not set(row.get("blockers", [])) - {
                    "beta_screening_cost_quote_expired", "beta_screening_waiting_for_capacity",
                    "beta_screening_unspent_invocation_recovery"}:
                result.add(key)
        elif configured["status"] == "unavailable":
            blockers = configured.get("blockers", [])
            if (not any(code in {"candidate_disabled", "commercial_use_not_admitted",
                                 "policy_identity_not_verified", "policy_identity_not_pinned"} for code in blockers)
                    and any(code.startswith("credential_unavailable:") or code == "runtime_candidate_binding_missing"
                            for code in blockers)):
                result.add(key)
    return result


def _unchanged(old: Mapping[str, Any], new: Mapping[str, Any], *, expectations=None) -> None:
    ignored = {"campaign_id", "plan_digest", "candidates", "maximum_planned_cost_usd"}
    if _same_bytes({k: v for k, v in old.items() if k not in ignored}) != _same_bytes(
            {k: v for k, v in new.items() if k not in ignored}):
        raise ValueError("beta_screening_refresh_scientific_binding_changed")
    before = {row["candidate_id"]: row for row in old["candidates"]}
    after = {row["candidate_id"]: row for row in new["candidates"]}
    if set(before) != set(after) or any(before[key]["configuration_digest"] != after[key]["configuration_digest"] for key in before):
        raise ValueError("beta_screening_refresh_configuration_changed")
    for key in before:
        if after[key]["status"] == "selected":
            expected = (_science(before[key]["execution_spec"]) if before[key]["status"] == "selected"
                        else (expectations or {}).get(key))
            if expected is None or expected != _science(after[key]["execution_spec"]):
                raise ValueError("beta_screening_refresh_new_configuration_identity_required")


def _remaining(ledger, state, selected, *, origins=None) -> Decimal:
    from .beta_policy_screening_coordinator import SITE_LIMIT, _budget
    commitment = Decimal(_budget(ledger, state["site_id"])["committed_upper_bound_usd"])
    rows = {key: record["row"] for key, record in origins.items()} if origins is not None else state["candidates"]
    transfer = sum((Decimal(rows[key]["attempt"]["maximum_cost_usd"]) for key in selected
                    if rows[key].get("attempt", {}).get("reservation_status") == "reserved"), Decimal(0))
    return SITE_LIMIT - commitment + transfer


def _budget_action(baseline):
    return {**baseline, "refresh_status": "cost_above_remaining_site_budget",
            "blockers": ["beta_screening_refresh_budget_action_required"],
            "next_action": "retain_site_commitments_and_review_cost_or_existing_authority"}


def maybe_refresh_campaign(*, queue_root: Path, base_campaign_id: str,
                           build: Callable[[Mapping[str, Any], str, Path], Mapping[str, Any]]) -> dict[str, Any]:
    """Reopen current source, rebuild unpaid facts, and atomically retain a generation."""
    from .beta_policy_screening_coordinator import (
        _changed, _ledger, _lock, _new_attempt, _now, _plan, _read, _root, _write,
        enqueue_screening_campaign,
    )
    from .beta_policy_screening_plan import validate_screening_plan
    from .beta_policy_screening_worker import require_current_source
    root = _root(queue_root)
    current_id = active_campaign_id(root, base_campaign_id)
    current = _plan(root, current_id)
    baseline = {"campaign_id": current_id, "status": "already_enqueued", "plan_digest": current["plan_digest"]}
    try:
        with _lock(root, ".worker.lock", nonblocking=True):
            current_id = active_campaign_id(root, base_campaign_id)
            current = _plan(root, current_id)
            baseline = {"campaign_id": current_id, "status": "already_enqueued", "plan_digest": current["plan_digest"]}
            expectations = _planning_bindings(root, base_campaign_id)
            with _lock(root):
                ledger = _ledger(root)
                state = deepcopy(ledger["campaigns"][current_id])
            origins = _origins(root, current, state, ledger, expectations=expectations)
            adoptions = {}
            for key, row in state["candidates"].items():
                origin = origins[key]
                prior = origin["row"]
                if origin["campaign_id"] != current_id and prior["status"] == "completed":
                    adoptions[key] = {"status": "completed", "blockers": [],
                        "summary": deepcopy(prior["summary"]),
                        "adopted_from_campaign_id": prior.get("adopted_from_campaign_id", origin["campaign_id"])}
            if adoptions:
                require_current_source(current["source"], site_id=current["site_id"])
                with _lock(root):
                    from .beta_policy_screening_coordinator import _save
                    ledger = _ledger(root)
                    retained_state = ledger["campaigns"][current_id]
                    retained_state["candidates"].update(adoptions)
                    _changed(retained_state)
                    _save(root, ledger)
                    state = deepcopy(retained_state)
                    origins = _origins(root, current, state, ledger, expectations=expectations)
            eligible = _eligible(root, current, state, origins=origins)
            if not eligible:
                return baseline
            checks = root / "refresh_checks"
            checks.mkdir(mode=0o700, exist_ok=True)
            check_path = checks / (current_id + ".json")
            if check_path.exists():
                previous = _read(check_path)
                deadline = datetime.fromisoformat(previous["retry_not_before"])
                if deadline.tzinfo is None:
                    raise ValueError("beta_screening_refresh_backoff_invalid")
                if deadline > datetime.now(timezone.utc):
                    return baseline
            _write(check_path, {"retry_not_before": (datetime.now(timezone.utc) + timedelta(seconds=REFRESH_RETRY_SECONDS)).isoformat()})
            # Every refresh reopens actual current authority; catalog/quote age
            # can never renew source consent or change owner task truth.
            reopen_refresh_source(current)
            generation = int(state.get("refresh", {}).get("generation", 0)) + 1
            identity = canonical_digest({"base_campaign_id": base_campaign_id,
                "supersedes": current["plan_digest"], "generation": generation})[7:39]
            next_id = "screening-refresh-" + identity
            pending_path = checks / (next_id + ".plan.json")
            if pending_path.exists():
                fresh = validate_screening_plan(_read(pending_path))
            else:
                fresh = validate_screening_plan(build(current, next_id, root / "inputs" / next_id))
            if fresh["campaign_id"] != next_id:
                raise ValueError("beta_screening_refresh_generation_invalid")
            _unchanged(current, fresh, expectations=expectations)
            selected = {row["candidate_id"] for row in fresh["candidates"] if row["status"] == "selected"} & eligible
            if any(row["candidate_id"] in eligible and "site_screening_budget_insufficient" in row["blockers"]
                   for row in fresh["candidates"]):
                return _budget_action(baseline)
            if not selected:
                return {**baseline, "refresh_status": "availability_unchanged"}
            from .beta_policy_screening_coordinator import _quote_current
            if any(not _quote_current(row) for row in fresh["candidates"] if row["candidate_id"] in selected):
                return {**baseline, "refresh_status": "fresh_cost_quote_required"}
            proposed = sum((Decimal(row["max_cost_usd"]) for row in fresh["candidates"]
                            if row["candidate_id"] in selected), Decimal(0))
            if proposed > _remaining(ledger, state, selected, origins=origins):
                return _budget_action(baseline)
            old_config = {row["candidate_id"]: row for row in current["candidates"]}
            # Historical selected descriptors retain the exact old receipt
            # identity. Their state below prevents any new dispatch.
            fresh["candidates"] = [deepcopy(old_config[row["candidate_id"]])
                if row["candidate_id"] not in selected and old_config[row["candidate_id"]]["status"] == "selected"
                else row for row in fresh["candidates"]]
            maximum = sum((Decimal(row["max_cost_usd"]) for row in fresh["candidates"] if row["status"] == "selected"), Decimal(0))
            if maximum > Decimal(fresh["budget_usd"]):
                return _budget_action(baseline)
            fresh["maximum_planned_cost_usd"] = format(maximum, ".2f")
            fresh["plan_digest"] = canonical_digest(fresh, digest_field="plan_digest")
            fresh = validate_screening_plan(fresh)
            if pending_path.exists():
                if _read(pending_path) != fresh:
                    raise ValueError("beta_screening_refresh_retained_generation_changed")
            else:
                _write(pending_path, fresh)
            # New pending work, old unspent supersession, completed adoption and
            # the active-generation pointer share ONE durable ledger commit.
            # A crash can never expose adopted completed rows as new pending work.
            def transition(ledger, next_state):
                old_state = ledger["campaigns"][current_id]
                actual = _origins(root, current, old_state, ledger, expectations=expectations)
                if (not selected <= _eligible(root, current, old_state, origins=actual)
                        or any(actual[key]["campaign_id"] != origins[key]["campaign_id"] for key in selected)):
                    raise ValueError("beta_screening_refresh_old_effect_state_changed")
                if proposed > _remaining(ledger, old_state, selected, origins=actual):
                    raise ValueError("beta_screening_refresh_budget_action_required")
                for key, old_row in old_state["candidates"].items():
                    if key in selected:
                        origin = actual[key]
                        origin_row = origin["row"]
                        if origin_row.get("attempt"):
                            origin_row["attempt"]["reservation_status"] = "released"
                        origin_row.update(status="deferred", blockers=["beta_screening_plan_superseded_unspent"], superseded_by=next_id)
                        if origin["campaign_id"] != current_id:
                            _changed(ledger["campaigns"][origin["campaign_id"]])
                        configured = next(row for row in fresh["candidates"] if row["candidate_id"] == key)
                        # Transfer the commitment in the same ledger replace;
                        # there is no zero-reservation window between plans.
                        next_state["candidates"][key].update(status="preparing",
                            attempt=_new_attempt(fresh, configured), refreshed_from_campaign_id=origin["campaign_id"])
                        old_row.update(status="deferred", blockers=["beta_screening_plan_superseded_unspent"], superseded_by=next_id)
                        _changed(old_state)
                    elif old_row["status"] == "completed":
                        next_state["candidates"][key] = {"status": "completed", "blockers": [],
                            "summary": deepcopy(old_row["summary"]),
                            "adopted_from_campaign_id": old_row.get("adopted_from_campaign_id", current_id)}
                    elif next_state["candidates"][key]["status"] == "pending":
                        next_state["candidates"][key] = {"status": "blocked",
                            "blockers": ["beta_screening_prior_attempt_not_retry_authority"],
                            "prior_campaign_id": actual[key]["campaign_id"]}
                next_state["refresh"] = {"generation": generation, "base_campaign_id": base_campaign_id,
                    "supersedes_campaign_id": current_id, "supersedes_plan_digest": current["plan_digest"],
                    "reason": "unspent_quote_or_availability_refresh", "recorded_at": _now()}
                ledger.setdefault("refreshes", {})[base_campaign_id] = {"active_campaign_id": next_id,
                    "generation": generation, "supersedes": current["plan_digest"]}
                _changed(next_state)
            outcome = enqueue_screening_campaign(fresh, queue_root=root, _transition=transition)
            return {**outcome, "refresh_status": "refreshed", "supersedes_campaign_id": current_id}
    except BlockingIOError:
        return {**baseline, "refresh_status": "worker_busy"}
