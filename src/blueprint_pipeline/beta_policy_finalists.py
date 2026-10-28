"""Continue completed beta screening into a source-bound 200-cell evaluation.

Preparation is unpaid. The existing per-site owner's capped, single-attempt
spend admission is required before the ordinary canonical allocator can run.
Diagnostic screening and even a full unqualified run never create a pilot
recommendation. Private matrix cells stay on the Pipeline control plane.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import os
from pathlib import Path
import re
from typing import Any, Mapping

from .decision_evidence_contracts import canonical_digest, cross_runtime_canonical_digest
from .beta_policy_screening_coordinator import _lock, _read, _root, _write, _safe_blockers, _capacity_refusal_proof

SCHEMA = "beta_policy_finalist_request.v1"
ROOT_ENV = "BLUEPRINT_BETA_POLICY_FINALIST_QUEUE_ROOT"
DIGEST = re.compile(r"sha256:[a-f0-9]{64}")
ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,191}")
BINDING_FIELDS = {"finalist_id", "screening_campaign_id", "screening_publication_digest",
                  "job_id", "brief_digest", "source_fingerprint"}


def finalist_id(request: Mapping[str, Any]) -> str:
    return "finalist-" + cross_runtime_canonical_digest({key: request[key] for key in (
        "campaign_id", "candidate_id", "job_id", "brief_digest", "source_fingerprint")})[7:39]


def validate_finalist_binding(binding: Mapping[str, Any]) -> None:
    if (not isinstance(binding, Mapping) or set(binding) - BINDING_FIELDS - {"owner_authority_digest"}
            or BINDING_FIELDS - set(binding)
            or any(not ID.fullmatch(str(binding[key])) for key in ("finalist_id", "screening_campaign_id", "job_id"))
            or any(not DIGEST.fullmatch(str(binding[key])) for key in (
                "screening_publication_digest", "brief_digest", "source_fingerprint"))
            or "owner_authority_digest" in binding and not DIGEST.fullmatch(str(binding["owner_authority_digest"]))):
        raise ValueError("beta_finalist_binding_invalid")


def _queue(root: str | Path | None = None) -> Path:
    target = _root(root or os.getenv(ROOT_ENV) or "/var/lib/blueprint/pipeline-control-plane/beta-policy-finalists")
    requests = target / "requests"
    if requests.is_symlink():
        raise ValueError("beta_finalist_queue_root_unsafe")
    requests.mkdir(exist_ok=True, mode=0o700)
    return target


def _receipt(record: Mapping[str, Any]) -> dict[str, Any]:
    plan = record["plan"]
    binding = plan["finalist_binding"]
    return {"schema_version": "beta_policy_finalist_receipt.v1", "finalist_id": plan["campaign_id"],
            "plan_digest": record["prepared_plan_digest"], "state": record["state"], "source": plan["source"],
            **{key: binding[key] for key in ("job_id", "brief_digest", "source_fingerprint")},
            "candidate_id": plan["candidates"][0]["candidate_id"], "episodes": 200,
            "maximum_planned_cost_usd": plan["maximum_planned_cost_usd"],
            "recommendation_eligible": False, "provider_mutation_performed": False,
            "paid_execution_requested": bool(record.get("authority")),
            **({"owner_authority_digest": record["authority"]["authority_digest"]} if record.get("authority") else {}),
            **({"publication": record["publication"]} if record.get("publication") else {})}


def require_finalist_authority(plan: Mapping[str, Any]) -> None:
    """Reopen the retained owner admission; a purpose flag alone grants nothing."""
    binding = plan.get("finalist_binding") or {}
    validate_finalist_binding(binding)
    path = _queue() / "requests" / (str(plan["campaign_id"]) + ".json")
    record = _read(path)
    authority = record.get("authority") or {}
    raw = {key: value for key, value in authority.items() if key != "authority_digest"}
    if (record.get("plan") != plan or record.get("state") not in {
            "authorized", "preparing", "launching", "running", "reconciliation_required", "completed", "blocked"}
            or authority.get("authority_digest") != cross_runtime_canonical_digest(raw)
            or binding.get("owner_authority_digest") != authority.get("authority_digest")
            or authority.get("plan_digest") != record.get("prepared_plan_digest")
            or authority.get("approval", {}).get("maxAttempts") != 1
            or Decimal(str(authority.get("approval", {}).get("sponsorCapUsd", 0))) < Decimal(plan["maximum_planned_cost_usd"])
            or datetime.fromisoformat(authority["approval"]["expiresAtIso"].replace("Z", "+00:00")) <= datetime.now(timezone.utc)):
        raise ValueError("beta_finalist_owner_authority_missing_or_expired")
    from .team_scene_packet_worker import resolve_capture_root
    from .team_candidate_quick10_registry import profile_for_capture
    profile = profile_for_capture(resolve_capture_root(plan["source"])) or {}
    candidate = plan["candidates"][0]
    if (authority["approved_by"] not in profile.get("allowed_founder_approvers", [])
            or Decimal(str(profile.get("max_hourly_rate_usd") or "0"))
            != Decimal(candidate["execution_spec"]["max_hourly_rate_usd"])):
        raise ValueError("beta_finalist_current_operator_authority_changed")


def prepare_finalist(request: Mapping[str, Any], *, queue_root=None, screening_root=None,
                     source_validator=None, profile_loader=None) -> dict[str, Any]:
    from .beta_policy_screening_coordinator import _ledger, _plan
    from .beta_policy_screening_result import campaign_status, screening_publication
    from .beta_policy_screening_worker import require_current_source
    from .beta_policy_screening_plan import validate_screening_plan, execution_cost_floor
    from .beta_policy_screening_launch import _bound_path
    from .team_candidate_quick10_registry import profile_for_capture, scene_facts
    from .team_scene_packet_worker import resolve_capture_root
    from .team_evaluation_plan import campaign_ttl_seconds, scored_cells

    expected = {"schema_version", "campaign_id", "publication_digest", "candidate_id", "job_id", "brief_digest", "source_fingerprint"}
    if (set(request) != expected or request.get("schema_version") != SCHEMA
            or any(not ID.fullmatch(str(request[key])) for key in ("campaign_id", "candidate_id", "job_id"))
            or any(not DIGEST.fullmatch(str(request[key])) for key in ("publication_digest", "brief_digest", "source_fingerprint"))):
        raise ValueError("beta_finalist_request_invalid")
    root = _queue(queue_root)
    requests = root / "requests"
    requests.mkdir(exist_ok=True, mode=0o700)
    fid = finalist_id(request)
    path = requests / (fid + ".json")
    with _lock(root):
        if path.exists():
            previous = _read(path)
            (source_validator or require_current_source)(previous["plan"]["source"], site_id=previous["plan"]["site_id"])
            return _receipt(previous)
        parent = _root(screening_root)
        with _lock(parent):
            ledger = _ledger(parent)
            state = ledger["campaigns"].get(request["campaign_id"])
            if not state or campaign_status(state["candidates"]) != "completed":
                raise ValueError("beta_finalist_screening_not_completed")
            plan = _plan(parent, request["campaign_id"], state["plan_digest"])
            publication = screening_publication(plan=plan, state=state, budget=state["site_budget"])
            if publication["publication_digest"] != request["publication_digest"]:
                snapshot = parent / "publications" / "history" / (request["publication_digest"][7:] + ".json")
                if not snapshot.is_file():
                    raise ValueError("beta_finalist_screening_publication_changed")
                retained = _read(snapshot)
                mutable = {"budget", "revision", "generated_at", "publication_digest"}
                if (retained.get("publication_digest") != request["publication_digest"]
                        or cross_runtime_canonical_digest(retained, digest_field="publication_digest") != request["publication_digest"]
                        or {key: value for key, value in retained.items() if key not in mutable}
                        != {key: value for key, value in publication.items() if key not in mutable}):
                    raise ValueError("beta_finalist_screening_publication_changed")
            candidate = next((row for row in plan["candidates"] if row["candidate_id"] == request["candidate_id"]), None)
            observed = state["candidates"].get(request["candidate_id"]) or {}
            if (not candidate or candidate["status"] != "selected" or observed.get("status") != "completed"
                    or (observed.get("summary") or {}).get("episodes_succeeded", 0) < 1):
                raise ValueError("beta_finalist_completed_candidate_required")
        if plan["site_id"] != "site-" + request["job_id"]:
            raise ValueError("beta_finalist_job_source_mismatch")
        (source_validator or require_current_source)(plan["source"], site_id=plan["site_id"])
        if plan.get("evaluation_backend") == "task_family_native.v1":
            raise ValueError("beta_finalist_family_scored_matrix_unavailable")
        capture = resolve_capture_root(plan["source"])
        profile = (profile_loader or profile_for_capture)(capture) or {}
        if profile.get("scene_id") != plan["site_id"]:
            raise ValueError("beta_finalist_scene_profile_unavailable")
        _bound_path(plan["scene_inputs"])
        scene = scene_facts(profile)["scene"]
        cells = scored_cells(scene_revision_digest=plan["scene_revision_digest"], scene_id=plan["scene_id"],
            salt=profile.get("scored_matrix_salt"), task_spec=scene["task_spec"],
            task_kind=scene["task_spec"]["task_kind"], nominal_dynamic_friction=profile.get("nominal_dynamic_friction"))
        cells = [{**cell, "cell_id": "finalist." + cell["cell_id"], "partition": "finalist",
                  "resolved_scenario_digest": cell["cell_spec_digest"]} for cell in cells]
        selected = deepcopy(candidate)
        spec = selected["execution_spec"]
        if "credential_file" in spec or "hosted_inference_payment" in spec:
            raise ValueError("beta_finalist_hosted_cost_quote_required")
        rate = Decimal(str(profile.get("max_hourly_rate_usd") or "0"))
        if not rate.is_finite() or rate <= 0:
            raise ValueError("beta_finalist_current_operator_price_bound_missing")
        spec["max_hourly_rate_usd"] = f"{rate:.2f}"
        spec["hard_ttl_seconds"] = campaign_ttl_seconds()
        selected["admission_digest"] = canonical_digest(spec)
        cap = execution_cost_floor(spec)
        if cap > Decimal("20"):
            raise ValueError("beta_finalist_existing_provider_cap_insufficient")
        quote = {**deepcopy(selected["cost_quote"]), "episode_count": 200, "maximum_cost_usd": f"{cap:.2f}",
                 "valid_until": (datetime.now(timezone.utc) + timedelta(days=7)).isoformat(),
                 "quote_id": fid + "-resource-bound",
                 "price_source": "operator_compute_ceiling:" + selected["admission_digest"], "quote_digest": ""}
        quote["quote_digest"] = canonical_digest(quote, digest_field="quote_digest")
        selected.update(cost_quote=quote, max_cost_usd=f"{cap:.2f}", episode_count=200, scenario_cells=cells)
        binding = {"finalist_id": fid, "screening_campaign_id": request["campaign_id"],
                   "screening_publication_digest": request["publication_digest"],
                   **{key: request[key] for key in ("job_id", "brief_digest", "source_fingerprint")}}
        plan.update(schema_version="beta_policy_finalist_plan.v1", campaign_id=fid, purpose="finalist",
                    budget_usd="20.00", maximum_planned_cost_usd=f"{cap:.2f}", scenario_cells=cells,
                    scenario_plan_digest=canonical_digest({"cells": cells}), candidates=[selected], finalist_binding=binding,
                    plan_digest="")
        plan["plan_digest"] = canonical_digest(plan, digest_field="plan_digest")
        validate_screening_plan(plan)
        record = {"request": dict(request), "plan": plan, "prepared_plan_digest": plan["plan_digest"],
                  "state": "awaiting_owner_authority", "revision": 1, "updated_at": datetime.now(timezone.utc).isoformat()}
        _write(path, record)
        return _receipt(record)


def inspect_finalist(fid: str, *, queue_root=None) -> dict[str, Any]:
    if not re.fullmatch(r"finalist-[a-f0-9]{32}", fid):
        raise ValueError("beta_finalist_identity_invalid")
    return _receipt(_read(_queue(queue_root) / "requests" / (fid + ".json")))


def authorize_finalist(fid: str, payload: Mapping[str, Any], *, queue_root=None) -> dict[str, Any]:
    root = _queue(queue_root)
    inspect_finalist(fid, queue_root=root)
    if set(payload) != {"plan_digest", "approved_by", "approval"} or not isinstance(payload.get("approved_by"), str) or not 1 <= len(payload["approved_by"].strip()) <= 192:
        raise ValueError("beta_finalist_owner_approval_invalid")
    approval = payload["approval"]
    if (not isinstance(approval, Mapping) or set(approval) != {
            "episodes", "sponsorCapUsd", "expiresAtIso", "maxAttempts", "evidenceScope"}
            or approval.get("episodes") != 200 or type(approval.get("episodes")) is not int
            or approval.get("maxAttempts") != 1 or type(approval.get("maxAttempts")) is not int
            or approval.get("evidenceScope") != "development_only"
            or type(approval.get("sponsorCapUsd")) not in (int, float) or not 0 < approval["sponsorCapUsd"] <= 20):
        raise ValueError("beta_finalist_owner_approval_invalid")
    expires = datetime.fromisoformat(str(approval.get("expiresAtIso")).replace("Z", "+00:00"))
    with _lock(root):
        path = root / "requests" / (fid + ".json")
        record = _read(path)
        plan = record["plan"]
        if payload["plan_digest"] != record["prepared_plan_digest"]:
            raise ValueError("beta_finalist_plan_changed")
        authority = dict(payload)
        authority["authority_digest"] = cross_runtime_canonical_digest(authority)
        if record.get("authority"):
            if record["authority"] != authority:
                raise ValueError("beta_finalist_owner_approval_conflict")
            return _receipt(record)
        if expires.tzinfo is None or not datetime.now(timezone.utc) < expires <= datetime.now(timezone.utc) + timedelta(days=7):
            raise ValueError("beta_finalist_owner_approval_expired")
        if expires > datetime.fromisoformat(plan["candidates"][0]["cost_quote"]["valid_until"].replace("Z", "+00:00")):
            raise ValueError("beta_finalist_owner_approval_exceeds_quote_validity")
        if Decimal(str(approval["sponsorCapUsd"])) < Decimal(plan["maximum_planned_cost_usd"]):
            raise ValueError("beta_finalist_owner_cap_insufficient")
        from .beta_policy_screening_worker import require_current_source
        from .team_scene_packet_worker import resolve_capture_root
        from .team_candidate_quick10_registry import profile_for_capture
        require_current_source(plan["source"], site_id=plan["site_id"])
        profile = profile_for_capture(resolve_capture_root(plan["source"])) or {}
        if payload["approved_by"] not in profile.get("allowed_founder_approvers", []):
            raise ValueError("beta_finalist_owner_not_authorized")
        plan["finalist_binding"]["owner_authority_digest"] = authority["authority_digest"]
        plan["budget_usd"] = format(Decimal(str(approval["sponsorCapUsd"])), ".2f")
        plan["plan_digest"] = canonical_digest(plan, digest_field="plan_digest")
        record.update(authority=authority, state="authorized", revision=record["revision"] + 1,
                      updated_at=datetime.now(timezone.utc).isoformat())
        _write(path, record)
        return _receipt(record)


def process_finalists(*, queue_root=None, executor=None, transport=None, max_requests=2) -> dict[str, Any]:
    """Resume authorized work through the same exact-once allocator bridge."""
    from . import beta_policy_screening_launch as native
    from .beta_policy_screening_result import candidate_result_summary, screening_publication
    from .beta_policy_screening_worker import publish_screening, require_current_source
    backend = executor or native
    root = _queue(queue_root)
    processed = 0
    paths = sorted((root / "requests").glob("*.json"))
    try:
        with _lock(root, ".worker.lock", nonblocking=True):
            for path in paths:
                record = _read(path)
                if (not record.get("authority") or processed >= max_requests
                        or record.get("state") in {"completed", "blocked"} and record.get("publication")
                        and record.get("publication_acknowledged_digest") == (record.get("publication") or {}).get("publication_digest")):
                    continue
                plan = record["plan"]
                candidate = plan["candidates"][0]
                state = record["state"]
                generation = int(record.get("attempt_generation", 0))
                attempt_key = plan["campaign_id"] + (f"-r{generation}" if generation else "")
                attempt_dir = root / "attempts" / attempt_key
                if attempt_dir.is_symlink():
                    raise ValueError("beta_finalist_attempt_path_unsafe")
                attempt_dir.mkdir(exist_ok=True, mode=0o700)
                if state not in {"completed", "blocked"}:
                    if state in {"authorized", "preparing"}:
                        try:
                            require_finalist_authority(plan)
                            require_current_source(plan["source"], site_id=plan["site_id"])
                            prepared = backend.prepare_candidate(plan=plan, candidate=candidate, attempt_dir=attempt_dir)
                            capacity_check = getattr(backend, "preflight_candidate_capacity", None)
                            capacity = capacity_check(plan=plan, candidate=candidate, attempt_dir=attempt_dir,
                                prepared=prepared) if callable(capacity_check) else {"capacity_available": True}
                            if not isinstance(capacity, Mapping) or type(capacity.get("capacity_available")) is not bool:
                                raise ValueError("beta_finalist_capacity_preflight_invalid")
                            if capacity["capacity_available"] is False:
                                if capacity.get("blockers") not in (["vast_paid_launch_lock_busy"], ["vast_release_deployment_in_progress"]):
                                    raise ValueError("beta_finalist_capacity_preflight_invalid")
                                observation = {"status": "preparing", "blockers": ["beta_screening_waiting_for_capacity"]}
                            else:
                                # Persist before crossing the provider boundary. A crash from
                                # this point reconciles the same exclusive allocator intent.
                                record.update(state="launching", prepared=prepared)
                                _write(path, record)
                                try:
                                    observation = backend.launch_candidate(plan=plan, candidate=candidate,
                                        attempt_dir=attempt_dir, prepared=prepared)
                                except Exception:  # a lost response can hide an allocated provider
                                    observation = {"status": "reconciliation_required",
                                        "blockers": ["beta_finalist_launch_response_unknown"]}
                        except (ValueError, OSError, RuntimeError) as exc:
                            observation = {"status": "blocked", "no_allocation_proven": True,
                                "provider_allocation_performed": False,
                                "blockers": _safe_blockers([str(exc)], "beta_finalist_prelaunch_refused")}
                    else:
                        try:
                            observation = backend.reconcile_candidate(plan=plan, candidate=candidate,
                                attempt_dir=attempt_dir, prepared=record.get("prepared") or {}, handle=record.get("handle"))
                        except (ValueError, OSError, RuntimeError):
                            observation = {"status": "reconciliation_required", "blockers": ["beta_finalist_reconciliation_failed"]}
                    observed_state = observation.get("status")
                    if (observed_state == "blocked" and record.get("allocated") is not True
                            and observation.get("retryable_capacity_refusal") is True
                            and observation.get("no_policy_api_calls_proven") is True
                            and _capacity_refusal_proof(root, {"attempt_id": attempt_key}, observation)):
                        record.setdefault("attempt_history", []).append({"attempt_key": attempt_key,
                            "prepared": record.get("prepared"), "capacity_refusal_receipt": observation["capacity_refusal_receipt"]})
                        record.update(attempt_generation=generation + 1, prepared=None, handle=None)
                        observed_state = "preparing"
                        observation = {"status": "preparing", "blockers": ["beta_screening_waiting_for_capacity"]}
                    allocated = record.get("allocated") is True or observation.get("provider_allocation_performed") is True
                    resolved = (observation.get("no_allocation_proven") is True
                                and observation.get("provider_allocation_performed") is False and not allocated
                                or observation.get("provider_zero_verified") is True and allocated)
                    if observed_state in {"completed", "blocked"} and not resolved:
                        observed_state = "reconciliation_required"
                    if observed_state not in {"preparing", "running", "completed", "blocked", "reconciliation_required"}:
                        observed_state = "reconciliation_required"
                    summary = None
                    if observed_state == "completed":
                        try:
                            summary = candidate_result_summary(observation["result"], plan=plan, candidate=candidate)
                        except (ValueError, KeyError, TypeError):
                            observed_state = "blocked"
                    record.update(state=observed_state, handle=observation.get("handle") or record.get("handle"),
                                  allocated=record.get("allocated") is True or observation.get("provider_allocation_performed") is True,
                                  blockers=_safe_blockers(observation.get("blockers"), "beta_finalist_execution_blocked")
                                      if observed_state in {"preparing", "blocked", "reconciliation_required"} else [], summary=summary,
                                  revision=record["revision"] + 1, updated_at=datetime.now(timezone.utc).isoformat())
                row = {"status": record["state"], "blockers": record.get("blockers") or []}
                if record.get("summary"):
                    row["summary"] = record["summary"]
                cap = candidate["max_cost_usd"]
                spent = cap if record.get("allocated") and record["state"] in {"completed", "blocked"} else "0"
                reserved = cap if record["state"] not in {"completed", "blocked"} else "0"
                budget = {"currency": "USD", "limit_usd": plan["budget_usd"],
                          "spent_upper_bound_usd": spent, "reserved_usd": reserved,
                          "committed_upper_bound_usd": str(Decimal(spent) + Decimal(reserved))}
                publication = screening_publication(plan=plan,
                    state={"candidates": {candidate["candidate_id"]: row}, "revision": record["revision"],
                           "updated_at": record["updated_at"]}, budget=budget)
                publication.update(purpose="finalist", finalist_binding=plan["finalist_binding"])
                publication["publication_digest"] = cross_runtime_canonical_digest(publication, digest_field="publication_digest")
                record["publication"] = publication
                _write(path, record)
                try:
                    (transport or publish_screening)(publication)
                    record["publication_acknowledged_digest"] = publication["publication_digest"]
                    _write(path, record)
                except (ValueError, OSError, RuntimeError):
                    pass  # durable accepted publication is retried on the next worker pass
                processed += 1
        return {"status": "processed", "processed": processed}
    except BlockingIOError:
        return {"status": "worker_busy", "processed": 0}


def require_finalist_session_authority(authority: Mapping[str, Any], *, job_dir: str | Path, max_hourly_rate_usd: float) -> dict[str, Any]:
    """Allocator-side authority/source check, independent of its caller's bundle."""
    rate = Decimal(str(max_hourly_rate_usd))
    if not rate.is_finite() or rate <= 0:
        raise ValueError("beta_finalist_allocator_rate_invalid")
    binding = authority.get("beta_screening_admission") or {}
    fid = binding.get("finalist_id")
    if not isinstance(fid, str) or not re.fullmatch(r"finalist-[a-f0-9]{32}", fid):
        raise ValueError("beta_finalist_session_identity_missing")
    record = _read(_queue() / "requests" / (fid + ".json"))
    plan = record["plan"]
    require_finalist_authority(plan)
    from .beta_policy_screening_worker import require_current_source
    from .beta_policy_screening_launch import _bound_path, _read as read_artifact
    require_current_source(plan["source"], site_id=plan["site_id"])
    candidate = plan["candidates"][0]
    prepared = record.get("prepared") or {}
    if (record.get("state") not in {"launching", "running", "reconciliation_required"}
            or Path(job_dir).resolve() != (Path(str(prepared.get("attempt_dir") or "")) / "allocator").resolve()
            or not prepared.get("authority") or read_artifact(_bound_path(prepared["authority"])) != authority
            or authority.get("episodes_per_policy") != 200
            or authority.get("candidate_ids") != [candidate["candidate_id"]]
            or Decimal(str(authority.get("hard_cap_usd"))) != Decimal(candidate["max_cost_usd"])
            or authority.get("hard_ttl_seconds") != candidate["execution_spec"]["hard_ttl_seconds"]
            or rate > Decimal(candidate["execution_spec"]["max_hourly_rate_usd"])
            or binding.get("plan_digest") != plan["plan_digest"]
            or binding.get("owner_authority_digest") != record["authority"]["authority_digest"]):
        raise ValueError("beta_finalist_allocator_authority_mismatch")
    return record


def consume_finalist_provider_create(authority: Mapping[str, Any], bundle: Mapping[str, Any], *, job_dir: str | Path, max_hourly_rate_usd: float) -> dict[str, Any]:
    """Final provider hook: freshness plus one globally retained create allowance."""
    from .beta_policy_screening_launch import _bound_path, _read as read_artifact, _write as exclusive_write
    record = require_finalist_session_authority(authority, job_dir=job_dir, max_hourly_rate_usd=max_hourly_rate_usd)
    prepared = record["prepared"]
    if read_artifact(_bound_path(prepared["bundle_receipt"])) != bundle:
        raise ValueError("beta_finalist_allocator_bundle_mismatch")
    fid = record["plan"]["campaign_id"]
    receipt = {"status": "consumed", "finalist_id": fid,
               "authority_digest": authority["authority_digest"], "bundle_sha256": bundle["bundle_sha256"],
               "owner_authority_digest": record["authority"]["authority_digest"],
               "maximum_provider_allocations": 1, "retry_cap": 0}
    path = _queue() / "attempts" / fid / "owner_provider_create_consumption.json"
    try:
        exclusive_write(path, receipt)
    except FileExistsError as exc:
        raise ValueError("beta_finalist_provider_create_authority_already_consumed") from exc
    return receipt
