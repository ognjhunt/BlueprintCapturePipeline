"""Test chosen policies on one stored job: a new, limited screening campaign over the existing pieces.

Plan 1.1 (``docs/launch-readiness/2026-11-11-evaluation-service-plan.md``). One call plans exactly the
named candidates for one website job (``site-<request id>``) and enqueues them under a new campaign id:
``build_planning_inputs`` -> ``plan_within_job_ceiling(candidate_ids=...)`` (or, for a standing test,
the same planner re-funded from the standing allowance) -> ``enqueue_screening_campaign``. Funding,
launch, settlement and publication are the coordinator's, unchanged: this module reserves, launches
and pays for nothing, and every money check stays where it is (the plan budget, the site allowance or
the standing allowance, the job's ceiling, the account month, the spend guard at paid admission).

Two funding modes:

- ``site_allowance``: the job's own screening money, only while its window is open
  (``beta_policy_screening_worker.job_window_closed``), planned within the job's ceiling.
- ``standing_test``: the standing-test allowance (``beta_policy_screening_standing_test``), refused while
  its monthly total is unset or the WebApp has not said the job may be tested after its window.

Nothing here waits or holds a job: every refusal is a typed result that spends nothing. A pair the job
already has a finished result for, or one that is still queued or running, is never tested again
(``beta_policy_screening_tested_pairs``). A job whose launch run went to cold storage, or whose scene
workspace was retired, is never restored from here: the refusal names the door operation that does.
A limited plan carries no scripted reference row (``append_control_row`` adds one only to a campaign
planned without ``candidate_ids``), so the reference row's rules are untouched.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from decimal import Decimal
import json
import os
from pathlib import Path
import re
import sys
from typing import Any

from .decision_evidence_contracts import canonical_digest

SITE_ALLOWANCE = "site_allowance"
STANDING_TEST = "standing_test"
FUNDING_MODES = (SITE_ALLOWANCE, STANDING_TEST)
PROTOCOL = "beta_policy_screening.job_test.v1"
RECEIPT_SCHEMA = "blueprint.beta_screening_job_test_receipt.v1"
RECEIPT_DIRNAME = "job_tests"
MAX_CANDIDATES = 8
RESTORE_LAUNCH_RUN = "restore-launch-run"
RESTORE_SCENE_WORKSPACE = "restore-scene-workspace"

REQUEST_INVALID = "job_test_request_invalid"
SOURCE_NOT_FOUND = "job_test_source_not_found"
SCENE_WORKSPACE_RETIRED = "job_test_scene_workspace_retired"
LAUNCH_RUN_OFFLOADED = "job_test_launch_run_offloaded"
SITE_WITHDRAWN = "job_test_site_withdrawn"
WITHDRAWAL_UNKNOWN = "job_test_site_withdrawal_unknown"
WINDOW_CLOSED = "job_test_window_closed"
ALREADY_TESTED = "job_test_pairs_already_tested"
NOTHING_FUNDED = "job_test_allowance_insufficient"
_SITE = re.compile(r"site-[A-Za-z0-9][A-Za-z0-9._:-]{0,191}")
_CANDIDATE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_REQUEST_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}")
_CODE = re.compile(r"[a-z][a-z0-9_:,.=-]{0,199}")


class JobTestRefused(ValueError):
    def __init__(self, code: str, **details: Any):
        super().__init__(code)
        self.code, self.details = code, details


def validate_request(value: Mapping[str, Any]) -> dict[str, Any]:
    """The request's fields (the door validates the same shape)."""
    fields = {"site_id", "candidate_ids", "funding", "authorization_reference"}
    ids = value.get("candidate_ids") if isinstance(value, Mapping) else None
    reference = value.get("authorization_reference") if isinstance(value, Mapping) else None
    if (not isinstance(value, Mapping) or set(value) != fields
            or not isinstance(value["site_id"], str) or not _SITE.fullmatch(value["site_id"])
            or not isinstance(ids, list) or not 1 <= len(ids) <= MAX_CANDIDATES or len(set(map(str, ids))) != len(ids)
            or any(not isinstance(row, str) or not _CANDIDATE.fullmatch(row) for row in ids)
            or value["funding"] not in FUNDING_MODES
            or not isinstance(reference, str) or reference != reference.strip()
            or not 1 <= len(reference) <= 300 or not reference.isprintable()):
        raise JobTestRefused(REQUEST_INVALID)
    return {key: (list(value[key]) if key == "candidate_ids" else value[key]) for key in sorted(fields)}


def _partition() -> Path | None:
    from .team_scene_capture_root import CAPTURE_PARTITION_ENV

    raw = os.environ.get(CAPTURE_PARTITION_ENV, "").strip()
    return Path(raw) if raw else None


def job_source(packet_root: Path, site_id: str) -> tuple[dict[str, Any], Path]:
    """The job's newest completed scene request and its capture root; refuses by name when it is cold."""
    from .beta_policy_screening_worker import newest_intent_requests
    from .team_scene_packet_worker import _read_json, resolve_capture_root, validated_request

    partition = _partition()
    if partition is not None and (partition / "scenes" / f"{site_id}.retired.v1.json").exists():
        raise JobTestRefused(SCENE_WORKSPACE_RETIRED, restore_operation=RESTORE_SCENE_WORKSPACE)
    newest = newest_intent_requests(packet_root)
    found: list[tuple[int, str, dict[str, Any], Path]] = []
    for path in sorted((packet_root / "processed").glob("*.json")):
        try:
            request = validated_request(_read_json(path))
            if newest.get(request["intent_id"]) != ("processed", path.name):
                continue
            status = _read_json(packet_root / "status" / path.name) or {}
            if status.get("status") not in {"completed", "skipped"}:
                continue
            capture = resolve_capture_root(request)
        except (OSError, ValueError, KeyError):
            continue
        if capture.parent.parent.name == site_id:
            found.append((path.stat().st_mtime_ns, path.name, request, capture))
    if not found:
        raise JobTestRefused(SOURCE_NOT_FOUND)
    _stamp, _name, request, capture = max(found, key=lambda row: row[:2])
    return request, capture


def launch_run_cold(request: Mapping[str, Any]) -> bool:
    """The job's configured-scene launch run went to cold storage (its offload pointer stands in its place)."""
    from .control_plane_evidence_offload import POINTER_SUFFIX

    launch = Path(request["source_launch_root"])
    return not launch.exists() and (launch.parent / f"{launch.name}{POINTER_SUFFIX}").exists()


def current_consent(site_id: str, packet_request: Mapping[str, Any],
                    capture: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """The privacy stop every use of a stored job passes first: ``(source, context)``, or a typed refusal.

    The site's withdrawal as ``_reserve`` reads it before any reservation (withdrawn or unreadable both
    refuse), then consent, rights and the task as the WebApp has them now (the read every paid attempt
    makes). Shared by this module and the launch-run restore, so neither can touch a withdrawn site's job."""
    from .beta_policy_screening_worker import require_current_source
    from .screening_site_withdrawal import site_withdrawal_state
    from .team_scene_packet_worker import _read_json

    withdrawal = site_withdrawal_state(site_id)
    if withdrawal in {"withdrawn", "unknown"}:
        raise JobTestRefused(SITE_WITHDRAWN if withdrawal == "withdrawn" else WITHDRAWAL_UNKNOWN)
    retained = _read_json(capture / "pipeline" / "website_task_context.json") or {}
    source = {"scene_intent_id": packet_request["intent_id"], "scene_intent_digest": packet_request["intent_digest"],
              "capture_id": packet_request["capture_id"], "task_id": packet_request["task_id"],
              "task_context_digest": retained.get("context_digest"),
              "scene_revision_digest": packet_request["scene_revision_digest"]}
    return source, require_current_source(source, site_id=site_id)


def _receipt(root: Path, request_id: str, value: Mapping[str, Any]) -> dict[str, Any]:
    from .beta_policy_screening_coordinator import _write

    receipt = {"schema_version": RECEIPT_SCHEMA, "request_id": request_id,
               "recorded_at": datetime.now(timezone.utc).isoformat(), **value, "receipt_digest": ""}
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    directory = root / RECEIPT_DIRNAME
    directory.mkdir(exist_ok=True, mode=0o700)
    path = directory / f"{request_id}.json"
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous.get("status") in {"enqueued", "already_enqueued"}:
            return previous  # the first outcome of an enqueued request is its record
    _write(path, receipt)
    return receipt


def screen_policies_on_job(request: Mapping[str, Any], *, request_id: str, queue_root: str | Path | None = None,
                       packet_queue_root: str | Path | None = None) -> dict[str, Any]:
    """Plan and enqueue one limited campaign, or refuse by name; never raises for a typed refusal."""
    from .beta_policy_screening_coordinator import _root

    if not isinstance(request_id, str) or not _REQUEST_ID.fullmatch(request_id):
        raise ValueError(REQUEST_INVALID)
    root = _root(queue_root)
    try:
        value = validate_request(request)
        outcome = _test(value, request_id=request_id, root=root, packet_queue_root=packet_queue_root)
    except JobTestRefused as exc:
        outcome = {"status": "refused", "reason": exc.code, **exc.details}
    except (ValueError, RuntimeError, OSError, KeyError) as exc:
        code = str(exc) if isinstance(exc, (ValueError, RuntimeError)) else ""
        outcome = {"status": "refused", "reason": code if _CODE.fullmatch(code) else "job_test_planning_failed"}
    if outcome["status"] == "refused":
        outcome["spent_usd"] = "0"
    return _receipt(root, request_id, {"request": dict(request), **outcome})


def _test(value: Mapping[str, Any], *, request_id: str, root: Path,
          packet_queue_root: str | Path | None) -> dict[str, Any]:
    from .beta_policy_screening_coordinator import _lock, _read, enqueue_screening_campaign
    from .beta_policy_screening_tested_pairs import (
        EVERY_PAIR_TESTED, planning_exclusions, policy_identity, record_tested_pairs, site_pairs,
    )
    from .beta_policy_screening_worker import job_window_closed
    from .team_scene_packet_worker import QUEUE_ROOT_ENV as PACKET_QUEUE_ENV

    site_id, funding = value["site_id"], value["funding"]
    raw = str(packet_queue_root or os.environ.get(PACKET_QUEUE_ENV, "")).strip()
    if not raw or not Path(raw).is_absolute():
        raise JobTestRefused(SOURCE_NOT_FOUND)
    packet_request, capture = job_source(Path(raw), site_id)
    if launch_run_cold(packet_request):
        raise JobTestRefused(LAUNCH_RUN_OFFLOADED, restore_operation=RESTORE_LAUNCH_RUN)
    source, context = current_consent(site_id, packet_request, capture)
    month = None
    if funding == SITE_ALLOWANCE:
        if job_window_closed(context):
            raise JobTestRefused(WINDOW_CLOSED)
    else:
        from .beta_policy_screening_standing_test import month_of, monthly_total_usd, require_site_permission

        month = month_of()
        monthly_total_usd(month=month)
        require_site_permission(context=context, capture_root=capture)
    from .task_evaluation_configured_controls_progression_worker import _validate_source

    result, receipt, _zero = _validate_source(Path(packet_request["source_launch_root"]))
    if (result.get("configured_scene_revision_digest") != source["scene_revision_digest"]
            or receipt.get("receipt_digest") != packet_request["launch_receipt_digest"]):
        raise JobTestRefused("beta_screening_completed_scene_binding_changed")
    from .beta_policy_candidate_catalog import load_default_catalog
    from .beta_policy_screening_team import catalog_for_profile
    from .team_candidate_quick10_registry import profile_for_capture

    profile = profile_for_capture(capture)
    if not profile or profile.get("scene_id") != site_id:
        raise JobTestRefused("beta_screening_scene_profile_unavailable")
    catalog = catalog_for_profile(profile, load_default_catalog())
    unknown = sorted(set(value["candidate_ids"]) - {row["candidate_id"] for row in catalog["candidates"]})
    if unknown:
        raise JobTestRefused("beta_screening_candidate_not_in_catalog", candidate_ids=unknown)
    campaign_id = "screening-" + canonical_digest({"source": source, "catalog_digest": catalog["catalog_digest"],
        "protocol": PROTOCOL, "request_id": request_id, "candidate_ids": sorted(value["candidate_ids"]),
        "funding": funding})[7:39]
    existing = root / "plans" / f"{campaign_id}.json"
    if existing.exists():
        return {"campaign_id": campaign_id, **enqueue_screening_campaign(_read(existing), queue_root=root),
                "funding": funding}
    pairs = site_pairs(root, site_id)
    # The rule the worker's re-plan uses too: never a pair already finished, queued or running.
    known = planning_exclusions(pairs)
    identities = {row["candidate_id"]: policy_identity(row)["policy_identity"] for row in catalog["candidates"]}
    from .beta_policy_screening_capabilities import build_planning_inputs
    from .website_candidate_ranking import retained_priority

    priority = retained_priority(capture / "pipeline", task_context_digest=source["task_context_digest"])
    intake_ranking = bool(priority)
    if not priority:
        from .beta_policy_screening_default_priority import default_screening_priority

        priority = default_screening_priority()
    inputs = build_planning_inputs(catalog=catalog, profile=profile, capture_root=capture, source=source,
                                   inputs_dir=root / "inputs" / campaign_id, priority=priority)
    options = {"capabilities": inputs["capabilities"], "cost_quotes": inputs["cost_quotes"],
               "priority": inputs.get("candidate_priority"), "candidate_ids": list(value["candidate_ids"]),
               "intake_ranking": intake_ranking, "finished_pairs": known}

    def skipped(plan: Mapping[str, Any] | None) -> dict[str, list[str]]:
        planned = {row["candidate_id"] for row in plan["candidates"]} if plan else set()
        rows = {"already_tested": [], "in_progress": []}
        for candidate_id in value["candidate_ids"]:
            if candidate_id in planned:
                continue
            done = any(entry["candidate_id"] == candidate_id or entry["policy_identity"] == identities[candidate_id]
                       for entry in pairs["finished"].values())
            rows["already_tested" if done else "in_progress"].append(candidate_id)
        return {key: ids for key, ids in rows.items() if ids}

    from .beta_policy_screening_plan import BetaPolicyScreeningPlanError, SITE_BUDGET_USD, build_screening_plan

    with _lock(root, ".standing.lock"):
        try:
            if funding == SITE_ALLOWANCE:
                from .beta_policy_screening_job_ceiling import plan_within_job_ceiling

                plan = plan_within_job_ceiling(catalog, inputs["scene"], campaign_id=campaign_id, queue_root=root,
                                               context=context, capture_root=capture, **options)
            else:
                plan = _standing(catalog, inputs["scene"], campaign_id=campaign_id, root=root, month=month,
                                 request_id=request_id, reference=value["authorization_reference"],
                                 site_id=site_id, options=options, budget=SITE_BUDGET_USD,
                                 build=build_screening_plan, identities=identities)
        except BetaPolicyScreeningPlanError as exc:
            if str(exc) == EVERY_PAIR_TESTED:
                raise JobTestRefused(ALREADY_TESTED, skipped=skipped(None)) from None
            raise
        selected = [row["candidate_id"] for row in plan["candidates"] if row["status"] == "selected"]
        if not selected:
            raise JobTestRefused(NOTHING_FUNDED, skipped=skipped(plan), candidates={
                row["candidate_id"]: {"status": row["status"], "blockers": row["blockers"]}
                for row in plan["candidates"]})
        from .website_candidate_selection import retain_campaign_catalog

        retain_campaign_catalog(queue_root=root, campaign_id=campaign_id, catalog=catalog)
        enqueued = enqueue_screening_campaign(plan, queue_root=root)
    record_tested_pairs(root, site_id)
    return {**enqueued, "funding": funding, "selected": selected, "skipped": skipped(plan),
            "maximum_planned_cost_usd": plan["maximum_planned_cost_usd"],
            "candidates": {row["candidate_id"]: {"status": row["status"], "blockers": row["blockers"],
                                                 **({"max_cost_usd": row["max_cost_usd"],
                                                     "hard_ttl_seconds": row["execution_spec"]["hard_ttl_seconds"]}
                                                    if row["status"] == "selected" else {})}
                           for row in plan["candidates"]}}


def _standing(catalog: Mapping[str, Any], scene: Mapping[str, Any], *, campaign_id: str, root: Path, month: str,
              request_id: str, reference: str, site_id: str, options: Mapping[str, Any], budget: Decimal,
              build: Any, identities: Mapping[str, str]) -> dict[str, Any]:
    """The limited plan, funded from the standing allowance (caller holds the standing lock).

    The $2 cap is per policy per job: one policy identity (``policy_identity``), whatever candidate id it was
    entered under, so the same checkpoint entered again never gets a second $2."""
    from .beta_policy_screening_standing_test import (
        POLICY_CAP_USD, monthly_total_usd, policy_used, standing_plan, standing_record, standing_usage,
    )

    total = monthly_total_usd(month=month)
    usage = standing_usage(root, month=month)
    plan = build(catalog, scene, campaign_id=campaign_id, budget_usd=f"{budget:.2f}", website=True, **options)
    policy_left = {candidate_id: max(Decimal(0), POLICY_CAP_USD - policy_used(
                       usage, site_id=site_id, candidate_id=candidate_id, identity=identities[candidate_id]))
                   for candidate_id in options["candidate_ids"]}
    return standing_plan(plan, policy_left=policy_left, month_left=max(Decimal(0), total - usage["month_usd"]),
                         record=standing_record(monthly_total=total, month=month, request_id=request_id,
                                                authorization_reference=reference))


def main(argv: Sequence[str] | None = None) -> int:
    """The door's entry: one request, one receipt printed as JSON (``door_outcome`` enqueued or refused)."""
    from .beta_policy_screening_coordinator import QUEUE_ROOT_ENV

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request-json", required=True)
    parser.add_argument("--door-request-id", required=True)
    parser.add_argument("--queue-root", default=os.environ.get(QUEUE_ROOT_ENV))
    parser.add_argument("--packet-queue-root", default=None)
    args = parser.parse_args(argv)
    request = json.loads(args.request_json)
    if isinstance(request, dict):
        request = {key: item for key, item in request.items() if key != "kind"}
    receipt = screen_policies_on_job(request, request_id=args.door_request_id, queue_root=args.queue_root,
                                 packet_queue_root=args.packet_queue_root)
    door = "enqueued" if receipt.get("status") in {"enqueued", "already_enqueued"} else "refused"
    print(json.dumps({**receipt, "door_outcome": door, "code": receipt.get("reason")}, sort_keys=True))
    return 0 if door == "enqueued" else 3


if __name__ == "__main__":
    sys.exit(main())
