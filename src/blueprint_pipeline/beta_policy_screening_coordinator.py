"""Durable, bounded beta screening over the canonical policy allocator.

Beta journey: a prepared site's candidate screening. A single atomic ledger
holds campaign state and site commitments, so a crash cannot publish a launch
without its reservation. The expensive worker lives behind the existing
allocator bridge; this coordinator never invokes a provider API. Ambiguous
launches retain their reservation and are reconciled, never retried blindly.

The standing allowance is USD 15 for the stable website site submission,
shared across candidate sets and scene revisions. Once allocation is observed,
the entire reserved upper bound is conservatively charged. A lower provider
estimate cannot silently replenish the allowance. A campaign completed within
it is the job's evaluation result; spend beyond it needs the owner's approval
and is never reserved by this queue.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import tempfile
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

from .beta_policy_screening_result import (
    ACTIVE_STATUSES, QUEUED_STATUSES, candidate_result_summary, campaign_status, money, screening_publication,
)
from .decision_evidence_contracts import canonical_digest

QUEUE_ROOT_ENV = "BLUEPRINT_BETA_POLICY_SCREENING_QUEUE_ROOT"
DEFAULT_QUEUE_ROOT = "/var/lib/blueprint/pipeline-control-plane/beta-policy-screening"
LEDGER_SCHEMA = "blueprint.beta_policy_screening_ledger.v1"
LEDGER_FILENAME = "screening_ledger.v1.json"
SITE_LIMIT = Decimal("15")
CAPACITY_RETRY_SECONDS = 60
CAPACITY_WAIT_BLOCKER = "beta_screening_waiting_for_capacity"
INFEASIBLE_BLOCKER = "candidate_infeasible_measured_geometry"
# The plan's own reason: published reasons stay a subset of the plan's.
WAITLIST_BLOCKER = "site_screening_budget_insufficient"
BUDGET_EXHAUSTED_BLOCKER = "beta_screening_site_budget_exhausted"
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_MONEY = re.compile(r"(?:0|[1-9][0-9]*)(?:\.[0-9]{1,6})?")
_BLOCKER = re.compile(r"[a-z][a-z0-9_:.=-]{0,199}")


class ScreeningCoordinatorError(ValueError):
    """A campaign cannot advance without weakening identity or cost controls."""


def _amount(value: Any) -> Decimal:
    if not isinstance(value, str) or not _MONEY.fullmatch(value) or len(value) > 32:
        raise ScreeningCoordinatorError("beta_screening_money_invalid")
    return Decimal(value)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _root(value: str | Path | None) -> Path:
    root = Path(value or os.environ.get(QUEUE_ROOT_ENV) or DEFAULT_QUEUE_ROOT)
    if not root.is_absolute() or ".." in root.parts or any(path.is_symlink() for path in (root, *root.parents)):
        raise ScreeningCoordinatorError("beta_screening_queue_root_unsafe")
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    for name in ("plans", "attempts", "publications"):
        target = root / name
        if target.is_symlink():
            raise ScreeningCoordinatorError("beta_screening_queue_root_unsafe")
        target.mkdir(exist_ok=True, mode=0o700)
    return root


@contextmanager
def _lock(root: Path, name: str = ".ledger.lock", *, nonblocking: bool = False) -> Iterator[None]:
    descriptor = os.open(root / name, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | (fcntl.LOCK_NB if nonblocking else 0))
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _write(path: Path, value: Mapping[str, Any]) -> None:
    if path.is_symlink():
        raise ScreeningCoordinatorError("beta_screening_state_path_unsafe")
    with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix=".screening-", suffix=".tmp",
                                     delete=False, encoding="utf-8") as stream:
        temporary = Path(stream.name)
        try:
            json.dump(value, stream, sort_keys=True, allow_nan=False, separators=(",", ":"))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def _read(path: Path) -> dict[str, Any]:
    if path.is_symlink():
        raise ScreeningCoordinatorError("beta_screening_state_path_unsafe")
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ScreeningCoordinatorError("beta_screening_state_invalid")
    return value


def _budget(ledger: Mapping[str, Any], site_id: str, *, campaign_id: str | None = None) -> dict[str, Any]:
    reserved, spent = Decimal(0), Decimal(0)
    for key, campaign in ledger["campaigns"].items():
        if campaign["site_id"] != site_id or campaign_id is not None and key != campaign_id:
            continue
        for row in campaign["candidates"].values():
            attempts = [*row.get("attempt_history", []), *([row["attempt"]] if row.get("attempt") else [])]
            for attempt in attempts:
                cap = _amount(attempt["maximum_cost_usd"])
                state = attempt["reservation_status"]
                if state == "reserved":
                    reserved += cap
                elif state == "committed":
                    spent += cap
                elif state != "released":
                    raise ScreeningCoordinatorError("beta_screening_reservation_state_invalid")
    return {"currency": "USD", "limit_usd": money(SITE_LIMIT),
            "committed_upper_bound_usd": money(reserved + spent),
            "spent_upper_bound_usd": money(spent), "reserved_usd": money(reserved)}


def _ledger(root: Path) -> dict[str, Any]:
    path = root / LEDGER_FILENAME
    if not path.exists():
        if any((root / "attempts").iterdir()):
            raise ScreeningCoordinatorError("beta_screening_ledger_missing_after_attempt")
        return {"schema_version": LEDGER_SCHEMA, "campaigns": {}}
    ledger = _read(path)
    if (ledger.get("schema_version") != LEDGER_SCHEMA or not isinstance(ledger.get("campaigns"), dict)
            or ledger.get("ledger_digest") != canonical_digest(ledger, digest_field="ledger_digest")):
        raise ScreeningCoordinatorError("beta_screening_ledger_invalid")
    for site_id in {row["site_id"] for row in ledger["campaigns"].values()}:
        if _amount(_budget(ledger, site_id)["committed_upper_bound_usd"]) > SITE_LIMIT:
            raise ScreeningCoordinatorError("beta_screening_site_budget_exceeded")
    return ledger


def _save(root: Path, ledger: dict[str, Any]) -> None:
    # A site's commitment is shared by all its revisions. Progress already
    # published for another campaign must receive a new revision if that
    # shared budget changes; same-revision bytes must never drift.
    for state in ledger["campaigns"].values():
        budget = _budget(ledger, state["site_id"])
        previous = state.get("site_budget")
        if previous is not None and previous != budget:
            _changed(state)
        state["site_budget"] = budget
    ledger["ledger_digest"] = canonical_digest(ledger, digest_field="ledger_digest")
    _write(root / LEDGER_FILENAME, ledger)


def _changed(state: dict[str, Any]) -> None:
    state["revision"] += 1
    state["updated_at"] = _now()


def _plan(root: Path, campaign_id: str, expected_digest: str | None = None) -> dict[str, Any]:
    from .beta_policy_screening_plan import validate_screening_plan

    if not _ID.fullmatch(campaign_id):
        raise ScreeningCoordinatorError("beta_screening_campaign_id_invalid")
    plan = validate_screening_plan(_read(root / "plans" / f"{campaign_id}.json"))
    if plan["campaign_id"] != campaign_id or expected_digest and plan["plan_digest"] != expected_digest:
        raise ScreeningCoordinatorError("beta_screening_campaign_identity_changed")
    return plan


def enqueue_screening_campaign(plan: Mapping[str, Any], *, queue_root: str | Path | None = None) -> dict[str, Any]:
    """Idempotently retain one immutable plan; no allocation or budget reset."""
    from .beta_policy_screening_plan import validate_screening_plan

    value = validate_screening_plan(plan)
    campaign_id = value["campaign_id"]
    if not _ID.fullmatch(campaign_id) or not value.get("site_id") or _amount(value["budget_usd"]) > SITE_LIMIT:
        raise ScreeningCoordinatorError("beta_screening_campaign_scope_invalid")
    root = _root(queue_root)
    with _lock(root):
        ledger = _ledger(root)
        old = ledger["campaigns"].get(campaign_id)
        if old:
            if old["plan_digest"] != value["plan_digest"]:
                raise ScreeningCoordinatorError("beta_screening_campaign_identity_changed")
            return {"campaign_id": campaign_id, "status": "already_enqueued", "plan_digest": value["plan_digest"]}
        path = root / "plans" / f"{campaign_id}.json"
        if path.exists():
            if _read(path) != value:
                raise ScreeningCoordinatorError("beta_screening_campaign_identity_changed")
        else:
            _write(path, value)
        timestamp = _now()
        ledger["campaigns"][campaign_id] = {
            "plan_digest": value["plan_digest"], "site_id": value["site_id"],
            "revision": 1, "created_at": timestamp, "updated_at": timestamp,
            "published_revision": 0, "published_digest": None,
            "candidates": {candidate["candidate_id"]: _initial_state(candidate) for candidate in value["candidates"]},
        }
        _save(root, ledger)
    return {"campaign_id": campaign_id, "status": "enqueued", "plan_digest": value["plan_digest"]}


COMMIT_CHANGED_BLOCKER = "beta_screening_implementation_commit_changed"
REPLANNED_BLOCKER = "beta_screening_replanned_for_new_implementation"


def _spent(row: Mapping[str, Any]) -> bool:
    """A candidate that ran or may have allocated is never planned again automatically."""
    attempts = [*row.get("attempt_history", []), *([row["attempt"]] if row.get("attempt") else [])]
    return (row["status"] in {"completed", *ACTIVE_STATUSES}
            or any(attempt["reservation_status"] != "released" or attempt.get("provider_allocation_observed")
                   for attempt in attempts))


def _replan_scope(ledger: Mapping[str, Any], plan: Mapping[str, Any], implementation_commit: str) -> list[str] | None:
    state = ledger["campaigns"][plan["campaign_id"]]
    rows = state["candidates"]
    if state.get("superseded_by") or any(row["status"] in ACTIVE_STATUSES for row in rows.values()):
        return None
    if _amount(_budget(ledger, state["site_id"], campaign_id=plan["campaign_id"])["reserved_usd"]) > 0:
        return None
    stale = False
    for candidate in plan["candidates"]:
        row = rows[candidate["candidate_id"]]
        frozen = (candidate.get("execution_spec") or {}).get("implementation_commit")
        if (row["status"] in QUEUED_STATUSES and frozen not in (None, implementation_commit)
                or row["status"] == "blocked" and COMMIT_CHANGED_BLOCKER in row.get("blockers", [])):
            stale = True
    if not stale:
        return None
    return [candidate["candidate_id"] for candidate in plan["candidates"]
            if not _spent(rows[candidate["candidate_id"]])] or None


def replan_scope(campaign_id: str, *, implementation_commit: str,
                 queue_root: str | Path | None = None) -> list[str] | None:
    """Candidates to plan again because the deployed code changed, or ``None``.

    A plan freezes the commit its candidates were prepared against, and
    preparation refuses another. When a deploy lands while some candidate is
    still unprepared (queued, or refused for the changed commit) and the
    campaign holds no reservation, every candidate that never ran or allocated
    is planned again at the new commit. Spent candidates keep their result.
    """
    if not re.fullmatch(r"[0-9a-f]{40}", implementation_commit):
        raise ScreeningCoordinatorError("beta_screening_implementation_commit_invalid")
    root = _root(queue_root)
    with _lock(root):
        ledger = _ledger(root)
        state = ledger["campaigns"].get(campaign_id)
        if state is None:
            return None
        return _replan_scope(ledger, _plan(root, campaign_id, state["plan_digest"]), implementation_commit)


def latest_campaign(campaign_id: str, *, queue_root: str | Path | None = None) -> str:
    """Follow re-plans from a source's first campaign to the one that is current."""
    root = _root(queue_root)
    with _lock(root):
        campaigns = _ledger(root)["campaigns"]
        seen = {campaign_id}
        while (successor := (campaigns.get(campaign_id) or {}).get("superseded_by")) is not None:
            if successor in seen:
                raise ScreeningCoordinatorError("beta_screening_supersession_cycle")
            seen.add(successor)
            campaign_id = successor
    return campaign_id


def supersede_campaign(campaign_id: str, plan: Mapping[str, Any], *, implementation_commit: str,
                       queue_root: str | Path | None = None) -> dict[str, Any]:
    """Atomically enqueue a re-plan and close the old campaign's unprepared rows.

    One ledger write does both, rechecking under the lock that the old campaign
    holds no reservation and that the new plan covers only unspent candidates
    of the same site and source. The site's budget is shared, so the new plan
    can never spend what the old one committed.
    """
    from .beta_policy_screening_plan import validate_screening_plan

    value = validate_screening_plan(plan)
    successor = value["campaign_id"]
    if not _ID.fullmatch(successor) or _amount(value["budget_usd"]) > SITE_LIMIT:
        raise ScreeningCoordinatorError("beta_screening_campaign_scope_invalid")
    root = _root(queue_root)
    with _lock(root):
        ledger = _ledger(root)
        old = ledger["campaigns"].get(campaign_id)
        if old is None:
            raise ScreeningCoordinatorError("beta_screening_campaign_unknown")
        if old.get("superseded_by") == successor:
            existing = ledger["campaigns"].get(successor)
            if existing is None or existing["plan_digest"] != value["plan_digest"]:
                raise ScreeningCoordinatorError("beta_screening_campaign_identity_changed")
            return {"campaign_id": successor, "status": "already_enqueued", "plan_digest": value["plan_digest"],
                    "superseded_campaign_id": campaign_id}
        previous = _plan(root, campaign_id, old["plan_digest"])
        scope = _replan_scope(ledger, previous, implementation_commit)
        if (scope is None or successor in ledger["campaigns"] or value["site_id"] != old["site_id"]
                or value["source"] != previous["source"]
                or not {row["candidate_id"] for row in value["candidates"]} <= set(scope)):
            raise ScreeningCoordinatorError("beta_screening_replan_not_admitted")
        path = root / "plans" / f"{successor}.json"
        if path.exists():
            if _read(path) != value:
                raise ScreeningCoordinatorError("beta_screening_campaign_identity_changed")
        else:
            _write(path, value)
        for row in old["candidates"].values():
            if row["status"] in QUEUED_STATUSES:
                row.update(status="blocked", blockers=[REPLANNED_BLOCKER])
        old["superseded_by"] = successor
        _changed(old)
        timestamp = _now()
        ledger["campaigns"][successor] = {
            "plan_digest": value["plan_digest"], "site_id": value["site_id"],
            "revision": 1, "created_at": timestamp, "updated_at": timestamp,
            "published_revision": 0, "published_digest": None, "supersedes": campaign_id,
            "candidates": {candidate["candidate_id"]: _initial_state(candidate) for candidate in value["candidates"]},
        }
        _save(root, ledger)
    return {"campaign_id": successor, "status": "enqueued", "plan_digest": value["plan_digest"],
            "superseded_campaign_id": campaign_id}


def _initial_state(candidate: Mapping[str, Any]) -> dict[str, Any]:
    status = candidate["status"]
    blockers = candidate["blockers"]
    if status == "infeasible":
        # The frozen plan keeps ``infeasible``; the published vocabulary has no
        # such state yet, so the site sees it unavailable with its measured reason.
        status, blockers = "unavailable", [INFEASIBLE_BLOCKER, *blockers]
    if status == "waitlisted":
        # The plan's reason: the site allowance, or the website job's ceiling.
        return {"status": "waitlisted", "blockers": _safe_blockers(blockers, WAITLIST_BLOCKER)}
    return {"status": "pending" if status == "selected" else status,
            "blockers": _safe_blockers(blockers, "beta_screening_candidate_unavailable") if blockers else []}


def _safe_blockers(value: Any, fallback: str) -> list[str]:
    # Backend exception/provider text can contain private URLs or tokens. Public
    # status carries typed codes only; full diagnostics stay in allocator logs.
    codes = [item for item in value or [] if isinstance(item, str) and _BLOCKER.fullmatch(item)]
    return sorted(set(codes)) or [fallback]


def _quote_current(candidate: Mapping[str, Any]) -> bool:
    quote = candidate.get("cost_quote") or {}
    try:
        deadline = datetime.fromisoformat(str(quote["valid_until"]).replace("Z", "+00:00"))
        return deadline.tzinfo is not None and deadline > datetime.now(timezone.utc)
    except (ValueError, KeyError):
        return False


def _new_attempt(plan: Mapping[str, Any], candidate: Mapping[str, Any], *, generation: int = 0) -> dict[str, Any]:
    identity = f"{plan['campaign_id']}:{candidate['candidate_id']}:{plan['plan_digest']}"
    if generation:
        identity += f":capacity-generation-{generation}"
    return {"attempt_id": "screen-" + hashlib.sha256(identity.encode()).hexdigest()[:32],
            "generation": generation, "maximum_cost_usd": money(_amount(candidate["max_cost_usd"])),
            "reservation_status": "reserved", "provider_allocation_observed": False}


def _retry_after_capacity() -> str:
    return (datetime.now(timezone.utc) + timedelta(seconds=CAPACITY_RETRY_SECONDS)).isoformat()


def _capacity_retry_ready(attempt: Mapping[str, Any]) -> bool:
    deadline = attempt.get("retry_not_before")
    if deadline is None:
        return True
    try:
        instant = datetime.fromisoformat(str(deadline).replace("Z", "+00:00"))
        return instant.tzinfo is not None and instant <= datetime.now(timezone.utc)
    except ValueError:
        return False


def _capacity_refusal_proof(root: Path, attempt: Mapping[str, Any], observation: Mapping[str, Any]) -> bool:
    """Only the retained canonical pre-create slot refusal can authorize retry."""
    from .task_evaluation_scene_execution_authority import proves_no_provider_allocation

    if (observation.get("retryable_capacity_refusal") is not True
            or observation.get("no_policy_api_calls_proven") is not True
            or observation.get("blockers") != ["vast_paid_launch_lock_busy"]):
        return False
    record = observation.get("capacity_refusal_receipt")
    if not isinstance(record, Mapping):
        return False
    path = root / "attempts" / attempt["attempt_id"] / "allocator_result.json"
    try:
        if path.is_symlink() or record.get("path") != str(path):
            return False
        raw = path.read_bytes()
        receipt = json.loads(raw)
        return (record.get("size_bytes") == len(raw)
                and record.get("sha256") == "sha256:" + hashlib.sha256(raw).hexdigest()
                and isinstance(receipt, Mapping) and proves_no_provider_allocation(receipt)
                and receipt.get("status") == "blocked"
                and receipt.get("blockers") == ["vast_paid_launch_lock_busy"])
    except (OSError, ValueError):
        return False


def _reserve(root: Path, plan: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, Any] | None:
    campaign_id, candidate_id = plan["campaign_id"], candidate["candidate_id"]
    with _lock(root):
        ledger = _ledger(root)
        state = ledger["campaigns"][campaign_id]
        row = state["candidates"][candidate_id]
        if row["status"] != "pending":
            return None
        cap = _amount(candidate["max_cost_usd"])
        site = _budget(ledger, plan["site_id"])
        campaign = _budget(ledger, plan["site_id"], campaign_id=campaign_id)
        blocker = None
        if not _quote_current(candidate):
            blocker = "beta_screening_cost_quote_expired"
        elif cap <= 0:
            blocker = "beta_screening_cost_bound_invalid"
        elif (_amount(site["committed_upper_bound_usd"]) + cap > SITE_LIMIT
              or _amount(campaign["committed_upper_bound_usd"]) + cap > _amount(plan["budget_usd"])):
            # Another attempt holds the money it needs. That attempt may still
            # release it before spend, so wait for the refill to decide.
            row.update(status="waitlisted", blockers=[WAITLIST_BLOCKER])
            _changed(state)
            _save(root, ledger)
            return None
        if blocker:
            row.update(status="deferred", blockers=[blocker])
            _changed(state)
            _save(root, ledger)
            return None
        attempt = _new_attempt(plan, candidate)
        row.update(status="preparing", blockers=[], attempt=attempt)
        _changed(state)
        _save(root, ledger)
        return dict(attempt)


def _refill_waitlist(root: Path) -> int:
    """Fund waitlisted candidates from money released before spend.

    Plan order decides who is funded first. A row is funded only if its whole
    cap fits under both the site allowance and its plan's budget, after every
    reservation, commitment and already pending (funded, unreserved) row. Once
    the site holds no reservation and no pending row, nothing can release money
    any more: every row that still does not fit closes as ``deferred``.
    Returns the number of rows changed.
    """
    changed = 0
    with _lock(root):
        ledger = _ledger(root)
        campaigns = ledger["campaigns"]
        sites = sorted({state["site_id"] for state in campaigns.values()
                        if any(row["status"] == "waitlisted" for row in state["candidates"].values())})
        if not sites:
            return 0
        plans: dict[str, dict[str, Any]] = {}

        def plan_for(campaign_id: str) -> dict[str, Any]:
            if campaign_id not in plans:
                plans[campaign_id] = _plan(root, campaign_id, campaigns[campaign_id]["plan_digest"])
            return plans[campaign_id]

        def pending_caps(campaign_id: str) -> Decimal:
            rows = campaigns[campaign_id]["candidates"]
            if not any(row["status"] == "pending" for row in rows.values()):
                return Decimal(0)
            return sum((_amount(candidate["max_cost_usd"]) for candidate in plan_for(campaign_id)["candidates"]
                        if rows[candidate["candidate_id"]]["status"] == "pending"), Decimal(0))

        for site_id in sites:
            ids = sorted((key for key, state in campaigns.items() if state["site_id"] == site_id),
                         key=lambda key: (campaigns[key]["created_at"], key))
            site = _budget(ledger, site_id)
            pending = {key: pending_caps(key) for key in ids}
            site_used = _amount(site["committed_upper_bound_usd"]) + sum(pending.values(), Decimal(0))
            in_flight = _amount(site["reserved_usd"]) > 0 or any(pending.values())
            for campaign_id in ids:
                state = campaigns[campaign_id]
                if not any(row["status"] == "waitlisted" for row in state["candidates"].values()):
                    continue
                plan = plan_for(campaign_id)
                campaign_used = (_amount(_budget(ledger, site_id, campaign_id=campaign_id)["committed_upper_bound_usd"])
                                 + pending[campaign_id])
                touched = False
                for candidate in plan["candidates"]:
                    row = state["candidates"][candidate["candidate_id"]]
                    if row["status"] != "waitlisted":
                        continue
                    cap = _amount(candidate["max_cost_usd"])
                    if not _quote_current(candidate):
                        row.update(status="deferred", blockers=["beta_screening_cost_quote_expired"])
                    elif site_used + cap <= SITE_LIMIT and campaign_used + cap <= _amount(plan["budget_usd"]):
                        row.update(status="pending", blockers=[])
                        site_used += cap
                        campaign_used += cap
                        in_flight = True
                    elif not in_flight:
                        row.update(status="deferred", blockers=[BUDGET_EXHAUSTED_BLOCKER])
                    else:
                        continue
                    touched = True
                    changed += 1
                if touched:
                    _changed(state)
        if changed:
            _save(root, ledger)
    return changed


def _update(root: Path, campaign_id: str, candidate_id: str, update) -> dict[str, Any]:
    with _lock(root):
        ledger = _ledger(root)
        state = ledger["campaigns"][campaign_id]
        row = state["candidates"][candidate_id]
        before = json.dumps(row, sort_keys=True, allow_nan=False)
        update(row)
        if json.dumps(row, sort_keys=True, allow_nan=False) != before:
            _changed(state)
            _save(root, ledger)
        return json.loads(json.dumps(row))


def _observe(root: Path, plan: Mapping[str, Any], candidate: Mapping[str, Any], observation: Mapping[str, Any]) -> None:
    if not isinstance(observation, Mapping) or (observation.get("handle") is not None
                                               and not isinstance(observation["handle"], Mapping)):
        observation = {"status": "reconciliation_required", "blockers": ["beta_screening_backend_response_invalid"]}

    def apply(row):
        attempt = row["attempt"]
        if observation.get("handle") is not None:
            attempt["handle"] = dict(observation["handle"])
        allocated = observation.get("provider_allocation_performed") is True
        attempt["provider_allocation_observed"] = attempt["provider_allocation_observed"] or allocated
        status = observation.get("status")
        if status == "running":
            row.update(status="running", blockers=[])
            return
        no_allocation = (observation.get("no_allocation_proven") is True
                         and observation.get("provider_allocation_performed") is False
                         and attempt["provider_allocation_observed"] is False)
        closed = observation.get("provider_zero_verified") is True and attempt["provider_allocation_observed"] is True
        if status in {"completed", "blocked"} and no_allocation:
            if _capacity_refusal_proof(root, attempt, observation):
                # This is a proven zero-GPU/zero-API pre-create refusal, not
                # another paid attempt. Retain every old intent and proof;
                # atomically transfer the same reservation to a fresh identity.
                attempt.update(reservation_status="released", closed_at=_now(),
                               terminal_reason="vast_paid_launch_lock_busy",
                               capacity_refusal_receipt=dict(observation["capacity_refusal_receipt"]),
                               no_policy_api_calls_proven=True)
                row.setdefault("attempt_history", []).append(json.loads(json.dumps(attempt)))
                row["attempt"] = {**_new_attempt(plan, candidate, generation=attempt.get("generation", 0) + 1),
                                  "retry_not_before": _retry_after_capacity()}
                row.update(status="preparing", blockers=[CAPACITY_WAIT_BLOCKER])
                return
            attempt["reservation_status"] = "released"
            row.update(status="blocked", blockers=_safe_blockers(observation.get("blockers"),
                                                                "beta_screening_not_allocated"))
            return
        if status in {"completed", "blocked"} and closed:
            # Charged at the full reserved bound even if a billing estimate is
            # lower: only separately reconciled authority could replenish it.
            attempt["reservation_status"] = "committed"
            if isinstance(observation.get("result"), Mapping):
                try:
                    summary = candidate_result_summary(observation["result"], plan=plan, candidate=candidate)
                except (ValueError, TypeError, KeyError):
                    row.update(status="blocked", blockers=["beta_screening_native_result_invalid"])
                else:
                    row.update(status="completed", blockers=[], summary=summary)
            else:
                row.update(status="blocked", blockers=_safe_blockers(observation.get("blockers"),
                                                                    "beta_screening_native_result_missing"))
            return
        row.update(status="reconciliation_required", blockers=_safe_blockers(observation.get("blockers"),
                                                                            "beta_screening_allocation_uncertain"))
    _update(root, plan["campaign_id"], candidate["candidate_id"], apply)


def _attempt_root(root: Path, attempt: Mapping[str, Any]) -> Path:
    path = root / "attempts" / attempt["attempt_id"]
    if path.is_symlink():
        raise ScreeningCoordinatorError("beta_screening_attempt_path_unsafe")
    path.mkdir(exist_ok=True, mode=0o700)
    return path


def _advance_candidate(root: Path, plan: Mapping[str, Any], candidate: Mapping[str, Any], *, executor: Any,
                       require_publication_ack: bool) -> bool:
    with _lock(root):
        ledger = _ledger(root)
        state = ledger["campaigns"][plan["campaign_id"]]
        row = state["candidates"][candidate["candidate_id"]]
        if require_publication_ack and row["status"] in {"pending", "preparing"}:
            publication = screening_publication(plan=plan, state=state, budget=_budget(ledger, state["site_id"]))
            if (state["published_revision"] != state["revision"]
                    or state["published_digest"] != publication["publication_digest"]):
                return False
    # Admission above occurs before reserving/preparing a new attempt. Progress
    # within this call changes the publication but does not require an extra
    # round trip between local staging and spawn. A restarted preparing attempt
    # must pass the latest acknowledgment again. Already committed launches
    # always reconcile, even while the website is unavailable.
    if row["status"] == "pending":
        attempt = _reserve(root, plan, candidate)
        if attempt is None:
            return False
        row = {"status": "preparing", "attempt": attempt}
    if row["status"] not in ACTIVE_STATUSES:
        return False
    attempt = row["attempt"]
    if row["status"] == "preparing" and not _capacity_retry_ready(attempt):
        return False
    attempt_dir = _attempt_root(root, attempt)
    if row["status"] != "preparing":
        try:
            observed = executor.reconcile_candidate(plan=plan, candidate=candidate, attempt_dir=attempt_dir,
                                                    prepared=attempt.get("prepared") or {},
                                                    handle=attempt.get("handle"))
        except Exception:  # noqa: BLE001 - uncertainty retains its reservation and never repeats launch
            observed = {"status": "reconciliation_required", "blockers": ["beta_screening_reconciliation_failed"]}
        _observe(root, plan, candidate, observed)
        return False
    # Preparation can be retried after a coordinator crash: its contract is
    # local staging only. No launch is repeated once a launch intent exists.
    try:
        prepared = executor.prepare_candidate(plan=plan, candidate=candidate, attempt_dir=attempt_dir)
        if not isinstance(prepared, Mapping):
            raise ValueError("beta_screening_prepared_record_invalid")
    except Exception as exc:  # noqa: BLE001 - a pre-spend refusal does not prevent unrelated candidates
        from .bounded_error import error_description

        blockers = _safe_blockers([str(exc)], "beta_screening_preparation_failed")
        # Only the blocker codes are published. The attempt keeps the type and
        # a redacted, trimmed message of a failure the generic code stands for.
        described = error_description(exc) if blockers == ["beta_screening_preparation_failed"] else None

        def blocked(item):
            item["attempt"]["reservation_status"] = "released"
            if described is not None:
                item["attempt"]["error"] = described
            item.update(status="blocked", blockers=blockers)
        _update(root, plan["campaign_id"], candidate["candidate_id"], blocked)
        return False
    if not _quote_current(candidate):
        def expired(item):
            item["attempt"]["reservation_status"] = "released"
            item.update(status="deferred", blockers=["beta_screening_cost_quote_expired"])
        _update(root, plan["campaign_id"], candidate["candidate_id"], expired)
        return False

    capacity_preflight = getattr(executor, "preflight_candidate_capacity", None)
    if callable(capacity_preflight):
        try:
            capacity = capacity_preflight(plan=plan, candidate=candidate, attempt_dir=attempt_dir, prepared=prepared)
            if not isinstance(capacity, Mapping) or type(capacity.get("capacity_available")) is not bool:
                raise ValueError("beta_screening_capacity_preflight_invalid")
            if capacity["capacity_available"] is False:
                if capacity.get("blockers") not in (["vast_paid_launch_lock_busy"],
                                                    ["vast_release_deployment_in_progress"]):
                    raise ValueError("beta_screening_capacity_preflight_invalid")

                def waiting(item):
                    item["attempt"].update(prepared=dict(prepared), retry_not_before=_retry_after_capacity())
                    item.update(status="preparing", blockers=[CAPACITY_WAIT_BLOCKER])
                _update(root, plan["campaign_id"], candidate["candidate_id"], waiting)
                return False
        except Exception as exc:  # noqa: BLE001 - this boundary is read-only and strictly pre-intent
            from .bounded_error import error_description

            described = error_description(exc)

            def refused(item):
                item["attempt"]["reservation_status"] = "released"
                item["attempt"]["error"] = described
                item.update(status="blocked", blockers=["beta_screening_capacity_preflight_failed"])
            _update(root, plan["campaign_id"], candidate["candidate_id"], refused)
            return False

    def launching(item):
        item["attempt"]["prepared"] = dict(prepared)
        item["attempt"]["launch_intent_at"] = _now()
        item.update(status="launching", blockers=[])
    _update(root, plan["campaign_id"], candidate["candidate_id"], launching)
    try:
        observed = executor.launch_candidate(plan=plan, candidate=candidate, attempt_dir=attempt_dir, prepared=prepared)
    except Exception:  # noqa: BLE001 - a failed response does not prove a failed allocation
        observed = {"status": "reconciliation_required", "blockers": ["beta_screening_launch_response_unknown"]}
    _observe(root, plan, candidate, observed)
    return True


def process_screening_queue(
    *, queue_root: str | Path | None = None, max_campaigns: int = 4,
    max_launches: int = 1, executor: Any = None, require_publication_ack: bool = True,
) -> dict[str, Any]:
    """Advance bounded work after exact website acknowledgment; reconcile always.

    ``require_publication_ack=False`` is for hermetic lifecycle rehearsals. The
    production worker retains the default and the real canonical allocator.
    """
    if type(max_campaigns) is not int or not 1 <= max_campaigns <= 100:
        raise ScreeningCoordinatorError("beta_screening_max_campaigns_invalid")
    if type(max_launches) is not int or not 0 <= max_launches <= 100:
        raise ScreeningCoordinatorError("beta_screening_max_launches_invalid")
    if type(require_publication_ack) is not bool:
        raise ScreeningCoordinatorError("beta_screening_publication_ack_requirement_invalid")
    if executor is None:
        from . import beta_policy_screening_launch as executor
    root = _root(queue_root)
    try:
        with _lock(root, ".worker.lock", nonblocking=True):
            # Money a candidate released before spend funds the next waitlisted
            # one. A funded row is pending and still needs the website's
            # acknowledgment of that revision before it is reserved.
            _refill_waitlist(root)
            with _lock(root):
                ledger = _ledger(root)
                active = [key for key, state in ledger["campaigns"].items()
                          if campaign_status(state["candidates"]) in {"queued", "running", "reconciliation_required"}]
                last = ledger.get("last_processed_campaign", "")
                active = sorted(key for key in active if key > last) + sorted(key for key in active if key <= last)
            launches = 0
            processed = []
            for campaign_id in active[:max_campaigns]:
                state = ledger["campaigns"][campaign_id]
                plan = _plan(root, campaign_id, state["plan_digest"])
                # Reconcile allocated/uncertain work before reserving another
                # candidate; new work follows the plan's (ranked) priority order.
                position = {row["candidate_id"]: index for index, row in enumerate(plan["candidates"])}
                ordered = sorted(plan["candidates"], key=lambda value: (
                    state["candidates"][value["candidate_id"]]["status"] not in ACTIVE_STATUSES,
                    position[value["candidate_id"]],
                ))
                for candidate in ordered:
                    current = state["candidates"][candidate["candidate_id"]]["status"]
                    if current == "pending" and launches >= max_launches:
                        continue
                    if current == "preparing" and launches >= max_launches:
                        continue
                    launches += _advance_candidate(root, plan, candidate, executor=executor,
                                                   require_publication_ack=require_publication_ack)
                processed.append(campaign_id)
                # A permanently uncertain provider response must not starve
                # unrelated sites. The reservation stays held while bounded
                # worker passes rotate through every active campaign.
                with _lock(root):
                    current_ledger = _ledger(root)
                    current_ledger["last_processed_campaign"] = campaign_id
                    _save(root, current_ledger)
            # Close or fund the waitlist after this pass's refusals as well, so
            # the publication that follows already says what happens next.
            refilled = _refill_waitlist(root)
            return {"status": "processed", "campaign_ids": processed, "launches_started": launches,
                    "waitlist_changes": refilled}
    except BlockingIOError:
        return {"status": "worker_busy", "campaign_ids": [], "launches_started": 0}


def pending_publications(*, queue_root: str | Path | None = None,
                         refresh_startable: bool = False) -> list[dict[str, Any]]:
    """Latest immutable payload per campaign; a failed delivery can be retried."""
    root = _root(queue_root)
    values = []
    with _lock(root):
        ledger = _ledger(root)
        invalidated = False
        for campaign_id, state in ledger["campaigns"].items():
            if refresh_startable and any(row["status"] in {"pending", "preparing"}
                                         for row in state["candidates"].values()):
                # A receipt from before a restart is not a fresh source check.
                # Persist invalidation before sending; failure cannot leave an
                # old accepted receipt eligible for a new paid attempt.
                state["published_revision"] = 0
                state["published_digest"] = None
                invalidated = True
            plan = _plan(root, campaign_id, state["plan_digest"])
            publication = screening_publication(plan=plan, state=state, budget=_budget(ledger, state["site_id"]))
            if (state["published_revision"] == publication["revision"]
                    and state["published_digest"] == publication["publication_digest"]):
                continue
            if state["published_revision"] == state["revision"]:
                # Acknowledged bytes changed without progress (a wire contract
                # upgrade). Send them as a new revision; the website refuses a
                # same-revision rewrite, which would leave the result unseen.
                _changed(state)
                invalidated = True
                publication = screening_publication(plan=plan, state=state, budget=_budget(ledger, state["site_id"]))
            _write(root / "publications" / f"{campaign_id}.json", publication)
            values.append(publication)
        if invalidated:
            _save(root, ledger)
    return values


def acknowledge_publication(
    campaign_id: str, publication_digest: str, *, queue_root: str | Path | None = None,
) -> bool:
    """Acknowledge only current bytes; a late response cannot swallow newer progress."""
    root = _root(queue_root)
    with _lock(root):
        ledger = _ledger(root)
        state = ledger["campaigns"].get(campaign_id)
        if state is None:
            raise ScreeningCoordinatorError("beta_screening_campaign_unknown")
        plan = _plan(root, campaign_id, state["plan_digest"])
        publication = screening_publication(plan=plan, state=state, budget=_budget(ledger, state["site_id"]))
        if publication["publication_digest"] != publication_digest:
            return False
        state["published_revision"] = state["revision"]
        state["published_digest"] = publication_digest
        _save(root, ledger)
        return True


__all__ = ["enqueue_screening_campaign", "process_screening_queue", "pending_publications",
           "acknowledge_publication", "latest_campaign", "replan_scope", "supersede_campaign",
           "ScreeningCoordinatorError"]
