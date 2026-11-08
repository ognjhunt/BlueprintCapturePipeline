"""Durable, bounded beta screening over the canonical policy allocator.

Beta journey: a prepared site's candidate screening. A single atomic ledger
holds campaign state and site commitments, so a crash cannot publish a launch
without its reservation. The expensive worker lives behind the existing
allocator bridge; this coordinator never invokes a provider API. Ambiguous
launches retain their reservation and are reconciled, never retried blindly.

The standing allowance is USD 30 (``SITE_LIMIT``) for the stable website site submission,
shared across candidate sets and scene revisions. Once allocation is observed,
the entire reserved upper bound is conservatively charged until the attempt
settles: once its own records prove the run is over and bound to it, it counts
at its budget ledger's estimate rounded up to the cent
(``beta_policy_screening_settlement``). Any doubt keeps the whole bound. A
campaign completed within the allowance is the job's evaluation result; spend
beyond it needs the owner's approval and is never reserved by this queue.

Settlement lowers only what funding counts (``_budget``). The WebApp refuses a
publication whose ``spent_upper_bound_usd`` is below the campaign's previous
one, so each campaign publishes its own spent bound as a floor
(``_publication_budget``); its committed total is never below the real one. A
campaign the WebApp already holds in full is not sent again for a settlement.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import tempfile
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

from .beta_policy_screening_result import (
    ACTIVE_STATUSES, QUEUED_STATUSES, candidate_result_summary, campaign_status, money, screening_publication,
)
from .beta_policy_screening_settlement import settled_cost
from .decision_evidence_contracts import canonical_digest
from .policy_execution_evidence import SCRIPTED
from .policy_canary_cell_setup_failure import MACHINE_FAULT_BLOCKERS, SETUP_FAILURE_BLOCKERS
from .website_candidate_feasibility import is_measured_refusal

QUEUE_ROOT_ENV = "BLUEPRINT_BETA_POLICY_SCREENING_QUEUE_ROOT"
DEFAULT_QUEUE_ROOT = "/var/lib/blueprint/pipeline-control-plane/beta-policy-screening"
# The bulk volume new attempts are reserved under (the unit sets
# /mnt/blueprint-work/beta-policy-screening-work): each attempt's sealed bundle, allocator job
# directory and launch receipts live in ``<work root>/attempts/<attempt id>``. The ledger, plans
# and publications stay in the queue root, beside a pointer per attempt. The ledger records each
# attempt's directory, so an attempt reserved without it (unset) keeps ``<queue>/attempts/<id>``.
WORK_ROOT_ENV = "BLUEPRINT_BETA_POLICY_SCREENING_WORK_ROOT"
ATTEMPT_DIR_FIELD = "attempt_dir"
ATTEMPT_POINTER_SCHEMA = "blueprint.beta_policy_screening_attempt_pointer.v1"
LEDGER_SCHEMA = "blueprint.beta_policy_screening_ledger.v1"
LEDGER_FILENAME = "screening_ledger.v1.json"
#: The site's screening allowance across its campaigns (raised from $15 on 2026-11-05, owner
#: approval). Each website job's ceiling still bounds what a job spends in all.
SITE_LIMIT = Decimal("30")
CAPACITY_RETRY_SECONDS = 60
CAPACITY_WAIT_BLOCKER = "beta_screening_waiting_for_capacity"
# No rentable GPU met the run's price, driver and region rules when the allocator searched. Nothing
# was created, so the attempt waits and searches again: 2, 4, 8, 16, then every 30 minutes, at most
# OFFER_SHORTAGE_RETRY_CAP times (about five hours) before the candidate is blocked by name.
OFFER_WAIT_BLOCKER = "beta_screening_waiting_for_gpu_capacity"
OFFER_SHORTAGE_EXHAUSTED_BLOCKER = "beta_screening_gpu_capacity_unavailable"
OFFER_SHORTAGE_RETRY_CAP = 12
OFFER_SHORTAGE_FIRST_RETRY_SECONDS = 120
OFFER_SHORTAGE_MAX_RETRY_SECONDS = 1800
# The spend admission lock (live billing evidence) was older than its window when the launch
# preflight read it, as it is for a few minutes after a deploy restarts its refresh: the money
# check refused before anything was created. The check stays: the attempt waits and the next one
# reads a fresh lock, 2, 4, 8, 16, then 30 minutes apart, at most SPEND_LOCK_RETRY_CAP times. A
# stale lock beside any other spend blocker is not this wait and keeps its refusal.
SPEND_LOCK_STALE_BLOCKER = "spend_admission:spend_admission_lock_stale"
SPEND_LOCK_RETRY_CAP = 6
#: The same stale-lock refusal as the G1 lane's allocator words it, without the ``spend_admission:`` prefix
#: (2026-11-08 02:40Z, the library site's pi0.5 row: blocked outright instead of waiting).
SPEND_LOCK_STALE_CODES = frozenset({SPEND_LOCK_STALE_BLOCKER, "spend_admission_lock_stale"})
_SPEND_LOCK_REFUSAL = frozenset({*SPEND_LOCK_STALE_CODES, "paid_resource_admission_has_blockers",
                                 "paid_resource_admission_not_admitted"})
# A candidate retries a refusal whose control read got no answer (a 429, a 5xx,
# a timeout) at most this many times in all: a launch preflight read again, or
# a fresh attempt after its allocator fenced its own create. Each fresh attempt
# rebuilds and restages its bundle and re-arms the watchdog. The next such
# refusal blocks only this candidate with TRANSPORT_RETRY_CAP_BLOCKER and
# releases its reservation as any other no-spend block does. Retries wait
# CAPACITY_RETRY_SECONDS, doubled after each one (1, 2, then 4 minutes).
TRANSPORT_RETRY_CAP = 3
TRANSPORT_RETRY_CAP_BLOCKER = "beta_screening_transport_retry_cap_exhausted"
# The launch module's retry_cause for a fresh attempt after an unanswered read.
TRANSPORT_RETRY_CAUSE = "control_unanswered"
# Published beside the capacity code, the only one the WebApp accepts for a return to preparing.
CONTROL_WAIT_BLOCKER = "beta_screening_waiting_for_control"
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
    """What the site (or one campaign) holds, as funding counts it.

    A committed attempt counts at its recorded settlement, else at its whole
    maximum; a reserved one at its maximum.
    """
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
                    settled = settled_cost(attempt)
                    spent += cap if settled is None else settled
                elif state != "released":
                    raise ScreeningCoordinatorError("beta_screening_reservation_state_invalid")
    return {"currency": "USD", "limit_usd": money(SITE_LIMIT),
            "committed_upper_bound_usd": money(reserved + spent),
            "spent_upper_bound_usd": money(spent), "reserved_usd": money(reserved)}


def _publication_budget(ledger: Mapping[str, Any], state: Mapping[str, Any]) -> dict[str, Any]:
    """The site budget one campaign would publish now: the WebApp's rules hold for every revision.

    The WebApp refuses a ``spent_upper_bound_usd`` below the campaign's previous
    publication (``screeningPublicationTransition``) and any committed total
    above the $15 limit. A settlement recorded after a campaign published an
    attempt at its whole maximum lowers the site's spend below that, so the
    campaign keeps its last spent bound (``site_budget``, never below anything
    it sent) as a floor. Its committed total is the larger of the site's real
    commitment and that floor, and its reserved figure is the part of that
    total above the floor: the total never understates what the site holds
    (the WebApp's job ceiling reads it) and never exceeds the limit. A campaign
    planned after the settlement has no floor and publishes the settled figures.
    """
    budget = _budget(ledger, state["site_id"])
    floor = _floor(state)
    if floor <= _amount(budget["spent_upper_bound_usd"]):
        return budget
    committed = max(_amount(budget["committed_upper_bound_usd"]), floor)
    return {**budget, "committed_upper_bound_usd": money(committed), "spent_upper_bound_usd": money(floor),
            "reserved_usd": money(committed - floor)}


def _floor(state: Mapping[str, Any]) -> Decimal:
    return _amount((state.get("site_budget") or {}).get("spent_upper_bound_usd", "0"))


def _published_budget(ledger: Mapping[str, Any], state: Mapping[str, Any]) -> dict[str, Any]:
    """The budget a campaign's publication carries: the snapshot ``_save`` recorded for it."""
    return dict(state.get("site_budget") or _publication_budget(ledger, state))


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
        budget = _publication_budget(ledger, state)
        previous = state.get("site_budget")
        if previous is not None and previous != budget:
            if (state["published_revision"] == state["revision"]
                    and _floor(state) > _amount(_budget(ledger, state["site_id"])["spent_upper_bound_usd"])):
                # A settlement lowered the site's spend below what this campaign
                # already published, and the WebApp holds all its progress: it
                # keeps those acknowledged bytes. The site's next publication,
                # from any campaign, carries the real figures; one from this
                # campaign carries its floor (``_publication_budget``).
                continue
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


#: The launch resolved the provider (zero verified) but our own harness returned no usable
#: result: no episode was scored, so nothing about the policy was measured. The untyped
#: fallback for a closed run with no reason is not here: it is not known to be ours. A cell
#: that could not open the runtime, pass the pre-policy gate or load the policy is named.
HARNESS_FAILURE_BLOCKERS = frozenset({"beta_screening_native_result_missing_or_invalid",
                                      "beta_screening_native_result_invalid",
                                      "g1_beta_native_result_missing_or_invalid",
                                      *SETUP_FAILURE_BLOCKERS})
#: How many times a site's candidate is planned again after a harness failure. It spent money,
#: so it is bounded: only after a deploy, inside the site's existing allowance, one candidate
#: at a time until a run gets through, and a run stops at its first cell that ran no policy.
#: Harness failures are ours, not the policy's: 2026-11-04 took three fixes in a row (Pink
#: startup, the RoboLab import closure), each shown by the run before it. Raised from 4 on
#: 2026-11-05 (owner approval) so candidates whose earlier fixes landed can be proven out, and to
#: 7 the same day: pi05's sixth and seventh runs both failed on the OpenPI client's ``ping_timeout``
#: keyword before its fix (#3399) shipped, so it gets one run on the fixed client.
HARNESS_RETRY_LIMIT = 7
#: A campaign freezes one commit, so a harness failure in it would fail its other candidates the
#: same way: they are held, unspent, and planned again after the next deploy. Matches the WebApp's
#: "launch" category ("Could not start in our simulator. This is ours to fix.").
HARNESS_HOLD_BLOCKER = "beta_screening_held_after_launch_failure"
#: How many times a site's candidate is planned again after its rented machine failed while starting
#: the simulator (``MACHINE_FAULT_BLOCKERS``: a fatal signal before Isaac finished launching, so no
#: policy ran). That is the machine, not our harness or the policy, so it never spends a harness
#: retry (owner decision, 2026-11-05) and needs no deploy: the candidate is planned again at the
#: same commit, on whatever machine the next offer search finds. It still spent money, so it is
#: bounded, inside the site's existing allowance.
MACHINE_FAULT_RETRY_LIMIT = 3


def _machine_faulted(row: Mapping[str, Any]) -> bool:
    blockers = set(row.get("blockers") or [])
    return row.get("status") == "blocked" and bool(blockers) and blockers <= MACHINE_FAULT_BLOCKERS


def _machine_fault_retry_allowed(ledger: Mapping[str, Any], site_id: str, candidate_id: str) -> bool:
    faults = sum(1 for state in ledger["campaigns"].values()
                 if state["site_id"] == site_id and _machine_faulted(state["candidates"].get(candidate_id) or {}))
    return 0 < faults <= MACHINE_FAULT_RETRY_LIMIT


def _measured_nothing(row: Mapping[str, Any]) -> bool:
    """A completed row whose every episode was void: the session closed, but the simulator
    stopped each episode (2026-11-05: all ten before the first policy query), so nothing about
    the policy was measured. Any scored episode is a result and never counts here."""
    summary = row.get("summary") or {}
    outcomes = summary.get("episode_outcomes")
    return (row.get("status") == "completed" and not summary.get("episodes_scored")
            and isinstance(outcomes, list) and bool(outcomes)
            and all(isinstance(outcome, Mapping) and outcome.get("outcome") == "void" for outcome in outcomes))


def _harness_failed(row: Mapping[str, Any]) -> bool:
    """A spent row our harness failed: blocked only by a harness code, or completed with every
    episode void; either way with no episode scored."""
    blockers = set(row.get("blockers") or [])
    return (row.get("status") == "blocked" and bool(blockers) and blockers <= HARNESS_FAILURE_BLOCKERS
            and not (row.get("summary") or {}).get("episodes_scored")) or _measured_nothing(row)


def _harness_failures(ledger: Mapping[str, Any], site_id: str, candidate_id: str) -> int:
    return sum(1 for state in ledger["campaigns"].values()
               if state["site_id"] == site_id and _harness_failed(state["candidates"].get(candidate_id) or {}))


def _harness_retry_allowed(ledger: Mapping[str, Any], site_id: str, candidate_id: str) -> bool:
    """Whether this site's candidate may be planned again: at most ``HARNESS_RETRY_LIMIT`` retries."""
    return 0 < _harness_failures(ledger, site_id, candidate_id) <= HARNESS_RETRY_LIMIT


def _dropped_harness_retries(ledger: Mapping[str, Any], state: Mapping[str, Any],
                             plan: Mapping[str, Any]) -> list[str]:
    """Candidates an earlier plan in this campaign's chain listed, and this plan does not.

    A re-plan keeps only unspent rows and harness failures with a retry left, so a
    candidate whose retries ran out leaves the chain. When ``HARNESS_RETRY_LIMIT`` is
    raised (2026-11-05, on the owner's approval) such a candidate may be retried again.
    It counts here only if the last row the chain holds for it is a harness failure
    (never a result) and the site's retry bound now allows it.
    """
    listed = {candidate["candidate_id"] for candidate in plan["candidates"]}
    latest: dict[str, Mapping[str, Any]] = {}
    seen: set[str] = set()
    previous = state.get("supersedes")
    while isinstance(previous, str) and previous not in seen and previous in ledger["campaigns"]:
        seen.add(previous)
        earlier = ledger["campaigns"][previous]
        for candidate_id, row in earlier["candidates"].items():
            latest.setdefault(candidate_id, row)
        previous = earlier.get("supersedes")
    return sorted(candidate_id for candidate_id, row in latest.items()
                  if candidate_id not in listed and _harness_failed(row)
                  and _harness_retry_allowed(ledger, state["site_id"], candidate_id))


def _unproven_retry_in_flight(ledger: Mapping[str, Any], state: Mapping[str, Any], candidate_id: str) -> bool:
    """A campaign retrying a harness failure runs one candidate at a time until one completes.

    The fix that prompted the retry is unproven until a run gets through. Another failure
    would hit every candidate already launched, each spending a bounded retry, so the next
    one waits (still pending, nothing reserved) for the active one's result.
    """
    rows = state["candidates"]
    if any(row["status"] == "completed" and not _harness_failed(row) for row in rows.values()):
        return False
    if not any(_harness_failures(ledger, state["site_id"], key) for key in rows):
        return False
    return any(row["status"] in ACTIVE_STATUSES for key, row in rows.items() if key != candidate_id)


def _result_not_counted(row: Mapping[str, Any], candidate: Mapping[str, Any], plan: Mapping[str, Any]) -> bool:
    """A result made under a harness revision a later one corrected for this candidate's backend.

    A scored result (completed, or a closed run that scored episodes) is normally final: the
    candidate is never planned again. When our harness made it measure something other than
    the policy (``RESULTS_NOT_COUNTED_BELOW``), it is not counted, and the candidate is planned
    again once at the next deploy, inside the site's existing allowance (owner approval
    2026-11-05). The new plan carries the current revision, so its result counts and the
    candidate is never planned again for the same defect.
    """
    from .beta_policy_screening_plan import RESULTS_NOT_COUNTED_BELOW, plan_harness_revision

    backend = (candidate.get("execution_spec") or {}).get("backend")
    if plan_harness_revision(plan) >= RESULTS_NOT_COUNTED_BELOW.get(backend, 0):
        return False
    return (row.get("status") in {"completed", "blocked"} and not _harness_failed(row)
            and bool((row.get("summary") or {}).get("episodes_scored")))


def _dropped_results_not_counted(ledger: Mapping[str, Any], state: Mapping[str, Any], plan: Mapping[str, Any],
                                 plans: Callable[[str, str], Mapping[str, Any]]) -> list[str]:
    """Candidates an earlier plan in this campaign's chain ran to a result that is not counted.

    A finished candidate leaves the chain, so its result sits in the campaign that ran it. It
    counts here only if the last row the chain holds for it is that result.
    """
    listed = {candidate["candidate_id"] for candidate in plan["candidates"]}
    latest: dict[str, tuple[Mapping[str, Any], str]] = {}
    seen: set[str] = set()
    previous = state.get("supersedes")
    while isinstance(previous, str) and previous not in seen and previous in ledger["campaigns"]:
        seen.add(previous)
        earlier = ledger["campaigns"][previous]
        for candidate_id, row in earlier["candidates"].items():
            latest.setdefault(candidate_id, (row, previous))
        previous = earlier.get("supersedes")
    dropped = []
    for candidate_id, (row, campaign_id) in sorted(latest.items()):
        if candidate_id in listed or not (row.get("summary") or {}).get("episodes_scored"):
            continue
        earlier_plan = plans(campaign_id, ledger["campaigns"][campaign_id]["plan_digest"])
        candidate = next((item for item in earlier_plan["candidates"] if item["candidate_id"] == candidate_id), None)
        if candidate is not None and _result_not_counted(row, candidate, earlier_plan):
            dropped.append(candidate_id)
    return dropped


def _replan_scope(ledger: Mapping[str, Any], plan: Mapping[str, Any], implementation_commit: str,
                  scene_half: str | None = None,
                  plans: Callable[[str, str], Mapping[str, Any]] | None = None) -> list[str] | None:
    state = ledger["campaigns"][plan["campaign_id"]]
    rows = state["candidates"]
    if state.get("superseded_by") or any(row["status"] in ACTIVE_STATUSES for row in rows.values()):
        return None
    if _amount(_budget(ledger, state["site_id"], campaign_id=plan["campaign_id"])["reserved_usd"]) > 0:
        return None
    stale = machine = False
    for candidate in plan["candidates"]:
        row = rows[candidate["candidate_id"]]
        frozen = (candidate.get("execution_spec") or {}).get("implementation_commit")
        if (row["status"] in QUEUED_STATUSES and frozen not in (None, implementation_commit)
                or row["status"] == "blocked" and COMMIT_CHANGED_BLOCKER in row.get("blockers", [])
                # Refused before any allocation at an older commit: a deploy may have fixed the
                # refusal, so the row is planned again once at the new commit, never spent twice.
                or row["status"] == "blocked" and frozen not in (None, implementation_commit)
                and not _spent(row)
                # Our harness failed after allocation at an older commit: the deploy may have fixed
                # it, so the candidate is planned again once per site (bounded, it spent money).
                or frozen not in (None, implementation_commit) and _harness_failed(row)
                and _harness_retry_allowed(ledger, state["site_id"], candidate["candidate_id"])
                # Closed for the site's budget at an older commit, never spent, while the site
                # still has money: the new plan may fit its quote to what is left. The site
                # limit still decides every reservation.
                or row["status"] == "deferred" and row.get("blockers") == [BUDGET_EXHAUSTED_BLOCKER]
                and frozen not in (None, implementation_commit) and not _spent(row)
                and _amount(_budget(ledger, state["site_id"])["committed_upper_bound_usd"]) < SITE_LIMIT
                # Its result was made under a harness revision a later one corrected: not counted,
                # so the candidate runs once more on the deployed harness.
                or frozen not in (None, implementation_commit) and _result_not_counted(row, candidate, plan)):
            stale = True
        # Its machine failed while starting the simulator: planned again at any commit, bounded.
        machine = machine or (_machine_faulted(row)
                              and _machine_fault_retry_allowed(ledger, state["site_id"], candidate["candidate_id"]))
    # A candidate whose retries ran out left the chain; a raised retry bound admits it again
    # at the next deploy, once (the plan that re-admits it freezes the new commit).
    frozen_commits = {(candidate.get("execution_spec") or {}).get("implementation_commit")
                      for candidate in plan["candidates"]} - {None}
    moved = bool(frozen_commits) and implementation_commit not in frozen_commits
    dropped = _dropped_harness_retries(ledger, state, plan) if moved else []
    if moved and plans is not None:
        dropped = sorted({*dropped, *_dropped_results_not_counted(ledger, state, plan, plans)})
    stale = stale or bool(dropped)
    if not stale and scene_half is not None:
        from .beta_policy_screening_scene_half import planned_half

        # The scene's team half was compiled again since this plan froze it: a half the plan
        # cannot read is not known to differ, so it never re-plans on its own.
        stale = planned_half(plan) not in (None, scene_half)
    if not stale and not machine:
        return None
    # A machine fault alone changed nothing of ours, so a harness failure beside it waits for its fix:
    # it is planned again only when the commit (or the scene half) changed.
    return [candidate["candidate_id"] for candidate in plan["candidates"]
            if not _spent(rows[candidate["candidate_id"]])
            or stale and _harness_failed(rows[candidate["candidate_id"]])
            and _harness_retry_allowed(ledger, state["site_id"], candidate["candidate_id"])
            or stale and _result_not_counted(rows[candidate["candidate_id"]], candidate, plan)
            or _machine_faulted(rows[candidate["candidate_id"]])
            and _machine_fault_retry_allowed(ledger, state["site_id"], candidate["candidate_id"])] + dropped or None


def replan_scope(campaign_id: str, *, implementation_commit: str, scene_half: str | None = None,
                 queue_root: str | Path | None = None) -> list[str] | None:
    """Candidates to plan again because the deployed code or the scene's half changed, or ``None``.

    A plan freezes the commit its candidates were prepared against, and
    preparation refuses another. When a deploy lands while some candidate is
    still unprepared (queued, or refused for the changed commit) and the
    campaign holds no reservation, every candidate that never ran or allocated
    is planned again at the new commit. Spent candidates keep their result.

    A plan also freezes the scene half it was built from
    (``beta_policy_screening_scene_half``). ``scene_half`` is the completed
    half the source's profile names now; when it differs from the frozen one
    and the campaign holds no reservation and no active row, every candidate
    that never ran or allocated is planned again, finished campaign or not.
    """
    if not re.fullmatch(r"[0-9a-f]{40}", implementation_commit):
        raise ScreeningCoordinatorError("beta_screening_implementation_commit_invalid")
    root = _root(queue_root)
    with _lock(root):
        ledger = _ledger(root)
        state = ledger["campaigns"].get(campaign_id)
        if state is None:
            return None
        return _replan_scope(ledger, _plan(root, campaign_id, state["plan_digest"]), implementation_commit,
                             scene_half, plans=lambda earlier, digest: _plan(root, earlier, digest))


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
    can never spend what the old one committed. A re-plan is admitted for a
    changed commit, or for a scene half the new plan froze that differs from
    the old plan's (read from both plans' own sealed scene inputs).
    """
    from .beta_policy_screening_plan import validate_screening_plan
    from .beta_policy_screening_scene_half import planned_half

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
        scope = _replan_scope(ledger, previous, implementation_commit, planned_half(value),
                              plans=lambda earlier, digest: _plan(root, earlier, digest))
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
        # such state yet, so the site sees it unavailable with its reason. Only a
        # measured geometry refusal carries the measured-geometry marker; a GPU
        # floor the screening lane cannot rent is not a fact about the site.
        measured = bool(blockers) and all(is_measured_refusal(code) for code in blockers)
        status, blockers = "unavailable", [INFEASIBLE_BLOCKER, *blockers] if measured else list(blockers)
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


def _work_root() -> Path | None:
    raw = str(os.environ.get(WORK_ROOT_ENV) or "").strip()
    if not raw:
        return None
    root = Path(raw)
    if not root.is_absolute() or ".." in root.parts:
        raise ScreeningCoordinatorError("beta_screening_work_root_unsafe")
    return root


def _new_attempt(plan: Mapping[str, Any], candidate: Mapping[str, Any], *, generation: int = 0) -> dict[str, Any]:
    identity = f"{plan['campaign_id']}:{candidate['candidate_id']}:{plan['plan_digest']}"
    if generation:
        identity += f":capacity-generation-{generation}"
    attempt_id = "screen-" + hashlib.sha256(identity.encode()).hexdigest()[:32]
    work = _work_root()
    return {"attempt_id": attempt_id,
            "generation": generation, "maximum_cost_usd": money(_amount(candidate["max_cost_usd"])),
            "reservation_status": "reserved", "provider_allocation_observed": False,
            **({ATTEMPT_DIR_FIELD: str(work / "attempts" / attempt_id)} if work is not None else {})}


def _attempt_path(root: Path, attempt: Mapping[str, Any]) -> Path:
    """The attempt's directory: the one the ledger recorded when it was reserved, else beside the ledger."""
    recorded = attempt.get(ATTEMPT_DIR_FIELD)
    if recorded is None:
        return root / "attempts" / attempt["attempt_id"]
    path = Path(recorded) if isinstance(recorded, str) else Path()
    if (not path.is_absolute() or ".." in path.parts or path.name != attempt["attempt_id"]
            or path.parent.name != "attempts"):
        raise ScreeningCoordinatorError("beta_screening_attempt_path_unsafe")
    return path


def _attempt_pointer(root: Path, attempt_id: str, path: Path) -> None:
    """Name a work-volume attempt beside the ledger, so a lost ledger is still detected."""
    pointer = root / "attempts" / f"{attempt_id}.attempt_dir.json"
    payload = json.dumps({"schema_version": ATTEMPT_POINTER_SCHEMA, "attempt_id": attempt_id,
                          ATTEMPT_DIR_FIELD: str(path)}, sort_keys=True) + "\n"
    try:
        descriptor = os.open(pointer, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    except FileExistsError:
        return
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _retry_after_capacity(earlier_retries: int = 0) -> str:
    delay = CAPACITY_RETRY_SECONDS * 2 ** earlier_retries
    return (datetime.now(timezone.utc) + timedelta(seconds=delay)).isoformat()


def _offer_shortage(observation: Mapping[str, Any]) -> bool:
    """The allocator's offer search found nothing to rent (its typed codes), before any create, or
    the offer it chose left the market before create and nothing was created."""
    from .vast_create_failure_diagnosis import OFFER_GONE_BEFORE_CREATE

    return any(isinstance(code, str) and (code.startswith("no_vast_offer_")
                                          or code in ("vast_offer_selection_blocked", OFFER_GONE_BEFORE_CREATE))
               for code in observation.get("blockers") or [])


def _spend_lock_stale(observation: Mapping[str, Any]) -> bool:
    """Refused before any create only because the spend admission lock was stale."""
    blockers = set(observation.get("blockers") or [])
    return bool(blockers & SPEND_LOCK_STALE_CODES) and blockers <= _SPEND_LOCK_REFUSAL


def _spend_lock_retries(row: Mapping[str, Any]) -> int:
    value = row.get("spend_lock_retries", 0)
    # An unreadable count is treated as spent, so the cap always holds.
    return value if type(value) is int and 0 <= value <= SPEND_LOCK_RETRY_CAP else SPEND_LOCK_RETRY_CAP


def _offer_shortage_retries(row: Mapping[str, Any]) -> int:
    value = row.get("offer_shortage_retries", 0)
    # An unreadable count is treated as spent, so the cap always holds.
    return value if type(value) is int and 0 <= value <= OFFER_SHORTAGE_RETRY_CAP else OFFER_SHORTAGE_RETRY_CAP


def _retry_after_offer_shortage(earlier_retries: int) -> str:
    delay = min(OFFER_SHORTAGE_MAX_RETRY_SECONDS, OFFER_SHORTAGE_FIRST_RETRY_SECONDS * 2 ** earlier_retries)
    return (datetime.now(timezone.utc) + timedelta(seconds=delay)).isoformat()


def _transport_retries(row: Mapping[str, Any]) -> int:
    value = row.get("transport_retries", 0)
    # An unreadable count is treated as spent, so the cap always holds.
    return value if type(value) is int and 0 <= value <= TRANSPORT_RETRY_CAP else TRANSPORT_RETRY_CAP


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
    path = _attempt_path(root, attempt) / "allocator_result.json"
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
        if _unproven_retry_in_flight(ledger, state, candidate_id):
            return None
        if any(_harness_failed(other) for key, other in state["candidates"].items() if key != candidate_id):
            # Never pay for a run this commit's harness already failed on a sibling.
            row.update(status="blocked", blockers=[HARNESS_HOLD_BLOCKER])
            _changed(state)
            _save(root, ledger)
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


def _switch_off_website_scripted(root: Path) -> int:
    """Close a website plan's queued scripted baselines before they reserve money.

    Website jobs never fund a scripted baseline (``beta_policy_screening_plan.website_switched_off``,
    owner decision 2026-11-04). Plans frozen before that decision still list some as selected or
    waitlisted: a row that never reserved money closes ``unavailable`` with the plan's own
    ``candidate_disabled``, the same reason a newer plan lists. One already preparing, launched
    or finished is left to its end. Returns the number of rows closed.
    """
    from .beta_policy_screening_plan import DISABLED_BLOCKER, JOB_CEILING_BASIS

    closed = 0
    with _lock(root):
        ledger = _ledger(root)
        for campaign_id, state in ledger["campaigns"].items():
            if not any(row["status"] in QUEUED_STATUSES for row in state["candidates"].values()):
                continue
            plan = _plan(root, campaign_id, state["plan_digest"])
            if plan.get("budget_basis") != JOB_CEILING_BASIS:
                continue
            touched = False
            for candidate in plan["candidates"]:
                row = state["candidates"][candidate["candidate_id"]]
                if (row["status"] in QUEUED_STATUSES and "attempt" not in row
                        and candidate.get("execution_evidence_kind") == SCRIPTED):
                    row.update(status="unavailable", blockers=[DISABLED_BLOCKER])
                    touched = True
                    closed += 1
            if touched:
                _changed(state)
        if closed:
            _save(root, ledger)
    return closed


def _unstarted_refusal_proof(root: Path, attempt: Mapping[str, Any], observation: Mapping[str, Any]) -> bool:
    """Reopen bytes from our exact job fence; caller booleans are not evidence."""
    from .allocator_invocation import manifest
    record = observation.get("unstarted_refusal_receipt") or {}
    directory = _attempt_path(root, attempt) / "invocation"
    path = directory / "no_create_fence.json"
    if (type(observation.get("retryable_unstarted_refusal")) is not bool
            or observation.get("no_policy_api_calls_proven") is not True
            or record.get("path") != str(path) or path.is_symlink()
            or (directory / "provider_create_intent.json").exists()):
        return False
    try:
        raw = path.read_bytes()
        proof, value = json.loads(raw), manifest(directory / "manifest.json")
        return bool(
            record.get("size_bytes") == len(raw)
            and record.get("sha256") == "sha256:" + hashlib.sha256(raw).hexdigest()
            and proof.get("schema_version") == "blueprint.allocator_no_create_fence.v1"
            and proof.get("manifest_digest") == value["manifest_digest"]
            and proof.get("invocation_id") == value["invocation_id"]
            and proof.get("binding") == value["binding"]
            and value["binding"].get("prepared_digest") == attempt.get("prepared", {}).get("prepared_digest")
            and proof.get("provider_create_prevented") is True
            and proof.get("no_policy_api_calls_proven") is True
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False


def _launch_unclaimed(root: Path, attempt: Mapping[str, Any]) -> bool:
    """No launch slot was ever claimed for this attempt, so no allocator can exist for it."""
    directory = _attempt_path(root, attempt)
    return not any(path.exists() or path.is_symlink() for path in (
        directory / "allocator_launch_intent.json", directory / "invocation" / "manifest.json",
        directory / "allocator_process.json", directory / "allocator_result.json"))


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
            row.pop("next_action", None)
            row.update(status="running", blockers=[])
            return
        no_allocation = (observation.get("no_allocation_proven") is True
                         and observation.get("provider_allocation_performed") is False
                         and attempt["provider_allocation_observed"] is False)
        closed = observation.get("provider_zero_verified") is True and attempt["provider_allocation_observed"] is True
        if (status == "deferred" and no_allocation and observation.get("retryable_control_refusal") is True
                and _launch_unclaimed(root, attempt)):
            # The control read before spawn got no answer (429, 5xx, timeout).
            # Nothing could start: keep this attempt and its reservation, read
            # again, up to the cap; past it, release and block this candidate.
            retries = _transport_retries(row)
            if retries >= TRANSPORT_RETRY_CAP:
                attempt["reservation_status"] = "released"
                row.update(status="blocked", blockers=[TRANSPORT_RETRY_CAP_BLOCKER])
                return
            row["transport_retries"] = retries + 1
            attempt["retry_not_before"] = _retry_after_capacity(retries)
            row.update(status="preparing", blockers=[CAPACITY_WAIT_BLOCKER, CONTROL_WAIT_BLOCKER])
            return
        if status in {"completed", "blocked"} and no_allocation:
            unstarted_proof = _unstarted_refusal_proof(root, attempt, observation)
            if "unstarted_refusal_receipt" in observation and not unstarted_proof:
                row.update(status="reconciliation_required", blockers=["beta_screening_unspent_fence_unproven"])
                return
            if unstarted_proof and observation.get("retryable_unstarted_refusal") is True:
                # A durable generation fence prevents every delayed old start;
                # transfer, never replenish, the site's existing reservation.
                attempt.update(reservation_status="released", closed_at=_now(),
                    terminal_reason="allocator_fenced_before_create",
                    unstarted_refusal_receipt=dict(observation["unstarted_refusal_receipt"]),
                    no_policy_api_calls_proven=True)
                # Only a fresh attempt after an unanswered read counts toward
                # the cap; a crash or deploy before create keeps its recovery.
                transport = observation.get("retry_cause") == TRANSPORT_RETRY_CAUSE
                retries = _transport_retries(row) if transport else 0
                if transport and retries >= TRANSPORT_RETRY_CAP:
                    row.update(status="blocked", blockers=[TRANSPORT_RETRY_CAP_BLOCKER])
                    return
                if transport:
                    row["transport_retries"] = retries + 1
                row.setdefault("attempt_history", []).append(json.loads(json.dumps(attempt)))
                row["attempt"] = {**_new_attempt(plan, candidate, generation=attempt.get("generation", 0) + 1),
                                  "retry_not_before": _retry_after_capacity(retries)}
                # The WebApp lets a published running candidate return to
                # preparing only with the capacity-wait code; it names the wait.
                row.update(status="preparing",
                           blockers=[CAPACITY_WAIT_BLOCKER, "beta_screening_unspent_invocation_recovery"])
                return
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
            if _offer_shortage(observation):
                # Proven zero allocation: the market had no qualifying offer. Wait and search again
                # under a fresh identity holding the same reservation, a bounded number of times.
                retries = _offer_shortage_retries(row)
                if retries < OFFER_SHORTAGE_RETRY_CAP:
                    attempt.update(reservation_status="released", closed_at=_now(),
                                   terminal_reason="no_qualifying_gpu_offer", no_policy_api_calls_proven=True)
                    row["offer_shortage_retries"] = retries + 1
                    row.setdefault("attempt_history", []).append(json.loads(json.dumps(attempt)))
                    row["attempt"] = {**_new_attempt(plan, candidate, generation=attempt.get("generation", 0) + 1),
                                      "retry_not_before": _retry_after_offer_shortage(retries)}
                    # The WebApp lets a published candidate return to preparing only with the
                    # capacity-wait code; the second code names which capacity.
                    row.update(status="preparing", blockers=[CAPACITY_WAIT_BLOCKER, OFFER_WAIT_BLOCKER])
                    return
                attempt["reservation_status"] = "released"
                row.update(status="blocked", blockers=[OFFER_SHORTAGE_EXHAUSTED_BLOCKER])
                return
            if _spend_lock_stale(observation) and _spend_lock_retries(row) < SPEND_LOCK_RETRY_CAP:
                # Proven zero allocation: the money check could not read fresh billing evidence.
                # Search again later under a fresh identity holding the same reservation; the next
                # preflight reads the lock again. Past the cap it keeps the refusal below.
                retries = _spend_lock_retries(row)
                attempt.update(reservation_status="released", closed_at=_now(),
                               terminal_reason="spend_admission_lock_stale", no_policy_api_calls_proven=True)
                row["spend_lock_retries"] = retries + 1
                row.setdefault("attempt_history", []).append(json.loads(json.dumps(attempt)))
                row["attempt"] = {**_new_attempt(plan, candidate, generation=attempt.get("generation", 0) + 1),
                                  "retry_not_before": _retry_after_offer_shortage(retries)}
                # The website words this pair "Waiting to confirm this job can still run."
                row.update(status="preparing", blockers=[CAPACITY_WAIT_BLOCKER, CONTROL_WAIT_BLOCKER])
                return
            attempt["reservation_status"] = "released"
            row.update(status="blocked", blockers=_safe_blockers(observation.get("blockers"),
                                                                "beta_screening_not_allocated"))
            return
        if status in {"completed", "blocked"} and closed:
            # Charged at the full reserved bound, then settled in this same
            # write when the run's own records prove its measured cost: no
            # publication shows it at the whole bound first. Doubt keeps it.
            attempt["reservation_status"] = "committed"
            _settle(root, attempt, candidate)
            if isinstance(observation.get("result"), Mapping):
                try:
                    summary = candidate_result_summary(observation["result"], plan=plan, candidate=candidate,
                                                       review_videos=observation.get("review_videos"))
                except (ValueError, TypeError, KeyError):
                    row.update(status="blocked", blockers=["beta_screening_native_result_invalid"])
                else:
                    row.update(status="completed", blockers=[], summary=summary)
            else:
                row.update(status="blocked", blockers=_safe_blockers(observation.get("blockers"),
                                                                    "beta_screening_native_result_missing"))
            if row["status"] == "blocked":
                # A run without a result keeps its videos for the site too, beside its capture;
                # only a completed run's summary references and publishes them.
                from .beta_screening_review_videos import valid_reference

                if (reference := valid_reference(observation.get("review_videos"))) is not None:
                    row["review_videos"] = reference
            return
        if observation.get("next_action") == "reconcile_exact_provider_identity":
            row["next_action"] = "reconcile_exact_provider_identity"
            checks = min(6, int(attempt.get("provider_reconciliation_checks", 0)) + 1)
            attempt["provider_reconciliation_checks"] = checks
            attempt["reconcile_not_before"] = (datetime.now(timezone.utc) + timedelta(
                seconds=min(600, 60 * 2 ** (checks - 1)))).isoformat()
        row.update(status="reconciliation_required", blockers=_safe_blockers(observation.get("blockers"),
                                                                            "beta_screening_allocation_uncertain"))
    _update(root, plan["campaign_id"], candidate["candidate_id"], apply)


SETTLEMENT_HELD_FIELD = "settlement_held"


def _settle(root: Path, attempt: dict[str, Any], candidate: Mapping[str, Any]) -> bool:
    """Record what a committed attempt's own records prove it cost; never raises.

    Each attempt is decided once: settled (``settled_cost_usd``), or held at
    its maximum for good with the reason (``settlement_held``) when its records
    do not prove one finished, torn-down run bound to it. An attempt directory
    that cannot be read at all (an unmounted volume) is tried again on a later
    pass. Returns whether the attempt changed.
    """
    from .beta_policy_screening_settlement import SettlementEvidenceError, attempt_settlement

    if (attempt.get("reservation_status") != "committed" or attempt.get("provider_allocation_observed") is not True
            or "settled_cost_usd" in attempt or SETTLEMENT_HELD_FIELD in attempt):
        return False
    try:
        attempt.update(attempt_settlement(attempt_dir=_attempt_path(root, attempt), attempt=attempt,
                                          candidate=candidate, settled_at=_now()))
    except SettlementEvidenceError as exc:
        attempt[SETTLEMENT_HELD_FIELD] = {"reason": _safe_blockers([str(exc)], "beta_screening_settlement_unproven")[0],
                                          "held_at": _now()}
    except Exception:  # noqa: BLE001 - unreadable now; it counts at its maximum and is tried again
        return False
    return True


def _settle_finished(root: Path) -> int:
    """Settle every committed attempt not decided yet, including ones committed before settlement existed.

    A settlement changes no row a publication shows. It can lower only what
    the site's funding counts; ``_save`` gives a campaign a new revision only
    if its published budget changes. Returns the number of attempts decided.
    """
    decided = 0
    with _lock(root):
        ledger = _ledger(root)
        for campaign_id, state in ledger["campaigns"].items():
            open_attempts = [(candidate_id, attempt) for candidate_id, row in state["candidates"].items()
                             if row["status"] in {"completed", "blocked"}
                             for attempt in [*row.get("attempt_history", []), *([row["attempt"]] if row.get("attempt") else [])]
                             if attempt["reservation_status"] == "committed"
                             and "settled_cost_usd" not in attempt and SETTLEMENT_HELD_FIELD not in attempt]
            if not open_attempts:
                continue
            plan = _plan(root, campaign_id, state["plan_digest"])
            candidates = {candidate["candidate_id"]: candidate for candidate in plan["candidates"]}
            for candidate_id, attempt in open_attempts:
                decided += _settle(root, attempt, candidates[candidate_id])
        if decided:
            _save(root, ledger)
    return decided


def _attempt_root(root: Path, attempt: Mapping[str, Any]) -> Path:
    path = _attempt_path(root, attempt)
    if path.parent != root / "attempts":
        # On the work volume: its root must already exist (the deploy provisions it). A missing
        # volume stops here, before any staging, and never re-creates a committed attempt.
        work = path.parent.parent
        if not work.is_dir() or any(item.is_symlink() for item in (path, path.parent, work, *work.parents)):
            raise ScreeningCoordinatorError("beta_screening_work_root_unavailable")
        path.parent.mkdir(exist_ok=True, mode=0o700)
        _attempt_pointer(root, attempt["attempt_id"], path)
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
            publication = screening_publication(plan=plan, state=state, budget=_published_budget(ledger, state))
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
        if attempt.get("reconcile_not_before") and datetime.fromisoformat(
                attempt["reconcile_not_before"].replace("Z", "+00:00")) > datetime.now(timezone.utc):
            return False
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
        observed = executor.launch_candidate(plan=plan, candidate=candidate, attempt_dir=attempt_dir,
                                             prepared=prepared, queue_root=root)
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
            # Finished attempts settle to their measured cost first (each once),
            # so the refill below and new plans count what they really spent.
            settled = _settle_finished(root)
            # A website plan's queued scripted baselines close first, so the
            # waitlist below never funds one.
            _switch_off_website_scripted(root)
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
                    "waitlist_changes": refilled, "settlements_decided": settled}
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
            publication = screening_publication(plan=plan, state=state, budget=_published_budget(ledger, state))
            if (state["published_revision"] == publication["revision"]
                    and state["published_digest"] == publication["publication_digest"]):
                continue
            if state["published_revision"] == state["revision"]:
                # Acknowledged bytes changed without progress (a wire contract
                # upgrade). Send them as a new revision; the website refuses a
                # same-revision rewrite, which would leave the result unseen.
                _changed(state)
                invalidated = True
                publication = screening_publication(plan=plan, state=state, budget=_published_budget(ledger, state))
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
        publication = screening_publication(plan=plan, state=state, budget=_published_budget(ledger, state))
        if publication["publication_digest"] != publication_digest:
            return False
        state["published_revision"] = state["revision"]
        state["published_digest"] = publication_digest
        _save(root, ledger)
        return True


__all__ = ["enqueue_screening_campaign", "process_screening_queue", "pending_publications",
           "acknowledge_publication", "latest_campaign", "replan_scope", "supersede_campaign",
           "ScreeningCoordinatorError"]
