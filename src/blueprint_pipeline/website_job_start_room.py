"""Before a website job's first paid step: start it only if the month has room for a whole typical job.

Owner decision 2026-11-11 ("yes replace them - $300 monthly limit, $100 fuse"): the only true
money stop is the account's monthly limit (``account_monthly_spend``). A job that starts when the
month is nearly spent can be stranded half-built: it pays for a build and gets no answer. So a
website job that has not started its paid work yet starts only when

    limit - counted - sum over started, unfinished jobs of their reserve  >=  TYPICAL_JOB_USD

``counted`` is the account report's ``counted_usd``: the month's reconciled spend plus every hold
the meter has open, so holds are already in it and are not subtracted twice.

A started job's reserve is an estimate with a hard end. It is ``max(0, TYPICAL_JOB_USD - what the
job has spent since it started)``. That spend is kept in the job's own start record, one figure per
UTC month (the meter's ``metered_usd`` for the job, never lowered within a month), so a job running
across a month boundary keeps what it spent before. The meter prices screening and Atlas purchases
today; the other admissions count as calls until stage 2 prices them, so the remainder errs high,
toward waiting. The reserve ends, and the record is marked finished and never read again, at the
first of:

- the job is over with nothing paid open: no screening row queued or in flight, and either every
  capture's preparation ended for good or its screening campaigns have all ended;
- its grant has expired (the later of ``expires_at_epoch`` and ``valid_until``), whatever its rows say;
- with no grant to read, ``NO_GRANT_BOUND_SECONDS`` after its start record: the longest any website
  job can run, the website's 7-day window plus an owner's 7-day extension (WebApp
  ``server/utils/websiteJobWindow.ts:20,25``; ``sceneExecutionWindowExtension.ts:18``).

Where it runs: the host listener, for a website browser capture, right after the browser admission
(a free read of the job's context and grant) and before ``run_e2e``, where every paid step of a
website preparation starts (the Atlas preview is enqueued from there too). Nothing paid has happened
yet, so the wait costs nothing.

A job that started is never stopped here: the first pass that finds room writes a start record
beside the job's scene (``START_FILE``), and every later attempt of that job, and every other
capture of it, passes. A job the meter already shows spend or admissions for (one in flight when
this shipped) gets a start record when first seen, and the same end rules apply to it.

The wait: the attempt ends ``failed_retryable`` with ``START_WAIT`` inside the shared paid-admission
refusal (``paid_resource_admission_blocked:account_monthly_spend_job_start_wait``). The retry policy
reads it as money (``job_retry_policy.hard_stop_kind``), so the host's own tick runs the job again
from its kept bytes from 2 minutes to hourly until the job's deadline, its grant's expiry; no attempt
count ends it. It starts by itself on the first pass that finds room: a new UTC month, a higher
limit, or a started job finishing or settling. The status read reports it as ``capacity_delayed``.
The job's grant is kept before it waits (``GRANT_FILE``, as the first step of ``run_e2e`` keeps it),
so the grant's expiry is its deadline from the first wait. A wait record beside the scene
(``WAIT_FILE``) feeds one warn on the account's alert row while any job waits
(``waiting_jobs``). A job whose grant runs out while it waits ends with its own code,
``ROOM_WAIT_EXPIRED_CODE`` (``room_wait_ending``), not ``consent_expired``: the site's consent did
not run out, Blueprint held the job.

It runs only with the switch at ``fuse`` (``website_job_limits``); at ``nested`` (shipped) it
returns at once and records nothing. The check reads the same report the account limit itself reads, so with the report stale, missing
or the limit reached the job waits on that refusal by its own name. Where no limit is configured
(tests, development, replays off the host) nothing is checked or recorded.
"""
from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from .common import write_json

#: What one website job costs from video to answer, used only to decide whether a new job may start.
#: An estimate, not a cap: it is the one job that has gone all the way (capture-447, about $71 as
#: counted: $44.52 build and $26.63 testing). No job is stopped or sized by it.
TYPICAL_JOB_USD = Decimal("70")
#: The wait, as the refusal code inside the shared paid-admission refusal.
START_WAIT = "account_monthly_spend_job_start_wait"
#: A job whose grant ran out while it waited here: Blueprint held it, the site's consent stands.
ROOM_WAIT_EXPIRED_CODE = "website_job_held_for_monthly_room_expired"
#: The listener's failure stage while the job waits here.
STAGE = "website_job_start_room"
#: Beside the job's scene (``<scenes>/site-<request id>/``): one per job, whatever its captures.
START_FILE = ".website_job_start.json"
START_SCHEMA = "website_job_start.v1"
WAIT_FILE = ".website_job_start_wait.json"
WAIT_SCHEMA = "website_job_start_wait.v1"
GRANT_FILE = "pipeline/website_scene_sponsorship.json"
#: The longest a website job can run: the 7-day window plus an owner's 7-day extension (see above).
NO_GRANT_BOUND_SECONDS = 14 * 86_400
#: A wait record counts for the alert while its job was refused this recently (the retry runs at least hourly).
WAIT_FRESH_SECONDS = 2 * 3_600
_TERMINAL_LEDGER = "terminal_authority_ended"
_CONSENT_EXPIRED = "consent_expired"
#: The screening ledger's file and the row states in which a row may still spend
#: (``beta_policy_screening_coordinator.LEDGER_FILENAME``; ``beta_policy_screening_result`` QUEUED and
#: ACTIVE statuses). Named here, not imported: the listener's import graph must not reach the GPU lanes
#: (``tests/test_live_pipeline_import_isolation.py``); a test pins them to the coordinator's own.
SCREENING_LEDGER_FILENAME = "screening_ledger.v1.json"
SCREENING_OPEN_STATUSES = frozenset({"pending", "waitlisted", "preparing", "launching", "running",
                                     "reconciliation_required"})


def _now(now: float | None) -> float:
    return datetime.now(timezone.utc).timestamp() if now is None else float(now)


def _iso(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, timezone.utc).isoformat().replace("+00:00", "Z")


def _month(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, timezone.utc).strftime("%Y-%m")


def _read(path: Path) -> Any:
    if path.is_symlink() or not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeDecodeError):
        return None


def _dollars(value: Any) -> Decimal:
    if isinstance(value, bool):
        return Decimal(0)
    try:
        amount = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        return Decimal(0)
    return amount if amount.is_finite() and amount > 0 else Decimal(0)


def _epoch(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
        return parsed.timestamp() if parsed.tzinfo is not None else None
    return None


def grant_end(grant: Any) -> float | None:
    """When a grant stops admitting spend: the later of its expiry and an owner extension's ``valid_until``."""
    if not isinstance(grant, Mapping):
        return None
    ends = [end for end in (_epoch(grant.get("expires_at_epoch")), _epoch(grant.get("valid_until"))) if end is not None]
    return max(ends) if ends else None


def _meter_jobs(report: Mapping[str, Any]) -> dict[str, tuple[Decimal, int]]:
    """``{request id: (metered dollars this month, admissions this month)}`` from the account report."""
    meter = report.get("meter") if isinstance(report.get("meter"), Mapping) else {}
    jobs: dict[str, tuple[Decimal, int]] = {}
    for row in meter.get("jobs") or []:
        if isinstance(row, Mapping) and isinstance(row.get("job_id"), str):
            admissions = row.get("admissions")
            jobs[row["job_id"]] = (_dollars(row.get("metered_usd")),
                                   admissions if type(admissions) is int and admissions > 0 else 0)
    return jobs


def _screening_sites(environ: Mapping[str, str]) -> dict[str, list[str]] | None:
    """Every row status of every screening campaign, by site; None when the ledger cannot be read."""
    from .paid_spend_meter import DEFAULT_SCREENING_QUEUE_ROOT, SCREENING_QUEUE_ROOT_ENV

    root = Path(str(environ.get(SCREENING_QUEUE_ROOT_ENV) or "").strip() or DEFAULT_SCREENING_QUEUE_ROOT)
    path = root / SCREENING_LEDGER_FILENAME
    if not path.exists():
        return {}
    ledger = _read(path)
    if not isinstance(ledger, Mapping) or not isinstance(ledger.get("campaigns"), Mapping):
        return None
    sites: dict[str, list[str]] = {}
    for state in ledger["campaigns"].values():
        if isinstance(state, Mapping) and isinstance(state.get("candidates"), Mapping):
            sites.setdefault(str(state.get("site_id")), []).extend(
                str(row.get("status")) for row in state["candidates"].values() if isinstance(row, Mapping))
    return sites


def end_reason(scene_dir: Path, record: Mapping[str, Any], *, sites: Mapping[str, list[str]] | None,
               now: float) -> str | None:
    """Why a started job's reserve has ended, or None while it may still spend (see the module)."""
    ends = [end for end in [grant_end(record.get("grant")),
                            *(grant_end(_read(path)) for path in scene_dir.glob("captures/*/" + GRANT_FILE))]
            if end is not None]
    if ends and max(ends) <= now:
        return "grant_expired"
    started = _epoch(record.get("started_at_epoch"))
    if not ends and (started is None or now - started >= NO_GRANT_BOUND_SECONDS):
        return "no_grant_bound_reached"
    statuses = None if sites is None else sites.get(scene_dir.name)
    if sites is None or (statuses and any(status in SCREENING_OPEN_STATUSES for status in statuses)):
        return None
    if statuses:
        return "screening_ended"
    ledgers = [_read(path) for path in scene_dir.glob("captures/*/pipeline_job_ledger.json")]
    if ledgers and all(isinstance(row, Mapping) and row.get("status") == _TERMINAL_LEDGER for row in ledgers):
        return "preparation_ended"
    return None


def _record(scene_dir: Path, *, now: float, basis: str, grant: Mapping[str, Any] | None = None,
            **extra: Any) -> dict[str, Any]:
    return {"schema_version": START_SCHEMA, "scene_id": scene_dir.name, "started_at": _iso(now),
            "started_at_epoch": now, "basis": basis, "spend_by_month": {},
            **({"grant": {key: grant[key] for key in ("expires_at_epoch", "valid_until") if key in grant}}
               if isinstance(grant, Mapping) else {}), **extra}


def _spent_since_start(record: dict[str, Any], *, meter_spent: Decimal, now: float) -> Decimal:
    """Fold this month's meter figure into the record (never lowered within a month); the sum of every month."""
    months = record.setdefault("spend_by_month", {})
    if not isinstance(months, dict):
        months = record["spend_by_month"] = {}
    period = _month(now)
    months[period] = str(max(_dollars(months.get(period)), meter_spent))
    return sum((_dollars(value) for value in months.values()), Decimal(0))


def started_job_reserve(scenes_dir: Path, *, exclude: str, meter: Mapping[str, tuple[Decimal, int]],
                        environ: Mapping[str, str], now: float) -> tuple[Decimal, list[dict[str, Any]]]:
    """The sum of the reserves of started, unfinished jobs, and one row per job still reserving.

    Updates each started job's record: its spend this month, or its end (then it is never read again).
    A job the meter shows activity for with no record yet gets one now.
    """
    for job_id, (spent, admissions) in meter.items():
        scene = scenes_dir / ("site-" + job_id)
        if (spent > 0 or admissions > 0) and scene.is_dir() and not (scene / START_FILE).exists():
            write_json(scene / START_FILE, _record(scene, now=now, basis="metered_before_check"))
    sites = _screening_sites(environ)
    total, rows = Decimal(0), []
    for path in sorted(scenes_dir.glob("site-*/" + START_FILE)):
        scene = path.parent
        record = _read(path)
        if scene.name == exclude or not isinstance(record, dict) or record.get("finished_at"):
            continue
        reason = end_reason(scene, record, sites=sites, now=now)
        if reason is not None:
            write_json(path, {**record, "finished_at": _iso(now), "finished_reason": reason})
            continue
        spent = _spent_since_start(record, meter_spent=meter.get(scene.name[len("site-"):], (Decimal(0), 0))[0],
                                   now=now)
        write_json(path, record)
        remainder = max(Decimal(0), TYPICAL_JOB_USD - spent)
        total += remainder
        rows.append({"scene_id": scene.name, "spent_since_start_usd": str(spent), "reserved_usd": str(remainder)})
    return total, rows


def _retain_grant(capture_root: Path, authority: Mapping[str, Any] | None) -> None:
    """Keep the job's grant, as the first step of ``run_e2e`` does, so its expiry is the wait's deadline."""
    path = Path(capture_root) / GRANT_FILE
    if authority is not None and not path.exists() and not path.is_symlink():
        write_json(path, dict(authority))


def _note_wait(scene_dir: Path, now: float) -> None:
    previous = _read(scene_dir / WAIT_FILE)
    since = _epoch(previous.get("since_epoch")) if isinstance(previous, Mapping) else None
    write_json(scene_dir / WAIT_FILE, {"schema_version": WAIT_SCHEMA, "since_epoch": since or now,
                                       "last_refused_epoch": now})


def require_job_start_room(*, capture_root: Path, authority: Mapping[str, Any] | None = None,
                           now: float | None = None, environ: Mapping[str, str] | None = None) -> dict[str, Any] | None:
    """Return the job's start record (or None where no limit is set); raise to wait when the month has no room.

    Raises ``PaidResourceAdmissionBlocked`` with ``START_WAIT``, or with the account limit's own
    refusal when its report cannot be used. Never stops a job that has started.
    """
    from . import account_monthly_spend as account
    from .paid_resource_admission import PaidResourceAdmissionBlocked

    from .website_job_limits import fuse

    if not fuse():
        return None  # nested (shipped): no start check, as before 2026-11-11.
    env = os.environ if environ is None else environ
    capture_root = Path(capture_root)
    scene_dir = capture_root.parent.parent
    if capture_root.parent.name != "captures" or not scene_dir.name.startswith("site-"):
        return None
    record_path = scene_dir / START_FILE
    existing = _read(record_path)
    if isinstance(existing, Mapping) and existing.get("schema_version") == START_SCHEMA:
        return dict(existing)
    policy, error, _config = account.configured_policy(env)
    if policy is None and error is None and not account.limit_required(env):
        return None
    moment = _now(now)
    blockers = account.admission_blockers(now=moment, environ=env)
    if blockers:
        _retain_grant(capture_root, authority)
        _note_wait(scene_dir, moment)
        raise PaidResourceAdmissionBlocked(sorted(set(blockers)))
    report = account.read_report(Path(policy["report_path"]))
    limit = account.effective_limit(policy, account.month_of(moment))
    counted = _dollars(report.get("counted_usd"))
    meter = _meter_jobs(report)
    spent, admissions = meter.get(scene_dir.name[len("site-"):], (Decimal(0), 0))
    checked = {"period": account.month_of(moment), "limit_usd": str(limit), "counted_usd": str(counted),
               "typical_job_usd": str(TYPICAL_JOB_USD)}
    if spent > 0 or admissions > 0:
        # Already spending (in flight when this check shipped): started, never stopped here.
        record = _record(scene_dir, now=moment, basis="metered_before_check", grant=authority, **checked)
    else:
        reserve, rows = started_job_reserve(scene_dir.parent, exclude=scene_dir.name, meter=meter, environ=env,
                                            now=moment)
        room = limit - counted - reserve
        record = _record(scene_dir, now=moment, basis="room_checked", grant=authority, **checked,
                         started_jobs_reserve_usd=str(reserve), started_jobs=rows, room_usd=str(room))
        if room < TYPICAL_JOB_USD:
            _retain_grant(capture_root, authority)
            _note_wait(scene_dir, moment)
            raise PaidResourceAdmissionBlocked([START_WAIT])
    write_json(record_path, record)
    (scene_dir / WAIT_FILE).unlink(missing_ok=True)
    return record


def waiting_jobs(storage_root: Path, *, now: float, limit: int = 10_000) -> dict[str, Any]:
    """How many jobs wait for this month's room now, and since when the oldest has waited."""
    count, oldest = 0, None
    for index, path in enumerate(Path(storage_root).glob("*/scenes/site-*/" + WAIT_FILE)):
        if index >= limit:
            break
        wait = _read(path)
        since = _epoch(wait.get("since_epoch")) if isinstance(wait, Mapping) else None
        last = _epoch(wait.get("last_refused_epoch")) if isinstance(wait, Mapping) else None
        if since is None or last is None or now - last > WAIT_FRESH_SECONDS or (path.parent / START_FILE).exists():
            continue
        count += 1
        oldest = since if oldest is None else min(oldest, since)
    return {"count": count, "oldest_since_epoch": oldest}


def room_wait_ending(code: str, history: Sequence[Mapping[str, Any]]) -> str:
    """The code a job's ending carries: its own when its grant ran out while it waited for room."""
    last = history[-1] if history and isinstance(history[-1], Mapping) else {}
    return ROOM_WAIT_EXPIRED_CODE if code == _CONSENT_EXPIRED and last.get("stage") == STAGE else code


__all__ = ["GRANT_FILE", "NO_GRANT_BOUND_SECONDS", "ROOM_WAIT_EXPIRED_CODE", "STAGE", "START_FILE", "START_SCHEMA",
           "START_WAIT", "TYPICAL_JOB_USD", "WAIT_FILE", "end_reason", "grant_end", "require_job_start_room",
           "room_wait_ending", "started_job_reserve", "waiting_jobs"]
