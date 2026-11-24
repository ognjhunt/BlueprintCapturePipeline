"""Meter every paid admission and its settlement against its job (metered spend, stage 1).

Owner decision 2026-11-07, "Meter + monthly cap": a job gets no budget or
confirmation of its own. Every paid call is metered against the job that made
it and shown, and one account-wide monthly limit
(``account_monthly_spend``) stops new spend. This module is the meter. It
reads and reports; it never admits, refuses, reserves or releases anything.

A meter row is one paid admission and, once known, its settlement. Rows come
from four places. Two of them already record admissions and settlements per
job, so they are read where they are, not copied:

``paid_resource_admission`` (journal)
    Every grant the shared chokepoint issues is appended here
    (``meter_chokepoint_admission``), keyed by the job the admission names
    (``admission["meter"]["job_id"]``, else the admission's own ``job_id``,
    ``website_request_id`` or ``request_id``, else ``BLUEPRINT_PAID_SPEND_JOB_ID``
    in the process environment, else ``unattributed``). An admission carries a
    price only when it names one (``admission["meter"]["held_usd"]``).
    A grant checked against the account's month at a maximum cost is also
    journaled with it (``maximum_cost_usd``). Until its settlement is journaled
    or a bill of its provider can show it (``BILL_COVERS_AFTER_SECONDS`` after
    it was admitted), that figure is a hold (``UNPRICED_HOLD_BASIS``, stress
    round 2, 2026-11-24): before, a grant counted only until the next report
    (written every minute), so at $380 of $400 one $12 call a minute was judged
    at $380 each time. The admissions judged before the next report count it
    too (``admitted_since``): the report names the bytes of each journal file
    it read, so a grant counts once, in the report or after it, whatever either
    clock says. ``record_settlement`` is the matching settlement record.
``beta_policy_screening`` (screening ledger)
    Each attempt in ``screening_ledger.v1.json``: held at its cap while
    reserved or committed without a verified settlement, then its settled
    cost (``beta_policy_screening_settlement.settled_cost``).
``atlas`` (World Labs attempt records)
    Each Atlas attempt that ``provider_spend_alerts.observe_world_labs_spend``
    counts: a completed operation at its reviewed price, one whose outcome is
    unknown held at that price (``atlas_spend_settlement``). Its settlement is
    also journaled (``atlas_spend_settlement.journal_atlas_purchase``, keyed
    by its reservation's binding): the purchase counts once, from the scan while
    its records are on disk and from the journal once its workspace is retired.
``openai_settlement`` (the OpenAI settlements each job keeps)
    The OpenAI calls of a job's Atlas stage (planners, reviews, image edits)
    that ``provider_spend_alerts.observe_openai_job_settlements`` finds settled
    with the WebApp, summed per job at their settled cost. The account still
    counts OpenAI from the organization's bill; these rows say which job spent it.
    A call the journal also settled (``website_task_context.meter_preparation_settlement``)
    counts once, here; the journal's record keeps it attributed once the job's
    workspace is retired and its receipts are gone.
    ARTISAN records the usage retained by its authoring attempt directly in
    the journal (``artisan_anthropic_month``), outside this scan.
``webapp_overhead`` (the WebApp's own paid agent runs)
    The first look, brief reading and the WebApp's other agent runs, per run
    kind and provider for the month (``provider_spend_alerts.observe_webapp_overhead``).
    The first look and brief reading are outside every job's $40 cap (owner rule 2026-11-20); the WebApp holds
    its other runs for a job (the free-text pass, the follow-up picker) against that job's cap on its own ledger
    (since 2026-11-24), and every run is counted here once all the same. They are keyed as
    named overhead (``overhead:<name>``), never as a job; they count toward the
    account's month like any other settled row.

Gemini calls reach the journal from ``website_gemini_receipts``: each preparation
admission names its job (``website_task_context.metered_admission``), and each
answered call settles at its reported usage (provider ``gcp``, where Gemini bills).

A website job is keyed by its WebApp request id: the Pipeline names the
job's scene and screening site ``site-<request id>``.

The account's monthly limit reads this journal (``admitted_since``), so it is
not best effort where it exists: a grant whose line cannot be written while
the meter root exists is refused (``JOURNAL_UNWRITABLE``), never admitted
unseen. A process with no meter root (off the host) journals nothing, as
before; a clear stop is never refused for it. Each account writes its month in
files of at most ``ROLL_JOURNAL_FILE_BYTES``, starting a new one before a file
would pass it, and readers read every file of the month: one they cannot read
whole refuses new spend (``JournalUnreadable``), and one past
``NEAR_CAP_JOURNAL_FILE_BYTES`` is flagged before it gets there.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import re
import stat
import time
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from datetime import datetime, timezone
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from pathlib import Path
from typing import Any

EVENT_SCHEMA = "blueprint.paid_spend_meter_event.v1"
VIEW_SCHEMA = "blueprint.paid_spend_meter_view.v1"
METER_ROOT_ENV = "BLUEPRINT_PAID_SPEND_METER_ROOT"
JOB_ID_ENV = "BLUEPRINT_PAID_SPEND_JOB_ID"
#: Used only where the control-plane state directory exists (the host).
DEFAULT_METER_ROOT = Path("/var/lib/blueprint/pipeline-control-plane/paid-spend-meter")
SCREENING_QUEUE_ROOT_ENV = "BLUEPRINT_BETA_POLICY_SCREENING_QUEUE_ROOT"
DEFAULT_SCREENING_QUEUE_ROOT = Path("/var/lib/blueprint/pipeline-control-plane/beta-policy-screening")
UNATTRIBUTED = "unattributed"
SOURCE_JOURNAL = "paid_resource_admission"
SOURCE_SCREENING = "beta_policy_screening"
SOURCE_ATLAS = "atlas"
SOURCE_OPENAI = "openai_settlement"
SOURCE_OVERHEAD = "webapp_overhead"
ARTISAN_OPENAI_METER_PREFIX = "artisan-openai-call:"
#: The job key of a named overhead row: spend that belongs to no job and still counts toward the account.
OVERHEAD_PREFIX = "overhead:"
#: The meter's provider for each name the WebApp reports a run under: the bill it lands on.
OVERHEAD_PROVIDERS = {"openai": "openai", "openai_responses": "openai", "gemini": "gcp", "google": "gcp",
                      "anthropic": "anthropic"}
_OVERHEAD_NAME = re.compile(r"[a-z][a-z0-9_.-]{0,79}")
MAX_JOURNAL_FILE_BYTES = 16 * 1024 * 1024
#: A writer starts the next file of its month (``uid-<euid>.<n>.jsonl``) before a file would pass this, so no file nears
#: the readers' cap and one account's busy month never refuses every admission until the month ends (stress round 2).
ROLL_JOURNAL_FILE_BYTES = 8 * 1024 * 1024
#: A file past this was not rolled (another writer, an old release): readers flag it (``NEAR_CAP_GAP``) to page first.
NEAR_CAP_JOURNAL_FILE_BYTES = 12 * 1024 * 1024
NEAR_CAP_GAP = "paid_spend_meter_file_near_cap"
#: A grant whose journal line could not be written while the meter root exists: refused, never admitted unseen.
JOURNAL_UNWRITABLE = "paid_spend_meter_journal_unwritable"
#: A grant that named its maximum cost and no held price is held at that cost (``journal_rows``) until it settles or
#: a bill of its provider read at least ``BILL_COVERS_AFTER_SECONDS`` after it was admitted can show it: the age to
#: which the account accepts billing evidence (``account_monthly_spend.BILLING_MAX_AGE_SECONDS``).
UNPRICED_HOLD_BASIS = "admitted_maximum_cost"
BILL_COVERS_AFTER_SECONDS = 24 * 60 * 60
#: The report lists each journal hold it counted by meter key, up to this many, so a reservation judged again is not
#: counted beside its own hold; more, and none is listed (each then counts twice, never less).
MAX_JOURNAL_HOLDS_LISTED = 1000
#: Each account's journal is readable by every account on the host, written only by its own (Codex review of #3716):
#: a grant one account journals must count against the month for an admission another account judges.
JOURNAL_FILE_MODE = 0o644


class JournalUnreadable(OSError):
    """A journal file of the month could not be read whole: its grants cannot be counted, so new spend waits."""


class JournalUnwritable(OSError):
    """A grant's journal line could not be written while the meter root exists: the account could not count it."""

MAX_LEDGER_BYTES = 32 * 1024 * 1024
#: Optional on a held row: the last moment its attempt could be billed, from the attempt's own record.
BILLABLE_UNTIL_FIELD = "billable_until_epoch"
#: On a journaled grant: the most its call can cost, as the chokepoint judged it against the account's month.
MAXIMUM_COST_FIELD = "maximum_cost_usd"
#: Held from the account check to the grant's journal line, so two admissions are never judged at one count.
ADMISSION_LOCK_FILENAME = "admission.lock"
_JOB_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,191}")
_MONTH = re.compile(r"[0-9]{4}-(0[1-9]|1[0-2])")
_CENT = Decimal("0.000001")

#: The provider each shared resource class spends with, where one class means one provider.
RESOURCE_CLASS_PROVIDERS = {
    "cloud_run_cpu_job": "gcp",
    "fullenrich_work_email": "fullenrich",
    "lambda_provider_adapter": "lambda",
    "openai_api_candidate": "openai",
    "parallel_findall": "parallel",
    "provider_reconstruction_api": "world_labs",
    "runpod_live_execution": "runpod",
    "runpod_provider_adapter": "runpod",
    "runpod_serverless_active_worker": "runpod",
    "runpod_wam_async": "runpod",
    "skypilot_vast_pilot": "vast",
    "unitree_unifolm_runpod": "runpod",
    "vast_provider_adapter": "vast",
    "vast_wam_async": "vast",
}


def _utc(epoch: float) -> datetime:
    return datetime.fromtimestamp(epoch, timezone.utc)


def month_of(epoch: float) -> str:
    """The UTC calendar month ``YYYY-MM`` of an epoch."""

    return _utc(epoch).strftime("%Y-%m")


def month_start(epoch: float) -> float:
    """The epoch at which the UTC calendar month of ``epoch`` began."""

    moment = _utc(epoch)
    return datetime(moment.year, moment.month, 1, tzinfo=timezone.utc).timestamp()


def money(value: Any) -> Decimal | None:
    """A finite, non-negative USD amount, or None."""

    if isinstance(value, bool) or not isinstance(value, (str, int, float, Decimal)):
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    try:
        amount = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None
    return amount if amount.is_finite() and amount >= 0 else None


def usd(value: Decimal) -> float:
    return float(value.quantize(_CENT, rounding=ROUND_HALF_UP))


def job_key(value: Any) -> str:
    """The job a row belongs to: the WebApp request id for a website job."""

    text = str(value or "").strip()
    if text.startswith("site-"):
        text = text[5:]
    return text if _JOB_ID.fullmatch(text) else UNATTRIBUTED


def meter_root(environ: Mapping[str, str] | None = None) -> Path | None:
    """The journal root: the configured one, else the host default where the host state exists."""

    env = os.environ if environ is None else environ
    configured = str(env.get(METER_ROOT_ENV) or "").strip()
    if configured:
        return Path(configured)
    return DEFAULT_METER_ROOT if DEFAULT_METER_ROOT.parent.is_dir() else None


def _event_id(kind: str, meter_key: str, job_id: str) -> str:
    return "sha256:" + hashlib.sha256(f"{kind}\0{meter_key}\0{job_id}".encode()).hexdigest()


def record_event(event: Mapping[str, Any], *, root: Path, now: float | None = None) -> dict[str, Any]:
    """Append one event to this account's journal for its month; returns the stored row.

    One file per month and per writing account, so a root-run operator command
    and the service account never share a file. Appends are whole lines under
    an exclusive lock. Raises OSError when the root cannot be written.
    """

    at = time.time() if now is None else float(now)
    row = {"schema_version": EVENT_SCHEMA, **dict(event), "at_epoch": at, "at": _utc(at).isoformat()}
    row["event_id"] = _event_id(str(row["event"]), str(row["meter_key"]), str(row["job_id"]))
    directory = Path(root) / "events" / month_of(at)
    directory.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode()
    euid = os.geteuid()
    part = _last_part(directory, euid)
    while True:
        path = directory / (f"uid-{euid}.jsonl" if part == 0 else f"uid-{euid}.{part}.jsonl")
        descriptor = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
                             JOURNAL_FILE_MODE)
        try:
            # Every admitting account reads every account's grants (``admitted_since``): a file made before this rule,
            # or under a stricter umask, is opened up by its own writer. It holds job ids, providers and amounts.
            if stat.S_IMODE(os.fstat(descriptor).st_mode) != JOURNAL_FILE_MODE:
                os.fchmod(descriptor, JOURNAL_FILE_MODE)
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            size = os.fstat(descriptor).st_size
            if size and size + len(data) > ROLL_JOURNAL_FILE_BYTES:
                part += 1  # full: the month goes on in the next file, under the same lock rule
                continue
            if os.write(descriptor, data) != len(data):
                raise OSError("paid_spend_meter_journal_short_write")
        finally:
            os.close(descriptor)
        return row


_PART = re.compile(r"uid-(?P<uid>[0-9]+)(?:\.(?P<part>[1-9][0-9]{0,5}))?\.jsonl")


def _last_part(directory: Path, euid: int) -> int:
    """The highest numbered file of this account's month already written (0 for the first)."""
    parts = [int(found["part"] or 0) for found in (_PART.fullmatch(name) for name in os.listdir(directory))
             if found and int(found["uid"]) == euid]
    return max(parts, default=0)


def record_admission(*, job_id: Any, meter_key: str, resource_class: str, held_usd: Any = None,
                     provider: str | None = None, source: str = SOURCE_JOURNAL, root: Path,
                     now: float | None = None, **extra: Any) -> dict[str, Any]:
    held = money(held_usd)
    return record_event({
        "event": "admission", "job_id": job_key(job_id), "meter_key": str(meter_key), "source": source,
        "resource_class": resource_class, "provider": provider or RESOURCE_CLASS_PROVIDERS.get(resource_class),
        "held_usd": None if held is None else str(held), **extra,
    }, root=root, now=now)


def record_settlement(*, job_id: Any, meter_key: str, settled_usd: Any, basis: str,
                      provider: str | None = None, root: Path, now: float | None = None) -> dict[str, Any]:
    """Settle one journal admission to what it cost; the stage-2 replacement for per-job settlement."""

    settled = money(settled_usd)
    if settled is None:
        raise ValueError("paid_spend_meter_settlement_amount_invalid")
    return record_event({
        "event": "settlement", "job_id": job_key(job_id), "meter_key": str(meter_key), "source": SOURCE_JOURNAL,
        "provider": provider, "settled_usd": str(settled), "basis": str(basis),
    }, root=root, now=now)


def admission_meter_key(admission: Mapping[str, Any]) -> str | None:
    """The meter key the chokepoint journals ``admission`` under, when the admission names one.

    Its meter's ``meter_key``, else its ``allocation_binding_digest``: one reservation, however often a grant
    is taken for it. An admission that names neither gets a fresh key of its own when it is journaled.
    """

    meter = admission.get("meter") if isinstance(admission.get("meter"), Mapping) else {}
    key = meter.get("meter_key") or admission.get("allocation_binding_digest")
    return str(key) if key else None


def meter_chokepoint_admission(admission: Mapping[str, Any], *, resource_class: str, schema_version: str,
                               spend_effect: str = "new_spend", environ: Mapping[str, str] | None = None,
                               now: float | None = None, maximum_cost_usd: Any = None) -> None:
    """Journal one grant the chokepoint issued; never changes the admission.

    ``maximum_cost_usd`` is the most the call can cost, as the account's month was judged for it; it is kept
    on the row (``MAXIMUM_COST_FIELD``): a hold until the grant settles (``journal_rows``), and counted by the
    admissions judged before the next report (``admitted_since``). The account's limit reads this line, so when it
    cannot be written while the meter root exists this raises ``JournalUnwritable`` (stress round 2: a full disk
    let every grant through at the report's stale count); the chokepoint refuses the grant. With no meter root (off
    the host) nothing is journaled and nothing is raised.
    """

    env = os.environ if environ is None else environ
    root = meter_root(env)
    if root is None:
        return
    try:
        meter = admission.get("meter") if isinstance(admission.get("meter"), Mapping) else {}
        job_id = (meter.get("job_id") or admission.get("job_id") or admission.get("website_request_id")
                  or admission.get("request_id") or env.get(JOB_ID_ENV))
        binding = admission.get("allocation_binding_digest")
        meter_key = admission_meter_key(admission) or ""
        if not meter_key:
            canonical = json.dumps(admission, sort_keys=True, separators=(",", ":"), default=str)
            meter_key = "admission:" + hashlib.sha256(
                f"{resource_class}\0{schema_version}\0{canonical}\0{os.getpid()}\0{time.time_ns()}".encode()
            ).hexdigest()
        quote = money(maximum_cost_usd)
        record_admission(job_id=job_id, meter_key=meter_key, resource_class=resource_class,
                         held_usd=meter.get("held_usd"), provider=meter.get("provider"), root=root, now=now,
                         admission_schema_version=schema_version, spend_effect=spend_effect,
                         allocation_binding_digest=str(binding) if binding is not None else None,
                         **({MAXIMUM_COST_FIELD: str(quote)} if quote is not None else {}))
    except Exception as exc:  # noqa: BLE001 - whatever stopped the line, the grant would go uncounted
        if Path(root).is_dir():
            raise JournalUnwritable(f"{JOURNAL_UNWRITABLE}:{type(exc).__name__}") from exc
        return  # no meter root this process can see: nothing is journaled here, as before


@contextmanager
def admission_lock(environ: Mapping[str, str] | None = None) -> Iterator[None]:
    """One new paid admission at a time on this host, from its account check to its journal line.

    Without it two processes could both be judged at the same count and both admitted. The lock file lives in
    the meter root beside the journal; a process that cannot open it (no meter root, or a sandbox that sees
    none) is not locked, and journals nothing either (``meter_chokepoint_admission``).
    """

    descriptor = None
    root = meter_root(environ)
    if root is not None:
        path = Path(root) / ADMISSION_LOCK_FILENAME
        nofollow = getattr(os, "O_NOFOLLOW", 0)
        try:
            Path(root).mkdir(parents=True, exist_ok=True)
            try:
                descriptor = os.open(path, os.O_RDWR | os.O_CREAT | nofollow, 0o644)
            except OSError:
                # Another account's lock file, or a read-only view of the root: a shared lock needs no write.
                descriptor = os.open(path, os.O_RDONLY | nofollow)
            fcntl.flock(descriptor, fcntl.LOCK_EX)
        except OSError:
            if descriptor is not None:
                os.close(descriptor)
            descriptor = None
    try:
        yield
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _read_file(path: Path, start: int = 0) -> tuple[list[str], int]:
    """The whole lines of a journal file from byte ``start`` on, and where they end, read under a shared lock (a
    writer appends whole lines under an exclusive one). Raises ``JournalUnreadable`` for a file that is not a regular
    one, is past ``MAX_JOURNAL_FILE_BYTES`` or cannot be read: its grants cannot be counted."""
    try:
        info = os.lstat(path)
        if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_JOURNAL_FILE_BYTES:
            raise JournalUnreadable(path.name)
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        try:
            fcntl.flock(descriptor, fcntl.LOCK_SH)
            with os.fdopen(os.dup(descriptor), "rb") as stream:
                stream.seek(start)
                data = stream.read(MAX_JOURNAL_FILE_BYTES + 1)
        finally:
            os.close(descriptor)
        if start + len(data) > MAX_JOURNAL_FILE_BYTES:
            raise JournalUnreadable(path.name)
        data.decode("utf-8")  # a file that is not text cannot be counted: refused whole, as before
        # Writers append whole lines under the exclusive lock, so a line with no end is a damaged one: skipped, as a
        # line that is not JSON is, and left after the mark, so a later reader sees it as it is then.
        whole = data[:data.rfind(b"\n") + 1]
        return whole.decode("utf-8").splitlines(), start + len(whole)
    except JournalUnreadable:
        raise
    except (OSError, UnicodeDecodeError) as exc:
        raise JournalUnreadable(path.name) from exc


def admitted_since(root: Path | None, since_epoch: float, *, now: float,
                   exclude_meter_key: str | None = None, after: Mapping[str, Any] | None = None) -> Decimal:
    """What the grants journaled since a report read the journal, in the UTC month of ``now``, were admitted for.

    The account's month is judged from a report written up to a few minutes before (``account_monthly_spend``);
    a grant journaled after it read the journal is in none of its figures. ``after`` is what that report read of
    each file of the month (``"<month>/<file>": bytes``, ``journal_read_marks``): the lines after it count, so no
    clock decides and a grant counts once, in the report or here; a file it does not name is read whole. Without
    ``after`` (a report that names none), the grants journaled at or after ``since_epoch`` count. Each counts at the
    most it was admitted or held for (``MAXIMUM_COST_FIELD``, ``held_usd``), once per meter key.
    ``exclude_meter_key`` is the reservation being judged, whose own cost the caller adds. A stop counts nothing,
    nor does an admission that names no amount: none is made up for it. A file of the month this process cannot read
    whole raises ``JournalUnreadable``: its grants cannot be counted, so the caller refuses new spend.
    """

    if root is None:
        return Decimal(0)
    month = month_of(now)
    directory = Path(root) / "events" / month
    if directory.is_symlink() or not directory.is_dir():
        return Decimal(0)
    amounts: dict[str, Decimal] = {}
    for path in sorted(directory.glob("uid-*.jsonl")):
        # Fail closed (Codex review of #3716): a file of the month this process cannot read whole, another account's
        # or a damaged one, may hold grants; counting it as nothing could admit spend past the limit.
        start = 0
        if after is not None:
            mark = after.get(f"{month}/{path.name}")
            size = path.lstat().st_size if path.exists() else 0
            start = int(mark) if isinstance(mark, int) and not isinstance(mark, bool) and 0 <= mark <= size else 0
        lines, _end = _read_file(path, start)
        for number, line in enumerate(lines):
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if (not isinstance(row, dict) or row.get("schema_version") != EVENT_SCHEMA
                    or row.get("event") != "admission" or row.get("spend_effect") == "stop"):
                continue
            at = _finite(row.get("at_epoch"))
            if at is None or (after is None and at < since_epoch):
                continue
            key = str(row.get("meter_key") or "") or f"{path.name}:{start}:{number}"
            if exclude_meter_key and key == exclude_meter_key:
                continue
            figures = [value for value in (money(row.get(MAXIMUM_COST_FIELD)), money(row.get("held_usd")))
                       if value is not None]
            if figures:
                amounts[key] = max(amounts.get(key, Decimal(0)), *figures)
    return sum(amounts.values(), Decimal(0))


def _journal_files(root: Path) -> list[Path]:
    events = Path(root) / "events"
    if not events.is_dir() or events.is_symlink():
        return []
    files = []
    for month in sorted(events.iterdir()):
        if month.is_symlink() or not month.is_dir() or not _MONTH.fullmatch(month.name):
            continue
        files.extend(sorted(path for path in month.glob("uid-*.jsonl") if not path.is_symlink()))
    return files


def read_journal(root: Path | None, *, marks: dict[str, int] | None = None
                 ) -> tuple[list[dict[str, Any]], list[str]]:
    """Every journal event, and a gap for each file that could not be read whole (``NEAR_CAP_GAP`` too for one past
    ``NEAR_CAP_JOURNAL_FILE_BYTES``). ``marks``, when given, is filled with how far each file was read
    (``"<month>/<file>": bytes``): a report names it so the admissions after it count only what it did not read."""

    if root is None:
        return [], []
    events, gaps = [], []
    for path in _journal_files(Path(root)):
        try:
            lines, end = _read_file(path)
        except JournalUnreadable:
            gaps.append("paid_spend_meter_journal_file_unreadable")
            continue
        if end > NEAR_CAP_JOURNAL_FILE_BYTES:
            gaps.append(NEAR_CAP_GAP)
        if marks is not None:
            marks[f"{path.parent.name}/{path.name}"] = end
        for line in lines:
            try:
                row = json.loads(line)
            except ValueError:
                gaps.append("paid_spend_meter_journal_line_invalid")
                continue
            if isinstance(row, dict) and row.get("schema_version") == EVENT_SCHEMA:
                events.append(row)
            else:
                gaps.append("paid_spend_meter_journal_line_invalid")
    return events, sorted(set(gaps))


def journal_rows(events: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Fold admissions and settlements into one row per meter key; repeats count once."""

    seen: set[str] = set()
    admissions: dict[tuple[str, str], dict[str, Any]] = {}
    settlements: dict[tuple[str, str], Mapping[str, Any]] = {}
    for event in sorted(events, key=lambda row: (float(row.get("at_epoch") or 0), str(row.get("event_id")))):
        event_id = str(event.get("event_id") or "")
        if event_id in seen:
            continue
        seen.add(event_id)
        key = (str(event.get("job_id") or UNATTRIBUTED), str(event.get("meter_key") or ""))
        if event.get("event") == "admission":
            admissions.setdefault(key, dict(event))
        elif event.get("event") == "settlement":
            settlements.setdefault(key, event)
    rows = []
    for key in sorted(set(admissions) | set(settlements)):
        admission = admissions.get(key, {})
        settlement = settlements.get(key)
        held = money(admission.get("held_usd"))
        row = {"source": SOURCE_JOURNAL, "job_id": key[0], "meter_key": key[1],
               "resource_class": admission.get("resource_class"),
               "provider": (settlement or {}).get("provider") or admission.get("provider"),
               "admitted_at_epoch": admission.get("at_epoch"), "held_usd": None, "settled_usd": None,
               "settled_at_epoch": None, "status": "unpriced",
               BILLABLE_UNTIL_FIELD: _finite(admission.get(BILLABLE_UNTIL_FIELD))}
        quote = money(admission.get(MAXIMUM_COST_FIELD))
        admitted = _finite(admission.get("at_epoch"))
        if settlement is not None:
            row.update(status="settled", settled_usd=money(settlement.get("settled_usd")),
                       settled_at_epoch=settlement.get("at_epoch"))
        elif held is not None:
            row.update(status="held", held_usd=held)
        elif quote is not None and admitted is not None and admission.get("spend_effect") != "stop":
            # It named the most it can cost and no price: held at that until it settles or a bill can show it, never
            # past the month after the bill could (``meter_view``).
            row.update(status="held", held_usd=quote, hold_basis=UNPRICED_HOLD_BASIS,
                       **{BILLABLE_UNTIL_FIELD: row[BILLABLE_UNTIL_FIELD] or admitted + BILL_COVERS_AFTER_SECONDS})
        rows.append(row)
    return rows


def _finite(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        return float(value)
    return None


def _epoch(value: Any) -> float | None:
    if _finite(value) is not None:
        return float(value)
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return parsed.timestamp() if parsed.tzinfo is not None else None


def atlas_rows(world_labs: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    """Atlas attempts as ``observe_world_labs_spend`` counted them, one row per attempt."""

    rows = []
    for charge in (world_labs or {}).get("charges") or []:
        if not isinstance(charge, Mapping):
            continue
        price, at = money(charge.get("usd")), _epoch(charge.get("at_epoch"))
        if price is None or at is None:
            continue
        held = charge.get("held") is True
        rows.append({"source": SOURCE_ATLAS, "job_id": job_key(charge.get("job_id")),
                     "meter_key": str(charge.get("attempt") or f"atlas:{at}"),
                     "resource_class": "provider_reconstruction_api", "provider": "world_labs",
                     "held_usd": price if held else None, "settled_usd": None if held else price,
                     "settled_at_epoch": None if held else at, "status": "held" if held else "settled"})
    return rows


def openai_rows(openai: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    """Each job's settled OpenAI calls as ``observe_openai_job_settlements`` summed them, one row per job."""

    rows = []
    for charge in (openai or {}).get("job_charges") or []:
        if not isinstance(charge, Mapping):
            continue
        price, at = money(charge.get("usd")), _epoch(charge.get("at_epoch"))
        if price is None or at is None:
            continue
        rows.append({"source": SOURCE_OPENAI, "job_id": job_key(charge.get("job_id")),
                     "meter_key": str(charge.get("attempt") or f"openai:{at}"),
                     "resource_class": "openai_api_candidate", "provider": "openai",
                     "held_usd": None, "settled_usd": price, "settled_at_epoch": at, "status": "settled"})
    return rows


def overhead_rows(overhead: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    """The WebApp's named overhead for the month, one settled row per name and provider."""

    rows = []
    for charge in (overhead or {}).get("charges") or []:
        if not isinstance(charge, Mapping):
            continue
        name, price, at = str(charge.get("name") or ""), money(charge.get("usd")), _epoch(charge.get("at_epoch"))
        provider = str(charge.get("provider") or "").strip().lower()
        if not _OVERHEAD_NAME.fullmatch(name) or not provider or price is None or at is None:
            continue
        meter_provider = OVERHEAD_PROVIDERS.get(provider, provider)
        rows.append({"source": SOURCE_OVERHEAD, "job_id": OVERHEAD_PREFIX + name,
                     "meter_key": f"{OVERHEAD_PREFIX}{name}:{meter_provider}", "resource_class": None,
                     "provider": meter_provider, "held_usd": None, "settled_usd": price, "settled_at_epoch": at,
                     "status": "settled"})
    return rows


def is_overhead(job_id: Any) -> bool:
    return str(job_id or "").startswith(OVERHEAD_PREFIX)


def _zero() -> dict[str, Decimal]:
    return {"settled_usd": Decimal(0), "held_usd": Decimal(0)}


def hold_counts_in_month(row: Mapping[str, Any], *, now: float) -> bool:
    """Whether a held row's attempt could still be billed in the UTC month of ``now``.

    A hold stands in for a bill that has not been read yet. Once the attempt's
    own record says when it ended (``billable_until_epoch``: its recorded close),
    the hold counts only in months the attempt was still live in: an attempt
    that ended before this month began was billed, if at all, in an earlier
    month's provider reading. One live across 00:00 UTC on the 1st counts in
    both months. A hold with no recorded end is open and counts every month.
    """

    until = _finite(row.get(BILLABLE_UNTIL_FIELD))
    return until is None or until >= month_start(now)


def covered_by_bill(row: Mapping[str, Any], bills: Mapping[str, float] | None) -> bool:
    """Whether a grant held at its maximum cost (``UNPRICED_HOLD_BASIS``) is now shown by its provider's bill: a
    reading of that bill (``bills``: provider to the epoch it was read at) taken ``BILL_COVERS_AFTER_SECONDS`` or
    more after the grant was admitted. The bill then counts it, and the hold would count it twice."""

    observed = (bills or {}).get(str(row.get("provider") or ""))
    admitted = _finite(row.get("admitted_at_epoch"))
    return (row.get("hold_basis") == UNPRICED_HOLD_BASIS and isinstance(observed, (int, float))
            and admitted is not None and observed >= admitted + BILL_COVERS_AFTER_SECONDS)


def meter_view(rows: Iterable[Mapping[str, Any]], *, now: float, gaps: Iterable[str] = (),
               bills: Mapping[str, float] | None = None, journal_marks: Mapping[str, int] | None = None
               ) -> dict[str, Any]:
    """Per-job metered spend for the UTC month of ``now``, and the totals of those rows.

    A settled row counts in the month it settled. A held row counts in every
    month its attempt could still be billed in (``hold_counts_in_month``); one
    whose attempt ended in an earlier month is listed under
    ``old_unsettled_holds`` for someone to settle and counts in no total. A
    grant held at its maximum cost (``UNPRICED_HOLD_BASIS``) counts as one until
    a bill can show it (``covered_by_bill``) and, past its month, as the call
    it was. An unpriced admission counts as a call in the month it was
    admitted. The totals are the sums of the job rows, by construction.

    ``journal``: how far the journal was read (``journal_marks``, for the
    admissions after it) and each journal hold counted, by meter key, so a
    reservation judged again is not counted beside its own hold.
    """

    month = month_of(now)
    jobs: dict[str, dict[str, Any]] = {}
    old_holds: list[dict[str, Any]] = []
    journal_holds: dict[str, Decimal] = {}
    for row in rows:
        job = jobs.setdefault(str(row.get("job_id") or UNATTRIBUTED), {
            **_zero(), "admissions": 0, "unpriced_admissions": 0, "by_source": {}, "by_provider_settled_usd": {}})
        source = job["by_source"].setdefault(str(row.get("source")), _zero())
        status = row.get("status")
        if status == "held" and row.get("hold_basis") == UNPRICED_HOLD_BASIS and (
                covered_by_bill(row, bills) or not hold_counts_in_month(row, now=now)):
            status = "unpriced"  # its bill counts it now, or its month is over: the call it was
        if status == "held" and row.get("source") == SOURCE_JOURNAL and hold_counts_in_month(row, now=now):
            key = str(row.get("meter_key") or "")
            journal_holds[key] = journal_holds.get(key, Decimal(0)) + (row.get("held_usd") or Decimal(0))
        settled_at, admitted_at = row.get("settled_at_epoch"), row.get("admitted_at_epoch")
        if status == "settled" and isinstance(settled_at, (int, float)) and month_of(settled_at) == month:
            amount = row.get("settled_usd") or Decimal(0)
            job["settled_usd"] += amount
            source["settled_usd"] += amount
            provider = str(row.get("provider") or "other")
            job["by_provider_settled_usd"][provider] = job["by_provider_settled_usd"].get(provider, Decimal(0)) + amount
        elif status == "held" and not hold_counts_in_month(row, now=now):
            old_holds.append({"job_id": str(row.get("job_id") or UNATTRIBUTED), "source": str(row.get("source")),
                              "meter_key": str(row.get("meter_key") or ""), "provider": row.get("provider"),
                              "held_usd": row.get("held_usd") or Decimal(0),
                              "billable_until": _utc(float(row[BILLABLE_UNTIL_FIELD])).isoformat(),
                              **({"reason": str(row["hold_reason"])} if row.get("hold_reason") else {})})
        elif status == "held":
            amount = row.get("held_usd") or Decimal(0)
            job["held_usd"] += amount
            source["held_usd"] += amount
        if row.get("source") == SOURCE_JOURNAL and isinstance(admitted_at, (int, float)) and month_of(admitted_at) == month:
            job["admissions"] += 1
            job["unpriced_admissions"] += int(status == "unpriced")
    totals = {**_zero(), "admissions": 0, "unpriced_admissions": 0, "by_source": {}, "by_provider_settled_usd": {}}
    overhead_by_provider: dict[str, Decimal] = {}
    for job_id, job in jobs.items():
        if is_overhead(job_id):
            for provider, amount in job["by_provider_settled_usd"].items():
                overhead_by_provider[provider] = overhead_by_provider.get(provider, Decimal(0)) + amount
    for job in jobs.values():
        for key in ("settled_usd", "held_usd", "admissions", "unpriced_admissions"):
            totals[key] += job[key]
        for name, amounts in job["by_source"].items():
            bucket = totals["by_source"].setdefault(name, _zero())
            for key in ("settled_usd", "held_usd"):
                bucket[key] += amounts[key]
        for provider, amount in job["by_provider_settled_usd"].items():
            totals["by_provider_settled_usd"][provider] = totals["by_provider_settled_usd"].get(provider, Decimal(0)) + amount

    def project(value: Mapping[str, Any]) -> dict[str, Any]:
        return {"settled_usd": usd(value["settled_usd"]), "held_usd": usd(value["held_usd"]),
                "metered_usd": usd(value["settled_usd"] + value["held_usd"]),
                "admissions": value["admissions"], "unpriced_admissions": value["unpriced_admissions"],
                "by_source": {name: {key: usd(amount) for key, amount in amounts.items()}
                              for name, amounts in sorted(value["by_source"].items())},
                "by_provider_settled_usd": {name: usd(amount) for name, amount in sorted(value["by_provider_settled_usd"].items())}}

    job_rows = sorted((row for row in ({"job_id": job_id, **project(value)} for job_id, value in jobs.items())
                       if row["metered_usd"] or row["admissions"]),
                      key=lambda row: (-row["metered_usd"], row["job_id"]))
    old_total = sum((hold["held_usd"] for hold in old_holds), Decimal(0))
    old_holds.sort(key=lambda hold: (-hold["held_usd"], hold["job_id"], hold["source"], hold["meter_key"]))
    journal = {"period": month, "read_bytes": dict(journal_marks) if journal_marks is not None else None,
               "holds_usd": ({key: str(amount) for key, amount in sorted(journal_holds.items())}
                             if len(journal_holds) <= MAX_JOURNAL_HOLDS_LISTED else None)}
    return {"schema_version": VIEW_SCHEMA, "period": month, "generated_at": _utc(now).isoformat(),
            "journal": journal, "jobs": job_rows,
            "totals": {**project(totals),
                       # Of the settled spend per provider, what is named overhead rather than a job's.
                       "overhead_by_provider_settled_usd": {name: usd(amount) for name, amount
                                                            in sorted(overhead_by_provider.items())}},
            "old_unsettled_held_usd": usd(old_total),
            "old_unsettled_holds": [{**hold, "held_usd": usd(hold["held_usd"])} for hold in old_holds],
            "_exact": {"settled_usd": totals["settled_usd"], "held_usd": totals["held_usd"],
                       "old_unsettled_held_usd": old_total,
                       "by_source": {name: dict(amounts) for name, amounts in totals["by_source"].items()},
                       "held_outside_provider_readings_usd": sum(
                           (amounts["held_usd"] for name, amounts in totals["by_source"].items() if name != SOURCE_ATLAS),
                           Decimal(0)),
                       "by_provider_settled_usd": dict(totals["by_provider_settled_usd"]),
                       "overhead_by_provider_settled_usd": dict(overhead_by_provider)},
            "gaps": sorted(set(gaps)), "complete": not list(gaps)}


def public_view(view: Mapping[str, Any], *, max_jobs: int | None = None) -> dict[str, Any]:
    """The view without its exact working figures, optionally keeping the largest jobs only."""

    result = {key: value for key, value in view.items() if not key.startswith("_")}
    for name in ("jobs", "old_unsettled_holds"):
        listed = list(view.get(name) or [])
        if max_jobs is not None and len(listed) > max_jobs:
            result[name] = listed[:max_jobs]
            result[f"{name}_omitted"] = len(listed) - max_jobs
    return result


__all__ = [
    "ADMISSION_LOCK_FILENAME", "BILL_COVERS_AFTER_SECONDS", "DEFAULT_METER_ROOT", "EVENT_SCHEMA", "JOB_ID_ENV",
    "JOURNAL_FILE_MODE", "JOURNAL_UNWRITABLE", "JournalUnreadable", "JournalUnwritable", "MAXIMUM_COST_FIELD",
    "NEAR_CAP_GAP", "NEAR_CAP_JOURNAL_FILE_BYTES", "ROLL_JOURNAL_FILE_BYTES", "UNPRICED_HOLD_BASIS", "covered_by_bill",
    "METER_ROOT_ENV", "RESOURCE_CLASS_PROVIDERS",
    "BILLABLE_UNTIL_FIELD", "OVERHEAD_PREFIX", "OVERHEAD_PROVIDERS", "SOURCE_OPENAI", "SOURCE_OVERHEAD",
    "UNATTRIBUTED", "VIEW_SCHEMA", "admission_lock", "admission_meter_key", "admitted_since", "atlas_rows",
    "openai_rows", "overhead_rows", "is_overhead",
    "hold_counts_in_month", "job_key",
    "journal_rows", "meter_chokepoint_admission", "meter_root", "meter_view", "month_of", "month_start",
    "public_view", "read_journal",
    "record_admission", "record_event", "record_settlement",
]

