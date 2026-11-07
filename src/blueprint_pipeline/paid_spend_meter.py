"""Meter every paid admission and its settlement against its job (metered spend, stage 1).

Owner decision 2026-11-07, "Meter + monthly cap": a job gets no budget or
confirmation of its own. Every paid call is metered against the job that made
it and shown, and one account-wide monthly limit
(``account_monthly_spend``) stops new spend. This module is the meter. It
reads and reports; it never admits, refuses, reserves or releases anything.

A meter row is one paid admission and, once known, its settlement. Rows come
from three places. Two of them already record admissions and settlements per
job, so they are read where they are, not copied:

``paid_resource_admission`` (journal)
    Every grant the shared chokepoint issues is appended here
    (``meter_chokepoint_admission``), keyed by the job the admission names
    (``admission["meter"]["job_id"]``, else the admission's own ``job_id``,
    ``website_request_id`` or ``request_id``, else ``BLUEPRINT_PAID_SPEND_JOB_ID``
    in the process environment, else ``unattributed``). An admission carries a
    price only when it names one (``admission["meter"]["held_usd"]``); no
    lane does yet, so these rows count calls, not dollars, until stage 2.
    ``record_settlement`` is the matching settlement record.
``beta_policy_screening`` (screening ledger)
    Each attempt in ``screening_ledger.v1.json``: held at its cap while
    reserved or committed without a verified settlement, then its settled
    cost (``beta_policy_screening_settlement.settled_cost``).
``atlas`` (World Labs attempt records)
    Each Atlas attempt that ``provider_spend_alerts.observe_world_labs_spend``
    counts: a completed operation at its reviewed price, one whose outcome is
    unknown held at that price (``atlas_spend_settlement``).

A website job is keyed by its WebApp request id: the Pipeline names the
job's scene and screening site ``site-<request id>``.

The journal is best effort. A process whose sandbox cannot write the meter
root records nothing and its admission proceeds, because the monthly limit is
read from reconciled provider billing, not from this journal.
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
from collections.abc import Iterable, Mapping
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
MAX_JOURNAL_FILE_BYTES = 16 * 1024 * 1024
MAX_LEDGER_BYTES = 32 * 1024 * 1024
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
    path = directory / f"uid-{os.geteuid()}.jsonl"
    descriptor = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        os.write(descriptor, (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode())
    finally:
        os.close(descriptor)
    return row


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


def meter_chokepoint_admission(admission: Mapping[str, Any], *, resource_class: str, schema_version: str,
                               environ: Mapping[str, str] | None = None, now: float | None = None) -> None:
    """Journal one grant the chokepoint issued. Never raises and never changes the admission."""

    try:
        env = os.environ if environ is None else environ
        root = meter_root(env)
        if root is None:
            return
        meter = admission.get("meter") if isinstance(admission.get("meter"), Mapping) else {}
        job_id = (meter.get("job_id") or admission.get("job_id") or admission.get("website_request_id")
                  or admission.get("request_id") or env.get(JOB_ID_ENV))
        binding = admission.get("allocation_binding_digest")
        meter_key = str(meter.get("meter_key") or binding or "")
        if not meter_key:
            canonical = json.dumps(admission, sort_keys=True, separators=(",", ":"), default=str)
            meter_key = "admission:" + hashlib.sha256(
                f"{resource_class}\0{schema_version}\0{canonical}\0{os.getpid()}\0{time.time_ns()}".encode()
            ).hexdigest()
        record_admission(job_id=job_id, meter_key=meter_key, resource_class=resource_class,
                         held_usd=meter.get("held_usd"), provider=meter.get("provider"), root=root, now=now,
                         admission_schema_version=schema_version,
                         allocation_binding_digest=str(binding) if binding is not None else None)
    except Exception:  # noqa: BLE001 - metering is visibility; the monthly limit reads billing, not this
        return


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


def read_journal(root: Path | None) -> tuple[list[dict[str, Any]], list[str]]:
    """Every journal event, and a gap for each file that could not be read whole."""

    if root is None:
        return [], []
    events, gaps = [], []
    for path in _journal_files(Path(root)):
        try:
            info = os.lstat(path)
            if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_JOURNAL_FILE_BYTES:
                gaps.append("paid_spend_meter_journal_file_unreadable")
                continue
            lines = path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError):
            gaps.append("paid_spend_meter_journal_file_unreadable")
            continue
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
               "settled_at_epoch": None, "status": "unpriced"}
        if settlement is not None:
            row.update(status="settled", settled_usd=money(settlement.get("settled_usd")),
                       settled_at_epoch=settlement.get("at_epoch"))
        elif held is not None:
            row.update(status="held", held_usd=held)
        rows.append(row)
    return rows


def _epoch(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
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


def _zero() -> dict[str, Decimal]:
    return {"settled_usd": Decimal(0), "held_usd": Decimal(0)}


def meter_view(rows: Iterable[Mapping[str, Any]], *, now: float, gaps: Iterable[str] = ()) -> dict[str, Any]:
    """Per-job metered spend for the UTC month of ``now``, and the totals of those rows.

    A settled row counts in the month it settled. A held row counts while it
    is held, whatever month it was admitted in: held money is held now. An
    unpriced admission counts as a call in the month it was admitted. The
    totals are the sums of the job rows, by construction.
    """

    month = month_of(now)
    jobs: dict[str, dict[str, Any]] = {}
    for row in rows:
        job = jobs.setdefault(str(row.get("job_id") or UNATTRIBUTED), {
            **_zero(), "admissions": 0, "unpriced_admissions": 0, "by_source": {}, "by_provider_settled_usd": {}})
        source = job["by_source"].setdefault(str(row.get("source")), _zero())
        status = row.get("status")
        settled_at, admitted_at = row.get("settled_at_epoch"), row.get("admitted_at_epoch")
        if status == "settled" and isinstance(settled_at, (int, float)) and month_of(settled_at) == month:
            amount = row.get("settled_usd") or Decimal(0)
            job["settled_usd"] += amount
            source["settled_usd"] += amount
            provider = str(row.get("provider") or "other")
            job["by_provider_settled_usd"][provider] = job["by_provider_settled_usd"].get(provider, Decimal(0)) + amount
        elif status == "held":
            amount = row.get("held_usd") or Decimal(0)
            job["held_usd"] += amount
            source["held_usd"] += amount
        if row.get("source") == SOURCE_JOURNAL and isinstance(admitted_at, (int, float)) and month_of(admitted_at) == month:
            job["admissions"] += 1
            job["unpriced_admissions"] += int(status == "unpriced")
    totals = {**_zero(), "admissions": 0, "unpriced_admissions": 0, "by_source": {}, "by_provider_settled_usd": {}}
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
    return {"schema_version": VIEW_SCHEMA, "period": month, "generated_at": _utc(now).isoformat(),
            "jobs": job_rows, "totals": project(totals),
            "_exact": {"settled_usd": totals["settled_usd"], "held_usd": totals["held_usd"],
                       "held_outside_provider_readings_usd": sum(
                           (amounts["held_usd"] for name, amounts in totals["by_source"].items() if name != SOURCE_ATLAS),
                           Decimal(0)),
                       "by_provider_settled_usd": dict(totals["by_provider_settled_usd"])},
            "gaps": sorted(set(gaps)), "complete": not list(gaps)}


def public_view(view: Mapping[str, Any], *, max_jobs: int | None = None) -> dict[str, Any]:
    """The view without its exact working figures, optionally keeping the largest jobs only."""

    result = {key: value for key, value in view.items() if not key.startswith("_")}
    jobs = list(view.get("jobs") or [])
    if max_jobs is not None and len(jobs) > max_jobs:
        result["jobs"] = jobs[:max_jobs]
        result["jobs_omitted"] = len(jobs) - max_jobs
    return result


__all__ = [
    "DEFAULT_METER_ROOT", "EVENT_SCHEMA", "JOB_ID_ENV", "METER_ROOT_ENV", "RESOURCE_CLASS_PROVIDERS",
    "UNATTRIBUTED", "VIEW_SCHEMA", "atlas_rows", "job_key", "journal_rows",
    "meter_chokepoint_admission", "meter_root", "meter_view", "month_of", "public_view", "read_journal",
    "record_admission", "record_event", "record_settlement",
]

