"""The off-host GPU reaper: destroys Blueprint's Vast machines that outlived their deadline.

Everything else that stops a rented machine runs on the control-plane host: each launch's
watchdog (``groot_oscar_runpod_watchdog``), the spend guard and the pending-teardown reaper. If
the host dies, hangs or is cut off, they stop with it and a machine keeps billing. This module
runs somewhere else (a Cloud Run job on a Cloud Scheduler tick, ``deploy/terraform``) and needs
nothing from the host but the one object the host publishes
(``off_host_gpu_reaper_heartbeat``): a heartbeat and the deadline of every live watchdog.

A Blueprint machine (its Vast label matches ``LABEL_PATTERN``) is destroyed when:

- it is past its published deadline plus ``DEADLINE_GRACE_SECONDS``, host alive or not; or
- the heartbeat is older than ``HEARTBEAT_STALE_SECONDS`` and the machine is past its published
  deadline; or it is absent from a list the host marked complete and started more than
  ``ORPHAN_START_MARGIN_SECONDS`` before that heartbeat (its watchdog died first); or it is absent
  otherwise and older than the longest deadline the host's code can arm (published with the
  heartbeat).

A machine inside its deadline is never destroyed. Nothing is acted on when no heartbeat was ever
published, when the published object or the provider's list is unreadable (the pass is
``unknown``), or for a label that is not Blueprint's (left, ownership unknown). Destroying stops
spend; it needs no budget admission. After the deletes the list is read again: a machine is
``destroyed`` only when it is gone from it or terminal there, otherwise ``failed``.

Each pass writes one report object and prints it. The job exits non-zero when the pass is
``unknown`` or ``failed``, so the job-failure alert in Terraform pages.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Protocol

STATE_SCHEMA_VERSION = "off_host_gpu_reaper_host_state.v1"
REPORT_SCHEMA_VERSION = "off_host_gpu_reaper_pass.v1"
#: The one object the host rewrites on every heartbeat, and where each pass's report goes.
STATE_OBJECT = "host-state/latest.json"
REPORT_PREFIX = "reports/"
BUCKET_ENV = "BLUEPRINT_OFF_HOST_GPU_REAPER_BUCKET"
#: A watchdog that is still running at its deadline is mid-teardown; give it this long.
DEADLINE_GRACE_SECONDS = 15 * 60
#: The host publishes every two minutes; this long without one means it is down or cut off.
HEARTBEAT_STALE_SECONDS = 30 * 60
#: A launch arms its watchdog before it creates the machine
#: (``vast_independent_watchdog_control.arm_independent_vast_watchdog``), so a machine that
#: existed at a publish was covered in it unless its watchdog was gone. The next publish is at most
#: about 4.25 minutes away (a 2-minute timer, 15 s jitter, a run of up to 2 minutes); twice that,
#: rounded up, leaves room for the clock difference between Vast's start time and the host's.
ORPHAN_START_MARGIN_SECONDS = 10 * 60
#: Every Blueprint launch labels its instance ``blueprint-...`` (the watchdog prefixes,
#: ``groot_oscar_runpod_watchdog.CANARY_NAME_PREFIXES``). Vast caps a label at 256 characters.
LABEL_PATTERN = re.compile(r"blueprint-[a-z0-9-]{1,246}")
MAX_STATE_BYTES = 4 * 1024 * 1024
MAX_STATE_ENTRIES = 10_000
VERIFY_ATTEMPTS = 3
VERIFY_DELAY_SECONDS = 10.0


class ObjectStore(Protocol):
    def read(self, name: str) -> bytes | None: ...

    def write(self, name: str, data: bytes) -> None: ...


@dataclass(frozen=True)
class Entry:
    """One live watchdog's scope and deadline, as the host published it."""

    deadline_epoch: float
    instance_id: int | None = None
    label: str | None = None
    label_prefix: str | None = None

    def covers(self, row: "Instance") -> bool:
        return bool((self.instance_id is not None and self.instance_id == row.instance_id)
                    or (self.label is not None and self.label == row.label)
                    or (self.label_prefix is not None and row.label.startswith(self.label_prefix)))


@dataclass(frozen=True)
class HostState:
    heartbeat_epoch: float
    longest_deadline_seconds: float
    entries: tuple[Entry, ...]
    #: The host read every live watchdog's deadline. Missing (an older publisher) reads as not complete.
    deadlines_complete: bool = False


@dataclass(frozen=True)
class Instance:
    instance_id: int
    label: str
    status: str
    started_epoch: float | None


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _positive_id(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    if isinstance(value, str) and value.isdigit():
        return int(value) if int(value) > 0 else None
    return None


def parse_host_state(raw: bytes) -> HostState:
    """The published object, strictly: anything unexpected raises ``ValueError``."""

    if len(raw) > MAX_STATE_BYTES:
        raise ValueError("host_state_too_large")
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, Mapping) or payload.get("schema_version") != STATE_SCHEMA_VERSION:
        raise ValueError("host_state_schema_invalid")
    heartbeat = _number(payload.get("heartbeat_epoch"))
    longest = _number(payload.get("longest_deadline_seconds"))
    rows = payload.get("entries")
    if heartbeat is None or heartbeat <= 0:
        raise ValueError("host_state_heartbeat_invalid")
    if longest is None or longest <= 0:
        raise ValueError("host_state_longest_deadline_invalid")
    if not isinstance(rows, list) or len(rows) > MAX_STATE_ENTRIES:
        raise ValueError("host_state_entries_invalid")
    complete = payload.get("deadlines_complete", False)
    if type(complete) is not bool:
        raise ValueError("host_state_completeness_invalid")
    entries = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("host_state_entry_invalid")
        deadline = _number(row.get("deadline_epoch"))
        instance_id = row.get("instance_id")
        label, prefix = row.get("label"), row.get("label_prefix")
        if (deadline is None or deadline <= 0
                or (instance_id is not None and _positive_id(instance_id) is None)
                or (label is not None and (not isinstance(label, str) or not LABEL_PATTERN.fullmatch(label)))
                or (prefix is not None and (not isinstance(prefix, str) or not LABEL_PATTERN.fullmatch(prefix)))
                or (label is None and prefix is None)):
            raise ValueError("host_state_entry_invalid")
        entries.append(Entry(deadline_epoch=deadline, instance_id=_positive_id(instance_id) if instance_id is not None
                             else None, label=label, label_prefix=prefix))
    return HostState(heartbeat_epoch=heartbeat, longest_deadline_seconds=longest, entries=tuple(entries),
                     deadlines_complete=complete)


def read_vast_inventory(api_key: str) -> list[Instance]:
    """Every instance on the account, from one ``GET /instances/``; raises when it cannot be read whole."""

    from .vast_provider_adapter import _api_json, _instance_inventory_valid, _instance_list_rows, _instance_status

    status, payload = _api_json(method="GET", path="/instances/", api_key=api_key, timeout_seconds=30)
    if status != 200 or not _instance_inventory_valid(payload):
        raise ValueError(f"vast_inventory_unreadable:http_{status}")
    rows = []
    for row in _instance_list_rows(payload):
        instance_id = _positive_id(row.get("id") or row.get("instance_id") or row.get("contract_id"))
        if instance_id is None:
            raise ValueError("vast_inventory_row_id_invalid")
        started = _number(row.get("start_date"))
        rows.append(Instance(instance_id=instance_id, label=str(row.get("label") or "").strip(),
                             status=str(_instance_status(row) or "").strip().lower(),
                             started_epoch=started if started is not None and started > 0 else None))
    return rows


def _terminal_statuses() -> frozenset[str]:
    from .vast_provider_adapter import VAST_TERMINAL_INSTANCE_STATUSES

    return frozenset(str(status).lower() for status in VAST_TERMINAL_INSTANCE_STATUSES)


def decide(row: Instance, state: HostState, *, now: float) -> dict[str, Any]:
    """What to do with one live instance: ``destroy``, ``leave`` or ``unknown``, and why."""

    if not LABEL_PATTERN.fullmatch(row.label):
        return {"action": "leave", "reason": "label_not_blueprint", "ownership": "unknown"}
    stale = now - state.heartbeat_epoch > HEARTBEAT_STALE_SECONDS
    age = None if row.started_epoch is None else max(0.0, now - row.started_epoch)
    deadlines = [entry.deadline_epoch for entry in state.entries if entry.covers(row)]
    if deadlines:
        # Two watchdogs can cover one machine while one supersedes the other: the later deadline holds.
        deadline = max(deadlines)
        decision: dict[str, Any] = {"deadline_epoch": deadline}
        if now > deadline + DEADLINE_GRACE_SECONDS:
            return {**decision, "action": "destroy", "reason": "past_deadline_and_grace"}
        if stale and now > deadline:
            return {**decision, "action": "destroy", "reason": "past_deadline_host_heartbeat_stale"}
        return {**decision, "action": "leave", "reason": "within_deadline"}
    if not stale:
        return {"action": "leave", "reason": "not_published_host_alive"}
    if age is None:
        return {"action": "unknown", "reason": "not_published_host_heartbeat_stale_age_unknown"}
    if (state.deadlines_complete and row.started_epoch is not None
            and row.started_epoch < state.heartbeat_epoch - ORPHAN_START_MARGIN_SECONDS):
        # It existed when the host last read every watchdog, and none covered it: its watchdog died
        # before the host did, and nothing is left to stop it.
        return {"action": "destroy", "reason": "not_published_in_complete_list_host_heartbeat_stale",
                "age_seconds": age}
    if age > state.longest_deadline_seconds:
        return {"action": "destroy", "reason": "not_published_older_than_longest_deadline_host_heartbeat_stale",
                "age_seconds": age}
    return {"action": "leave", "reason": "not_published_within_longest_deadline", "age_seconds": age}


def _observe(store: ObjectStore, inventory: Callable[[], Sequence[Instance]]) -> tuple[HostState, list[Instance]]:
    try:
        raw = store.read(STATE_OBJECT)
    except Exception:  # noqa: BLE001 - an unreadable store is unknown, never "no heartbeat"
        raise LookupError("host_state_unreadable") from None
    if raw is None:
        raise LookupError("host_heartbeat_never_published")
    try:
        state = parse_host_state(raw)
    except (ValueError, UnicodeDecodeError) as error:
        raise LookupError(str(error) if str(error).startswith("host_state_") else "host_state_unreadable") from None
    try:
        rows = list(inventory())
    except Exception:  # noqa: BLE001 - a failed or partial list is unknown, never "nothing running"
        raise LookupError("provider_inventory_unknown") from None
    return state, rows


def _plan(state: HostState, rows: Sequence[Instance], *, now: float,
          terminal: frozenset[str]) -> list[tuple[Instance, dict[str, Any]]]:
    return [(row, decide(row, state, now=now)) for row in rows if row.status not in terminal]


def run_pass(*, store: ObjectStore, inventory: Callable[[], Sequence[Instance]],
             destroy: Callable[[int], Mapping[str, Any]], clock: Callable[[], float] = time.time,
             sleep: Callable[[float], None] = time.sleep, terminal: frozenset[str] | None = None,
             verify_attempts: int = VERIFY_ATTEMPTS, verify_delay_seconds: float = VERIFY_DELAY_SECONDS) -> dict[str, Any]:
    """One pass. Never raises for a provider or store problem: the report says ``unknown``."""

    terminal = _terminal_statuses() if terminal is None else terminal
    now = clock()
    report: dict[str, Any] = {"schema_version": REPORT_SCHEMA_VERSION, "observed_at_epoch": now,
                              "observed_at": datetime.fromtimestamp(now, timezone.utc).isoformat(),
                              "status": "ok", "blockers": [], "instances": [], "destroyed": [], "failed": [],
                              "budget_admission": "not_required_spend_effect_stop"}
    try:
        state, rows = _observe(store, inventory)
    except LookupError as error:
        report.update(status="unknown", blockers=[str(error)])
        return report
    report.update(heartbeat_epoch=state.heartbeat_epoch, heartbeat_age_seconds=now - state.heartbeat_epoch,
                  heartbeat_stale=now - state.heartbeat_epoch > HEARTBEAT_STALE_SECONDS,
                  published_deadline_count=len(state.entries), deadlines_complete=state.deadlines_complete,
                  longest_deadline_seconds=state.longest_deadline_seconds)
    plan = _plan(state, rows, now=now, terminal=terminal)
    candidates = {row.instance_id for row, decision in plan if decision["action"] == "destroy"}
    if candidates:
        # The object and the list are read again at the moment of acting; only a machine both
        # readings condemn is destroyed.
        now = clock()
        try:
            fresh_state, fresh_rows = _observe(store, inventory)
        except LookupError as error:
            report.update(status="unknown", blockers=[f"reread:{error}"])
            report["instances"] = [_row(row, decision, acted=False) for row, decision in plan]
            return report
        plan = _plan(fresh_state, fresh_rows, now=now, terminal=terminal)
        candidates &= {row.instance_id for row, decision in plan if decision["action"] == "destroy"}
    deletes: dict[int, Mapping[str, Any]] = {}
    for row, decision in plan:
        if row.instance_id in candidates:
            try:
                deletes[row.instance_id] = dict(destroy(row.instance_id))
            except Exception as error:  # noqa: BLE001 - recorded as failed; the error may carry a key
                deletes[row.instance_id] = {"status": "stop_failed", "error_type": type(error).__name__}
    verified = _verify(set(deletes), inventory, terminal=terminal, attempts=verify_attempts,
                       delay=verify_delay_seconds, sleep=sleep) if deletes else {}
    for row, decision in plan:
        entry = _row(row, decision, acted=row.instance_id in deletes)
        if row.instance_id in deletes:
            result = deletes[row.instance_id]
            proof = verified.get(row.instance_id)
            entry["delete"] = {key: result[key] for key in ("status", "http", "already_gone", "error_type")
                               if key in result}
            if result.get("status") == "stopped" and proof is not None:
                entry.update(outcome="destroyed", proof=proof)
                report["destroyed"].append(row.instance_id)
            else:
                entry.update(outcome="failed", proof=proof or "absence_not_verified")
                report["failed"].append(row.instance_id)
        report["instances"].append(entry)
    if report["failed"]:
        report["status"] = "failed"
    elif any(entry["outcome"] == "unknown" for entry in report["instances"]):
        report["status"] = "unknown"
        report["blockers"].append("instance_reapability_unknown")
    return report


def _row(row: Instance, decision: Mapping[str, Any], *, acted: bool) -> dict[str, Any]:
    """A report row. A foreign label is not copied: it is someone else's name."""

    outcome = "unknown" if decision["action"] == "unknown" else "left"
    if decision["action"] == "destroy" and not acted:
        outcome, decision = "left", {**decision, "reason": "not_confirmed_on_reread:" + str(decision["reason"])}
    return {"instance_id": row.instance_id, "label": row.label if LABEL_PATTERN.fullmatch(row.label) else None,
            "status": row.status, "outcome": outcome,
            **{key: value for key, value in decision.items() if key != "action"}}


def _verify(ids: set[int], inventory: Callable[[], Sequence[Instance]], *, terminal: frozenset[str], attempts: int,
            delay: float, sleep: Callable[[float], None]) -> dict[int, str]:
    """``{id: proof}`` for each deleted machine the list shows gone or terminal."""

    proven: dict[int, str] = {}
    for attempt in range(max(1, attempts)):
        if attempt:
            sleep(delay)
        try:
            rows = {row.instance_id: row for row in inventory()}
        except Exception:  # noqa: BLE001 - an unreadable list proves nothing
            continue
        for instance_id in ids - set(proven):
            row = rows.get(instance_id)
            if row is None:
                proven[instance_id] = "absent_from_provider_list"
            elif row.status in terminal:
                proven[instance_id] = f"provider_status_{row.status}"
        if set(proven) == ids:
            break
    return proven


def exit_code(report: Mapping[str, Any]) -> int:
    return 0 if report.get("status") == "ok" else 1


class GcsObjectStore:
    """The reaper bucket through google-cloud-storage, imported only when used."""

    def __init__(self, bucket: str, *, credentials: Any = None, project: str | None = None) -> None:
        from google.cloud import storage

        self._bucket = storage.Client(project=project, credentials=credentials).bucket(bucket)

    def read(self, name: str) -> bytes | None:
        from google.api_core.exceptions import NotFound

        blob = self._bucket.blob(name)
        try:
            blob.reload()
            if blob.size is not None and int(blob.size) > MAX_STATE_BYTES:
                raise ValueError("object_too_large")
            return blob.download_as_bytes()
        except NotFound:
            return None

    def write(self, name: str, data: bytes) -> None:
        self._bucket.blob(name).upload_from_string(bytes(data), content_type="application/json")


def report_object_name(report: Mapping[str, Any]) -> str:
    stamp = datetime.fromtimestamp(float(report["observed_at_epoch"]), timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{REPORT_PREFIX}{stamp}.json"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="One off-host GPU reaper pass.")
    parser.add_argument("--bucket", default=os.getenv(BUCKET_ENV, ""))
    args = parser.parse_args(argv)
    if not args.bucket:
        print(json.dumps({"schema_version": REPORT_SCHEMA_VERSION, "status": "unknown",
                          "blockers": ["reaper_bucket_not_configured"]}, sort_keys=True))
        return 1
    from .gpu_render_providers import get_render_provider

    provider = get_render_provider("vast")
    key = provider._key()  # noqa: SLF001 - the provider's own file-based key lookup
    store = GcsObjectStore(args.bucket)

    def inventory() -> list[Instance]:
        if not key:
            raise ValueError("vast_api_key_missing")
        return read_vast_inventory(key)

    report = run_pass(store=store, inventory=inventory, destroy=lambda instance_id: provider.stop(str(instance_id)))
    if not key:
        report["blockers"].append("vast_api_key_missing")
    text = json.dumps(report, sort_keys=True)
    print(text)
    try:
        store.write(report_object_name(report), text.encode("utf-8"))
    except Exception as error:  # noqa: BLE001 - the printed report is in the job's log
        print(json.dumps({"report_write_failed": type(error).__name__}), file=sys.stderr)
        return 1
    return exit_code(report)


__all__ = [
    "BUCKET_ENV", "DEADLINE_GRACE_SECONDS", "HEARTBEAT_STALE_SECONDS", "LABEL_PATTERN", "REPORT_SCHEMA_VERSION",
    "STATE_OBJECT", "STATE_SCHEMA_VERSION", "Entry", "GcsObjectStore", "HostState", "Instance", "decide",
    "exit_code", "parse_host_state", "read_vast_inventory", "run_pass",
]


if __name__ == "__main__":
    raise SystemExit(main())
