"""Record a failed systemd unit and tell the operator once, then hourly.

``blueprint-unit-failure@.service`` runs this when a unit that names it in
``OnFailure=`` enters the failed state (a crash, a start timeout, an OOM kill, a
non-zero exit). Before it, such a failure was a journal line only.

Each run rewrites one small record, ``<record root>/<unit>.json``, which the
operator door can read, and posts through the capacity controller's own alert
route and posting code (``control_plane_capacity_controller.post_alert`` on
``BLUEPRINT_OPERATOR_ALERT_WEBHOOK_URL``). A unit that fails every minute posts
once and then once an hour, the controller's own repeat interval. It reads the
failure from the ``MONITOR_*`` variables systemd (251 and later) passes to an
``OnFailure=`` service; what systemd does not say stays ``null``.

It changes no unit, queue or job, and it never exits non-zero for a failed
post: the record keeps the error and the next failure retries it.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .core.common import write_json

SCHEMA_VERSION = "blueprint.unit_failure_record.v1"
RECORD_ROOT_ENV = "BLUEPRINT_UNIT_FAILURE_RECORD_ROOT"
DEFAULT_RECORD_ROOT = Path("/var/lib/blueprint/pipeline-control-plane/unit-failures")
ALERT_CODE = "unit_failed"
RUNBOOK = "docs/runbooks/beta-operations.md#1a-a-unit-failed"
MAX_RECORD_BYTES = 64 * 1024
# A unit this repository installs; the name becomes a file name and part of the alert.
_UNIT = re.compile(r"blueprint-[A-Za-z0-9_.:@-]{1,200}\.(?:service|timer|path)\Z")
_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,63}\Z")


def _word(value: Any) -> str | None:
    """A short systemd word (a result, an exit status, an invocation id), or None."""

    text = str(value or "").strip()
    return text if _WORD.match(text) else None


def observed_failure(unit: str, environ: Mapping[str, str]) -> dict[str, str | None]:
    """What systemd said about the failure; ``None`` where it said nothing.

    The variables describe the unit named by ``MONITOR_UNIT``. When that is not
    the unit this run was started for, nothing is taken from them.
    """

    named = str(environ.get("MONITOR_UNIT") or "").strip()
    trusted = named == unit
    return {
        "result": _word(environ.get("MONITOR_SERVICE_RESULT")) if trusted else None,
        "exit_code": _word(environ.get("MONITOR_EXIT_CODE")) if trusted else None,
        "exit_status": _word(environ.get("MONITOR_EXIT_STATUS")) if trusted else None,
        "invocation_id": _word(environ.get("MONITOR_INVOCATION_ID")) if trusted else None,
    }


def _previous(path: Path) -> dict[str, Any]:
    try:
        if path.is_symlink() or path.stat().st_size > MAX_RECORD_BYTES:
            return {}
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) and value.get("schema_version") == SCHEMA_VERSION else {}


def _number(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def alert_report(record: Mapping[str, Any]) -> dict[str, Any]:
    """The one-alert report ``post_alert`` formats: it names an alert by its code."""

    detail = str(record.get("result") or "unknown")
    if detail == "exit-code" and record.get("exit_status"):
        detail += f":{record['exit_status']}"
    return {
        "level": "critical",
        "mounts": [],
        "alerts": [{
            "code": f"{ALERT_CODE}:{record['unit']}:{detail}",
            "severity": "page",
            "provider": "systemd",
            "unit": record["unit"],
            "result": record.get("result"),
            "exit_status": record.get("exit_status"),
            "failure_count": record.get("failure_count"),
            "unit_runbook": RUNBOOK,
        }],
    }


def record_unit_failure(
    unit: str,
    *,
    record_root: Path,
    webhook_url: str,
    environ: Mapping[str, str] | None = None,
    now: float | None = None,
    poster: Callable[[str, Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Write the unit's failure record, post when one is due, and return the record."""

    if not _UNIT.match(unit):
        raise ValueError("unit_failure_unit_name_invalid")
    observed = time.time() if now is None else float(now)
    path = Path(record_root) / f"{unit}.json"
    previous = _previous(path)
    count = previous.get("failure_count")
    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "unit": unit,
        **observed_failure(unit, os.environ if environ is None else environ),
        "failure_count": (count if isinstance(count, int) and not isinstance(count, bool) and count > 0 else 0) + 1,
        "first_failure_epoch": _number(previous.get("first_failure_epoch")) or observed,
        "last_failure_epoch": observed,
        "last_failure_at": datetime.fromtimestamp(observed, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "last_alert_epoch": _number(previous.get("last_alert_epoch")),
        "alert_posted": False,
        "alert_error": None,
        "runbook": RUNBOOK,
    }
    # The failure is on disk before any network call: a post that hangs or dies loses nothing.
    write_json(path, record)
    if not webhook_url:
        # The capacity controller already warns about the missing route on every tick.
        record["alert_error"] = "operator_alert_route_unconfigured"
        write_json(path, record)
        return record
    # Imported here: the controller is the one webhook client, and its repeat interval the one dedupe.
    from . import control_plane_capacity_controller as capacity

    last = record["last_alert_epoch"]
    due = (
        last is None
        or observed - last >= capacity.ALERT_REPEAT_SECONDS
        or bool(previous.get("alert_error"))  # the last delivery failed: retry it
    )
    if due:
        try:
            (poster or capacity.post_alert)(webhook_url, alert_report(record))
            record["alert_posted"] = True
            record["last_alert_epoch"] = observed
        except Exception as exc:  # noqa: BLE001 - a failed post must not fail the notifier; it may carry the URL
            record["alert_error"] = type(exc).__name__
        write_json(path, record)
    return record


def main(argv: Sequence[str] | None = None) -> int:
    from .control_plane_capacity_controller import WEBHOOK_URL_ENV

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--unit", required=True, help="the failed unit (the template instance, %%i)")
    parser.add_argument("--record-root", default=os.getenv(RECORD_ROOT_ENV) or str(DEFAULT_RECORD_ROOT))
    args = parser.parse_args(argv)
    record = record_unit_failure(
        args.unit, record_root=Path(args.record_root), webhook_url=str(os.getenv(WEBHOOK_URL_ENV) or "").strip()
    )
    print(json.dumps(record, sort_keys=True))
    return 0


__all__ = [
    "ALERT_CODE",
    "DEFAULT_RECORD_ROOT",
    "RECORD_ROOT_ENV",
    "SCHEMA_VERSION",
    "alert_report",
    "main",
    "observed_failure",
    "record_unit_failure",
]


if __name__ == "__main__":  # pragma: no cover - exercised through the unit
    sys.exit(main())
