"""Read-only, privacy-safe progress projection of the existing screening ledger.

The snapshot carries no launch, funding, publication or recovery authority. It
does not copy the outbox or alter campaign state. Only this summary is written.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import os
import stat
import tempfile
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest

SCHEMA_VERSION = "blueprint.beta_policy_screening_worker_status.v1"
STATUS_FILENAME = "worker_status.json"
_MAX_LEDGER_BYTES = 4 * 1024 * 1024
_MAX_STATUS_BYTES = 32 * 1024
_STATES = frozenset(("pending", "preparing", "launching", "running", "reconciliation_required",
                     "completed", "blocked", "unavailable", "deferred", "stopped", "cancelled"))
_FAILURES = frozenset(("auth", "source", "ack", "unavailable", "unknown"))


def _read(path: Path, limit: int) -> dict[str, Any]:
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
            raise ValueError("screening_status_input_unsafe")
        raw = os.read(fd, limit + 1)
    finally:
        os.close(fd)
    if len(raw) > limit:
        raise ValueError("screening_status_input_too_large")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("screening_status_input_invalid")
    return value


def _timestamp(value: Any) -> float | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed.timestamp() if parsed.tzinfo is not None else None
    except (ValueError, OverflowError):
        return None


def _number(value: Any) -> float | int | None:
    return value if type(value) in (int, float) and math.isfinite(value) and value >= 0 else None


def _failure(delivery: Any) -> dict[str, Any] | None:
    if not isinstance(delivery, Mapping):
        return None
    category = delivery.get("last_failure_category")
    if not isinstance(category, str) or category not in _FAILURES:
        return None
    return {"category": category, "owner": "pipeline_transport",
            "next_action": {"auth": "check_existing_sync_credential", "source": "reconcile_current_task_source",
                            "ack": "reconcile_exact_publication_receipt", "unavailable": "retry_existing_outbox",
                            "unknown": "inspect_retained_transport_diagnostic"}[category],
            "next_retry_at_epoch": _number(delivery.get("next_retry_at_epoch"))}


def build_worker_status(root: str | Path, *, publication_before: Any = None,
                        publication_after: Any = None, execution: Any = None,
                        now: float | None = None) -> dict[str, Any]:
    observed = time.time() if now is None else now
    directory = Path(root)
    result: dict[str, Any] = {"schema_version": SCHEMA_VERSION, "observed_at_epoch": observed,
                              "status": "unknown", "coverage": "unavailable", "performs_live_actions": False}
    if not directory.is_absolute() or any(path.is_symlink() for path in (directory, *directory.parents)):
        return {**result, "reason": "screening_status_root_unsafe"}
    previous = {}
    try:
        previous = _read(directory / STATUS_FILENAME, _MAX_STATUS_BYTES)
        if previous.get("schema_version") != SCHEMA_VERSION:
            previous = {}
    except (OSError, ValueError):
        pass
    try:
        ledger = _read(directory / "screening_ledger.v1.json", _MAX_LEDGER_BYTES)
    except FileNotFoundError:
        try:
            with os.scandir(directory / "attempts") as entries:
                if next(entries, None) is not None:
                    return {**result, "reason": "screening_ledger_missing_after_attempt"}
        except OSError:
            return {**result, "reason": "screening_attempt_inventory_unavailable"}
        ledger = {"schema_version": "blueprint.beta_policy_screening_ledger.v1", "campaigns": {}}
        ledger["ledger_digest"] = canonical_digest(ledger, digest_field="ledger_digest")
    except (OSError, ValueError):
        return {**result, "reason": "screening_ledger_unavailable"}
    if (ledger.get("schema_version") != "blueprint.beta_policy_screening_ledger.v1"
            or not isinstance(ledger.get("campaigns"), dict)
            or ledger.get("ledger_digest") != canonical_digest(ledger, digest_field="ledger_digest")):
        return {**result, "reason": "screening_ledger_invalid"}
    counts = {state: 0 for state in sorted(_STATES)}
    publication_pending = 0
    eligible_since = []
    updates = []
    categories: dict[str, int] = {}
    try:
        for campaign in ledger["campaigns"].values():
            if not isinstance(campaign, dict) or not isinstance(campaign.get("candidates"), dict):
                raise ValueError
            publication_pending += int(campaign["revision"] != campaign["published_revision"])
            updated = _timestamp(campaign.get("updated_at"))
            if updated is not None:
                updates.append(updated)
            for candidate in campaign["candidates"].values():
                state = candidate["status"]
                if state not in counts:
                    raise ValueError
                counts[state] += 1
                if state in {"pending", "preparing", "launching", "running", "reconciliation_required"}:
                    created = _timestamp(campaign.get("created_at"))
                    if created is not None:
                        eligible_since.append(created)
                for code in candidate.get("blockers", []):
                    if not isinstance(code, str):
                        continue
                    category = ("intentional_hold" if "hold" in code else "authority" if any(part in code for part in ("consent", "grant", "revoked", "expired"))
                                else "budget" if any(part in code for part in ("budget", "cost", "funding"))
                                else "capacity" if "capacity" in code else "capability_preparation" if state == "unavailable"
                                else "reconciliation" if state == "reconciliation_required" else "preparation")
                    categories[category] = categories.get(category, 0) + 1
    except (KeyError, TypeError, ValueError):
        return {**result, "reason": "screening_campaign_state_invalid"}
    acknowledgment_count = sum((_number(value.get("acknowledged")) or 0)
                                for value in (publication_before, publication_after) if isinstance(value, Mapping))
    failure = _failure(publication_after) or _failure(publication_before)
    previous_failure = previous.get("last_publication_failure")
    if failure is not None:
        failure["observed_at_epoch"] = observed
    elif isinstance(previous_failure, Mapping):
        failure = _failure({"last_failure_category": previous_failure.get("category"),
                            "next_retry_at_epoch": previous_failure.get("next_retry_at_epoch")})
        if failure is not None:
            failure["observed_at_epoch"] = _number(previous_failure.get("observed_at_epoch"))
    active = sum(counts[state] for state in ("launching", "running", "reconciliation_required"))
    waiting = counts["pending"] + counts["preparing"]
    status = "working" if active else "waiting" if waiting or publication_pending else "preparation_needed" if counts["unavailable"] or counts["blocked"] or counts["deferred"] else "idle"
    return {**result, "status": status, "coverage": "current_ledger", "reason": None,
            "ledger_digest": ledger["ledger_digest"], "campaign_count": len(ledger["campaigns"]),
            "candidate_counts": counts, "publication_pending": publication_pending,
            "oldest_eligible_wait_seconds": max(0, observed - min(eligible_since)) if eligible_since else None,
            "last_ledger_progress_at_epoch": max(updates) if updates else None,
            "last_successful_publication_at_epoch": observed if acknowledgment_count else _number(previous.get("last_successful_publication_at_epoch")),
            "last_publication_failure": failure, "blocker_categories": categories,
            "worker_busy": isinstance(execution, Mapping) and execution.get("status") == "worker_busy"}


def write_worker_status(root: str | Path, publication_before: Any = None,
                        publication_after: Any = None, execution: Any = None) -> dict[str, Any]:
    result = build_worker_status(root, publication_before=publication_before,
                                 publication_after=publication_after, execution=execution)
    directory = Path(root)
    if result.get("reason") == "screening_status_root_unsafe" or not directory.is_dir():
        return result
    path = directory / STATUS_FILENAME
    if path.is_symlink():
        return {**result, "status": "unknown", "reason": "screening_status_path_unsafe"}
    try:
        fd, temporary = tempfile.mkstemp(prefix=".worker-status-", dir=directory)
    except OSError:
        return {**result, "status": "unknown", "reason": "screening_status_unwritten"}
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(result, stream, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        descriptor = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except (OSError, ValueError):
        return {**result, "status": "unknown", "reason": "screening_status_unwritten"}
    finally:
        try:
            Path(temporary).unlink(missing_ok=True)
        except OSError:
            pass
    return result
