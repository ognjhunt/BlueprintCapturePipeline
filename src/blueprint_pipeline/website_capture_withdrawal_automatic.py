"""Consent-withdrawal deletion with no operator: one timer pass plans, and applies under a standing approval.

Beta journey step: site withdrawal. The intake writes a tombstone when a site withdraws
(``website_capture_withdrawal.acknowledge_withdrawal``) and never deletes. Before this module
nothing on the Pipeline host was deleted unless the owner ran the deletion tool by hand, and
nothing said a withdrawal was waiting. ``blueprint-website-withdrawal-deletion.timer`` starts
``run`` hourly, in its own unit with its own limits; the intake starts nothing.

Each pass, with no operator:

1. It finds every acknowledged tombstone (``<site>/website_withdrawal/tombstone.json`` under
   the site roots) that has no sealed deletion receipt with every covered copy deleted.
2. It does nothing while a paid stage is live on the host, by the definition ``door status``
   and the beta deploy policy use (AGENTS.md): a held ``paid_launch_locks`` entry (a lock in
   ``provider-locks/`` the kernel lists as held, read from ``/proc/locks`` by the stall alarm's
   own reader, taking no lock) or beta-screening candidates ``launching`` or ``running`` in the
   screening worker's status. A check that cannot look counts as busy.
3. It takes at most one site, the one whose last attempt is oldest, and records its state
   beside the tombstone (``automatic_deletion_state.json``, the latest plan beside it):
   ``planning`` or ``applying`` before it starts (a run killed there is a failure the next
   pass records), then ``planned`` (with what it waits on), ``unresolved`` (with the reasons),
   ``legal_hold``, ``failed`` (with the error and the next attempt), or ``applied``.
   A failure backs off (1 h, 4 h, then 24 h) and is retried; nothing becomes terminal.
4. It applies only under the owner's standing approval (``APPROVAL_PATH``, validated strictly
   by ``load_approval``), and only a plan in the approved policy class: an acknowledged
   withdrawal, no legal hold, nothing unresolved, the site's money settled (no screening
   attempt deferred or unsettled, no rigid probe job waiting on its money) and the approval's
   cool-off elapsed since the acknowledgement. The apply is the deletion tool's own
   (``apply_withdrawal_deletion``) with the plan's exact digest, so its re-plan and every
   refusal still run, and its sealed receipt names the approval (``authorization``).
   Without a valid approval it only plans. Money records, ledgers, the audit journal and
   other sites are the deletion tool's to keep, and it keeps them.
5. It alerts through the capacity controller's posting code
   (``control_plane_capacity_controller.post_alert``, the same webhook) when it cannot finish:
   a site blocked (unresolved or money unsettled) longer than ``BLOCKED_ALERT_SECONDS``, a site
   failing ``FAILING_ALERT_COUNT`` times in a row, a withdrawal older than
   ``OVERDUE_ALERT_SECONDS`` without a receipt (not under legal hold), an approval file that
   does not validate or has expired, or site roots it could not list. A new alert posts at once;
   the same set again after ``ALERT_REPEAT_SECONDS``. An absent approval alone is not an alert:
   ``summary.json`` counts the withdrawals waiting only on it, and the capacity controller
   copies that count into the summary ``door status`` reads.

The nine Pipeline roots the deletion reports but does not scan are recorded per site from the
plan (``uncovered_pipeline_roots``) and named in each receipt's ``remaining``.
"""
from __future__ import annotations

import argparse
import errno
import fcntl
import hashlib
import json
import os
import re
import stat
import sys
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .core.common import write_json

UNIT = "blueprint-website-withdrawal-deletion.service"
# The owner installs this file once, by hand; no deploy step writes it (deploy/website-withdrawal-deletion/).
APPROVAL_PATH = Path("/etc/blueprint/website-withdrawal-deletion-approval.json")
APPROVAL_SCHEMA = "website_withdrawal_deletion_standing_approval.v1"
POLICY_CLASS = "website_withdrawal_deletion.acknowledged_unheld_resolved_settled_cooled.v1"
_APPROVAL_REQUIRED = frozenset({"schema_version", "approved_by", "approved_at", "policy_class", "cool_off_hours"})
# ``garbage_collection_policy_class``: the owner's separate, explicit approval for deleting content blobs
# nobody refers to (website_withdrawal_content_blobs.plan_gc); without it that plan only plans.
_APPROVAL_OPTIONAL = frozenset({"expires_at", "note", "garbage_collection_policy_class"})
MAX_APPROVAL_BYTES = 16 * 1024
MAX_COOL_OFF_HOURS = 720
CLOCK_SKEW_SECONDS = 300

STATE_SCHEMA = "website_withdrawal_automatic_deletion_state.v1"
STATE_FILENAME = "automatic_deletion_state.json"
PLAN_FILENAME = "automatic_deletion_plan.json"
SUMMARY_SCHEMA = "website_withdrawal_automatic_deletion_summary.v1"
SUMMARY_FILENAME = "summary.json"
ALERT_STATE_FILENAME = "alerts.json"
LOCK_FILENAME = "run.lock"
SITE_INDEX_DIRNAME = "site-index"
DEFAULT_REPORT_ROOT = Path("/var/lib/blueprint/pipeline-control-plane/website-withdrawal-deletion")
DEFAULT_SITE_ROOTS = (Path("/var/lib/blueprint/pubsub-handoffs"),)
DEFAULT_PROVIDER_LOCK_DIR = Path("/var/lib/blueprint/pipeline-control-plane/provider-locks")
DEFAULT_SCREENING_STATUS = Path("/var/lib/blueprint/pipeline-control-plane/beta-policy-screening/worker_status.json")
SITE_ROOTS_ENV = "BLUEPRINT_WEBSITE_WITHDRAWAL_SITE_ROOTS"
REPORT_ROOT_ENV = "BLUEPRINT_WEBSITE_WITHDRAWAL_DELETION_REPORT_ROOT"
SOURCE_BUCKET_ENV = "BLUEPRINT_WEBSITE_WITHDRAWAL_SOURCE_BUCKET"

# A failed attempt waits 1 h, then 4 h, then 24 h between tries, for as long as it fails.
BACKOFF_SECONDS = (3600, 4 * 3600, 24 * 3600)
FAILING_ALERT_COUNT = 3
# A plan that waits re-reads every run's records this often; a clean plan waiting only on the
# approval or its cool-off is taken as soon as both allow.
REPLAN_SECONDS = 6 * 3600
BLOCKED_ALERT_SECONDS = 72 * 3600
# The privacy policy promises completion within 30 days; this leaves nine to act.
OVERDUE_ALERT_SECONDS = 21 * 86400
ALERT_REPEAT_SECONDS = 24 * 3600
_APPROVAL_WAITS = frozenset({"standing_approval_absent", "standing_approval_invalid", "standing_approval_expired"})
_CODE = re.compile(r"[a-z][a-z0-9_:.-]{0,159}")
_SITE = re.compile(r"site-[A-Za-z0-9][A-Za-z0-9._-]{0,119}")
_MAX_SUMMARY_SITES = 50


def _time(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed.astimezone(timezone.utc) if parsed.tzinfo is not None else None


def _invalid(reason: str, digest: str | None = None) -> dict[str, Any]:
    return {"status": "invalid", "reason": reason, "digest": digest, "approval": None}


def load_approval(path: Path = APPROVAL_PATH, *, now: float, owner_uid: int = 0) -> dict[str, Any]:
    """The standing approval at ``path``: ``absent``, ``invalid`` (with a reason), ``expired`` or ``valid``.

    Strict: a regular file (never a link) owned by ``owner_uid`` (root on the host) that no group
    or other user may write, at most 16 KiB of one JSON object with exactly the known fields, the
    current schema, the exact policy class, an approval time with a zone that is not in the
    future, an optional expiry after it, and a whole number of cool-off hours from 0 to 720.
    """

    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC)
    except FileNotFoundError:
        return {"status": "absent", "reason": None, "digest": None, "approval": None}
    except OSError as error:
        return _invalid("approval_symlink" if error.errno == errno.ELOOP else "approval_unreadable")
    with os.fdopen(descriptor, "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode):
            return _invalid("approval_not_a_regular_file")
        if info.st_uid != owner_uid:
            return _invalid("approval_owner_unexpected")
        if info.st_mode & 0o022:
            return _invalid("approval_writable_by_others")
        raw = stream.read(MAX_APPROVAL_BYTES + 1)
    digest = "sha256:" + hashlib.sha256(raw).hexdigest()
    if len(raw) > MAX_APPROVAL_BYTES:
        return _invalid("approval_too_large", digest)
    try:
        value = json.loads(raw)
    except ValueError:
        return _invalid("approval_not_json", digest)
    if not isinstance(value, dict):
        return _invalid("approval_not_an_object", digest)
    if set(value) - _APPROVAL_REQUIRED - _APPROVAL_OPTIONAL:
        return _invalid("approval_unknown_field", digest)
    if _APPROVAL_REQUIRED - set(value):
        return _invalid("approval_field_missing", digest)
    if value["schema_version"] != APPROVAL_SCHEMA:
        return _invalid("approval_schema_unknown", digest)
    approved_by = value["approved_by"]
    if (not isinstance(approved_by, str) or not 0 < len(approved_by) <= 200 or approved_by.strip() != approved_by
            or not approved_by.isprintable()):
        return _invalid("approval_approved_by_invalid", digest)
    approved_at = _time(value["approved_at"])
    if approved_at is None:
        return _invalid("approval_approved_at_invalid", digest)
    if approved_at.timestamp() > now + CLOCK_SKEW_SECONDS:
        return _invalid("approval_approved_at_in_future", digest)
    expires_at = None
    if value.get("expires_at") is not None:
        expires_at = _time(value["expires_at"])
        if expires_at is None:
            return _invalid("approval_expires_at_invalid", digest)
        if expires_at <= approved_at:
            return _invalid("approval_expires_before_approval", digest)
    if value["policy_class"] != POLICY_CLASS:
        return _invalid("approval_policy_class_mismatch", digest)
    hours = value["cool_off_hours"]
    if type(hours) is not int or not 0 <= hours <= MAX_COOL_OFF_HOURS:
        return _invalid("approval_cool_off_hours_invalid", digest)
    if "note" in value and (not isinstance(value["note"], str) or len(value["note"]) > 1000):
        return _invalid("approval_note_invalid", digest)
    from .website_withdrawal_content_blobs import GC_POLICY_CLASS
    gc_class = value.get("garbage_collection_policy_class")
    if gc_class is not None and gc_class != GC_POLICY_CLASS:
        return _invalid("approval_garbage_collection_policy_class_mismatch", digest)
    approval = {"approved_by": approved_by, "approved_at": approved_at.isoformat(),
                "expires_at": expires_at.isoformat() if expires_at else None,
                "policy_class": POLICY_CLASS, "cool_off_hours": hours,
                "garbage_collection_approved": gc_class == GC_POLICY_CLASS}
    if expires_at is not None and now >= expires_at.timestamp():
        return {"status": "expired", "reason": "approval_expired", "digest": digest, "approval": approval}
    return {"status": "valid", "reason": None, "digest": digest, "approval": approval}


def paid_stage_reasons(*, provider_lock_dir: Path = DEFAULT_PROVIDER_LOCK_DIR,
                       screening_status: Path = DEFAULT_SCREENING_STATUS,
                       lock_reader: Callable[[], Mapping[tuple[int, int, int], Sequence[int]]] | None = None
                       ) -> list[str]:
    """Why a paid stage counts as live now, by ``door status``'s definition; empty when the host is quiet.

    A held lock in ``provider_lock_dir`` is ``paid_launch_locks`` held (the deploy tool refuses
    or reports such a slot as a paid run in flight); ``launching`` or ``running`` screening
    candidates are the beta deploy policy's third signal. Nothing is locked or changed. A check
    that cannot look returns a reason too: unknown is never quiet.
    """

    reasons: list[str] = []
    try:
        locks = sorted(path for path in Path(provider_lock_dir).glob("*.lock") if not path.is_symlink())
    except OSError:
        locks, reasons = [], ["paid_launch_locks_unreadable"]
    if locks:
        try:
            if lock_reader is None:
                from .control_plane_stall_alarm import read_lock_holders as lock_reader
            holders = lock_reader()
        except Exception:  # noqa: BLE001 - an unreadable lock table is unknown, which is busy
            reasons.append("paid_launch_locks_unknown")
        else:
            for path in locks:
                try:
                    info = os.stat(path)
                except FileNotFoundError:
                    continue
                except OSError:
                    reasons.append("paid_launch_locks_unknown")
                    continue
                if (os.major(info.st_dev), os.minor(info.st_dev), info.st_ino) in holders:
                    reasons.append("paid_launch_lock_held:" + path.name[:80])
    try:
        raw = Path(screening_status).read_bytes()[:64 * 1024]
    except FileNotFoundError:
        raw = None
    except OSError:
        raw, reasons = None, [*reasons, "screening_status_unknown"]
    if raw is not None:
        try:
            counts = json.loads(raw).get("candidate_counts") or {}
            live = sum(int(counts.get(state) or 0) for state in ("launching", "running"))
        except (ValueError, AttributeError, TypeError):
            reasons.append("screening_status_unknown")
        else:
            if live > 0:
                reasons.append("screening_candidates_live")
    return reasons


def _read_json(path: Path, max_bytes: int = 1024 * 1024) -> dict[str, Any] | None:
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC)
    except OSError:
        return None
    with os.fdopen(descriptor, "rb") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            return None
        raw = stream.read(max_bytes + 1)
    if len(raw) > max_bytes:
        return None
    try:
        value = json.loads(raw)
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


def discover_withdrawals(site_roots: Sequence[Path]) -> tuple[list[Path], list[dict[str, str]]]:
    """Every site directory with a tombstone under ``<root>/scenes/`` or ``<root>/*/scenes/``, and what could not be listed.

    A root, bucket or ``scenes`` directory that cannot be listed (or a missing root) is reported, never read as empty.
    """

    sites: list[Path] = []
    problems: list[dict[str, str]] = []

    def children(directory: Path) -> list[os.DirEntry[str]]:
        try:
            with os.scandir(directory) as entries:
                return sorted(entries, key=lambda entry: entry.name)
        except FileNotFoundError:
            return []
        except OSError as error:
            problems.append({"path": str(directory), "reason": "unlistable:" + type(error).__name__})
            return []

    for root in map(Path, site_roots):
        if not root.is_dir() or root.is_symlink():
            problems.append({"path": str(root), "reason": "site_root_missing"})
            continue
        bases = [root, *(Path(entry.path) for entry in children(root)
                         if entry.name != "scenes" and entry.is_dir(follow_symlinks=False))]
        for base in bases:
            scenes = base / "scenes"
            if scenes.is_symlink() or not scenes.is_dir():
                continue
            for entry in children(scenes):
                if not (_SITE.fullmatch(entry.name) and entry.is_dir(follow_symlinks=False)):
                    continue
                tombstone = Path(entry.path) / "website_withdrawal" / "tombstone.json"
                if tombstone.is_file() and not tombstone.is_symlink():
                    sites.append(Path(entry.path))
    return sites, problems


def _sealed_receipt(site: Path, tombstone_digest: str | None) -> dict[str, Any] | None:
    """A sealed receipt of this tombstone whose ``deletion_status`` is finished, or None.

    Finished is ``complete`` or ``complete_provider_retention_remains`` (named copies stay at a provider
    that offers no delete); the status is returned as it is, never shortened to "complete".

    A receipt from before ``deletion_status`` existed stays readable but is not complete: its
    roots were never indexed, so the site is planned again.
    """

    from .decision_evidence_contracts import canonical_digest
    from .website_capture_withdrawal_deletion import FINISHED_STATUSES, RECEIPT_SCHEMA

    directory = site / "website_withdrawal" / "deletion_receipts"
    if tombstone_digest is None or directory.is_symlink() or not directory.is_dir():
        return None
    for path in sorted(directory.glob("*.json")):
        value = _read_json(path, max_bytes=16 * 1024 * 1024)
        if (value and value.get("schema_version") == RECEIPT_SCHEMA
                and value.get("tombstone_digest") == tombstone_digest
                and value.get("deletion_status") in FINISHED_STATUSES
                and value.get("digest") == canonical_digest(value, digest_field="digest")):
            return {"digest": value["digest"], "applied_at_iso": value.get("applied_at_iso"),
                    "deletion_status": value["deletion_status"],
                    "provider_retention_remains": sorted({str(row.get("provider"))
                                                          for row in value.get("provider_retention_remains") or []}),
                    "authorization": (value.get("authorization") or {}).get("kind")}
    return None


def _state_path(site: Path) -> Path:
    return site / "website_withdrawal" / STATE_FILENAME


def _write_state(site: Path, state: Mapping[str, Any]) -> None:
    path = _state_path(site)
    if path.is_symlink() or path.parent.is_symlink():
        raise OSError("website_withdrawal_automatic_state_symlink")
    write_json(path, state)


def _site_row(site: Path, now: float) -> dict[str, Any]:
    """What the pass knows about one withdrawn site without planning: tombstone, receipt and state."""

    tombstone_path = site / "website_withdrawal" / "tombstone.json"
    tombstone = _read_json(tombstone_path) or {}
    digest = tombstone.get("digest") if isinstance(tombstone.get("digest"), str) else None
    acknowledged = None
    try:
        # The intake writes the tombstone once, when it acknowledges; the command's own time is earlier.
        acknowledged = os.lstat(tombstone_path).st_mtime
        requested = _time(tombstone.get("consent_revoked_at"))
        if requested is not None:
            acknowledged = max(acknowledged, requested.timestamp())
    except OSError:
        pass
    state = _read_json(_state_path(site))
    if not state or state.get("schema_version") != STATE_SCHEMA or state.get("tombstone_digest") != digest:
        state = None
    elif state.get("status") in {"planning", "applying"}:
        # This pass holds the host lock, so the run that wrote this ended inside the step.
        state = _failed(state, "interrupted_while_" + state["status"], now)
        _write_state(site, state)
    request_id = site.name[5:]
    return {"site": site, "capture_root": site / "captures" / f"walkthrough-{request_id}", "scene_id": site.name,
            "tombstone_digest": digest, "acknowledged_at_epoch": acknowledged,
            "receipt": _sealed_receipt(site, digest), "state": state}


def _failed(state: Mapping[str, Any], error: str, now: float) -> dict[str, Any]:
    count = int(state.get("failure_count") or 0) + 1
    return {**state, "status": "failed", "last_error": error, "failure_count": count,
            "failed_at_epoch": now, "next_attempt_at_epoch": now + BACKOFF_SECONDS[min(count, len(BACKOFF_SECONDS)) - 1]}


def _cool_off_end(row: Mapping[str, Any], approval: Mapping[str, Any]) -> float | None:
    if approval["status"] != "valid" or row["acknowledged_at_epoch"] is None:
        return None
    return float(row["acknowledged_at_epoch"]) + approval["approval"]["cool_off_hours"] * 3600


def _due(row: Mapping[str, Any], approval: Mapping[str, Any], now: float) -> bool:
    state = row["state"]
    if state is None:
        return True
    if state["status"] in {"failed", "incomplete"}:
        return now >= float(state.get("next_attempt_at_epoch") or 0)
    if now >= float(state.get("last_attempt_at_epoch") or 0) + REPLAN_SECONDS:
        return True
    waiting = set(state.get("waiting_on") or ())
    end = _cool_off_end(row, approval)
    # A clean plan that waited only on the approval, its cool-off or a paid stage goes as soon as all allow.
    return (state["status"] == "planned" and bool(waiting) and waiting <= _APPROVAL_WAITS | {"cool_off", "host_busy"}
            and end is not None and now >= end)


def _error_code(error: BaseException) -> str:
    text = str(error)
    return text if _CODE.fullmatch(text) else type(error).__name__


def _money(plan: Mapping[str, Any]) -> tuple[bool, dict[str, Any]]:
    from . import website_capture_withdrawal_screening as screening

    screening_open = screening.incomplete(plan["screening"])
    probes = plan["team_scene"].get("probe_jobs_waiting") or []
    return (not screening_open and not probes), {
        "screening_attempts_deferred": len(plan["screening"].get("deferred") or []),
        "screening_incomplete": screening_open, "rigid_probe_jobs_waiting": len(probes)}


def _handle(row: Mapping[str, Any], *, approval: Mapping[str, Any], approval_path: Path, now: float,
            resolution: Callable[[], Mapping[str, Any]], quiet: Callable[[], list[str]],
            planner: Callable[..., dict[str, Any]], applier: Callable[..., dict[str, Any]]) -> dict[str, Any]:
    """Plan one site, and apply it when the standing approval's class allows; every outcome is recorded."""

    site, previous = row["site"], row["state"] or {}
    state: dict[str, Any] = {
        "schema_version": STATE_SCHEMA, "request_id": site.name[5:], "scene_id": site.name,
        "tombstone_digest": row["tombstone_digest"], "acknowledged_at_epoch": row["acknowledged_at_epoch"],
        "first_observed_at_epoch": previous.get("first_observed_at_epoch", now),
        "failure_count": int(previous.get("failure_count") or 0),
        "blocked_since_epoch": previous.get("blocked_since_epoch"),
        "status": "planning", "attempt_started_at_epoch": now, "last_attempt_at_epoch": now, "unit": UNIT}
    _write_state(site, state)
    try:
        from . import website_capture_withdrawal as withdrawal

        inspected = withdrawal.inspect_withdrawal(capture_root=row["capture_root"])
        if inspected["legal_hold"]:
            state.update(status="legal_hold", waiting_on=["legal_hold"], failure_count=0, blocked_since_epoch=None)
            _write_state(site, state)
            return state
        resolved = dict(resolution())
        plan = planner(capture_root=row["capture_root"], **resolved)
        write_json(site / "website_withdrawal" / PLAN_FILENAME, plan)
        settled, money = _money(plan)
        reasons: dict[str, int] = {}
        for item in plan["unresolved"]:
            reasons[str(item.get("reason"))] = reasons.get(str(item.get("reason")), 0) + 1
        state.update(plan_digest=plan["digest"], planned_at_epoch=now, failure_count=0, last_error=None,
                     unresolved_count=len(plan["unresolved"]), unresolved_reasons=dict(sorted(reasons.items())[:20]),
                     money=money, legal_hold=plan["legal_hold"],
                     uncovered_pipeline_roots=list(plan["uncovered_pipeline_roots"]))
        if plan["legal_hold"]:
            state.update(status="legal_hold", waiting_on=["legal_hold"], blocked_since_epoch=None)
        elif plan["unresolved"] or not settled:
            state.update(status="unresolved" if plan["unresolved"] else "planned",
                         waiting_on=["unresolved_copies"] if plan["unresolved"] else ["money_unsettled"],
                         blocked_since_epoch=state["blocked_since_epoch"] or now)
        else:
            state["blocked_since_epoch"] = None
            end = _cool_off_end(row, approval)
            if approval["status"] != "valid":
                waiting = ["standing_approval_" + approval["status"]]
            elif end is None or now < end:
                waiting = ["cool_off"]
            else:
                # The plan took a while: a paid stage that started meanwhile defers the apply too.
                waiting = ["host_busy"] if quiet() else []
            state.update(cool_off_ends_at_epoch=end)
            if waiting:
                state.update(status="planned", waiting_on=waiting)
            else:
                state.update(status="applying", waiting_on=[], approval_digest=approval["digest"])
                _write_state(site, state)
                authorization = {
                    "kind": "standing_owner_approval", "approval_path": str(approval_path),
                    "approval_digest": approval["digest"], **approval["approval"],
                    "class_checks": {"pipeline_acknowledged": inspected["pipeline_acknowledged"] is True,
                                     "acknowledged_at_epoch": row["acknowledged_at_epoch"],
                                     "cool_off_elapsed_at_epoch": end, "legal_hold": False,
                                     "unresolved_count": 0, "money_settled": True},
                    "unit": UNIT, "authorized_at_epoch": now}
                receipt = applier(capture_root=row["capture_root"], expected_plan_digest=plan["digest"],
                                  authorize_complete_deletion=True, authorization=authorization, **resolved)
                reasons = list(receipt.get("incomplete_reasons") or [])
                state.update(status="applied" if not reasons else "incomplete", receipt_digest=receipt["digest"],
                             applied_at_epoch=now, covered_copies_deleted=receipt.get("covered_copies_deleted"),
                             deletion_status=receipt.get("deletion_status"),
                             waiting_on=sorted({str(row.get("holder")) for row in reasons}),
                             incomplete_reasons=reasons[:20],
                             # A failed provider delete is retried on the next hourly pass; anything else
                             # is planned again after REPLAN_SECONDS.
                             next_attempt_at_epoch=now + (BACKOFF_SECONDS[0] if any(
                                 row.get("holder") == "provider_copies_not_deleted" for row in reasons)
                                 else REPLAN_SECONDS),
                             blocked_since_epoch=(previous.get("blocked_since_epoch") or now) if reasons else None)
    except Exception as error:  # noqa: BLE001 - every failure is recorded and retried; nothing is terminal
        state = _failed(state, _error_code(error), now)
    _write_state(site, state)
    return state


def _alerts(rows: Sequence[Mapping[str, Any]], approval: Mapping[str, Any], problems: Sequence[Mapping[str, str]],
            now: float) -> list[dict[str, Any]]:
    alerts: list[dict[str, Any]] = []
    if approval["status"] in {"invalid", "expired"}:
        alerts.append({"code": "website_withdrawal_approval_invalid", "reason": approval["reason"],
                       "target": "standing_approval"})
    if problems:
        alerts.append({"code": "website_withdrawal_scan_incomplete", "target": "site_roots",
                       "count": len(problems), "reason": problems[0]["reason"]})
    for row in rows:
        if row["receipt"] is not None:
            continue
        state = row["state"] or {}
        held = state.get("status") == "legal_hold"
        if state.get("status") == "failed" and int(state.get("failure_count") or 0) >= FAILING_ALERT_COUNT:
            alerts.append({"code": "website_withdrawal_deletion_failing", "target": row["scene_id"],
                           "reason": state.get("last_error"), "count": state.get("failure_count")})
        blocked = state.get("blocked_since_epoch")
        if isinstance(blocked, (int, float)) and now - blocked >= BLOCKED_ALERT_SECONDS:
            alerts.append({"code": "website_withdrawal_deletion_unresolved", "target": row["scene_id"],
                           "blockers": sorted(state.get("unresolved_reasons") or {}) or list(state.get("waiting_on") or []),
                           "age_seconds": round(now - blocked)})
        acknowledged = row["acknowledged_at_epoch"]
        if not held and isinstance(acknowledged, (int, float)) and now - acknowledged >= OVERDUE_ALERT_SECONDS:
            alerts.append({"code": "website_withdrawal_deletion_overdue", "target": row["scene_id"],
                           "age_seconds": round(now - acknowledged), "status": state.get("status") or "not_planned"})
    for alert in alerts:
        alert.update(severity="page", provider="website_withdrawal")
    return alerts


def _post(report_root: Path, alerts: Sequence[Mapping[str, Any]], *, now: float, webhook_url: str,
          poster: Callable[[str, Mapping[str, Any]], None] | None) -> dict[str, Any]:
    """Post through the capacity controller's route: a new alert at once, the same ones again daily."""

    path = report_root / ALERT_STATE_FILENAME
    previous = _read_json(path) or {}
    keys = sorted({f"{alert['code']}:{alert['target']}" for alert in alerts})
    posted = set(previous.get("posted_keys") or [])
    last = previous.get("last_post_epoch")
    record = {"posted_keys": sorted(posted & set(keys)), "last_post_epoch": last,
              "last_error": previous.get("last_error"), "alert_keys": keys, "posted_now": False}
    due = bool(keys) and (bool(set(keys) - posted) or bool(previous.get("last_error"))
                          or not isinstance(last, (int, float)) or now - last >= ALERT_REPEAT_SECONDS)
    if due:
        if not webhook_url:
            record["last_error"] = "operator_alert_route_unconfigured"
        else:
            from . import control_plane_capacity_controller as capacity

            try:
                (poster or capacity.post_alert)(webhook_url, {"level": "critical", "mounts": [],
                                                              "alerts": [dict(alert) for alert in alerts]})
            except Exception as error:  # noqa: BLE001 - a failed post is retried next pass; it may carry the URL
                record["last_error"] = type(error).__name__
            else:
                record.update(posted_keys=keys, last_post_epoch=now, last_error=None, posted_now=True)
    elif not keys:
        record["last_error"] = None
    write_json(path, record)
    return record


def _summary(rows: Sequence[Mapping[str, Any]], *, approval: Mapping[str, Any], busy: Sequence[str],
             problems: Sequence[Mapping[str, str]], alerts: Sequence[Mapping[str, Any]],
             action: Mapping[str, Any] | None, now: float) -> dict[str, Any]:
    counts = dict.fromkeys(("total", "completed", "waiting", "not_yet_planned", "awaiting_owner_approval", "cool_off",
                            "unresolved", "money_unsettled", "legal_hold", "failing", "overdue", "incomplete",
                            "complete", "complete_provider_retention_remains"), 0)
    sites = []
    overdue = {alert["target"] for alert in alerts if alert["code"] == "website_withdrawal_deletion_overdue"}
    for row in rows:
        counts["total"] += 1
        if row["receipt"] is not None:
            counts["completed"] += 1
            counts[row["receipt"]["deletion_status"]] += 1
            sites.append({"scene_id": row["scene_id"], "status": row["receipt"]["deletion_status"],
                          "provider_retention_remains": row["receipt"]["provider_retention_remains"],
                          "acknowledged_at_epoch": row["acknowledged_at_epoch"]})
            continue
        counts["waiting"] += 1
        state = row["state"] or {}
        status, waiting = state.get("status"), set(state.get("waiting_on") or ())
        counts["not_yet_planned"] += state == {}
        counts["awaiting_owner_approval"] += status == "planned" and bool(waiting & _APPROVAL_WAITS)
        counts["cool_off"] += status == "planned" and "cool_off" in waiting
        counts["unresolved"] += status == "unresolved"
        counts["money_unsettled"] += "money_unsettled" in waiting
        counts["legal_hold"] += status == "legal_hold"
        counts["failing"] += status == "failed"
        counts["incomplete"] += status == "incomplete"
        counts["overdue"] += row["scene_id"] in overdue
        sites.append({"scene_id": row["scene_id"], "status": status or "not_planned",
                      "waiting_on": sorted(waiting), "failure_count": state.get("failure_count") or 0,
                      "acknowledged_at_epoch": row["acknowledged_at_epoch"]})
    return {"schema_version": SUMMARY_SCHEMA, "observed_at_epoch": now,
            "approval": {"status": approval["status"], "reason": approval["reason"], "digest": approval["digest"],
                         "expires_at": (approval["approval"] or {}).get("expires_at"),
                         "cool_off_hours": (approval["approval"] or {}).get("cool_off_hours")},
            "host": {"quiet": not busy, "reasons": list(busy)}, "scan_complete": not problems,
            "scan_problems": list(problems)[:20], "withdrawals": counts, "action": dict(action) if action else None,
            "alerts": [{key: alert[key] for key in ("code", "target", "reason") if key in alert} for alert in alerts],
            "sites": sites[:_MAX_SUMMARY_SITES]}


@contextmanager
def _host_lock(report_root: Path) -> Iterator[bool]:
    path = report_root / LOCK_FILENAME
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
            return
        yield True
    finally:
        os.close(descriptor)


def run_pass(*, site_roots: Sequence[Path], report_root: Path, resolution: Callable[[], Mapping[str, Any]],
             quiet: Callable[[], list[str]], approval_path: Path = APPROVAL_PATH, approval_owner_uid: int = 0,
             now: float | None = None, webhook_url: str = "",
             poster: Callable[[str, Mapping[str, Any]], None] | None = None,
             planner: Callable[..., dict[str, Any]] | None = None,
             applier: Callable[..., dict[str, Any]] | None = None,
             site_index_root: Path | None = None, indexed_roots: Sequence[Path] = (),
             index_budget: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """One timer pass: at most one site planned (and applied when approved), every state recorded, alerts posted."""

    if planner is None or applier is None:
        from . import website_capture_withdrawal_deletion as deletion

        planner = planner or deletion.plan_withdrawal_deletion
        applier = applier or deletion.apply_withdrawal_deletion
    observed = time.time() if now is None else float(now)
    report_root = Path(report_root)
    report_root.mkdir(parents=True, exist_ok=True, mode=0o750)
    with _host_lock(report_root) as held:
        if not held:
            return {"schema_version": SUMMARY_SCHEMA, "observed_at_epoch": observed, "skipped": "another_pass_running"}
        approval = load_approval(approval_path, now=observed, owner_uid=approval_owner_uid)
        sites, problems = discover_withdrawals(site_roots)
        rows = []
        for site in sites:
            try:
                rows.append(_site_row(site, observed))
            except OSError as error:
                # One site's unreadable journal must not stop the others; the scan reports it.
                problems.append({"path": str(site), "reason": "site_unreadable:" + _error_code(error)})
        busy = list(quiet())
        action = None
        index_report = None
        if site_index_root is not None and indexed_roots and not busy:
            from .website_withdrawal_site_index import advance

            # $0 and read-only, bounded per pass and resumable: the ownership index of the roots
            # with no per-site binding, read before the plan so it sees the newest entries.
            index_report = advance(site_index_root, indexed_roots, now=observed, **dict(index_budget or {}))
        due = sorted((row for row in rows if row["receipt"] is None and _due(row, approval, observed)),
                     key=lambda row: (float((row["state"] or {}).get("last_attempt_at_epoch") or -1), str(row["site"])))
        if due and not busy:
            row = due[0]
            state = _handle(row, approval=approval, approval_path=Path(approval_path), now=observed,
                            resolution=resolution, quiet=quiet, planner=planner, applier=applier)
            row.update(state=state, receipt=_sealed_receipt(row["site"], row["tombstone_digest"]))
            action = {"scene_id": row["scene_id"], "status": state["status"],
                      "waiting_on": state.get("waiting_on") or [], "last_error": state.get("last_error"),
                      "deletion_status": state.get("deletion_status")}
        alerts = _alerts(rows, approval, problems, observed)
        summary = _summary(rows, approval=approval, busy=busy, problems=problems, alerts=alerts, action=action,
                           now=observed)
        summary["site_index"] = index_report
        if site_index_root is not None and indexed_roots:
            summary["content_gc"] = _content_gc(site_index_root, indexed_roots, report_root=report_root,
                                                approval=approval, busy=busy, now=observed)
        summary["alert_delivery"] = _post(report_root, alerts, now=observed, webhook_url=webhook_url, poster=poster)
        write_json(report_root / SUMMARY_FILENAME, summary)
        return summary


def _content_gc(index_root: Path, roots: Sequence[Path], *, report_root: Path, approval: Mapping[str, Any],
                busy: Sequence[str], now: float) -> dict[str, Any]:
    """Count the content blobs nobody refers to, plan their garbage collection, and apply it only when the
    approval file names its own policy class (with every store fully read and a fresh check per blob)."""

    from . import website_withdrawal_content_blobs as content_blobs
    from .website_withdrawal_site_index import read_index

    unreferenced = {"count": 0, "bytes": 0}
    for root in map(Path, roots):
        store = ((read_index(index_root, root) or {}).get("entries") or {}).get(content_blobs.STORE_ENTRY) or {}
        for blob in (store.get("blobs") or {}).values():
            if blob["nlink"] <= 1:
                unreferenced["count"] += 1
                unreferenced["bytes"] += blob["size"]
    plan = content_blobs.plan_gc(index_root, roots, now=now)
    approved = approval["status"] == "valid" and (approval["approval"] or {}).get("garbage_collection_approved")
    result: dict[str, Any] = {"unreferenced_blobs": unreferenced, "plan_complete": plan["complete"],
                              "candidates": len(plan["candidates"]), "candidate_bytes": plan["bytes"],
                              "approved": bool(approved), "applied": None}
    if approved and plan["candidates"] and not busy:
        applied = content_blobs.apply_gc(plan, now=now)
        write_json(report_root / "content-gc" / f"{int(now)}.json", {**applied, "policy_class": plan["policy_class"],
                                                                      "approval_digest": approval["digest"]})
        result["applied"] = {"deleted": len(applied["deleted"]), "kept": len(applied["kept"]),
                             "bytes_deleted": applied["bytes_deleted"]}
    return result


_PROJECTED_COUNTS = ("total", "completed", "waiting", "not_yet_planned", "awaiting_owner_approval", "cool_off",
                     "unresolved", "money_unsettled", "legal_hold", "failing", "overdue", "incomplete",
                     "complete", "complete_provider_retention_remains")


def capacity_projection(summary: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """The secret-free counts the capacity summary carries for ``door status``; None when nothing was read."""

    if summary is None:
        return None
    if summary.get("schema_version") != SUMMARY_SCHEMA:
        return {"status": "unreadable"}
    counts = summary.get("withdrawals") if isinstance(summary.get("withdrawals"), Mapping) else {}
    approval = summary.get("approval") if isinstance(summary.get("approval"), Mapping) else {}
    host = summary.get("host") if isinstance(summary.get("host"), Mapping) else {}
    observed = summary.get("observed_at_epoch")
    status = approval.get("status")
    return {
        "status": "observed",
        "observed_at_epoch": observed if isinstance(observed, (int, float)) and not isinstance(observed, bool) else None,
        "approval_status": status if status in {"absent", "invalid", "expired", "valid"} else "unknown",
        "host_quiet": host.get("quiet") if isinstance(host.get("quiet"), bool) else None,
        "scan_complete": summary.get("scan_complete") if isinstance(summary.get("scan_complete"), bool) else None,
        "withdrawals": {key: counts[key] for key in _PROJECTED_COUNTS
                        if type(counts.get(key)) is int and counts[key] >= 0},
        "unreferenced_content_blobs": {key: value for key, value in (
            (summary.get("content_gc") or {}).get("unreferenced_blobs") or {}).items()
            if key in {"count", "bytes"} and type(value) is int},
        "alert_codes": sorted({str(alert.get("code")) for alert in summary.get("alerts") or []
                               if isinstance(alert, Mapping) and _CODE.fullmatch(str(alert.get("code")))}),
    }


def _production_resolution() -> dict[str, Any]:
    """The roots the deletion tool defaults to (the units' environment) and the existing store clients."""

    from . import website_capture_withdrawal as withdrawal
    from .website_capture_withdrawal_deletion import B2ObjectVersions

    bucket = str(os.getenv(SOURCE_BUCKET_ENV) or "").strip()
    from .website_capture_withdrawal_provider_copies import production_deleters
    from .website_capture_withdrawal_unclaimed import DEFAULT_ROOT as UNCLAIMED_ROOT

    report_root = Path(os.getenv(REPORT_ROOT_ENV) or DEFAULT_REPORT_ROOT)
    return {"object_store": B2ObjectVersions.configured(), "unclaimed_handoff_root": UNCLAIMED_ROOT,
            "site_index_root": report_root / SITE_INDEX_DIRNAME, "provider_copies": True,
            "provider_deleters": production_deleters(),
            "cloud_storage": withdrawal.GoogleWebsiteCleanupStorage(bucket) if bucket else None}


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m blueprint_pipeline.website_capture_withdrawal_automatic``: one pass, as the timer runs it."""

    from .control_plane_capacity_controller import WEBHOOK_URL_ENV

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--site-root", type=Path, action="append",
                        help="Root holding <bucket>/scenes/site-* (repeatable); default: " + SITE_ROOTS_ENV)
    parser.add_argument("--report-root", type=Path, default=Path(os.getenv(REPORT_ROOT_ENV) or DEFAULT_REPORT_ROOT))
    args = parser.parse_args(argv)
    roots = args.site_root or [Path(item) for item in str(os.getenv(SITE_ROOTS_ENV) or "").split(":") if item] \
        or list(DEFAULT_SITE_ROOTS)
    from .website_capture_withdrawal_deletion import UNCOVERED_ROOTS

    summary = run_pass(site_roots=roots, report_root=args.report_root, resolution=_production_resolution,
                       quiet=paid_stage_reasons, webhook_url=str(os.getenv(WEBHOOK_URL_ENV) or "").strip(),
                       site_index_root=args.report_root / SITE_INDEX_DIRNAME, indexed_roots=UNCOVERED_ROOTS)
    print(json.dumps({key: summary.get(key) for key in ("approval", "host", "withdrawals", "action", "alerts",
                                                        "skipped") if key in summary}, sort_keys=True))
    return 0


__all__ = [
    "APPROVAL_PATH",
    "APPROVAL_SCHEMA",
    "POLICY_CLASS",
    "STATE_SCHEMA",
    "SUMMARY_SCHEMA",
    "capacity_projection",
    "discover_withdrawals",
    "load_approval",
    "main",
    "paid_stage_reasons",
    "run_pass",
]


if __name__ == "__main__":  # pragma: no cover - exercised through the unit
    sys.exit(main())
