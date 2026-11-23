"""Page the owner once when a website preparation stops on a mismatch in Blueprint's own records.

Three preparation stops are ours to fix, never the site's (2026-11-11):

- ``website_registration_anchor_frame_missing`` and ``website_registration_anchor_frame_invalid``: the
  reconstruction's anchor does not bind to the source views Blueprint itself recorded;
- ``website_registration_scale_conflicts_declared``: the object's estimated size and the scene's
  declared scale, both Blueprint's estimates, disagree by more than a third.

Such a preparation fails, gets the shared retry policy's six timed retries (about 46 hours) and is then
left until the job's window ends it. The site is told only that the job needs our team's review, and
before this nothing told the owner: no stall check reads the preparation ledger. This check pages once
per capture and code, on the stall alarm's route. It gates nothing and writes nothing but the alarm's
memory: it reads each ledger's ``status`` and ``last_error``.

Run inside the stall alarm's tick (``control_plane_stall_alarm``). A persisting page repeats hourly
there, so the row is emitted only for ``ANNOUNCE_SECONDS`` after the tick that first saw the stop, as
the stall alarm's other announced pages do.
"""
from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

CODE = "website_preparation_blueprint_fault"
ANNOUNCE_SECONDS = 30 * 60
LEDGER_GLOB = "*/scenes/site-*/captures/*/pipeline_job_ledger.json"
MAX_LEDGERS = 2048
MAX_LEDGER_BYTES = 1024 * 1024
FAULT_CODES = ("website_registration_anchor_frame_invalid", "website_registration_anchor_frame_missing",
               "website_registration_scale_conflicts_declared",
               # Since 2026-11-18: a preparation step failed in Blueprint's own code, not on a typed code
               # (``website_scene_handoff.DEFECT``); it waits for a fix release until the job's window ends.
               "website_preparation_defect",
               # A host configuration the step needs is missing (a provider key, a segmenter profile).
               "website_assembly_coverage_key_missing", "website_task_grounding_key_missing",
               "website_sam31_profile_missing")
_FAULT = re.compile(r"\b(" + "|".join(FAULT_CODES) + r")\b")
#: Since 2026-11-18: a website job became ready with the depth-fused draft as its room's look for good
#: (``look_fallback`` on its Atlas job, ``website_atlas_preview``): the refined splat could not be had within
#: the job's allowance or before its window's last two hours (``atlas_refined_splat_retry``). Paged once per
#: capture, on the same route and memory as the faults above; it gates nothing.
LOOK_CODE = "website_atlas_look_draft_final"
ATLAS_JOB_GLOB = "*/scenes/site-*/captures/*/pipeline/atlas/preview_job.json"
#: Since 2026-11-22: an outside service refused Blueprint's own key or account (401, 402, 403) and a room build
#: waits on it as on an outage (``job_retry_policy.provider_account_refused``). Paged once per service while any
#: job waits on it; the jobs resume by themselves once the key or account works. It gates nothing.
KEY_CODE = "website_provider_account_refused"
_ATLAS_API_ACCOUNT = re.compile(r"website_atlas_api_40[123]")
#: The blocker a provider's "no credit left" answer is paged with (``job_retry_policy.provider_funding_refused``).
_NO_CREDIT = "insufficient_quota"
_STOPPED = frozenset({"failed_retryable", "retryable_blocked", "failed_terminal"})


def _fault(ledger: Mapping[str, Any]) -> str | None:
    if ledger.get("status") not in _STOPPED:
        return None
    texts = [ledger.get("last_error"), *(ledger.get("retry_blockers") or [])]
    for text in texts:
        found = _FAULT.search(text) if isinstance(text, str) else None
        if found:
            return found.group(1)
    return None


def check_preparation_faults(root: Path | None, *, now: float, remembered: Mapping[str, Any] | None = None,
                             memory_out: dict[str, Any] | None = None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """One page row naming each capture newly stopped on a Blueprint-side code; unreadable ledgers are counted."""
    remembered = remembered if isinstance(remembered, Mapping) else {}
    if root is None:
        return {"status": "not_configured"}, []
    if not Path(root).is_dir():
        return {"status": "absent"}, []
    paths = sorted(Path(root).glob(LEDGER_GLOB))
    truncated = len(paths) > MAX_LEDGERS
    offenders, unreadable, stopped = [], 0, 0
    for path in paths[:MAX_LEDGERS]:
        try:
            if path.is_symlink() or path.stat().st_size > MAX_LEDGER_BYTES:
                raise ValueError("ledger_unreadable")
            ledger = json.loads(path.read_text(encoding="utf-8"))
            code = _fault(ledger) if isinstance(ledger, Mapping) else None
        except (OSError, ValueError):
            unreadable += 1
            continue
        if code is None:
            continue
        stopped += 1
        key = f"{path.parent.relative_to(root)}:{code}"
        prior = remembered.get(key) if isinstance(remembered.get(key), Mapping) else {}
        since = prior.get("since")
        since = since if isinstance(since, (int, float)) and not isinstance(since, bool) else now
        if memory_out is not None:
            memory_out[key] = {"since": since}
        if now - since < ANNOUNCE_SECONDS:
            offenders.append({"target": key, "label": path.parent.name[:120], "status": ledger.get("status"),
                              "blockers": [code], "age_seconds": round(now - since)})
    looks, look_unreadable, look_truncated = _draft_looks(Path(root), now=now, remembered=remembered,
                                                          memory_out=memory_out)
    section = {"status": "observed", "ledgers": min(len(paths), MAX_LEDGERS), "truncated": truncated,
               "stopped": stopped, "announced": len(offenders), "unreadable": unreadable,
               "draft_looks_announced": len(looks), "draft_looks_unreadable": look_unreadable}
    alerts: list[dict[str, Any]] = []
    keys = _account_refusals(Path(root), now=now, remembered=remembered, memory_out=memory_out)
    section["provider_account_refusals_announced"] = len(keys)
    for code, rows in ((CODE, offenders), (LOOK_CODE, looks), (KEY_CODE, keys)):
        if rows:
            rows.sort(key=lambda row: row["target"])
            alerts.append({"code": code, "count": len(rows), "targets": [row["target"] for row in rows[:10]], **rows[0]})
    if look_truncated:
        alerts.append({"code": "stall_check_incomplete", "reason": "atlas_jobs_truncated"})
    if truncated:
        # More ledgers than one tick reads: what was not read is unknown, never "nothing found".
        alerts.append({"code": "stall_check_incomplete", "reason": "preparation_ledgers_truncated",
                       "count": len(paths)})
    return section, alerts


def _draft_looks(root: Path, *, now: float, remembered: Mapping[str, Any],
                 memory_out: dict[str, Any] | None) -> tuple[list[dict[str, Any]], int, bool]:
    """One row per capture whose ready Atlas job's look is the draft for good, within its announce window."""
    paths = sorted(root.glob(ATLAS_JOB_GLOB))
    rows, unreadable = [], 0
    for path in paths[:MAX_LEDGERS]:
        try:
            if path.is_symlink() or path.stat().st_size > MAX_LEDGER_BYTES:
                raise ValueError("job_unreadable")
            job = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            unreadable += 1
            continue
        reason = job.get("look_fallback") if isinstance(job, Mapping) and job.get("state") == "ready" else None
        if not isinstance(reason, str):
            continue
        capture = path.parents[2]  # <capture>/pipeline/atlas/preview_job.json
        key = f"{capture.relative_to(root)}:{LOOK_CODE}"
        prior = remembered.get(key) if isinstance(remembered.get(key), Mapping) else {}
        since = prior.get("since")
        since = since if isinstance(since, (int, float)) and not isinstance(since, bool) else now
        if memory_out is not None:
            memory_out[key] = {"since": since}
        if now - since < ANNOUNCE_SECONDS:
            rows.append({"target": key, "label": capture.name[:120], "status": "ready", "blockers": [reason[:120]],
                         "age_seconds": round(now - since)})
    return rows, unreadable, len(paths) > MAX_LEDGERS


def _account_refusals(root: Path, *, now: float, remembered: Mapping[str, Any],
                      memory_out: dict[str, Any] | None) -> list[dict[str, Any]]:
    """One row per outside service whose refusal of Blueprint's key or account a waiting Atlas job last met: a
    coded 401, 402 or 403, or (since 2026-11-23) a provider SDK's answer that the account has no credit left, read
    from the job's retained ``last_error``; the row counts the jobs waiting on it."""
    from .job_retry_policy import provider_account_refused, provider_funding_refused
    services: dict[str, str] = {}
    waiting: dict[str, int] = {}
    for path in sorted(root.glob(ATLAS_JOB_GLOB))[:MAX_LEDGERS]:
        try:
            if path.is_symlink() or path.stat().st_size > MAX_LEDGER_BYTES:
                continue
            job = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue  # counted by the draft-look read of the same files
        reason = job.get("last_transient_reason") if isinstance(job, Mapping) else None
        if not (isinstance(reason, str) and job.get("transient_failures") and job.get("state") == "processing"):
            continue
        error = job.get("last_error") if isinstance(job.get("last_error"), Mapping) else {}
        if provider_account_refused(reason) or _ATLAS_API_ACCOUNT.fullmatch(reason):
            head = reason.partition(":")[0]
            service = head.rsplit("_http_", 1)[0] if "_http_" in head else head[:-4]
        elif provider_funding_refused(error.get("message")) and isinstance(error.get("type"), str):
            head, service = _NO_CREDIT, error["type"].partition(".")[0][:60]
        else:
            continue
        services.setdefault(service, head)
        waiting[service] = waiting.get(service, 0) + 1
    rows = []
    for service, head in sorted(services.items()):
        label = f"{service} ({waiting[service]} job{'s' if waiting[service] != 1 else ''} waiting)"
        key = f"provider:{service}:{KEY_CODE}"
        prior = remembered.get(key) if isinstance(remembered.get(key), Mapping) else {}
        since = prior.get("since")
        since = since if isinstance(since, (int, float)) and not isinstance(since, bool) else now
        if memory_out is not None:
            memory_out[key] = {"since": since}
        if now - since < ANNOUNCE_SECONDS:
            rows.append({"target": key, "label": label[:120], "status": "processing", "blockers": [head[:120]],
                         "age_seconds": round(now - since)})
    return rows


def describe(alert: Mapping[str, Any]) -> str:
    more = int(alert.get("count") or 1) - 1
    tail = f" (+{more} more)" if more > 0 else ""
    blockers = alert.get("blockers") or ["?"]
    if alert.get("code") == KEY_CODE and blockers[0] == _NO_CREDIT:
        return (f"an outside service says Blueprint's account has no credit left [{blockers[0]}]; only the owner can "
                f"add it; room builds wait on it as on an outage and resume by themselves once it works: "
                f"{alert.get('label')}{tail}")
    if alert.get("code") == KEY_CODE:
        return (f"an outside service refused Blueprint's own key or account [{blockers[0]}]; room builds wait on it "
                f"as on an outage and resume by themselves once it works: {alert.get('label')}{tail}")
    if alert.get("code") == LOOK_CODE:
        return (f"website job ready with the depth-fused draft as its room's look [{blockers[0]}]; no refined splat "
                f"within its allowance or window: {alert.get('label') or alert.get('target')}{tail}")
    return (f"website preparation stopped on Blueprint's own records or code [{blockers[0]}], ours to fix; the site "
            f"is told only that it needs our review: {alert.get('label') or alert.get('target')}{tail}")
