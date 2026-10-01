"""Treat control-plane disk capacity as a plan, not a floor.

The disk-admission ledger refuses a stage when free space would drop under the
floor.  That guard is correct and it is also the only capacity signal the host
had: the first anyone learned that the disk was full was a ``503`` at intake,
after four fills.  Nothing measured, nothing forecast, nothing alerted, and
nothing grew.

Every tick this controller measures each configured mount, projects admission
per stage from the live reservation ledger exactly as intake will, appends the
observation to an append-only history, forecasts when the floor is reached at
the observed growth rate, alerts the operator webhook when a mount crosses the
warning or critical fraction or when any stage would be refused, and, when a
resizable block volume is configured and acknowledged, grows it one step and
resizes the filesystem online.  The report it writes is ``evidence_hot``: the
capacity record of the host, never pruned.

At most hourly it also surveys usage (``control_plane_disk_usage.survey_usage``):
every byte on its mounts and the root disk, once per inode, by storage class, root
and owner.  The survey is written to ``usage-latest.json``, a compact projection is
embedded in the report, and every tick writes ``summary.json``: a secret-free view
of the report that the operator door, which runs as the service account, can read.

It spends nothing unless the resize acknowledgement is set, and even then it
grows only the configured volume, only up to the configured maximum, only when
the volume's mount is critical.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from collections.abc import Callable, Collection, Mapping, Sequence
from pathlib import Path
from typing import Any

from .control_plane_capacity_evidence import (
    CHAIN_ROLES,
    CRITICAL_FRACTION,
    WARNING_FRACTION,
    _read_attention_summary,
    capacity_eta,
    chain_footprints,
    footprint_basis,
    measure_mount,
    whole_chain_admission,  # noqa: F401 - compatibility re-export
)

from . import control_plane_disk_budget as disk_budget
from .control_plane_disk_budget import DEFAULT_RESERVATION_ROOT
from .control_plane_disk_usage import (
    SURVEY_SCHEMA_VERSION, is_orphan_scratch_root, sanitize_public_survey, survey_usage,
)
from .decision_evidence_contracts import canonical_digest

SCHEMA_VERSION = "control_plane_capacity_report.v1"
SUMMARY_SCHEMA_VERSION = "control_plane_capacity_summary.v1"
USAGE_FILENAME = "usage-latest.json"
USAGE_ATTEMPT_FILENAME = "usage-attempt.json"
USAGE_ATTEMPT_SCHEMA_VERSION = "control_plane_usage_survey_attempt.v1"
SUMMARY_FILENAME = "summary.json"
# The door's status reader refuses files over 256 KiB; the summary keeps half that.
SUMMARY_MAX_BYTES = 128 * 1024
DEFAULT_SURVEY_INTERVAL_SECONDS = 60 * 60
USAGE_UNCLASSIFIED_ALERT_BYTES = 1024**3
USAGE_CONTAINER_RUNTIME_ALERT_BYTES = 20 * 1024**3
ORPHAN_SCRATCH_AGGREGATE_PAGE_BYTES = 5 * 1024**3
ORPHAN_SCRATCH_SINGLE_PAGE_BYTES = 2 * 1024**3
USAGE_ATTRIBUTION_ALERT_FRACTION = 0.9
USAGE_PROJECTED_UNCLASSIFIED_ROOTS = 20
RESIZE_RECEIPT_SCHEMA_VERSION = "control_plane_volume_resize_receipt.v1"
DEFAULT_REPORT_ROOT = Path("/var/lib/blueprint/pipeline-control-plane/capacity")
DEFAULT_RELEASE_RETIREMENT_SUMMARY = Path(
    "/var/lib/blueprint/pipeline-control-plane/release-retention/latest-deploy-retirement.json"
)
DEFAULT_BREAK_GLASS_NOTES_ROOT = Path("/var/lib/blueprint/pipeline-control-plane/cleanup-receipts")
DEFAULT_STORAGE_GC_SUMMARY = Path(
    "/var/lib/blueprint/pipeline-control-plane/storage-gc/summary.json"
)
GC_SUMMARY_INTERVAL_SECONDS = 3600
DEFAULT_MOUNTS: tuple[str, ...] = ("/var/lib/blueprint",)
WORK_VOLUME_MOUNT = "/mnt/blueprint-work"
_DEFAULT_SURVEY = object()
FORECAST_WINDOW_SECONDS = 7 * 24 * 60 * 60
ALERT_REPEAT_SECONDS = 60 * 60
RESIZE_ACK = "grow-control-plane-volume"
DEFAULT_RESIZE_STEP_GIB = 50
GIB = 1024**3
# The week trend averages a burst away, and a reclaim inside the week makes it
# read ``not_growing`` while a writer fills the disk (2026-10-01: a local image
# unpack took root from 55 to 31 GB with the floor hours away, and nothing paged).
# The recent window projects only the last hour of ticks, and only growth the
# reservation ledger does not explain (admitted writers are already budgeted).
# A page set below six hours clears at twelve, so a writer near the threshold
# does not post on every tick.
RECENT_FORECAST_WINDOW_SECONDS = 60 * 60
RECENT_FORECAST_MIN_SECONDS = 30 * 60
RECENT_DECLINE_MIN_BYTES = 2 * GIB
FLOOR_WITHIN_HOURS_PAGE = 6.0
FLOOR_WITHIN_HOURS_CLEAR = 12.0
PAGE_ALERT_CODES = frozenset({
    "floor_within_three_days", "floor_within_hours", "admission_refused", "critical_admission_refused",
    "mount_unreadable", "volume_growth_blocked", "operator_alert_route_unconfigured",
    "reclaim_ineffective",
    "orphan_scratch_large",
})


def _severity(code: str) -> str:
    return "page" if code in PAGE_ALERT_CODES else "warn"


def _annotate_alerts(alerts: list[dict[str, Any]]) -> None:
    for alert in alerts:
        alert["severity"] = _severity(str(alert.get("code") or ""))


def alert_fingerprint(report: Mapping[str, Any]) -> str:
    """Stable identity of page-severity alert targets, independent of row order."""
    rows = sorted({
        (str(alert.get("code") or ""), str(alert.get("mount") or ""), "page")
        for alert in report.get("alerts") or []
        if isinstance(alert, Mapping) and alert.get("severity") == "page"
    })
    return "sha256:" + hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest()



MOUNTS_ENV = "BLUEPRINT_CAPACITY_MOUNTS"
REPORT_ROOT_ENV = "BLUEPRINT_CAPACITY_REPORT_ROOT"
RESERVATION_ROOT_ENV = "BLUEPRINT_CONTROL_PLANE_DISK_RESERVATION_ROOT"
WEBHOOK_URL_ENV = "BLUEPRINT_OPERATOR_ALERT_WEBHOOK_URL"
VOLUME_ID_ENV = "BLUEPRINT_CAPACITY_VOLUME_ID"
VOLUME_MOUNT_ENV = "BLUEPRINT_CAPACITY_VOLUME_MOUNT"
VOLUME_DEVICE_ENV = "BLUEPRINT_CAPACITY_VOLUME_DEVICE"
VOLUME_MAX_GIB_ENV = "BLUEPRINT_CAPACITY_VOLUME_MAX_GIB"
VOLUME_STEP_GIB_ENV = "BLUEPRINT_CAPACITY_VOLUME_STEP_GIB"
RESIZE_ACK_ENV = "BLUEPRINT_CAPACITY_AUTORESIZE_ACK"
SURVEY_INTERVAL_ENV = "BLUEPRINT_CAPACITY_SURVEY_INTERVAL_SECONDS"
DO_TOKEN_FILE_ENV = "DIGITALOCEAN_API_TOKEN_FILE"
DO_VOLUME_ACTIONS_URL = "https://api.digitalocean.com/v2/volumes/{volume_id}/actions"



class ControlPlaneCapacityError(RuntimeError):
    """The controller's configuration or a resize could not be trusted."""


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return loaded if isinstance(loaded, dict) else None




def _finite_number(value: int | float) -> bool:
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _reclaim_outlook(
    summary: Mapping[str, Any] | None, *, now: float, volume_growth: str
) -> tuple[dict[str, Any], list[str], bool]:
    """Use only a fresh applying GC summary for reclaim bytes and page reasons."""

    reclaim_phases = (
        "derived_directories", "content_store", "scratch_directories",
        "workspace_bundles", "result_artifact_offload", "evidence_offload",
        "replay_caches", "scene_workspaces", "result_residue_offload",
    )
    sources = dict.fromkeys(reclaim_phases)
    outlook: dict[str, Any] = {
        "observed_at_epoch": None, "next_reclaim_epoch": None,
        "reclaimable_bytes": None, "sources": sources, "volume_growth": volume_growth,
    }
    if not isinstance(summary, Mapping) or summary.get("schema_version") != "control_plane_storage_gc_summary.v1":
        return outlook, [], False
    observed = summary.get("observed_at_epoch")
    if (
        type(observed) not in (int, float) or not _finite_number(observed)
        or not 0 <= now - observed <= 2 * GC_SUMMARY_INTERVAL_SECONDS
        or summary.get("status") != "applied"
    ):
        return outlook, [], False
    phases = summary.get("phases")
    opt_in = summary.get("opt_in")
    if not isinstance(phases, Mapping) or not isinstance(opt_in, Mapping):
        return outlook, [], False
    skipped_roots = summary.get("skipped_roots")
    phase_errors = summary.get("phase_errors")
    if (not isinstance(skipped_roots, list) or not isinstance(phase_errors, list)
            or skipped_roots or phase_errors):
        return outlook, [], False
    enabled = [name for name in reclaim_phases[:4] if name in phases]
    if opt_in.get("evidence_offload") is True:
        if "result_artifact_offload" in phases:
            enabled.append("result_artifact_offload")
        if "evidence_offload" in phases:
            enabled.append("evidence_offload")
    if opt_in.get("replay_cache_retention") is True and "replay_caches" in phases:
        enabled.append("replay_caches")
    if opt_in.get("scene_workspace_retirement") is True:
        if "scene_workspaces" in phases:
            enabled.append("scene_workspaces")
    # Residue offload needs both owner switches. Older summaries omitted its
    # switch, so an enabled-looking phase is unknown until both are explicit.
    residue_switch = opt_in.get("result_residue_offload")
    residue_held: Mapping[str, Any] | None = None
    if (opt_in.get("evidence_offload") is not False and residue_switch is not False
            and ("result_residue_offload" in phases or residue_switch is True)):
        residue = phases.get("result_residue_offload")
        if (opt_in.get("evidence_offload") is not True or residue_switch is not True
                or not isinstance(residue, Mapping) or residue.get("enabled") is not True):
            return outlook, [], False
        # The producer's candidate total may include rows retained after a
        # publication or reference recheck. It does not partition those bytes
        # from actionable rows, so such a phase supplies no forecast bytes. It
        # no longer blanks the other phases, though: with residue offload on by
        # default, steady-state retained rows (a missing dispatch receipt, a hot
        # run, one already offloaded) would otherwise hide every forecast and
        # silence the reclaim_ineffective page.
        retained = residue.get("retained_by_reason")
        if not isinstance(retained, Mapping):
            return outlook, [], False
        if retained:
            residue_held = residue
        else:
            enabled.append("result_residue_offload")
    # Pin releases remove no bytes. Derived-directory candidates already
    # account for any reproducible files they make eligible in this tick.
    if not enabled:
        return outlook, [], False
    total_candidate = 0
    total_reclaimed = 0
    total_remaining = 0
    for name in enabled:
        phase = phases.get(name)
        if (
            not isinstance(phase, Mapping)
            or (name != "result_artifact_offload" and phase.get("status") != "applied")
        ):
            return outlook, [], False
        candidate = phase.get("candidate_bytes")
        reclaimed = phase.get("removed_or_offloaded_bytes")
        if (
            type(candidate) is not int or candidate < 0
            or type(reclaimed) is not int or reclaimed < 0
        ):
            return outlook, [], False
        # Candidate totals describe the pre-apply plan. Completed reclamation
        # cannot be promised again to a scene waiting after this observation.
        remaining = max(0, candidate - reclaimed)
        sources[name] = remaining
        total_candidate += candidate
        total_reclaimed += reclaimed
        total_remaining += remaining
    # A held residue phase forecasts nothing, but bytes it still holds or has
    # moved mean reclaim is not ineffective.
    held_bytes = 0
    if residue_held is not None:
        for field in ("candidate_bytes", "removed_or_offloaded_bytes"):
            value = residue_held.get(field)
            if type(value) is not int or value < 0:
                return outlook, [], False
            held_bytes += value
    outlook.update({
        "observed_at_epoch": observed,
        "next_reclaim_epoch": observed + GC_SUMMARY_INTERVAL_SECONDS,
        "reclaimable_bytes": total_remaining,
    })
    # Older summaries have only capped phase rows, which cannot prove the
    # largest reasons across phases. Wait for a complete producer summary.
    reason_rows = summary.get("top_retained_reasons")
    if not isinstance(reason_rows, list):
        return outlook, [], False
    reason_bytes: dict[str, int] = {}
    for row in reason_rows:
        if not (isinstance(row, Mapping)
                and type(row.get("bytes")) is int and row["bytes"] > 0
                and isinstance(row.get("reason"), str)
                and re.fullmatch(r"[a-z][a-z0-9_:+.-]{0,79}", row["reason"])):
            return outlook, [], False
        reason_bytes[row["reason"]] = reason_bytes.get(row["reason"], 0) + row["bytes"]
    reasons = sorted(reason_bytes, key=lambda reason: (-reason_bytes[reason], reason))[:3]
    return outlook, reasons, total_candidate == total_reclaimed == held_bytes == 0


def live_reserved_bytes(
    reservation_root: Path, *, now: float, mount: str | Path = DEFAULT_MOUNTS[0]
) -> tuple[int, int]:
    """Bytes and count of live reservations on ``mount``'s device, as admission counts them.

    Kept for compatibility; liveness is the ledger's own (device, TTL, live pid).
    """

    return disk_budget.live_reservations(
        reservation_root, device=disk_budget.target_device(mount), now=now
    )










def _recent_forecast(rows: Sequence[Mapping[str, Any]], current: Mapping[str, Any], *,
                     now: float) -> dict[str, Any]:
    """The last hour's decline in free bytes, and hours until admission headroom is
    gone if the part the reservation ledger does not explain continues."""

    def epoch(row: Mapping[str, Any]) -> float:
        return float(row["observed_at_epoch"])

    recent = [row for row in rows if now - epoch(row) <= RECENT_FORECAST_WINDOW_SECONDS]
    if not recent:
        return {"status": "insufficient_history"}
    oldest = min(recent, key=epoch)
    elapsed = now - epoch(oldest)
    later = [row for row in recent if epoch(row) > epoch(oldest)]
    if elapsed < RECENT_FORECAST_MIN_SECONDS or not later:
        return {"status": "insufficient_history"}
    middle = min(later, key=lambda row: abs(epoch(row) - (epoch(oldest) + now) / 2))
    free = int(current["free_bytes"])
    decline = int(oldest["free_bytes"]) - free
    # Bytes written under a reservation were admitted against the floor; the
    # largest reservation the window saw bounds what admitted writers explain.
    reserved = max([value for value in (row.get("reserved_bytes") for row in [*recent, current])
                    if type(value) is int] or [0])
    result: dict[str, Any] = {"status": "not_declining", "window_seconds": int(elapsed),
                              "decline_bytes_per_hour": int(decline / elapsed * 3600),
                              "reserved_bytes": reserved}
    if decline < RECENT_DECLINE_MIN_BYTES:
        return result
    if decline - reserved < RECENT_DECLINE_MIN_BYTES:
        result["status"] = "explained_by_reservations"
        return result
    halves = (int(oldest["free_bytes"]) - int(middle["free_bytes"]), int(middle["free_bytes"]) - free)
    if min(halves) < RECENT_DECLINE_MIN_BYTES / 2:
        result["status"] = "not_sustained"
        return result
    available = current.get("available_bytes")
    if type(available) is not int:
        available = free - int(current["floor_bytes"]) - int(current.get("reserved_bytes") or 0)
    result["status"] = "declining"
    result["hours_until_floor"] = round(max(0, available) / ((decline - reserved) / elapsed * 3600), 2)
    return result


def forecast(history: Sequence[Mapping[str, Any]], current: Mapping[str, Any], *, now: float) -> dict[str, Any]:
    """Growth per day and days until the floor, from the oldest row inside the window,
    and under ``recent`` the same projection from the last hour alone."""

    mount = current.get("mount")
    rows = [
        row
        for row in history
        if isinstance(row, Mapping)
        and row.get("mount") == mount
        and row.get("status") == "measured"
        and isinstance(row.get("observed_at_epoch"), (int, float))
        and type(row.get("free_bytes")) is int
        and now - float(row["observed_at_epoch"]) <= FORECAST_WINDOW_SECONDS
    ]
    if current.get("status") != "measured":
        return {"status": "insufficient_history"}
    recent = _recent_forecast(rows, current, now=now)
    if not rows:
        return {"status": "insufficient_history", "recent": recent}
    oldest = min(rows, key=lambda row: float(row["observed_at_epoch"]))
    elapsed = now - float(oldest["observed_at_epoch"])
    if elapsed < 3600:
        return {"status": "insufficient_history", "recent": recent}
    growth_per_day = (int(oldest["free_bytes"]) - int(current["free_bytes"])) / elapsed * 86400
    headroom = int(current["free_bytes"]) - int(current["floor_bytes"])
    if growth_per_day <= 0:
        return {"status": "not_growing", "growth_bytes_per_day": int(growth_per_day), "recent": recent}
    return {
        "status": "growing",
        "growth_bytes_per_day": int(growth_per_day),
        "days_until_floor": round(max(0.0, headroom / growth_per_day), 2),
        "recent": recent,
    }


def load_history(report_root: Path, *, limit: int = 4096) -> list[dict[str, Any]]:
    path = report_root / "history.jsonl"
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()[-limit:]
    except OSError:
        return []
    for line in lines:
        try:
            loaded = json.loads(line)
        except ValueError:
            continue
        if isinstance(loaded, dict):
            rows.append(loaded)
    return rows


def build_capacity_report(
    *,
    mounts: Sequence[str | Path],
    reservation_root: str | Path = DEFAULT_RESERVATION_ROOT,
    history: Sequence[Mapping[str, Any]] = (),
    disk_usage: Callable[[str | os.PathLike[str]], Any] = shutil.disk_usage,
    now: float | None = None,
    burst_paging: Collection[str] = (),
) -> dict[str, Any]:
    """``burst_paging`` names mounts whose previous report paged ``floor_within_hours``."""

    observed = time.time() if now is None else float(now)
    measured = [
        measure_mount(mount, reservation_root=reservation_root, disk_usage=disk_usage, now=observed)
        for mount in mounts
    ]
    for row in measured:
        row["observed_at_epoch"] = observed
        row["forecast"] = forecast(history, row, now=observed)
    levels = [row.get("level", "ok") for row in measured]
    level = "critical" if "critical" in levels or any(r["status"] not in {"measured", "absent"} for r in measured) else (
        "warning" if "warning" in levels else "ok"
    )
    alerts = []
    for row in measured:
        if row["status"] != "measured":
            alerts.append({"mount": row["mount"], "code": "mount_unreadable"
                           if row["status"] == "unreadable" else f"mount_{row['status']}"})
            continue
        if row["refused_roles"]:
            alerts.append({"mount": row["mount"], "code": "admission_refused", "roles": row["refused_roles"]})
        if row["critical_roles_refused"]:
            alerts.append({"mount": row["mount"], "code": "critical_admission_refused",
                           "roles": row["critical_roles_refused"]})
        recent = row["forecast"].get("recent") or {}
        hours = recent.get("hours_until_floor")
        limit = FLOOR_WITHIN_HOURS_CLEAR if row["mount"] in burst_paging else FLOOR_WITHIN_HOURS_PAGE
        if isinstance(hours, (int, float)) and hours < limit:
            alerts.append({"mount": row["mount"], "code": "floor_within_hours", "hours_until_floor": hours,
                           "decline_bytes_per_hour": recent["decline_bytes_per_hour"],
                           "window_seconds": recent["window_seconds"]})
        if row["level"] != "ok":
            alerts.append({"mount": row["mount"], "code": f"utilization_{row['level']}", "used_fraction": row["used_fraction"]})
        days = row["forecast"].get("days_until_floor")
        if isinstance(days, (int, float)) and days < 3:
            alerts.append({"mount": row["mount"], "code": "floor_within_three_days", "days_until_floor": days})
    _annotate_alerts(alerts)
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "observed_at_epoch": observed,
        "level": level,
        "mounts": measured,
        "alerts": alerts,
        "report_digest": "",
    }
    report["report_digest"] = canonical_digest(report, digest_field="report_digest")
    return report


def survey_mounts(mounts: Sequence[str | Path]) -> list[str]:
    """Survey controller mounts, the root disk, and the attached bulk volume."""

    listed = [str(mount) for mount in mounts]
    if "/" not in {os.path.normpath(mount) for mount in listed}:
        listed.append("/")
    if (os.path.ismount(WORK_VOLUME_MOUNT)
            and WORK_VOLUME_MOUNT not in {os.path.normpath(mount) for mount in listed}):
        listed.append(WORK_VOLUME_MOUNT)
    return listed


def _write_public_json(path: Path, document: Mapping[str, Any]) -> None:
    """Replace ``path`` atomically with a world-readable (0644) JSON document.

    The unit's ``UMask=0077`` makes new files root-only, so the mode is set before
    the rename and the published file is never unreadable to the door.
    """

    temporary = path.with_name(f".{path.name}-{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.chmod(temporary, 0o644)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _refresh_usage(
    report_root: Path,
    *,
    survey: Callable[..., Mapping[str, Any]] | None,
    mounts: Sequence[str | Path],
    interval_seconds: float,
    force: bool,
    now: float,
) -> tuple[dict[str, Any] | None, str | None]:
    """The latest usage survey, paced by attempted walks even when one fails."""

    path = report_root / USAGE_FILENAME
    latest = _read_json(path)
    if latest is not None and latest.get("schema_version") != SURVEY_SCHEMA_VERSION:
        latest = None
    if latest is not None:
        safe_latest = sanitize_public_survey(latest)
        if safe_latest != latest:
            try:
                _write_public_json(path, safe_latest)
            except OSError:
                pass  # keep names out of this tick's projection even if repair fails
        latest = safe_latest
    if survey is None:
        return latest, None
    marker_path = report_root / USAGE_ATTEMPT_FILENAME
    marker = _read_json(marker_path)
    if marker is not None and marker.get("schema_version") != USAGE_ATTEMPT_SCHEMA_VERSION:
        marker = None
    observed_at = (latest or {}).get("observed_at_epoch")
    attempted_at = (marker or {}).get("attempted_at_epoch")
    if isinstance(attempted_at, (int, float)) and not isinstance(attempted_at, bool):
        if not isinstance(observed_at, (int, float)) or isinstance(observed_at, bool) or attempted_at > observed_at:
            observed_at = attempted_at
    fresh = (
        isinstance(observed_at, (int, float))
        and not isinstance(observed_at, bool)
        and 0 <= now - float(observed_at) < interval_seconds
    )
    if fresh and not force:
        error = None
        if marker is not None and marker.get("attempted_at_epoch") == observed_at:
            error = marker.get("error") if isinstance(marker.get("error"), str) else None
            if marker.get("status") == "running":
                error = "usage_survey_interrupted"
        return latest, error

    def record_attempt(status: str, error: str | None = None) -> None:
        document: dict[str, Any] = {
            "schema_version": USAGE_ATTEMPT_SCHEMA_VERSION,
            "attempted_at_epoch": now,
            "status": status,
        }
        if error:
            document["error"] = error
        _write_public_json(marker_path, document)

    try:
        report_root.mkdir(parents=True, exist_ok=True, mode=0o750)
        record_attempt("running")
    except OSError as exc:
        return latest, f"usage_survey_attempt_unwritten:{type(exc).__name__}"
    try:
        result = survey(mounts=survey_mounts(mounts))
    except Exception as exc:  # noqa: BLE001 - a failed survey must never stop the capacity tick
        error = f"usage_survey_failed:{type(exc).__name__}"
        try:
            record_attempt("failed", error)
        except OSError:
            pass  # the pre-walk marker still prevents an immediate retry
        return latest, error
    if not isinstance(result, Mapping) or result.get("schema_version") != SURVEY_SCHEMA_VERSION:
        error = "usage_survey_invalid"
        try:
            record_attempt("failed", error)
        except OSError:
            pass
        return latest, error
    result = sanitize_public_survey(result)
    try:
        _write_public_json(path, result)
    except OSError as exc:
        error = f"usage_survey_unwritten:{type(exc).__name__}"
        try:
            record_attempt("failed", error)
        except OSError:
            pass
        return dict(result), error
    try:
        record_attempt("complete")
    except OSError:
        pass  # the saved survey itself supplies the cadence
    return dict(result), None


def usage_projection(
    survey: Mapping[str, Any] | None, *, now: float, error: str | None = None
) -> dict[str, Any]:
    """The compact usage view embedded in the report and in the door-readable summary."""

    if survey is None:
        projection: dict[str, Any] = {"status": "unavailable"}
    else:
        observed_at = survey.get("observed_at_epoch")
        timed = isinstance(observed_at, (int, float)) and not isinstance(observed_at, bool)
        projection = {
            "observed_at_epoch": observed_at if timed else None,
            "age_seconds": round(max(0.0, now - float(observed_at)), 1) if timed else None,
            "status": survey.get("status"),
            "survey_digest": survey.get("survey_digest"),
            **{key: list(survey.get(key) or []) for key in ("mounts", "by_class", "top_roots", "top_owners")},
            "unclassified_roots": list(survey.get("unclassified_roots") or [])[
                :USAGE_PROJECTED_UNCLASSIFIED_ROOTS
            ],
            "orphan_scratch_bytes": survey.get("orphan_scratch_bytes"),
            "orphan_scratch_count": survey.get("orphan_scratch_count"),
            "orphan_scratch_largest_bytes": survey.get("orphan_scratch_largest_bytes"),
            "orphan_scratch_roots": list(survey.get("orphan_scratch_roots") or [])[
                :USAGE_PROJECTED_UNCLASSIFIED_ROOTS
            ],
            "container_runtime_bytes": survey.get("container_runtime_bytes"),
            "container_runtime_complete": survey.get("container_runtime_complete"),
            "container_runtime_roots": list(survey.get("container_runtime_roots") or []),
        }
    if error:
        projection["error"] = error
    return projection


def usage_alerts(survey: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Warnings and one owner page for large unregistered scratch."""

    def number(value: Any) -> bool:
        return isinstance(value, (int, float)) and not isinstance(value, bool)

    alerts: list[dict[str, Any]] = []
    total = survey.get("orphan_scratch_bytes")
    count = survey.get("orphan_scratch_count")
    largest = survey.get("orphan_scratch_largest_bytes")
    has_scratch_summary = all(type(value) is int and value >= 0 for value in (total, count, largest))
    if has_scratch_summary and (total > ORPHAN_SCRATCH_AGGREGATE_PAGE_BYTES
                                or largest > ORPHAN_SCRATCH_SINGLE_PAGE_BYTES):
        alerts.append({"code": "orphan_scratch_large", "allocated_bytes": total,
                       "count": count, "largest_bytes": largest})
    for row in survey.get("unclassified_roots") or []:
        size = row.get("allocated_bytes") if isinstance(row, Mapping) else None
        root = row.get("root") if isinstance(row, Mapping) else None
        if has_scratch_summary and isinstance(root, str) and is_orphan_scratch_root(root):
            continue  # one aggregate page; raw scratch folder names stay out of alerts
        if number(size) and size > USAGE_UNCLASSIFIED_ALERT_BYTES:
            alerts.append({"code": "usage_unclassified_root", "root": root, "allocated_bytes": size})
    for row in survey.get("container_runtime_roots") or []:
        size = row.get("allocated_bytes") if isinstance(row, Mapping) else None
        if number(size) and size > USAGE_CONTAINER_RUNTIME_ALERT_BYTES:
            alerts.append({"code": "usage_container_runtime_large", "root": row.get("root"),
                           "allocated_bytes": size})
    for row in survey.get("mounts") or []:
        fraction = row.get("attributed_fraction") if isinstance(row, Mapping) else None
        if number(fraction) and fraction < USAGE_ATTRIBUTION_ALERT_FRACTION:
            alerts.append({"mount": row.get("mount"), "code": "usage_attribution_low",
                           "attributed_fraction": fraction})
    return alerts


_SUMMARY_MOUNT_KEYS = (
    "mount", "status", "total_bytes", "free_bytes", "used_fraction", "floor_bytes",
    "reserved_bytes", "available_bytes", "refused_roles", "forecast", "level",
)
_SUMMARY_ALERT_KEYS = (
    "code", "mount", "provider", "roles", "used_fraction", "days_until_floor", "root",
    "allocated_bytes", "attributed_fraction", "severity", "reason", "status", "alert_count", "count",
    "largest_bytes",
    "top_retained_reasons",
    "hours_until_floor", "decline_bytes_per_hour", "window_seconds",
)


def capacity_summary(report: Mapping[str, Any], *, max_bytes: int = SUMMARY_MAX_BYTES) -> dict[str, Any]:
    """The secret-free projection of a report that the operator door reads.

    Named keys only: no project spend, no provider funding, no alert error text and
    no URLs. When the document would exceed ``max_bytes`` its longest list is halved
    until it fits, and ``truncated`` says so.
    """

    usage = report.get("usage")
    resize = report.get("volume_resize")
    summary: dict[str, Any] = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "observed_at_epoch": report.get("observed_at_epoch"),
        "level": report.get("level"),
        "report_digest": report.get("report_digest"),
        "alerts": [
            {key: alert[key] for key in _SUMMARY_ALERT_KEYS if key in alert}
            for alert in report.get("alerts") or []
            if isinstance(alert, Mapping)
        ],
        "mounts": [
            {key: row[key] for key in _SUMMARY_MOUNT_KEYS if key in row}
            for row in report.get("mounts") or []
            if isinstance(row, Mapping)
        ],
        "usage": dict(usage) if isinstance(usage, Mapping) else {"status": "unavailable"},
        "volume_resize": (
            {key: resize[key] for key in ("status", "reason") if key in resize}
            if isinstance(resize, Mapping)
            else None
        ),
        "reclaim_outlook": report.get("reclaim_outlook"),
    }

    def size() -> int:
        return len(json.dumps(summary, indent=2, sort_keys=True)) + 1

    if size() > max_bytes:
        summary["truncated"] = True
    while size() > max_bytes:
        usage_lists = [(summary["usage"], key) for key in
                       ("unclassified_roots", "top_owners", "top_roots", "by_class", "mounts")]
        candidates = [
            (len(json.dumps(holder[key])), holder, key)
            for holder, key in [(summary, "alerts"), (summary, "mounts"), *usage_lists]
            if isinstance(holder.get(key), list) and holder[key]
        ]
        if not candidates:
            break
        _size, holder, key = max(candidates, key=lambda candidate: candidate[0])
        holder[key] = holder[key][: len(holder[key]) // 2]
    return summary


def write_report(report_root: Path, report: Mapping[str, Any]) -> Path:
    report_root.mkdir(parents=True, exist_ok=True, mode=0o750)
    latest = report_root / "latest.json"
    temporary = report_root / f".latest-{os.getpid()}.tmp"
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, latest)
    # history.jsonl is never pruned: it keeps each tick's measurement, while the
    # forecast and the per-role footprints stay in latest.json.
    with (report_root / "history.jsonl").open("a", encoding="utf-8") as stream:
        for row in report.get("mounts") or []:
            kept = {k: v for k, v in row.items() if k not in {"forecast", "footprints"}}
            stream.write(json.dumps(kept, sort_keys=True) + "\n")
    # latest.json and history.jsonl stay root-only; the door (the service account)
    # reads the secret-free summary, so the directory itself becomes traversable.
    _write_public_json(report_root / SUMMARY_FILENAME, capacity_summary(report))
    try:
        os.chmod(report_root, 0o755)
    except PermissionError:
        # Deploy creates a missing sandbox directory as the service account, and the
        # unit holds no CAP_FOWNER. The door owns such a directory and reads the
        # summary anyway; if it cannot, its status names the error.
        pass
    return latest


def alert_due(previous: Mapping[str, Any] | None, report: Mapping[str, Any], *, now: float) -> bool:
    """Alert on escalation or a new affected target, and retry failed delivery."""

    level = report.get("level")
    has_page = any(row.get("severity") == "page" for row in report.get("alerts") or []
                   if isinstance(row, Mapping))
    if level == "ok" and not has_page:
        return False
    if previous is None or previous.get("level") != level:
        return True
    # alert_posted describes this tick, so a quiet tick after a successful
    # delivery must not turn the following tick into a retry.
    if previous.get("alert_error") or not isinstance(previous.get("last_alert_epoch"), (int, float)):
        return True
    if has_page and alert_fingerprint(report) != previous.get("last_alert_fingerprint", alert_fingerprint(previous)):
        return True

    def actionable_alerts(value: Mapping[str, Any]) -> set[tuple[str, str, str, str, tuple[str, ...]]]:
        return {
            (code, str(row.get("mount") or ""), str(row.get("provider") or ""),
             str(row.get("root") or ""),
             tuple(sorted(str(role) for role in row.get("roles") or [])))
            for row in value.get("alerts") or []
            if isinstance(row, Mapping)
            and isinstance((code := row.get("code")), str)
            and not code.startswith("usage_")
        }

    if actionable_alerts(report) - actionable_alerts(previous):
        return True
    last = previous.get("last_alert_epoch")
    return (has_page or level == "critical") and (
        not isinstance(last, (int, float)) or now - float(last) >= ALERT_REPEAT_SECONDS
    )


def post_alert(url: str, report: Mapping[str, Any], *, timeout_seconds: float = 10.0) -> None:
    page_alerts = [a for a in report.get("alerts") or [] if a.get("severity") == "page"]
    urgent = page_alerts or list(report.get("alerts") or [])
    first = urgent[0] if urgent else {}
    summary = f"{first.get('mount') or 'control plane'}: {first.get('code') or report.get('level')}"
    if first.get("code") == "floor_within_three_days":
        summary = f"{first.get('mount')}: floor in {float(first.get('days_until_floor') or 0):.1f} days"
    elif first.get("code") == "floor_within_hours":
        minutes = round(float(first.get("window_seconds") or 0) / 60)
        summary = (f"{first.get('mount')}: floor in {float(first.get('hours_until_floor') or 0):.1f} hours"
                   f" at the last {minutes} minutes' rate")
    elif first.get("code") == "volume_growth_blocked":
        summary = f"{first.get('mount')}: volume growth blocked ({first.get('reason')})"
    elif first.get("code") == "orphan_scratch_large":
        gib = float(first.get("allocated_bytes") or 0) / GIB
        summary = f"control plane: {gib:.1f} GiB unowned scratch in {first.get('count')} folders"
    ineffective = next(
        (row for row in page_alerts if row.get("code") == "reclaim_ineffective"), None
    )
    if ineffective and ineffective.get("top_retained_reasons"):
        summary += "; retained: " + ", ".join(ineffective["top_retained_reasons"])
    payload = {
        "schema_version": "control_plane_capacity_alert.v1",
        "level": report.get("level"),
        "severity": "page" if page_alerts else "warn",
        "page": bool(page_alerts),
        "fingerprint": alert_fingerprint(report),
        "runbook": "docs/runbooks/control-plane-capacity.md",
        "summary": summary[:200],
        "alerts": report.get("alerts"),
        "mounts": [
            {
                "mount": row.get("mount"),
                "free_gib": round(int(row.get("free_bytes") or 0) / GIB, 2),
                "used_fraction": row.get("used_fraction"),
                "refused_roles": row.get("refused_roles"),
                "forecast": row.get("forecast"),
            }
            for row in report.get("mounts") or []
        ],
        "text": "control-plane capacity "
        + str(report.get("level"))
        + ": "
        + "; ".join(f"{a.get('mount')} {a.get('code')}"
                    + (f" {float(a.get('hours_until_floor') or 0):.1f}h"
                       if a.get("code") == "floor_within_hours" else "")
                    for a in report.get("alerts") or []),
    }
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"}, method="POST"
    )
    if not url.startswith("https://"):
        raise ControlPlaneCapacityError("control_plane_capacity_webhook_not_https")
    with urllib.request.urlopen(  # nosec B310 - operator webhook, https-only, checked above
        request, timeout=timeout_seconds
    ) as response:
        status = int(getattr(response, "status", 0) or 0)
        if status < 200 or status >= 300:
            raise ControlPlaneCapacityError(f"control_plane_capacity_webhook_http_{status}")


def plan_volume_resize(
    report: Mapping[str, Any],
    *,
    volume_id: str,
    volume_mount: str,
    current_size_gib: int,
    max_gib: int,
    step_gib: int = DEFAULT_RESIZE_STEP_GIB,
) -> dict[str, Any] | None:
    """Grow one step when the volume's mount is critical and the maximum allows it."""

    if not volume_id or step_gib <= 0 or max_gib <= 0:
        return None
    row = next((r for r in report.get("mounts") or [] if r.get("mount") == volume_mount), None)
    if row is None or row.get("level") != "critical":
        return None
    target = min(current_size_gib + step_gib, max_gib)
    if target <= current_size_gib:
        return {"status": "blocked", "reason": "volume_at_maximum", "volume_id": volume_id, "current_size_gib": current_size_gib, "max_gib": max_gib}
    return {
        "status": "planned",
        "volume_id": volume_id,
        "mount": volume_mount,
        "current_size_gib": current_size_gib,
        "target_size_gib": target,
    }


def _do_request(url: str, *, token: str, method: str, payload: Mapping[str, Any] | None = None) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=None if payload is None else json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        method=method,
    )
    with urllib.request.urlopen(  # nosec B310 - fixed https://api.digitalocean.com origin
        request, timeout=30
    ) as response:
        return json.loads(response.read().decode("utf-8") or "{}")


def resize_volume(
    plan: Mapping[str, Any],
    *,
    ack: str,
    token: str,
    device: str,
    api: Callable[..., dict[str, Any]] = _do_request,
    runner: Callable[..., Any] = subprocess.run,
    now: float | None = None,
) -> dict[str, Any]:
    """Grow the block volume through the provider API, then the filesystem online."""

    if ack != RESIZE_ACK:
        raise ControlPlaneCapacityError("control_plane_capacity_resize_not_acknowledged")
    if plan.get("status") != "planned" or not token or not device.startswith("/dev/"):
        raise ControlPlaneCapacityError("control_plane_capacity_resize_plan_invalid")
    action = api(
        DO_VOLUME_ACTIONS_URL.format(volume_id=plan["volume_id"]),
        token=token,
        method="POST",
        payload={"type": "resize", "size_gigabytes": int(plan["target_size_gib"])},
    )
    status = str((action.get("action") or {}).get("status") or "")
    if status not in {"completed", "in-progress"}:
        raise ControlPlaneCapacityError(f"control_plane_capacity_resize_rejected:{status or 'unknown'}")
    completed = runner(["resize2fs", device], check=False, capture_output=True, text=True, timeout=600)
    if getattr(completed, "returncode", 1) != 0:
        raise ControlPlaneCapacityError("control_plane_capacity_filesystem_resize_failed")
    receipt: dict[str, Any] = {
        "schema_version": RESIZE_RECEIPT_SCHEMA_VERSION,
        "status": "applied",
        "volume_id": plan["volume_id"],
        "device": device,
        "from_size_gib": plan["current_size_gib"],
        "to_size_gib": plan["target_size_gib"],
        "provider_action_status": status,
        "resized_at_epoch": time.time() if now is None else float(now),
        "provider_mutation_performed": True,
        "receipt_digest": "",
    }
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    return receipt


def _env_int(name: str, default: int) -> int:
    raw = str(os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ControlPlaneCapacityError(f"control_plane_capacity_environment_int_invalid:{name}") from exc


def _read_secret(path_text: str) -> str:
    try:
        return Path(path_text).read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def run_controller(
    *,
    mounts: Sequence[str],
    report_root: Path,
    reservation_root: Path,
    webhook_url: str,
    volume: Mapping[str, Any] | None,
    ack: str,
    token: str,
    poster: Callable[..., None] = post_alert,
    resizer: Callable[..., dict[str, Any]] = resize_volume,
    disk_usage: Callable[[str | os.PathLike[str]], Any] = shutil.disk_usage,
    now: float | None = None,
    credit_collector: Callable[[], Mapping[str, Any]] | None = None,
    credit_warning_usd: float = 5.0,
    credit_reserve_usd: float = 1.0,
    survey: Callable[..., Mapping[str, Any]] | None | object = _DEFAULT_SURVEY,
    survey_interval_seconds: float = DEFAULT_SURVEY_INTERVAL_SECONDS,
    force_survey: bool = False,
    release_retirement_summary_path: Path = DEFAULT_RELEASE_RETIREMENT_SUMMARY,
    break_glass_notes_root: Path = DEFAULT_BREAK_GLASS_NOTES_ROOT,
    storage_gc_summary_path: Path = DEFAULT_STORAGE_GC_SUMMARY,
) -> dict[str, Any]:
    """One tick. By default, survey when stale or forced. Pass ``survey=None``
    only to reuse an existing report without scanning."""

    observed = time.time() if now is None else float(now)
    previous = _read_json(report_root / "latest.json")
    report = build_capacity_report(
        mounts=mounts,
        reservation_root=reservation_root,
        history=load_history(report_root),
        disk_usage=disk_usage,
        now=observed,
        burst_paging={
            str(alert.get("mount")) for alert in (previous or {}).get("alerts") or []
            if isinstance(alert, Mapping) and alert.get("code") == "floor_within_hours"
        },
    )
    from .task_evaluation_scene_spend import observe_configured_scene_project_spend
    try:
        if project_spend := observe_configured_scene_project_spend(now=observed):
            report["project_spend"] = project_spend
    except (OSError, ValueError, TypeError) as exc:
        report["level"] = "critical"
        reason = str(exc) if str(exc) in {
            "scene_spend_monitor_config_invalid", "scene_spend_monitor_seed_reference_invalid",
            "scene_spend_pointer_invalid_or_stale", "scene_spend_pointer_source_changed",
            "scene_spend_pointer_outside_output_root",
        } else "scene_spend_snapshot_unreadable_or_invalid"
        report["alerts"].append({"code": "project_spend_observation_blocked", "reason": reason})
    if credit_collector is not None:
        from .provider_credit_admission import credit_admission

        try:
            observation = credit_collector()
        except Exception:  # never include credential-bearing provider exceptions
            observation = {}
        # Judge freshness against a clock no earlier than the observation itself. ``observed``
        # is captured at pass start, before the disk measurement and the credit GET, so the
        # observation's own epoch is later; comparing to ``observed`` gave a negative age and
        # flagged every just-taken observation stale.  The upper bound still catches a genuinely
        # old observation (epoch far in the past -> age > maximum_age).
        credit_epoch = observation.get("observed_at_epoch")
        credit_now = max(observed, float(credit_epoch)) if isinstance(credit_epoch, (int, float)) and not isinstance(credit_epoch, bool) else observed
        funding = credit_admission(observation, required_usd=credit_warning_usd,
                                   reserve_usd=credit_reserve_usd, now=credit_now)
        report["provider_funding"] = funding
        if funding["blockers"]:
            report["level"] = "critical"
            report["alerts"].extend({"provider": "vast", "code": code}
                                    for code in funding["blockers"])
    if survey is _DEFAULT_SURVEY:
        survey = survey_usage
    usage, usage_error = _refresh_usage(
        report_root, survey=survey, mounts=mounts, interval_seconds=survey_interval_seconds,
        force=force_survey, now=observed,
    )
    if usage is not None or usage_error is not None:
        report["usage"] = usage_projection(usage, now=observed, error=usage_error)
    if usage is not None and (warnings := usage_alerts(usage)):
        report["alerts"].extend(warnings)
        if report["level"] == "ok":
            report["level"] = "warning"
    retirement = _read_attention_summary(release_retirement_summary_path)
    if retirement is not None:
        retirement_alerts = retirement.get("alerts")
        alert_count = len(retirement_alerts) if isinstance(retirement_alerts, list) else 0
        raw_status = retirement.get("status")
        retirement_status = raw_status if raw_status in ("applied", "blocked", "skipped") else "unreadable"
        if retirement_status != "applied" or alert_count:
            report["alerts"].append({"code": "release_retirement_attention",
                                     "status": retirement_status,
                                     "alert_count": alert_count})
            if report["level"] == "ok":
                report["level"] = "warning"
    from .control_plane_break_glass import unreported_notes
    try:
        unreported_count = len(unreported_notes(break_glass_notes_root))
    except (OSError, ValueError):
        report["alerts"].append({"code": "break_glass_notes_unreadable"})
        unreported_count = 0
    if unreported_count:
        report["alerts"].append({"code": "break_glass_notes_unreported", "count": unreported_count})
    if unreported_count or any(a["code"] == "break_glass_notes_unreadable" for a in report["alerts"]):
        if report["level"] == "ok":
            report["level"] = "warning"
    if not webhook_url:
        report["alerts"].append({"code": "operator_alert_route_unconfigured"})
        if report["level"] == "ok":
            report["level"] = "warning"
    if volume:
        plan = plan_volume_resize(
            report,
            volume_id=str(volume.get("id") or ""),
            volume_mount=str(volume.get("mount") or ""),
            current_size_gib=int(volume.get("current_size_gib") or 0),
            max_gib=int(volume.get("max_gib") or 0),
            step_gib=int(volume.get("step_gib") or DEFAULT_RESIZE_STEP_GIB),
        )
        report["volume_resize"] = plan or {"status": "not_needed"}
        if plan and plan.get("status") == "planned":
            if ack != RESIZE_ACK or not token:
                report["volume_resize"] = {**plan, "status": "blocked", "reason": "resize_not_acknowledged"}
            else:
                try:
                    report["volume_resize"] = resizer(
                        plan, ack=ack, token=token, device=str(volume.get("device") or ""), now=observed
                    )
                except ControlPlaneCapacityError as exc:
                    code = str(exc).split(":", 1)[0]
                    if code not in {
                        "control_plane_capacity_resize_rejected",
                        "control_plane_capacity_filesystem_resize_failed",
                        "control_plane_capacity_resize_not_acknowledged",
                        "control_plane_capacity_resize_plan_invalid",
                    }:
                        code = "control_plane_capacity_resize_failed"
                    report["volume_resize"] = {**plan, "status": "blocked", "reason": code}
        if report["volume_resize"].get("status") == "blocked":
            report["alerts"].append({"code": "volume_growth_blocked",
                                     "mount": str(volume.get("mount") or ""),
                                     "reason": report["volume_resize"].get("reason")})
    gc_summary = _read_attention_summary(storage_gc_summary_path, max_bytes=256 * 1024)
    growth = (report.get("volume_resize") or {}).get("status", "not_configured")
    outlook, retained_reasons, reclaim_ineffective = _reclaim_outlook(
        gc_summary, now=observed, volume_growth=growth,
    )
    report["reclaim_outlook"] = outlook
    if report["level"] == "critical" and reclaim_ineffective:
        report["alerts"].append({
            "code": "reclaim_ineffective",
            "top_retained_reasons": retained_reasons,
        })
    _annotate_alerts(report["alerts"])
    report["alert_posted"] = False
    if webhook_url and alert_due(previous, report, now=observed):
        try:
            poster(webhook_url, report)
            report["alert_posted"] = True
            report["last_alert_epoch"] = observed
            report["last_alert_fingerprint"] = alert_fingerprint(report)
        except Exception as exc:  # noqa: BLE001 - alerting must never stop measurement
            report["alert_error"] = f"{type(exc).__name__}: {exc}"[:200]
    elif previous is not None and isinstance(previous.get("last_alert_epoch"), (int, float)):
        report["last_alert_epoch"] = previous["last_alert_epoch"]
        report["last_alert_fingerprint"] = previous.get("last_alert_fingerprint", alert_fingerprint(previous))
    report["report_digest"] = ""
    report["report_digest"] = canonical_digest(report, digest_field="report_digest")
    write_report(report_root, report)
    return report


def _volume_from_environment() -> dict[str, Any] | None:
    volume_id = str(os.getenv(VOLUME_ID_ENV) or "").strip()
    if not volume_id:
        return None
    mount = str(os.getenv(VOLUME_MOUNT_ENV) or "").strip()
    device = str(os.getenv(VOLUME_DEVICE_ENV) or "").strip()
    try:
        current = shutil.disk_usage(mount).total // GIB if mount else 0
    except OSError:
        current = 0
    return {
        "id": volume_id,
        "mount": mount,
        "device": device,
        "current_size_gib": int(current),
        "max_gib": _env_int(VOLUME_MAX_GIB_ENV, 0),
        "step_gib": _env_int(VOLUME_STEP_GIB_ENV, DEFAULT_RESIZE_STEP_GIB),
    }


def main(argv: Sequence[str] | None = None) -> int:
    from .provider_credit_admission import (
        ENABLED_ENV, RESERVE_ENV, WARNING_ENV, observe_vast_credit,
    )

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--mount", action="append", default=None)
    parser.add_argument("--report-root", default=os.getenv(REPORT_ROOT_ENV) or str(DEFAULT_REPORT_ROOT))
    parser.add_argument(
        "--reservation-root", default=os.getenv(RESERVATION_ROOT_ENV) or str(DEFAULT_RESERVATION_ROOT)
    )
    parser.add_argument("--webhook-url", default=os.getenv(WEBHOOK_URL_ENV) or "")
    parser.add_argument("--print", action="store_true", dest="print_report")
    parser.add_argument("--survey", action="store_true",
                        help="survey disk usage now, whatever the age of the last survey")
    args = parser.parse_args(argv)
    mounts = args.mount or [item for item in str(os.getenv(MOUNTS_ENV) or "").split(":") if item] or list(DEFAULT_MOUNTS)
    report = run_controller(
        mounts=mounts,
        report_root=Path(args.report_root),
        reservation_root=Path(args.reservation_root),
        webhook_url=args.webhook_url,
        volume=_volume_from_environment(),
        ack=str(os.getenv(RESIZE_ACK_ENV) or "").strip(),
        token=_read_secret(str(os.getenv(DO_TOKEN_FILE_ENV) or "")) if os.getenv(VOLUME_ID_ENV) else "",
        credit_collector=(observe_vast_credit if os.getenv(ENABLED_ENV, "false").lower()
                          not in {"false", "0", ""} else None),
        credit_warning_usd=float(os.getenv(WARNING_ENV, "5")),
        credit_reserve_usd=float(os.getenv(RESERVE_ENV, "1")),
        survey=survey_usage,
        survey_interval_seconds=_env_int(SURVEY_INTERVAL_ENV, DEFAULT_SURVEY_INTERVAL_SECONDS),
        force_survey=args.survey,
    )
    if args.print_report:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(json.dumps({k: report[k] for k in ("level", "alerts", "report_digest")}, sort_keys=True))
    return 0


__all__ = [
    "CHAIN_ROLES",
    "CRITICAL_FRACTION",
    "RESIZE_ACK",
    "SCHEMA_VERSION",
    "SUMMARY_SCHEMA_VERSION",
    "WARNING_FRACTION",
    "ControlPlaneCapacityError",
    "alert_due",
    "build_capacity_report",
    "capacity_eta",
    "capacity_summary",
    "chain_footprints",
    "footprint_basis",
    "forecast",
    "live_reserved_bytes",
    "main",
    "measure_mount",
    "plan_volume_resize",
    "resize_volume",
    "run_controller",
    "survey_mounts",
    "usage_alerts",
    "usage_projection",
    "write_report",
]


if __name__ == "__main__":  # pragma: no cover - exercised through module CLI
    sys.exit(main())
