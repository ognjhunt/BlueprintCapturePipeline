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

It spends nothing unless the resize acknowledgement is set, and even then it
grows only the configured volume, only up to the configured maximum, only when
the volume's mount is critical.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .control_plane_disk_budget import (
    DEFAULT_FLOOR_BYTES,
    DEFAULT_FLOOR_FRACTION,
    DEFAULT_RESERVATION_ROOT,
    ROLE_FOOTPRINT_BYTES,
)
from .decision_evidence_contracts import canonical_digest

SCHEMA_VERSION = "control_plane_capacity_report.v1"
RESIZE_RECEIPT_SCHEMA_VERSION = "control_plane_volume_resize_receipt.v1"
DEFAULT_REPORT_ROOT = Path("/var/lib/blueprint/pipeline-control-plane/capacity")
DEFAULT_MOUNTS: tuple[str, ...] = ("/var/lib/blueprint",)
WARNING_FRACTION = 0.70
CRITICAL_FRACTION = 0.85
FORECAST_WINDOW_SECONDS = 7 * 24 * 60 * 60
ALERT_REPEAT_SECONDS = 60 * 60
RESIZE_ACK = "grow-control-plane-volume"
DEFAULT_RESIZE_STEP_GIB = 50
GIB = 1024**3

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
DO_TOKEN_FILE_ENV = "DIGITALOCEAN_API_TOKEN_FILE"
DO_VOLUME_ACTIONS_URL = "https://api.digitalocean.com/v2/volumes/{volume_id}/actions"

CHAIN_ROLES: tuple[str, ...] = (
    "launch_preparation",
    "episode_compilation",
    "launch_activation",
    "launch_dispatch",
    "policy_canary_dispatch",
)


class ControlPlaneCapacityError(RuntimeError):
    """The controller's configuration or a resize could not be trusted."""


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return loaded if isinstance(loaded, dict) else None


def live_reserved_bytes(reservation_root: Path, *, now: float) -> tuple[int, int]:
    """Sum the bytes of unexpired reservations exactly as admission counts them."""

    reserved = 0
    count = 0
    if not reservation_root.is_dir():
        return 0, 0
    for path in sorted(reservation_root.iterdir()):
        if path.name.startswith(".") or not path.is_file() or path.is_symlink():
            continue
        document = _read_json(path) or {}
        expires = document.get("expires_at_epoch")
        if isinstance(expires, (int, float)) and not isinstance(expires, bool) and expires < now:
            continue
        try:
            reserved += int(document.get("expected_bytes") or document.get("reserved_bytes") or 0)
        except (TypeError, ValueError):
            continue
        count += 1
    return reserved, count


def measure_mount(
    mount: str | Path,
    *,
    reservation_root: str | Path = DEFAULT_RESERVATION_ROOT,
    disk_usage: Callable[[str | os.PathLike[str]], Any] = shutil.disk_usage,
    now: float | None = None,
) -> dict[str, Any]:
    """One mount's admission projection, computed the way intake computes it."""

    observed = time.time() if now is None else float(now)
    path = Path(mount)
    try:
        usage = disk_usage(path)
    except OSError as exc:
        return {"mount": str(path), "status": "unreadable", "errno": exc.errno}
    floor = max(DEFAULT_FLOOR_BYTES, int(usage.total * DEFAULT_FLOOR_FRACTION))
    reserved, live = live_reserved_bytes(Path(reservation_root), now=observed)
    available = max(0, int(usage.free) - floor - reserved)
    refused = sorted(role for role in CHAIN_ROLES if ROLE_FOOTPRINT_BYTES[role] > available)
    used_fraction = 0.0 if not usage.total else (usage.total - usage.free) / usage.total
    if used_fraction >= CRITICAL_FRACTION or refused:
        level = "critical"
    elif used_fraction >= WARNING_FRACTION:
        level = "warning"
    else:
        level = "ok"
    return {
        "mount": str(path),
        "status": "measured",
        "total_bytes": int(usage.total),
        "free_bytes": int(usage.free),
        "used_fraction": round(used_fraction, 4),
        "floor_bytes": floor,
        "reserved_bytes": reserved,
        "live_reservations": live,
        "available_bytes": available,
        "refused_roles": refused,
        "free_needed_for_one_role_bytes": floor + ROLE_FOOTPRINT_BYTES["launch_preparation"],
        "free_needed_for_whole_chain_bytes": floor
        + sum(ROLE_FOOTPRINT_BYTES[role] for role in CHAIN_ROLES),
        "level": level,
    }


def forecast(history: Sequence[Mapping[str, Any]], current: Mapping[str, Any], *, now: float) -> dict[str, Any]:
    """Growth per day and days until the floor, from the oldest row inside the window."""

    mount = current.get("mount")
    rows = [
        row
        for row in history
        if isinstance(row, Mapping)
        and row.get("mount") == mount
        and row.get("status") == "measured"
        and isinstance(row.get("observed_at_epoch"), (int, float))
        and now - float(row["observed_at_epoch"]) <= FORECAST_WINDOW_SECONDS
    ]
    if not rows or current.get("status") != "measured":
        return {"status": "insufficient_history"}
    oldest = min(rows, key=lambda row: float(row["observed_at_epoch"]))
    elapsed = now - float(oldest["observed_at_epoch"])
    if elapsed < 3600:
        return {"status": "insufficient_history"}
    growth_per_day = (int(oldest["free_bytes"]) - int(current["free_bytes"])) / elapsed * 86400
    headroom = int(current["free_bytes"]) - int(current["floor_bytes"])
    if growth_per_day <= 0:
        return {"status": "not_growing", "growth_bytes_per_day": int(growth_per_day)}
    return {
        "status": "growing",
        "growth_bytes_per_day": int(growth_per_day),
        "days_until_floor": round(max(0.0, headroom / growth_per_day), 2),
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
) -> dict[str, Any]:
    observed = time.time() if now is None else float(now)
    measured = [
        measure_mount(mount, reservation_root=reservation_root, disk_usage=disk_usage, now=observed)
        for mount in mounts
    ]
    for row in measured:
        row["observed_at_epoch"] = observed
        row["forecast"] = forecast(history, row, now=observed)
    levels = [row.get("level", "critical") for row in measured]
    level = "critical" if "critical" in levels or any(r["status"] != "measured" for r in measured) else (
        "warning" if "warning" in levels else "ok"
    )
    alerts = []
    for row in measured:
        if row["status"] != "measured":
            alerts.append({"mount": row["mount"], "code": "mount_unreadable"})
            continue
        if row["refused_roles"]:
            alerts.append({"mount": row["mount"], "code": "admission_refused", "roles": row["refused_roles"]})
        if row["level"] != "ok":
            alerts.append({"mount": row["mount"], "code": f"utilization_{row['level']}", "used_fraction": row["used_fraction"]})
        days = row["forecast"].get("days_until_floor")
        if isinstance(days, (int, float)) and days < 3:
            alerts.append({"mount": row["mount"], "code": "floor_within_three_days", "days_until_floor": days})
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


def write_report(report_root: Path, report: Mapping[str, Any]) -> Path:
    report_root.mkdir(parents=True, exist_ok=True, mode=0o750)
    latest = report_root / "latest.json"
    temporary = report_root / f".latest-{os.getpid()}.tmp"
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, latest)
    with (report_root / "history.jsonl").open("a", encoding="utf-8") as stream:
        for row in report.get("mounts") or []:
            stream.write(json.dumps({k: v for k, v in row.items() if k != "forecast"}, sort_keys=True) + "\n")
    return latest


def alert_due(previous: Mapping[str, Any] | None, report: Mapping[str, Any], *, now: float) -> bool:
    """Alert on every escalation, and re-alert hourly while critical."""

    level = report.get("level")
    if level == "ok":
        return False
    if previous is None or previous.get("level") != level:
        return True
    last = previous.get("last_alert_epoch")
    return level == "critical" and (not isinstance(last, (int, float)) or now - float(last) >= ALERT_REPEAT_SECONDS)


def post_alert(url: str, report: Mapping[str, Any], *, timeout_seconds: float = 10.0) -> None:
    payload = {
        "schema_version": "control_plane_capacity_alert.v1",
        "level": report.get("level"),
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
        + "; ".join(f"{a.get('mount')} {a.get('code')}" for a in report.get("alerts") or []),
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
) -> dict[str, Any]:
    observed = time.time() if now is None else float(now)
    previous = _read_json(report_root / "latest.json")
    report = build_capacity_report(
        mounts=mounts,
        reservation_root=reservation_root,
        history=load_history(report_root),
        disk_usage=disk_usage,
        now=observed,
    )
    from .task_evaluation_scene_spend import refresh_configured_scene_project_spend
    try:
        if project_spend := refresh_configured_scene_project_spend():
            report["project_spend"] = project_spend
    except (OSError, ValueError, TypeError):
        report["level"] = "critical"
        report["alerts"].append({"code": "project_spend_refresh_blocked"})
    if credit_collector is not None:
        from .provider_credit_admission import credit_admission

        try:
            observation = credit_collector()
        except Exception:  # never include credential-bearing provider exceptions
            observation = {}
        funding = credit_admission(observation, required_usd=credit_warning_usd,
                                   reserve_usd=credit_reserve_usd, now=observed)
        report["provider_funding"] = funding
        if funding["blockers"]:
            report["level"] = "critical"
            report["alerts"].extend({"provider": "vast", "code": code}
                                    for code in funding["blockers"])
    report["alert_posted"] = False
    if webhook_url and alert_due(previous, report, now=observed):
        try:
            poster(webhook_url, report)
            report["alert_posted"] = True
            report["last_alert_epoch"] = observed
        except Exception as exc:  # noqa: BLE001 - alerting must never stop measurement
            report["alert_error"] = f"{type(exc).__name__}: {exc}"[:200]
    elif previous is not None and isinstance(previous.get("last_alert_epoch"), (int, float)):
        report["last_alert_epoch"] = previous["last_alert_epoch"]
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
                    report["volume_resize"] = {**plan, "status": "blocked", "reason": str(exc)}
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
    "WARNING_FRACTION",
    "ControlPlaneCapacityError",
    "alert_due",
    "build_capacity_report",
    "forecast",
    "live_reserved_bytes",
    "main",
    "measure_mount",
    "plan_volume_resize",
    "resize_volume",
    "run_controller",
    "write_report",
]


if __name__ == "__main__":  # pragma: no cover - exercised through module CLI
    sys.exit(main())
