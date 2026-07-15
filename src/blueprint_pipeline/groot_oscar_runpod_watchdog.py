"""Independent name-bound hard-TTL watchdog for the GR00T + OSCAR canary."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .common import utc_now_iso, write_json
from .gpu_render_providers import get_render_provider

SCHEMA_VERSION = "groot_oscar_runpod_canary_watchdog.v1"
EVIDENCE_NAME = "groot_oscar_runpod_canary_watchdog.json"


def arm_watchdog(
    *, out_dir: str | Path, pod_name_prefix: str, deadline_epoch: float, pid: int | None = None
) -> dict[str, Any]:
    root = Path(out_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    if not pod_name_prefix.startswith("blueprint-groot-oscar-canary-"):
        raise ValueError("watchdog_pod_name_prefix_not_canary_scoped")
    if float(deadline_epoch) <= time.time() + 60:
        raise ValueError("watchdog_deadline_must_be_more_than_60_seconds_future")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "armed",
        "independent_process": True,
        "pid": int(pid if pid is not None else os.getpid()),
        "armed_at": utc_now_iso(),
        "deadline_epoch": float(deadline_epoch),
        "pod_name_prefix": pod_name_prefix,
        "watchdog_out_dir": str(root),
        "provider_mutations_performed": 0,
        "raw_secret_values_recorded": False,
    }
    write_json(root / EVIDENCE_NAME, payload)
    return payload


def terminate_canary_resources(
    *, provider: Any, pod_name_prefix: str, armed: Mapping[str, Any]
) -> dict[str, Any]:
    try:
        inventory = provider.billable_inventory(name_prefix=pod_name_prefix)
    except Exception as exc:  # noqa: BLE001 - persist terminal watchdog uncertainty
        return {
            **dict(armed),
            "status": "teardown_unverified",
            "completed_at": utc_now_iso(),
            "initial_inventory": {"api_confirmed": False},
            "terminations": [],
            "final_inventory": {"api_confirmed": False},
            "provider_absence_confirmed": False,
            "provider_mutations_performed": 0,
            "teardown_error_type": type(exc).__name__,
            "raw_secret_values_recorded": False,
        }
    resources = inventory.get("resources")
    resources = resources if isinstance(resources, list) else []
    terminations: list[dict[str, Any]] = []
    for row in resources:
        row = row if isinstance(row, Mapping) else {}
        instance_id = str(row.get("instance_id") or row.get("id") or "").strip()
        if not instance_id:
            terminations.append({"status": "blocked", "reason": "resource_id_missing"})
            continue
        try:
            result = provider.terminate(instance_id)
        except Exception as exc:  # noqa: BLE001 - continue bounded name-scope cleanup
            result = {
                "status": "teardown_unverified",
                "error_type": type(exc).__name__,
            }
        terminations.append({"instance_id": instance_id, **dict(result)})
    try:
        final_inventory = provider.billable_inventory(name_prefix=pod_name_prefix)
        final_error_type = None
    except Exception as exc:  # noqa: BLE001 - persist terminal watchdog uncertainty
        final_inventory = {"api_confirmed": False}
        final_error_type = type(exc).__name__
    absent = bool(
        final_inventory.get("api_confirmed") is True
        and final_inventory.get("live_resource_count") == 0
    )
    return {
        **dict(armed),
        "status": "provider_terminal" if absent else "teardown_unverified",
        "completed_at": utc_now_iso(),
        "initial_inventory": inventory,
        "terminations": terminations,
        "final_inventory": final_inventory,
        "provider_absence_confirmed": absent,
        "provider_mutations_performed": len(terminations),
        "teardown_error_type": final_error_type,
        "raw_secret_values_recorded": False,
    }


def run_watchdog(
    *,
    out_dir: str | Path,
    pod_name_prefix: str,
    deadline_epoch: float,
    provider_factory: Callable[[str], Any] = get_render_provider,
    clock: Callable[[], float] = time.time,
    sleeper: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    root = Path(out_dir).expanduser().resolve()
    armed = arm_watchdog(
        out_dir=root,
        pod_name_prefix=pod_name_prefix,
        deadline_epoch=deadline_epoch,
    )
    while clock() < deadline_epoch:
        sleeper(min(10.0, max(0.0, deadline_epoch - clock())))
    try:
        provider = provider_factory("runpod")
    except Exception as exc:  # noqa: BLE001 - evidence must survive provider init failure
        result = {
            **dict(armed),
            "status": "teardown_unverified",
            "completed_at": utc_now_iso(),
            "provider_absence_confirmed": False,
            "provider_mutations_performed": 0,
            "teardown_error_type": type(exc).__name__,
            "raw_secret_values_recorded": False,
        }
    else:
        result = terminate_canary_resources(
            provider=provider,
            pod_name_prefix=pod_name_prefix,
            armed=armed,
        )
    receipt_path = root / "provider_lane_handoff_receipt.json"
    receipt_control_required = receipt_path.exists() or receipt_path.is_symlink()
    receipt_safe = False
    if receipt_control_required:
        try:
            receipt_stat = receipt_path.lstat()
            receipt_safe = bool(
                receipt_path.is_file()
                and not receipt_path.is_symlink()
                and not receipt_stat.st_mode & 0o077
            )
        except OSError:
            receipt_safe = False
    if receipt_safe and result.get("provider_absence_confirmed") is True:
        try:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            receipt = {}
        if (
            isinstance(receipt, Mapping)
            and receipt.get("pod_name_prefix") == pod_name_prefix
        ):
            from .paid_lane_guard import (
                cancel_pending_teardown,
                close_pending_teardown,
            )
            from .paid_provider_lane_lease import (
                restore_paid_provider_lane_lease_to_retained_watchdog,
            )

            pending_path = str(receipt.get("pod_pending_teardown_record") or "")
            try:
                pending_record = json.loads(Path(pending_path).read_text(encoding="utf-8"))
            except (OSError, ValueError, json.JSONDecodeError):
                pending_record = {}
            pending_valid = bool(
                isinstance(pending_record, Mapping)
                and pending_record.get("status") == "open"
                and pending_record.get("provider") == "runpod"
                and pending_record.get("lane") == "groot_oscar_gpu_canary"
                and pending_record.get("resource_kind") == "compute_instance"
                and str(pending_record.get("resource_name") or "").startswith(
                    pod_name_prefix
                )
            )
            if pending_valid and receipt.get("pod_id"):
                pending_close = close_pending_teardown(
                    pending_path,
                    {
                        "status": "PASS",
                        "provider_absence_confirmed": True,
                        "instance_id": receipt.get("pod_id"),
                    },
                )
            elif pending_valid:
                pending_close = cancel_pending_teardown(
                    pending_path,
                    reason="canary_watchdog_provider_inventory_verified_zero",
                    evidence={"provider_absence_confirmed": True},
                )
            else:
                pending_close = {"status": "invalid"}
            result["pod_pending_teardown_close"] = pending_close
            result["provider_lane_owner_return"] = (
                restore_paid_provider_lane_lease_to_retained_watchdog(receipt)
            )
    if receipt_control_required:
        control_terminal = bool(
            receipt_safe
            and result.get("pod_pending_teardown_close", {}).get("status")
            in {"closed", "cancelled_no_allocation"}
            and result.get("provider_lane_owner_return", {}).get("status")
            in {"restored", "already_released"}
        )
        if not control_terminal:
            result["status"] = "provider_terminal_control_plane_open"
            result["control_plane_terminal"] = False
        else:
            result["control_plane_terminal"] = True
    write_json(root / EVIDENCE_NAME, result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--pod-name-prefix", required=True)
    parser.add_argument("--deadline-epoch", type=float, required=True)
    args = parser.parse_args(argv)
    result = run_watchdog(
        out_dir=args.out_dir,
        pod_name_prefix=args.pod_name_prefix,
        deadline_epoch=args.deadline_epoch,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "provider_terminal" else 2


if __name__ == "__main__":
    raise SystemExit(main())
