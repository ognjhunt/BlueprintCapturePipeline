"""Independent hard-TTL watchdog for a paid production warm GPU rehearsal."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence, cast

from .common import utc_now_iso, write_json
from .gpu_render_providers import get_render_provider
from .paid_lane_guard import close_pending_teardown, load_pending_teardowns
from .paid_provider_allocation_lifecycle import teardown_proof_from_attempt
from .production_gpu_worker_agent import _post_json, _read_token
from .production_gpu_campaign_budget import ProductionGpuCampaignBudget

SCHEMA_VERSION = "production_gpu_warm_watchdog.v1"
EVIDENCE_FILENAME = "production_gpu_warm_watchdog.json"
MARKER_FILENAME = "warm_serve_pod.json"
CANCEL_FILENAME = "production_gpu_warm_watchdog.cancel"


def _evidence_path(out_dir: str | Path) -> Path:
    return Path(out_dir).expanduser().resolve() / EVIDENCE_FILENAME


def _marker_path(out_dir: str | Path) -> Path:
    return Path(out_dir).expanduser().resolve() / MARKER_FILENAME


def _read_mapping(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}
    return dict(value) if isinstance(value, Mapping) else {}


def _discover_started_id(out_dir: Path) -> str:
    ids: set[str] = set()
    for path in out_dir.glob("**/started_pod_id.txt"):
        try:
            value = path.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if value:
            ids.add(value)
    return next(iter(ids)) if len(ids) == 1 else ""


def arm_watchdog_evidence(
    *,
    out_dir: str | Path,
    deadline_epoch: float,
    pid: int | None = None,
    campaign_budget_ledger: str | Path | None = None,
    campaign_reservation_id: str | None = None,
    pool_base_url: str | None = None,
    pool_token_file: str | Path | None = None,
    clock: Callable[[], float] = time.time,
) -> dict[str, Any]:
    root = Path(out_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    deadline = float(deadline_epoch)
    if deadline <= clock() + 60:
        raise ValueError("watchdog_deadline_must_be_more_than_60_seconds_future")
    path = _evidence_path(root)
    armed_epoch = clock()
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "armed",
        "independent_process": True,
        "pid": int(pid if pid is not None else os.getpid()),
        "armed_at": utc_now_iso(),
        "armed_at_epoch": armed_epoch,
        "deadline_epoch": deadline,
        "evidence_path": str(path),
        "provider": "runpod",
        "provider_mutations_performed": 0,
        "campaign_budget_ledger": (
            str(Path(campaign_budget_ledger).expanduser().resolve())
            if campaign_budget_ledger
            else None
        ),
        "campaign_reservation_id": campaign_reservation_id,
        "pool_base_url": pool_base_url,
        "pool_token_file": (
            str(Path(pool_token_file).expanduser().resolve()) if pool_token_file else None
        ),
    }
    write_json(path, payload)
    return payload


def _close_matching_pending(instance_id: str, proof: Mapping[str, Any]) -> list[dict[str, Any]]:
    closures: list[dict[str, Any]] = []
    for record in load_pending_teardowns():
        if (
            record.get("status") == "open"
            and str(record.get("provider") or "") == "runpod"
            and str(record.get("instance_id") or "") == instance_id
            and record.get("path")
        ):
            closures.append(close_pending_teardown(str(record["path"]), proof))
    return closures


def _settle_campaign_budget(
    evidence: Mapping[str, Any], *, charged_gpu_seconds: int, outcome: str
) -> dict[str, Any]:
    ledger_path = str(evidence.get("campaign_budget_ledger") or "").strip()
    reservation_id = str(evidence.get("campaign_reservation_id") or "").strip()
    if not ledger_path or not reservation_id:
        return {"status": "not_configured"}
    path = Path(ledger_path).expanduser().resolve()
    ledger_state = _read_mapping(path)
    try:
        ledger = ProductionGpuCampaignBudget(
            path,
            initial_spent_usd=float(cast(Any, ledger_state.get("initial_spent_usd"))),
            initial_used_gpu_seconds=int(
                cast(Any, ledger_state.get("initial_used_gpu_seconds"))
            ),
            total_spend_cap_usd=float(
                cast(Any, ledger_state.get("total_spend_cap_usd"))
            ),
            combined_gpu_wall_cap_seconds=int(
                cast(Any, ledger_state.get("combined_gpu_wall_cap_seconds"))
            ),
        )
        reservation = next(
            row
            for row in ledger.snapshot()["reservations"]
            if row.get("reservation_id") == reservation_id
        )
        seconds = min(
            max(0, int(charged_gpu_seconds)),
            int(reservation.get("reserved_gpu_seconds") or 0),
        )
        rate = float(cast(Any, reservation.get("max_hourly_rate_usd") or 0.0))
        settled = ledger.settle(
            reservation_id=reservation_id,
            charged_gpu_seconds=seconds,
            charged_usd=round(rate * seconds / 3600.0, 6),
            outcome=outcome,
        )
        return {"status": "settled", "reservation": settled}
    except (KeyError, StopIteration, TypeError, ValueError) as exc:
        return {"status": "blocked", "error_type": type(exc).__name__}


def _elapsed_campaign_seconds(evidence: Mapping[str, Any]) -> int:
    try:
        return max(
            0,
            math.ceil(
                time.time() - float(cast(Any, evidence.get("armed_at_epoch")))
            ),
        )
    except (TypeError, ValueError):
        return 0


def terminate_at_watchdog_boundary(
    *,
    out_dir: str | Path,
    provider_factory: Callable[[str], Any] = get_render_provider,
    pool_sender: Callable[[str, str, Mapping[str, Any], str], dict[str, Any]] = _post_json,
) -> dict[str, Any]:
    root = Path(out_dir).expanduser().resolve()
    evidence = _read_mapping(_evidence_path(root))
    marker = _read_mapping(_marker_path(root))
    instance_id = str(marker.get("pod_id") or "").strip() or _discover_started_id(root)
    provider = provider_factory("runpod")
    pool_quarantine: dict[str, Any] = {"status": "not_configured"}
    pool_base_url = str(evidence.get("pool_base_url") or "").strip()
    pool_token_file = str(evidence.get("pool_token_file") or "").strip()
    if instance_id and pool_base_url and pool_token_file:
        try:
            pool_quarantine = pool_sender(
                pool_base_url,
                f"/v1/workers/{instance_id}/quarantine",
                {"reason": "watchdog_provider_teardown"},
                _read_token(pool_token_file),
            )
            pool_quarantine["status"] = (
                "quarantined"
                if pool_quarantine.get("state") == "quarantined"
                else "blocked"
            )
        except Exception as exc:  # noqa: BLE001
            pool_quarantine = {"status": "blocked", "error_type": type(exc).__name__}
    if not instance_id:
        inventory = provider.billable_inventory(name_prefix="blueprint-")
        absent = bool(
            inventory.get("api_confirmed") is True
            and inventory.get("live_resource_count") == 0
        )
        result = {
            **evidence,
            "status": "closed_no_allocation" if absent else "blocked_inventory_not_empty",
            "completed_at": utc_now_iso(),
            "instance_id": None,
            "provider_inventory": inventory,
            "pool_quarantine": pool_quarantine,
            "api_confirmed_absent": absent,
        }
        if absent:
            result["campaign_budget_settlement"] = _settle_campaign_budget(
                evidence, charged_gpu_seconds=0, outcome="provider_confirmed_no_allocation"
            )
        write_json(_evidence_path(root), result)
        return result
    try:
        teardown = provider.terminate(instance_id)
    except Exception as exc:  # noqa: BLE001
        teardown = {"status": "terminate_failed", "error_type": type(exc).__name__}
    proof = teardown_proof_from_attempt(
        provider=provider,
        instance_id=instance_id,
        teardown=teardown,
        action="terminate",
    )
    inventory = provider.billable_inventory(name_prefix="blueprint-")
    absent = bool(
        inventory.get("api_confirmed") is True
        and inventory.get("live_resource_count") == 0
    )
    closures = _close_matching_pending(instance_id, proof)
    passed = bool(
        str(proof.get("status") or "").upper() == "PASS"
        and absent
        and all(row.get("status") == "closed" for row in closures)
        and pool_quarantine.get("status") in {"quarantined", "not_configured"}
    )
    result = {
        **evidence,
        "status": "PASS" if passed else "blocked_teardown_incomplete",
        "completed_at": utc_now_iso(),
        "instance_id": instance_id,
        "provider_mutations_performed": 1,
        "teardown": teardown,
        "teardown_proof": proof,
        "provider_inventory": inventory,
        "api_confirmed_absent": absent,
        "pending_teardown_closures": closures,
        "pool_quarantine": pool_quarantine,
    }
    if passed:
        result["campaign_budget_settlement"] = _settle_campaign_budget(
            evidence,
            charged_gpu_seconds=_elapsed_campaign_seconds(evidence),
            outcome="watchdog_teardown_and_absence_proven",
        )
    write_json(_evidence_path(root), result)
    if marker:
        marker["status"] = "terminated" if passed else "teardown_blocked"
        marker["watchdog_evidence_path"] = str(_evidence_path(root))
        marker["terminated_at"] = result["completed_at"] if passed else None
        write_json(_marker_path(root), marker)
    return result


def run_watchdog(
    *,
    out_dir: str | Path,
    deadline_epoch: float,
    poll_interval_seconds: float = 10.0,
    clock: Callable[[], float] = time.time,
    sleeper: Callable[[float], None] = time.sleep,
    provider_factory: Callable[[str], Any] = get_render_provider,
    campaign_budget_ledger: str | Path | None = None,
    campaign_reservation_id: str | None = None,
    pool_base_url: str | None = None,
    pool_token_file: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(out_dir).expanduser().resolve()
    interval = float(poll_interval_seconds)
    if not 1 <= interval <= 60:
        raise ValueError("watchdog_poll_interval_out_of_range")
    armed = arm_watchdog_evidence(
        out_dir=root,
        deadline_epoch=deadline_epoch,
        pid=os.getpid(),
        campaign_budget_ledger=campaign_budget_ledger,
        campaign_reservation_id=campaign_reservation_id,
        pool_base_url=pool_base_url,
        pool_token_file=pool_token_file,
        clock=clock,
    )
    while clock() < float(deadline_epoch):
        if (root / CANCEL_FILENAME).is_file():
            return terminate_at_watchdog_boundary(
                out_dir=root, provider_factory=provider_factory
            )
        marker_path = _marker_path(root)
        marker = _read_mapping(marker_path)
        if marker.get("status") in {"terminated", "teardown_blocked"}:
            result = {**armed, "status": "owner_teardown_observed", "completed_at": utc_now_iso()}
            write_json(_evidence_path(root), result)
            return result
        if marker.get("status") == "serving":
            marker["heartbeat_at"] = utc_now_iso()
            marker["lease_expires_at"] = time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime(float(deadline_epoch))
            )
            marker["watchdog_evidence_path"] = str(_evidence_path(root))
            write_json(marker_path, marker)
        sleeper(min(interval, max(0.0, float(deadline_epoch) - clock())))
    return terminate_at_watchdog_boundary(
        out_dir=root, provider_factory=provider_factory
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--deadline-epoch", type=float, required=True)
    parser.add_argument("--poll-interval-seconds", type=float, default=10)
    parser.add_argument("--campaign-budget-ledger", default=None)
    parser.add_argument("--campaign-reservation-id", default=None)
    parser.add_argument("--pool-base-url", default=None)
    parser.add_argument("--pool-token-file", default=None)
    args = parser.parse_args(argv)
    result = run_watchdog(
        out_dir=args.out_dir,
        deadline_epoch=args.deadline_epoch,
        poll_interval_seconds=args.poll_interval_seconds,
        campaign_budget_ledger=args.campaign_budget_ledger,
        campaign_reservation_id=args.campaign_reservation_id,
        pool_base_url=args.pool_base_url,
        pool_token_file=args.pool_token_file,
    )
    print(json.dumps(result, sort_keys=True))
    return 0 if result.get("status") in {
        "PASS", "closed_no_allocation", "cancelled_before_allocation", "owner_teardown_observed"
    } else 1


if __name__ == "__main__":
    raise SystemExit(main())
