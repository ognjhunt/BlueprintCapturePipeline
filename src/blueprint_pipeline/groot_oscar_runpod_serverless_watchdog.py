"""Independent hard-TTL teardown watchdog for RunPod Serverless resources."""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .common import write_json
from .gpu_render_providers import _runpod_call
from .paid_lane_guard import (
    TEARDOWN_STATUS_SOURCE_PROVIDER_API,
    build_teardown_proof,
    close_pending_teardown,
)
from .paid_provider_lane_lease import (
    restore_paid_provider_lane_lease_to_retained_watchdog,
)
from .production_gpu_campaign_budget import ProductionGpuCampaignBudget


SCHEMA_VERSION = "groot_oscar_runpod_serverless_watchdog.v1"
TERMINAL_REQUEST_NAME = "teardown.request.json"
TEARDOWN_PROOF_NAME = "serverless_teardown_proof.json"


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("serverless_watchdog_state_not_object")
    return dict(value)


def _key(path: Path) -> str:
    if path.is_symlink() or not path.is_file() or path.stat().st_mode & 0o077:
        raise ValueError("runpod_api_key_file_unsafe")
    value = path.read_text(encoding="utf-8").strip()
    if not value:
        raise ValueError("runpod_api_key_missing")
    return value


def _inventory(kind: str, *, key: str) -> tuple[int, list[dict[str, Any]]]:
    http, payload = _runpod_call("GET", f"/{kind}", None, key=key, timeout=15)
    rows = payload if isinstance(payload, list) else []
    return http, [dict(row) for row in rows if isinstance(row, Mapping)]


def teardown_matching_resources(
    *,
    resource_name_prefix: str,
    api_key: str,
    request_reason: str,
    clock: Any = time.time,
    sleeper: Any = time.sleep,
    convergence_attempts: int = 20,
) -> dict[str, Any]:
    started = float(clock())
    endpoints_http, endpoints = _inventory("endpoints", key=api_key)
    templates_http, templates = _inventory("templates", key=api_key)
    matching_endpoints = [
        row for row in endpoints if str(row.get("name") or "").startswith(resource_name_prefix)
    ]
    matching_templates = [
        row for row in templates if str(row.get("name") or "").startswith(resource_name_prefix)
    ]
    deletes: list[dict[str, Any]] = []
    if endpoints_http == 200:
        for row in matching_endpoints:
            resource_id = str(row.get("id") or "")
            http, _ = _runpod_call(
                "DELETE", f"/endpoints/{resource_id}", None, key=api_key, timeout=15
            )
            deletes.append({"kind": "endpoint", "id": resource_id, "http": http})
    endpoint_absent = False
    for attempt in range(max(1, convergence_attempts)):
        final_endpoints_http, final_endpoints = _inventory("endpoints", key=api_key)
        endpoint_absent = final_endpoints_http == 200 and not any(
            str(row.get("name") or "").startswith(resource_name_prefix)
            for row in final_endpoints
        )
        if endpoint_absent or final_endpoints_http != 200:
            break
        if attempt + 1 < convergence_attempts:
            sleeper(3)
    if templates_http == 200 and endpoint_absent:
        for row in matching_templates:
            resource_id = str(row.get("id") or "")
            http, _ = _runpod_call(
                "DELETE", f"/templates/{resource_id}", None, key=api_key, timeout=15
            )
            deletes.append({"kind": "template", "id": resource_id, "http": http})
    template_absent = False
    for attempt in range(max(1, convergence_attempts)):
        final_templates_http, final_templates = _inventory("templates", key=api_key)
        template_absent = final_templates_http == 200 and not any(
            str(row.get("name") or "").startswith(resource_name_prefix)
            for row in final_templates
        )
        if template_absent or final_templates_http != 200:
            break
        if attempt + 1 < convergence_attempts:
            sleeper(3)
    passed = endpoint_absent and template_absent
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if passed else "BLOCKED",
        "reason": request_reason,
        "resource_name_prefix": resource_name_prefix,
        "pre_teardown": {
            "endpoints_http": endpoints_http,
            "templates_http": templates_http,
            "matching_endpoint_count": len(matching_endpoints),
            "matching_template_count": len(matching_templates),
        },
        "deletes": deletes,
        "provider_absence": {
            "endpoints_http": final_endpoints_http,
            "templates_http": final_templates_http,
            "matching_endpoints_absent": endpoint_absent,
            "matching_templates_absent": template_absent,
            "billing_compute_stopped": endpoint_absent,
        },
        "elapsed_seconds": round(float(clock()) - started, 3),
        "raw_secret_values_recorded": False,
    }


def _settle_budget(state: Mapping[str, Any], proof: Mapping[str, Any]) -> dict[str, Any]:
    budget = state.get("campaign_budget")
    budget = dict(budget) if isinstance(budget, Mapping) else {}
    if not budget or proof.get("status") != "PASS":
        return {"status": "not_settled", "reason": "teardown_not_proven_or_budget_missing"}
    allocated_at = float(state.get("endpoint_allocated_at_epoch") or 0)
    measurement = "endpoint_wall_clock"
    if not allocated_at:
        pre_teardown = proof.get("pre_teardown")
        pre_teardown = (
            dict(pre_teardown) if isinstance(pre_teardown, Mapping) else {}
        )
        if int(pre_teardown.get("matching_endpoint_count") or 0) > 0:
            allocated_at = float(state.get("endpoint_create_requested_at_epoch") or 0)
            measurement = "endpoint_request_wall_clock_provider_presence_confirmed"
    ended_at = time.time()
    seconds = max(0, math.ceil(ended_at - allocated_at)) if allocated_at else 0
    reserved_seconds = int(budget.get("reserved_gpu_seconds") or 0)
    seconds = min(seconds, reserved_seconds)
    rate = float(budget.get("max_hourly_rate_usd") or 0)
    charged = round(seconds * rate / 3600.0, 6)
    ledger = ProductionGpuCampaignBudget(
        str(budget.get("ledger_path") or ""),
        initial_spent_usd=float(budget.get("initial_spent_usd") or 0),
        initial_used_gpu_seconds=int(budget.get("initial_used_gpu_seconds") or 0),
        total_spend_cap_usd=float(budget.get("total_spend_cap_usd") or 20.0),
        combined_gpu_wall_cap_seconds=int(
            budget.get("combined_gpu_wall_cap_seconds") or 21_000
        ),
    )
    settled = ledger.settle(
        reservation_id=str(budget.get("reservation_id") or ""),
        charged_gpu_seconds=seconds,
        charged_usd=charged,
        outcome=str(proof.get("reason") or "serverless_endpoint_torn_down"),
    )
    return {
        "status": "settled",
        "measurement": measurement,
        "selected_gpu_name": budget.get("selected_gpu_name"),
        "hourly_rate_usd": rate,
        "billing_rate_basis": budget.get("billing_rate_basis")
        or "runpod_public_active_worker_l40s_ceiling",
        **settled,
    }


def _close_pending(state: Mapping[str, Any], proof: Mapping[str, Any]) -> dict[str, Any]:
    path = str(state.get("pending_teardown_record") or "")
    endpoint_id = str(state.get("endpoint_id") or "")
    if not path or proof.get("status") != "PASS":
        return {"status": "not_closed"}
    teardown = build_teardown_proof(
        provider="runpod",
        allocation_id=endpoint_id or str(state.get("resource_name_prefix") or ""),
        terminate_requested=True,
        provider_terminal_status="not_found",
        verified_at=datetime.now(timezone.utc).isoformat(),
        status_source=TEARDOWN_STATUS_SOURCE_PROVIDER_API,
    )
    close_pending_teardown(path, teardown)
    return {"status": "closed", "teardown_proof": teardown}


def execute_teardown(state_path: Path, *, reason: str) -> dict[str, Any]:
    state = _read(state_path)
    proof = teardown_matching_resources(
        resource_name_prefix=str(state["resource_name_prefix"]),
        api_key=_key(Path(str(state["api_key_file"])).expanduser()),
        request_reason=reason,
    )
    proof["pending_teardown"] = _close_pending(state, proof)
    try:
        proof["campaign_budget_settlement"] = _settle_budget(state, proof)
    except Exception as exc:  # fail closed; open reservation retains worst case
        proof["campaign_budget_settlement"] = {
            "status": "blocked",
            "error_type": type(exc).__name__,
        }
        proof["status"] = "BLOCKED"
    receipt = state.get("provider_lane_handoff_acceptance")
    if isinstance(receipt, Mapping) and proof.get("provider_absence", {}).get(
        "billing_compute_stopped"
    ) is True:
        proof["provider_lane_restore"] = (
            restore_paid_provider_lane_lease_to_retained_watchdog(receipt)
        )
    else:
        proof["provider_lane_restore"] = {"status": "not_attempted"}
    write_json(state_path.parent / TEARDOWN_PROOF_NAME, proof)
    return proof


def watchdog_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", required=True)
    parser.add_argument("--resource-name-prefix", required=True)
    parser.add_argument("--deadline-epoch", required=True, type=float)
    args = parser.parse_args(argv)
    state_path = Path(args.state).expanduser().resolve()
    out_dir = state_path.parent
    write_json(
        out_dir / "watchdog_armed.json",
        {
            "schema_version": SCHEMA_VERSION,
            "status": "armed",
            "pid": os.getpid(),
            "resource_name_prefix": args.resource_name_prefix,
            "deadline_epoch": args.deadline_epoch,
            "raw_secret_values_recorded": False,
        },
    )
    terminate = False

    def requested(_signum: int, _frame: Any) -> None:
        nonlocal terminate
        terminate = True

    signal.signal(signal.SIGTERM, requested)
    signal.signal(signal.SIGINT, requested)
    while True:
        request = out_dir / TERMINAL_REQUEST_NAME
        now = time.time()
        if terminate or request.is_file() or now >= args.deadline_epoch:
            reason = (
                "signal_requested"
                if terminate
                else "synchronous_teardown_requested"
                if request.is_file()
                else "hard_ttl_reached"
            )
            proof = execute_teardown(state_path, reason=reason)
            return 0 if proof.get("status") == "PASS" else 2
        time.sleep(min(5.0, max(0.1, args.deadline_epoch - now)))


def main(argv: Sequence[str] | None = None) -> int:
    return watchdog_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
