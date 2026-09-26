"""Consume one accepted G1 team intent through the canonical paid allocator.

The queue is single attempt. A durable execution-start record is written before
the paid allocator call, so a stopped dispatcher never silently launches again.
The allocator owns spend admission, provider locks, watchdog, and teardown.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import cross_runtime_canonical_digest as digest
from .native_g1_team_campaign_intake import INTENT_SCHEMA, _read
from .native_g1_team_campaign_preparation import (
    _verified_intent,
    prepare_g1_team_campaign,
)
from .task_evaluation_launch_preparation_queue import (
    _write_launch_preparation_record_exclusive_locked as write_exclusive,
)


DRY_SCHEMA = "native_g1_team_campaign_dry_run.v1"
START_SCHEMA = "native_g1_team_campaign_execution_start.v1"
FINAL_SCHEMA = "native_g1_team_campaign_dispatch_result.v1"
AllocatorRunner = Callable[[list[str], Path], int]


def _run_allocator(command: list[str], log_path: Path) -> int:
    with log_path.open("x", encoding="utf-8") as stream:
        result = subprocess.run(
            command, stdout=stream, stderr=subprocess.STDOUT,
            check=False, timeout=5 * 3600,
        )
    return result.returncode


def _allocator_command(
    *, receipt_path: Path, run_root: Path, authorization: dict[str, Any],
    machine_avoidlist_path: Path | None, execute: bool,
) -> list[str]:
    command = [
        sys.executable, "-m", "blueprint_pipeline.paid_resource_allocator",
        "gpu-canary", "--provider", "vast", "--probe-kind",
        "native-g1-development-campaign", "--g1-campaign-bundle-receipt",
        str(receipt_path), "--adp-job-dir", str(run_root),
        "--adp-max-hourly-rate-usd", "3.0",
        "--adp-max-spend-usd", str(authorization["maximum_cost_usd"]),
        "--adp-hard-ttl-seconds", str(authorization["hard_ttl_seconds"]),
        "--admission-out", str(run_root / ("admission_paid.json" if execute else "admission_dry.json")),
        "--adapter-output", str(run_root / ("adapter_paid.json" if execute else "adapter_dry.json")),
    ]
    if machine_avoidlist_path is not None:
        command.extend(["--adp-machine-avoidlist", str(machine_avoidlist_path)])
    if execute:
        command.append("--execute")
    return command


def _sealed(path: Path, field: str) -> dict[str, Any]:
    return _read(path, field=field)


def _dispatch_one_locked(
    *, intent_path: Path, registry_path: Path, work_root: Path,
    implementation_commit: str, machine_avoidlist_path: Path | None,
    execute: bool, allocator_runner: AllocatorRunner,
) -> dict[str, Any]:
    intent, _ = _verified_intent(intent_path, registry_path)
    prepared = prepare_g1_team_campaign(
        intent_path=intent_path, registry_path=registry_path,
        work_root=work_root, implementation_commit=implementation_commit,
    )
    directory = work_root / intent["intent_id"]
    final_path = directory / "dispatch_final.json"
    if final_path.exists() or final_path.is_symlink():
        final = _sealed(final_path, "dispatch_digest")
        if final.get("intent_digest") != intent["intent_digest"]:
            raise ValueError("g1_team_campaign_dispatch_conflict")
        return final
    start_path = directory / "execution_started.json"
    if start_path.exists() or start_path.is_symlink():
        started = _sealed(start_path, "start_digest")
        if started.get("intent_digest") != intent["intent_digest"]:
            raise ValueError("g1_team_campaign_dispatch_conflict")
        return {
            "schema_version": FINAL_SCHEMA,
            "status": "execution_already_started_reconcile_exact_attempt",
            "intent_id": intent["intent_id"],
            "intent_digest": intent["intent_digest"],
            "provider_mutation_unproven": True,
        }
    run_root = directory / "run"
    if run_root.is_symlink():
        raise ValueError("g1_team_campaign_run_root_unsafe")
    run_root.mkdir(mode=0o750, exist_ok=True)
    dry_path = directory / "dry_run.json"
    if dry_path.exists() or dry_path.is_symlink():
        dry = _sealed(dry_path, "dry_run_digest")
        if (
            dry.get("preparation_digest") != prepared["preparation_digest"]
            or dry.get("status") != "dry_run_ready"
        ):
            raise ValueError("g1_team_campaign_dry_run_changed")
    else:
        command = _allocator_command(
            receipt_path=Path(prepared["bundle_receipt_path"]), run_root=run_root,
            authorization=prepared["authorization"],
            machine_avoidlist_path=machine_avoidlist_path, execute=False,
        )
        code = allocator_runner(command, run_root / f"allocator_dry_{time.time_ns()}.log")
        adapter_path = run_root / "adapter_dry.json"
        adapter = json.loads(adapter_path.read_text(encoding="utf-8")) if adapter_path.is_file() else {}
        if code != 0 or adapter.get("status") != "dry_run_ready":
            return {
                "schema_version": FINAL_SCHEMA,
                "status": "blocked_before_provider",
                "intent_id": intent["intent_id"],
                "blockers": adapter.get("blockers") or ["g1_team_campaign_dry_run_failed"],
                "provider_mutation_performed": False,
            }
        dry = {
            "schema_version": DRY_SCHEMA,
            "status": "dry_run_ready",
            "intent_id": intent["intent_id"],
            "intent_digest": intent["intent_digest"],
            "preparation_digest": prepared["preparation_digest"],
            "provider_mutation_performed": False,
        }
        dry["dry_run_digest"] = digest(dry, digest_field="dry_run_digest")
        write_exclusive(dry_path, dry)
    if not execute:
        return dry
    # Intake expiry, registry, packet binding and rights are reopened after the
    # dry run. The canonical allocator rechecks exact release and bundle again
    # immediately before a provider create call.
    intent, _ = _verified_intent(intent_path, registry_path)
    if time.time() >= intent["request"]["authorization"]["expires_at_epoch"]:
        raise ValueError("g1_team_campaign_authority_expired_before_execute")
    start = {
        "schema_version": START_SCHEMA,
        "status": "execution_started_once",
        "intent_id": intent["intent_id"],
        "intent_digest": intent["intent_digest"],
        "preparation_digest": prepared["preparation_digest"],
        "dry_run_digest": dry["dry_run_digest"],
        "implementation_commit": implementation_commit,
        "started_at_epoch": time.time(),
        "retry_cap": 0,
    }
    start["start_digest"] = digest(start, digest_field="start_digest")
    write_exclusive(start_path, start)
    command = _allocator_command(
        receipt_path=Path(prepared["bundle_receipt_path"]), run_root=run_root,
        authorization=prepared["authorization"],
        machine_avoidlist_path=machine_avoidlist_path, execute=True,
    )
    code = allocator_runner(command, run_root / "allocator_paid.log")
    adapter_path = run_root / "adapter_paid.json"
    adapter = json.loads(adapter_path.read_text(encoding="utf-8")) if adapter_path.is_file() else {}
    verified = adapter.get("g1_output_verification") or {}
    controller_complete = (
        code == 0 and adapter.get("status") == "completed"
        and adapter.get("continuing_spend_from_this_run") is False
        and verified.get("status") == "verified_development_only"
        and len(verified.get("episodes") or []) == 4
    )
    final = {
        "schema_version": FINAL_SCHEMA,
        "status": (
            "controller_completed_pending_billing_and_private_delivery"
            if controller_complete else "blocked_after_allocator_attempt"
        ),
        "intent_id": intent["intent_id"],
        "intent_digest": intent["intent_digest"],
        "start_digest": start["start_digest"],
        "implementation_commit": implementation_commit,
        "paid_allocator_exit_code": code,
        "adapter_status": adapter.get("status"),
        "four_episodes_verified": controller_complete,
        "run_teardown_confirmed_by_adapter": adapter.get("continuing_spend_from_this_run") is False,
        "global_provider_zero_verified": False,
        "official_billing_reconciled": False,
        "private_review_delivered": False,
        "blockers": [] if controller_complete else adapter.get("blockers") or ["g1_team_campaign_terminal_incomplete"],
        "claim_ceiling": "development_only",
    }
    final["dispatch_digest"] = digest(final, digest_field="dispatch_digest")
    write_exclusive(final_path, final)
    return final


def dispatch_one_g1_team_campaign(
    *, queue_root: Path, registry_path: Path, work_root: Path,
    implementation_commit: str, machine_avoidlist_path: Path | None = None,
    execute: bool = False, allocator_runner: AllocatorRunner = _run_allocator,
) -> dict[str, Any]:
    """Process at most one pending intent; never repeat a paid attempt."""

    queue = Path(queue_root)
    work = Path(work_root)
    if (
        not queue.is_absolute() or queue.is_symlink() or not queue.is_dir()
        or not work.is_absolute() or work.is_symlink()
    ):
        raise ValueError("g1_team_campaign_dispatch_paths_invalid")
    work.mkdir(parents=True, exist_ok=True, mode=0o750)
    descriptor = os.open(work / ".dispatch.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        unresolved_started = []
        for intent_path in sorted(queue.glob("g1-*/intent.json")):
            if intent_path.parent.is_symlink() or intent_path.is_symlink():
                raise ValueError("g1_team_campaign_intent_path_unsafe")
            intent = _read(intent_path, field="intent_digest")
            intent_id = intent.get("intent_id")
            if (
                not isinstance(intent_id, str)
                or re.fullmatch(r"g1-[0-9a-f]{64}", intent_id) is None
                or intent_path.parent.name != intent_id
            ):
                raise ValueError("g1_team_campaign_intent_id_invalid")
            final_path = work / intent_id / "dispatch_final.json"
            started_path = final_path.with_name("execution_started.json")
            if final_path.is_file():
                continue
            if started_path.is_file():
                unresolved_started.append(intent_id)
                continue
            # Expired authority is terminal before preparation or provider work.
            # Seal it so a stale first entry cannot starve later team requests.
            request = intent.get("request")
            authorization = request.get("authorization") if isinstance(request, dict) else None
            expiry = authorization.get("expires_at_epoch") if isinstance(authorization, dict) else None
            if (
                intent.get("schema_version") != INTENT_SCHEMA
                or intent.get("status") != "accepted_not_dispatched"
                or type(expiry) not in (int, float)
            ):
                raise ValueError("g1_team_campaign_intent_invalid")
            if time.time() >= expiry:
                final_path.parent.mkdir(mode=0o750, exist_ok=True)
                final = {
                    "schema_version": FINAL_SCHEMA,
                    "status": "authorization_expired_before_provider",
                    "intent_id": intent_id,
                    "intent_digest": intent["intent_digest"],
                    "provider_mutation_performed": False,
                    "four_episodes_verified": False,
                    "global_provider_zero_verified": False,
                    "official_billing_reconciled": False,
                    "private_review_delivered": False,
                    "blockers": ["g1_team_campaign_authority_expired"],
                    "claim_ceiling": "development_only",
                }
                final["dispatch_digest"] = digest(final, digest_field="dispatch_digest")
                write_exclusive(final_path, final)
                return final
            return _dispatch_one_locked(
                intent_path=intent_path, registry_path=Path(registry_path),
                work_root=work, implementation_commit=implementation_commit,
                machine_avoidlist_path=machine_avoidlist_path,
                execute=execute, allocator_runner=allocator_runner,
            )
        return {
            "schema_version": FINAL_SCHEMA,
            "status": (
                "awaiting_exact_attempt_reconciliation" if unresolved_started
                else "no_pending_intent"
            ),
            "unresolved_started_intent_ids": unresolved_started,
        }
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-root", type=Path, required=True)
    parser.add_argument("--registry-path", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--implementation-commit", required=True)
    parser.add_argument("--machine-avoidlist-path", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    result = dispatch_one_g1_team_campaign(
        queue_root=args.queue_root, registry_path=args.registry_path,
        work_root=args.work_root, implementation_commit=args.implementation_commit,
        machine_avoidlist_path=args.machine_avoidlist_path, execute=args.execute,
    )
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] in {
        "no_pending_intent", "dry_run_ready",
        "authorization_expired_before_provider",
        "controller_completed_pending_billing_and_private_delivery",
    } else 2


if __name__ == "__main__":
    raise SystemExit(main())
