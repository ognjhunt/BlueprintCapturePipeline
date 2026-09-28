"""Bridge a signed selected-policy intent to the canonical G1 paid allocator.

Operator approvals, credentials and SONIC cache are separate from team intake.
An execution-start record prevents replacement launches after interruption.
Controller reports remain pending independent billing and private delivery.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import re
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .control_plane_disk_budget import (
    DEFAULT_RESERVATION_ROOT, ControlPlaneDiskBudgetError, reserve_control_plane_disk,
)
from .control_plane_disk_reservation_heartbeat import keep_reservation_live
from .decision_evidence_contracts import cross_runtime_canonical_digest as digest
from .native_g1_team_campaign_dispatcher import (
    AdmissionRefresher, AllocatorRunner, _refresh_paid_admission, _run_allocator,
)
from .native_g1_team_campaign_intake import _read
from .native_g1_team_policy_authority import verify_g1_team_policy_authority
from .native_g1_team_policy_preparation import _operator_directory, prepare_g1_team_policy
from .native_g1_team_policy_run_intake import INTENT_SCHEMA
from .task_evaluation_launch_preparation_queue import (
    _write_launch_preparation_record_exclusive_locked as write_exclusive,
)


FINAL_SCHEMA = "native_g1_team_policy_dispatch_result.v1"
START_SCHEMA = "native_g1_team_policy_execution_start.v1"
DRY_SCHEMA = "native_g1_team_policy_dry_run.v1"
_ID = re.compile(r"g1-team-policy-[0-9a-f]{64}\Z")


def _result(status: str, *, intent: dict | None = None, **fields: Any) -> dict[str, Any]:
    value = {"schema_version": FINAL_SCHEMA, "status": status,
             "claim_ceiling": "development_only", "official_billing_reconciled": False,
             "global_provider_zero_verified": False, "private_review_delivered": False,
             "public_redistribution_authorized": False, **fields}
    if intent is not None:
        value.update(intent_id=intent["intent_id"], intent_digest=intent["intent_digest"])
    return value


def _seal(path: Path, value: dict[str, Any]) -> dict[str, Any]:
    value["dispatch_digest"] = digest(value, digest_field="dispatch_digest")
    write_exclusive(path, value)
    return value


def _adapter(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= 16 * 1024 * 1024:
        return {}
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        return {}
    # Reject nonfinite JSON before it can enter a durable receipt.
    digest(value)
    return value


def _command(*, live, prepared, run, credential_registry, avoidlist, execute):
    nonce = str(time.time_ns())
    phase = "paid" if execute else "dry"
    command = [sys.executable, "-m", "blueprint_pipeline.paid_resource_allocator",
               "gpu-canary", "--provider", "vast", "--probe-kind", "native-g1-team-policy"]
    paths = {
        "g1-team-bundle-receipt": prepared["bundle_receipt_path"],
        "g1-team-intent": live["intent_path"], "g1-team-registry": live["registry_path"],
        "g1-team-approval": live["approval_path"],
        "g1-team-credential-registry": credential_registry,
        "adp-job-dir": run, "expected-source-commit": prepared["implementation_commit"],
        "admission-out": run / f"admission_{phase}_{nonce}.json",
        "adapter-output": run / f"adapter_{phase}_{nonce}.json",
        "adp-max-hourly-rate-usd": "3.0",
        "adp-max-spend-usd": prepared["authorization"]["maximum_cost_usd"],
        "adp-hard-ttl-seconds": prepared["authorization"]["hard_ttl_seconds"],
    }
    for name, value in paths.items():
        command.extend(["--" + name, str(value)])
    for client in sorted(live["trusted_clients"]):
        command.extend(["--g1-team-trusted-client", client])
    if avoidlist is not None:
        command.extend(["--adp-machine-avoidlist", str(avoidlist)])
    if execute:
        command.append("--execute")
    return command, run / f"allocator_{phase}_{nonce}.log", Path(paths["adapter-output"])


def _dispatch_locked(*, live, intent, work, sonic, commit, credentials, avoidlist,
                     execute, allocator_runner, admission_refresher):
    directory = work / intent["intent_id"]
    _operator_directory(directory, required=False)
    directory.mkdir(mode=0o750, exist_ok=True)
    final_path, start_path = directory / "dispatch_final.json", directory / "execution_started.json"
    if final_path.exists() or final_path.is_symlink():
        final = _read(final_path, field="dispatch_digest")
        if final.get("intent_digest") != intent["intent_digest"]:
            raise ValueError("g1_team_policy_dispatch_final_conflict")
        return None
    if start_path.exists() or start_path.is_symlink():
        started = _read(start_path, field="start_digest")
        if started.get("intent_digest") != intent["intent_digest"]:
            raise ValueError("g1_team_policy_dispatch_start_conflict")
        return _result("execution_already_started_reconcile_exact_attempt", intent=intent,
                       provider_mutation_unproven=True)
    try:
        authority = verify_g1_team_policy_authority(**live)
        reservation = reserve_control_plane_disk(
            "launch_preparation", target_root=work,
            reservation_root=os.getenv("BLUEPRINT_CONTROL_PLANE_DISK_RESERVATION_ROOT", str(DEFAULT_RESERVATION_ROOT)),
            workspace=directory, workload="g1_team_policy_preparation",
        )
        with reservation, keep_reservation_live(reservation) as health:
            health.check()
            prepared = prepare_g1_team_policy(
                authority_arguments=live, work_root=work, sonic_asset_dir=sonic, implementation_commit=commit,
            )
            health.check()
            run = directory / "run"
            _operator_directory(run, required=False)
            run.mkdir(mode=0o750, exist_ok=True)
            command, log, output = _command(live=live, prepared=prepared, run=run,
                                           credential_registry=credentials, avoidlist=avoidlist, execute=False)
            code = allocator_runner(command, log)
            observed = _adapter(output)
            health.check()
    except (OSError, ValueError, ControlPlaneDiskBudgetError):
        return _result("blocked_before_provider", intent=intent, provider_mutation_performed=False,
                       blockers=["g1_team_policy_preparation_or_dry_admission_failed"])
    if code != 0 or observed.get("status") != "dry_run_ready" or observed.get("provider_mutations_performed") != 0:
        return _result("blocked_before_provider", intent=intent, provider_mutation_performed=False,
                       blockers=observed.get("blockers") or ["g1_team_policy_dry_run_failed"])
    dry = {"schema_version": DRY_SCHEMA, "status": "dry_run_ready",
           "intent_id": intent["intent_id"], "intent_digest": intent["intent_digest"],
           "preparation_digest": prepared["preparation_digest"], "implementation_commit": commit,
           "adapter_digest": digest(observed), "provider_mutation_performed": False,
           "launch_readiness": "static_transport_only", "claim_ceiling": "development_only"}
    dry["dry_run_digest"] = digest(dry, digest_field="dry_run_digest")
    write_exclusive(directory / f"dry_run_{time.time_ns()}.json", dry)
    if not execute:
        return dry
    blockers = admission_refresher(run)
    if blockers:
        return _result("blocked_before_provider", intent=intent, provider_mutation_performed=False, blockers=blockers)
    try:
        fresh = verify_g1_team_policy_authority(**live)
        if fresh != authority:
            raise ValueError("g1_team_policy_authority_changed_after_guard")
    except (OSError, ValueError):
        return _result("blocked_before_provider", intent=intent, provider_mutation_performed=False,
                       blockers=["g1_team_policy_authority_changed_after_guard"])
    start = {"schema_version": START_SCHEMA, "status": "execution_started_once",
             "intent_id": intent["intent_id"], "intent_digest": intent["intent_digest"],
             "preparation_digest": prepared["preparation_digest"], "dry_run_digest": dry["dry_run_digest"],
             "implementation_commit": commit, "started_at_epoch": time.time(), "retry_cap": 0}
    start["start_digest"] = digest(start, digest_field="start_digest")
    write_exclusive(start_path, start)
    command, log, output = _command(live=live, prepared=prepared, run=run,
                                   credential_registry=credentials, avoidlist=avoidlist, execute=True)
    # Unexpected interruption propagates. The durable start remains and every
    # future dispatcher refuses a replacement until exact reconciliation.
    code = allocator_runner(command, log)
    try:
        observed = _adapter(output)
    except (OSError, ValueError):
        observed = {}
    if (not observed or (observed.get("continuing_spend_from_this_run") is not False
                         and observed.get("provider_mutations_performed") != 0)):
        return _result("execution_already_started_reconcile_exact_attempt", intent=intent,
                       provider_mutation_unproven=True, blockers=["g1_team_policy_allocator_closeout_unproven"])
    proof = observed.get("g1_team_output_verification")
    reported_complete = (code == 0 and observed.get("status") == "completed"
                         and isinstance(proof, dict) and proof.get("status") == "verified_development_only"
                         and type(proof.get("policy_query_count")) is int and proof["policy_query_count"] > 0
                         and observed.get("continuing_spend_from_this_run") is False)
    final = _result("controller_completed_pending_billing_and_private_delivery" if reported_complete
                    else "blocked_after_allocator_attempt", intent=intent,
                    start_digest=start["start_digest"], implementation_commit=commit,
                    paid_allocator_exit_code=code, adapter_status=observed.get("status"),
                    adapter_result_path=str(output), adapter_result_digest=digest(observed),
                    controller_reported_episode_verified=reported_complete,
                    run_teardown_confirmed_by_adapter=observed.get("continuing_spend_from_this_run") is False,
                    blockers=[] if reported_complete else observed.get("blockers") or ["g1_team_policy_terminal_incomplete"])
    return _seal(final_path, final)


def dispatch_one_g1_team_policy(
    *, queue_root: Path, registry_path: Path, approval_root: Path, trusted_clients: set[str],
    work_root: Path, sonic_asset_dir: Path, credential_registry_path: Path,
    implementation_commit: str, machine_avoidlist_path: Path | None = None,
    execute: bool = False, allocator_runner: AllocatorRunner = _run_allocator,
    admission_refresher: AdmissionRefresher = _refresh_paid_admission,
) -> dict[str, Any]:
    """Inspect queued choices without waiting on another locked intent."""
    if re.fullmatch(r"[0-9a-f]{40}", implementation_commit) is None or not trusted_clients:
        raise ValueError("g1_team_policy_dispatch_configuration_invalid")
    queue, work, approvals = Path(queue_root), Path(work_root), Path(approval_root)
    for directory, required in ((queue, True), (work, False), (approvals, True)):
        _operator_directory(directory, required=required)
    work.mkdir(parents=True, exist_ok=True, mode=0o750)
    locks = work / ".intent-locks"
    _operator_directory(locks, required=False)
    locks.mkdir(mode=0o750, exist_ok=True)
    waiting, started, held = [], [], []
    for path in sorted(queue.glob("g1-team-policy-*/intent.json")):
        if path.parent.is_symlink() or path.stat().st_size > 1024 * 1024:
            raise ValueError("g1_team_policy_dispatch_intent_path_invalid")
        intent = _read(path, field="intent_digest")
        identity = intent.get("intent_id")
        request = intent.get("request")
        authorization = request.get("authorization") if isinstance(request, dict) else None
        expiry = authorization.get("expires_at_epoch") if isinstance(authorization, dict) else None
        if (not isinstance(identity, str) or _ID.fullmatch(identity) is None or path.parent.name != identity
                or intent.get("schema_version") != INTENT_SCHEMA
                or intent.get("status") != "accepted_pending_operator_approval"
                or type(expiry) not in (int, float) or not math.isfinite(expiry)):
            raise ValueError("g1_team_policy_dispatch_intent_invalid")
        descriptor = os.open(locks / (identity + ".lock"), os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                held.append(identity)
                continue
            directory = work / identity
            _operator_directory(directory, required=False)
            final_path = directory / "dispatch_final.json"
            if final_path.exists() or final_path.is_symlink():
                if _read(final_path, field="dispatch_digest").get("intent_digest") != intent["intent_digest"]:
                    raise ValueError("g1_team_policy_dispatch_final_conflict")
                continue
            if (directory / "execution_started.json").exists() or (directory / "execution_started.json").is_symlink():
                if _read(directory / "execution_started.json", field="start_digest").get("intent_digest") != intent["intent_digest"]:
                    raise ValueError("g1_team_policy_dispatch_start_conflict")
                started.append(identity)
                continue
            if time.time() >= expiry:
                directory.mkdir(mode=0o750, exist_ok=True)
                return _seal(final_path, _result("authorization_expired_before_provider", intent=intent,
                             provider_mutation_performed=False, blockers=["g1_team_policy_authority_expired"]))
            preparation_path = directory / "preparation.json"
            if preparation_path.exists() or preparation_path.is_symlink():
                preparation = _read(preparation_path, field="preparation_digest")
                if preparation.get("intent_digest") != intent["intent_digest"]:
                    raise ValueError("g1_team_policy_dispatch_preparation_conflict")
                if preparation.get("implementation_commit") != implementation_commit:
                    return _seal(final_path, _result(
                        "superseded_before_provider_by_release", intent=intent,
                        preparation_digest=preparation["preparation_digest"],
                        prepared_implementation_commit=preparation.get("implementation_commit"),
                        implementation_commit=implementation_commit,
                        provider_mutation_performed=False,
                        blockers=["g1_team_policy_prepared_on_prior_release"],
                    ))
            approval = approvals / (identity + ".json")
            if not approval.exists() and not approval.is_symlink():
                waiting.append(identity)
                continue
            result = _dispatch_locked(
                live={"intent_path": path, "registry_path": Path(registry_path), "approval_path": approval,
                      "trusted_clients": trusted_clients}, intent=intent, work=work, sonic=Path(sonic_asset_dir),
                commit=implementation_commit, credentials=Path(credential_registry_path), avoidlist=machine_avoidlist_path,
                execute=execute, allocator_runner=allocator_runner, admission_refresher=admission_refresher,
            )
            if result is not None:
                return result
        finally:
            os.close(descriptor)
    status = ("awaiting_exact_attempt_reconciliation" if started else "awaiting_operator_approval" if waiting
              else "other_intent_active" if held else "no_pending_intent")
    return _result(status, unresolved_started_intent_ids=started, pending_approval_intent_ids=waiting,
                   held_intent_ids=held)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("queue-root", "registry-path", "approval-root", "work-root", "sonic-asset-dir", "credential-registry-path"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--trusted-client", action="append", required=True)
    parser.add_argument("--implementation-commit", required=True)
    parser.add_argument("--machine-avoidlist-path", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    result = dispatch_one_g1_team_policy(
        queue_root=args.queue_root, registry_path=args.registry_path, approval_root=args.approval_root,
        trusted_clients=set(args.trusted_client), work_root=args.work_root, sonic_asset_dir=args.sonic_asset_dir,
        credential_registry_path=args.credential_registry_path, implementation_commit=args.implementation_commit,
        machine_avoidlist_path=args.machine_avoidlist_path, execute=args.execute,
    )
    print(json.dumps(result, sort_keys=True))
    return 2 if result["status"].startswith("blocked_") else 0


if __name__ == "__main__":
    raise SystemExit(main())
