"""Independent liveness, teardown, provider-zero, and orphan reconciliation.

This process never launches or retries paid work.  It observes the immutable
launch state, consumes the separately produced GPU spend-guard inventory, and
only closes an abandoned processing lease after every provider required by the
Pipeline-owned profile is API-confirmed at zero.  Provider mutation remains in
the canonical allocator and the independent GPU spend guard.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .task_evaluation_launch_progress import build_launch_progress
from .task_evaluation_launch_webapp_sync import sync_launch_progress_to_webapp
from .task_evaluation_launch_dispatcher import (
    TaskEvaluationLaunchError,
    canonical_digest,
)


RECONCILIATION_SCHEMA_VERSION = "task_evaluation_launch_reconciliation.v1"
ORPHAN_RECOVERY_SCHEMA_VERSION = "task_evaluation_launch_orphan_recovery.v1"


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise TaskEvaluationLaunchError(f"json_object_required:{path.name}")
    return dict(value)


def _write_immutable(path: Path, value: Mapping[str, Any]) -> None:
    payload = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as stream:
            stream.write(payload)
    except FileExistsError:
        if path.read_bytes() != payload:
            raise TaskEvaluationLaunchError(f"immutable_reconciliation_conflict:{path.name}")


def _timestamp(value: Any) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(str(value or "").replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(timezone.utc)


def _guard_provider_zero(
    *,
    guard: Mapping[str, Any],
    required_providers: Sequence[str],
    max_age_seconds: int,
    now: datetime,
    not_before: datetime | None,
) -> tuple[bool, list[str]]:
    blockers: list[str] = []
    if not required_providers:
        blockers.append("gpu_required_provider_scope_missing")
    if guard.get("schema_version") != "gpu_spend_guard.v1":
        blockers.append("gpu_spend_guard_schema_invalid")
    generated_at = _timestamp(guard.get("generated_at"))
    if generated_at is None:
        blockers.append("gpu_spend_guard_timestamp_invalid")
    else:
        age = (now - generated_at).total_seconds()
        if age < -60 or age > max_age_seconds:
            blockers.append("gpu_spend_guard_stale")
        if not_before is not None and generated_at < not_before:
            blockers.append("gpu_spend_guard_predates_launch")
    if guard.get("reap_mode") is not True:
        blockers.append("gpu_spend_guard_reap_mode_missing")
    if guard.get("live_instance_count") != 0:
        blockers.append("gpu_provider_nonzero")
    if guard.get("reap_candidate_ids") not in ([], ()):
        blockers.append("gpu_orphan_reap_candidates_remaining")
    for result in guard.get("reap_results") or []:
        if isinstance(result, Mapping) and result.get("status") != "terminated":
            blockers.append("gpu_orphan_reap_not_confirmed")

    inventories = {
        str(row.get("provider")): row
        for row in guard.get("inventory_results") or []
        if isinstance(row, Mapping)
    }
    for provider in required_providers:
        inventory = inventories.get(str(provider))
        if inventory is None:
            blockers.append(f"gpu_inventory_missing:{provider}")
        elif inventory.get("status") != "succeeded":
            blockers.append(f"gpu_inventory_not_confirmed:{provider}")
    return not blockers, sorted(set(blockers))


def _queue_destination(queue_root: Path, status: str) -> Path:
    return queue_root / (
        "completed" if status in {"completed", "dry_run_completed"} else "blocked"
    )


def reconcile_launches(
    *,
    queue_root: str | Path,
    state_root: str | Path,
    guard_report_path: str | Path,
    now: datetime | None = None,
    fallback_stale_seconds: int = 14_400,
    publish_progress: bool = True,
) -> dict[str, Any]:
    """Reconcile all launch leases without invoking or retrying the allocator."""

    observed_at = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    queue = Path(queue_root).expanduser().resolve()
    state = Path(state_root).expanduser().resolve()
    guard_path = Path(guard_report_path).expanduser().resolve()
    guard: dict[str, Any] = {}
    guard_error: str | None = None
    try:
        guard = _read(guard_path)
    except (OSError, json.JSONDecodeError, TaskEvaluationLaunchError):
        guard_error = "gpu_spend_guard_report_unavailable"

    rows: list[dict[str, Any]] = []
    processing = queue / "processing"
    processing.mkdir(parents=True, exist_ok=True)
    for request_path in sorted(processing.glob("*.json")):
        try:
            request = _read(request_path)
            launch_id = str(request.get("launch_id") or request_path.stem)
            run_root = state / launch_id
            receipt_path = run_root / "launch_receipt.json"
            if receipt_path.is_file():
                receipt = _read(receipt_path)
                destination = _queue_destination(queue, str(receipt.get("status") or "blocked"))
                destination.mkdir(parents=True, exist_ok=True)
                os.replace(request_path, destination / request_path.name)
                rows.append({"launch_id": launch_id, "status": "terminal_queue_repaired"})
                continue

            started_path = run_root / "launch_started.json"
            started = _read(started_path) if started_path.is_file() else {}
            profile_path = run_root / "launch_profile.json"
            profile = _read(profile_path) if profile_path.is_file() else {}
            started_at = _timestamp(started.get("started_at"))
            ttl = started.get("hard_ttl_seconds")
            ttl_seconds = (
                int(ttl)
                if isinstance(ttl, int) and not isinstance(ttl, bool) and ttl > 0
                else fallback_stale_seconds
            )
            lease_age = (
                (observed_at - started_at).total_seconds()
                if started_at is not None
                else observed_at.timestamp() - request_path.stat().st_mtime
            )
            if lease_age <= ttl_seconds:
                # The dispatcher blocks on the allocator for the whole run, so
                # this timer is the only place that can report progress while
                # the launch is still in flight. Best effort: a failed publish
                # is recorded and never affects the run.
                progress_result = None
                if publish_progress:
                    try:
                        progress_result = sync_launch_progress_to_webapp(
                            progress=build_launch_progress(
                                run_root=run_root,
                                request=request,
                                guard=guard,
                                elapsed_seconds=lease_age,
                                observed_at=observed_at,
                            )
                        )
                    except (OSError, ValueError) as exc:
                        progress_result = {
                            "status": "failed",
                            "reason": type(exc).__name__.lower(),
                        }
                rows.append({
                    "launch_id": launch_id,
                    "status": "processing_within_ttl",
                    "lease_age_seconds": round(max(0.0, lease_age), 3),
                    "progress_publish": (
                        progress_result.get("status") if progress_result else "disabled"
                    ),
                })
                continue

            reconciliation = profile.get("reconciliation")
            reconciliation = reconciliation if isinstance(reconciliation, Mapping) else {}
            required_providers = [
                str(item) for item in reconciliation.get("required_providers") or []
            ]
            max_guard_age = reconciliation.get("max_guard_age_seconds")
            max_guard_age_seconds = (
                int(max_guard_age)
                if isinstance(max_guard_age, int)
                and not isinstance(max_guard_age, bool)
                and max_guard_age > 0
                else 300
            )
            provider_zero, blockers = _guard_provider_zero(
                guard=guard,
                required_providers=required_providers,
                max_age_seconds=max_guard_age_seconds,
                now=observed_at,
                not_before=started_at,
            ) if guard_error is None else (False, [guard_error])
            recovery = {
                "schema_version": ORPHAN_RECOVERY_SCHEMA_VERSION,
                "launch_id": launch_id,
                "request_digest": request.get("request_digest"),
                "observed_at": observed_at.isoformat(),
                "status": "provider_zero_confirmed" if provider_zero else "recovery_pending",
                "lease_age_seconds": round(max(0.0, lease_age), 3),
                "hard_ttl_seconds": ttl_seconds,
                "required_providers": required_providers,
                "guard_report_path": str(guard_path),
                "guard_report_sha256": (
                    "sha256:" + hashlib.sha256(guard_path.read_bytes()).hexdigest()
                    if guard_path.is_file()
                    else None
                ),
                "provider_zero_confirmed": provider_zero,
                "automatic_retry_performed": False,
                "allocator_invoked": False,
                "blockers": blockers,
            }
            recovery["recovery_digest"] = canonical_digest(
                recovery, digest_field="recovery_digest"
            )
            _write_immutable(
                run_root / "reconciliations" / f"{recovery['recovery_digest'][7:]}.json",
                recovery,
            )
            if provider_zero:
                _write_immutable(run_root / "orphan_recovery_receipt.json", recovery)
                destination = queue / "blocked"
                destination.mkdir(parents=True, exist_ok=True)
                os.replace(request_path, destination / request_path.name)
            rows.append({
                "launch_id": launch_id,
                "status": recovery["status"],
                "provider_zero_confirmed": provider_zero,
                "blockers": blockers,
            })
        except (OSError, json.JSONDecodeError, TaskEvaluationLaunchError) as exc:
            rows.append({
                "launch_id": request_path.stem,
                "status": "reconciliation_blocked",
                "blockers": ["launch_reconciliation_input_invalid"],
                "error_type": type(exc).__name__,
            })

    sync_rows: list[dict[str, Any]] = []
    from .task_evaluation_launch_webapp_sync import sync_launch_receipt_to_webapp

    for receipt_path in sorted(state.glob("*/launch_receipt.json")):
        run_root = receipt_path.parent
        if (run_root / "webapp_sync_succeeded.json").is_file():
            continue
        try:
            receipt = _read(receipt_path)
            profile = _read(run_root / "launch_profile.json")
            sync_policy = profile.get("webapp_sync")
            sync_policy = sync_policy if isinstance(sync_policy, Mapping) else {}
            max_attempts = int(sync_policy.get("max_attempts") or 0)
            attempt_dir = run_root / "webapp_sync_attempts"
            prior_attempts = []
            for path in sorted(attempt_dir.glob("*.json")):
                value = _read(path)
                if value.get("status") == "failed":
                    prior_attempts.append(value)
            if len(prior_attempts) >= max_attempts:
                sync_rows.append({
                    "launch_id": receipt.get("launch_id"),
                    "status": "webapp_sync_retry_exhausted",
                    "attempts": len(prior_attempts),
                    "provider_mutation_performed": False,
                })
                continue
            sync_result = sync_launch_receipt_to_webapp(receipt=receipt)
            if sync_result.get("status") == "skipped":
                sync_rows.append({
                    "launch_id": receipt.get("launch_id"),
                    "status": "webapp_sync_not_configured",
                    "attempts": len(prior_attempts),
                    "provider_mutation_performed": False,
                })
                continue
            attempt = {
                **sync_result,
                "attempt_number": len(prior_attempts) + 1,
                "attempted_at": observed_at.isoformat(),
                "provider_mutation_performed": False,
            }
            attempt["sync_result_digest"] = canonical_digest(
                attempt, digest_field="sync_result_digest"
            )
            _write_immutable(
                attempt_dir / f"{attempt['sync_result_digest'][7:]}.json", attempt
            )
            if sync_result.get("status") == "succeeded":
                _write_immutable(run_root / "webapp_sync_succeeded.json", attempt)
            sync_rows.append({
                "launch_id": receipt.get("launch_id"),
                "status": f"webapp_sync_{sync_result.get('status')}",
                "attempts": attempt["attempt_number"],
                "provider_mutation_performed": False,
            })
        except (OSError, ValueError, json.JSONDecodeError, TaskEvaluationLaunchError) as exc:
            sync_rows.append({
                "launch_id": run_root.name,
                "status": "webapp_sync_reconciliation_blocked",
                "blockers": ["webapp_sync_reconciliation_input_invalid"],
                "error_type": type(exc).__name__,
                "provider_mutation_performed": False,
            })

    report = {
        "schema_version": RECONCILIATION_SCHEMA_VERSION,
        "observed_at": observed_at.isoformat(),
        "status": "passed" if all(
            row.get("status") not in {
                "recovery_pending",
                "reconciliation_blocked",
                "webapp_sync_retry_exhausted",
                "webapp_sync_not_configured",
                "webapp_sync_failed",
                "webapp_sync_reconciliation_blocked",
            }
            for row in [*rows, *sync_rows]
        ) else "blocked",
        "processing_count": len(rows),
        "launches": rows,
        "webapp_sync": sync_rows,
        "automatic_retry_performed": False,
        "allocator_invoked": False,
    }
    report["reconciliation_digest"] = canonical_digest(
        report, digest_field="reconciliation_digest"
    )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-root", required=True)
    parser.add_argument("--state-root", required=True)
    parser.add_argument("--guard-report", required=True)
    parser.add_argument("--report-out", required=True)
    parser.add_argument("--fallback-stale-seconds", type=int, default=14_400)
    args = parser.parse_args(argv)
    result = reconcile_launches(
        queue_root=args.queue_root,
        state_root=args.state_root,
        guard_report_path=args.guard_report,
        fallback_stale_seconds=max(1, args.fallback_stale_seconds),
    )
    output = Path(args.report_out).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "passed" else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
