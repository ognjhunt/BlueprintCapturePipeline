#!/usr/bin/env python3
"""Close one stopped launch claim using host liveness and fresh provider-zero evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from deploy_control_plane_commit import DEFAULT_PAID_LAUNCH_LOCKS, _holding_paid_launch_locks
from blueprint_pipeline.task_evaluation_launch_reconciler import (
    ORPHAN_RECOVERY_SCHEMA_VERSION, TaskEvaluationLaunchError, _guard_provider_zero,
    _processing_profile, _read, _required_provider_scope, _timestamp, _write_immutable,
    canonical_digest,
)


def _observe_dispatcher():
    result = subprocess.run([
        "systemctl", "show", "blueprint-task-evaluation-launch-dispatcher.service",
        "-p", "LoadState", "-p", "ActiveState", "-p", "MainPID", "-p", "ControlGroup",
        "-p", "ExecMainExitTimestamp",
    ], check=True, capture_output=True, text=True, timeout=10,
        env={**os.environ, "LC_ALL": "C", "TZ": "UTC"})
    return dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)


def _process_alive(pid):
    if type(pid) is not int or pid <= 0:
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def reconcile_stopped_launch(*, queue_root, state_root, guard_report_path, launch_id, now=None):
    if not launch_id or Path(launch_id).name != launch_id or launch_id in {".", ".."}:
        raise TaskEvaluationLaunchError("stopped_launch_id_invalid")
    queue, state, guard_path = Path(queue_root), Path(state_root), Path(guard_report_path)
    observed_at = now or datetime.now(timezone.utc)
    with _holding_paid_launch_locks(DEFAULT_PAID_LAUNCH_LOCKS):
        unit = _observe_dispatcher()
        if (unit.get("LoadState") != "loaded" or unit.get("ActiveState") not in {"inactive", "failed"}
                or unit.get("MainPID") != "0" or unit.get("ControlGroup") != ""):
            raise TaskEvaluationLaunchError("stopped_launch_dispatcher_not_drained")
        try:
            exited_at = datetime.strptime(unit["ExecMainExitTimestamp"], "%a %Y-%m-%d %H:%M:%S %Z").replace(tzinfo=timezone.utc)
        except (KeyError, ValueError) as exc:
            raise TaskEvaluationLaunchError("stopped_launch_exit_time_missing") from exc
        matches = list((queue / "processing").glob(launch_id + "-*.json"))
        if len(matches) != 1 or matches[0].is_symlink():
            raise TaskEvaluationLaunchError("stopped_launch_claim_ambiguous")
        request_path = matches[0]
        request = _read(request_path)
        run_root = state / launch_id
        started = _read(run_root / "launch_started.json")
        if (request.get("launch_id") != launch_id or started.get("launch_id") != launch_id
                or not request.get("request_digest")
                or started.get("request_digest") != request["request_digest"]
                or _process_alive(started.get("process_id"))):
            raise TaskEvaluationLaunchError("stopped_launch_writer_or_binding_unproven")
        profile, source, profile_path = _processing_profile(
            run_root=run_root, request=request, published_profile_dir=None)
        providers, scope = _required_provider_scope(profile, expected_profile_digest=request.get("launch_profile_digest"))
        raw_guard = guard_path.read_bytes()
        zero, blockers = _guard_provider_zero(
            guard=json.loads(raw_guard), required_providers=providers, max_age_seconds=300,
            now=observed_at, not_before=exited_at)
        if not zero or not providers:
            raise TaskEvaluationLaunchError("stopped_launch_provider_zero_required:" + ";".join(blockers))
        started_at = _timestamp(started.get("started_at"))
        if started_at is None or not started_at <= exited_at <= observed_at:
            raise TaskEvaluationLaunchError("stopped_launch_time_binding_invalid")
        secrets = run_root / "allocator/scene-configuration-job/runtime-secrets"
        if secrets.exists() or secrets.is_symlink():
            if any(p.is_symlink() for p in (secrets, *secrets.parents)):
                raise TaskEvaluationLaunchError("stopped_launch_secret_root_unsafe")
            from blueprint_pipeline.task_evaluation_scene_configuration_vast import _discard_staged_runtime_secrets
            if _discard_staged_runtime_secrets(secrets):
                raise TaskEvaluationLaunchError("stopped_launch_secret_cleanup_failed")
        guard = json.loads(raw_guard)
        guard_bytes = (json.dumps(guard, sort_keys=True, separators=(",", ":")) + "\n").encode()
        guard_digest = hashlib.sha256(guard_bytes).hexdigest()
        guard_snapshot = run_root / "reconciliations" / (guard_digest + ".provider-zero.json")
        _write_immutable(guard_snapshot, guard)
        receipt = {
            "schema_version": ORPHAN_RECOVERY_SCHEMA_VERSION, "launch_id": launch_id,
            "request_digest": request["request_digest"], "observed_at": observed_at.isoformat(),
            "status": "provider_zero_confirmed", "recovery_basis": "stopped_dispatcher_and_fresh_provider_zero",
            "dispatcher_state": unit, "started_digest": started.get("started_digest"),
            "lease_age_seconds": (observed_at - started_at).total_seconds(),
            "hard_ttl_seconds": started.get("hard_ttl_seconds"), "required_providers": providers,
            "launch_profile_digest": profile["profile_digest"], "provider_scope_source": scope,
            "profile_record_source": source, "profile_record_path": str(profile_path),
            "guard_report_path": str(guard_snapshot), "guard_report_sha256": "sha256:" + guard_digest,
            "source_guard_report_path": str(guard_path), "temporary_runtime_secrets_removed": not secrets.exists(),
            "provider_zero_confirmed": True, "automatic_retry_performed": False,
            "allocator_invoked": False, "historical_spend_settled": False, "blockers": [],
        }
        receipt["recovery_digest"] = canonical_digest(receipt, digest_field="recovery_digest")
        receipt_path = run_root / "orphan_recovery_receipt.json"
        if receipt_path.exists():
            prior = _read(receipt_path)
            if (any(prior.get(k) != receipt[k] for k in ("schema_version", "launch_id", "request_digest", "launch_profile_digest", "provider_zero_confirmed"))
                    or prior.get("recovery_digest") != canonical_digest(prior, digest_field="recovery_digest")):
                raise TaskEvaluationLaunchError("stopped_launch_recovery_conflict")
            receipt = prior
        else:
            _write_immutable(receipt_path, receipt)
        destination = queue / "blocked"
        destination.mkdir(parents=True, exist_ok=True)
        if _read(request_path) != request or (
            (destination / request_path.name).exists()
            and (destination / request_path.name).read_bytes() != request_path.read_bytes()
        ):
            raise TaskEvaluationLaunchError("stopped_launch_claim_changed")
        os.replace(request_path, destination / request_path.name)
        return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("queue-root", "state-root", "guard-report-path", "launch-id"):
        parser.add_argument("--" + name, required=True)
    print(json.dumps(reconcile_stopped_launch(**vars(parser.parse_args())), sort_keys=True))


if __name__ == "__main__":
    main()
