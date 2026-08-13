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
POST_TEARDOWN_PROVIDER_ZERO_SCHEMA_VERSION = (
    "task_evaluation_post_teardown_provider_zero.v1"
)
POST_TEARDOWN_PROVIDER_ZERO_FILENAME = "post_teardown_provider_zero_receipt.json"
PROVIDER_ZERO_GUARD_SNAPSHOT_SCHEMA_VERSION = (
    "task_evaluation_provider_zero_guard_snapshot.v1"
)
WEBAPP_SYNC_TERMINAL_UNMATCHED_SCHEMA_VERSION = (
    "task_evaluation_launch_webapp_sync_terminal_unmatched.v1"
)
WEBAPP_SYNC_TERMINAL_UNMATCHED_FILENAME = "webapp_sync_terminal_unmatched.json"
WEBAPP_SYNC_SUCCEEDED_FILENAME = "webapp_sync_succeeded.json"


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


def _is_sha256_digest(value: Any) -> bool:
    text = str(value or "")
    return (
        text.startswith("sha256:")
        and len(text) == len("sha256:") + 64
        and all(character in "0123456789abcdef" for character in text[7:])
    )


def _file_digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _guard_provider_zero(
    *,
    guard: Mapping[str, Any],
    required_providers: Sequence[str],
    max_age_seconds: int,
    now: datetime,
    not_before: datetime | None,
    not_before_subject: str = "launch",
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
            blockers.append(f"gpu_spend_guard_predates_{not_before_subject}")
    if guard.get("reap_mode") is not True:
        blockers.append("gpu_spend_guard_reap_mode_missing")
    if guard.get("provider_zero_verified") is not True:
        blockers.append("gpu_provider_zero_not_verified")
    provider_zero = guard.get("provider_zero")
    provider_zero = provider_zero if isinstance(provider_zero, Mapping) else {}
    if provider_zero.get("status") != "verified":
        blockers.append("gpu_provider_zero_status_unverified")
    if guard.get("live_instance_count") != 0:
        blockers.append("gpu_provider_nonzero")
    if guard.get("total_burn_per_hour_usd") not in (0, 0.0):
        blockers.append("gpu_provider_nonzero_burn")
    if provider_zero.get("global_live_instance_count") != 0:
        blockers.append("gpu_provider_zero_global_inventory_nonzero")
    if provider_zero.get("global_total_burn_per_hour_usd") not in (0, 0.0):
        blockers.append("gpu_provider_zero_global_burn_nonzero")
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
        elif inventory.get("row_count") != 0:
            blockers.append(f"gpu_inventory_nonzero:{provider}")
        elif inventory.get("required") is not True:
            blockers.append(f"gpu_inventory_scope_not_required:{provider}")
        elif provider not in {
            str(item) for item in provider_zero.get("required_provider_ids") or []
        }:
            blockers.append(f"gpu_provider_zero_scope_missing:{provider}")
    return not blockers, sorted(set(blockers))


def _terminal_teardown_evidence(
    *,
    receipt: Mapping[str, Any],
) -> tuple[dict[str, Any] | None, list[str]]:
    """Validate the retained teardown artifact named by a terminal receipt.

    This deliberately uses the digest-bearing artifact descriptor in the
    immutable dispatcher receipt rather than rediscovering an arbitrary JSON
    file beneath a run directory.  A provider-zero observation before this
    point cannot be attributed to a completed teardown.
    """

    blockers: list[str] = []
    terminal = receipt.get("terminal_evidence")
    terminal = terminal if isinstance(terminal, Mapping) else {}
    artifacts = terminal.get("artifacts")
    artifacts = artifacts if isinstance(artifacts, Mapping) else {}
    descriptor = artifacts.get("teardown_manifest_path")
    descriptor = descriptor if isinstance(descriptor, Mapping) else {}
    raw_path = str(descriptor.get("path") or "").strip()
    expected_digest = descriptor.get("digest")
    if descriptor.get("exists") is not True or not raw_path:
        blockers.append("terminal_teardown_manifest_descriptor_missing")
        return None, blockers
    if not _is_sha256_digest(expected_digest):
        blockers.append("terminal_teardown_manifest_digest_invalid")
        return None, blockers
    source = Path(raw_path).expanduser()
    if not source.is_absolute() or source.is_symlink() or not source.is_file():
        blockers.append("terminal_teardown_manifest_missing")
        return None, blockers
    path = source.resolve()
    actual_digest = _file_digest(path)
    if actual_digest != expected_digest:
        blockers.append("terminal_teardown_manifest_digest_mismatch")
        return None, blockers
    try:
        manifest = _read(path)
    except (OSError, json.JSONDecodeError, TaskEvaluationLaunchError):
        blockers.append("terminal_teardown_manifest_invalid")
        return None, blockers
    generated_at = _timestamp(manifest.get("generated_at"))
    if generated_at is None:
        blockers.append("terminal_teardown_manifest_timestamp_invalid")
    status = str(manifest.get("status") or "")
    # A lane that blocks before it ever obtains an instance writes a
    # `not_required_*` manifest rather than a completed teardown, because no
    # teardown happened -- there was nothing to tear down. Demanding
    # `completed` there demands an event that by definition never occurred, so
    # those runs could never close, and the reconciler unit failed on every
    # sweep forever after. A permanently red provider-zero signal is not a
    # strict one; it is one nobody can read.
    #
    # It is accepted only on the lane's own proof that nothing was allocated.
    # Wherever an instance id exists, `completed` is still the only answer.
    allocated = True
    if status != "completed":
        if (
            status.startswith("not_required_")
            and manifest.get("vast_instance_ids") == []
            and manifest.get("teardown_actions_performed") == []
        ):
            allocated = False
        else:
            blockers.append("terminal_teardown_manifest_not_completed")
    if manifest.get("continuing_spend_from_this_run") is not False:
        blockers.append("terminal_teardown_continuing_spend_not_false")
    if blockers:
        return None, sorted(set(blockers))
    return {
        "path": str(path),
        "digest": actual_digest,
        "schema_version": manifest.get("schema_version"),
        "status": manifest.get("status"),
        "generated_at": manifest.get("generated_at"),
        "continuing_spend_from_this_run": False,
        # Carried into the closure receipt so a reader can never mistake
        # "torn down" for "never allocated".
        "provider_resource_allocated": allocated,
        "zero_continuing_spend_scope": manifest.get("zero_continuing_spend_scope"),
    }, []


def _terminal_preprovider_admission_blocked(*, receipt: Mapping[str, Any]) -> bool:
    """Recognize the allocator's retained fail-closed pre-provider result.

    ``provider_mutation_attempted`` means the dispatcher crossed the canonical
    allocator boundary; it does not prove that the allocator reached a provider
    API.  An exact, digest-bound allocator admission rejection therefore has no
    teardown obligation.  Any other missing teardown remains a closure blocker.
    """

    terminal = receipt.get("terminal_evidence")
    terminal = terminal if isinstance(terminal, Mapping) else {}
    descriptor = terminal.get("result")
    descriptor = descriptor if isinstance(descriptor, Mapping) else {}
    raw_path = str(descriptor.get("path") or "").strip()
    expected_digest = descriptor.get("digest")
    if (
        descriptor.get("exists") is not True
        or not raw_path
        or not _is_sha256_digest(expected_digest)
    ):
        return False
    source = Path(raw_path).expanduser()
    if not source.is_absolute() or source.is_symlink() or not source.is_file():
        return False
    path = source.resolve()
    if _file_digest(path) != expected_digest:
        return False
    try:
        result = _read(path)
    except (OSError, json.JSONDecodeError, TaskEvaluationLaunchError):
        return False
    blockers = {
        str(item)
        for item in result.get("blockers") or []
        if isinstance(item, str) and item
    }
    return result.get("status") == "blocked" and {
        "paid_resource_admission_has_blockers",
        "paid_resource_admission_not_admitted",
    }.issubset(blockers)


def _retain_guard_snapshot(
    *,
    run_root: Path,
    guard: Mapping[str, Any],
    guard_path: Path,
    guard_bytes: bytes,
) -> tuple[Path, dict[str, Any]]:
    """Copy the exact independently observed guard into immutable run evidence."""

    source_digest = "sha256:" + hashlib.sha256(guard_bytes).hexdigest()
    snapshot = {
        "schema_version": PROVIDER_ZERO_GUARD_SNAPSHOT_SCHEMA_VERSION,
        "source_guard_report_path": str(guard_path),
        "source_guard_report_sha256": source_digest,
        "source_guard_generated_at": guard.get("generated_at"),
        "guard": dict(guard),
    }
    snapshot["snapshot_digest"] = canonical_digest(
        snapshot, digest_field="snapshot_digest"
    )
    path = (
        run_root
        / "provider_zero_guard_snapshots"
        / f"{snapshot['snapshot_digest'][7:]}.json"
    )
    _write_immutable(path, snapshot)
    return path, snapshot


def _validated_post_teardown_provider_zero_receipt(
    *,
    path: Path,
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a previously retained closure receipt only when bound to this run."""

    value = _read(path)
    if (
        value.get("schema_version") != POST_TEARDOWN_PROVIDER_ZERO_SCHEMA_VERSION
        or value.get("status") != "provider_zero_confirmed"
        or value.get("provider_zero_verified") is not True
        or value.get("continuing_spend_from_this_run") is not False
        or value.get("allocator_invoked") is not False
        or value.get("provider_mutation_performed") is not False
        or value.get("automatic_retry_performed") is not False
        or value.get("blockers") != []
        or value.get("provider_zero_receipt_digest")
        != canonical_digest(value, digest_field="provider_zero_receipt_digest")
        or any(
            value.get(field) != receipt.get(field)
            for field in (
                "launch_id",
                "run_id",
                "request_digest",
                "receipt_digest",
                "launch_profile_digest",
            )
        )
    ):
        raise TaskEvaluationLaunchError("post_teardown_provider_zero_receipt_invalid")
    return value


def _reconcile_terminal_provider_zero(
    *,
    run_root: Path,
    receipt: Mapping[str, Any],
    profile: Mapping[str, Any],
    guard: Mapping[str, Any],
    guard_path: Path,
    guard_bytes: bytes | None,
    guard_error: str | None,
    observed_at: datetime,
) -> dict[str, Any]:
    """Retain a post-teardown provider-zero receipt without provider mutation.

    The reconciler never launches, retries, or tears down providers here.  It
    only records a closure receipt after the independently scheduled GPU guard
    proves that the entire required provider scope is zero *after* this run's
    digest-bound teardown manifest.
    """

    retained_path = run_root / POST_TEARDOWN_PROVIDER_ZERO_FILENAME
    if retained_path.is_file():
        retained = _validated_post_teardown_provider_zero_receipt(
            path=retained_path,
            receipt=receipt,
        )
        return {
            "launch_id": receipt.get("launch_id"),
            "status": "provider_zero_receipt_retained",
            "provider_zero_confirmed": True,
            "provider_zero_receipt_digest": retained.get("provider_zero_receipt_digest"),
            "provider_mutation_performed": False,
            "allocator_invoked": False,
            "automatic_retry_performed": False,
            "blockers": [],
        }

    if _terminal_preprovider_admission_blocked(receipt=receipt):
        return {
            "launch_id": receipt.get("launch_id"),
            "status": "provider_zero_not_applicable_pre_provider_admission_blocked",
            "provider_zero_confirmed": None,
            "provider_zero_receipt_required": False,
            "provider_mutation_performed": False,
            "allocator_invoked": False,
            "automatic_retry_performed": False,
            "blockers": [
                "paid_resource_admission_has_blockers",
                "paid_resource_admission_not_admitted",
            ],
        }

    blockers: list[str] = []
    receipt_digest = receipt.get("receipt_digest")
    if (
        not _is_sha256_digest(receipt_digest)
        or receipt_digest != canonical_digest(receipt, digest_field="receipt_digest")
    ):
        blockers.append("terminal_launch_receipt_digest_invalid")
    if receipt.get("launch_id") != run_root.name:
        blockers.append("terminal_launch_receipt_run_binding_invalid")
    if receipt.get("launch_profile_digest") != profile.get("profile_digest"):
        blockers.append("terminal_launch_profile_binding_invalid")
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
    controls = profile.get("required_controls")
    controls = controls if isinstance(controls, Mapping) else {}
    if controls.get("provider_zero_required") is not True:
        blockers.append("terminal_provider_zero_control_missing")
    teardown, teardown_blockers = _terminal_teardown_evidence(receipt=receipt)
    blockers.extend(teardown_blockers)
    if not blockers and teardown is not None:
        provider_zero, guard_blockers = (
            _guard_provider_zero(
                guard=guard,
                required_providers=required_providers,
                max_age_seconds=max_guard_age_seconds,
                now=observed_at,
                not_before=_timestamp(teardown["generated_at"]),
                not_before_subject="teardown",
            )
            if guard_error is None
            else (False, [guard_error])
        )
        blockers.extend(guard_blockers)
    else:
        provider_zero = False
    if blockers or not provider_zero or teardown is None or guard_bytes is None:
        return {
            "launch_id": receipt.get("launch_id"),
            "status": "provider_zero_pending",
            "provider_zero_confirmed": False,
            "required_providers": required_providers,
            "provider_mutation_performed": False,
            "allocator_invoked": False,
            "automatic_retry_performed": False,
            "blockers": sorted(set(blockers or ["gpu_spend_guard_report_unavailable"])),
        }

    snapshot_path, snapshot = _retain_guard_snapshot(
        run_root=run_root,
        guard=guard,
        guard_path=guard_path,
        guard_bytes=guard_bytes,
    )
    closure = {
        "schema_version": POST_TEARDOWN_PROVIDER_ZERO_SCHEMA_VERSION,
        "status": "provider_zero_confirmed",
        "launch_id": receipt.get("launch_id"),
        "run_id": receipt.get("run_id"),
        "request_digest": receipt.get("request_digest"),
        "receipt_digest": receipt.get("receipt_digest"),
        "launch_profile_digest": profile.get("profile_digest"),
        "required_providers": required_providers,
        "teardown_manifest": teardown,
        "independent_guard_snapshot": {
            "path": str(snapshot_path),
            "snapshot_digest": snapshot["snapshot_digest"],
            "source_guard_report_sha256": snapshot["source_guard_report_sha256"],
            "source_guard_generated_at": snapshot["source_guard_generated_at"],
        },
        "observed_at": observed_at.isoformat(),
        "provider_zero_verified": True,
        "continuing_spend_from_this_run": False,
        "allocator_invoked": False,
        "provider_mutation_performed": False,
        "automatic_retry_performed": False,
        "claim_boundary": (
            "Resource-closure evidence only; this receipt does not convert a "
            "scientific or policy blocker into a completed evaluation."
        ),
        "blockers": [],
    }
    closure["provider_zero_receipt_digest"] = canonical_digest(
        closure, digest_field="provider_zero_receipt_digest"
    )
    _write_immutable(retained_path, closure)
    return {
        "launch_id": receipt.get("launch_id"),
        "status": "provider_zero_confirmed",
        "provider_zero_confirmed": True,
        "provider_zero_receipt_digest": closure["provider_zero_receipt_digest"],
        "provider_mutation_performed": False,
        "allocator_invoked": False,
        "automatic_retry_performed": False,
        "blockers": [],
    }


def _queue_destination(queue_root: Path, status: str) -> Path:
    return queue_root / (
        "completed" if status in {"completed", "dry_run_completed"} else "blocked"
    )


def _terminal_unmatched_webapp_sync_receipt(
    *,
    receipt: Mapping[str, Any],
    attempt: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind an irrecoverable WebApp 404 to the exact terminal receipt.

    A 404 means the WebApp has no durable, website-created launch record for
    this identity.  It must remain visible as evidence, but retrying it cannot
    create that missing provenance and must never become a launch retry.
    """

    unmatched = {
        "schema_version": WEBAPP_SYNC_TERMINAL_UNMATCHED_SCHEMA_VERSION,
        "status": "terminal_unmatched",
        "launch_id": receipt.get("launch_id"),
        "run_id": receipt.get("run_id"),
        "request_digest": receipt.get("request_digest"),
        "receipt_digest": receipt.get("receipt_digest"),
        "sync_result_digest": attempt.get("sync_result_digest"),
        "attempt_number": attempt.get("attempt_number"),
        "detected_at": attempt.get("attempted_at"),
        "reason": "http_error:404",
        "webapp_record_bound": False,
        "website_trigger_proven": False,
        "provider_mutation_performed": False,
        "allocator_invoked": False,
        "automatic_retry_performed": False,
        "blockers": ["webapp_launch_record_missing"],
    }
    unmatched["terminal_unmatched_digest"] = canonical_digest(
        unmatched, digest_field="terminal_unmatched_digest"
    )
    return unmatched


def validated_succeeded_webapp_sync_row(
    *, receipt: Mapping[str, Any], attempt: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate the WebApp's exact terminal binding before claiming website origin."""

    response = attempt.get("response")
    response = response if isinstance(response, Mapping) else {}
    if (
        attempt.get("schema_version")
        != "task_evaluation_launch_webapp_sync_result.v1"
        or attempt.get("status") != "succeeded"
        or attempt.get("provider_mutation_performed") is not False
        or attempt.get("sync_result_digest")
        != canonical_digest(attempt, digest_field="sync_result_digest")
        or not isinstance(attempt.get("attempt_number"), int)
        or isinstance(attempt.get("attempt_number"), bool)
        or attempt["attempt_number"] < 1
        or _timestamp(attempt.get("attempted_at")) is None
        or any(
            attempt.get(field) != receipt.get(field)
            for field in ("launch_id", "run_id", "request_digest", "receipt_digest")
        )
        or any(
            response.get(field) != receipt.get(field)
            for field in ("launch_id", "run_id", "request_digest", "receipt_digest")
        )
    ):
        raise TaskEvaluationLaunchError("webapp_sync_succeeded_invalid")
    return {
        "launch_id": receipt.get("launch_id"),
        "status": "webapp_sync_succeeded",
        "attempts": attempt["attempt_number"],
        "blockers": [],
        "webapp_record_bound": True,
        "website_trigger_proven": True,
        "provider_mutation_performed": False,
        "allocator_invoked": False,
        "automatic_retry_performed": False,
        "receipt": {
            "sync_result_digest": attempt.get("sync_result_digest"),
            "launch_id": receipt.get("launch_id"),
            "run_id": receipt.get("run_id"),
            "request_digest": receipt.get("request_digest"),
            "receipt_digest": receipt.get("receipt_digest"),
        },
    }


def _validated_terminal_unmatched_webapp_sync_row(
    *,
    run_root: Path,
    receipt: Mapping[str, Any],
    unmatched: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the retained row only when it is bound to this receipt."""

    sync_result_digest = unmatched.get("sync_result_digest")
    if (
        unmatched.get("schema_version")
        != WEBAPP_SYNC_TERMINAL_UNMATCHED_SCHEMA_VERSION
        or unmatched.get("status") != "terminal_unmatched"
        or unmatched.get("reason") != "http_error:404"
        or unmatched.get("webapp_record_bound") is not False
        or unmatched.get("website_trigger_proven") is not False
        or unmatched.get("provider_mutation_performed") is not False
        or unmatched.get("allocator_invoked") is not False
        or unmatched.get("automatic_retry_performed") is not False
        or unmatched.get("blockers") != ["webapp_launch_record_missing"]
        or unmatched.get("terminal_unmatched_digest")
        != canonical_digest(unmatched, digest_field="terminal_unmatched_digest")
        or not _is_sha256_digest(sync_result_digest)
        or any(
            unmatched.get(field) != receipt.get(field)
            for field in ("launch_id", "run_id", "request_digest", "receipt_digest")
        )
        or not isinstance(unmatched.get("attempt_number"), int)
        or isinstance(unmatched.get("attempt_number"), bool)
        or unmatched["attempt_number"] < 1
    ):
        raise TaskEvaluationLaunchError("webapp_sync_terminal_unmatched_invalid")
    attempt_path = run_root / "webapp_sync_attempts" / f"{str(sync_result_digest)[7:]}.json"
    attempt = _read(attempt_path)
    if (
        attempt.get("sync_result_digest") != sync_result_digest
        or attempt.get("sync_result_digest")
        != canonical_digest(attempt, digest_field="sync_result_digest")
        or attempt.get("status") != "failed"
        or attempt.get("reason") != "http_error:404"
        or attempt.get("attempt_number") != unmatched["attempt_number"]
        or attempt.get("attempted_at") != unmatched.get("detected_at")
        or _timestamp(attempt.get("attempted_at")) is None
        or any(
            attempt.get(field) != receipt.get(field)
            for field in ("launch_id", "run_id", "request_digest", "receipt_digest")
        )
    ):
        raise TaskEvaluationLaunchError("webapp_sync_terminal_unmatched_attempt_invalid")
    return {
        "launch_id": receipt.get("launch_id"),
        "status": "webapp_sync_terminal_unmatched",
        "attempts": unmatched["attempt_number"],
        "blockers": ["webapp_launch_record_missing"],
        "webapp_record_bound": False,
        "website_trigger_proven": False,
        "provider_mutation_performed": False,
        "allocator_invoked": False,
        "automatic_retry_performed": False,
    }


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
    guard_bytes: bytes | None = None
    guard_error: str | None = None
    try:
        guard_bytes = guard_path.read_bytes()
        value = json.loads(guard_bytes.decode("utf-8"))
        if not isinstance(value, Mapping):
            raise TaskEvaluationLaunchError(f"json_object_required:{guard_path.name}")
        guard = dict(value)
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
                    "sha256:" + hashlib.sha256(guard_bytes).hexdigest()
                    if guard_bytes is not None
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

    terminal_provider_zero_rows: list[dict[str, Any]] = []
    receipt_paths = sorted(state.glob("*/launch_receipt.json"))
    for receipt_path in receipt_paths:
        run_root = receipt_path.parent
        try:
            receipt = _read(receipt_path)
            # Dry dispatch never mutates a provider.  It must not manufacture a
            # paid-run closure obligation simply because it shares the same
            # state-machine and WebApp sync path.
            if (
                receipt.get("execute_requested") is not True
                or receipt.get("provider_mutation_attempted") is not True
            ):
                continue
            profile = _read(run_root / "launch_profile.json")
            terminal_provider_zero_rows.append(
                _reconcile_terminal_provider_zero(
                    run_root=run_root,
                    receipt=receipt,
                    profile=profile,
                    guard=guard,
                    guard_path=guard_path,
                    guard_bytes=guard_bytes,
                    guard_error=guard_error,
                    observed_at=observed_at,
                )
            )
        except (OSError, json.JSONDecodeError, TaskEvaluationLaunchError) as exc:
            terminal_provider_zero_rows.append({
                "launch_id": run_root.name,
                "status": "provider_zero_reconciliation_blocked",
                "provider_zero_confirmed": False,
                "provider_mutation_performed": False,
                "allocator_invoked": False,
                "automatic_retry_performed": False,
                "blockers": ["terminal_provider_zero_reconciliation_input_invalid"],
                "error_type": type(exc).__name__,
            })

    sync_rows: list[dict[str, Any]] = []
    from .task_evaluation_launch_webapp_sync import sync_launch_receipt_to_webapp

    for receipt_path in receipt_paths:
        run_root = receipt_path.parent
        try:
            receipt = _read(receipt_path)
            succeeded_path = run_root / WEBAPP_SYNC_SUCCEEDED_FILENAME
            if succeeded_path.is_file():
                sync_rows.append(
                    validated_succeeded_webapp_sync_row(
                        receipt=receipt,
                        attempt=_read(succeeded_path),
                    )
                )
                continue
            terminal_unmatched_path = run_root / WEBAPP_SYNC_TERMINAL_UNMATCHED_FILENAME
            if terminal_unmatched_path.is_file():
                sync_rows.append(
                    _validated_terminal_unmatched_webapp_sync_row(
                        run_root=run_root,
                        receipt=receipt,
                        unmatched=_read(terminal_unmatched_path),
                    )
                )
                continue
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
                _write_immutable(run_root / WEBAPP_SYNC_SUCCEEDED_FILENAME, attempt)
            if (
                sync_result.get("status") == "failed"
                and sync_result.get("reason") == "http_error:404"
            ):
                unmatched = _terminal_unmatched_webapp_sync_receipt(
                    receipt=receipt,
                    attempt=attempt,
                )
                _write_immutable(run_root / WEBAPP_SYNC_TERMINAL_UNMATCHED_FILENAME, unmatched)
                sync_rows.append(
                    _validated_terminal_unmatched_webapp_sync_row(
                        run_root=run_root,
                        receipt=receipt,
                        unmatched=unmatched,
                    )
                )
            elif sync_result.get("status") == "succeeded":
                sync_rows.append(
                    validated_succeeded_webapp_sync_row(
                        receipt=receipt,
                        attempt=attempt,
                    )
                )
            else:
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
                "provider_zero_pending",
                "provider_zero_reconciliation_blocked",
                "webapp_sync_retry_exhausted",
                "webapp_sync_not_configured",
                "webapp_sync_failed",
                "webapp_sync_reconciliation_blocked",
            }
            for row in [*rows, *terminal_provider_zero_rows, *sync_rows]
        ) else "blocked",
        "processing_count": len(rows),
        "launches": rows,
        "terminal_provider_zero": terminal_provider_zero_rows,
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
