"""Consume one activated internal policy canary through the canonical allocator.

This module is intentionally canary-only.  It cannot dispatch qualified
evaluations, cannot promote a scene, and cannot retry a paid allocation.  Its
closeout is resumable: once allocator output exists, later invocations may
collect official billing, fresh provider-zero, Website sync, and notification
readback without ever invoking the allocator again.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any, Callable, Mapping, Sequence

from .common import write_json
from .decision_evidence_contracts import canonical_digest, cross_runtime_canonical_digest
from .control_plane_disk_budget import (
    ControlPlaneDiskBudgetError,
    reserve_control_plane_disk,
)
from .control_plane_storage_pins import (
    ControlPlaneStoragePinError,
    pins_root_from_environment,
    release_storage_pin,
)
from .native_task_arena_policy_canary_bundle import (
    build_policy_canary_session_bundle,
)
from .native_task_arena_policy_canary_session import (
    CANDIDATE_IDS,
    CLAIM_CEILING,
    LEARNED_ROLLOUT_COUNT,
    PROBE_KIND,
    RUN_KIND,
    build_session_authority,
    validate_provider_bundle,
    validate_runtime_input_manifest,
)
from .task_evaluation_policy_canary_result_projection import (
    build_policy_canary_result_projection,
    derive_policy_canary_episode_blockers,
)
from .task_evaluation_canary_hotfix_overlay import (
    canary_hotfix_execution_release,
    verify_canary_hotfix_overlay,
)
from .task_evaluation_policy_canary_scene_setup import (
    PolicyCanarySetupError,
    materialize_scene839873_policy_canary_setup_from_template,
)
from .task_evaluation_result_delivery import (
    POLICY_CANARY_DELIVERY_SCHEMA_VERSION,
    TaskEvaluationResultDeliveryError,
    materialize_policy_canary_result_delivery,
    materialize_policy_canary_website_delivery,
)
from .task_evaluation_run_webapp_sync import (
    sync_policy_canary_preprovider_blocked_to_webapp,
    sync_task_evaluation_policy_canary_to_webapp,
)
from .task_evaluation_launch_webapp_sync import sync_launch_progress_to_webapp
from .vast_official_billing_extractor import (
    VastOfficialBillingExtractionError,
    materialize_vast_official_same_goal_reconciliation,
    validate_vast_official_same_goal_reconciliation,
)


SCHEMA_VERSION = "task_evaluation_policy_canary_dispatch.v1"
SETUP_SCHEMA_VERSION = "task_evaluation_policy_canary_execution_setup.v1"
ACTIVATION_SCHEMA_VERSION = "task_evaluation_launch_activation_result.v1"
ACTIVATION_FILENAME = "task_evaluation_policy_campaign_activation.v1.json"
ALLOCATOR_MODULE = "blueprint_pipeline.paid_resource_allocator"


class TaskEvaluationPolicyCanaryDispatchError(ValueError):
    """The canary consumer could not prove a safe next step."""


AllocatorRunner = Callable[[Sequence[str]], int]
ProviderZeroCollector = Callable[[], Mapping[str, Any]]
SyncRunner = Callable[..., Mapping[str, Any]]


def _proves_no_provider_allocation(adapter: Mapping[str, Any]) -> bool:
    instance_ids = adapter.get("vast_instance_ids")
    return bool(
        instance_ids in (None, [])
        and adapter.get("provider_mutations_performed") in {0, False}
        and adapter.get("provider_create_attempted") is not True
        and adapter.get("vast_side_effects_may_have_occurred") is not True
        and adapter.get("continuing_spend_from_this_run") is not True
    )


def _sync_pending_progress(
    *,
    runner: SyncRunner,
    run_id: str,
    request_digest: str,
    phase: str,
    blocker: str,
) -> dict[str, Any]:
    return dict(
        runner(
            progress={
                "schema_version": "task_evaluation_launch_progress.v1",
                "launch_id": run_id,
                "run_id": run_id,
                "request_digest": request_digest,
                "phase": phase,
                "phase_status": blocker,
                "observed_at_iso": datetime.now(timezone.utc).isoformat(),
                "elapsed_seconds": 0.0,
            }
        )
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _is_digest(value: Any) -> bool:
    return bool(re.fullmatch(r"sha256:[0-9a-f]{64}", str(value or "")))


def _read(path: str | Path, *, code: str) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TaskEvaluationPolicyCanaryDispatchError(code) from exc
    if source.is_symlink() or not source.is_file() or not isinstance(value, Mapping):
        raise TaskEvaluationPolicyCanaryDispatchError(code)
    return dict(value)


def _record(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    if source.is_symlink() or not source.is_file():
        raise TaskEvaluationPolicyCanaryDispatchError("policy_canary_dispatch_record_invalid")
    return {
        "path": str(source),
        "size_bytes": source.stat().st_size,
        "sha256": _sha256(source),
    }


def _record_path(value: Any, *, code: str) -> Path:
    if not isinstance(value, Mapping):
        raise TaskEvaluationPolicyCanaryDispatchError(code)
    path = Path(str(value.get("path") or "")).expanduser().resolve()
    if (
        path.is_symlink()
        or not path.is_file()
        or path.stat().st_size != value.get("size_bytes")
        or _sha256(path) != value.get("sha256")
    ):
        raise TaskEvaluationPolicyCanaryDispatchError(code)
    return path


def validate_policy_canary_execution_setup(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    setup = json.loads(json.dumps(dict(value), allow_nan=False))
    records = setup.get("records")
    if (
        setup.get("schema_version") != SETUP_SCHEMA_VERSION
        or setup.get("status") != "verified_runnable"
        or str(setup.get("scene_id")) != "839873"
        or not str(setup.get("configured_source_launch_id") or "")
        or not _is_digest(setup.get("scene_revision_digest"))
        or not _is_digest(setup.get("activation_digest"))
        or not re.fullmatch(r"[0-9a-f]{40}", str(setup.get("source_commit") or ""))
        or setup.get("provider") != "vast"
        or tuple(setup.get("candidate_ids") or ()) != CANDIDATE_IDS
        or not str(setup.get("capture_session_id") or "")
        or not str(setup.get("intake_id") or "")
        or not _is_digest(setup.get("request_digest"))
        or not isinstance(records, Mapping)
        or setup.get("setup_digest") != canonical_digest(setup, digest_field="setup_digest")
    ):
        raise TaskEvaluationPolicyCanaryDispatchError("policy_canary_scene839873_setup_invalid")
    expected = {
        "pi05_execution_spec",
        "groot_execution_spec",
        "pi05_checkpoint_inventory",
    }
    if set(records) != expected:
        raise TaskEvaluationPolicyCanaryDispatchError(
            "policy_canary_scene839873_setup_records_invalid"
        )
    for name in sorted(expected):
        _record_path(records[name], code=f"policy_canary_setup_record_invalid:{name}")
    return setup


def _write_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    payload = (json.dumps(dict(value), sort_keys=True, separators=(",", ":")) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError:
        if path.read_bytes() != payload:
            raise TaskEvaluationPolicyCanaryDispatchError(
                f"policy_canary_dispatch_immutable_conflict:{path.name}"
            )


def _event(root: Path, *, stage: str, status: str, **details: Any) -> dict[str, Any]:
    path = root / "status_events.jsonl"
    sequence = 1
    previous_digest = None
    if path.is_file():
        rows = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
        sequence = len(rows) + 1
        if rows:
            previous_digest = json.loads(rows[-1]).get("event_digest")
    observed_at = datetime.now(timezone.utc)
    event = {
        "schema_version": "task_evaluation_policy_canary_status_event.v1",
        "sequence": sequence,
        "stage": stage,
        "status": status,
        "observed_at_iso": observed_at.isoformat(),
        "previous_event_digest": previous_digest,
        **details,
        "event_digest": "",
    }
    event["event_digest"] = canonical_digest(event, digest_field="event_digest")
    with path.open("ab") as stream:
        stream.write((json.dumps(event, sort_keys=True) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())
    return event


def _sync_status_event_progress(
    *,
    root: Path,
    event: Mapping[str, Any],
    run_id: str,
    request_digest: str,
    runner: SyncRunner,
) -> dict[str, Any]:
    """Publish one real canary event without making progress authoritative."""

    observed_at = str(event["observed_at_iso"])
    elapsed_seconds = 0.0
    try:
        first_line = next(
            line
            for line in (root / "status_events.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line
        )
        first = json.loads(first_line)
        started = datetime.fromisoformat(str(first["observed_at_iso"]))
        observed = datetime.fromisoformat(observed_at)
        elapsed_seconds = max(0.0, (observed - started).total_seconds())
    except (OSError, StopIteration, KeyError, TypeError, ValueError, json.JSONDecodeError):
        # The event still carries a real timestamp; elapsed time is optional
        # context and must never make status delivery affect execution.
        elapsed_seconds = 0.0
    progress = {
        "schema_version": "task_evaluation_launch_progress.v1",
        "launch_id": run_id,
        "run_id": run_id,
        "request_digest": request_digest,
        "phase": str(event["stage"]),
        "phase_status": str(event["status"]),
        "observed_at_iso": observed_at,
        "elapsed_seconds": round(elapsed_seconds, 3),
    }
    try:
        result = dict(runner(progress=progress))
    except Exception as exc:  # noqa: BLE001 - observational delivery boundary
        result = {"status": "failed", "reason": type(exc).__name__}
    receipt = {
        "schema_version": "task_evaluation_policy_canary_progress_sync.v1",
        "event_digest": event["event_digest"],
        "run_id": run_id,
        "request_digest": request_digest,
        "phase": progress["phase"],
        "phase_status": progress["phase_status"],
        "sync_status": str(result.get("status") or "failed"),
        "sync_reason": str(result.get("reason") or "") or None,
    }
    with (root / "status_progress_sync.jsonl").open("ab") as stream:
        stream.write((json.dumps(receipt, sort_keys=True) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())
    return result


def _event_and_sync(
    root: Path,
    *,
    stage: str,
    status: str,
    run_id: str,
    request_digest: str,
    runner: SyncRunner,
    **details: Any,
) -> dict[str, Any]:
    event = _event(root, stage=stage, status=status, **details)
    _sync_status_event_progress(
        root=root,
        event=event,
        run_id=run_id,
        request_digest=request_digest,
        runner=runner,
    )
    return event


def _default_allocator_runner(argv: Sequence[str]) -> int:
    completed = subprocess.run(  # nosec B603 - fixed module and validated argv
        [sys.executable, "-m", ALLOCATOR_MODULE, *argv],
        check=False,
        shell=False,
    )
    return int(completed.returncode)


def collect_policy_canary_vast_provider_zero() -> dict[str, Any]:
    """Collect fresh authenticated global Vast inventory without other-provider coupling."""

    from .gpu_render_providers import VastRenderProvider

    inventory = dict(VastRenderProvider().billable_inventory(name_prefix=""))
    resources = inventory.get("resources")
    api_confirmed = inventory.get("api_confirmed") is True
    blockers = [str(item) for item in inventory.get("blockers") or [] if str(item)]
    verified = (
        inventory.get("provider") == "vast"
        and inventory.get("status") == "observed"
        and api_confirmed
        and isinstance(resources, list)
        and not resources
        and not blockers
    )
    guard_path = str(os.getenv("BLUEPRINT_GPU_SPEND_GUARD_REPORT") or "").strip()
    guard_record = None
    if guard_path:
        path = Path(guard_path).expanduser().resolve()
        if path.is_file() and not path.is_symlink():
            guard_record = _record(path)
    value: dict[str, Any] = {
        "schema_version": "task_evaluation_policy_canary_vast_provider_zero.v1",
        "status": "provider_zero_confirmed" if verified else "blocked",
        "provider": "vast",
        "inventory_scope": "global_billable_resources",
        "api_confirmed": api_confirmed,
        "live_instance_count": len(resources) if isinstance(resources, list) else None,
        "provider_zero_verified": verified,
        "global_gpu_guard_snapshot": guard_record,
        "blockers": [] if verified else blockers or ["vast_provider_zero_unproven"],
        "raw_provider_response_recorded": False,
        "receipt_digest": "",
    }
    value["receipt_digest"] = canonical_digest(value, digest_field="receipt_digest")
    return value


def _adapter_instance_ids(adapter: Mapping[str, Any]) -> list[int]:
    values = adapter.get("vast_instance_ids")
    watchdog = adapter.get("independent_watchdog")
    if values is None and isinstance(watchdog, Mapping) and (
        watchdog.get("status") == "provider_terminal"
        and watchdog.get("provider_absence_confirmed") is True
    ):
        values = watchdog.get("instance_ids")
    if not isinstance(values, list) or any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in values
    ):
        return []
    return list(values)


def _join_session_closeout(
    *, inner: Mapping[str, Any], adapter: Mapping[str, Any], provider_zero: Mapping[str, Any]
) -> dict[str, Any]:
    value = json.loads(json.dumps(dict(inner), allow_nan=False))
    episodes = value.get("episodes")
    instance_ids = _adapter_instance_ids(adapter)
    closeout = adapter.get("provider_closeout")
    teardown_complete = (
        isinstance(closeout, Mapping)
        and closeout.get("provider_zero_confirmed") is True
        and closeout.get("warm_session_retained") is False
        and closeout.get("all_staged_objects_absent") is True
        and adapter.get("continuing_spend_from_this_run") is False
    )
    global_zero = (
        provider_zero.get("schema_version") == "task_evaluation_policy_canary_vast_provider_zero.v1"
        and provider_zero.get("provider_zero_verified") is True
        and provider_zero.get("live_instance_count") == 0
        and provider_zero.get("blockers") == []
    )
    value["provider_allocations_observed"] = len(instance_ids)
    value["session_closeout"] = {
        "status": "closed" if teardown_complete and global_zero else "blocked",
        "provider_allocations_observed": len(instance_ids),
        "teardown_completed": teardown_complete,
        "provider_zero_confirmed": global_zero,
    }
    completed = (
        isinstance(episodes, list)
        and len(episodes) == LEARNED_ROLLOUT_COUNT
        and all(row.get("status") == "completed" for row in episodes)
    )
    value["status"] = (
        "completed_unqualified"
        if completed and len(instance_ids) == 1 and teardown_complete and global_zero
        else "blocked"
    )
    blockers = [str(item) for item in value.get("blockers") or [] if str(item)]
    if not completed:
        blockers.extend(derive_policy_canary_episode_blockers(episodes))
    if len(instance_ids) != 1:
        blockers.append("policy_canary_provider_allocation_count_invalid")
    if not teardown_complete:
        blockers.append("policy_canary_teardown_incomplete")
    if not global_zero:
        blockers.append("policy_canary_global_provider_zero_unproven")
    value["blockers"] = sorted(set(blockers))
    value["result_digest"] = canonical_digest(value, digest_field="result_digest")
    return value


def _partial_policy_canary_result(
    *,
    native_path: Path,
    fallback: Mapping[str, Any],
    runtime_inputs: Mapping[str, Any],
    specs: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], Path] | None:
    """Preserve sealed child cells when a later isolated cell times out."""

    if (
        fallback.get("status") == "runtime_completed_unqualified_pending_closeout"
        and isinstance(fallback.get("episodes"), list)
        and len(fallback["episodes"]) == LEARNED_ROLLOUT_COUNT
        and isinstance(fallback.get("artifact_inventory"), list)
    ):
        # The provider already produced the authoritative full-session
        # aggregation. Rebuilding it from child receipts would incorrectly turn
        # a complete run into a "partial" result and would rebind child-owned
        # artifacts that may still have been open when the child sealed.
        return None

    evidence_root = native_path.parent
    cells = list(runtime_inputs.get("cells") or [])
    if len(cells) != 10:
        return None
    partial_episodes: list[dict[str, Any]] = []
    partial_artifacts: list[dict[str, Any]] = []
    completed_indices: set[int] = set()
    observed_indices: set[int] = set()
    for path in sorted(
        evidence_root.glob(
            "cell_runs/*/native_task_arena_policy_canary_session_result.v1.json"
        )
    ):
        try:
            index = int(path.parent.name)
        except ValueError:
            continue
        if index < 0 or index >= len(cells):
            continue
        child = _read(path, code="policy_canary_partial_cell_result_invalid")
        episodes = child.get("episodes")
        if (
            child.get("selected_cell_index") != index
            or child.get("result_digest")
            != canonical_digest(child, digest_field="result_digest")
            or not isinstance(episodes, list)
            or not isinstance(child.get("artifact_inventory"), list)
        ):
            raise TaskEvaluationPolicyCanaryDispatchError(
                "policy_canary_partial_cell_result_invalid"
            )
        child_completed = (
            child.get("status")
            == "runtime_selected_cell_completed_pending_aggregation"
            and len(episodes) == len(CANDIDATE_IDS)
        )
        child_blocked = child.get("status") == "blocked" and not episodes
        if not child_completed and not child_blocked:
            raise TaskEvaluationPolicyCanaryDispatchError(
                "policy_canary_partial_cell_result_invalid"
            )
        expected = {
            (candidate, str(cells[index]["cell_id"]), int(cells[index]["seed"]))
            for candidate in CANDIDATE_IDS
        }
        observed = {
            (
                str(row.get("candidate_id")),
                str(row.get("cell_id")),
                int(row.get("seed")),
            )
            for row in episodes
            if isinstance(row, Mapping)
        }
        if child_completed and observed != expected:
            raise TaskEvaluationPolicyCanaryDispatchError(
                "policy_canary_partial_cell_pairing_invalid"
            )
        if child_blocked and observed:
            raise TaskEvaluationPolicyCanaryDispatchError(
                "policy_canary_partial_cell_pairing_invalid"
            )
        prefix = f"cell_runs/{index:02d}"
        for row in episodes:
            episode = json.loads(json.dumps(dict(row), allow_nan=False))
            evidence = episode.get("evidence_artifacts")
            if isinstance(evidence, Mapping):
                episode["evidence_artifacts"] = {
                    role: (
                        {
                            **dict(record),
                            "relative_path": f"{prefix}/{record['relative_path']}",
                        }
                        if isinstance(record, Mapping)
                        and isinstance(record.get("relative_path"), str)
                        else record
                    )
                    for role, record in evidence.items()
                }
            partial_episodes.append(episode)
        for record in child.get("artifact_inventory") or []:
            if not isinstance(record, Mapping):
                continue
            copied = dict(record)
            if str(copied.get("relative_path") or "").endswith(
                "/worker_console.log"
            ) or copied.get("relative_path") == "worker_console.log":
                # Legacy child receipts could include the parent-owned stdout
                # log before the parent appended the final exit lines. It is a
                # mutable diagnostic, not episode evidence, so partial recovery
                # must not publish its stale digest.
                continue
            if isinstance(copied.get("relative_path"), str):
                copied["relative_path"] = f"{prefix}/{copied['relative_path']}"
            partial_artifacts.append(copied)
        observed_indices.add(index)
        if child_completed:
            completed_indices.add(index)
    if not partial_episodes and not partial_artifacts:
        return None
    gap_root = evidence_root / "partial_terminal_evidence"
    gap_root.mkdir(parents=True, exist_ok=True)
    gap_path = gap_root / "typed_media_gap.json"
    gap_value = {
        "schema_version": "task_evaluation_policy_canary_media_gap.v1",
        "type": "cell_not_completed_before_terminal_failure",
        "reason": (fallback.get("blockers") or ["policy_canary_worker_timeout"])[0],
        "completed_cell_indices": sorted(completed_indices),
        "observed_cell_indices": sorted(observed_indices),
        "candidate_policy_queried": any(
            row.get("candidate_policy_queried") is True for row in partial_episodes
        ),
    }
    write_json(gap_path, gap_value)
    observed_keys = {
        (str(row["candidate_id"]), str(row["cell_id"]), int(row["seed"]))
        for row in partial_episodes
    }
    missing_episodes = []
    for candidate in CANDIDATE_IDS:
        spec = specs[candidate]
        for cell in cells:
            key = (candidate, str(cell["cell_id"]), int(cell["seed"]))
            if key in observed_keys:
                continue
            missing_episodes.append(
                {
                    "candidate_id": candidate,
                    "cell_id": cell["cell_id"],
                    "seed": cell["seed"],
                    "status": "blocked",
                    "candidate_policy_queried": False,
                    "actions_reached_robot": False,
                    "arm_moved": False,
                    "policy_outcome_interpretable": False,
                    "typed_harness_failure": (
                        "cell_not_completed_before_terminal_failure"
                    ),
                    "checkpoint_digest": spec["checkpoint_digest"],
                    "runtime_identity_digest": spec["runtime_identity_digest"],
                    "reset_state_digest": canonical_digest(
                        {
                            "resolved_scenario": cell["resolved_scenario"],
                            "seed": cell["seed"],
                            "execution_performed": False,
                        }
                    ),
                    "visual_evidence": {
                        "media_gap": {
                            "type": gap_value["type"],
                            "reason": gap_value["reason"],
                        }
                    },
                    "evidence_artifacts": {},
                }
            )
    value: dict[str, Any] = {
        "schema_version": "native_task_arena_policy_canary_session_result.v1",
        "status": "blocked",
        "run_kind": RUN_KIND,
        "claim_ceiling": CLAIM_CEILING,
        "candidate_ids": list(CANDIDATE_IDS),
        "episodes_per_policy": 10,
        "learned_policy_rollout_count": 20,
        "episodes": [*partial_episodes, *missing_episodes],
        "artifact_inventory": [
            *partial_artifacts,
            {
                "role": "typed_media_gap",
                "relative_path": gap_path.relative_to(evidence_root).as_posix(),
                "media_type": "application/json",
                "size_bytes": gap_path.stat().st_size,
                "sha256": _sha256(gap_path),
            },
        ],
        "candidate_policy_queried": any(
            row.get("candidate_policy_queried") is True for row in partial_episodes
        ),
        "completed_cell_count": len(completed_indices),
        "incomplete_cell_count": len(cells) - len(completed_indices),
        "official_ranking_performed": False,
        "scene_promotion_performed": False,
        "blockers": sorted(
            set(
                [str(item) for item in fallback.get("blockers") or []]
                + [
                    "policy_canary_partial_cell_results_preserved",
                    f"policy_canary_incomplete_cell_count:{len(cells) - len(completed_indices)}",
                ]
            )
        ),
        "result_digest": "",
    }
    value["result_digest"] = canonical_digest(value, digest_field="result_digest")
    output_path = evidence_root / "policy_canary_partial_provider_result.v1.json"
    write_json(output_path, value)
    return value, output_path


def _materialize_official_billing_if_posted(
    *,
    billing_audit_root: str | Path,
    adapter_result_path: Path,
    adapter: Mapping[str, Any],
    launch_label: str,
    output_path: Path,
) -> bool:
    if output_path.is_file():
        validate_vast_official_same_goal_reconciliation(output_path)
        return True
    instance_ids = _adapter_instance_ids(adapter)
    if len(instance_ids) != 1:
        return False
    audit = Path(billing_audit_root).expanduser().resolve()
    if not audit.is_dir() or audit.is_symlink():
        return False
    candidates = sorted(
        audit.rglob("provider_billing_source_receipt.json"),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )
    for source in candidates:
        try:
            materialize_vast_official_same_goal_reconciliation(
                provider_billing_source_receipt_path=source,
                expected_instances=[(int(instance_ids[0]), launch_label, adapter_result_path)],
                output_path=output_path,
            )
        except (OSError, VastOfficialBillingExtractionError):
            continue
        return True
    return False


def _projection(
    *, setup: Mapping[str, Any], result: Mapping[str, Any], delivery: Mapping[str, Any]
) -> dict[str, Any]:
    return build_policy_canary_result_projection(
        setup=setup,
        result=result,
        delivery=delivery,
        error_factory=TaskEvaluationPolicyCanaryDispatchError,
    )
AccessChecker = Callable[[Path, int], bool]


def _release_activation_storage_pin(root: Path) -> None:
    """Let the storage reaper retire this activation's derived inputs.

    The dispatch directory is named by the activation id, so the terminal
    receipt is the moment the preparation, compilation, and launch set it
    consumed stop being needed.  Best effort: a missing pin or ledger never
    disturbs a sealed receipt.
    """

    pins_root = pins_root_from_environment()
    if pins_root is None:
        return
    try:
        release_storage_pin(pins_root=pins_root, kind="activation", owner_id=root.name)
    except (ControlPlaneStoragePinError, OSError):
        return


def _default_access_checker(path: Path, mode: int) -> bool:
    """Ask the kernel whether this process's effective identity may use ``path``."""

    if os.access in os.supports_effective_ids:
        return os.access(path, mode, effective_ids=True)
    return os.access(path, mode)


def _nearest_existing_ancestor(path: Path) -> Path:
    candidate = path
    while not candidate.exists():
        parent = candidate.parent
        if parent == candidate:
            return candidate
        candidate = parent
    return candidate


def service_access_blockers(
    *,
    readable_files: Mapping[str, Path],
    readable_directories: Mapping[str, Path],
    writable_directories: Mapping[str, Path],
    access: AccessChecker = _default_access_checker,
) -> list[str]:
    """Return blockers for every input the dispatcher's own identity cannot use.

    The dispatcher runs as a hardened service account.  Two paid attempts were
    lost to inputs staged by root that this identity could not open (a signed
    overlay archive, a run directory pre-created for a machine avoidlist).
    Every role here is checked with the kernel's answer for the effective
    identity, before any read, write, or allocator invocation, and each
    blocker names the role rather than a host path.
    """

    blockers: list[str] = []

    def parent_untraversable(target: Path) -> bool:
        parent = _nearest_existing_ancestor(target.parent)
        return parent.is_dir() and not access(parent, os.X_OK)

    # Absent inputs are left to the dispatch's own typed schema and record
    # errors; this check only speaks to what exists and cannot be used.
    for role, path in sorted(readable_files.items()):
        target = Path(path).expanduser()
        if parent_untraversable(target):
            blockers.append(f"policy_canary_dispatch_input_unreadable:{role}")
        elif target.is_symlink():
            blockers.append(f"policy_canary_dispatch_input_symlink:{role}")
        elif target.is_file() and not access(target, os.R_OK):
            blockers.append(f"policy_canary_dispatch_input_unreadable:{role}")
    for role, path in sorted(readable_directories.items()):
        target = Path(path).expanduser()
        if parent_untraversable(target):
            blockers.append(f"policy_canary_dispatch_input_unreadable:{role}")
        elif target.is_dir() and not access(target, os.R_OK | os.X_OK):
            blockers.append(f"policy_canary_dispatch_input_unreadable:{role}")
    for role, path in sorted(writable_directories.items()):
        target = _nearest_existing_ancestor(Path(path).expanduser().resolve())
        if not target.is_dir():
            blockers.append(f"policy_canary_dispatch_output_not_directory:{role}")
        elif not access(target, os.W_OK | os.X_OK):
            blockers.append(f"policy_canary_dispatch_output_unwritable:{role}")
    return sorted(set(blockers))


def _dispatch_input_access_blockers(
    *,
    execution_setup_path: Path,
    activation_result_path: Path,
    output_root: Path,
    hotfix_overlay_path: Path | None,
    machine_avoidlist_path: Path | None,
    access: AccessChecker,
) -> list[str]:
    """Check every path the dispatch will touch, resolved from the two entry files.

    Inputs are located exactly as the dispatch locates them, but read only
    enough to find the next path; digests and schemas are verified by the
    dispatch itself once access is proven.
    """

    readable_files: dict[str, Path] = {
        "execution_setup": execution_setup_path,
        "activation_result": activation_result_path,
    }
    readable_directories: dict[str, Path] = {}
    if hotfix_overlay_path is not None:
        readable_files["hotfix_overlay"] = hotfix_overlay_path
    if machine_avoidlist_path is not None:
        readable_files["machine_avoidlist"] = machine_avoidlist_path
    blockers = service_access_blockers(
        readable_files=readable_files,
        readable_directories=readable_directories,
        writable_directories={"output_root": output_root},
        access=access,
    )
    if blockers:
        return blockers

    def _paths_from(mapping: Mapping[str, Any], roles: Mapping[str, str]) -> None:
        for role, key in roles.items():
            record = mapping.get(key)
            raw = record.get("path") if isinstance(record, Mapping) else record
            if isinstance(raw, str) and raw:
                readable_files[role] = Path(raw)

    try:
        setup = json.loads(execution_setup_path.read_text(encoding="utf-8"))
        activation_result = json.loads(activation_result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        # The dispatch reports the typed schema/digest failure itself.
        return []
    if isinstance(setup, Mapping) and isinstance(setup.get("records"), Mapping):
        _paths_from(
            setup["records"],
            {
                "pi05_execution_spec": "pi05_execution_spec",
                "groot_execution_spec": "groot_execution_spec",
                "pi05_checkpoint_inventory": "pi05_checkpoint_inventory",
            },
        )
    runtime_inputs: Mapping[str, Any] | None = None
    if isinstance(activation_result, Mapping):
        raw_runtime = activation_result.get("policy_canary_runtime_inputs_path")
        if isinstance(raw_runtime, str) and raw_runtime:
            runtime_path = Path(raw_runtime).expanduser()
            readable_files["runtime_inputs"] = runtime_path
            readable_files["activation_manifest"] = runtime_path.parent / ACTIVATION_FILENAME
            readable_directories["activation_root"] = runtime_path.parent
            if runtime_path.is_file() and access(runtime_path, os.R_OK):
                try:
                    loaded = json.loads(runtime_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    loaded = None
                runtime_inputs = loaded if isinstance(loaded, Mapping) else None
    if runtime_inputs is not None:
        _paths_from(
            runtime_inputs,
            {
                "base_native_packet": "base_native_packet",
                "runtime_source": "runtime_source",
                "construction_result": "construction_result",
            },
        )
        packet = readable_files.get("base_native_packet")
        if packet is not None:
            readable_directories["base_native_packet_dir"] = packet.parent
        source = readable_files.get("runtime_source")
        if source is not None:
            readable_directories["runtime_source_dir"] = source.parent
    return service_access_blockers(
        readable_files=readable_files,
        readable_directories=readable_directories,
        writable_directories={"output_root": output_root},
        access=access,
    )


def _finish_policy_canary_delivery(
    *,
    root: Path,
    setup: Mapping[str, Any],
    runtime_inputs: Mapping[str, Any],
    authority: Mapping[str, Any],
    bundle: Mapping[str, Any],
    joined: Mapping[str, Any],
    joined_path: Path,
    delivery: Mapping[str, Any],
    closure: Mapping[str, Mapping[str, Any]],
    sync_runner: SyncRunner,
    progress_sync_runner: SyncRunner,
    allocator_invoked: bool,
) -> dict[str, Any]:
    """Publish one already sealed delivery without re-entering provider closeout."""

    website_delivery = materialize_policy_canary_website_delivery(
        run_root=root,
        delivery=delivery,
    )
    projection = _projection(
        setup=setup,
        result=joined,
        delivery=website_delivery,
    )
    _event_and_sync(
        root,
        stage="report_generating",
        status="completed",
        run_id=str(joined["run_id"]),
        request_digest=str(setup["request_digest"]),
        runner=progress_sync_runner,
    )
    sync = dict(
        sync_runner(
            capture_session_id=setup["capture_session_id"],
            intake_id=setup["intake_id"],
            run_id=str(joined["run_id"]),
            request_digest=setup["request_digest"],
            configuration_digest=runtime_inputs["configuration_digest"],
            result_status=str(joined["status"]),
            result_delivery=website_delivery,
            policy_canary_result=projection,
        )
    )
    notification = sync.get("notification_delivery")
    if (
        sync.get("status") != "succeeded"
        or not isinstance(notification, Mapping)
        or notification.get("status") not in {"accepted", "delivered", "failed"}
    ):
        pending = {
            "schema_version": SCHEMA_VERSION,
            "status": "awaiting_website_sync_or_notification",
            "run_id": joined["run_id"],
            "allocator_invoked": allocator_invoked,
            "automatic_retry_performed": False,
            "blockers": ["policy_canary_website_notification_readback_missing"],
        }
        write_json(root / "dispatch_pending.json", pending)
        return pending
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "status": joined["status"],
        "run_id": joined["run_id"],
        "run_kind": RUN_KIND,
        "claim_ceiling": CLAIM_CEILING,
        "authority_digest": authority["authority_digest"],
        "bundle_sha256": bundle["bundle_sha256"],
        "terminal_result": _record(joined_path),
        "result_delivery_digest": website_delivery["delivery_digest"],
        "policy_canary_projection_digest": projection["projection_digest"],
        "notification_delivery": sync["notification_delivery"],
        "official_billing": closure["billing"],
        "teardown": closure["teardown"],
        "provider_zero": closure["provider_zero"],
        "allocator_invoked": allocator_invoked,
        "automatic_retry_performed": False,
        "scene_promotion_performed": False,
        "official_ranking_performed": False,
        "retry_cap": 0,
        "receipt_digest": "",
    }
    receipt["receipt_digest"] = canonical_digest(
        receipt, digest_field="receipt_digest"
    )
    _write_exclusive(root / "dispatch_receipt.json", receipt)
    _release_activation_storage_pin(root)
    _event_and_sync(
        root,
        stage="billing_teardown",
        status="completed",
        run_id=str(joined["run_id"]),
        request_digest=str(setup["request_digest"]),
        runner=progress_sync_runner,
    )
    _event_and_sync(
        root,
        stage=str(joined["status"]),
        status="completed",
        run_id=str(joined["run_id"]),
        request_digest=str(setup["request_digest"]),
        runner=progress_sync_runner,
    )
    return receipt


def _sealed_provider_zero(path: Path) -> dict[str, Any] | None:
    """Reuse immutable terminal absence evidence across billing closeout resumes."""

    if not path.is_file():
        return None
    try:
        value = _read(path, code="policy_canary_materialized_provider_zero_invalid")
    except TaskEvaluationPolicyCanaryDispatchError:
        return None
    if (
        value.get("schema_version")
        != "task_evaluation_policy_canary_vast_provider_zero.v1"
        or value.get("status") != "provider_zero_confirmed"
        or value.get("api_confirmed") is not True
        or value.get("provider_zero_verified") is not True
        or value.get("live_instance_count") != 0
        or value.get("blockers") != []
        or value.get("receipt_digest")
        != canonical_digest(value, digest_field="receipt_digest")
    ):
        return None
    return value


def _resume_materialized_policy_canary_delivery(
    *,
    root: Path,
    setup: Mapping[str, Any],
    runtime_inputs: Mapping[str, Any],
    authority: Mapping[str, Any],
    bundle: Mapping[str, Any],
    adapter: Mapping[str, Any],
    sync_runner: SyncRunner,
    progress_sync_runner: SyncRunner = sync_launch_progress_to_webapp,
) -> dict[str, Any] | None:
    """Resume only Website publication after evidence packaging already sealed.

    Provider-zero receipts contain observation timestamps, so collecting a new
    one after an immutable delivery was written would necessarily change the
    package. Resume from the exact sealed result and closure instead.
    """

    joined_path = root / "policy_canary_terminal_result.json"
    delivery_path = root / "artifacts" / "result_delivery" / "delivery.json"
    if not delivery_path.exists():
        return None
    if not joined_path.is_file() or not delivery_path.is_file():
        raise TaskEvaluationPolicyCanaryDispatchError(
            "policy_canary_materialized_delivery_partial"
        )
    joined = _read(joined_path, code="policy_canary_terminal_result_invalid")
    delivery = _read(delivery_path, code="policy_canary_result_delivery_invalid")
    if (
        joined.get("run_kind") != RUN_KIND
        or joined.get("claim_ceiling") != CLAIM_CEILING
        or joined.get("status") not in {"completed_unqualified", "blocked", "cancelled"}
        or joined.get("result_digest")
        != canonical_digest(joined, digest_field="result_digest")
        or delivery.get("schema_version") != POLICY_CANARY_DELIVERY_SCHEMA_VERSION
        or delivery.get("run_id") != joined.get("run_id")
        or delivery.get("result_status") != joined.get("status")
        or delivery.get("claim_ceiling") != CLAIM_CEILING
        or delivery.get("delivery_digest")
        != cross_runtime_canonical_digest(delivery, digest_field="delivery_digest")
    ):
        raise TaskEvaluationPolicyCanaryDispatchError(
            "policy_canary_materialized_delivery_invalid"
        )
    report_path = root / "artifacts" / "result_delivery" / "policy_canary_full_report.json"
    report_record = (delivery.get("report") or {}).get("machine_readable_report")
    if (
        not isinstance(report_record, Mapping)
        or not report_path.is_file()
        or report_record.get("digest") != _sha256(report_path)
        or report_record.get("size_bytes") != report_path.stat().st_size
        or _read(report_path, code="policy_canary_full_report_invalid") != joined
    ):
        raise TaskEvaluationPolicyCanaryDispatchError(
            "policy_canary_materialized_report_invalid"
        )
    billing_path = root / "official_billing_reconciliation.json"
    validate_vast_official_same_goal_reconciliation(billing_path)
    teardown_path = Path(str(adapter.get("teardown_manifest_path") or "")).resolve()
    provider_zero_path = root / "post_teardown_global_provider_zero.json"
    provider_zero = _read(
        provider_zero_path,
        code="policy_canary_materialized_provider_zero_invalid",
    )
    if provider_zero.get("provider_zero_verified") is not True:
        raise TaskEvaluationPolicyCanaryDispatchError(
            "policy_canary_materialized_provider_zero_invalid"
        )
    closure = {
        "billing": {**_record(billing_path), "official_billing_sealed": True},
        "teardown": {**_record(teardown_path), "teardown_completed": True},
        "provider_zero": {
            **_record(provider_zero_path),
            "provider_zero_verified": True,
        },
    }
    delivered_closure = delivery.get("closure")
    if not isinstance(delivered_closure, Mapping):
        raise TaskEvaluationPolicyCanaryDispatchError(
            "policy_canary_materialized_closure_invalid"
        )
    for role, record in closure.items():
        delivered = delivered_closure.get(role)
        if (
            not isinstance(delivered, Mapping)
            or delivered.get("digest") != record["sha256"]
            or delivered.get("size_bytes") != record["size_bytes"]
        ):
            raise TaskEvaluationPolicyCanaryDispatchError(
                f"policy_canary_materialized_closure_invalid:{role}"
            )
    _event_and_sync(
        root,
        stage="artifacts_syncing",
        status="resumed_from_sealed_delivery",
        run_id=str(joined["run_id"]),
        request_digest=str(setup["request_digest"]),
        runner=progress_sync_runner,
    )
    return _finish_policy_canary_delivery(
        root=root,
        setup=setup,
        runtime_inputs=runtime_inputs,
        authority=authority,
        bundle=bundle,
        joined=joined,
        joined_path=joined_path,
        delivery=delivery,
        closure=closure,
        sync_runner=sync_runner,
        progress_sync_runner=progress_sync_runner,
        allocator_invoked=False,
    )


def dispatch_policy_canary_activation(
    *,
    activation_result_path: str | Path,
    execution_setup_path: str | Path,
    output_root: str | Path,
    implementation_commit: str,
    execute: bool = False,
    official_billing_receipt_path: str | Path | None = None,
    billing_audit_root: str | Path | None = None,
    hotfix_overlay_path: str | Path | None = None,
    machine_avoidlist_path: str | Path | None = None,
    allocator_runner: AllocatorRunner | None = None,
    provider_zero_collector: ProviderZeroCollector = collect_policy_canary_vast_provider_zero,
    sync_runner: SyncRunner = sync_task_evaluation_policy_canary_to_webapp,
    blocked_sync_runner: SyncRunner = sync_policy_canary_preprovider_blocked_to_webapp,
    progress_sync_runner: SyncRunner = sync_launch_progress_to_webapp,
    access: AccessChecker = _default_access_checker,
) -> dict[str, Any]:
    """Dispatch or resume exactly one Scene 839873 policy canary."""

    if not re.fullmatch(r"[0-9a-f]{40}", implementation_commit):
        raise TaskEvaluationPolicyCanaryDispatchError(
            "policy_canary_dispatch_source_commit_invalid"
        )
    # Prove the service identity can use every input and the run directory
    # before anything is read for real or written.  A denial here is a typed
    # pre-provider blocker instead of an OSError after the allocator ran.
    access_blockers = _dispatch_input_access_blockers(
        execution_setup_path=Path(execution_setup_path).expanduser(),
        activation_result_path=Path(activation_result_path).expanduser(),
        output_root=Path(output_root).expanduser(),
        hotfix_overlay_path=(
            Path(hotfix_overlay_path).expanduser() if hotfix_overlay_path is not None else None
        ),
        machine_avoidlist_path=(
            Path(machine_avoidlist_path).expanduser()
            if machine_avoidlist_path is not None
            else None
        ),
        access=access,
    )
    if access_blockers:
        raise TaskEvaluationPolicyCanaryDispatchError(
            "policy_canary_dispatch_service_access_denied:" + ",".join(access_blockers)
        )
    setup = validate_policy_canary_execution_setup(
        _read(
            execution_setup_path,
            code="policy_canary_scene839873_setup_receipt_missing",
        )
    )
    activation_result_path = Path(activation_result_path).expanduser().resolve()
    activation_result = _read(
        activation_result_path, code="policy_canary_activation_result_invalid"
    )
    if (
        activation_result.get("schema_version") != ACTIVATION_SCHEMA_VERSION
        or activation_result.get("status") != "policy_campaign_queue_materialized_no_execution"
        or activation_result.get("run_kind") != RUN_KIND
        or activation_result.get("claim_ceiling") != CLAIM_CEILING
        or activation_result.get("provider_mutation_performed") is not False
        or activation_result.get("paid_execution_requested") is not False
        or activation_result.get("result_digest")
        != canonical_digest(activation_result, digest_field="result_digest")
    ):
        raise TaskEvaluationPolicyCanaryDispatchError("policy_canary_activation_result_invalid")
    runtime_path = (
        Path(str(activation_result.get("policy_canary_runtime_inputs_path") or ""))
        .expanduser()
        .resolve()
    )
    runtime_inputs = validate_runtime_input_manifest(
        _read(runtime_path, code="policy_canary_runtime_inputs_invalid")
    )
    activation_path = runtime_path.parent / ACTIVATION_FILENAME
    activation = _read(activation_path, code="policy_canary_activation_manifest_invalid")
    resource = runtime_inputs.get("resource_authority")
    if (
        setup["source_commit"] != implementation_commit
        or setup["activation_digest"] != activation["activation_digest"]
        or setup["scene_revision_digest"] != runtime_inputs.get("scene_revision_digest")
        or activation_result.get("source_commit") != implementation_commit
        or not isinstance(resource, Mapping)
        or resource.get("user_confirmed") is not True
        or not str(resource.get("resource_name") or "").startswith(
            "blueprint-native-task-policy-canary-"
        )
    ):
        raise TaskEvaluationPolicyCanaryDispatchError(
            "policy_canary_dispatch_activation_setup_mismatch"
        )
    root = Path(output_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    _event_and_sync(
        root,
        stage="queued",
        status="completed",
        run_id=str(activation["run_id"]),
        request_digest=str(setup["request_digest"]),
        runner=progress_sync_runner,
    )
    hotfix_manifest = (
        verify_canary_hotfix_overlay(hotfix_overlay_path)
        if hotfix_overlay_path is not None
        else None
    )
    execution_release = (
        canary_hotfix_execution_release(hotfix_manifest)
        if hotfix_manifest is not None
        else None
    )
    if (
        execution_release is not None
        and execution_release["base_release_commit"] != implementation_commit
    ):
        raise TaskEvaluationPolicyCanaryDispatchError(
            "policy_canary_hotfix_base_release_mismatch"
        )
    authority = build_session_authority(
        activation_manifest=activation,
        activation_record=_record(activation_path),
        runtime_inputs=runtime_inputs,
        runtime_input_record=_record(runtime_path),
        resource_name=str(resource["resource_name"]),
        hard_cap_usd=float(resource["hard_cap_usd"]),
        hard_ttl_seconds=int(resource["hard_ttl_seconds"]),
        execution_release=execution_release,
    )
    authority_path = root / "policy_canary_session_authority.json"
    _write_exclusive(authority_path, authority)
    records = setup["records"]
    _event_and_sync(
        root,
        stage="preparing",
        status="running",
        run_id=str(activation["run_id"]),
        request_digest=str(setup["request_digest"]),
        runner=progress_sync_runner,
    )
    bundle_receipt_path = (
        root / "bundle" / "native_task_arena_policy_canary_session_bundle_receipt.v1.json"
    )
    if bundle_receipt_path.is_file():
        bundle = validate_provider_bundle(
            _read(bundle_receipt_path, code="policy_canary_bundle_receipt_invalid"),
            authority=authority,
        )
    else:
        bundle = build_policy_canary_session_bundle(
            job_dir=root / "bundle",
            packet_dir=Path(runtime_inputs["base_native_packet"]["path"]).parent,
            runtime_source_packet_receipt=runtime_inputs["runtime_source"]["path"],
            runtime_input_manifest_path=runtime_path,
            session_authority_path=authority_path,
            pi05_execution_spec_path=records["pi05_execution_spec"]["path"],
            groot_execution_spec_path=records["groot_execution_spec"]["path"],
            pi05_checkpoint_inventory_path=records["pi05_checkpoint_inventory"]["path"],
            implementation_commit=implementation_commit,
            hotfix_overlay_path=hotfix_overlay_path,
        )
    _event_and_sync(
        root,
        stage="preparing",
        status="completed",
        run_id=str(activation["run_id"]),
        request_digest=str(setup["request_digest"]),
        runner=progress_sync_runner,
        authority_digest=authority["authority_digest"],
        bundle_sha256=bundle["bundle_sha256"],
    )
    admission_path = root / "paid_admission.json"
    adapter_path = root / "allocator_result.json"
    invocation_started_path = root / "allocator_invocation_started.json"
    invocation_finished_path = root / "allocator_invocation_finished.json"
    argv = [
        "gpu-canary",
        "--provider",
        "vast",
        "--probe-kind",
        PROBE_KIND,
        "--native-task-arena-policy-canary-session-authority",
        str(authority_path),
        "--native-task-arena-policy-canary-session-bundle-receipt",
        str(bundle_receipt_path),
        "--adp-job-dir",
        str(root / "allocator"),
        "--adp-max-hourly-rate-usd",
        str(resource["maximum_hourly_rate_usd"]),
        "--adp-max-spend-usd",
        str(resource["hard_cap_usd"]),
        "--adp-hard-ttl-seconds",
        str(resource["hard_ttl_seconds"]),
        "--admission-out",
        str(admission_path),
        "--adapter-output",
        str(adapter_path),
    ]
    if machine_avoidlist_path is not None:
        # A canonical, access-checked avoidlist input replaces hand-staging a
        # file inside the run directory (which root once created, locking the
        # service account out of its own status journal).
        argv.extend(
            [
                "--adp-machine-avoidlist",
                str(Path(machine_avoidlist_path).expanduser().resolve()),
            ]
        )
    allocator_invoked = False
    if not adapter_path.is_file():
        if invocation_started_path.is_file():
            raise TaskEvaluationPolicyCanaryDispatchError(
                "policy_canary_allocator_previous_invocation_without_result"
            )
        _event_and_sync(
            root,
            stage="provider_allocating",
            status="running",
            run_id=str(activation["run_id"]),
            request_digest=str(setup["request_digest"]),
            runner=progress_sync_runner,
        )
        if execute:
            argv.append("--execute")
        invocation_started = {
            "schema_version": "task_evaluation_policy_canary_allocator_invocation.v1",
            "status": "started",
            "run_id": activation["run_id"],
            "execute": execute,
            "allocator_argv_digest": canonical_digest({"argv": argv}),
            "allocator_invoked": True,
            "automatic_retry_authorized": False,
            "invocation_digest": "",
        }
        invocation_started["invocation_digest"] = canonical_digest(
            invocation_started,
            digest_field="invocation_digest",
        )
        _write_exclusive(invocation_started_path, invocation_started)
        exit_code = int((allocator_runner or _default_allocator_runner)(argv))
        allocator_invoked = True
        invocation_finished = {
            "schema_version": "task_evaluation_policy_canary_allocator_invocation.v1",
            "status": "finished",
            "run_id": activation["run_id"],
            "execute": execute,
            "allocator_argv_digest": invocation_started["allocator_argv_digest"],
            "exit_code": exit_code,
            "adapter_result_present": adapter_path.is_file(),
            "allocator_invoked": True,
            "automatic_retry_performed": False,
            "invocation_digest": "",
        }
        invocation_finished["invocation_digest"] = canonical_digest(
            invocation_finished,
            digest_field="invocation_digest",
        )
        _write_exclusive(invocation_finished_path, invocation_finished)
        if not adapter_path.is_file():
            raise TaskEvaluationPolicyCanaryDispatchError(
                f"policy_canary_allocator_exit_{exit_code}_without_result"
            )
    adapter = _read(adapter_path, code="policy_canary_allocator_result_invalid")
    if not execute:
        receipt = {
            "schema_version": SCHEMA_VERSION,
            "status": "prepared_no_execution",
            "run_id": activation["run_id"],
            "run_kind": RUN_KIND,
            "claim_ceiling": CLAIM_CEILING,
            "authority_digest": authority["authority_digest"],
            "bundle_sha256": bundle["bundle_sha256"],
            "allocator_argv": argv,
            "allocator_invoked": allocator_invoked,
            "provider_mutation_performed": False,
            "retry_cap": 0,
            "receipt_digest": "",
        }
        receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
        _write_exclusive(root / "dispatch_receipt.json", receipt)
        return receipt

    resumed = _resume_materialized_policy_canary_delivery(
        root=root,
        setup=setup,
        runtime_inputs=runtime_inputs,
        authority=authority,
        bundle=bundle,
        adapter=adapter,
        sync_runner=sync_runner,
        progress_sync_runner=progress_sync_runner,
    )
    if resumed is not None:
        return resumed

    if _proves_no_provider_allocation(adapter):
        blockers = list(adapter.get("blockers") or ["policy_canary_provider_not_allocated"])
        terminal_sync = dict(
            blocked_sync_runner(
                activation_id=activation_result["activation_id"],
                capture_session_id=setup["capture_session_id"],
                intake_id=setup["intake_id"],
                request_digest=setup["request_digest"],
                blockers=blockers,
            )
        )
        blocked = {
            "schema_version": SCHEMA_VERSION,
            "status": (
                "blocked_without_provider_allocation"
                if terminal_sync.get("status") == "succeeded"
                else "blocked_without_provider_allocation_awaiting_notification"
            ),
            "run_id": activation["run_id"],
            "run_kind": RUN_KIND,
            "claim_ceiling": CLAIM_CEILING,
            "allocator_invoked": allocator_invoked,
            "provider_allocation_performed": False,
            "provider_mutation_performed": False,
            "automatic_retry_performed": False,
            "blockers": blockers,
            "terminal_sync": terminal_sync,
            "receipt_digest": "",
        }
        blocked["receipt_digest"] = canonical_digest(
            blocked, digest_field="receipt_digest"
        )
        write_json(root / "no_provider_allocation_blocked.json", blocked)
        return blocked

    provider_zero_path = root / "post_teardown_global_provider_zero.json"
    provider_zero = _sealed_provider_zero(provider_zero_path)
    if provider_zero is None:
        provider_zero = dict(provider_zero_collector())
        write_json(provider_zero_path, provider_zero)
    native_path = Path(str(adapter.get("native_control_result_path") or ""))
    specs = {
        candidate: _read(
            records[
                "pi05_execution_spec"
                if candidate == "pi05_droid"
                else "groot_execution_spec"
            ]["path"],
            code="policy_canary_execution_spec_invalid",
        )
        for candidate in CANDIDATE_IDS
    }
    if native_path.is_file():
        inner = _read(native_path, code="policy_canary_provider_result_missing")
        partial = _partial_policy_canary_result(
            native_path=native_path,
            fallback=inner,
            runtime_inputs=runtime_inputs,
            specs=specs,
        )
        if partial is not None:
            inner, native_path = partial
    else:
        gap_root = root / "preprovider_evidence"
        gap_root.mkdir(parents=True, exist_ok=True)
        gap_path = gap_root / "typed_media_gap.json"
        gap_value = {
            "schema_version": "task_evaluation_policy_canary_media_gap.v1",
            "type": "before_first_observation",
            "reason": (adapter.get("blockers") or ["provider_result_missing"])[0],
            "candidate_policy_queried": False,
        }
        write_json(gap_path, gap_value)
        inner = {
            "schema_version": "native_task_arena_policy_canary_session_result.v1",
            "status": "blocked",
            "run_kind": RUN_KIND,
            "claim_ceiling": CLAIM_CEILING,
            "episodes": [
                {
                    "candidate_id": candidate,
                    "cell_id": cell["cell_id"],
                    "seed": cell["seed"],
                    "status": "blocked",
                    "candidate_policy_queried": False,
                    "actions_reached_robot": False,
                    "arm_moved": False,
                    "policy_outcome_interpretable": False,
                    "typed_harness_failure": "before_first_observation",
                    "checkpoint_digest": specs[candidate]["checkpoint_digest"],
                    "runtime_identity_digest": specs[candidate]["runtime_identity_digest"],
                    "reset_state_digest": canonical_digest(
                        {
                            "resolved_scenario": cell["resolved_scenario"],
                            "seed": cell["seed"],
                            "execution_performed": False,
                        }
                    ),
                    "visual_evidence": {
                        "media_gap": {
                            "type": "before_first_observation",
                            "reason": gap_value["reason"],
                        }
                    },
                    "evidence_artifacts": {},
                }
                for candidate in CANDIDATE_IDS
                for cell in runtime_inputs["cells"]
            ],
            "artifact_inventory": [
                {
                    "role": "typed_media_gap",
                    "relative_path": gap_path.name,
                    "media_type": "application/json",
                    "size_bytes": gap_path.stat().st_size,
                    "sha256": _sha256(gap_path),
                }
            ],
            "blockers": list(adapter.get("blockers") or ["provider_result_missing"]),
            "result_digest": "",
        }
        inner["result_digest"] = canonical_digest(inner, digest_field="result_digest")
        native_path = gap_root / "policy_canary_provider_gap_result.json"
        write_json(native_path, inner)
    joined = _join_session_closeout(inner=inner, adapter=adapter, provider_zero=provider_zero)
    joined["run_id"] = activation["run_id"]
    joined["configuration_digest"] = runtime_inputs["configuration_digest"]
    joined["scene_revision_digest"] = setup["scene_revision_digest"]
    joined["provider"] = "vast"
    joined["provider_instance_ids"] = list(adapter.get("vast_instance_ids") or [])
    selected_container = str(adapter.get("selected_container_image") or "")
    container_match = re.search(r"sha256:[0-9a-f]{64}", selected_container)
    if container_match:
        joined["runtime_container_digest"] = container_match.group(0)
    started_at = str(adapter.get("generated_at") or "")
    teardown_for_timing = Path(str(adapter.get("teardown_manifest_path") or ""))
    completed_at = ""
    if teardown_for_timing.is_file():
        completed_at = str(
            _read(teardown_for_timing, code="policy_canary_teardown_manifest_invalid").get(
                "generated_at"
            )
            or ""
        )
    if started_at and completed_at:
        joined["started_at_iso"] = started_at
        joined["completed_at_iso"] = completed_at
        joined["duration_seconds"] = max(
            0.0,
            (
                datetime.fromisoformat(completed_at)
                - datetime.fromisoformat(started_at)
            ).total_seconds(),
        )
    joined["result_digest"] = canonical_digest(joined, digest_field="result_digest")
    joined_path = root / "policy_canary_terminal_result.json"
    write_json(joined_path, joined)
    completed_by_candidate = {
        candidate: sum(
            row.get("candidate_id") == candidate and row.get("status") == "completed"
            for row in joined.get("episodes") or []
        )
        for candidate in CANDIDATE_IDS
    }
    for candidate in CANDIDATE_IDS:
        _event_and_sync(
            root,
            stage=f"policy_{candidate}_running",
            status="completed",
            run_id=str(activation["run_id"]),
            request_digest=str(setup["request_digest"]),
            runner=progress_sync_runner,
            completed_episode_count=completed_by_candidate[candidate],
            expected_episode_count=10,
        )
    if provider_zero.get("provider_zero_verified") is not True:
        progress_sync = _sync_pending_progress(
            runner=progress_sync_runner,
            run_id=activation["run_id"],
            request_digest=setup["request_digest"],
            phase="awaiting_authenticated_vast_provider_zero",
            blocker="policy_canary_global_provider_zero_unproven",
        )
        pending = {
            "schema_version": SCHEMA_VERSION,
            "status": "awaiting_authenticated_vast_provider_zero",
            "run_id": activation["run_id"],
            "allocator_invoked": allocator_invoked,
            "automatic_retry_performed": False,
            "blockers": ["policy_canary_global_provider_zero_unproven"],
            "website_progress_sync": progress_sync,
        }
        write_json(root / "dispatch_pending.json", pending)
        return pending
    billing_path = (
        Path(official_billing_receipt_path or root / "official_billing_reconciliation.json")
        .expanduser()
        .resolve()
    )
    if not billing_path.is_file():
        _materialize_official_billing_if_posted(
            billing_audit_root=(
                billing_audit_root
                or os.getenv("BLUEPRINT_PROVIDER_BILLING_AUDIT_ROOT")
                or "/var/lib/blueprint/pipeline-control-plane/gpu_spend_guard/billing-audit"
            ),
            adapter_result_path=adapter_path,
            adapter=adapter,
            launch_label=str(resource["resource_name"]),
            output_path=billing_path,
        )
    if not billing_path.is_file():
        progress_sync = _sync_pending_progress(
            runner=progress_sync_runner,
            run_id=activation["run_id"],
            request_digest=setup["request_digest"],
            phase="awaiting_official_billing",
            blocker="policy_canary_official_billing_receipt_missing",
        )
        pending = {
            "schema_version": SCHEMA_VERSION,
            "status": "awaiting_official_billing",
            "run_id": activation["run_id"],
            "allocator_invoked": allocator_invoked,
            "automatic_retry_performed": False,
            "blockers": ["policy_canary_official_billing_receipt_missing"],
            "website_progress_sync": progress_sync,
        }
        write_json(root / "dispatch_pending.json", pending)
        return pending
    validate_vast_official_same_goal_reconciliation(billing_path)
    billing = _read(billing_path, code="policy_canary_official_billing_invalid")
    official_total_usd = billing.get("official_total_usd")
    if isinstance(official_total_usd, (int, float)) and not isinstance(
        official_total_usd, bool
    ):
        joined["official_total_usd"] = float(official_total_usd)
        joined["result_digest"] = canonical_digest(joined, digest_field="result_digest")
        write_json(joined_path, joined)
    teardown_path = Path(str(adapter.get("teardown_manifest_path") or "")).resolve()
    closure = {
        "billing": {**_record(billing_path), "official_billing_sealed": True},
        "teardown": {**_record(teardown_path), "teardown_completed": True},
        "provider_zero": {
            **_record(provider_zero_path),
            "provider_zero_verified": provider_zero.get("provider_zero_verified") is True,
        },
    }
    _event_and_sync(
        root,
        stage="artifacts_syncing",
        status="running",
        run_id=str(activation["run_id"]),
        request_digest=str(setup["request_digest"]),
        runner=progress_sync_runner,
    )
    try:
        delivery = materialize_policy_canary_result_delivery(
            run_root=root,
            run_id=activation["run_id"],
            result_status=joined["status"],
            session_result=joined,
            evidence_root=native_path.parent,
            closure_records=closure,
        )
    except TaskEvaluationResultDeliveryError as exc:
        raise TaskEvaluationPolicyCanaryDispatchError(str(exc)) from exc
    return _finish_policy_canary_delivery(
        root=root,
        setup=setup,
        runtime_inputs=runtime_inputs,
        authority=authority,
        bundle=bundle,
        joined=joined,
        joined_path=joined_path,
        delivery=delivery,
        closure=closure,
        sync_runner=sync_runner,
        progress_sync_runner=progress_sync_runner,
        allocator_invoked=allocator_invoked,
    )


def process_policy_canary_activation_results(
    *,
    activation_results_root: str | Path,
    execution_setup_root: str | Path,
    dispatch_root: str | Path,
    implementation_commit: str,
    execute: bool,
    hotfix_overlay_path: str | Path | None = None,
    machine_avoidlist_path: str | Path | None = None,
    billing_audit_root: str | Path | None = None,
    max_messages: int = 1,
    access: AccessChecker = _default_access_checker,
) -> dict[str, Any]:
    """Consume activation results automatically; never re-run an allocator output.

    Operator inputs (signed overlay, machine avoidlist, billing audit root) are
    forwarded exactly as the queue mode forwards them; a flag accepted by the
    CLI and silently dropped in one mode would be a fail-open surface.
    """

    if (
        not isinstance(max_messages, int)
        or isinstance(max_messages, bool)
        or not 1 <= max_messages <= 8
    ):
        raise TaskEvaluationPolicyCanaryDispatchError("policy_canary_dispatch_max_messages_invalid")
    results = Path(activation_results_root).expanduser().resolve()
    setups = Path(execution_setup_root).expanduser().resolve()
    outputs = Path(dispatch_root).expanduser().resolve()
    if not results.is_dir() or not setups.is_dir():
        raise TaskEvaluationPolicyCanaryDispatchError("policy_canary_dispatch_queue_roots_invalid")
    outputs.mkdir(parents=True, exist_ok=True)
    processed: list[dict[str, Any]] = []
    for path in sorted(results.glob("*.json")):
        if len(processed) >= max_messages:
            break
        try:
            result = _read(path, code="policy_canary_activation_result_invalid")
        except TaskEvaluationPolicyCanaryDispatchError:
            continue
        if result.get("run_kind") != RUN_KIND:
            continue
        activation_id = str(result.get("activation_id") or "")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,191}", activation_id):
            processed.append(
                {"status": "blocked", "blockers": ["policy_canary_activation_id_invalid"]}
            )
            continue
        setup_path = setups / f"{activation_id}.json"
        if not setup_path.is_file():
            processed.append(
                {
                    "status": "waiting_for_scene839873_execution_setup",
                    "activation_id": activation_id,
                    "allocator_invoked": False,
                    "provider_mutation_performed": False,
                }
            )
            continue
        output = outputs / activation_id
        terminal = output / "dispatch_receipt.json"
        if terminal.is_file():
            processed.append(_read(terminal, code="policy_canary_dispatch_receipt_invalid"))
            continue
        processed.append(
            dispatch_policy_canary_activation(
                activation_result_path=path,
                execution_setup_path=setup_path,
                output_root=output,
                implementation_commit=implementation_commit,
                execute=execute,
                hotfix_overlay_path=hotfix_overlay_path,
                machine_avoidlist_path=machine_avoidlist_path,
                official_billing_receipt_path=(output / "official_billing_reconciliation.json"),
                billing_audit_root=billing_audit_root,
                access=access,
            )
        )
    return {
        "schema_version": "task_evaluation_policy_canary_dispatch_queue_run.v1",
        "status": "processed" if processed else "idle",
        "processed_count": len(processed),
        "results": processed,
    }


def process_policy_canary_dispatch_queue(
    *,
    dispatch_queue_root: str | Path,
    execution_setup_root: str | Path,
    dispatch_root: str | Path,
    implementation_commit: str,
    execute: bool,
    execution_setup_template_path: str | Path | None = None,
    billing_audit_root: str | Path | None = None,
    hotfix_overlay_path: str | Path | None = None,
    machine_avoidlist_path: str | Path | None = None,
    max_messages: int = 1,
    blocked_sync_runner: SyncRunner = sync_policy_canary_preprovider_blocked_to_webapp,
    provider_zero_collector: ProviderZeroCollector = collect_policy_canary_vast_provider_zero,
    access: AccessChecker = _default_access_checker,
    disk_reservation_root: str | Path | None = None,
) -> dict[str, Any]:
    """Consume the activation worker's sealed canary-only paid queue."""

    queue = Path(dispatch_queue_root).expanduser().resolve()
    setups = Path(execution_setup_root).expanduser().resolve()
    outputs = Path(dispatch_root).expanduser().resolve()
    for name in ("pending", "processing", "completed", "blocked"):
        if not (queue / name).is_dir():
            raise TaskEvaluationPolicyCanaryDispatchError(
                "policy_canary_dispatch_queue_root_invalid"
            )
    processed: list[dict[str, Any]] = []

    def block_before_paid_dispatch(
        *,
        envelope_path: Path,
        envelope: Mapping[str, Any],
        activation_id: str,
        blockers: Sequence[str],
    ) -> dict[str, Any]:
        blocked: dict[str, Any] = {
            "schema_version": "task_evaluation_policy_canary_preprovider_blocked.v1",
            "status": "blocked_before_paid_dispatch",
            "activation_id": activation_id,
            "run_kind": RUN_KIND,
            "claim_ceiling": CLAIM_CEILING,
            "allocator_invoked": False,
            "provider_mutation_performed": False,
            "automatic_retry_performed": False,
            "blockers": list(blockers),
            "blocked_result_digest": "",
        }
        blocked["blocked_result_digest"] = canonical_digest(
            blocked, digest_field="blocked_result_digest"
        )
        blocked_root = outputs / activation_id
        try:
            blocked_root.mkdir(parents=True, exist_ok=True)
            write_json(blocked_root / "preprovider_blocked.json", blocked)
        except OSError:
            # The run directory itself may be what the service account cannot
            # use (root pre-created it).  Retain the typed block beside the
            # queue instead of crashing before the Website is told.
            blocked_root = outputs / "unwritable-runs" / activation_id
            blocked_root.mkdir(parents=True, exist_ok=True)
            write_json(blocked_root / "preprovider_blocked.json", blocked)
        sync = dict(
            blocked_sync_runner(
                activation_id=activation_id,
                capture_session_id=envelope["capture_session_id"],
                intake_id=envelope["intake_id"],
                request_digest=envelope["request_digest"],
                blockers=list(blockers),
            )
        )
        blocked["terminal_sync"] = sync
        if sync.get("status") == "succeeded":
            os.replace(envelope_path, queue / "blocked" / envelope_path.name)
        else:
            blocked["status"] = "blocked_awaiting_website_notification"
        processed.append(blocked)
        return blocked

    for envelope_path in sorted((queue / "pending").glob("*.json"))[:max_messages]:
        try:
            envelope = _read(
                envelope_path,
                code="policy_canary_dispatch_envelope_invalid",
            )
            activation_record = envelope.get("activation_result")
            if (
                envelope.get("schema_version")
                != "task_evaluation_policy_canary_dispatch_envelope.v1"
                or envelope.get("run_kind") != RUN_KIND
                or envelope.get("claim_ceiling") != CLAIM_CEILING
                or envelope.get("maximum_provider_allocations") != 1
                or envelope.get("retry_cap") != 0
                or envelope.get("automatic_retry_authorized") is not False
                or envelope.get("provider_mutation_performed") is not False
                or envelope.get("paid_execution_requested") is not False
                or envelope.get("envelope_digest")
                != canonical_digest(envelope, digest_field="envelope_digest")
            ):
                raise TaskEvaluationPolicyCanaryDispatchError(
                    "policy_canary_dispatch_envelope_invalid"
                )
            activation_path = _record_path(
                activation_record,
                code="policy_canary_dispatch_activation_record_invalid",
            )
        except (OSError, ValueError, TypeError, KeyError) as exc:
            invalid = {
                "schema_version": "task_evaluation_policy_canary_invalid_envelope.v1",
                "status": "blocked_invalid_envelope",
                "envelope_filename": envelope_path.name,
                "envelope_size_bytes": envelope_path.stat().st_size,
                "envelope_sha256": _sha256(envelope_path),
                "allocator_invoked": False,
                "provider_mutation_performed": False,
                "automatic_retry_performed": False,
                "blockers": [str(exc) or type(exc).__name__],
                "receipt_digest": "",
            }
            invalid["receipt_digest"] = canonical_digest(
                invalid,
                digest_field="receipt_digest",
            )
            invalid_root = outputs / "invalid-envelopes" / envelope_path.stem
            invalid_root.mkdir(parents=True, exist_ok=True)
            write_json(invalid_root / "invalid_envelope_receipt.json", invalid)
            os.replace(envelope_path, queue / "blocked" / envelope_path.name)
            processed.append(invalid)
            continue
        activation_id = str(envelope["activation_id"])
        setup_candidates = (
            setups / f"{activation_id}.json",
            setups / activation_id / "task_evaluation_policy_canary_execution_setup.v1.json",
        )
        setup_path = next((path for path in setup_candidates if path.is_file()), None)
        if setup_path is None and execution_setup_template_path is not None:
            setup_directory = setups / activation_id
            try:
                materialize_scene839873_policy_canary_setup_from_template(
                    template_path=execution_setup_template_path,
                    activation_envelope=envelope,
                    output_dir=setup_directory,
                )
            except PolicyCanarySetupError as exc:
                block_before_paid_dispatch(
                    envelope_path=envelope_path,
                    envelope=envelope,
                    activation_id=activation_id,
                    blockers=exc.blockers,
                )
                continue
            setup_path = setup_directory / "task_evaluation_policy_canary_execution_setup.v1.json"
        if setup_path is None:
            waiting = {
                "schema_version": "task_evaluation_policy_canary_preprovider_wait.v1",
                "status": "waiting_for_scene839873_execution_setup",
                "activation_id": activation_id,
                "allocator_invoked": False,
                "provider_mutation_performed": False,
                "automatic_retry_performed": False,
                "waiting_digest": "",
            }
            waiting["waiting_digest"] = canonical_digest(waiting, digest_field="waiting_digest")
            wait_root = outputs / activation_id
            wait_root.mkdir(parents=True, exist_ok=True)
            _write_exclusive(wait_root / "preprovider_waiting.json", waiting)
            processed.append(waiting)
            continue
        output = outputs / activation_id
        try:
            reservation = (
                reserve_control_plane_disk(
                    "policy_canary_dispatch",
                    target_root=outputs,
                    reservation_root=disk_reservation_root,
                )
                if disk_reservation_root is not None
                else contextlib.nullcontext()
            )
            with reservation:
                result = dispatch_policy_canary_activation(
                    activation_result_path=activation_path,
                    execution_setup_path=setup_path,
                    output_root=output,
                    implementation_commit=implementation_commit,
                    execute=execute,
                    hotfix_overlay_path=hotfix_overlay_path,
                    machine_avoidlist_path=machine_avoidlist_path,
                    official_billing_receipt_path=(output / "official_billing_reconciliation.json"),
                    billing_audit_root=billing_audit_root,
                    access=access,
                )
        except (
            TaskEvaluationPolicyCanaryDispatchError,
            ControlPlaneDiskBudgetError,
        ) as exc:
            invocation_started = output / "allocator_invocation_started.json"
            if invocation_started.is_file():
                try:
                    provider_zero = dict(provider_zero_collector())
                except Exception as zero_exc:  # pragma: no cover - defensive boundary
                    provider_zero = {
                        "status": "provider_zero_unproven",
                        "provider_zero_verified": False,
                        "blockers": [type(zero_exc).__name__],
                    }
                after_allocator = {
                    "schema_version": ("task_evaluation_policy_canary_post_allocator_blocked.v1"),
                    "status": (
                        "blocked_after_allocator_invocation_provider_zero"
                        if provider_zero.get("provider_zero_verified") is True
                        else "awaiting_post_allocator_provider_zero"
                    ),
                    "activation_id": activation_id,
                    "run_kind": RUN_KIND,
                    "claim_ceiling": CLAIM_CEILING,
                    "allocator_invoked": True,
                    "provider_mutation_status": ("unknown_after_allocator_invocation"),
                    "automatic_retry_performed": False,
                    "blockers": [str(exc)],
                    "allocator_invocation": _record(invocation_started),
                    "provider_zero": provider_zero,
                    "blocked_result_digest": "",
                }
                after_allocator["blocked_result_digest"] = canonical_digest(
                    after_allocator,
                    digest_field="blocked_result_digest",
                )
                write_json(output / "post_allocator_blocked.json", after_allocator)
                if provider_zero.get("provider_zero_verified") is True:
                    os.replace(
                        envelope_path,
                        queue / "blocked" / envelope_path.name,
                    )
                processed.append(after_allocator)
                continue
            block_before_paid_dispatch(
                envelope_path=envelope_path,
                envelope=envelope,
                activation_id=activation_id,
                blockers=[str(exc)],
            )
            continue
        processed.append(result)
        if (output / "dispatch_receipt.json").is_file():
            os.replace(envelope_path, queue / "completed" / envelope_path.name)
        elif (
            result.get("status") == "blocked_without_provider_allocation"
            and result.get("terminal_sync", {}).get("status") == "succeeded"
        ):
            os.replace(envelope_path, queue / "blocked" / envelope_path.name)
    return {
        "schema_version": "task_evaluation_policy_canary_dispatch_queue_run.v1",
        "status": "processed" if processed else "idle",
        "processed_count": len(processed),
        "results": processed,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activation-result")
    parser.add_argument("--execution-setup")
    parser.add_argument("--output-root")
    parser.add_argument("--activation-results-root")
    parser.add_argument("--dispatch-queue-root")
    parser.add_argument("--execution-setup-root")
    parser.add_argument("--execution-setup-template")
    parser.add_argument("--dispatch-root")
    parser.add_argument("--implementation-commit", required=True)
    parser.add_argument("--official-billing-receipt")
    parser.add_argument("--billing-audit-root")
    parser.add_argument("--hotfix-overlay")
    parser.add_argument("--machine-avoidlist")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    try:
        legacy_queue_mode = all(
            (args.activation_results_root, args.execution_setup_root, args.dispatch_root)
        )
        queue_mode = all((args.dispatch_queue_root, args.execution_setup_root, args.dispatch_root))
        direct_mode = all((args.activation_result, args.execution_setup, args.output_root))
        if sum((legacy_queue_mode, queue_mode, direct_mode)) != 1:
            raise TaskEvaluationPolicyCanaryDispatchError("policy_canary_dispatch_cli_mode_invalid")
        result = (
            process_policy_canary_dispatch_queue(
                dispatch_queue_root=args.dispatch_queue_root,
                execution_setup_root=args.execution_setup_root,
                dispatch_root=args.dispatch_root,
                implementation_commit=args.implementation_commit,
                execute=args.execute,
                execution_setup_template_path=args.execution_setup_template,
                billing_audit_root=args.billing_audit_root,
                hotfix_overlay_path=args.hotfix_overlay,
                machine_avoidlist_path=args.machine_avoidlist,
                disk_reservation_root=os.getenv(
                    "BLUEPRINT_CONTROL_PLANE_DISK_RESERVATION_ROOT"
                ),
            )
            if queue_mode
            else process_policy_canary_activation_results(
                activation_results_root=args.activation_results_root,
                execution_setup_root=args.execution_setup_root,
                dispatch_root=args.dispatch_root,
                implementation_commit=args.implementation_commit,
                execute=args.execute,
                hotfix_overlay_path=args.hotfix_overlay,
                machine_avoidlist_path=args.machine_avoidlist,
                billing_audit_root=args.billing_audit_root,
            )
            if legacy_queue_mode
            else dispatch_policy_canary_activation(
                activation_result_path=args.activation_result,
                execution_setup_path=args.execution_setup,
                output_root=args.output_root,
                implementation_commit=args.implementation_commit,
                execute=args.execute,
                official_billing_receipt_path=args.official_billing_receipt,
                billing_audit_root=args.billing_audit_root,
                hotfix_overlay_path=args.hotfix_overlay,
                machine_avoidlist_path=args.machine_avoidlist,
            )
        )
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(json.dumps({"status": "blocked", "blockers": [str(exc)]}, sort_keys=True))
        return 2
    print(json.dumps(result, sort_keys=True))
    return (
        0
        if result["status"]
        in {
            "prepared_no_execution",
            "completed_unqualified",
            "processed",
            "idle",
        }
        else 2
    )


__all__ = [
    "SCHEMA_VERSION",
    "SETUP_SCHEMA_VERSION",
    "TaskEvaluationPolicyCanaryDispatchError",
    "dispatch_policy_canary_activation",
    "collect_policy_canary_vast_provider_zero",
    "main",
    "process_policy_canary_activation_results",
    "process_policy_canary_dispatch_queue",
    "validate_policy_canary_execution_setup",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
