"""Consume one activated internal policy canary through the canonical allocator.

This module is intentionally canary-only.  It cannot dispatch qualified
evaluations, cannot promote a scene, and cannot retry a paid allocation.  Its
closeout is resumable: once allocator output exists, later invocations may
collect official billing, fresh provider-zero, Website sync, and notification
readback without ever invoking the allocator again.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Callable, Mapping, Sequence

from .common import write_json
from .decision_evidence_contracts import canonical_digest
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
from .task_evaluation_policy_canary_result import validate_policy_canary_result
from .task_evaluation_policy_canary_scene_setup import (
    PolicyCanarySetupError,
    materialize_scene839873_policy_canary_setup_from_template,
)
from .task_evaluation_result_delivery import (
    TaskEvaluationResultDeliveryError,
    materialize_policy_canary_result_delivery,
)
from .task_evaluation_run_webapp_sync import (
    sync_policy_canary_preprovider_blocked_to_webapp,
    sync_task_evaluation_policy_canary_to_webapp,
)
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
        raise TaskEvaluationPolicyCanaryDispatchError(
            "policy_canary_dispatch_record_invalid"
        )
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
        or setup.get("setup_digest")
        != canonical_digest(setup, digest_field="setup_digest")
    ):
        raise TaskEvaluationPolicyCanaryDispatchError(
            "policy_canary_scene839873_setup_invalid"
        )
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


def _event(root: Path, *, stage: str, status: str, **details: Any) -> None:
    path = root / "status_events.jsonl"
    sequence = 1
    previous_digest = None
    if path.is_file():
        rows = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
        sequence = len(rows) + 1
        if rows:
            previous_digest = json.loads(rows[-1]).get("event_digest")
    event = {
        "schema_version": "task_evaluation_policy_canary_status_event.v1",
        "sequence": sequence,
        "stage": stage,
        "status": status,
        "previous_event_digest": previous_digest,
        **details,
        "event_digest": "",
    }
    event["event_digest"] = canonical_digest(event, digest_field="event_digest")
    with path.open("ab") as stream:
        stream.write((json.dumps(event, sort_keys=True) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())


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


def _join_session_closeout(
    *, inner: Mapping[str, Any], adapter: Mapping[str, Any], provider_zero: Mapping[str, Any]
) -> dict[str, Any]:
    value = json.loads(json.dumps(dict(inner), allow_nan=False))
    episodes = value.get("episodes")
    instance_ids = adapter.get("vast_instance_ids") or []
    closeout = adapter.get("provider_closeout")
    teardown_complete = (
        isinstance(closeout, Mapping)
        and closeout.get("provider_zero_confirmed") is True
        and closeout.get("warm_session_retained") is False
        and closeout.get("all_staged_objects_absent") is True
        and adapter.get("continuing_spend_from_this_run") is False
    )
    global_zero = (
        provider_zero.get("schema_version")
        == "task_evaluation_policy_canary_vast_provider_zero.v1"
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
    if len(instance_ids) != 1:
        blockers.append("policy_canary_provider_allocation_count_invalid")
    if not teardown_complete:
        blockers.append("policy_canary_teardown_incomplete")
    if not global_zero:
        blockers.append("policy_canary_global_provider_zero_unproven")
    value["blockers"] = sorted(set(blockers))
    value["result_digest"] = canonical_digest(value, digest_field="result_digest")
    return value


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
    instance_ids = adapter.get("vast_instance_ids")
    if (
        not isinstance(instance_ids, list)
        or len(instance_ids) != 1
        or isinstance(instance_ids[0], bool)
        or not isinstance(instance_ids[0], int)
    ):
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
                expected_instances=[
                    (int(instance_ids[0]), launch_label, adapter_result_path)
                ],
                output_path=output_path,
            )
        except (OSError, VastOfficialBillingExtractionError):
            continue
        return True
    return False


def _projection(
    *, setup: Mapping[str, Any], result: Mapping[str, Any], delivery: Mapping[str, Any]
) -> dict[str, Any]:
    episodes = list(result.get("episodes") or [])
    def compact_artifact(record: Mapping[str, Any]) -> dict[str, Any]:
        return {
            key: record[key]
            for key in ("artifact_id", "digest", "size_bytes")
        }

    report = {
        "machine_readable_report": compact_artifact(
            delivery["report"]["machine_readable_report"]
        ),
        "evidence_manifest": compact_artifact(
            delivery["report"]["evidence_manifest"]
        ),
    }
    public_artifacts = delivery.get("artifacts") or []

    def bound_artifact(record: Any) -> dict[str, Any] | None:
        if not isinstance(record, Mapping):
            return None
        matches = [
            artifact
            for artifact in public_artifacts
            if artifact.get("role") == record.get("role")
            and artifact.get("digest") == record.get("sha256")
            and artifact.get("size_bytes") == record.get("size_bytes")
        ]
        if len(matches) != 1:
            return None
        return {
            key: matches[0][key]
            for key in ("artifact_id", "digest", "size_bytes")
        }
    projected_episodes: list[dict[str, Any]] = []
    for row in episodes:
        candidate = str(row.get("candidate_id") or "")
        cell_id = str(row.get("cell_id") or "")
        episode_id = f"{result.get('run_id') or setup['scene_id']}--{cell_id}--{candidate}"
        if len(episode_id) > 192:
            episode_id = episode_id[:150] + "-" + hashlib.sha256(
                episode_id.encode()
            ).hexdigest()[:32]
        source_artifacts = row.get("evidence_artifacts")
        source_artifacts = (
            dict(source_artifacts) if isinstance(source_artifacts, Mapping) else {}
        )
        evidence_roles = {
            "reset_state": "reset_state",
            "frame_manifest": "frame_manifest",
            "review_video": "review_video",
            "policy_query_receipt": "policy_query_receipt",
            "action_sequence": "action_sequence",
            "action_delivery_readback": "action_delivery_readback",
            "state_trace": "state_trace",
            "contact_force_trace": "contact_force_trace",
            "task_object_trajectory": "task_object_trajectory",
            "score_receipt": "score_receipt",
        }
        bound = {
            target: bound_artifact(source_artifacts.get(source))
            for target, source in evidence_roles.items()
        }
        evidence_gaps = sorted(
            target for target, artifact in bound.items() if artifact is None
        )
        if row.get("status") == "completed" and evidence_gaps:
            raise TaskEvaluationPolicyCanaryDispatchError(
                "policy_canary_completed_episode_evidence_missing:"
                + episode_id
                + ":"
                + ",".join(evidence_gaps)
            )
        checkpoint_digest = row.get("checkpoint_digest")
        runtime_identity_digest = row.get("runtime_identity_digest")
        reset_state_digest = (
            source_artifacts.get("reset_state", {}).get("sha256")
            if isinstance(source_artifacts.get("reset_state"), Mapping)
            else row.get("reset_state_digest")
        )
        if not all(
            _is_digest(value)
            for value in (
                checkpoint_digest,
                runtime_identity_digest,
                reset_state_digest,
            )
        ):
            raise TaskEvaluationPolicyCanaryDispatchError(
                "policy_canary_episode_identity_evidence_missing:" + episode_id
            )
        evidence = {
            "checkpoint_digest": checkpoint_digest,
            "runtime_identity_digest": runtime_identity_digest,
            "reset_state_digest": reset_state_digest,
            **bound,
            "evidence_gaps": evidence_gaps,
        }
        gap = ((row.get("visual_evidence") or {}).get("media_gap") or {}).get(
            "reason"
        )
        if gap:
            evidence["typed_media_gap"] = str(gap)
        projected_episodes.append(
            {
                "episode_id": episode_id,
                "candidate_id": candidate,
                "cell_id": cell_id,
                "seed": row.get("seed"),
                "terminal_state": (
                    "completed" if row.get("status") == "completed" else "blocked"
                ),
                "candidate_policy_queried": row.get("candidate_policy_queried")
                is True,
                "actions_reached_robot": row.get("actions_reached_robot") is True,
                "arm_moved": row.get("arm_moved") is True,
                "policy_outcome_interpretable": row.get(
                    "policy_outcome_interpretable"
                )
                is True,
                "failure_taxonomy": row.get("typed_harness_failure"),
                "evidence": evidence,
            }
        )
    candidate_results = []
    delivered_candidate_results = {
        row["candidate_id"]: row for row in delivery.get("candidate_results") or []
    }
    for candidate in CANDIDATE_IDS:
        rows = [row for row in projected_episodes if row["candidate_id"] == candidate]
        failures: dict[str, int] = {}
        for row in rows:
            if row["terminal_state"] != "completed":
                name = str(row.get("failure_taxonomy") or "unclassified")
                failures[name] = failures.get(name, 0) + 1
        delivered_metrics = dict(delivered_candidate_results.get(candidate) or {})
        candidate_results.append(
            {
                "candidate_id": candidate,
                "episodes_completed": sum(
                    row["terminal_state"] == "completed" for row in rows
                ),
                "interpretable_episode_count": sum(
                    row["policy_outcome_interpretable"] for row in rows
                ),
                "actions_delivered_episode_count": sum(
                    row["actions_reached_robot"] for row in rows
                ),
                "metrics": {
                    key: value
                    for key, value in delivered_metrics.items()
                    if key != "candidate_id"
                },
                "failure_counts": failures,
            }
        )
    cell_sets = [
        {row["cell_id"] for row in projected_episodes if row["candidate_id"] == candidate}
        for candidate in CANDIDATE_IDS
    ]
    result_status = str(delivery["result_status"])
    value: dict[str, Any] = {
        "schema_version": "task_evaluation_policy_canary_result_projection.v1",
        "run_id": delivery["run_id"],
        "request_digest": setup["request_digest"],
        "configuration_digest": result["configuration_digest"],
        "result_delivery_digest": delivery["delivery_digest"],
        "run_kind": RUN_KIND,
        "claim_ceiling": CLAIM_CEILING,
        "scene_controls_status": "configured_controls_pending",
        "result_status": result_status,
        "warning": "Controls pending — results are unqualified.",
        "counts": {
            "policy_count": 2,
            "episodes_per_policy": 10,
            "learned_policy_rollout_count": 20,
            "completed_learned_policy_rollout_count": sum(
                row["terminal_state"] == "completed" for row in projected_episodes
            ),
            "diagnostic_control_rollout_count": 20,
            "completed_diagnostic_control_rollout_count": 0,
        },
        "candidate_ids": list(CANDIDATE_IDS),
        "candidate_results": candidate_results,
        "episodes": projected_episodes,
        "comparison": {
            "matched_cell_count": len(cell_sets[0] & cell_sets[1]),
            "winner_declared": False,
            "official_ranking_contribution": False,
        },
        "report": {
            "result_digest": result["result_digest"],
            "permanent_result_path": f"/internal/task-evaluation-runs/{delivery['run_id']}",
            **report,
        },
        "closure": {
            "billing": compact_artifact(delivery["closure"]["billing"]),
            "teardown": compact_artifact(delivery["closure"]["teardown"]),
            "provider_zero": {
                **compact_artifact(delivery["closure"]["provider_zero"]),
                "provider_zero_verified": True,
            },
        },
        "notification_delivery": {
            "terminal_state": (
                "completed" if result_status == "completed_unqualified" else result_status
            ),
            "status": "pending",
            "attempts": 0,
            "provider": "website_terminal_handler",
            "message_id": None,
            "delivered_at": None,
            "run_result_digest": result["result_digest"],
        },
        "blockers": list(result.get("blockers") or []),
        "projection_digest": "",
    }
    value["projection_digest"] = canonical_digest(
        value, digest_field="projection_digest"
    )
    return validate_policy_canary_result(value)


def dispatch_policy_canary_activation(
    *,
    activation_result_path: str | Path,
    execution_setup_path: str | Path,
    output_root: str | Path,
    implementation_commit: str,
    execute: bool = False,
    official_billing_receipt_path: str | Path | None = None,
    billing_audit_root: str | Path | None = None,
    allocator_runner: AllocatorRunner | None = None,
    provider_zero_collector: ProviderZeroCollector = collect_policy_canary_vast_provider_zero,
    sync_runner: SyncRunner = sync_task_evaluation_policy_canary_to_webapp,
) -> dict[str, Any]:
    """Dispatch or resume exactly one Scene 839873 policy canary."""

    if not re.fullmatch(r"[0-9a-f]{40}", implementation_commit):
        raise TaskEvaluationPolicyCanaryDispatchError(
            "policy_canary_dispatch_source_commit_invalid"
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
        or activation_result.get("status")
        != "policy_campaign_queue_materialized_no_execution"
        or activation_result.get("run_kind") != RUN_KIND
        or activation_result.get("claim_ceiling") != CLAIM_CEILING
        or activation_result.get("provider_mutation_performed") is not False
        or activation_result.get("paid_execution_requested") is not False
        or activation_result.get("result_digest")
        != canonical_digest(activation_result, digest_field="result_digest")
    ):
        raise TaskEvaluationPolicyCanaryDispatchError(
            "policy_canary_activation_result_invalid"
        )
    runtime_path = Path(
        str(activation_result.get("policy_canary_runtime_inputs_path") or "")
    ).expanduser().resolve()
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
    _event(root, stage="queued", status="completed", run_id=activation["run_id"])
    authority = build_session_authority(
        activation_manifest=activation,
        activation_record=_record(activation_path),
        runtime_inputs=runtime_inputs,
        runtime_input_record=_record(runtime_path),
        resource_name=str(resource["resource_name"]),
        hard_cap_usd=float(resource["hard_cap_usd"]),
        hard_ttl_seconds=int(resource["hard_ttl_seconds"]),
    )
    authority_path = root / "policy_canary_session_authority.json"
    _write_exclusive(authority_path, authority)
    records = setup["records"]
    _event(root, stage="preparing", status="running")
    bundle_receipt_path = (
        root
        / "bundle"
        / "native_task_arena_policy_canary_session_bundle_receipt.v1.json"
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
        )
    _event(
        root,
        stage="preparing",
        status="completed",
        authority_digest=authority["authority_digest"],
        bundle_sha256=bundle["bundle_sha256"],
    )
    admission_path = root / "paid_admission.json"
    adapter_path = root / "allocator_result.json"
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
    allocator_invoked = False
    if not adapter_path.is_file():
        _event(root, stage="provider_allocating", status="running")
        if execute:
            argv.append("--execute")
        exit_code = int((allocator_runner or _default_allocator_runner)(argv))
        allocator_invoked = True
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
        receipt["receipt_digest"] = canonical_digest(
            receipt, digest_field="receipt_digest"
        )
        _write_exclusive(root / "dispatch_receipt.json", receipt)
        return receipt

    provider_zero_path = root / "post_teardown_global_provider_zero.json"
    provider_zero = dict(provider_zero_collector())
    write_json(provider_zero_path, provider_zero)
    native_path = Path(str(adapter.get("native_control_result_path") or ""))
    if native_path.is_file():
        inner = _read(native_path, code="policy_canary_provider_result_missing")
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
                    "runtime_identity_digest": specs[candidate][
                        "runtime_identity_digest"
                    ],
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
        inner["result_digest"] = canonical_digest(
            inner, digest_field="result_digest"
        )
        native_path = gap_root / "policy_canary_provider_gap_result.json"
        write_json(native_path, inner)
    joined = _join_session_closeout(
        inner=inner, adapter=adapter, provider_zero=provider_zero
    )
    joined["run_id"] = activation["run_id"]
    joined["configuration_digest"] = runtime_inputs["configuration_digest"]
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
        _event(
            root,
            stage=f"policy_{candidate}_running",
            status="completed",
            completed_episode_count=completed_by_candidate[candidate],
            expected_episode_count=10,
        )
    if provider_zero.get("provider_zero_verified") is not True:
        pending = {
            "schema_version": SCHEMA_VERSION,
            "status": "awaiting_authenticated_vast_provider_zero",
            "run_id": activation["run_id"],
            "allocator_invoked": allocator_invoked,
            "automatic_retry_performed": False,
            "blockers": ["policy_canary_global_provider_zero_unproven"],
        }
        write_json(root / "dispatch_pending.json", pending)
        return pending
    billing_path = Path(
        official_billing_receipt_path
        or root / "official_billing_reconciliation.json"
    ).expanduser().resolve()
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
        pending = {
            "schema_version": SCHEMA_VERSION,
            "status": "awaiting_official_billing",
            "run_id": activation["run_id"],
            "allocator_invoked": allocator_invoked,
            "automatic_retry_performed": False,
            "blockers": ["policy_canary_official_billing_receipt_missing"],
        }
        write_json(root / "dispatch_pending.json", pending)
        return pending
    validate_vast_official_same_goal_reconciliation(billing_path)
    teardown_path = Path(str(adapter.get("teardown_manifest_path") or "")).resolve()
    closure = {
        "billing": {**_record(billing_path), "official_billing_sealed": True},
        "teardown": {**_record(teardown_path), "teardown_completed": True},
        "provider_zero": {
            **_record(provider_zero_path),
            "provider_zero_verified": provider_zero.get("provider_zero_verified")
            is True,
        },
    }
    _event(root, stage="artifacts_syncing", status="running")
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
    projection = _projection(setup=setup, result=joined, delivery=delivery)
    _event(root, stage="report_generating", status="completed")
    sync = dict(
        sync_runner(
            capture_session_id=setup["capture_session_id"],
            intake_id=setup["intake_id"],
            run_id=activation["run_id"],
            request_digest=setup["request_digest"],
            configuration_digest=runtime_inputs["configuration_digest"],
            result_status=joined["status"],
            result_delivery=delivery,
            policy_canary_result=projection,
        )
    )
    if sync.get("status") != "succeeded" or not isinstance(
        sync.get("notification_delivery"), Mapping
    ):
        pending = {
            "schema_version": SCHEMA_VERSION,
            "status": "awaiting_website_sync_or_notification",
            "run_id": activation["run_id"],
            "allocator_invoked": allocator_invoked,
            "automatic_retry_performed": False,
            "blockers": ["policy_canary_website_notification_readback_missing"],
        }
        write_json(root / "dispatch_pending.json", pending)
        return pending
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "status": joined["status"],
        "run_id": activation["run_id"],
        "run_kind": RUN_KIND,
        "claim_ceiling": CLAIM_CEILING,
        "authority_digest": authority["authority_digest"],
        "bundle_sha256": bundle["bundle_sha256"],
        "terminal_result": _record(joined_path),
        "result_delivery_digest": delivery["delivery_digest"],
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
    _event(root, stage="billing_teardown", status="completed")
    _event(root, stage=joined["status"], status="completed")
    return receipt


def process_policy_canary_activation_results(
    *,
    activation_results_root: str | Path,
    execution_setup_root: str | Path,
    dispatch_root: str | Path,
    implementation_commit: str,
    execute: bool,
    max_messages: int = 1,
) -> dict[str, Any]:
    """Consume activation results automatically; never re-run an allocator output."""

    if not isinstance(max_messages, int) or isinstance(max_messages, bool) or not 1 <= max_messages <= 8:
        raise TaskEvaluationPolicyCanaryDispatchError(
            "policy_canary_dispatch_max_messages_invalid"
        )
    results = Path(activation_results_root).expanduser().resolve()
    setups = Path(execution_setup_root).expanduser().resolve()
    outputs = Path(dispatch_root).expanduser().resolve()
    if not results.is_dir() or not setups.is_dir():
        raise TaskEvaluationPolicyCanaryDispatchError(
            "policy_canary_dispatch_queue_roots_invalid"
        )
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
                official_billing_receipt_path=(
                    output / "official_billing_reconciliation.json"
                ),
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
    max_messages: int = 1,
    blocked_sync_runner: SyncRunner = sync_policy_canary_preprovider_blocked_to_webapp,
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
        envelope = _read(envelope_path, code="policy_canary_dispatch_envelope_invalid")
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
            activation_record, code="policy_canary_dispatch_activation_record_invalid"
        )
        activation_id = str(envelope["activation_id"])
        setup_candidates = (
            setups / f"{activation_id}.json",
            setups
            / activation_id
            / "task_evaluation_policy_canary_execution_setup.v1.json",
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
            setup_path = (
                setup_directory
                / "task_evaluation_policy_canary_execution_setup.v1.json"
            )
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
            waiting["waiting_digest"] = canonical_digest(
                waiting, digest_field="waiting_digest"
            )
            wait_root = outputs / activation_id
            wait_root.mkdir(parents=True, exist_ok=True)
            _write_exclusive(wait_root / "preprovider_waiting.json", waiting)
            processed.append(waiting)
            continue
        output = outputs / activation_id
        try:
            result = dispatch_policy_canary_activation(
                activation_result_path=activation_path,
                execution_setup_path=setup_path,
                output_root=output,
                implementation_commit=implementation_commit,
                execute=execute,
                official_billing_receipt_path=(
                    output / "official_billing_reconciliation.json"
                ),
                billing_audit_root=billing_audit_root,
            )
        except TaskEvaluationPolicyCanaryDispatchError as exc:
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
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    try:
        legacy_queue_mode = all(
            (args.activation_results_root, args.execution_setup_root, args.dispatch_root)
        )
        queue_mode = all(
            (args.dispatch_queue_root, args.execution_setup_root, args.dispatch_root)
        )
        direct_mode = all(
            (args.activation_result, args.execution_setup, args.output_root)
        )
        if sum((legacy_queue_mode, queue_mode, direct_mode)) != 1:
            raise TaskEvaluationPolicyCanaryDispatchError(
                "policy_canary_dispatch_cli_mode_invalid"
            )
        result = (
            process_policy_canary_dispatch_queue(
                dispatch_queue_root=args.dispatch_queue_root,
                execution_setup_root=args.execution_setup_root,
                dispatch_root=args.dispatch_root,
                implementation_commit=args.implementation_commit,
                execute=args.execute,
                execution_setup_template_path=args.execution_setup_template,
                billing_audit_root=args.billing_audit_root,
            )
            if queue_mode
            else process_policy_canary_activation_results(
                activation_results_root=args.activation_results_root,
                execution_setup_root=args.execution_setup_root,
                dispatch_root=args.dispatch_root,
                implementation_commit=args.implementation_commit,
                execute=args.execute,
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
            )
        )
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(json.dumps({"status": "blocked", "blockers": [str(exc)]}, sort_keys=True))
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] in {
        "prepared_no_execution",
        "completed_unqualified",
        "processed",
        "idle",
    } else 2


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
