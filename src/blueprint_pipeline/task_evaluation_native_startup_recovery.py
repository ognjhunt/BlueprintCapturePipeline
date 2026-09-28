"""ADP-009D/day-21: retry only terminal pre-execution native allocations.

Reuse the compiled episode and original team request. Every successor gets a
fresh, ledger-debited single-use authority through ordinary activation/dispatch.
This module does not launch providers or repair runtime/placement failures.
"""
from __future__ import annotations

import copy
from datetime import datetime
from pathlib import Path
import time

from .decision_evidence_contracts import canonical_digest
from .task_evaluation_scene_configuration_submission_inputs import read, require, sha
from .task_evaluation_public_scene_attempt_factory import record
from .task_evaluation_scene_recovery import provider_null_evidence
from . import task_evaluation_scene_intake as intake


# No-start markers alone do not distinguish an unavailable host from a broken
# runtime. Only these explicit infrastructure outcomes authorize replacement.
STARTUP_LOSS = frozenset({"vast_heartbeat_instance_exited", "vast_heartbeat_container_missing"})
STARTUP_CONSEQUENCES = STARTUP_LOSS | {"vast_heartbeat_output_missing_success_marker"}


def configuration(controls: dict, progression: dict) -> dict:
    """Use the existing scene worker's ownership scope, not invented defaults."""
    require(controls["scene_root"] == progression["intent_root"], "native_retry_scene_scope_changed")
    return {"scene_root": controls["scene_root"], **{key: progression[key] for key in (
        "provider_guard_path", "ownership_roots", "child_execution_root", "launch_execution_root",
        "child_queue_root", "launch_queue_root")}}


def _owned_path(root: Path, value: str) -> Path:
    path = Path(value)
    require(path.is_absolute() and path.is_relative_to(root), "native_retry_artifact_outside_run")
    read(path)  # The shared reader rejects symlinks, including parent components.
    return path


def inspect_failure(*, run_root: Path, launch: dict, activation: dict,
                    scene_root: Path) -> tuple[dict, dict] | None:
    """A missing/running/completed launch never authorizes replacement."""
    receipt_path = run_root / "launch_receipt.json"
    if not receipt_path.exists():
        return None
    receipt = read(receipt_path, digest_field="receipt_digest")
    if receipt.get("status") == "completed":
        return None
    require(receipt.get("status") == "blocked", "native_retry_launch_not_terminal")
    profile = read(run_root / "launch_profile.json", digest_field="profile_digest")
    require(receipt.get("launch_id") == launch["launch_id"] == run_root.name
            and receipt.get("launch_profile_digest") == launch["profile_digest"] == profile["profile_digest"]
            and receipt.get("source_commit") == activation["expected_production_commit"] == profile["source_commit"],
            "native_retry_launch_binding_changed")
    ref = receipt["terminal_evidence"]["result"]
    result_path = _owned_path(run_root, ref["path"])
    require(sha(result_path) == ref["digest"], "native_retry_terminal_changed")
    result = read(result_path, digest_field="result_digest")
    close = result.get("independent_watchdog_close") or {}
    require(result.get("status") == "blocked" and result.get("continuing_spend_from_this_run") is False
            and result.get("all_staged_objects_absent") is True
            and result.get("native_control_result_path") is None
            and close.get("status") == "provider_terminal" and close.get("provider_absence_confirmed") is True,
            "native_retry_teardown_or_execution_not_admitted")
    adapter_path = _owned_path(run_root, result["adapter_result_path"])
    manifest_ref = receipt["terminal_evidence"]["artifacts"]["artifact_manifest_path"]
    manifest_path = _owned_path(run_root, result["artifact_manifest_path"])
    require(str(manifest_path) == manifest_ref["path"] and sha(manifest_path) == manifest_ref["digest"],
            "native_retry_manifest_changed")
    manifest = read(manifest_path, digest_field="manifest_digest")
    matches = [row for row in manifest["files"] if "allocator_adapter_result" in row.get("roles", [])]
    require(len(matches) == 1 and manifest_path.parent / matches[0]["relative_path"] == adapter_path
            and sha(adapter_path) == matches[0]["sha256"]
            and adapter_path.stat().st_size == matches[0]["size_bytes"], "native_retry_adapter_changed")
    adapter = read(adapter_path)
    require(provider_null_evidence(adapter) and adapter.get("continuing_spend_from_this_run") is False,
            "native_retry_not_pre_execution_failure")
    causes = set(adapter["provider_attempt_classification"]["blockers"]) | set(adapter.get("blockers", []))
    require(bool(causes & STARTUP_LOSS) and causes <= STARTUP_CONSEQUENCES,
            "native_retry_runtime_repair_required")
    old_owner = activation["activation_request"]["authorization"]["scene_owner_attempt"]
    binding = old_owner["scene_attempt_binding"]
    require(profile.get("scene_attempt_binding") == binding, "native_retry_owner_changed")
    prior = intake._read(scene_root / binding["intent_id"] / "attempts" / (binding["attempt_id"] + ".json"), "attempt_digest")
    require(all(prior.get(k) == v for k, v in binding.items() if k != "schema_version"), "native_retry_attempt_changed")
    failure = {"schema_version": "task_evaluation_scene_attempt_failure.v1", "status": "failed",
               "attempt_digest": prior["attempt_digest"], "failure_kind": "provider_null",
               "observed_at_epoch": datetime.fromisoformat(adapter["generated_at"].replace("Z", "+00:00")).timestamp(),
               "producer_result": record(adapter_path), "launch_receipt": record(receipt_path)}
    failure["failure_digest"] = canonical_digest(failure, digest_field="failure_digest")
    return prior, failure


def retain_failure(*, run_root: Path, launch: dict, activation: dict,
                   scene_root: Path, output_root: Path) -> tuple[dict, Path] | None:
    verified = inspect_failure(run_root=run_root, launch=launch, activation=activation, scene_root=scene_root)
    if verified is None:
        return None
    prior, failure = verified
    output_root.mkdir(parents=True, exist_ok=True, mode=0o750)
    path = output_root / "failure.json"
    if path.exists():
        require(read(path, digest_field="failure_digest") == failure, "native_retry_failure_changed")
    else:
        intake.write_exclusive(path, failure)
    return prior, path


def advance(*, config: dict, plan: dict, state: Path, launch_root: Path,
            launch: dict, activation: dict, phase: dict, base: dict, preparation: dict,
            activation_queue_root: Path, publisher, submitter_factory, now: float | None = None):
    """Advance one successor transition; return (effective launch, pending status).

    Immutable files make replay after interruption idempotent. The scene ledger
    serializes reservations and forbids two successors of the same attempt.
    """
    from . import task_evaluation_configured_controls_progression_worker as worker
    from .task_evaluation_configured_controls_progression import (
        stage_configured_controls_activation, submit_authorized_progression_launch)
    from .task_evaluation_scene_progression_recovery import reconcile_ownership
    from .task_evaluation_scene_execution_authority import bind_scene_attempt
    from .task_evaluation_scene_owner_attempt_profiles import make_owner_attempt_record

    moment = time.time() if now is None else now
    current, current_activation = launch, activation
    # The ledger's owner limit is authoritative; this additionally bounds scanning.
    for _ in range(9):
        retry_root = state / "startup-recovery" / current["launch_id"]
        retry_launch_path = retry_root / "launch.json"
        retry_activation_path = retry_root / "activation.json"
        if retry_launch_path.exists():
            current = read(retry_launch_path, digest_field="progression_digest")
            current_activation = read(retry_activation_path, digest_field="progression_digest")
            continue
        failure = retain_failure(run_root=launch_root / current["launch_id"], launch=current,
            activation=current_activation, scene_root=Path(config["scene_root"]), output_root=retry_root)
        if failure is None:
            return current, None
        prior, failure_path = failure
        owner = intake._read(Path(config["scene_root"]) / prior["intent_id"] / "intent.json", "intent_digest")
        require(owner["request"]["execution"]["max_retries"] > 0, "native_retry_owner_limit_exhausted")
        token = read(failure_path)["failure_digest"][7:19]
        activation_id = current_activation["activation_request"]["activation_id"] + "-boot-" + token
        require(len(activation_id) <= 192, "native_retry_identity_exhausted")
        reservation_path = retry_root / "reservation.json"
        attempt_id = "native-startup-" + read(failure_path)["failure_digest"][7:47]
        ledger_path = Path(config["scene_root"]) / prior["intent_id"] / "attempts" / (attempt_id + ".json")
        if ledger_path.exists() and not reservation_path.exists():
            retained = intake._read(ledger_path, "attempt_digest")
            require(retained.get("recovery", {}).get("prior_attempt_id") == prior["attempt_id"]
                    and retained["recovery"].get("failure_digest") == read(failure_path)["failure_digest"],
                    "native_retry_reservation_changed")
            worker._write_immutable(reservation_path, retained)
        if reservation_path.exists():
            reservation = read(reservation_path, digest_field="attempt_digest")
            retained = intake._read(Path(config["scene_root"]) / prior["intent_id"] / "attempts" /
                                   (reservation["attempt_id"] + ".json"), "attempt_digest")
            require(retained == reservation and reservation.get("recovery", {}).get("prior_attempt_id") == prior["attempt_id"],
                    "native_retry_reservation_changed")
        else:
            evidence = reconcile_ownership(attempt=prior, failure_path=failure_path, config=config,
                                           output_root=retry_root / "ownership", now=moment)
            reservation = intake.reserve_scene_attempt(queue_root=config["scene_root"], intent_id=prior["intent_id"],
                attempt_id=attempt_id,
                source_commit=prior["source_commit"], runtime_digest=prior["runtime_digest"],
                input_digest=prior["input_digest"], provider=prior["provider"],
                maximum_spend_usd=prior["maximum_spend_usd"], recovery_from_attempt_id=prior["attempt_id"],
                recovery_evidence=evidence, now=moment)
            worker._write_immutable(reservation_path, reservation)
        if not retry_activation_path.exists():
            authorization = copy.deepcopy(current_activation["activation_request"]["authorization"])
            # Native profiles are immutable and keyed by revision, not activation id.
            authorization["profile_revision"] = attempt_id
            old_owner = authorization["scene_owner_attempt"]
            authorization["scene_owner_attempt"] = make_owner_attempt_record(owner_fields=bind_scene_attempt(reservation),
                **{k: old_owner[k] for k in ("phase", "team_namespace", "scene_id", "task_id", "runtime_source_bundle_digest")})
            lineage = current_activation["activation_request"]["lineage"]
            lane = current_activation["lane"]
            require(lane == "native_task_arena_construction", "native_retry_lane_not_supported")
            window = worker._materialize_phase_release_window(state=base, preparation=preparation, phase=phase,
                lineage=lineage, authorization=authorization, lane=lane, root=retry_root,
                publisher=publisher(), activation_id=activation_id)
            staged = stage_configured_controls_activation(progression=base, preparation_result=preparation,
                release_window=window, lineage=lineage, authorization=authorization, lane=lane,
                queue_root=activation_queue_root, submitted_by=plan["submitted_by"], activation_id=activation_id)
            worker._write_immutable(retry_activation_path, staged)
            return current, "startup_replacement_activation_queued"
        staged = read(retry_activation_path, digest_field="progression_digest")
        ready = worker._activation_authority(activation_queue_root=activation_queue_root,
            profile_dir=Path(plan["profile_dir"]), activation_id=staged["activation_request"]["activation_id"])
        if ready is None:
            return current, "awaiting_startup_replacement_activation"
        activated, profile = ready
        queued = submit_authorized_progression_launch(activation_progression=staged, activation_result=activated,
            profile=profile, launch_authority=read(phase["launch_authority_path"]), submitter=submitter_factory())
        worker._write_immutable(retry_launch_path, queued)
        return queued, "startup_replacement_launch_queued"
    raise ValueError("native_startup_recovery_scan_limit")


def effective_launch(state: Path, launch: dict) -> dict:
    """Resolve durable startup successors for downstream evidence consumers."""
    for _ in range(9):
        path = state / "startup-recovery" / launch["launch_id"] / "launch.json"
        if not path.exists():
            return launch
        successor = read(path, digest_field="progression_digest")
        require(successor.get("status") == launch["status"]
                and successor.get("configured_scene_revision_digest") == launch["configured_scene_revision_digest"],
                "native_retry_successor_changed")
        launch = successor
    raise ValueError("native_startup_recovery_scan_limit")
