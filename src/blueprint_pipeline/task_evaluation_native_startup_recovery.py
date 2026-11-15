"""ADP-009D/day-21: retry only terminal pre-execution native allocations.

Reuse the compiled episode and original team request. Every successor gets a
fresh, ledger-debited single-use authority through ordinary activation/dispatch.
This module does not launch providers or repair runtime/placement failures.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from .decision_evidence_contracts import canonical_digest
from .task_evaluation_scene_configuration_submission_inputs import read, require, sha
from .task_evaluation_public_scene_attempt_factory import record
from .task_evaluation_scene_recovery import provider_null_evidence
from . import task_evaluation_scene_intake as intake


# No-start markers alone do not distinguish an unavailable host from a broken
# runtime. Only these explicit infrastructure outcomes authorize replacement.
STARTUP_LOSS = frozenset({"vast_heartbeat_instance_exited", "vast_heartbeat_container_missing"})
STARTUP_CONSEQUENCES = STARTUP_LOSS | {"vast_heartbeat_output_missing_success_marker"}


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
