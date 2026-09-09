"""Retire controls holds that never became eligible before publication failed.

An original blocked launch projection is the admission boundary: controls and
placement cannot start until it becomes completed, delivered, and provider-zero
qualified. Preserve its bytes before retiring only its sealed downstream holds.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .decision_evidence_contracts import canonical_digest, cross_runtime_canonical_digest

SCHEMA = "task_evaluation_unstarted_controls_cancellation.v1"
DIRECTORY = "cancelled-unstarted-controls"


def _read(path: Path) -> dict[str, Any]:
    if not path.is_file() or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("unstarted_controls_evidence_unsafe")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("unstarted_controls_evidence_invalid")
    return value


def _file(path: Path) -> dict[str, Any]:
    _read(path)
    return {"path": str(path), "digest": "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()}


def validated_cancellation(directory: Path, attempt: Mapping[str, Any]) -> dict[str, Any] | None:
    path = directory / DIRECTORY / (str(attempt["attempt_id"]) + ".json")
    if not path.exists() and not path.is_symlink():
        return None
    receipt = _read(path)
    if receipt.get('schema_version') == 'task_evaluation_unused_native_plan_cancellation.v1':
        from .task_evaluation_completed_placement_adoption import validate_cancellation
        validate_cancellation(receipt=receipt,attempt=attempt)
        return receipt
    if receipt.get('schema_version') == 'task_evaluation_unstarted_native_after_visual_review.v1':
        from .task_evaluation_visual_review_continuation import validate_native_retirement
        validate_native_retirement(receipt=receipt, attempt=attempt)
        return receipt
    if receipt.get('schema_version') == 'task_evaluation_unmaterialized_adoption_cancellation.v1':
        from .task_evaluation_terminal_adoption_retirement import validate_retirement
        validate_retirement(receipt=receipt, attempt=attempt)
        return receipt
    original = receipt.get("original_blocked_launch_receipt") or {}
    if (
        receipt.get("schema_version") != SCHEMA
        or receipt.get("status") != "cancelled_before_controls_eligibility"
        or receipt.get("receipt_digest") != canonical_digest(receipt, digest_field="receipt_digest")
        or receipt.get("attempt_digest") != attempt.get("attempt_digest")
        or receipt.get("intent_digest") != attempt.get("intent_digest")
        or receipt.get("attempt_id") != attempt.get("attempt_id")
        or receipt.get("maximum_spend_usd") != attempt.get("maximum_spend_usd")
        or receipt.get("provider") != attempt.get("provider")
        or original.get("schema_version") != "task_evaluation_launch_receipt.v1"
        or original.get("status") != "blocked"
        or original.get("receipt_digest") != cross_runtime_canonical_digest(original, digest_field="receipt_digest")
        or original.get("source_commit") != attempt.get("source_commit")
        or receipt.get("downstream_execution_eligible") is not False
        or receipt.get("provider_mutation_performed") is not False
    ):
        raise ValueError("unstarted_controls_cancellation_invalid")
    return receipt


def cancel_unstarted_controls_reservations(*, launch_root: str | Path, scene_root: str | Path, dry_run: bool = False) -> dict[str, Any]:
    from .task_evaluation_scene_configuration_publication_recovery import publication_projection_lock
    with publication_projection_lock(Path(launch_root)):
        return _cancel_unstarted_controls_reservations(launch_root=launch_root, scene_root=scene_root, dry_run=dry_run)


def _cancel_unstarted_controls_reservations(*, launch_root: str | Path, scene_root: str | Path, dry_run: bool = False) -> dict[str, Any]:
    from . import task_evaluation_scene_intake as intake
    from .task_evaluation_configured_controls_autostart import validate_configured_controls_autostart_intent

    run = Path(launch_root).resolve(strict=True)
    launch_path = run / "launch_receipt.json"
    launch = _read(launch_path)
    profile_path = run / "launch_profile.json"
    profile = _read(profile_path)
    if (
        launch.get("status") != "blocked"
        or launch.get("receipt_digest") != cross_runtime_canonical_digest(launch, digest_field="receipt_digest")
        or profile.get("profile_digest") != canonical_digest(profile, digest_field="profile_digest")
        or launch.get("launch_profile_digest") != profile.get("profile_digest")
        or profile.get("source_commit") != launch.get("source_commit")
        or (profile.get("task_evaluation_run") or {}).get("run_mode") != "scene_configuration"
        or (launch.get("terminal_evidence") or {}).get("status") != "blocked"
    ):
        raise ValueError("unstarted_controls_source_was_eligible_or_invalid")
    entries = [r for r in profile.get("immutable_inputs", []) if r.get("name") == "configured_controls_autostart_intent"]
    if len(entries) != 1:
        raise ValueError("unstarted_controls_intent_missing")
    intent_path = Path(entries[0]["path"])
    if _file(intent_path)["digest"] != entries[0].get("digest"):
        raise ValueError("unstarted_controls_intent_changed")
    intent = validate_configured_controls_autostart_intent(_read(intent_path))
    if intent["expected_production_commit"] != launch["source_commit"] or intent["configuration_adoption"] != {"mode": "same_commit_automatic"}:
        raise ValueError("unstarted_controls_source_binding_invalid")
    phase_bindings = []
    for phase in ("construction", "controls"):
        authorization = _read(Path(intent["phases"][phase]["authorization_path"]))
        owner = authorization.get("scene_owner_attempt") or {}
        binding = owner.get("scene_attempt_binding") or {}
        if owner.get("phase") != phase or binding.get("source_commit") != launch["source_commit"]:
            raise ValueError("unstarted_controls_owner_binding_invalid")
        phase_bindings.append(binding)
    first, second = phase_bindings
    stem = str(first.get("attempt_id", "")).removesuffix("-construction")
    if not stem.startswith("controls-") or second.get("attempt_id") != stem + "-controls" or any(
        first.get(k) != second.get(k) for k in ("intent_id", "intent_digest", "input_digest", "source_commit", "runtime_digest")
    ):
        raise ValueError("unstarted_controls_phase_binding_invalid")
    root = Path(scene_root)
    directory = root / first["intent_id"]
    with intake._lock(root):
        owner_intent = intake._read(directory / "intent.json", "intent_digest")
        if owner_intent["intent_digest"] != first["intent_digest"]:
            raise ValueError("unstarted_controls_owner_intent_invalid")
        # Reopen immediately before durable retirement. A completed projection
        # is never eligible for this cancellation path, even if a child failed.
        if _read(launch_path) != launch:
            raise ValueError("unstarted_controls_source_changed")
        receipts = []
        for phase in ("construction", "controls", "placement"):
            attempt_path = directory / "attempts" / (stem + "-" + phase + ".json")
            attempt = intake._read(attempt_path, "attempt_digest")
            if any(attempt.get(k) != first.get(k) for k in ("intent_id", "intent_digest", "input_digest", "source_commit", "runtime_digest")):
                raise ValueError("unstarted_controls_reservation_binding_invalid")
            if attempt["provider"] != ("openai" if phase == "placement" else "vast"):
                raise ValueError("unstarted_controls_reservation_provider_invalid")
            cancellation = validated_cancellation(directory, attempt)
            if cancellation is None:
                cancellation = {
                    "schema_version": SCHEMA, "status": "cancelled_before_controls_eligibility",
                    "attempt_id": attempt["attempt_id"], "attempt_digest": attempt["attempt_digest"],
                    "intent_digest": attempt["intent_digest"], "maximum_spend_usd": attempt["maximum_spend_usd"],
                    "provider": attempt["provider"], "original_blocked_launch_receipt": launch,
                    "source_profile": _file(profile_path), "source_autostart_intent": _file(intent_path),
                    "downstream_execution_eligible": False, "provider_mutation_performed": False,
                }
                cancellation["receipt_digest"] = canonical_digest(cancellation, digest_field="receipt_digest")
                target = directory / DIRECTORY / (attempt["attempt_id"] + ".json")
                if not dry_run:
                    target.parent.mkdir(mode=0o750, exist_ok=True)
                    intake.write_exclusive(target, cancellation)
            receipts.append(cancellation)
    return {"status": "would_cancel_unstarted_controls" if dry_run else "cancelled_unstarted_controls", "cancellations": receipts, "provider_mutation_performed": False}
