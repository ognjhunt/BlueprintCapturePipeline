"""Publish conservative project exposure from retained scene reservations.

This never calls a provider or infers a zero bill. Every unreconciled reservation
stays charged at its full cap, including expired/revoked and failed attempts.
The retained official-source seed remains the opening accounting authority.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest
from .task_evaluation_scene_intake import _read as read_scene, _lock


def _record(path: Path) -> dict[str, Any]:
    if not path.is_file() or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("scene_spend_source_unsafe")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"path": str(path), "sha256": "sha256:" + digest, "size_bytes": path.stat().st_size}


def scene_reservation_spend_record(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Normalize a reservation for accounting, never into a provider launch grant."""
    record = _record(path)
    attempt = read_scene(path, "attempt_digest")
    intent_path = path.parent.parent / "intent.json"
    intent_record = _record(intent_path)
    intent = read_scene(intent_path, "intent_digest")
    cap = attempt.get("maximum_spend_usd")
    if (attempt.get("schema_version") != "task_evaluation_scene_attempt.v1"
            or attempt.get("intent_digest") != intent.get("intent_digest")
            or attempt.get("intent_id") != intent.get("intent_id")
            or attempt.get("provider") not in intent["request"]["execution"]["allowed_providers"]
            or isinstance(cap, bool) or not isinstance(cap, (int, float)) or not math.isfinite(cap) or cap <= 0):
        raise ValueError("scene_spend_reservation_invalid")
    return {**attempt, "authorization_digest": attempt["attempt_digest"], "hard_attempt_spend_cap_usd": cap}, {
        **record, "authorization_digest": attempt["attempt_digest"], "hard_attempt_spend_cap_usd": cap,
        "accounting_kind": "persistent_scene_reservation", "owner_intent": intent_record,
    }


def _publish_current_scene_project_spend_locked(*, scene_root: str | Path, seed_reconciliation_path: str | Path,
                                        output_root: str | Path, current_path: str | Path,
                                        now: float | None = None) -> dict[str, Any]:
    """Reopen the seed and every enrolled hold, then publish a fresh checked pointer."""
    from .project_spend_reconciliation import (
        materialize_project_spend_reconciliation, validate_project_spend_reconciliation,
    )
    from .task_evaluation_launch_preparation_queue import _write_launch_preparation_record_exclusive_locked

    seed, seed_record = validate_project_spend_reconciliation(seed_reconciliation_path)
    root = Path(scene_root)
    if not root.is_dir() or any(p.is_symlink() for p in (root, *root.parents)):
        raise ValueError("scene_spend_root_unsafe")
    records = []
    for path in sorted(root.glob("scene-*/attempts/*.json")):
        records.append(scene_reservation_spend_record(path)[1])
    # Recompute holds from the enrollment store, never add a prior snapshot's
    # same reservation a second time. Official seed increments are unchanged.
    legacy = [r for r in seed["unposted_authorities"] if r.get("accounting_kind") != "persistent_scene_reservation"]
    coverage = sorted({str(row["attempt_id"]) for row in seed["posted_entries"]}
                      | {str(row["authorization_digest"]) for row in [*legacy, *records]})
    inventory = {"seed": seed_record, "scene_reservations": records, "legacy_unposted": legacy,
                 "expected_coverage_ids": coverage}
    snapshot_digest = canonical_digest(inventory)
    destination = Path(output_root) / snapshot_digest[7:]
    if any(p.is_symlink() for p in (destination, *destination.parents)):
        raise ValueError("scene_spend_output_unsafe")
    destination.mkdir(parents=True, exist_ok=True, mode=0o750)
    evidence_path = destination / "source_inventory.json"
    if not evidence_path.exists():
        _write_launch_preparation_record_exclusive_locked(evidence_path, inventory)
    elif json.loads(evidence_path.read_text()) != inventory:
        raise ValueError("scene_spend_inventory_conflict")
    receipt_path = destination / "project_spend_reconciliation.json"
    if not receipt_path.exists():
        authority = seed["completeness_authority"]
        materialize_project_spend_reconciliation(
            baseline_authority_path=seed["baseline_authority"]["path"],
            posted_reconciliation_paths=[r["path"] for r in seed["posted_reconciliations"]],
            unposted_authority_paths=[r["path"] for r in [*legacy, *records]],
            expected_coverage_ids=coverage,
            completeness_reference=authority["authority_reference"] + "/scene-reservations/" + snapshot_digest,
            authorized_by=authority["authorized_by"], authorized_on=authority["authorized_on"],
            output_path=receipt_path,
        )
    value, record = validate_project_spend_reconciliation(receipt_path)
    # Freshness is the time these retained sources were actually reopened, not
    # a claim that the provider posted a new bill or that reserved funds were spent.
    pointer = {"schema_version": "task_evaluation_project_spend_current.v1", "path": str(receipt_path),
               "digest": record["sha256"], "observed_at_epoch": time.time() if now is None else now}
    pointer["receipt_digest"] = canonical_digest(pointer, digest_field="receipt_digest")
    current = Path(current_path)
    if any(p.is_symlink() for p in (current, *current.parents)):
        raise ValueError("scene_spend_pointer_unsafe")
    current.parent.mkdir(parents=True, exist_ok=True, mode=0o750)
    temporary = current.with_name("." + current.name + "." + str(os.getpid()))
    try:
        with temporary.open("x") as stream:
            json.dump(pointer, stream, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.chmod(0o440)
        os.replace(temporary, current)
    finally:
        temporary.unlink(missing_ok=True)
    return {"status": "current_project_exposure_published", "pointer": pointer,
            "total_cost_usd": value["total_cost_usd"], "scene_reservation_count": len(records),
            "accounting_scope": "retained_official_source_seed_plus_full_enrolled_reservation_caps",
            "provider_mutation_performed": False, "reserved_caps_are_not_actual_billing": True}


def publish_current_scene_project_spend(**kwargs: Any) -> dict[str, Any]:
    root = Path(kwargs["scene_root"])
    if not root.is_dir() or any(p.is_symlink() for p in (root, *root.parents)):
        raise ValueError("scene_spend_root_unsafe")
    # The same lock is used by intake reservations. A new hold cannot appear
    # between inventory enumeration and publication of its current pointer.
    with _lock(root):
        return _publish_current_scene_project_spend_locked(**kwargs)


def refresh_configured_scene_project_spend() -> dict[str, Any] | None:
    configured = os.getenv("BLUEPRINT_SCENE_PROJECT_SPEND_CONFIG", "")
    if not configured:
        return None
    path = Path(configured)
    _record(path)
    value = json.loads(path.read_text())
    if (value.get("schema_version") != "task_evaluation_scene_project_spend_monitor.v1"
            or value.get("config_digest") != canonical_digest(value, digest_field="config_digest")
            or set(value) != {"schema_version", "scene_root", "seed_reconciliation_path", "output_root",
                              "current_path", "config_digest"}):
        raise ValueError("scene_spend_monitor_config_invalid")
    return publish_current_scene_project_spend(**{k: value[k] for k in (
        "scene_root", "seed_reconciliation_path", "output_root", "current_path")})
