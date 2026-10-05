"""Publish conservative project exposure from retained scene reservations.

This never calls a provider or infers a zero bill. Every unreconciled reservation
stays charged at its full cap, including expired/revoked and failed attempts.
Only a digest-bound cancellation proving no retained exposure, or authoritative
posted cost bound to the exact reservation, releases a hold. Terminal execution
alone is not evidence of a zero bill.
The retained official-source seed remains the opening accounting authority.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from .task_evaluation_scene_reservation_spend_evidence import (
    _record,
    scene_reservation_spend_record,
)

from .decision_evidence_contracts import canonical_digest
from .task_evaluation_scene_intake import _read as read_scene, _lock
from .validation_file_digests import file_digest_scope

MONITOR_MANAGED_BY = "blueprint_pipeline.task_evaluation_scene_preparation_installation"


def _require_reservation_posted_coverage(record: dict[str, Any], entry: dict[str, Any]) -> None:
    """A compute-only/zero-provider bill cannot erase retained authoring cost.

    Until coverage of the whole retained exposure is proven, fail closed and
    preserve the previous conservative pointer instead of publishing a lower
    total. Matching a digest alone is necessary but insufficient.
    """
    retained = float(record["hard_attempt_spend_cap_usd"])
    if entry.get("authority_digest") != record["authorization_digest"]:
        raise ValueError("scene_spend_posted_reservation_identity_mismatch")
    if float(entry["cost_usd"]) + 1e-9 < retained:
        raise ValueError("scene_spend_posted_reservation_partial_coverage")






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
    cancelled = []
    posted = {str(row.get("authority_digest")): row for row in seed["posted_entries"]
              if row.get("authority_digest")}
    covered = []
    for path in sorted(root.glob("scene-*/attempts/*.json")):
        _, record = scene_reservation_spend_record(path)
        digest = record["authorization_digest"]
        if digest in posted:
            _require_reservation_posted_coverage(record, posted[digest])
            # Only exact authority identity in the already validated official
            # seed replaces this hold. Matching an attempt name is insufficient.
            covered.append({"attempt": record, "posted_attempt_id": posted[digest]["attempt_id"]})
        elif record["hard_attempt_spend_cap_usd"] == 0:
            cancelled.append({"attempt": record, "cancellation_digest": record["settlement_digest"]})
        else:
            records.append(record)
    # Recompute holds from the enrollment store, never add a prior snapshot's
    # same reservation a second time. Official seed increments are unchanged.
    legacy = [r for r in seed["unposted_authorities"] if r.get("accounting_kind") != "persistent_scene_reservation"]
    coverage = sorted({str(row["attempt_id"]) for row in seed["posted_entries"]}
                      | {str(row["authorization_digest"]) for row in [*legacy, *records]})
    inventory = {"seed": seed_record, "scene_reservations": records, "legacy_unposted": legacy,
                 "cancelled_before_controls_eligibility": cancelled,
                 "reservations_covered_by_posted_seed": covered,
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


@file_digest_scope()
def publish_current_scene_project_spend(**kwargs: Any) -> dict[str, Any]:
    root = Path(kwargs["scene_root"])
    if not root.is_dir() or any(p.is_symlink() for p in (root, *root.parents)):
        raise ValueError("scene_spend_root_unsafe")
    # The same lock is used by intake reservations. A new hold cannot appear
    # between inventory enumeration and publication of its current pointer.
    with _lock(root):
        return _publish_current_scene_project_spend_locked(**kwargs)


def _configured_monitor() -> dict[str, Any] | None:
    configured = os.getenv("BLUEPRINT_SCENE_PROJECT_SPEND_CONFIG", "")
    if not configured:
        return None
    path = Path(configured)
    _record(path)
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("scene_spend_monitor_config_invalid")
    base_keys = {"schema_version", "scene_root", "seed_reconciliation_path", "output_root",
                 "current_path", "config_digest"}
    managed_keys = base_keys | {"managed_by"}
    reference_keys = {"seed_reconciliation_reference"}
    actual_keys = set(value)
    if (value.get("schema_version") != "task_evaluation_scene_project_spend_monitor.v1"
            or value.get("config_digest") != canonical_digest(value, digest_field="config_digest")
            or actual_keys not in (base_keys, managed_keys, base_keys | reference_keys,
                                   managed_keys | reference_keys)
            or ("managed_by" in value and value["managed_by"] != MONITOR_MANAGED_BY)):
        raise ValueError("scene_spend_monitor_config_invalid")
    reference = value.get("seed_reconciliation_reference")
    if "seed_reconciliation_reference" in value:
        if (not isinstance(reference, dict)
                or set(reference) != {"path", "sha256", "size_bytes"}
                or reference.get("path") != value.get("seed_reconciliation_path")):
            raise ValueError("scene_spend_monitor_seed_reference_invalid")
        try:
            if _record(Path(reference["path"])) != reference:
                raise ValueError("scene_spend_monitor_seed_reference_invalid")
        except (OSError, TypeError, ValueError):
            raise ValueError("scene_spend_monitor_seed_reference_invalid") from None
    return value


def observe_configured_scene_project_spend(*, now: float | None = None) -> dict[str, Any] | None:
    """Observe the publisher's fresh checked exposure without taking its write lock.

    Capacity's sandbox permits writing only capacity reports. The dedicated
    refresh service and activation remain the publication owners; observation
    never restamps freshness, enumerates new reservations or grants execution.
    """
    from .project_spend_reconciliation import validate_project_spend_reconciliation

    value = _configured_monitor()
    if value is None:
        return None
    _record(Path(value["current_path"]))
    pointer = read_scene(Path(value["current_path"]), "receipt_digest")
    observed = pointer.get("observed_at_epoch")
    clock = time.time() if now is None else now
    source = Path(str(pointer.get("path") or ""))
    output = Path(value["output_root"])
    if (set(pointer) != {"schema_version", "path", "digest", "observed_at_epoch", "receipt_digest"}
            or pointer.get("schema_version") != "task_evaluation_project_spend_current.v1"
            or isinstance(observed, bool) or not isinstance(observed, (int, float))
            or not 0 <= clock - observed <= 900
            or not source.is_absolute()
            or source.name != "project_spend_reconciliation.json"):
        raise ValueError("scene_spend_pointer_invalid_or_stale")
    # Resolve both paths, but reject the original traversal/symlink spelling
    # first. A lexical parent test accepts output_root/../receipt.json.
    if (".." in source.parts or ".." in output.parts
            or any(p.is_symlink() for p in (source, *source.parents, output, *output.parents))):
        raise ValueError("scene_spend_pointer_outside_output_root")
    try:
        relative = source.resolve(strict=True).relative_to(output.resolve(strict=True))
    except (OSError, ValueError):
        raise ValueError("scene_spend_pointer_outside_output_root") from None
    if len(relative.parts) != 2:
        raise ValueError("scene_spend_pointer_outside_output_root")
    if _record(source)["sha256"] != pointer["digest"]:
        raise ValueError("scene_spend_pointer_source_changed")
    receipt, record = validate_project_spend_reconciliation(source)
    if record["sha256"] != pointer["digest"]:
        raise ValueError("scene_spend_pointer_source_changed")
    return {"status": "published_project_exposure_observed", "pointer": pointer,
            "total_cost_usd": receipt["total_cost_usd"],
            "accounting_scope": "last_checked_publication_not_new_reservation_admission",
            "provider_mutation_performed": False, "reserved_caps_are_not_actual_billing": True}


def refresh_configured_scene_project_spend(*, now: float | None = None) -> dict[str, Any] | None:
    value = _configured_monitor()
    if value is None:
        return None
    # Stamp the pointer with the caller's ``now`` (the activation tick captures one
    # ``now`` for the whole pass, then gates on ``0 <= now - observed_at_epoch <= 900``;
    # defaulting to ``time.time()`` here stamps a moment LATER than that ``now`` and the
    # gate inverts to a permanent ``project_spend_stale`` on every tick).
    return publish_current_scene_project_spend(now=now, **{k: value[k] for k in (
        "scene_root", "seed_reconciliation_path", "output_root", "current_path")})
