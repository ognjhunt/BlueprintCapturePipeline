#!/usr/bin/env python3
"""Fail closed when the cross-surface data-retention policy drifts toward premature deletion.

Finding **R048 (P1)**: data-retention was agent-scoped to WebApp Firestore only, unenforced,
and did not reach pipeline output artifacts, hosted/derived world models, or storage. There
was no single, enforced, cross-surface retention contract.

This validator guards the committed machine-readable retention contract
(``configs/data_retention_policy.json``) so it stays a *coherent, enforceable* policy and
cannot silently be edited into aggressive deletion of authoritative capture truth. It mirrors
the fail-closed idiom of ``BlueprintCapture/scripts/validate_storage_lifecycle.py`` (R042).

Capture-truth / provenance guardrails (see docs/DATA_RETENTION_POLICY.md):

  * Raw capture bundles (``scenes/``) are AUTHORITATIVE and are the LONGEST-lived data class.
    Their retention must be >= the documented capture-truth floor and they must be managed by
    the committed GCS lifecycle file in BlueprintCapture (never a plain fast delete here).
  * NOTHING is retained longer than raw capture truth; DERIVED / hosted artifacts are retained
    STRICTLY SHORTER than raw.
  * Any DESTRUCTIVE action must clear the documented minimum retention floor for its class.
  * FINANCIAL / legal-hold data is never auto-deleted (review_then_delete / retain_indefinite
    only) and clears the 7-year financial floor.
  * A Firestore TTL entry must name its ttl_field, and a firestore_ttl entry must be a TTL
    delete -- so the policy and the applied Firestore TTL policy stay in lock-step.

This validator does NOT touch live infrastructure. Apply commands are in the policy doc.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

RAW_CLASS = "raw_capture_authoritative"
RAW_PREFIX = "scenes/"
RAW_LIFECYCLE_FILE = "storage.lifecycle.json"

# Actions that permanently destroy data (must clear a retention floor).
DESTRUCTIVE_ACTIONS = {"delete", "ttl_delete", "tier_then_delete"}
# Non-destructive / gated actions.
_ALLOWED_ACTIONS = DESTRUCTIVE_ACTIONS | {"review_then_delete", "retain_indefinite"}

_ALLOWED_ENFORCERS = {
    "gcs_lifecycle",
    "firestore_ttl",
    "scheduled_job",
    "manual_review",
}

# Classes that carry financial/legal obligations: never auto-deleted.
FINANCIAL_CLASSES = {"financial", "transactional"}


def fail(message: str) -> None:
    print(f"Data retention validation failed: {message}", file=sys.stderr)
    sys.exit(1)


def _int_field(entry: dict, name: str, label: str, *, allow_none: bool = False):
    value = entry.get(name)
    if value is None and allow_none:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        fail(f"{label}: '{name}' must be an integer number of days")
    if value < 0:
        fail(f"{label}: '{name}' must be non-negative")
    return value


def _validate_common(entry: dict, label: str) -> dict:
    if not isinstance(entry, dict):
        fail(f"{label}: each entry must be an object")

    cls = entry.get("class")
    if not isinstance(cls, str) or not cls:
        fail(f"{label}: 'class' must be a non-empty string")

    action = entry.get("action")
    if action not in _ALLOWED_ACTIONS:
        fail(f"{label}: unsupported action {action!r}; expected one of {sorted(_ALLOWED_ACTIONS)}")

    enforced_by = entry.get("enforced_by")
    if enforced_by not in _ALLOWED_ENFORCERS:
        fail(
            f"{label}: unsupported enforced_by {enforced_by!r}; "
            f"expected one of {sorted(_ALLOWED_ENFORCERS)}"
        )

    legal_hold = entry.get("legal_hold", False)
    if not isinstance(legal_hold, bool):
        fail(f"{label}: 'legal_hold' must be a boolean when present")

    # retain_indefinite may omit a numeric retention; everything else needs one.
    allow_none = action == "retain_indefinite"
    retention = _int_field(entry, "retention_days", label, allow_none=allow_none)

    # ── TTL consistency: policy config and applied Firestore TTL must agree. ──
    ttl_field = entry.get("ttl_field")
    if enforced_by == "firestore_ttl":
        if action != "ttl_delete":
            fail(f"{label}: enforced_by=firestore_ttl requires action=ttl_delete (got {action!r})")
        if not isinstance(ttl_field, str) or not ttl_field:
            fail(f"{label}: enforced_by=firestore_ttl requires a non-empty 'ttl_field'")
    elif ttl_field is not None:
        fail(f"{label}: 'ttl_field' is only valid when enforced_by=firestore_ttl")
    if action == "ttl_delete" and enforced_by != "firestore_ttl":
        fail(f"{label}: action=ttl_delete requires enforced_by=firestore_ttl")

    # ── Financial / legal-hold guardrail: never auto-delete. ──
    if cls in FINANCIAL_CLASSES or legal_hold:
        if action in DESTRUCTIVE_ACTIONS:
            fail(
                f"{label}: financial/legal-hold data (class={cls!r}, legal_hold={legal_hold}) must "
                f"use review_then_delete or retain_indefinite, never the auto-destructive {action!r}"
            )

    return {"class": cls, "action": action, "retention_days": retention, "legal_hold": legal_hold}


def validate_policy(config: object, source: str) -> None:
    if not isinstance(config, dict):
        fail(f"{source}: top-level policy must be a JSON object")

    capture_floor = config.get("capture_truth_floor_days")
    if not isinstance(capture_floor, int) or isinstance(capture_floor, bool) or capture_floor <= 0:
        fail(f"{source}: 'capture_truth_floor_days' must be a positive integer")

    financial_floor = config.get("financial_floor_days")
    if not isinstance(financial_floor, int) or isinstance(financial_floor, bool) or financial_floor <= 0:
        fail(f"{source}: 'financial_floor_days' must be a positive integer")

    pii_floor = config.get("pii_floor_days")
    if not isinstance(pii_floor, int) or isinstance(pii_floor, bool) or pii_floor <= 0:
        fail(f"{source}: 'pii_floor_days' must be a positive integer")

    surfaces = config.get("surfaces")
    if not isinstance(surfaces, dict):
        fail(f"{source}: 'surfaces' must be an object")

    storage_prefixes = surfaces.get("storage_prefixes")
    firestore_collections = surfaces.get("firestore_collections")
    if not isinstance(storage_prefixes, list) or not storage_prefixes:
        fail(f"{source}: surfaces.storage_prefixes must be a non-empty array")
    if not isinstance(firestore_collections, list) or not firestore_collections:
        fail(f"{source}: surfaces.firestore_collections must be a non-empty array")

    # ── Validate every entry and locate the authoritative raw entry. ──
    raw_entry = None
    normalized: list[dict] = []

    for index, entry in enumerate(storage_prefixes):
        label = f"{source}: storage_prefixes[{index}]"
        prefix = entry.get("prefix")
        if not isinstance(prefix, str) or not prefix:
            fail(f"{label}: 'prefix' must be a non-empty string")
        bucket = entry.get("bucket")
        if not isinstance(bucket, str) or not bucket:
            fail(f"{label}: 'bucket' must be a non-empty string")
        info = _validate_common(entry, f"{label} (prefix={prefix})")
        info["kind"] = "storage"
        info["id"] = prefix

        if info["class"] == RAW_CLASS:
            if raw_entry is not None:
                fail(f"{label}: multiple entries declare class {RAW_CLASS!r}; there must be exactly one")
            if prefix != RAW_PREFIX:
                fail(f"{label}: the {RAW_CLASS!r} entry must target the {RAW_PREFIX!r} prefix")
            if info["action"] != "tier_then_delete":
                fail(f"{label}: raw capture truth must use action=tier_then_delete (cost-tier before any delete)")
            if entry.get("enforced_by") != "gcs_lifecycle":
                fail(f"{label}: raw capture truth must be enforced_by=gcs_lifecycle")
            managed_by = entry.get("managed_by", "")
            if RAW_LIFECYCLE_FILE not in str(managed_by):
                fail(
                    f"{label}: raw capture truth must be managed_by the committed "
                    f"{RAW_LIFECYCLE_FILE} (BlueprintCapture, finding R042)"
                )
            raw_entry = info

        normalized.append(info)

    for index, entry in enumerate(firestore_collections):
        label = f"{source}: firestore_collections[{index}]"
        collection = entry.get("collection")
        if not isinstance(collection, str) or not collection:
            fail(f"{label}: 'collection' must be a non-empty string")
        info = _validate_common(entry, f"{label} (collection={collection})")
        info["kind"] = "firestore"
        info["id"] = collection
        if info["class"] == RAW_CLASS:
            fail(f"{label}: a Firestore collection may not claim the {RAW_CLASS!r} class")
        normalized.append(info)

    if raw_entry is None:
        fail(f"{source}: exactly one entry must declare the authoritative {RAW_CLASS!r} class ({RAW_PREFIX})")

    raw_retention = raw_entry["retention_days"]

    # ── Capture-truth guardrail: raw >= floor, and raw is the longest-lived class. ──
    if raw_retention < capture_floor:
        fail(
            f"{source}: raw capture retention {raw_retention}d is below the capture-truth floor "
            f"of {capture_floor}d ({capture_floor // 365}y); raw bundles are authoritative provenance"
        )

    # ── Per-entry floors + ordering vs raw. ──
    for info in normalized:
        cls = info["class"]
        retention = info["retention_days"]
        label = f"{source}: {info['kind']}:{info['id']} (class={cls})"

        if cls == RAW_CLASS:
            continue

        # retain_indefinite entries carry no numeric retention -> only the financial floor
        # semantics above apply; nothing lives "longer than raw" in a comparable way.
        if retention is None:
            if not info["legal_hold"] and cls not in FINANCIAL_CLASSES:
                fail(f"{label}: only financial/legal-hold data may use retain_indefinite (no retention_days)")
            continue

        # Nothing may be retained LONGER than authoritative raw capture truth.
        if retention > raw_retention:
            fail(
                f"{label}: retention {retention}d exceeds raw capture retention {raw_retention}d; "
                "no data class may outlive authoritative raw capture truth"
            )

        # Derived / hosted artifacts must be STRICTLY shorter than raw.
        if cls.startswith("derived") and retention >= raw_retention:
            fail(
                f"{label}: derived retention {retention}d must be strictly less than raw "
                f"retention {raw_retention}d; derived artifacts get shorter retention"
            )

        # Financial floor.
        if cls in FINANCIAL_CLASSES or info["legal_hold"]:
            if retention < financial_floor:
                fail(
                    f"{label}: financial/legal-hold retention {retention}d is below the "
                    f"{financial_floor}d ({financial_floor // 365}y) financial floor"
                )

        # PII floor: destructive PII deletion must clear a minimum non-trivial floor so a
        # future edit cannot set an accidental 0/near-0-day purge of lead PII.
        if cls.endswith("lead") and info["action"] in DESTRUCTIVE_ACTIONS:
            if retention < pii_floor:
                fail(f"{label}: PII lead retention {retention}d is below the {pii_floor}d PII floor")

    # ── Validate the default fall-through, if present. ──
    default = config.get("default_firestore_retention")
    if default is not None:
        info = _validate_common(default, f"{source}: default_firestore_retention")
        if info["class"] == RAW_CLASS:
            fail(f"{source}: default_firestore_retention may not claim the raw class")
        if info["retention_days"] is not None and info["retention_days"] > raw_retention:
            fail(f"{source}: default_firestore_retention may not exceed raw capture retention")

    n_storage = len(storage_prefixes)
    n_fs = len(firestore_collections)
    print(
        "Data retention validation passed: "
        f"raw capture truth {RAW_PREFIX} retained {raw_retention}d (>= {capture_floor}d floor, "
        f"managed by BlueprintCapture/{RAW_LIFECYCLE_FILE}); "
        f"{n_storage} storage prefixes + {n_fs} Firestore collections; "
        "no class outlives raw; derived shorter than raw; financial/legal-hold never auto-deleted; "
        "Firestore TTL entries consistent."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the committed cross-surface data-retention policy (finding R048)."
    )
    parser.add_argument(
        "--policy",
        type=Path,
        default=None,
        help="Path to the retention policy JSON (defaults to configs/data_retention_policy.json).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    policy_path = args.policy or (repo_root / "configs" / "data_retention_policy.json")

    if not policy_path.exists():
        fail(f"{policy_path} is missing")

    try:
        config = json.loads(policy_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        fail(f"{policy_path} is not valid JSON: {exc}")

    validate_policy(config, policy_path.name)


if __name__ == "__main__":
    main()
