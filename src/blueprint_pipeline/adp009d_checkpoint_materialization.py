"""Materialize a frozen ADP-009D policy checkpoint on the GPU worker.

The orchestrating machine never needs a copy.  The worker is ephemeral and
cannot read local disk, so staging weights locally would mean three transfers
(bucket to laptop, laptop to object store, object store to worker) where one
suffices, and the slowest leg would be a home connection rather than a
datacenter one.

Every frozen candidate is reachable without credentials, which was measured
rather than assumed: ``gs://openpi-assets`` lists anonymously over the GCS JSON
API, and all three NVIDIA repositories return ``gated=false, private=false`` at
their pinned revisions with no Authorization header.  So the ADP lanes keep
``forward_hf_token=False`` and no secret is ever handed to a rented host.

This module plans and verifies; it does not fetch.  The download itself is an
injected callable so the contract is testable without a network or a GPU.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

from .adp009d_policy_candidate_admission import EXPECTED_CANDIDATES, PROGRAM_ID
from .decision_evidence_contracts import canonical_digest

MATERIALIZATION_SCHEMA_VERSION = "adp009d_checkpoint_materialization.v1"

GCS_JSON_API = "https://storage.googleapis.com/storage/v1/b"
HUGGINGFACE_API = "https://huggingface.co/api/models"

SOURCE_PUBLIC_GCS = "public_gcs"
SOURCE_PUBLIC_HUGGINGFACE = "public_huggingface"

# Measured 2026-08-07: every one of these listed with no credentials at all.
CANDIDATE_SOURCES: dict[str, str] = {
    "pi05_droid": SOURCE_PUBLIC_GCS,
    "groot_n17_droid": SOURCE_PUBLIC_HUGGINGFACE,
    "groot_n16_droid": SOURCE_PUBLIC_HUGGINGFACE,
    "cosmos3_edge_policy_droid": SOURCE_PUBLIC_HUGGINGFACE,
}

BLOCKER_UNKNOWN_CANDIDATE = "checkpoint_materialization_unknown_candidate"
BLOCKER_CREDENTIALS_FORWARDED = "checkpoint_materialization_credentials_forwarded"
BLOCKER_BYTE_COUNT_MISMATCH = "checkpoint_materialization_byte_count_mismatch"
BLOCKER_NOT_ON_WORKER = "checkpoint_materialization_not_on_gpu_worker"
BLOCKER_NO_OBJECTS = "checkpoint_materialization_no_objects_retained"
BLOCKER_REVISION_MISMATCH = "checkpoint_materialization_revision_mismatch"


class CheckpointMaterializationError(ValueError):
    """Fail-closed checkpoint materialization contract errors."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(sorted({str(e) for e in errors if str(e)}))
        super().__init__(";".join(self.errors))


def plan_checkpoint_materialization(candidate_id: str) -> dict[str, Any]:
    """Describe exactly what the worker must fetch, and from where."""

    if candidate_id not in EXPECTED_CANDIDATES:
        raise CheckpointMaterializationError(
            [f"{BLOCKER_UNKNOWN_CANDIDATE}:{candidate_id}"]
        )
    expected = EXPECTED_CANDIDATES[candidate_id]
    source = CANDIDATE_SOURCES[candidate_id]
    repository = str(expected["checkpoint_repository"])

    if source == SOURCE_PUBLIC_GCS:
        prefix = repository.removeprefix("gs://")
        bucket, _, object_prefix = prefix.partition("/")
        listing_url = (
            f"{GCS_JSON_API}/{bucket}/o?prefix={object_prefix.rstrip('/')}/"
        )
    else:
        listing_url = (
            f"{HUGGINGFACE_API}/{repository.removeprefix('https://huggingface.co/')}"
            f"/revision/{expected['checkpoint_revision']}"
        )

    return {
        "schema_version": MATERIALIZATION_SCHEMA_VERSION,
        "program_id": PROGRAM_ID,
        "candidate_id": candidate_id,
        "source": source,
        "credentials_required": False,
        "checkpoint_repository": repository,
        "checkpoint_revision": str(expected["checkpoint_revision"]),
        "expected_total_bytes": int(expected["checkpoint_total_bytes"]),
        "expected_inventory_digest": str(expected["checkpoint_inventory_digest"]),
        "listing_url": listing_url,
        "materialize_on": "gpu_worker",
        "stage_locally": False,
    }


def verify_materialized_checkpoint(
    *,
    candidate_id: str,
    objects: Sequence[Mapping[str, Any]],
    materialized_on: str,
    credentials_forwarded: bool,
    observed_revision: str | None = None,
) -> dict[str, Any]:
    """Bind materialized bytes to the frozen candidate identity, or fail closed.

    ``objects`` are the retained per-object rows -- name and size at minimum --
    exactly as the worker observed them after download.
    """

    if candidate_id not in EXPECTED_CANDIDATES:
        raise CheckpointMaterializationError(
            [f"{BLOCKER_UNKNOWN_CANDIDATE}:{candidate_id}"]
        )
    expected = EXPECTED_CANDIDATES[candidate_id]
    errors: list[str] = []

    rows = [dict(row) for row in objects]
    if not rows:
        errors.append(BLOCKER_NO_OBJECTS)

    if materialized_on != "gpu_worker":
        # Local staging is not merely wasteful here, it breaks the claim that
        # the bytes the worker ran are the bytes this receipt describes.
        errors.append(f"{BLOCKER_NOT_ON_WORKER}:{materialized_on}")

    if credentials_forwarded:
        # Every frozen candidate is public.  A run that forwarded a token
        # either did not need to, or fetched something other than the frozen
        # public artifact -- both are worth stopping for.
        errors.append(BLOCKER_CREDENTIALS_FORWARDED)

    observed_bytes = 0
    for index, row in enumerate(rows):
        try:
            observed_bytes += int(row.get("size_bytes", row.get("size", 0)))
        except (TypeError, ValueError):
            errors.append(f"checkpoint_materialization_object_size_invalid:{index}")

    expected_bytes = int(expected["checkpoint_total_bytes"])
    if rows and observed_bytes != expected_bytes:
        errors.append(
            f"{BLOCKER_BYTE_COUNT_MISMATCH}:{observed_bytes}!={expected_bytes}"
        )

    expected_revision = str(expected["checkpoint_revision"])
    if observed_revision is not None and observed_revision != expected_revision:
        errors.append(f"{BLOCKER_REVISION_MISMATCH}:{observed_revision}")

    if errors:
        raise CheckpointMaterializationError(errors)

    manifest = sorted(
        (
            {
                "name": str(row.get("name", "")),
                "size_bytes": int(row.get("size_bytes", row.get("size", 0))),
            }
            for row in rows
        ),
        key=lambda item: item["name"],
    )
    receipt: dict[str, Any] = {
        "schema_version": MATERIALIZATION_SCHEMA_VERSION,
        "program_id": PROGRAM_ID,
        "status": "materialized",
        "candidate_id": candidate_id,
        "source": CANDIDATE_SOURCES[candidate_id],
        "checkpoint_repository": str(expected["checkpoint_repository"]),
        "checkpoint_revision": expected_revision,
        "object_count": len(manifest),
        "total_bytes": observed_bytes,
        "object_manifest_sha256": "sha256:"
        + hashlib.sha256(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "materialized_on": materialized_on,
        "credentials_forwarded": False,
        "staged_locally": False,
        "candidate_policy_queried": False,
    }
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    return receipt


__all__ = [
    "CANDIDATE_SOURCES",
    "MATERIALIZATION_SCHEMA_VERSION",
    "SOURCE_PUBLIC_GCS",
    "SOURCE_PUBLIC_HUGGINGFACE",
    "CheckpointMaterializationError",
    "plan_checkpoint_materialization",
    "verify_materialized_checkpoint",
]
