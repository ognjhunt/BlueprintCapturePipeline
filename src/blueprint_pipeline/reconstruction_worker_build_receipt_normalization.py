"""Normalize paid image-build evidence into the canonical worker receipt.

The remote build script, shared CPU builder, and teardown lane deliberately emit
separate records.  This module joins those records without upgrading an image
build into GPU-runtime or scientific evidence.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any, Mapping

from .decision_evidence_contracts import canonical_digest
from .reconstruction_worker_build_packet import (
    PAID_ENVELOPE_SCHEMA_VERSION,
    REMOTE_PACKET_SCHEMA_VERSION,
    validate_reconstruction_worker_archive,
)
from .reconstruction_worker_contracts import (
    ReconstructionWorkerContractError,
    build_worker_build_receipt,
    build_worker_stack_manifest,
)


class ReconstructionWorkerBuildNormalizationError(ValueError):
    def __init__(self, codes: list[str]) -> None:
        self.codes = tuple(sorted(set(code for code in codes if code)))
        super().__init__("; ".join(self.codes))


def _clone(value: Mapping[str, Any]) -> dict[str, Any]:
    try:
        return json.loads(json.dumps(dict(value), allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ReconstructionWorkerBuildNormalizationError(
            ["worker_build_normalization_input_not_json"]
        ) from exc


def _finite(value: Any, *, minimum: float = 0.0) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and float(value) >= minimum
    )


def _packet_digest(packet: Mapping[str, Any]) -> str:
    unsigned = _clone(packet)
    unsigned.pop("manifest_path", None)
    return canonical_digest(unsigned, digest_field="remote_build_packet_digest")


def _artifact_digest(value: Mapping[str, Any]) -> str:
    return canonical_digest(_clone(value), digest_field="__evidence_digest__")


def compile_reconstruction_worker_build_receipt(
    *,
    worker_stack_manifest: Mapping[str, Any],
    remote_build_packet: Mapping[str, Any],
    remote_build_receipt: Mapping[str, Any],
    builder_run_result: Mapping[str, Any],
    teardown_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Join exact-source build and provider-zero evidence, failing closed.

    The returned v1 receipt is the stable contract consumed by pose and trainer
    request compilers.  Governance bindings remain present as additional fields
    so replay cannot detach the image from the accepted license and paid envelope.
    """

    try:
        stack = build_worker_stack_manifest(worker_stack_manifest)
    except ReconstructionWorkerContractError as exc:
        raise ReconstructionWorkerBuildNormalizationError(
            ["worker_build_stack_manifest_invalid"]
        ) from exc
    packet = _clone(remote_build_packet)
    remote = _clone(remote_build_receipt)
    builder = _clone(builder_run_result)
    teardown = _clone(teardown_receipt)
    errors: list[str] = []

    if (
        packet.get("schema_version") != REMOTE_PACKET_SCHEMA_VERSION
        or packet.get("packet_kind") != "reconstruction_worker_image"
        or packet.get("status") != "ready"
        or packet.get("blockers") != []
        or packet.get("source_worktree_dirty") is not False
        or packet.get("provider_launch_performed_by_packet") is not False
        or packet.get("raw_secret_values_recorded") is not False
        or packet.get("remote_build_packet_digest") != _packet_digest(packet)
    ):
        errors.append("worker_build_remote_packet_not_accepted")
    archive_blockers = validate_reconstruction_worker_archive(packet)
    if archive_blockers:
        errors.append("worker_build_remote_packet_archive_invalid")

    envelope_value = packet.get("paid_execution_envelope")
    envelope = _clone(envelope_value) if isinstance(envelope_value, Mapping) else {}
    if (
        envelope.get("schema_version") != PAID_ENVELOPE_SCHEMA_VERSION
        or envelope.get("paid_execution_envelope_digest")
        != canonical_digest(envelope, digest_field="paid_execution_envelope_digest")
        or envelope.get("paid_execution_envelope_digest")
        != packet.get("paid_execution_envelope_digest")
        or envelope.get("authorized_action") != "cpu-build"
        or envelope.get("paid_mutation_authorized") is not True
        or envelope.get("authority_issued_by_agent") is not False
        or not isinstance(envelope.get("authority_id"), str)
        or not envelope.get("authority_id")
        or envelope.get("source_commit_sha") != packet.get("source_commit")
        or envelope.get("worker_stack_manifest_digest")
        != packet.get("worker_stack_manifest_digest")
        or envelope.get("license_inventory_digest") != packet.get("license_inventory_digest")
        or envelope.get("license_review_receipt_digest")
        != packet.get("license_review_receipt_digest")
        or isinstance(envelope.get("retry_cap"), bool)
        or not isinstance(envelope.get("retry_cap"), int)
        or not 0 <= envelope.get("retry_cap") <= 1
    ):
        errors.append("worker_build_paid_execution_envelope_invalid")

    if (
        remote.get("schema_version") != "reconstruction_worker_build_receipt.v2"
        or remote.get("build_receipt_digest")
        != canonical_digest(remote, digest_field="build_receipt_digest")
        or remote.get("status") != "built"
        or remote.get("blockers") != []
        or remote.get("scientific_qualification_inferred") is not False
        or remote.get("build_healthcheck_embedded") is not True
        or remote.get("runtime_gpu_healthcheck_completed") is not False
        or remote.get("raw_secret_values_recorded") is not False
        or remote.get("proof_effect") != "none"
        or remote.get("claim_ceiling") != "resolved_worker_image_build_only"
    ):
        errors.append("worker_build_remote_receipt_not_accepted")
    resolved_image = str(remote.get("resolved_image_digest") or "")
    expected_image_ref = str(packet.get("image_ref") or "")
    expected_repository = (
        expected_image_ref.rsplit("/", 1)[0]
        + "/"
        + expected_image_ref.rsplit("/", 1)[-1].rsplit(":", 1)[0]
    )
    if (
        re.fullmatch(r"[^\s@]+@sha256:[0-9a-f]{64}", resolved_image) is None
        or resolved_image.split("@", 1)[0] != expected_repository
    ):
        errors.append("worker_build_resolved_image_binding_invalid")

    binding_pairs = (
        (packet.get("source_commit"), stack.get("source_commit_sha")),
        (packet.get("source_commit"), remote.get("source_commit_sha")),
        (
            packet.get("worker_stack_manifest_digest"),
            stack.get("worker_stack_manifest_digest"),
        ),
        (
            packet.get("worker_stack_manifest_digest"),
            remote.get("worker_stack_manifest_digest"),
        ),
        (packet.get("license_inventory_digest"), remote.get("license_inventory_digest")),
        (
            packet.get("license_review_receipt_digest"),
            remote.get("license_review_receipt_digest"),
        ),
        (
            packet.get("paid_execution_envelope_digest"),
            remote.get("paid_execution_envelope_digest"),
        ),
        (
            "sha256:" + str(packet.get("context_manifest_sha256") or ""),
            remote.get("build_context_digest"),
        ),
    )
    if any(left != right for left, right in binding_pairs):
        errors.append("worker_build_artifact_binding_mismatch")

    if (
        builder.get("schema_version") != "groot_oscar_digitalocean_builder_run.v1"
        or builder.get("status") != "completed"
        or builder.get("blockers") != []
        or builder.get("build_exit_code") != 0
        or builder.get("packet_kind") != "reconstruction_worker_image"
        or builder.get("source_commit") != packet.get("source_commit")
        or builder.get("provider_absence_confirmed") is not True
        or builder.get("local_capability_cleanup_verified") is not True
        or builder.get("raw_secret_values_recorded") is not False
    ):
        errors.append("worker_build_outer_builder_result_not_accepted")
    claim_boundary = builder.get("claim_boundary")
    if not isinstance(claim_boundary, Mapping) or any(
        claim_boundary.get(field) is not True
        for field in (
            "image_build_is_not_model_cache_verification",
            "image_build_is_not_runpod_startup",
            "image_build_is_not_task_success",
        )
    ):
        errors.append("worker_build_outer_claim_boundary_invalid")

    if (
        teardown.get("schema_version") != "groot_oscar_digitalocean_builder_teardown.v1"
        or teardown.get("droplet_id") != builder.get("droplet_id")
        or teardown.get("verify_http_status") != 404
        or teardown.get("provider_absence_confirmed") is not True
        or teardown.get("raw_secret_values_recorded") is not False
    ):
        errors.append("worker_build_teardown_not_accepted")

    duration = teardown.get("elapsed_seconds")
    cost = teardown.get("maximum_compute_spend_usd")
    if (
        not _finite(duration)
        or not _finite(cost)
        or not _finite(remote.get("duration_seconds"))
        or not _finite(remote.get("cost_usd"))
        or float(remote.get("duration_seconds", 0)) > float(duration or 0)
        or float(remote.get("cost_usd", 0)) > float(cost or 0)
        or builder.get("maximum_compute_spend_usd") != cost
        or not _finite(envelope.get("max_spend_usd"), minimum=0.0000001)
        or not isinstance(envelope.get("hard_ttl_seconds"), int)
        or isinstance(envelope.get("hard_ttl_seconds"), bool)
        or float(cost or 0) > float(envelope.get("max_spend_usd") or 0)
        or float(duration or 0) > float(envelope.get("hard_ttl_seconds") or 0)
    ):
        errors.append("worker_build_cost_or_duration_outside_envelope")

    if errors:
        raise ReconstructionWorkerBuildNormalizationError(errors)

    logs = [
        {"artifact_id": "remote_build_packet", "digest": packet["remote_build_packet_digest"]},
        {"artifact_id": "remote_build_receipt", "digest": remote["build_receipt_digest"]},
        {"artifact_id": "builder_run_result", "digest": _artifact_digest(builder)},
        {"artifact_id": "provider_teardown", "digest": _artifact_digest(teardown)},
    ]
    value = {
        "worker_stack_manifest_digest": stack["worker_stack_manifest_digest"],
        "status": "built",
        "resolved_image_digest": remote["resolved_image_digest"],
        "source_commit_sha": stack["source_commit_sha"],
        "build_context_digest": remote["build_context_digest"],
        "duration_seconds": float(duration),
        "cost_usd": float(cost),
        "logs": logs,
        "blockers": [],
        "scientific_qualification_inferred": False,
        "remote_build_packet_digest": packet["remote_build_packet_digest"],
        "remote_build_receipt_digest": remote["build_receipt_digest"],
        "license_inventory_digest": packet["license_inventory_digest"],
        "license_review_receipt_digest": packet["license_review_receipt_digest"],
        "paid_execution_envelope_digest": packet["paid_execution_envelope_digest"],
        "paid_authority_id": envelope["authority_id"],
        "provider_zero_verified": True,
        "runtime_gpu_healthcheck_completed": False,
        "proof_effect": "none",
        "claim_ceiling": "resolved_worker_image_build_only",
        "warnings": [
            "image build and provider-zero evidence do not prove GPU runtime or reconstruction quality"
        ],
    }
    try:
        return build_worker_build_receipt(value)
    except ReconstructionWorkerContractError as exc:
        raise ReconstructionWorkerBuildNormalizationError(
            ["worker_build_canonical_receipt_invalid"]
        ) from exc


__all__ = [
    "ReconstructionWorkerBuildNormalizationError",
    "compile_reconstruction_worker_build_receipt",
]
