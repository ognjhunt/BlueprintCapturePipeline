"""Immutable queue for production compilation of robot-specific episodes."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest
from .task_evaluation_launch_preparation_queue import (
    write_launch_preparation_record_exclusive,
)
from .task_evaluation_scene_construction_queue import (
    ensure_scene_construction_queue_root,
)


ENVELOPE_SCHEMA_VERSION = "task_evaluation_episode_compilation_envelope.v1"


class TaskEvaluationEpisodeCompilationQueueError(ValueError):
    """A configured-scene evaluation could not be handed off immutably."""


def stage_episode_compilation(
    *,
    request: Mapping[str, Any],
    preparation_result: Mapping[str, Any],
    configured_revision: Mapping[str, Any],
    configured_scene_bundle_reference: Mapping[str, Any],
    queue_root: str | Path,
) -> dict[str, Any]:
    if (
        request.get("run_mode") != "episode_evaluation"
        or request.get("construction", {}).get("mode")
        != "reuse_configured_scene"
        or preparation_result.get("full_byte_service_account_readback_passed")
        is not True
        or configured_revision.get("status") != "configured"
        or configured_revision.get("configured_scene_bundle")
        != {
            key: configured_scene_bundle_reference.get(key)
            for key in ("uri", "digest", "size_bytes")
        }
        or configured_scene_bundle_reference.get(
            "full_byte_service_account_readback_passed"
        )
        is not True
    ):
        raise TaskEvaluationEpisodeCompilationQueueError(
            "episode_compilation_handoff_binding_invalid"
        )
    envelope: dict[str, Any] = {
        "schema_version": ENVELOPE_SCHEMA_VERSION,
        "compilation_id": request["preparation_id"],
        "preparation_id": request["preparation_id"],
        "run_id": request["run_id"],
        "team_namespace": request["team_namespace"],
        "expected_production_commit": request["expected_production_commit"],
        "configured_scene_revision_digest": configured_revision[
            "revision_digest"
        ],
        "configured_scene_bundle": dict(configured_scene_bundle_reference),
        "materialized_references": [
            dict(row) for row in preparation_result.get("references") or []
        ],
        "request": dict(request),
        "preparation_result_digest": preparation_result["result_digest"],
        "automatic_progression_required": True,
        "robot_specific_episode_packet_compiled_in_production": True,
        "customer_supplied_prebuilt_episode_packet": False,
        "production_compiler_owns_episode_packet": True,
        "provider_mutation_performed": False,
        "paid_execution_requested": False,
        "envelope_digest": "",
    }
    envelope["envelope_digest"] = canonical_digest(
        envelope, digest_field="envelope_digest"
    )
    root = ensure_scene_construction_queue_root(queue_root)
    filename = (
        f"{request['preparation_id']}-"
        f"{envelope['envelope_digest'].removeprefix('sha256:')}.json"
    )
    existing = [
        root / state / filename
        for state in ("pending", "processing", "completed", "blocked")
        if (root / state / filename).exists()
    ]
    if existing:
        if len(existing) != 1 or existing[0].is_symlink():
            raise TaskEvaluationEpisodeCompilationQueueError(
                "episode_compilation_queue_identity_ambiguous"
            )
        try:
            observed = json.loads(existing[0].read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise TaskEvaluationEpisodeCompilationQueueError(
                "episode_compilation_queue_existing_invalid"
            ) from exc
        if observed != envelope:
            raise TaskEvaluationEpisodeCompilationQueueError(
                "episode_compilation_queue_immutable_conflict"
            )
        destination = existing[0]
        created = False
    else:
        destination = root / "pending" / filename
        try:
            write_launch_preparation_record_exclusive(destination, envelope)
        except FileExistsError as exc:
            raise TaskEvaluationEpisodeCompilationQueueError(
                "episode_compilation_queue_race_conflict"
            ) from exc
        created = True
    receipt: dict[str, Any] = {
        "schema_version": "task_evaluation_episode_compilation_intake_receipt.v1",
        "status": "queued_for_production_episode_compilation",
        "compilation_id": request["preparation_id"],
        "run_id": request["run_id"],
        "configured_scene_revision_digest": configured_revision[
            "revision_digest"
        ],
        "envelope_digest": envelope["envelope_digest"],
        "queue_path": str(destination),
        "created": created,
        "automatic_progression_required": True,
        "provider_mutation_performed": False,
        "paid_execution_requested": False,
        "receipt_digest": "",
    }
    receipt["receipt_digest"] = canonical_digest(
        receipt, digest_field="receipt_digest"
    )
    return receipt


__all__ = [
    "ENVELOPE_SCHEMA_VERSION",
    "TaskEvaluationEpisodeCompilationQueueError",
    "stage_episode_compilation",
]
