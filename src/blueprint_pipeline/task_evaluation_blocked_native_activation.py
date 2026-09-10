"""Recognize terminal prelaunch failures without treating them as GPU execution."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

from .decision_evidence_contracts import canonical_digest
from .task_evaluation_retained_controls_evidence import _read, _file


def terminal_blocked_activation(
    *, marker_path: Path, plan: Mapping[str, Any], config: Mapping[str, Any]
) -> list[dict] | None:
    """Return sealed evidence only when the old activation cannot execute."""
    from .task_evaluation_release_identity import running_release_commit

    current = running_release_commit()
    if not current or current == plan["expected_production_commit"]:
        return None
    try:
        marker = _read(marker_path)
        request = marker["activation_request"]
        activation_id = request["activation_id"]
        phase = next(
            (
                name
                for name, row in plan["future_outputs"].items()
                if row["expected_activation_id"] == activation_id
            ),
            None,
        )
        if (
            phase is None
            or marker.get("schema_version") != "task_evaluation_configured_controls_progression.v1"
            or not str(marker.get("status", "")).endswith("_activation_queued")
            or marker.get("expected_production_commit") != plan["expected_production_commit"]
            or marker.get("progression_digest")
            != canonical_digest(marker, digest_field="progression_digest")
            or marker.get("provider_mutation_performed") is not False
            or marker.get("paid_execution_requested") is not False
            or marker.get("activation_executed_provider") is not False
            or request.get("expected_production_commit") != plan["expected_production_commit"]
            or request.get("authorization")
            != _read(Path(plan["phases"][phase]["authorization_path"]))
        ):
            return None
        queue = Path(
            config.get("activation_queue_root")
            or os.getenv(
                "BLUEPRINT_TASK_EVALUATION_LAUNCH_ACTIVATION_QUEUE_ROOT",
                str(Path(config["scene_root"]).parent / "task-evaluation-launch-activations"),
            )
        )
        pattern = activation_id + "-*.json"
        if any(
            list((queue / state).glob(pattern)) for state in ("pending", "processing", "prepared", "completed")
        ):
            return None
        blocked = list((queue / "blocked").glob(pattern))
        if len(blocked) != 1:
            return None
        envelope = _read(blocked[0])
        if (
            envelope.get("schema_version") != "task_evaluation_launch_activation_envelope.v1"
            or envelope.get("envelope_digest")
            != canonical_digest(envelope, digest_field="envelope_digest")
            or envelope.get("request") != request
            or envelope.get("request_digest") != canonical_digest(request)
            or envelope.get("provider_mutation_performed_inside_intake") is not False
            or envelope.get("paid_execution_requested") is not False
        ):
            return None
        result_path = queue / "results" / blocked[0].name
        result = _read(result_path)
        if (
            result.get("schema_version") != "task_evaluation_launch_activation_result.v1"
            or result.get("activation_id") != activation_id
            or result.get("status") != "blocked"
            or result.get("provider_mutation_performed") is not False
            or result.get("paid_execution_requested") is not False
            or not result.get("blockers")
            or result.get("result_digest") != canonical_digest(result, digest_field="result_digest")
        ):
            return None
        return [_file(p) for p in (marker_path, blocked[0], result_path)]
    except (OSError, ValueError, KeyError, TypeError, StopIteration):
        return None
