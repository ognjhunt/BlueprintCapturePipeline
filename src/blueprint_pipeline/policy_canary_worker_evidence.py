"""Retain native policy failures, lossless media, and indexed telemetry."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import time
from typing import Any, Mapping

from .decision_evidence_contracts import canonical_digest
from .policy_canary_media_integrity import bound_media_artifact as _bound_media_artifact

# Also shipped in standalone policy bundles, which have no paired-session module.
PROVIDER_RESULT_FILENAME = "native_task_arena_policy_canary_session_result.v1.json"


def _digest(value):
    return canonical_digest({'value': value})


def _sha256(path):
    return 'sha256:' + hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_episode_failure_gap(
    *,
    output_root: Path,
    run_id: str,
    context: Mapping[str, Any],
    failure: Exception,
    progress: Mapping[str, Any] | None = None,
) -> Path:
    """Retain the strongest episode evidence reached before a typed failure."""

    raw_message = str(failure).strip().replace("\n", " ").replace("\r", " ")
    safe_message = re.sub(r"(?<![A-Za-z0-9])/(?:[^\s/:]+/)*[^\s:]+", "<path>", raw_message)
    safe_message = safe_message[:512]
    episode_id = (
        f"{run_id}--{context.get('cell_id')}--{context.get('candidate_id')}"
    )
    progress = progress if isinstance(progress, Mapping) else {}
    first_observation_retained = progress.get("first_observation_retained") is True
    candidate_policy_queried = progress.get("candidate_policy_queried") is True
    query_attempted = candidate_policy_queried or progress.get("candidate_policy_query_attempted") is True
    query_attempt = {
        "candidate_policy_query_attempted": query_attempted,
        "policy_response_status": "received" if candidate_policy_queried else "unproven" if query_attempted else "not_attempted",
    }
    candidate_action_returned = progress.get("candidate_action_returned") is True
    action_applied = progress.get("candidate_action_applied") is True
    violations = [
        str(item)
        for item in getattr(failure, "errors", ())
        if str(item).startswith("candidate_action_joint_position_bounds_invalid")
    ]
    action_rejected = bool(
        candidate_policy_queried
        and candidate_action_returned
        and not action_applied
        and progress.get("phase") == "policy_action_bounds_refused"
        and violations
    )
    failure_stage = (
        "action_delivery_rejected"
        if action_rejected
        else "after_first_observation"
        if first_observation_retained
        else "before_first_observation"
    )
    suffix = "failure_evidence" if first_observation_retained else "failure_gap"
    path = output_root / "episodes" / f"{episode_id}.{suffix}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    visual_evidence: dict[str, Any]
    media_artifacts: list[dict[str, Any]] = []
    if first_observation_retained:
        finalizer = progress.get("_failure_media_finalizer")
        if callable(finalizer):
            try:
                visual, artifacts = finalizer(
                    failure_reason=f"{type(failure).__name__}:{safe_message}"
                )
                visual_evidence = dict(visual)
                media_artifacts = [
                    dict(item) for item in artifacts if isinstance(item, Mapping)
                ]
            except Exception as exc:  # noqa: BLE001 - preserve the primary failure
                visual_evidence = {
                    "status": "incomplete_after_first_observation",
                    "media_gap": {
                        "type": "after_first_observation_media_seal_failed",
                        "reason": type(exc).__name__,
                    },
                }
        else:
            visual_evidence = {
                "status": "incomplete_after_first_observation",
                "media_gap": {
                    "type": "after_first_observation_media_finalizer_missing",
                    "reason": "policy_canary_episode_runner_failed",
                },
            }
    else:
        visual_evidence = {
            "status": "unavailable_before_first_observation",
            "media_gap": {
                "type": "before_first_observation",
                "reason": "policy_canary_episode_runner_failed",
            },
        }
    raw_queries = [
        dict(item)
        for item in progress.get("candidate_policy_action_queries") or []
        if isinstance(item, Mapping)
    ]
    commanded_actions = [
        dict(item)
        for item in progress.get("commanded_actions") or []
        if isinstance(item, Mapping)
    ]
    action_rejection = None
    if action_rejected:
        action_rejection = {
            "schema_version": "policy_canary_action_delivery_rejection.v1",
            "status": "rejected_before_robot",
            "reason": "hard_joint_limit_violation",
            "violations": violations,
            "clamping_performed": False,
            "delivery_attempted": False,
            "actions_reached_robot": False,
            "rejection_digest": "",
        }
        action_rejection["rejection_digest"] = canonical_digest(
            action_rejection, digest_field="rejection_digest"
        )
    if progress.get("media_integrity_failure"):
        visual_evidence = {
            **visual_evidence,
            "status": "incomplete_after_first_observation",
            "media_gap": {
                "type": "after_first_observation_media_integrity_failed",
                "reason": progress["media_integrity_failure"],
            },
        }
    evidence_artifacts: dict[str, Any] = {}
    media_root = output_root / "episodes"
    if media_artifacts:
        evidence_artifacts["frame_manifest"] = _bound_media_artifact(
            output_root,
            media_root=media_root,
            artifacts=media_artifacts,
            role="lossless_frame_manifest",
            role_match=lambda name: "frame_manifest" in name,
        )
        evidence_artifacts["review_video"] = _bound_media_artifact(
            output_root,
            media_root=media_root,
            artifacts=media_artifacts,
            role="review_video",
            role_match=lambda name: "video" in name,
        )
    if query_attempted:
        evidence_artifacts["policy_query_receipt"] = _write_episode_json_artifact(
            output_root,
            episode_id=episode_id,
            role="policy_query_receipt",
            value={
                **query_attempt,
                "candidate_policy_queried": candidate_policy_queried,
                "candidate_action_returned": candidate_action_returned,
                "policy_request_artifacts": progress.get("policy_request_artifacts") or [],
                "policy_queries": raw_queries,
            },
        )
    if candidate_action_returned:
        evidence_artifacts["action_sequence"] = _write_episode_json_artifact(
            output_root,
            episode_id=episode_id,
            role="action_sequence",
            value=raw_queries,
        )
    if action_rejection is not None:
        evidence_artifacts["action_delivery_readback"] = _write_episode_json_artifact(
            output_root,
            episode_id=episode_id,
            role="action_delivery_readback",
            value=action_rejection,
        )
    value = {
        "schema_version": "policy_canary_episode_failure_evidence.v2",
        "status": "blocked",
        "run_kind": "internal_policy_canary",
        "claim_ceiling": "diagnostic_policy_execution",
        "candidate_id": context.get("candidate_id"),
        "cell_id": context.get("cell_id"),
        "seed": context.get("seed"),
        "episode_failure_stage": failure_stage,
        "scientific_reset": progress.get("scientific_reset"),
        "prestart_readiness": progress.get("prestart_readiness"),
        "policy_inference_evidence": progress.get("policy_inference_evidence"),
        "first_observation_retained": first_observation_retained,
        "reset_state_digest": canonical_digest(
            {
                "resolved_scenario": context.get("resolved_scenario"),
                "seed": context.get("seed"),
                "execution_performed": first_observation_retained,
            }
        ),
        "candidate_policy_queried": candidate_policy_queried,
        **query_attempt,
        "candidate_action_returned": candidate_action_returned,
        "candidate_action_shape_validated": (
            progress.get("candidate_action_shape_validated") is True
        ),
        "candidate_action_finite_validated": (
            progress.get("candidate_action_finite_validated") is True
        ),
        "candidate_action_bounds_validated": (
            progress.get("candidate_action_bounds_validated") is True
        ),
        "actions_reached_robot": action_applied,
        "arm_moved": False,
        "policy_outcome_interpretable": False,
        "failure_type": type(failure).__name__,
        "typed_harness_failure": type(failure).__name__,
        "failure_message": safe_message or None,
        "failure_message_digest": _digest(raw_message),
        "candidate_policy_action_queries": raw_queries,
        "commanded_actions": commanded_actions,
        "action_delivery_rejection": action_rejection,
        "visual_evidence": visual_evidence,
        "lossless_frame_manifest_digest": (
            _digest(visual_evidence) if first_observation_retained else None
        ),
        "review_video_digest": (
            _digest(media_artifacts) if media_artifacts else None
        ),
        "returned_action_sequence_digest": (
            _digest(raw_queries) if candidate_action_returned else None
        ),
        "action_delivery_readback_digest": (
            action_rejection["rejection_digest"]
            if action_rejection is not None
            else None
        ),
        "evidence_artifacts": evidence_artifacts,
        "episode": {
            **query_attempt,
            "episode_id": episode_id,
            "policy_request_artifacts": progress.get("policy_request_artifacts") or [],
            "scientific_reset": progress.get("scientific_reset"),
            "candidate_policy_action_queries": raw_queries,
            "commanded_actions": commanded_actions,
            "visual_evidence": visual_evidence,
            "media_artifacts": media_artifacts,
            "motion_evidence": {
                "actions_reached_robot": action_applied,
                "arm_moved": False,
                "policy_outcome_interpretable": False,
                "action_delivery_rejection": action_rejection,
            },
            "score": {
                "status": "not_scored",
                "blockers": ["policy_outcome_uninterpretable"],
            },
        },
        "gap_digest": "",
    }
    value["gap_digest"] = canonical_digest(value, digest_field="gap_digest")
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _write_episode_json_artifact(
    output_root: Path, *, episode_id: str, role: str, value: Any
) -> dict[str, Any]:
    path = output_root / "episodes" / f"{episode_id}.{role}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "role": role,
        "relative_path": path.relative_to(output_root).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _write_indexed_telemetry(
    output_root: Path, episodes: list[Mapping[str, Any]]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = [
        {
            "run_kind": episode.get("run_kind"),
            "candidate_id": episode.get("candidate_id"),
            "cell_id": episode.get("cell_id"),
            "seed": episode.get("seed"),
            "telemetry": episode.get("telemetry"),
        }
        for episode in episodes
    ]
    telemetry_path = output_root / "policy_canary_telemetry.jsonl"
    telemetry_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    channels = {
        "observations": sum(bool((row.get("telemetry") or {}).get("channels")) for row in rows),
        "episode_envelopes": len(rows),
    }
    schema = {
        "schema_version": "policy_canary_telemetry_schema.v1",
        "timebase": "unix_ns",
        "channels": {
            "episode_envelopes": "policy_canary_episode_telemetry.v1",
            "observations": "native_policy_observation_manifest_reference.v1",
        },
    }
    schema_path = output_root / "policy_canary_telemetry_schema.json"
    schema_path.write_text(json.dumps(schema, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    primary_path = telemetry_path
    primary_format = "typed_jsonl"
    mcap_gap: str | None = None
    try:
        from mcap.writer import Writer

        mcap_path = output_root / "policy_canary_telemetry.mcap"
        with mcap_path.open("wb") as stream:
            writer = Writer(stream)
            writer.start(profile="blueprint-policy-canary", library="blueprint_pipeline")
            schema_id = writer.register_schema(
                name="policy_canary_episode_telemetry.v1",
                encoding="jsonschema",
                data=json.dumps(schema, sort_keys=True).encode("utf-8"),
            )
            channel_id = writer.register_channel(
                topic="/blueprint/policy_canary/episode",
                message_encoding="json",
                schema_id=schema_id,
            )
            for row in rows:
                telemetry = row.get("telemetry") or {}
                timestamp = int(telemetry.get("completed_at_unix_ns") or time.time_ns())
                writer.add_message(
                    channel_id=channel_id,
                    log_time=timestamp,
                    publish_time=timestamp,
                    data=json.dumps(row, sort_keys=True).encode("utf-8"),
                )
            writer.finish()
        primary_path = mcap_path
        primary_format = "mcap"
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
        mcap_gap = f"mcap_unavailable:{type(exc).__name__}"
    index = {
        "schema_version": "policy_canary_telemetry_index.v1",
        "format": primary_format,
        "artifact": {
            "path": primary_path.name,
            "size_bytes": primary_path.stat().st_size,
            "sha256": _sha256(primary_path),
        },
        "schema": {
            "path": schema_path.name,
            "size_bytes": schema_path.stat().st_size,
            "sha256": _sha256(schema_path),
        },
        "channel_message_counts": channels,
        "message_count": len(rows),
        "attachments": [],
        "calibration_references": [
            (row.get("telemetry") or {}).get("camera_calibration") for row in rows
        ],
        "mcap_gap": mcap_gap,
        "evidence_gaps": sorted(
            {
                gap
                for row in rows
                for gap in (row.get("telemetry") or {}).get("evidence_gaps", [])
            }
        ),
        "index_digest": "",
    }
    index["index_digest"] = canonical_digest(index, digest_field="index_digest")
    index_path = output_root / "policy_canary_telemetry_index.json"
    index_path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    import mimetypes

    artifacts = []
    seen: set[str] = set()
    for path in sorted(output_root.rglob("*")):
        # The parent process owns this file and keeps the child's stdout stream
        # open until after the child has sealed its result.  Including it here
        # races the final interpreter-shutdown writes and produces an inventory
        # entry whose size/digest no longer match by delivery time.
        if (
            not path.is_file()
            or path.name == PROVIDER_RESULT_FILENAME
            or path.name == "worker_console.log"
        ):
            continue
        relative = path.relative_to(output_root).as_posix()
        if relative in seen:
            continue
        seen.add(relative)
        lowered = relative.lower()
        typed_evidence_role = next(
            (
                evidence_role
                for evidence_role in (
                    "reset_state",
                    "policy_query_receipt",
                    "action_sequence",
                    "action_delivery_readback",
                    "state_trace",
                    "contact_force_trace",
                    "task_object_trajectory",
                    "score_receipt",
                    "episode_receipt",
                )
                if f".{evidence_role}.json" in lowered
            ),
            None,
        )
        role = (
            "indexed_episode_telemetry"
            if path in {primary_path, telemetry_path}
            else "telemetry_schema"
            if path == schema_path
            else "telemetry_index"
            if path == index_path
            else "review_video"
            if path.suffix.lower() in {".mp4", ".mov", ".webm"}
            else "lossless_frame_manifest"
            if "frame" in lowered and "manifest" in lowered
            else "exact_policy_request"
            if "/policy-requests/" in lowered and path.suffix.lower() == ".json"
            else typed_evidence_role
            if typed_evidence_role is not None
            else "episode_evidence"
            if "episode" in lowered
            else "runtime_supporting_evidence"
        )
        artifacts.append(
            {
                "role": role,
                "media_type": mimetypes.guess_type(path.name)[0]
                or "application/octet-stream",
                "relative_path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return index, artifacts
