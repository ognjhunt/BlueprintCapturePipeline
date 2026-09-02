"""Safe Website projection for one internal policy-canary result delivery."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Mapping
from typing import Any

from .decision_evidence_contracts import (
    canonical_digest,
    cross_runtime_canonical_digest,
)
from .native_task_arena_policy_canary_session import (
    CANDIDATE_IDS,
    CLAIM_CEILING,
    RUN_KIND,
)
from .task_evaluation_policy_canary_result import validate_policy_canary_result


ErrorFactory = Callable[[str], Exception]


def _is_digest(value: Any) -> bool:
    return bool(re.fullmatch(r"sha256:[0-9a-f]{64}", str(value or "")))


def build_policy_canary_result_projection(
    *,
    setup: Mapping[str, Any],
    result: Mapping[str, Any],
    delivery: Mapping[str, Any],
    error_factory: ErrorFactory = ValueError,
) -> dict[str, Any]:
    """Project evidence-bound result fields without exposing provider internals."""

    episodes = list(result.get("episodes") or [])

    def compact_artifact(record: Mapping[str, Any]) -> dict[str, Any]:
        return {key: record[key] for key in ("artifact_id", "digest", "size_bytes")}

    report = {
        "machine_readable_report": compact_artifact(
            delivery["report"]["machine_readable_report"]
        ),
        "evidence_manifest": compact_artifact(delivery["report"]["evidence_manifest"]),
    }
    public_artifacts = delivery.get("artifacts") or []

    def bound_artifact(record: Any) -> dict[str, Any] | None:
        if not isinstance(record, Mapping):
            return None
        relative_path = record.get("relative_path")
        if not isinstance(relative_path, str) or not relative_path:
            return None
        matches = [
            artifact
            for artifact in public_artifacts
            if artifact.get("role") == record.get("role")
            and artifact.get("digest") == record.get("sha256")
            and artifact.get("size_bytes") == record.get("size_bytes")
            and artifact.get("relative_path") == relative_path
        ]
        if len(matches) != 1:
            return None
        return {
            key: matches[0][key] for key in ("artifact_id", "digest", "size_bytes")
        }

    projected_episodes: list[dict[str, Any]] = []
    for row in episodes:
        candidate = str(row.get("candidate_id") or "")
        cell_id = str(row.get("cell_id") or "")
        episode_id = f"{result.get('run_id') or setup['scene_id']}--{cell_id}--{candidate}"
        if len(episode_id) > 192:
            episode_id = (
                episode_id[:150]
                + "-"
                + hashlib.sha256(episode_id.encode()).hexdigest()[:32]
            )
        source_artifacts = row.get("evidence_artifacts")
        source_artifacts = (
            dict(source_artifacts) if isinstance(source_artifacts, Mapping) else {}
        )
        evidence_roles = {
            "reset_state": "reset_state",
            "frame_manifest": "frame_manifest",
            "review_video": "review_video",
            "policy_query_receipt": "policy_query_receipt",
            "action_sequence": "action_sequence",
            "action_delivery_readback": "action_delivery_readback",
            "state_trace": "state_trace",
            "contact_force_trace": "contact_force_trace",
            "task_object_trajectory": "task_object_trajectory",
            "score_receipt": "score_receipt",
        }
        bound = {
            target: bound_artifact(source_artifacts.get(source))
            for target, source in evidence_roles.items()
        }
        evidence_gaps = sorted(
            target for target, artifact in bound.items() if artifact is None
        )
        if row.get("status") == "completed" and evidence_gaps:
            raise error_factory(
                "policy_canary_completed_episode_evidence_missing:"
                + episode_id
                + ":"
                + ",".join(evidence_gaps)
            )
        checkpoint_digest = row.get("checkpoint_digest")
        runtime_identity_digest = row.get("runtime_identity_digest")
        reset_state_digest = (
            source_artifacts.get("reset_state", {}).get("sha256")
            if isinstance(source_artifacts.get("reset_state"), Mapping)
            else row.get("reset_state_digest")
        )
        if (
            not _is_digest(reset_state_digest)
            and row.get("status") != "completed"
            and isinstance(row.get("resolved_scenario"), Mapping)
            and isinstance(row.get("seed"), int)
            and not isinstance(row.get("seed"), bool)
        ):
            # Older blocked provider results predate the explicit reset digest.
            # Their immutable cell and seed still define the exact attempted
            # reset. Completed episodes may never use this fallback.
            reset_state_digest = canonical_digest(
                {
                    "resolved_scenario": row["resolved_scenario"],
                    "seed": row["seed"],
                    "execution_performed": False,
                }
            )
        if not all(
            _is_digest(value)
            for value in (
                checkpoint_digest,
                runtime_identity_digest,
                reset_state_digest,
            )
        ):
            raise error_factory(
                "policy_canary_episode_identity_evidence_missing:" + episode_id
            )
        evidence = {
            "checkpoint_digest": checkpoint_digest,
            "runtime_identity_digest": runtime_identity_digest,
            "reset_state_digest": reset_state_digest,
            **bound,
            "evidence_gaps": evidence_gaps,
        }
        gap = ((row.get("visual_evidence") or {}).get("media_gap") or {}).get(
            "reason"
        )
        if gap:
            evidence["typed_media_gap"] = str(gap)
        projected_episodes.append(
            {
                "episode_id": episode_id,
                "candidate_id": candidate,
                "cell_id": cell_id,
                "seed": row.get("seed"),
                "terminal_state": (
                    "completed" if row.get("status") == "completed" else "blocked"
                ),
                "candidate_policy_queried": row.get("candidate_policy_queried") is True,
                "actions_reached_robot": row.get("actions_reached_robot") is True,
                "arm_moved": row.get("arm_moved") is True,
                "policy_outcome_interpretable": (
                    row.get("policy_outcome_interpretable") is True
                ),
                "failure_taxonomy": row.get("typed_harness_failure"),
                "evidence": evidence,
            }
        )
    candidate_results = []
    delivered_candidate_results = {
        row["candidate_id"]: row for row in delivery.get("candidate_results") or []
    }
    for candidate in CANDIDATE_IDS:
        rows = [row for row in projected_episodes if row["candidate_id"] == candidate]
        failures: dict[str, int] = {}
        for row in rows:
            if row["terminal_state"] != "completed":
                name = str(row.get("failure_taxonomy") or "unclassified")
                failures[name] = failures.get(name, 0) + 1
        delivered_metrics = dict(delivered_candidate_results.get(candidate) or {})
        candidate_results.append(
            {
                "candidate_id": candidate,
                "episodes_completed": sum(
                    row["terminal_state"] == "completed" for row in rows
                ),
                "interpretable_episode_count": sum(
                    row["policy_outcome_interpretable"] for row in rows
                ),
                "actions_delivered_episode_count": sum(
                    row["actions_reached_robot"] for row in rows
                ),
                "metrics": {
                    key: value
                    for key, value in delivered_metrics.items()
                    if key != "candidate_id"
                },
                "failure_counts": failures,
            }
        )
    cell_sets = [
        {
            row["cell_id"]
            for row in projected_episodes
            if row["candidate_id"] == candidate
        }
        for candidate in CANDIDATE_IDS
    ]
    result_status = str(delivery["result_status"])
    delivery_reproducibility = dict(delivery.get("reproducibility") or {})
    reproducibility_artifacts = {
        "evidence_manifest": delivery_reproducibility.get("evidence_manifest")
        or (delivery.get("report") or {}).get("evidence_manifest"),
        "billing_receipt": delivery_reproducibility.get("billing_receipt")
        or (delivery.get("closure") or {}).get("billing"),
        "teardown_receipt": delivery_reproducibility.get("teardown_receipt")
        or (delivery.get("closure") or {}).get("teardown"),
        "provider_zero_receipt": delivery_reproducibility.get(
            "provider_zero_receipt"
        )
        or (delivery.get("closure") or {}).get("provider_zero"),
    }
    value: dict[str, Any] = {
        "schema_version": "task_evaluation_policy_canary_result_projection.v1",
        "run_id": delivery["run_id"],
        "request_digest": setup["request_digest"],
        "configuration_digest": result["configuration_digest"],
        "result_delivery_digest": delivery["delivery_digest"],
        "run_kind": RUN_KIND,
        "claim_ceiling": CLAIM_CEILING,
        "scene_controls_status": "configured_controls_pending",
        "result_status": result_status,
        "warning": "Controls pending — results are unqualified.",
        "matrix_digest": delivery.get("matrix_digest"),
        "counts": {
            "policy_count": 2,
            "episodes_per_policy": 10,
            "learned_policy_rollout_count": 20,
            "completed_learned_policy_rollout_count": sum(
                row["terminal_state"] == "completed" for row in projected_episodes
            ),
            "diagnostic_control_rollout_count": 20,
            "completed_diagnostic_control_rollout_count": 0,
        },
        "candidate_ids": list(CANDIDATE_IDS),
        "candidate_results": candidate_results,
        "episodes": projected_episodes,
        "comparison": {
            "matched_cell_count": len(cell_sets[0] & cell_sets[1]),
            "winner_declared": False,
            "official_ranking_contribution": False,
        },
        "report": {
            "result_digest": result["result_digest"],
            "permanent_result_path": (
                f"/internal/task-evaluation-runs/{delivery['run_id']}"
            ),
            **report,
        },
        "closure": {
            "billing": compact_artifact(delivery["closure"]["billing"]),
            "teardown": compact_artifact(delivery["closure"]["teardown"]),
            "provider_zero": {
                **compact_artifact(delivery["closure"]["provider_zero"]),
                "provider_zero_verified": True,
            },
        },
        "notification_delivery": {
            "terminal_state": (
                "completed" if result_status == "completed_unqualified" else result_status
            ),
            "status": "pending",
            "attempts": 0,
            "provider": "website_terminal_handler",
            "message_id": None,
            "delivered_at": None,
            "run_result_digest": result["result_digest"],
        },
        "blockers": list(result.get("blockers") or []),
        "projection_digest": "",
    }
    scene_revision_digest = setup.get("scene_revision_digest") or result.get(
        "scene_revision_digest"
    )
    if _is_digest(scene_revision_digest) and all(
        isinstance(record, Mapping) for record in reproducibility_artifacts.values()
    ):
        value["reproducibility"] = {
            "scene_revision_digest": scene_revision_digest,
            "runtime_container_digest": delivery_reproducibility.get(
                "runtime_container_digest"
            ),
            "scoring_version": delivery_reproducibility.get("scoring_version"),
            "observation_schema_id": delivery_reproducibility.get(
                "observation_schema_id"
            ),
            "action_schema_id": delivery_reproducibility.get("action_schema_id"),
            **{
                name: compact_artifact(record)
                for name, record in reproducibility_artifacts.items()
            },
            "official_total_usd": delivery_reproducibility.get(
                "official_total_usd"
            ),
            "started_at_iso": delivery_reproducibility.get("started_at_iso"),
            "completed_at_iso": delivery_reproducibility.get("completed_at_iso"),
            "duration_seconds": delivery_reproducibility.get("duration_seconds"),
            "provider": delivery_reproducibility.get("provider"),
            "provider_instance_ids": delivery_reproducibility.get(
                "provider_instance_ids"
            ),
        }
    value["projection_digest"] = cross_runtime_canonical_digest(
        value, digest_field="projection_digest"
    )
    return validate_policy_canary_result(value)


__all__ = ["build_policy_canary_result_projection"]
