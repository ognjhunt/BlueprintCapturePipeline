"""Seal terminal ADP Task Evaluation abstention before any episode exists."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest, canonical_json

SCHEMA_VERSION = "adp_task_evaluation_run_abstention.v1"
CONSTRUCTION_SCHEMA_VERSION = "articulated_public_scene_construction_run.v2"
FREEZE_SCHEMA_VERSION = "second_scene_scene_task_freeze.v1"


class TaskEvaluationAbstentionError(ValueError):
    """Fail-closed terminal abstention validation error."""


def _clone(value: Mapping[str, Any], *, code: str) -> dict[str, Any]:
    try:
        result = json.loads(json.dumps(dict(value), allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise TaskEvaluationAbstentionError(code) from exc
    return result


def materialize_task_evaluation_abstention(
    *,
    construction_run: Mapping[str, Any],
    scene_task_freeze: Mapping[str, Any],
    output_path: str | Path | None = None,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Turn observed terminal construction receipts into completion path B."""

    run = _clone(construction_run, code="task_evaluation_construction_run_invalid")
    freeze = _clone(scene_task_freeze, code="task_evaluation_freeze_invalid")
    errors: list[str] = []
    if (
        run.get("schema_version") != CONSTRUCTION_SCHEMA_VERSION
        or run.get("run_digest") != canonical_digest(run, digest_field="run_digest")
        or run.get("status") != "typed_abstention_before_simready_build"
    ):
        errors.append("task_evaluation_construction_run_invalid")
    if (
        freeze.get("schema_version") != FREEZE_SCHEMA_VERSION
        or freeze.get("freeze_digest")
        != canonical_digest(freeze, digest_field="freeze_digest")
    ):
        errors.append("task_evaluation_freeze_invalid")
    scene = run.get("scene") or {}
    freeze_scene = freeze.get("scene") or {}
    if (
        scene.get("publisher_scene_id") != freeze_scene.get("publisher_scene_id")
        or scene.get("target_instance_id") != freeze_scene.get("target_instance_id")
        or scene.get("freeze_digest") != freeze.get("freeze_digest")
        or run.get("frozen_candidates") != ["pi05_droid", "groot_n17_droid"]
    ):
        errors.append("task_evaluation_construction_freeze_join_invalid")
    stage_receipts = run.get("stage_receipts") or {}
    stage_status = run.get("stage_status") or {}
    blockers = run.get("blockers")
    if (
        not isinstance(blockers, list)
        or not blockers
        or run.get("smallest_blocker") != blockers[0]
    ):
        errors.append("task_evaluation_terminal_blocker_missing")
    if (
        not stage_receipts.get("aura_execution")
        or not stage_receipts.get("joint_agent_execution")
        or stage_status.get("released_code_inpainting_executed") is not True
        or stage_status.get("joint_agent_execution_attempted") is not True
    ):
        errors.append("task_evaluation_construction_attempt_receipts_incomplete")
    if (
        stage_status.get("simready_replacement_materialized") is not False
        or stage_status.get("native_simulator_qualified") is not False
        or stage_status.get("controls_executed") is not False
        or stage_status.get("learned_candidates_executed") is not False
    ):
        errors.append("task_evaluation_abstention_after_episode_state_invalid")
    if not any(
        str(blocker).startswith("joint_agent_topology_execution_abstained:")
        for blocker in (blockers or [])
    ):
        errors.append("task_evaluation_observed_execution_abstention_missing")
    if errors:
        raise TaskEvaluationAbstentionError(";".join(sorted(set(errors))))

    receipt: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "program_id": "arm-decision-proof-v1",
        "status": "typed_evidence_backed_abstention",
        "scene_id": scene["publisher_scene_id"],
        "target_instance_id": scene["target_instance_id"],
        "task_id": (freeze.get("task_spec") or {}).get("task_id"),
        "task_kind": scene.get("task_kind"),
        "freeze_digest": freeze["freeze_digest"],
        "candidate_ids": ["pi05_droid", "groot_n17_droid"],
        "smallest_missing_capability": run["smallest_blocker"],
        "all_terminal_construction_blockers": list(blockers),
        "construction_run_digest": run["run_digest"],
        "stage_receipts": dict(stage_receipts),
        "controls_executed": False,
        "learned_candidate_episodes_executed": False,
        "episode_media_exists": False,
        "comparison_exists": False,
        "automatic_paid_retry_executed": False,
        "claim_ceiling": (
            "public_dataset_construction_rehearsal_only; no partner capture, "
            "real_site_fidelity, deployment readiness, physical performance, "
            "or learned_policy_comparison"
        ),
        "next_action": run.get("next_action"),
        "receipt_digest": "",
    }
    receipt["receipt_digest"] = canonical_digest(
        receipt, digest_field="receipt_digest"
    )
    if output_path is not None:
        if repo_root is None:
            raise TaskEvaluationAbstentionError("task_evaluation_repo_root_missing")
        repo = Path(repo_root).expanduser().resolve()
        output = Path(output_path).expanduser().resolve()
        if not output.is_relative_to(repo) or output.is_symlink():
            raise TaskEvaluationAbstentionError(
                "task_evaluation_abstention_output_outside_repo"
            )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(canonical_json(receipt) + "\n", encoding="utf-8")
    return receipt


__all__ = [
    "SCHEMA_VERSION",
    "TaskEvaluationAbstentionError",
    "materialize_task_evaluation_abstention",
]
