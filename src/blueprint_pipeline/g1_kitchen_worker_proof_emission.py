"""Build worker proof rows from signed leaf artifacts produced by one episode."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .g1_kitchen_leaf_evidence import load_attempt_identity, write_attested_leaf
from .g1_kitchen_attempt_closure import persistent_task_identity_rows


def legacy_worker_proof_rows(
    *,
    proof: Mapping[str, Any],
    task_completion_results: Sequence[Mapping[str, Any]],
    manipulation_success_judge: Mapping[str, Any],
    proven_task_completion_transition: Mapping[str, Any],
    consistency_results: Sequence[Mapping[str, Any]],
    forward_consistency_proven: bool,
    inverse_consistency_proven: bool,
) -> dict[str, Any]:
    """Compatibility rows for non-attempt-bound fixture and local smoke callers."""
    return {
        **persistent_task_identity_rows(task_completion_results),
        "controller_fk": {
            "status": "passed"
            if proof["fresh_actions_all_controller_fk_conditioned"]
            and proof["fresh_action_conditioning_differentiation_proven"]
            else "blocked",
            "evidence": {
                "fresh_actions_all_controller_fk_conditioned": proof[
                    "fresh_actions_all_controller_fk_conditioned"
                ],
                "fresh_action_conditioning_differentiation_proven": proof[
                    "fresh_action_conditioning_differentiation_proven"
                ],
                "action_conditioning_evidence": proof["action_conditioning_evidence"],
            },
        },
        "persistent_simulator_transition": {
            "status": "passed"
            if manipulation_success_judge.get("manipulation_success_proven") is True
            and proven_task_completion_transition
            else "blocked",
            "evidence": {
                "registered_task_completion_transition": proven_task_completion_transition,
                "task_completion_results": list(task_completion_results),
            },
        },
        "forward_consistency": {
            "status": "passed" if forward_consistency_proven else "blocked",
            "evidence": {"strict_external_scorer_results": list(consistency_results)},
        },
        "inverse_consistency": {
            "status": "passed" if inverse_consistency_proven else "blocked",
            "evidence": {"strict_external_scorer_results": list(consistency_results)},
        },
        "semantic_review": {
            "status": "blocked",
            "blockers": ["full_ordered_episode_semantic_review_required_post_collection"],
            "evidence": {
                "generated_video_success_label_is_not_full_episode_semantic_review": True
            },
        },
    }


def _leaf(
    *, root: Path, name: str, payload: Mapping[str, Any], identity: Mapping[str, Any], role: str
) -> dict[str, Any]:
    if root.name == "episode_001" and root.parent.name == "closed_loop_out":
        relative = f"closed_loop_out/episode_001/proof_leaves/{name}"
    else:
        relative = f"closed_loop_out/proof_leaves/{name}"
    return write_attested_leaf(
        payload=payload,
        path=root / "proof_leaves" / name,
        reference_path=relative,
        identity=identity,
        role=role,
    )


def _row(
    identity: Mapping[str, Any],
    leafs: Sequence[Mapping[str, Any]],
    *,
    verdict_passed: bool = True,
) -> dict[str, Any]:
    refs = [dict(item) for item in leafs]
    passed = bool(refs) and verdict_passed
    return {
        "status": "passed" if passed else "blocked",
        "identity_binding": dict(identity),
        "leaf_artifacts": refs,
        "blockers": (
            []
            if passed
            else [
                "signed_leaf_artifacts_missing"
                if not refs
                else "signed_leaf_verdict_not_passed"
            ]
        ),
    }


def emit_worker_proof_rows(
    *,
    output_dir: str | Path,
    attempt_input_manifest: str | Path,
    task_completion_results: Sequence[Mapping[str, Any]],
    controller_result_paths: Sequence[str | Path],
    consistency_results: Sequence[Mapping[str, Any]],
    manipulation_success_judge: Mapping[str, Any],
    action_sha256s: Sequence[str],
    planned_max_steps: int,
    termination_reason: str,
    task_completed: bool,
    scenario_count: int,
    geometry_results: Mapping[str, Mapping[str, Any]] | None = None,
    policy_actions: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Emit the exact leaf set consumed by ``validate_worker_proof_rows``.

    Any absent live producer yields a blocked row.  There is no conversion of a
    worker boolean into evidence and no attempt-identity repair.
    """
    root = Path(output_dir)
    identity = load_attempt_identity(attempt_input_manifest)
    startup_rows_path = root / "startup_gates" / "startup_proof_rows.json"
    if (
        not startup_rows_path.is_file()
        and root.name == "episode_001"
        and root.parent.name == "closed_loop_out"
    ):
        startup_rows_path = root.parent / "startup_gates" / "startup_proof_rows.json"
    startup_rows: dict[str, Any] = {}
    if startup_rows_path.is_file():
        loaded = json.loads(startup_rows_path.read_text(encoding="utf-8"))
        if isinstance(loaded, Mapping):
            startup_rows = dict(loaded)
    measurement_leafs: list[dict[str, Any]] = []
    normalized_measurements: list[dict[str, Any]] = []
    for index, raw in enumerate(task_completion_results):
        payload = dict(raw)
        payload["schema_version"] = "task_transition_measurement.v1"
        payload["runtime_source_step_index"] = payload.get("source_step_index")
        payload["source_step_index"] = index
        normalized_measurements.append(payload)
        measurement_leafs.append(
            _leaf(
                root=root,
                name=f"task_transition_{index:04d}.json",
                payload=payload,
                identity=identity,
                role="task_transition",
            )
        )

    completion_by_action = {
        str(row.get("source_action_sha256") or ""): dict(row)
        for row in task_completion_results
        if str(row.get("source_action_sha256") or "")
    }
    controller_leafs: list[dict[str, Any]] = []
    controller_payloads: list[dict[str, Any]] = []
    for index, raw_path in enumerate(controller_result_paths):
        payload = json.loads(Path(raw_path).read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("controller_fk_leaf_not_object")
        payload = dict(payload)
        completion = completion_by_action.get(
            str(payload.get("source_action_sha256") or ""), {}
        )
        if "official_controller_action_applied" not in payload:
            payload["official_controller_action_applied"] = (
                completion.get("official_controller_action_applied") is True
            )
        if completion:
            payload["persistent_simulator_state_applied"] = (
                completion.get("persistent_simulator_state_applied") is True
            )
            payload["simulator_session_id"] = completion.get("simulator_session_id")
            payload["stage_id"] = completion.get("stage_id")
        controller_payloads.append(payload)
        controller_leafs.append(
            _leaf(
                root=root,
                name=f"controller_fk_{index:04d}.json",
                payload=payload,
                identity=identity,
                role="controller",
            )
        )

    action_list = [str(item) for item in action_sha256s]
    policy_leaf = _leaf(
        root=root,
        name="policy_action_sequence.json",
        payload={
            "schema_version": "g1_kitchen_policy_action_sequence.v1",
            "source_action_sha256s": action_list,
            "actions": [dict(item) for item in (policy_actions or [])],
        },
        identity=identity,
        role="policy",
    )
    consistency_payload = {
        "schema_version": "strict_action_aware_consistency_contract.v1",
        "forward_consistency_proven": bool(consistency_results)
        and all(
            row.get("forward_dynamics_consistency_proven") is True
            for row in consistency_results
        ),
        "inverse_consistency_proven": bool(consistency_results)
        and all(
            row.get("inverse_dynamics_consistency_proven") is True
            for row in consistency_results
        ),
        "source_action_sha256s": action_list,
        "per_step_results": [dict(row) for row in consistency_results],
    }
    consistency_leaf = _leaf(
        root=root,
        name="strict_action_consistency.json",
        payload=consistency_payload,
        identity=identity,
        role="scorer",
    )
    judge = {**dict(manipulation_success_judge)}
    judge["schema_version"] = "isaac_manipulation_success_evaluator_results.v1"
    judge_leaf = _leaf(
        root=root,
        name="manipulation_success_judge.json",
        payload=judge,
        identity=identity,
        role="task_transition",
    )
    session_ids = {str(row.get("simulator_session_id") or "") for row in normalized_measurements}
    stage_ids = {str(row.get("stage_id") or "") for row in normalized_measurements}
    horizon = {
        "schema_version": "g1_kitchen_terminal_horizon.v1",
        "planned_max_steps": int(planned_max_steps),
        "executed_step_count": len(normalized_measurements),
        "terminal_step_index": len(normalized_measurements) - 1,
        "termination_reason": str(termination_reason),
        "task_completed": bool(task_completed),
        "scenario_count": int(scenario_count),
        "source_action_sha256s": action_list,
        "simulator_session_id": next(iter(session_ids)) if len(session_ids) == 1 else "",
        "stage_id": next(iter(stage_ids)) if len(stage_ids) == 1 else "",
    }
    horizon_leaf = _leaf(
        root=root,
        name="terminal_horizon.json",
        payload=horizon,
        identity=identity,
        role="task_transition",
    )
    geometry = dict(geometry_results or {})
    geometry_rows: dict[str, dict[str, Any]] = {}
    for row_id, schema in (
        ("stance", "g1_kitchen_live_stance_validation.v1"),
        ("collision", "g1_kitchen_live_collision_validation.v1"),
    ):
        payload = dict(geometry.get(row_id) or {})
        refs = []
        if payload.get("schema_version") == schema:
            refs.append(
                _leaf(
                    root=root,
                    name=f"live_{row_id}_validation.json",
                    payload=payload,
                    identity=identity,
                    role="geometry",
                )
            )
        geometry_passed = (
            payload.get("stance_valid") is True
            and payload.get("reach_valid") is True
            and payload.get("facing_valid") is True
            if row_id == "stance"
            else payload.get("collision_free") is True
            and payload.get("clearance_valid") is True
        )
        geometry_rows[row_id] = _row(
            identity, refs, verdict_passed=geometry_passed
        )
    transition_leafs = [*measurement_leafs, judge_leaf, horizon_leaf]
    return {
        **startup_rows,
        "scene_load": _row(identity, measurement_leafs),
        "target": _row(identity, measurement_leafs),
        **geometry_rows,
        "controller_fk": _row(
            identity,
            [policy_leaf, *controller_leafs],
            verdict_passed=bool(controller_leafs)
            and all(
                payload.get("official_controller_action_applied") is True
                for payload in controller_payloads
            ),
        ),
        "persistent_simulator_transition": _row(
            identity,
            transition_leafs,
            verdict_passed=judge.get("manipulation_success_proven") is True
            and judge.get("did_target_manipulation_succeed") is True
            and bool(task_completed),
        ),
        "forward_consistency": _row(
            identity,
            [consistency_leaf],
            verdict_passed=consistency_payload["forward_consistency_proven"] is True,
        ),
        "inverse_consistency": _row(
            identity,
            [consistency_leaf],
            verdict_passed=consistency_payload["inverse_consistency_proven"] is True,
        ),
    }


def emit_rows_from_closed_loop_state(state: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Narrow adapter keeping proof-emission mechanics out of the loop orchestrator."""
    results = [dict(item) for item in state["task_completion_results"]]
    geometry: dict[str, dict[str, Any]] = {}
    for result in results:
        for row_id in ("stance", "collision"):
            candidate = result.get(f"live_{row_id}_validation")
            if isinstance(candidate, Mapping) and candidate:
                geometry[row_id] = dict(candidate)
    output = Path(state["resolved_out"])
    def canonical_sha256(value: Any) -> str:
        import hashlib

        return hashlib.sha256(
            json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
        ).hexdigest()
    return emit_worker_proof_rows(
        output_dir=output,
        attempt_input_manifest=state["attempt_input_manifest"],
        task_completion_results=results,
        controller_result_paths=sorted(
            output.glob("controller_fk_skeleton/step_*/controller_fk_output.json")
        ),
        consistency_results=state["consistency_results"],
        manipulation_success_judge=state["manipulation_success_judge"],
        action_sha256s=[canonical_sha256(action) for action in state["action_history"]],
        planned_max_steps=state["bounded_steps"],
        termination_reason=state["episode_termination_reason"],
        task_completed=state["task_completed_early"],
        scenario_count=1,
        geometry_results=geometry,
        policy_actions=state["action_history"],
    )
