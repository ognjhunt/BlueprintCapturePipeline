"""All-cell controls then paired diagnostic, with failure evidence sealed."""
from __future__ import annotations

from copy import deepcopy

import json
import math
import mimetypes
from pathlib import Path
from typing import Any, Callable, Mapping

from .decision_evidence_contracts import canonical_digest
from .native_policy_canary_control_gate import _file, verify_files

RESULT_FILENAME = "native_task_arena_policy_canary_session_result.v1.json"


def paired_gate(child: Mapping[str, Any], *, root: Path, candidate_ids: list[str]) -> dict[str, Any]:
    blockers = []
    rows = child.get("episodes") or []
    if {row.get("candidate_id") for row in rows} != set(candidate_ids) or len(rows) != len(candidate_ids):
        blockers.append("strict_paired_candidate_records_incomplete")
    for row in rows:
        episode = row.get("episode") or {}
        score = episode.get("score") or {}
        media = episode.get("visual_evidence") or row.get("visual_evidence") or {}
        artifacts = row.get("evidence_artifacts") or {}
        if (row.get("status") != "completed"
                or (row.get("embodiment_parity_diagnostic") or {}).get("status") != "passed"
                or score.get("status") != "scored"
                or (score.get("measurements") or {}).get("destination_pose_readback_complete") is not True
                or media.get("status") != "complete"
                or any(not artifacts.get(role) for role in ("frame_manifest", "review_video", "action_sequence", "state_trace"))):
            blockers.append("strict_paired_episode_evidence_incomplete:"+str(row.get("candidate_id")))
    files = [_file(path, root) for path in sorted(root.rglob("*")) if path.is_file()]
    if not verify_files(files, root):
        blockers.append("strict_paired_retained_files_invalid")
    return {"schema_version": "policy_canary_strict_paired_gate.v1", "status": "blocked" if blockers else "passed",
            "blockers": blockers, "files": files, "deterministic_success_required": False}



def _retained_episodes(children: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    episodes = []
    for index, child in enumerate(children):
        for source in child.get("episodes", []):
            row = deepcopy(source)
            row["evidence_artifacts"] = {
                role: {**artifact, "relative_path": f"cell_runs/{index:02d}/" + artifact["relative_path"]}
                if isinstance(artifact, Mapping) and isinstance(artifact.get("relative_path"), str) else artifact
                for role, artifact in (row.get("evidence_artifacts") or {}).items()}
            episodes.append(row)
    return episodes

def execute_strict_matrix(*, runtime: Path, output_root: Path, inputs: Mapping[str, Any],
                          authority: Mapping[str, Any], spawn: Callable[..., int],
                          aggregate: Callable[..., dict[str, Any]],
                          seal: Callable[..., Any], construction_lineage_mode: str,
                          transfer_pair: Callable[..., Mapping[str, Any]], stage: str = "all") -> int:
    controls: list[dict[str, Any]] = []
    children: list[Mapping[str, Any]] = []
    blockers: list[str] = []
    diagnostic: dict[str, Any] | None = None
    delivery: Mapping[str, Any] | None = None
    result: dict[str, Any] = {}
    try:
        if stage not in {"all", "controls", "policies"}:
            raise RuntimeError("strict_matrix_stage_invalid")
        for index, cell in enumerate(inputs["cells"]):
            child_root = output_root / "control_runs" / f"{index:02d}"
            if stage != "policies":
                child_root.mkdir(parents=True, exist_ok=False)
                spawn(index=index, runtime_root=runtime, output_root=output_root,
                      child_root=child_root, controls_only=True)
            path = child_root / "policy_canary_cell_controls.v1.json"
            receipt = json.loads(path.read_text())
            for control in receipt.get("controls") or []:
                controls.append({"cell_id": cell["cell_id"], "seed": cell["seed"],
                                 "control_id": control["control_id"], "control_passed": control.get("control_passed"),
                                 "receipt": control, "evidence_root": f"control_runs/{index:02d}",
                                 "cell_receipt_artifact": _file(path, output_root),
                                 "evidence_files": receipt.get("files", [])})
            if (receipt.get("status") != "passed" or receipt.get("cell_id") != cell["cell_id"]
                    or receipt.get("seed") != cell["seed"]
                    or receipt.get("receipt_digest") != canonical_digest(receipt, digest_field="receipt_digest")
                    or not verify_files(receipt.get("files") or [], child_root)):
                raise RuntimeError(f"strict_controls_cell_failed:{index}")
        if len(controls) != 2*len(inputs["cells"]) or not all(row["control_passed"] is True for row in controls):
            raise RuntimeError("strict_controls_complete_matrix_required")
        if stage == "controls":
            prerequisite = {"schema_version": "policy_canary_controls_prerequisite.v1",
                            "status": "passed", "authority_digest": authority["authority_digest"],
                            "runtime_inputs_digest": inputs["runtime_inputs_digest"], "controls": controls,
                            "candidate_policy_queried": False, "receipt_digest": ""}
            prerequisite["receipt_digest"] = canonical_digest(prerequisite, digest_field="receipt_digest")
            (output_root / "policy_canary_controls_prerequisite.v1.json").write_text(json.dumps(prerequisite, indent=2))
            return 0
        for index, _cell in enumerate(inputs["cells"]):
            child_root = output_root / "cell_runs" / f"{index:02d}"
            child_root.mkdir(parents=True, exist_ok=False)
            spawn(index=index, runtime_root=runtime, output_root=output_root, child_root=child_root)
            child = json.loads((child_root / RESULT_FILENAME).read_text())
            children.append(child)
            if index == 0:
                diagnostic = paired_gate(child, root=child_root, candidate_ids=list(inputs["candidate_ids"]))
                if diagnostic["status"] != "passed":
                    raise RuntimeError("strict_paired_diagnostic_failed")
                delivery = transfer_pair(root=child_root, output_root=output_root, authority=authority,
                                         runtime_inputs_digest=inputs["runtime_inputs_digest"], runtime=runtime)
                if delivery.get("status") != "uploaded_and_readback_verified":
                    raise RuntimeError("strict_paired_delivery_unproven")
        result = aggregate(authority=authority, inputs=inputs, child_results=children,
                           output_root=output_root, construction_lineage_mode=construction_lineage_mode)
    except Exception as exc:
        blockers.append(str(exc))
        result = {
            "schema_version": "native_task_arena_policy_canary_session_result.v1", "status": "blocked",
            "run_kind": "internal_policy_canary", "claim_ceiling": "diagnostic_policy_execution",
            "candidate_ids": inputs["candidate_ids"], "task_success_contract": inputs["task_success_contract"],
            "task_success_contract_digest": inputs["task_success_contract_digest"],
            "episodes_per_policy": 10, "learned_policy_rollout_count": 2*len(inputs["cells"]),
            "episodes": _retained_episodes(children),
            "retry_cap": 0, "provider_allocations_observed": None, "warm_session_open_count": 1,
            "policy_loads": [], "session_failure_type": type(exc).__name__,
            "session_closeout": {"status": "runtime_closed_pending_provider_teardown", "runtime_closed": True,
                                 "provider_closeout_pending": True},
            "scene_promotion_performed": False, "official_ranking_performed": False,
            "provider_zero_required_after_return": True,
            "candidate_policy_queried": any(child.get("candidate_policy_queried") is True for child in children),
        }
    result.update(controls=controls, control_episode_count=len(controls),
                  controls_gate={"status": "passed" if len(controls) == 20 and all(row["control_passed"] is True for row in controls) else "blocked",
                                 "required_control_episode_count": 20, "candidate_policies_loaded_during_controls": False},
                  strict_paired_gate=diagnostic, paired_delivery=delivery, strict_gate_blockers=blockers)
    result["unexecuted_learned_episode_count"] = 2*len(inputs["cells"]) - len(result.get("episodes") or [])
    inventory = list(result.get("artifact_inventory") or [])
    for index in range(len(inputs["cells"])):
        control_root = output_root / "control_runs" / f"{index:02d}"
        if control_root.is_dir():
            inventory.extend({**_file(path, output_root), "role": "control_evidence",
                              "media_type": mimetypes.guess_type(path.name)[0] or "application/octet-stream"}
                             for path in sorted(control_root.rglob("*")) if path.is_file())
    result["artifact_inventory"] = inventory
    result["artifact_inventory_digest"] = canonical_digest({"value": inventory})
    seal(result_path=output_root / RESULT_FILENAME, result=result)
    return 0 if not blockers else 1


DROID_PARITY_MINIMUM_APPROACH_M = 0.05


def _episode_embodiment_parity_diagnostic(
    episode: Mapping[str, Any], *, observation_support_qualified: bool
) -> dict[str, Any]:
    """Measure harness parity without treating task success as the authority."""

    trace = episode.get("state_trace")
    rows = list(trace.get("task_state_samples") or []) if isinstance(trace, Mapping) else []
    distances: list[float] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        grasp = row.get("grasp_frame_position_world_m") or row.get(
            "gripper_body_midpoint_world_m"
        )
        task = row.get("task_object_pose_world") or row.get("can_pose_world")
        if (
            isinstance(grasp, list)
            and isinstance(task, list)
            and len(grasp) >= 3
            and len(task) >= 3
        ):
            distances.append(math.dist(grasp[:3], task[:3]))
    queries = [row for row in episode.get("queries") or [] if isinstance(row, Mapping)]
    joint_limit_clean = bool(
        queries and all(row.get("any_joint_limit_clamped") is False for row in queries)
    )
    motion = episode.get("motion_evidence")
    actions_reached_robot = bool(
        isinstance(motion, Mapping) and motion.get("actions_reached_robot") is True
    )
    arm_moved = bool(isinstance(motion, Mapping) and motion.get("arm_moved") is True)
    initial = distances[0] if distances else None
    minimum = min(distances) if distances else None
    final = distances[-1] if distances else None
    approach = initial - minimum if initial is not None and minimum is not None else None
    blockers = []
    if observation_support_qualified is not True:
        blockers.append("droid_observation_outside_checkpoint_support")
    if not actions_reached_robot:
        blockers.append("droid_actions_did_not_reach_robot")
    if not arm_moved:
        blockers.append("droid_arm_did_not_move")
    if not joint_limit_clean:
        blockers.append("droid_action_joint_limit_or_query_evidence_invalid")
    if approach is None:
        blockers.append("droid_gripper_task_distance_unavailable")
    elif approach < DROID_PARITY_MINIMUM_APPROACH_M:
        blockers.append("droid_gripper_did_not_approach_task")
    value: dict[str, Any] = {
        "schema_version": "droid_policy_canary_embodiment_parity.v1",
        "status": "passed" if not blockers else "blocked",
        "observation_support_qualified": observation_support_qualified,
        "actions_reached_robot": actions_reached_robot,
        "arm_moved": arm_moved,
        "joint_limit_clean": joint_limit_clean,
        "initial_gripper_to_task_distance_m": initial,
        "minimum_gripper_to_task_distance_m": minimum,
        "final_gripper_to_task_distance_m": final,
        "approach_distance_m": approach,
        "minimum_required_approach_m": DROID_PARITY_MINIMUM_APPROACH_M,
        "blockers": blockers,
        "diagnostic_only": True,
        "task_success_claimed": False,
        "receipt_digest": "",
    }
    value["receipt_digest"] = canonical_digest(value, digest_field="receipt_digest")
    return value

