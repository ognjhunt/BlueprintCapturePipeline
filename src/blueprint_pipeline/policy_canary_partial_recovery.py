"""Recover sealed partial cell evidence without allocating or grading policies.

The dispatcher supplies its existing readers, writers and error class so its
public test/operational seams continue to apply to this recovery boundary.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .decision_evidence_contracts import canonical_digest


def recover_partial_policy_canary_result(
    *,
    native_path: Path,
    fallback: Mapping[str, Any],
    runtime_inputs: Mapping[str, Any],
    specs: Mapping[str, Mapping[str, Any]],
    read_record: Callable[..., dict[str, Any]],
    sha256: Callable[[Path], str],
    write_record: Callable[[Path, Mapping[str, Any]], Any],
    error_factory: Callable[[str], Exception],
    candidate_ids: Sequence[str],
    learned_rollout_count: int,
    run_kind: str,
    claim_ceiling: str,
) -> tuple[dict[str, Any], Path] | None:
    """Preserve sealed child cells when a later isolated cell times out."""

    if (
        fallback.get("status") == "runtime_completed_unqualified_pending_closeout"
        and isinstance(fallback.get("episodes"), list)
        and len(fallback["episodes"]) == learned_rollout_count
        and isinstance(fallback.get("artifact_inventory"), list)
    ):
        # The provider already produced the authoritative full-session
        # aggregation. Rebuilding it from child receipts would incorrectly turn
        # a complete run into a "partial" result and would rebind child-owned
        # artifacts that may still have been open when the child sealed.
        return None

    evidence_root = native_path.parent
    cells = list(runtime_inputs.get("cells") or [])
    if len(cells) != 10:
        return None
    partial_episodes: list[dict[str, Any]] = []
    partial_artifacts: list[dict[str, Any]] = []
    completed_indices: set[int] = set()
    observed_indices: set[int] = set()
    for path in sorted(
        evidence_root.glob(
            "cell_runs/*/native_task_arena_policy_canary_session_result.v1.json"
        )
    ):
        try:
            index = int(path.parent.name)
        except ValueError:
            continue
        if index < 0 or index >= len(cells):
            continue
        child = read_record(path, code="policy_canary_partial_cell_result_invalid")
        episodes = child.get("episodes")
        if (
            child.get("selected_cell_index") != index
            or child.get("result_digest")
            != canonical_digest(child, digest_field="result_digest")
            or not isinstance(episodes, list)
            or not isinstance(child.get("artifact_inventory"), list)
        ):
            raise error_factory(
                "policy_canary_partial_cell_result_invalid"
            )
        child_completed = (
            child.get("status")
            == "runtime_selected_cell_completed_pending_aggregation"
            and len(episodes) == len(candidate_ids)
        )
        child_blocked = (
            child.get("status") == "blocked"
            and len(episodes) <= len(candidate_ids)
        )
        if not child_completed and not child_blocked:
            raise error_factory(
                "policy_canary_partial_cell_result_invalid"
            )
        expected = {
            (candidate, str(cells[index]["cell_id"]), int(cells[index]["seed"]))
            for candidate in candidate_ids
        }
        observed = {
            (
                str(row.get("candidate_id")),
                str(row.get("cell_id")),
                int(row.get("seed")),
            )
            for row in episodes
            if isinstance(row, Mapping)
        }
        if child_completed and observed != expected:
            raise error_factory(
                "policy_canary_partial_cell_pairing_invalid"
            )
        if child_blocked and (
            not observed.issubset(expected) or len(observed) != len(episodes)
        ):
            raise error_factory(
                "policy_canary_partial_cell_pairing_invalid"
            )
        if child_blocked and episodes:
            if (
                child.get("task_success_contract") != runtime_inputs.get("task_success_contract")
                or child.get("task_success_contract_digest")
                != runtime_inputs.get("task_success_contract_digest")
                or any(
                    any(
                        row.get(field) != specs[row["candidate_id"]].get(field)
                        for field in ("checkpoint_digest", "runtime_identity_digest")
                    )
                    for row in episodes
                )
            ):
                raise error_factory(
                    "policy_canary_partial_cell_execution_binding_invalid"
                )
        prefix = f"cell_runs/{index:02d}"
        for row in episodes:
            episode = json.loads(json.dumps(dict(row), allow_nan=False))
            evidence = episode.get("evidence_artifacts")
            if isinstance(evidence, Mapping):
                episode["evidence_artifacts"] = {
                    role: (
                        {
                            **dict(record),
                            "relative_path": f"{prefix}/{record['relative_path']}",
                        }
                        if isinstance(record, Mapping)
                        and isinstance(record.get("relative_path"), str)
                        else record
                    )
                    for role, record in evidence.items()
                }
            partial_episodes.append(episode)
        for record in child.get("artifact_inventory") or []:
            if not isinstance(record, Mapping):
                continue
            copied = dict(record)
            if str(copied.get("relative_path") or "").endswith(
                "/worker_console.log"
            ) or copied.get("relative_path") == "worker_console.log":
                # Legacy child receipts could include the parent-owned stdout
                # log before the parent appended the final exit lines. It is a
                # mutable diagnostic, not episode evidence, so partial recovery
                # must not publish its stale digest.
                continue
            if isinstance(copied.get("relative_path"), str):
                copied["relative_path"] = f"{prefix}/{copied['relative_path']}"
            partial_artifacts.append(copied)
        observed_indices.add(index)
        if child_completed:
            completed_indices.add(index)
    if not partial_episodes and not partial_artifacts:
        return None
    gap_root = evidence_root / "partial_terminal_evidence"
    gap_root.mkdir(parents=True, exist_ok=True)
    gap_path = gap_root / "typed_media_gap.json"
    gap_value = {
        "schema_version": "task_evaluation_policy_canary_media_gap.v1",
        "type": "cell_not_completed_before_terminal_failure",
        "reason": (fallback.get("blockers") or ["policy_canary_worker_timeout"])[0],
        "completed_cell_indices": sorted(completed_indices),
        "observed_cell_indices": sorted(observed_indices),
        "candidate_policy_queried": any(
            row.get("candidate_policy_queried") is True for row in partial_episodes
        ),
    }
    write_record(gap_path, gap_value)
    observed_keys = {
        (str(row["candidate_id"]), str(row["cell_id"]), int(row["seed"]))
        for row in partial_episodes
    }
    missing_episodes = []
    for candidate in candidate_ids:
        spec = specs[candidate]
        for cell in cells:
            key = (candidate, str(cell["cell_id"]), int(cell["seed"]))
            if key in observed_keys:
                continue
            missing_episodes.append(
                {
                    "candidate_id": candidate,
                    "cell_id": cell["cell_id"],
                    "seed": cell["seed"],
                    "status": "blocked",
                    "candidate_policy_queried": False,
                    "actions_reached_robot": False,
                    "arm_moved": False,
                    "policy_outcome_interpretable": False,
                    "typed_harness_failure": (
                        "cell_not_completed_before_terminal_failure"
                    ),
                    "checkpoint_digest": spec["checkpoint_digest"],
                    "runtime_identity_digest": spec["runtime_identity_digest"],
                    "reset_state_digest": canonical_digest(
                        {
                            "resolved_scenario": cell["resolved_scenario"],
                            "seed": cell["seed"],
                            "execution_performed": False,
                        }
                    ),
                    "visual_evidence": {
                        "media_gap": {
                            "type": gap_value["type"],
                            "reason": gap_value["reason"],
                        }
                    },
                    "evidence_artifacts": {},
                }
            )
    value: dict[str, Any] = {
        "schema_version": "native_task_arena_policy_canary_session_result.v1",
        "status": "blocked",
        "run_kind": run_kind,
        "claim_ceiling": claim_ceiling,
        "candidate_ids": list(candidate_ids),
        "task_success_contract": runtime_inputs["task_success_contract"],
        "task_success_contract_digest": runtime_inputs[
            "task_success_contract_digest"
        ],
        "episodes_per_policy": 10,
        "learned_policy_rollout_count": 20,
        "episodes": [*partial_episodes, *missing_episodes],
        "artifact_inventory": [
            *partial_artifacts,
            {
                "role": "typed_media_gap",
                "relative_path": gap_path.relative_to(evidence_root).as_posix(),
                "media_type": "application/json",
                "size_bytes": gap_path.stat().st_size,
                "sha256": sha256(gap_path),
            },
        ],
        "candidate_policy_queried": any(
            row.get("candidate_policy_queried") is True for row in partial_episodes
        ),
        "completed_cell_count": len(completed_indices),
        "incomplete_cell_count": len(cells) - len(completed_indices),
        "official_ranking_performed": False,
        "scene_promotion_performed": False,
        "blockers": sorted(
            set(
                [str(item) for item in fallback.get("blockers") or []]
                + [
                    "policy_canary_partial_cell_results_preserved",
                    f"policy_canary_incomplete_cell_count:{len(cells) - len(completed_indices)}",
                ]
            )
        ),
        "result_digest": "",
    }
    value["result_digest"] = canonical_digest(value, digest_field="result_digest")
    output_path = evidence_root / "policy_canary_partial_provider_result.v1.json"
    write_record(output_path, value)
    return value, output_path

