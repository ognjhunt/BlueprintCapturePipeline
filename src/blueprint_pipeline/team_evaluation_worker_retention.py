"""Bound completed team-cell memory and retain digest-bound raw evidence (beta L5)."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Callable, Mapping

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.policy_execution_evidence import (
    candidate_execution_kinds, evidence_kind, execution_evidence_valid, rollout_counts_valid,
    session_evidence_binding_valid, retained_rollout_counts, SCRIPTED,
)
from blueprint_pipeline.native_task_arena_policy_canary_session import BETA_RUN_KIND, EXECUTION_MODES, PROVIDER_RESULT_FILENAME, TEAM_RUN_KIND


def _prefix_episode_evidence_paths(
    episode: Mapping[str, Any], *, prefix: str
) -> dict[str, Any]:
    value = deepcopy(dict(episode))
    evidence = value.get("evidence_artifacts")
    if isinstance(evidence, Mapping):
        rewritten: dict[str, Any] = {}
        for role, record in evidence.items():
            if isinstance(record, Mapping) and isinstance(
                record.get("relative_path"), str
            ):
                rewritten[role] = {
                    **record,
                    "relative_path": f"{prefix}/{record['relative_path']}",
                    **({"media_root_relative": f"{prefix}/{record['media_root_relative']}"}
                       if isinstance(record.get("media_root_relative"), str) else {}),
                }
            else:
                rewritten[role] = record
        value["evidence_artifacts"] = rewritten
    return value


def _validate_isolated_cell_result(*, child: Mapping[str, Any], inputs: Mapping[str, Any], index: int) -> bool:
    """Validate one completed child; False denotes the retained pre-policy failure shape."""
    candidate_ids = tuple(str(value) for value in inputs["candidate_ids"])
    if type(child.get("selected_cell_index")) is not int or child["selected_cell_index"] != index:
        raise RuntimeError("policy_canary_isolated_cell_result_invalid")
    if (child.get("result_digest") == canonical_digest(child, digest_field="result_digest")
            and child.get("task_success_contract") == inputs.get("task_success_contract")
            and child.get("task_success_contract_digest") == inputs.get("task_success_contract_digest")
            and child.get("status") == "blocked" and child.get("episodes") == []
            and child.get("policy_loads") == [] and child.get("candidate_policy_queried") is False
            and child.get("preload_observation_gate") is None
            and (child.get("session_closeout") or {}).get("runtime_closed") is True
            and (child.get("telemetry") or {}).get("message_count") == 0
            and (child.get("telemetry") or {}).get("channel_message_counts")
                == {"observations": 0, "episode_envelopes": 0}
            and child.get("session_failure_type")
            and any(frame.get("function") == "prepolicy_observation_gate"
                for frame in child.get("session_failure_source_trace") or [])):
        # Retain the raw pre-policy failure separately. It is not an episode,
        # policy outcome, loaded candidate, or measured runtime identity.
        return False
    if (
        child.get("selected_cell_index") != index
        or child.get("status")
        != "runtime_selected_cell_completed_pending_aggregation"
        or not isinstance(child.get("episodes"), list)
        or len(child["episodes"]) != len(candidate_ids)
        or ("candidate_execution_evidence_kinds" in inputs and (not rollout_counts_valid(child, 1)
            or candidate_execution_kinds(child) != candidate_execution_kinds(inputs)))
        or child.get("task_success_contract")
        != inputs.get("task_success_contract")
        or child.get("task_success_contract_digest")
        != inputs.get("task_success_contract_digest")
        or child.get("result_digest") != canonical_digest(child, digest_field="result_digest")
    ):
        raise RuntimeError("policy_canary_isolated_cell_result_invalid")
    cell = inputs["cells"][index]
    for row in child["episodes"]:
        if (not isinstance(row, Mapping)
                or (row.get("telemetry") is not None and not isinstance(row["telemetry"], Mapping))):
            raise RuntimeError("policy_canary_isolated_cell_result_invalid")
        if (evidence_kind(row) != candidate_execution_kinds(inputs).get(row.get("candidate_id"))
                or not execution_evidence_valid(row)):
            raise RuntimeError("policy_canary_isolated_cell_execution_evidence_invalid")
        if any(row.get(field) != cell.get(field) for field in (
            "cell_id", "seed", "cell_spec_digest", "family",
            "resolved_scenario", "resolved_scenario_digest",
        )):
            raise RuntimeError("policy_canary_isolated_cell_scenario_binding_invalid")
    if {str(row.get("candidate_id")) for row in child["episodes"]} != set(candidate_ids):
        raise RuntimeError("policy_canary_isolated_cell_pairing_invalid")
    return True


def _compact_team_child_result(child: Mapping[str, Any], *, child_root: Path, output_root: Path) -> dict[str, Any]:
    """Keep bounded scoring envelopes in memory; retain the sealed complete child on disk."""
    import hashlib

    path = child_root / PROVIDER_RESULT_FILENAME
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    source = {"relative_path": path.relative_to(output_root).as_posix(), "size_bytes": path.stat().st_size,
              "sha256": "sha256:" + digest.hexdigest(), "result_digest": child["result_digest"]}
    rows = []
    for row in child.get("episodes", []):
        episode = row.get("episode") if isinstance(row.get("episode"), Mapping) else {}
        motion = episode.get("motion_evidence") if isinstance(episode.get("motion_evidence"), Mapping) else {}
        summary = {**row, "episode_summary_kind": "team_terminal_episode_summary.v1",
                   "source_child_result": source,
                   "episode": {**({"score": episode["score"]} if "score" in episode else {}),
                               "motion_evidence": dict(motion)}}
        if len(json.dumps(summary, sort_keys=True, allow_nan=False).encode()) > 512 * 1024:
            raise RuntimeError("team_evaluation_episode_summary_budget_exceeded")
        rows.append(summary)
    result = {key: value for key, value in child.items() if key not in {"episodes", "artifact_inventory", "result_digest"}}
    result.update(episodes=rows, source_child_result=source)
    result["result_digest"] = canonical_digest(result, digest_field="result_digest")
    return result


def _team_checkpoint(*, output_root: Path, inputs: Mapping[str, Any],
                     authority: Mapping[str, Any], children: list[Mapping[str, Any]],
                     seal_result: Callable[..., None]) -> None:
    """Fsync a digest-bound completed prefix; never authorize replay or another GPU."""
    from blueprint_pipeline.native_task_arena_policy_canary_session import _episode_evidence_valid

    newest = children[-1]
    if newest.get("schema_version") != "native_task_arena_policy_canary_session_result.v1":
        raise RuntimeError("team_evaluation_checkpoint_child_schema_invalid")
    _validate_isolated_cell_result(child=newest, inputs=inputs, index=len(children) - 1)
    first_rows = next((child["episodes"] for child in children if child.get("episodes")), [])
    identities = {row["candidate_id"]: (row.get("checkpoint_digest"), row.get("runtime_identity_digest"))
                  for row in first_rows}
    for row in newest.get("episodes", []):
        if (row.get("run_kind") != TEAM_RUN_KIND or row.get("claim_ceiling") != "diagnostic_policy_execution"
                or row.get("ranking_eligible") is not False or not _episode_evidence_valid(row)):
            raise RuntimeError("team_evaluation_checkpoint_episode_evidence_invalid")
        if (row.get("checkpoint_digest"), row.get("runtime_identity_digest")) != identities[row["candidate_id"]]:
            raise RuntimeError("policy_canary_session_candidate_identity_drift")
    refs = []
    for index, child in enumerate(children):
        # Earlier children were checked before their checkpoint and are never
        # changed here. Rehashing all media-bearing records on every cell
        # would turn a 200-cell campaign into quadratic serialization work.
        if child.get("selected_cell_index") != index:
            raise RuntimeError("team_evaluation_checkpoint_child_binding_invalid")
        refs.append({"cell_id": inputs["cells"][index]["cell_id"],
                     "relative_path": f"cell_runs/{index:02d}/{PROVIDER_RESULT_FILENAME}",
                     "result_digest": (child["result_digest"] if index == len(children) - 1 else
                         (child.get("source_child_result") or {}).get("result_digest", child["result_digest"]))})
    value = {"schema_version": "team_evaluation_checkpoint.v1", "status": "in_progress",
             "runtime_inputs_digest": inputs["runtime_inputs_digest"],
             "authority_digest": authority["authority_digest"], "planned_cells": len(inputs["cells"]),
             "completed_child_results": refs, "provider_allocations_authorized": 1,
             "completed_attempt_replay_authorized": False, "result_digest": ""}
    value["result_digest"] = canonical_digest(value, digest_field="result_digest")
    seal_result(result_path=output_root / "team_evaluation_checkpoint.json", result=value)



def _validate_provider_manifest(
    value: Mapping[str, Any],
    *,
    runtime_inputs: Mapping[str, Any],
    authority: Mapping[str, Any],
) -> dict[str, Any]:
    manifest = json.loads(json.dumps(dict(value), allow_nan=False))
    if (
        manifest.get("schema_version")
        != "native_task_arena_policy_canary_provider_bundle.v1"
        or manifest.get("execution_mode")
        != EXECUTION_MODES.get(str(runtime_inputs.get("run_kind")))
        or manifest.get("run_kind") != runtime_inputs.get("run_kind")
        or manifest.get("claim_ceiling") != "diagnostic_policy_execution"
        or manifest.get("runtime_inputs_digest")
        != runtime_inputs.get("runtime_inputs_digest")
        or manifest.get("task_success_contract_digest")
        != runtime_inputs.get("task_success_contract_digest")
        or manifest.get("authority_digest") != authority.get("authority_digest")
        or not session_evidence_binding_valid(manifest, authority, runtime_inputs)
        or manifest.get("input_digest")
        != canonical_digest(manifest, digest_field="input_digest")
    ):
        raise RuntimeError("policy_canary_provider_manifest_invalid")
    return manifest


def seal_scripted_interruption(*, inputs: Mapping[str, Any], authority: Mapping[str, Any],
        children: list[Mapping[str, Any]], child_root: Path, output_root: Path, index: int,
        reason: str, seal_result: Callable[..., None]) -> bool:
    """Retain known scripted episodes without inventing the interrupted cell's state.

    The legacy interruption reader understands only the frozen learned pair.
    A scripted timeout retains its own already sealed children and an explicit
    unknown for an unsealed current cell, without allocating or retrying.
    """
    kinds = candidate_execution_kinds(inputs)
    if inputs.get("run_kind") != BETA_RUN_KIND or set(kinds.values()) != {SCRIPTED}:
        return False
    retained = list(children)
    current = child_root / PROVIDER_RESULT_FILENAME
    verified_current = False
    if current.is_file() and not current.is_symlink() and current.stat().st_size <= 128 * 1024**2:
        try:
            child = json.loads(current.read_text())
            verified_current = _validate_isolated_cell_result(child=child, inputs=inputs, index=index)
            if verified_current:
                retained.append(child)
        except (ValueError, TypeError, KeyError, RuntimeError, OSError):
            pass  # Original bytes remain on disk; unverified content is not execution evidence.
    episodes = [_prefix_episode_evidence_paths(row, prefix=f"cell_runs/{ordinal:02d}")
                for ordinal, child in enumerate(retained) for row in child.get("episodes", [])]
    inventory = [{**row, "relative_path": f"cell_runs/{ordinal:02d}/{row['relative_path']}"}
                 for ordinal, child in enumerate(retained) for row in child.get("artifact_inventory", [])]
    value = {"schema_version": "native_task_arena_policy_canary_session_result.v1", "status": "blocked",
        "run_id": authority["run_id"], "run_kind": BETA_RUN_KIND, "claim_ceiling": "diagnostic_policy_execution",
        "candidate_ids": list(kinds), **retained_rollout_counts(kinds, episodes),
        "episodes_per_policy": len(retained), "planned_episodes_per_policy": len(inputs["cells"]),
        "planned_learned_policy_rollout_count": 0, "planned_scripted_rollout_count": len(inputs["cells"]),
        "episodes": episodes, "artifact_inventory": inventory,
        "task_success_contract": inputs["task_success_contract"],
        "task_success_contract_digest": inputs["task_success_contract_digest"],
        "runtime_inputs_digest": inputs["runtime_inputs_digest"], "authority_digest": authority["authority_digest"],
        **{key: inputs[key] for key in ("beta_screening_admission", "scene_revision_digest", "plan_digest", "configuration_digest")},
        "candidate_policy_queried": False, "candidate_policy_query_attempted": False,
        "interrupted_cell_index": index, "interrupted_cell_receipt_verified": verified_current,
        "interrupted_scripted_invocation_status": "retained_receipt" if verified_current else "unknown",
        "retry_cap": 0, "warm_session_open_count": 1, "provider_allocations_observed": None,
        "session_closeout": {"status": "interrupted_runtime_close_unproven", "runtime_closed": False,
                             "provider_closeout_pending": True},
        "automatic_retry_performed": False, "official_ranking_performed": False, "scene_promotion_performed": False,
        "provider_zero_required_after_return": True, "blockers": [reason]}
    value["artifact_inventory_digest"] = canonical_digest({"value": inventory})
    value["result_digest"] = canonical_digest(value, digest_field="result_digest")
    seal_result(result_path=output_root / PROVIDER_RESULT_FILENAME, result=value)
    return True
