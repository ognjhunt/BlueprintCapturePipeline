"""Retain canary result prefixes, bounded raw evidence and exact runtime closeout."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Callable, Mapping

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.native_task_arena_policy_canary_session import (
    BETA_RUN_KIND, PROVIDER_RESULT_FILENAME, PROVIDER_STATUS_FILENAME, PROVIDER_STATUS_SCHEMA, TEAM_RUN_KIND,
)


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



def write_team_status_sidecar(*, result_path: Path, result: Mapping[str, Any], seal_result: Callable, sha256: Callable) -> None:
    """Bind small diagnostic status to the exact sealed terminal bytes, without grading them."""
    summary = {"schema_version": PROVIDER_STATUS_SCHEMA, "run_kind": TEAM_RUN_KIND,
               "claim_ceiling": "diagnostic_policy_execution", "status": result.get("status"),
               "blockers": result.get("blockers") or [],
               "source_result": {"file_name": result_path.name, "size_bytes": result_path.stat().st_size,
                                 "sha256": sha256(result_path), "result_digest": result["result_digest"]},
               "inspection_scope": "hash_bound_status_metadata_only", "result_digest": ""}
    summary["result_digest"] = canonical_digest(summary, digest_field="result_digest")
    if len(json.dumps(summary).encode()) > 64 * 1024:
        raise RuntimeError("team_status_summary_budget_exceeded")
    seal_result(result_path=result_path.with_name(PROVIDER_STATUS_FILENAME), result=summary)


def seal_before_simulation_close(*, result_path: Path, result: Mapping[str, Any], simulation_app: Any, seal_result: Callable) -> None:
    """Seal the provider result before Isaac can terminate the interpreter."""

    close = getattr(simulation_app, "close", None)
    if not callable(close):
        raise RuntimeError("policy_canary_simulation_close_unavailable")
    seal_result(result_path=result_path, result=result)
    close()


def seal_runtime_stop_sidecar(*, guard, output_root: Path, seal_result: Callable) -> None:
    """Keep a control stop alongside an independently sealed strict-matrix result."""
    if guard is not None and (stopped := guard.stop_receipt()) is not None:
        seal_result(result_path=output_root / "beta_screening_runtime_stop.v1.json", result={
            **stopped, "result_digest": canonical_digest(stopped)})


def seal_team_prefix(*, output_root: Path, inputs: Mapping[str, Any], authority: Mapping[str, Any],
                     children: list[Mapping[str, Any]], construction_lineage_mode: str, reason: str,
                     aggregate: Callable, seal_result: Callable, stopped_before_next_cell: bool = False,
                     interrupted_cell_index: int | None = None) -> None:
    if inputs.get("run_kind") != TEAM_RUN_KIND or not children:
        return
    result = aggregate(
        inputs={**inputs, "cells": inputs["cells"][:len(children)]}, authority=authority,
        child_results=children, output_root=output_root, construction_lineage_mode=construction_lineage_mode)
    result.update(status="blocked", blockers=[reason], planned_episodes_per_policy=len(inputs["cells"]),
                  execution_stopped_after_cell=len(children),
                  stopped_before_next_cell=stopped_before_next_cell)
    if interrupted_cell_index is not None:
        result["interrupted_cell_index"] = interrupted_cell_index
        if result.get("candidate_policy_queried") is not True:
            result["candidate_policy_queried"] = None
        if not result.get("policy_loads"):
            result["policy_loads"] = None
    result["result_digest"] = canonical_digest(result, digest_field="result_digest")
    seal_result(result_path=output_root / PROVIDER_RESULT_FILENAME, result=result)


def seal_beta_control_stop(*, output_root: Path, inputs: Mapping[str, Any], authority: Mapping[str, Any],
                           children: list[Mapping[str, Any]], construction_lineage_mode: str,
                           receipt: Mapping[str, Any], aggregate: Callable, validate_child: Callable,
                           sha256: Callable, seal_result: Callable) -> None:
    from .task_family_policy_canary_execution import beta_result_binding
    prefix, partial = [], []
    for index, child in enumerate(children):
        try:
            validate_child(child=child, inputs=inputs, index=index)
        except RuntimeError:
            # A revoked mid-query child is retained as its exact raw bytes. It
            # cannot masquerade as a completed scored cell in aggregation.
            path = output_root / "cell_runs" / f"{index:02d}" / PROVIDER_RESULT_FILENAME
            partial.append({"relative_path": path.relative_to(output_root).as_posix(),
                            "size_bytes": path.stat().st_size, "sha256": sha256(path),
                            "result_digest": child.get("result_digest")})
        else:
            if partial:
                raise RuntimeError("beta_screening_runtime_nonprefix_result")
            prefix.append(child)
    result = (aggregate(inputs={**inputs, "cells": inputs["cells"][:len(prefix)]},
        authority=authority, child_results=prefix, output_root=output_root,
        construction_lineage_mode=construction_lineage_mode) if prefix else {
            "schema_version": "native_task_arena_policy_canary_session_result.v1", "run_kind": BETA_RUN_KIND,
            "run_id": inputs["run_id"], "authority_digest": authority["authority_digest"],
            "claim_ceiling": "diagnostic_policy_execution", "candidate_ids": inputs["candidate_ids"],
            "task_success_contract": inputs["task_success_contract"],
            "task_success_contract_digest": inputs["task_success_contract_digest"],
            "episodes": [], "policy_loads": [], "candidate_policy_queried": False,
            "artifact_inventory": [], "scene_promotion_performed": False, "official_ranking_performed": False,
            "session_closeout": {"runtime_closed": True, "provider_closeout_pending": True},
            **beta_result_binding(inputs)})
    result.update(status="blocked", blockers=[receipt["reason"]], planned_episodes_per_policy=len(inputs["cells"]),
                  execution_stopped_after_cell=len(children), stopped_before_next_cell=True,
                  execution_control_stop=dict(receipt), applicability="historical_only",
                  retained_partial_cell_results=partial,
                  new_processing_authorized=False, read_only_closeout_allowed=True)
    result["result_digest"] = canonical_digest(result, digest_field="result_digest")
    seal_result(result_path=output_root / PROVIDER_RESULT_FILENAME, result=result)
