"""Select retained scores across an explicitly bound pre-policy reset recovery.

Original session/episode receipts remain untouched. A scored policy attempt
can never be replaced, even when it failed the task.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest


def recovery_arguments(plan: Mapping[str, Any], *, request_digest: str,
                       admission_digest: str, episodes: int) -> dict[str, Any]:
    """Bind the selection to unchanged original request, admission and frozen shards."""
    from .controlled_policy_configuration import canonical_request_digest

    if (plan.get("schema_version") != "team_candidate_pre_policy_reset_recovery_plan.v1"
            or plan.get("plan_digest") != canonical_digest(plan, digest_field="plan_digest")
            or plan.get("canonical_request_digest") != request_digest
            or plan.get("admission_digest") != admission_digest):
        raise ValueError("team_candidate_recovery_plan_binding_invalid")
    records = plan.get("source_records")
    if not isinstance(records, list):
        raise ValueError("team_candidate_recovery_sources_missing")
    sources: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        path = Path(record["path"])
        if not path.is_absolute() or path.is_symlink() or not path.is_file():
            raise ValueError("team_candidate_recovery_source_record_invalid")
        data = path.read_bytes()
        if len(data) != record["size_bytes"] or "sha256:" + hashlib.sha256(data).hexdigest() != record["sha256"]:
            raise ValueError("team_candidate_recovery_source_record_changed")
        sources[record["role"]].append(json.loads(data))
    if (set(sources) != {"prior_terminal", "original_request", "original_admission", "original_shard", "retained_native_session"}
            or any(len(sources[key]) != 1 for key in ("prior_terminal", "original_request", "original_admission"))):
        raise ValueError("team_candidate_recovery_source_roles_invalid")
    terminal = sources["prior_terminal"][0]
    if (terminal.get("status") != "blocked" or terminal.get("provider_zero_verified") is not True
            or terminal.get("canonical_request_digest") != request_digest
            or canonical_request_digest(sources["original_request"][0]) != request_digest
            or sources["original_admission"][0].get("admission_digest") != admission_digest):
        raise ValueError("team_candidate_recovery_prior_binding_invalid")
    shards = sorted(sources["original_shard"], key=lambda value: value["index"])
    if [s["index"] for s in shards] != list(range(len(shards))):
        raise ValueError("team_candidate_recovery_original_shards_invalid")
    cells = [cell for shard in shards for cell in shard["cells"]]
    if cells != plan.get("expected_cells") or len(cells) != episodes:
        raise ValueError("team_candidate_recovery_frozen_cells_mismatch")
    retained = {}
    for session in sources["retained_native_session"]:
        for row in session["episodes"]:
            cell_id = row["cell_id"]
            if cell_id in retained:
                raise ValueError("team_candidate_recovery_retained_cell_duplicate")
            retained[cell_id] = canonical_digest(row)
    return {"expected_cells": cells, "pre_policy_recovery_cells": plan["recovery_cell_ids"],
            "retained_episode_digests": retained}


def select_pre_policy_reset_recovery(
    session_rows: Sequence[tuple[str, Mapping[str, Any]]], *,
    expected_cells: Sequence[Mapping[str, Any]], recovery_cell_ids: Sequence[str],
    retained_episode_digests: Mapping[str, str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    expected, allowed = [cell["cell_id"] for cell in expected_cells], list(recovery_cell_ids)
    frozen = {cell["cell_id"]: cell for cell in expected_cells}
    if (not expected or len(set(expected)) != len(expected)
            or not allowed or len(set(allowed)) != len(allowed)
            or any(not isinstance(x, str) or not x for x in [*expected, *allowed])
            or not set(allowed).issubset(expected)):
        raise ValueError("team_candidate_recovery_cell_plan_invalid")
    grouped: dict[str, list[tuple[str, Mapping[str, Any]]]] = defaultdict(list)
    for digest, row in session_rows:
        grouped[str(row.get("cell_id") or "")].append((digest, row))
    if set(grouped) != set(expected):
        raise ValueError("team_candidate_recovery_frozen_cells_mismatch")
    if (not retained_episode_digests or not set(allowed).issubset(retained_episode_digests)
            or not set(retained_episode_digests).issubset(expected)):
        raise ValueError("team_candidate_recovery_retained_bindings_missing")
    for cell_id, digest in retained_episode_digests.items():
        if sum(canonical_digest(row) == digest for _, row in grouped[cell_id]) != 1:
            raise ValueError("team_candidate_recovery_retained_episode_changed")
    duplicate = {key for key, rows in grouped.items() if len(rows) > 1}
    if duplicate != set(allowed):
        raise ValueError("team_candidate_recovery_duplicate_cells_mismatch")
    selected, selected_sources, refusals = [], [], []
    identities = ("candidate_id", "cell_id", "seed", "cell_spec_digest", "checkpoint_digest", "runtime_identity_digest")
    for cell_id in expected:
        candidates = grouped[cell_id]
        completed = [(digest, row) for digest, row in candidates if row.get("status") == "completed"]
        if len(completed) != 1:
            raise ValueError("team_candidate_recovery_completed_attempt_count_invalid")
        chosen_digest, chosen = completed[0]
        if any(chosen.get(key) != frozen[cell_id].get(key) or frozen[cell_id].get(key) is None
               for key in ("seed", "cell_spec_digest", "resolved_scenario_digest")):
            raise ValueError("team_candidate_recovery_frozen_cell_identity_mismatch")
        score = (chosen.get("episode") or {}).get("score") or {}
        if (chosen.get("scoring_authority") != "deterministic_simulator_state"
                or score.get("status") != "scored" or type(score.get("task_succeeded")) is not bool
                or chosen.get("candidate_policy_queried") is not True or chosen.get("actions_reached_robot") is not True):
            raise ValueError("team_candidate_recovery_selected_episode_not_scored")
        for digest, row in candidates:
            if row is chosen:
                continue
            if (any(row.get(key) != chosen.get(key) or row.get(key) is None for key in identities)
                    or type(row.get("seed")) is not int):
                raise ValueError("team_candidate_recovery_episode_identity_mismatch")
            episode = row.get("episode") or {}
            if (row.get("status") != "blocked"
                    or row.get("episode_failure_stage") != "before_first_observation"
                    or row.get("failure_message") != "scientific_reset_scenario_application_mismatch"
                    or row.get("candidate_policy_query_attempted") is not False
                    or row.get("candidate_policy_queried") is not False
                    or row.get("actions_reached_robot") is not False
                    or episode.get("candidate_policy_query_attempted") is not False
                    or episode.get("policy_response_status") != "not_attempted"
                    or (episode.get("score") or {}).get("status") == "scored"):
                raise ValueError("team_candidate_recovery_refusal_not_before_policy")
            refusals.append({"cell_id": cell_id, "session_result_digest": digest,
                "episode_receipt_digest": canonical_digest(row),
                "failure_message": row["failure_message"], "candidate_policy_query_attempted": False})
        selected.append(dict(chosen))
        selected_sources.append({"cell_id": cell_id, "session_result_digest": chosen_digest})
    receipt = {"schema_version": "team_candidate_pre_policy_reset_recovery.v1",
        "selected_episode_count": len(selected), "selected_episode_sources": selected_sources,
        "superseded_pre_policy_refusals": refusals, "original_session_results_preserved": True,
        "authenticated_retained_episode_count": len(retained_episode_digests),
        "completed_policy_attempts_replaced": 0, "receipt_digest": ""}
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    return selected, receipt
