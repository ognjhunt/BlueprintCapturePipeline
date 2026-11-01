"""The beta job's evaluation result: a bounded simulation estimate.

Beta journey: a prepared site's automatic candidate evaluation. The run under
the site allowance is the job's evaluation; there is no second scoring tier.
Native session validation owns media, simulator scoring and provider closeout.
This projection adds exact campaign/configuration/cell bindings and publishes
each candidate's sample size and counts as they are. It is an estimate for the
recorded task and conditions, never certified physical performance or safety,
and it never enters Arm Decision Proof qualified or physical evidence.
"""
from __future__ import annotations

from collections.abc import Mapping
from decimal import Decimal
from typing import Any

from .decision_evidence_contracts import canonical_digest, cross_runtime_canonical_digest
from .team_policy_episode_outcome import episode_outcome
from .policy_execution_evidence import (
    candidate_execution_kinds, evidence_kind, SCRIPTED, SCRIPTED_ACTION_SOURCE, scripted_episode_outcome,
)

PUBLICATION_SCHEMA = "beta_policy_screening_publication.v2"
# The job-facing ceiling. The native session keeps its own runtime ceiling,
# which is what keeps beta runs out of ADP qualified evidence.
CLAIM_CEILING = "simulation_estimate"
SOURCE_FIELDS = (
    "scene_intent_id", "scene_intent_digest", "capture_id", "task_id",
    "task_context_digest", "scene_revision_digest",
)
CANDIDATE_STATUSES = frozenset({
    "pending", "waitlisted", "preparing", "launching", "running", "reconciliation_required",
    "completed", "blocked", "unavailable", "deferred",
})
# ``waitlisted``: runnable, waiting for site budget a funded candidate may
# release before spend. Not terminal: it becomes ``pending`` when funded, or
# ``deferred`` once nothing at the site can release money any more.
QUEUED_STATUSES = frozenset({"pending", "waitlisted"})
ACTIVE_STATUSES = frozenset({"preparing", "launching", "running", "reconciliation_required"})
TERMINAL_STATUSES = frozenset({"completed", "blocked", "unavailable", "deferred"})


def candidate_result_summary(
    value: Mapping[str, Any], *, plan: Mapping[str, Any], candidate: Mapping[str, Any],
) -> dict[str, Any]:
    """Count only the exact simulator-bound, closed native session's episodes."""
    if candidate.get("execution_spec", {}).get("backend") == "native_g1.v1":
        from .beta_policy_screening_g1 import summarize_result
        return summarize_result(value, plan=plan, candidate=candidate)
    from .native_task_arena_policy_canary_session import BETA_RUN_KIND, validate_session_result

    result = validate_session_result(value)
    execution = candidate["execution_spec"]
    success_digest = execution.get("task_success_contract_digest") or plan["scene_inputs"].get(
        "task_success_contract_digest"
    )
    binding = result.get("beta_screening_admission") or {}
    family = execution.get("backend") == "task_family_native.v1"
    readiness_bound = (binding.get("backend") == "task_family_native.v1"
        and binding.get("pre_policy_site_qualification_required") is True
        and plan.get("pre_policy_site_qualification_required") is True
        and binding.get("family_static_readiness_digest") == plan.get("family_static_readiness_digest")) if family else (
            binding.get("robot_free_probe_receipt_digest") == plan["robot_free_probe_digest"])
    if (result["run_kind"] != BETA_RUN_KIND
            or result["candidate_ids"] != [candidate["candidate_id"]]
            or not success_digest
            or result.get("task_success_contract_digest") != success_digest
            or binding.get("plan_digest") != plan["plan_digest"]
            or binding.get("configuration_digest") != candidate["configuration_digest"]
            or binding.get("admission_digest") != candidate["admission_digest"]
            or not readiness_bound
            or binding.get("scene_revision_digest") != plan["scene_revision_digest"]):
        raise ValueError("beta_screening_result_binding_mismatch")
    expected = {(cell["cell_id"], cell["seed"]) for cell in candidate["scenario_cells"]}
    episodes = result["episodes"]
    observed = {(row["cell_id"], row["seed"]) for row in episodes}
    if (len(episodes) != candidate["episode_count"] or len(expected) != len(episodes)
            or observed != expected):
        raise ValueError("beta_screening_result_cells_mismatch")
    outcomes = []
    expected_cells = {cell["cell_id"]: cell for cell in candidate["scenario_cells"]}
    for row in episodes:
        cell = expected_cells[row["cell_id"]]
        if (row["runtime_identity_digest"] != execution["runtime_digest"]
                or row.get("cell_spec_digest") != cell["cell_spec_digest"]
                or row.get("resolved_scenario_digest") != cell["resolved_scenario_digest"]
                or row.get("resolved_scenario") != cell["resolved_scenario"]):
            raise ValueError("beta_screening_result_execution_mismatch")
        if row.get("status") == "completed":
            episode = row.get("episode")
            score = episode.get("score") if isinstance(episode, Mapping) else None
            if row.get("deterministic_score_digest") != canonical_digest({"value": score}):
                raise ValueError("beta_screening_result_score_binding_mismatch")
        outcome, reason = scripted_episode_outcome(row) if evidence_kind(row) == SCRIPTED else episode_outcome(row)
        outcomes.append({"cell_id": row["cell_id"], "outcome": outcome, "reason": reason})
    succeeded = sum(row["outcome"] == "success" for row in outcomes)
    failed = sum(row["outcome"] == "failure" for row in outcomes)
    void = sum(row["outcome"] == "void" for row in outcomes)
    scored = succeeded + failed
    kind = candidate_execution_kinds(result)[candidate["candidate_id"]]
    return {
        **({"execution_evidence_kind": SCRIPTED, "action_source": SCRIPTED_ACTION_SOURCE,
            "scripted_rollout_count": result["scripted_rollout_count"], "learned_policy_rollout_count": 0,
            "scripted_controller_invoked": result.get("scripted_controller_invoked") is True,
            "candidate_policy_queried": False} if kind == SCRIPTED else {}),
        "episodes_scored": scored, "episodes_succeeded": succeeded,
        "episodes_failed": failed, "episodes_void": void,
        "success_rate": succeeded / scored if scored else None,
        "session_result_digest": result["result_digest"], "episode_outcomes": outcomes,
    }


def campaign_status(candidates: Mapping[str, Mapping[str, Any]]) -> str:
    statuses = [row["status"] for row in candidates.values()]
    if "reconciliation_required" in statuses:
        return "reconciliation_required"
    if any(status in ACTIVE_STATUSES for status in statuses):
        return "running"
    if QUEUED_STATUSES & set(statuses):
        return "queued"
    return "completed" if "completed" in statuses else "blocked"


def money(value: Decimal) -> str:
    """Decimal money on the wire; never round a commitment down."""
    if not value.is_finite() or value < 0 or value != value.quantize(Decimal("0.000001")):
        raise ValueError("beta_screening_money_precision_invalid")
    return format(value, ".6f").rstrip("0").rstrip(".") if value else "0"


def _publication_execution_evidence(configured: Mapping[str, Any], summary: Mapping[str, Any]) -> dict[str, Any]:
    """Retain explicit execution provenance; absence in old results stays unknown."""
    kind = configured.get("execution_evidence_kind")
    if kind is None:
        if summary.get("execution_evidence_kind") == "scripted_simulator_state_baseline":
            raise ValueError("beta_screening_result_evidence_kind_invalid")
        # Do not add a new identity field halfway through a retained campaign.
        return {}
    scripted = kind == "scripted_simulator_state_baseline"
    if (kind not in {"learned_policy", "scripted_simulator_state_baseline"}
            or scripted and configured.get("execution_evidence_kind") != kind
            or summary and summary.get("execution_evidence_kind") != kind):
        raise ValueError("beta_screening_result_evidence_kind_invalid")
    fields = {"execution_evidence_kind": kind,
              "action_source": "scripted_simulator_state" if scripted else "learned_policy",
              "scripted_rollout_count": 0, "learned_policy_rollout_count": 0,
              "scripted_controller_invoked": False, "candidate_policy_queried": False}
    if summary:
        if not fields.keys() <= summary.keys():
            raise ValueError("beta_screening_result_execution_evidence_missing")
        fields = {key: summary[key] for key in fields}
    counts = (fields["scripted_rollout_count"], fields["learned_policy_rollout_count"])
    scored = summary.get("episodes_scored", 0)
    if (any(type(count) is not int or count < 0 for count in counts)
            or sum(counts) > configured.get("episode_count", 0)
            or any(type(fields[key]) is not bool for key in ("scripted_controller_invoked", "candidate_policy_queried"))
            or fields["scripted_controller_invoked"] and fields["scripted_rollout_count"] == 0
            or fields["candidate_policy_queried"] and fields["learned_policy_rollout_count"] == 0
            or fields["action_source"] != ("scripted_simulator_state" if scripted else "learned_policy")
            or (scripted and (fields["learned_policy_rollout_count"] != 0 or fields["candidate_policy_queried"]
                or scored > fields["scripted_rollout_count"] or scored and not fields["scripted_controller_invoked"]))
            or (not scripted and (fields["scripted_rollout_count"] != 0 or fields["scripted_controller_invoked"]
                or scored > fields["learned_policy_rollout_count"] or scored and not fields["candidate_policy_queried"]))):
        raise ValueError("beta_screening_result_execution_evidence_invalid")
    return fields


def screening_publication(
    *, plan: Mapping[str, Any], state: Mapping[str, Any], budget: Mapping[str, Any],
) -> dict[str, Any]:
    """Customer-safe progress: no endpoint, local path, credential or prompt bytes."""
    candidates = []
    for configured in plan["candidates"]:
        row = state["candidates"][configured["candidate_id"]]
        summary = row.get("summary") or {}
        candidates.append({
            "candidate_id": configured["candidate_id"],
            "configuration_digest": configured["configuration_digest"],
            "display_name": configured.get("display_name") or configured["candidate_id"],
            "status": row["status"], "blockers": list(row.get("blockers") or []),
            "episodes_planned": configured.get("episode_count", 0),
            "episodes_scored": summary.get("episodes_scored", 0),
            "episodes_succeeded": summary.get("episodes_succeeded", 0),
            "episodes_failed": summary.get("episodes_failed", 0),
            "episodes_void": summary.get("episodes_void", 0),
            "success_rate": summary.get("success_rate"),
            "session_result_digest": summary.get("session_result_digest"),
            **_publication_execution_evidence(configured, summary),
        })
    value = {
        "schema_version": PUBLICATION_SCHEMA,
        "campaign_id": plan["campaign_id"], "site_id": plan["site_id"],
        "plan_digest": plan["plan_digest"], "catalog_digest": plan["catalog_digest"],
        "scenario_plan_digest": plan["scenario_plan_digest"],
        "source": {field: plan["source"][field] for field in SOURCE_FIELDS},
        "revision": state["revision"], "status": campaign_status(state["candidates"]),
        "purpose": "job_evaluation", "execution_operator": {"kind": "blueprint"},
        "claim_ceiling": CLAIM_CEILING,
        "candidates": candidates, "budget": dict(budget),
        "generated_at": state["updated_at"], "publication_digest": "",
    }
    value["publication_digest"] = cross_runtime_canonical_digest(value, digest_field="publication_digest")
    return value
