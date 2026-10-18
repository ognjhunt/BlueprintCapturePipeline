"""The scoring statement every agent-run result carries to the WebApp.

Beta journey step: L5 team policy evaluation. The WebApp's result route used to
receive two numbers, episodes run and episodes succeeded, with nothing saying
who decided the second. Every result this repository reports now states:

- ``runtime``: which execution path produced the episodes;
- ``success_source``: who decided each episode's success;
- the bindings that make a simulator score mean "against this job": the
  success-contract, scene-revision, robot-free-probe, team-candidate-admission
  and session-result digests; and
- ``claim_ceiling``: carried unchanged, never upgraded.

The WebApp admits a result as recommendation evidence only through a positive
allowlist (today exactly ``team_evaluation_scored`` with ``simulator_scored``
and every binding present). This module states facts; it
does not decide admission. Mirror: Blueprint-WebApp
``server/utils/runResultScoring.ts``.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

SCHEMA_VERSION = "blueprint.agent_run_result_scoring.v1"
# A team's evaluation: 200 hidden cells, the only run a recommendation may rest on.
TEAM_SCORED = "team_evaluation_scored"
TEAM_RUNTIMES = {"scored": TEAM_SCORED}
CONTROLLED_NATIVE_SEALED_TASK = "controlled_native_sealed_task"
LEGACY_COMMAND_BATCH = "legacy_robot_eval_command_batch"
SIMULATOR_SCORED = "simulator_scored"
UNVERIFIED = "unverified"
DIGEST_FIELDS = (
    "success_contract_digest",
    "scene_revision_digest",
    "robot_free_probe_receipt_digest",
    "team_candidate_admission_digest",
    "session_result_digest",
)
FIELDS = frozenset({"schema_version", "runtime", "success_source", "scored_episode_count",
                    "claim_ceiling", *DIGEST_FIELDS})
_LABEL = re.compile(r"^[a-z0-9_.-]{1,100}$")
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")


class AgentRunResultScoringError(ValueError):
    """A scoring statement is malformed and must not be reported."""


def validate_scoring_statement(value: Mapping[str, Any], *, episodes_run: int) -> dict[str, Any]:
    """Strict mirror of the WebApp wire schema, plus the impossible-count check."""

    row = dict(value) if isinstance(value, Mapping) else {}
    count = row.get("scored_episode_count")
    if (set(row) not in (FIELDS, FIELDS | {"site_task_qualification"}) or row.get("schema_version") != SCHEMA_VERSION
            or not all(isinstance(row.get(key), str) and _LABEL.fullmatch(row[key])
                       for key in ("runtime", "success_source", "claim_ceiling"))
            or type(count) is not int or not 0 <= count <= episodes_run
            or not all(row[key] is None or (isinstance(row[key], str) and _DIGEST.fullmatch(row[key]))
                       for key in DIGEST_FIELDS)):
        raise AgentRunResultScoringError("agent_run_result_scoring_invalid")
    if "site_task_qualification" in row:
        from .site_task_simulation_qualification import validate_projection
        projection = validate_projection(row["site_task_qualification"])
        if (row["claim_ceiling"] != "site_task_simulation" or row["runtime"] != TEAM_SCORED
                or row["success_source"] != SIMULATOR_SCORED
                or projection["scene_revision_digest"] != row["scene_revision_digest"]):
            raise AgentRunResultScoringError("agent_run_result_qualification_mismatch")
    elif row["claim_ceiling"] == "site_task_simulation":
        raise AgentRunResultScoringError("agent_run_result_qualification_missing")
    return row


def scoring_statement(*, runtime: str, success_source: str, scored_episode_count: int,
                      claim_ceiling: str, site_task_qualification: Mapping[str, Any] | None = None,
                      **digests: str | None) -> dict[str, Any]:
    unknown = set(digests) - set(DIGEST_FIELDS)
    if unknown:
        raise AgentRunResultScoringError("agent_run_result_scoring_field_unknown")
    value = {"schema_version": SCHEMA_VERSION, "runtime": runtime, "success_source": success_source,
             "scored_episode_count": scored_episode_count, "claim_ceiling": claim_ceiling,
             **{key: digests.get(key) for key in DIGEST_FIELDS}}
    if site_task_qualification is not None:
        value["site_task_qualification"] = dict(site_task_qualification)
    return validate_scoring_statement(value, episodes_run=scored_episode_count)


def unstated_scoring() -> dict[str, Any]:
    """A terminal that said nothing about scoring is reported as exactly that."""

    return scoring_statement(runtime="unstated", success_source=UNVERIFIED,
                             scored_episode_count=0, claim_ceiling="unverified")


def legacy_command_batch_scoring() -> dict[str, Any]:
    """Legacy terminal artifacts: the count is a command batch's, not a scored team policy's."""

    return scoring_statement(runtime=LEGACY_COMMAND_BATCH, success_source=UNVERIFIED,
                             scored_episode_count=0, claim_ceiling="unverified_legacy_batch")


def controlled_native_scoring(terminal: Mapping[str, Any]) -> dict[str, Any]:
    """One native episode scored on the operator's sealed task, not on a website job."""

    private = terminal.get("private_execution_result")
    private = private if isinstance(private, Mapping) else {}
    return scoring_statement(
        runtime=CONTROLLED_NATIVE_SEALED_TASK, success_source=SIMULATOR_SCORED,
        scored_episode_count=int(terminal.get("episodes_run") or 0), claim_ceiling="development_only",
        success_contract_digest=private.get("task_spec_digest"),
        session_result_digest=private.get("execution_receipt_digest"),
    )


__all__ = [
    "CONTROLLED_NATIVE_SEALED_TASK",
    "LEGACY_COMMAND_BATCH",
    "SCHEMA_VERSION",
    "SIMULATOR_SCORED",
    "TEAM_RUNTIMES",
    "TEAM_SCORED",
    "AgentRunResultScoringError",
    "controlled_native_scoring",
    "legacy_command_batch_scoring",
    "scoring_statement",
    "unstated_scoring",
    "validate_scoring_statement",
]
