"""Public real-world benchmark anchors for validating the evaluator harness.

Blueprint's rank-fidelity machinery is gated on *accepted real-world anchors*,
and its calibration path rejects any real-world outcome that does not join to a
prediction Blueprint itself produced in the same job.  Both rules are right for
customer work.  Taken together, though, they made the platform unprovable: the
only way to demonstrate that the evaluator ranks policies correctly was to find
a customer willing to buy an evaluator whose ranking had never been
demonstrated.

There is a way out that costs no robot time and no customer.  Public real-world
leaderboards already publish success rates for open policy checkpoints, measured
on physical robots by independent parties.  Correlating Blueprint's evaluator
against one of those is a genuine measurement of the harness -- the scoring
rubric, the aggregation, the interval machinery, the ranking -- on real-world
outcomes Blueprint did not produce and cannot influence.

This module supplies the two producers that were missing:

* :func:`build_anchor_snapshot` canonicalises an operator-supplied leaderboard
  export into a digest-pinned snapshot.  Its ``snapshot_sha256`` is the value the
  RoboWorld admission checklist already requires as ``roboarena_snapshot_sha256``
  and for which no producer existed.
* :func:`build_external_reference_results` converts that snapshot into the
  ``external_reference_results.v1`` artifact
  :func:`~blueprint_pipeline.benchmark_protocol.build_external_rank_fidelity_report`
  consumes -- a schema that, until now, had a constant and no producer anywhere
  in the repository.

What this deliberately does not do is loosen the customer gate.  A public anchor
measures a different embodiment, a different site, and a different task family
from any customer deployment, so results carry the distinct
``harness_validation_public_anchor`` scope and are structurally barred from
upgrading site-specific rank-fidelity or deployment claims.  Passing here means
the harness works, not that a customer's policy will.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .common import read_json_any, utc_now_iso, write_json


SNAPSHOT_SCHEMA_VERSION = "public_benchmark_anchor_snapshot.v1"
HARNESS_VALIDATION_SCHEMA_VERSION = "harness_validation_scope.v1"
EXTERNAL_REFERENCE_SCHEMA_VERSION = "external_reference_results.v1"

HARNESS_VALIDATION_SCOPE = "harness_validation_public_anchor"

# Public real-world benchmarks whose published outcomes may back a harness
# validation.  Each entry records what the benchmark actually measures, so a
# result cannot silently be read as evidence about a different embodiment.
PUBLIC_ANCHOR_REGISTRY: dict[str, dict[str, Any]] = {
    "roboarena": {
        "benchmark_id": "roboarena",
        "display_name": "RoboArena",
        "outcome_kind": "real_robot",
        "embodiment_family": "droid_franka_single_arm",
        "action_schema_family": "end_effector_cartesian",
        "task_family": "tabletop_manipulation",
        "site_alignment": "aggregate_only",
        "distributed_evaluation": True,
        "notes": (
            "distributed real-robot evaluation over the DROID platform; "
            "academic leaderboard, not a buyer-facing reference cell"
        ),
    },
    "droid_public_eval": {
        "benchmark_id": "droid_public_eval",
        "display_name": "DROID public evaluation",
        "outcome_kind": "real_robot",
        "embodiment_family": "droid_franka_single_arm",
        "action_schema_family": "end_effector_cartesian",
        "task_family": "tabletop_manipulation",
        "site_alignment": "aggregate_only",
        "distributed_evaluation": True,
        "notes": "published real-robot outcomes on the open DROID platform",
    },
}

MIN_ANCHOR_POLICY_COUNT = 3


def _string(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _rows(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [dict(item) for item in value if isinstance(item, Mapping)]
    return []


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _integer(value: Any, *, minimum: int = 0) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        return None
    return value


def _digest(value: Any) -> str:
    text = _string(value).lower().removeprefix("sha256:")
    return text if len(text) == 64 and all(c in "0123456789abcdef" for c in text) else ""


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_anchor_snapshot(
    *,
    benchmark_id: str,
    source_uri: str,
    retrieved_at: str,
    policy_results: Sequence[Mapping[str, Any]],
    acceptance: Mapping[str, Any] | None = None,
    task_mapping: Mapping[str, Any] | None = None,
    terms: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Canonicalise a public leaderboard export into a digest-pinned snapshot.

    Every policy row must carry a resolved ``checkpoint_sha256``.  A public
    leaderboard names policies; it does not identify weights, and an anchor whose
    rows cannot be tied to exact checkpoints cannot support the exact-match join
    the fidelity report performs.  For open policies this is satisfiable, so it
    is required rather than inferred.
    """

    blockers: list[str] = []
    registry_entry = PUBLIC_ANCHOR_REGISTRY.get(_string(benchmark_id))
    if registry_entry is None:
        blockers.append("public_anchor_benchmark_not_registered")
        registry_entry = {}
    if not _string(source_uri).startswith("https://"):
        blockers.append("public_anchor_source_uri_missing_or_not_https")
    if not _string(retrieved_at):
        blockers.append("public_anchor_retrieved_at_missing")

    normalized: list[dict[str, Any]] = []
    seen_policies: set[str] = set()
    seen_checkpoints: set[str] = set()
    for index, row in enumerate(_rows(policy_results)):
        policy_id = _string(row.get("policy_id"))
        checkpoint = _digest(row.get("checkpoint_sha256"))
        score = _number(row.get("success_rate"))
        trials = _integer(row.get("trial_count"), minimum=1)
        if not policy_id:
            blockers.append(f"public_anchor_policy_id_missing:{index}")
            continue
        if not checkpoint:
            blockers.append(f"public_anchor_checkpoint_digest_missing:{policy_id}")
            continue
        if score is None or not 0.0 <= score <= 1.0:
            blockers.append(f"public_anchor_success_rate_invalid:{policy_id}")
            continue
        if trials is None:
            blockers.append(f"public_anchor_trial_count_missing:{policy_id}")
            continue
        if policy_id in seen_policies:
            blockers.append(f"public_anchor_duplicate_policy:{policy_id}")
            continue
        if checkpoint in seen_checkpoints:
            # Two leaderboard rows resolving to one checkpoint would inflate the
            # apparent cohort size without adding a degree of freedom.
            blockers.append(f"public_anchor_duplicate_checkpoint:{policy_id}")
            continue
        seen_policies.add(policy_id)
        seen_checkpoints.add(checkpoint)
        normalized.append(
            {
                "policy_id": policy_id,
                "checkpoint_sha256": checkpoint,
                "success_rate": round(score, 6),
                "trial_count": trials,
                "policy_source_uri": _string(row.get("policy_source_uri")) or None,
            }
        )

    if len(normalized) < MIN_ANCHOR_POLICY_COUNT:
        blockers.append("public_anchor_requires_three_resolved_policies")

    acceptance_row = _mapping(acceptance)
    accepted = (
        acceptance_row.get("independently_accepted") is True
        and bool(_string(acceptance_row.get("accepted_by")))
        and bool(_string(acceptance_row.get("accepted_at")))
        and bool(_digest(acceptance_row.get("source_artifact_sha256")))
    )
    if not accepted:
        # The module never asserts independence on the operator's behalf.
        blockers.append("public_anchor_acceptance_record_incomplete")

    mapping_row = _mapping(task_mapping)
    task_mapping_sha256 = canonical_sha256(mapping_row) if mapping_row else ""
    if not mapping_row:
        blockers.append("public_anchor_task_mapping_missing")

    terms_row = _mapping(terms)
    if not _string(terms_row.get("usage_terms_uri")):
        blockers.append("public_anchor_usage_terms_missing")

    core = {
        "benchmark_id": _string(benchmark_id),
        "source_uri": _string(source_uri),
        "retrieved_at": _string(retrieved_at),
        "policy_results": normalized,
        "task_mapping_sha256": task_mapping_sha256 or None,
    }
    snapshot_sha256 = canonical_sha256(core)

    blockers = sorted(set(blockers))
    return {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "generated_at": utc_now_iso(),
        "status": "snapshot_ready" if not blockers else "blocked",
        "benchmark": dict(registry_entry),
        **core,
        "policy_count": len(normalized),
        "total_trial_count": sum(row["trial_count"] for row in normalized),
        # This is the digest the RoboWorld admission checklist requires as
        # `roboarena_snapshot_sha256`; it had no producer before.
        "snapshot_sha256": snapshot_sha256,
        "acceptance": acceptance_row or None,
        "usage_terms": terms_row or None,
        "blockers": blockers,
        "claim_boundary": {
            "snapshot_is_third_party_published_outcomes": True,
            "snapshot_is_not_a_blueprint_measurement": True,
            "snapshot_is_not_a_site_specific_anchor": True,
            "public_claim_upgrade_allowed": False,
        },
    }


def build_external_reference_results(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    """Convert an accepted snapshot into ``external_reference_results.v1``.

    This is the producer for the schema the external rank-fidelity report
    consumes.  ``site_alignment`` is taken from the benchmark registry rather
    than from the caller, so a distributed academic leaderboard cannot be
    presented as a same-site reference.
    """

    blockers: list[str] = []
    if snapshot.get("schema_version") != SNAPSHOT_SCHEMA_VERSION:
        blockers.append("public_anchor_snapshot_schema_missing_or_unsupported")
    if snapshot.get("status") != "snapshot_ready":
        blockers.append("public_anchor_snapshot_not_ready")

    benchmark = _mapping(snapshot.get("benchmark"))
    outcome_kind = _string(benchmark.get("outcome_kind"))
    site_alignment = _string(benchmark.get("site_alignment"))
    if outcome_kind not in {"real_robot", "simulator", "world_model"}:
        blockers.append("public_anchor_outcome_kind_invalid")
    if site_alignment not in {"same_site", "different_site", "aggregate_only"}:
        blockers.append("public_anchor_site_alignment_invalid")
    if site_alignment == "same_site":
        # A public distributed benchmark is never same-site with a captured
        # customer facility, whatever the registry says.
        blockers.append("public_anchor_may_not_claim_same_site_alignment")

    acceptance = _mapping(snapshot.get("acceptance"))
    source_artifact_sha256 = _digest(acceptance.get("source_artifact_sha256"))
    task_mapping_sha256 = _digest(snapshot.get("task_mapping_sha256"))
    if not source_artifact_sha256:
        blockers.append("public_anchor_source_artifact_digest_missing")
    if not task_mapping_sha256:
        blockers.append("public_anchor_task_mapping_digest_missing")

    policy_results = [
        {
            "policy_id": row.get("policy_id"),
            "checkpoint_sha256": row.get("checkpoint_sha256"),
            "score": row.get("success_rate"),
            "trial_count": row.get("trial_count"),
        }
        for row in _rows(snapshot.get("policy_results"))
    ]
    if len(policy_results) < MIN_ANCHOR_POLICY_COUNT:
        blockers.append("public_anchor_requires_three_resolved_policies")

    blockers = sorted(set(blockers))
    return {
        "schema_version": EXTERNAL_REFERENCE_SCHEMA_VERSION,
        "status": "ready" if not blockers else "blocked",
        "reference_id": f"public_anchor:{_string(snapshot.get('benchmark_id'))}",
        "reference_type": outcome_kind or None,
        "site_alignment": site_alignment or None,
        "independently_accepted": not blockers,
        "source_uri": snapshot.get("source_uri"),
        "source_artifact_sha256": source_artifact_sha256 or None,
        "task_mapping_sha256": task_mapping_sha256 or None,
        "snapshot_sha256": snapshot.get("snapshot_sha256"),
        "policy_results": policy_results,
        "blockers": blockers,
        "claim_boundary": {
            "reference_is_public_third_party_outcomes": True,
            "reference_is_not_site_specific": True,
            "scope": HARNESS_VALIDATION_SCOPE,
            "public_claim_upgrade_allowed": False,
        },
    }


def build_harness_validation_scope(
    *,
    snapshot: Mapping[str, Any],
    fidelity_report: Mapping[str, Any],
    customer_embodiment_id: str | None = None,
    customer_site_id: str | None = None,
) -> dict[str, Any]:
    """Bound what a public-anchor fidelity result is allowed to mean.

    The result is scoped to the harness.  It says the scoring, aggregation and
    ranking machinery reproduces an independently measured real-world ordering on
    the benchmark's own embodiment and task family.  It says nothing about a
    customer's robot, site, or task, and this record makes that explicit and
    machine-checkable rather than leaving it to prose.
    """

    benchmark = _mapping(snapshot.get("benchmark"))
    report_status = _string(fidelity_report.get("status"))
    measured = report_status == "measured"
    headline = _mapping(fidelity_report.get("headline"))

    transfer_blockers: list[str] = []
    if _string(customer_embodiment_id) and _string(customer_embodiment_id) != _string(
        benchmark.get("embodiment_family")
    ):
        transfer_blockers.append("public_anchor_embodiment_differs_from_customer")
    if _string(customer_site_id):
        transfer_blockers.append("public_anchor_site_differs_from_customer")

    return {
        "schema_version": HARNESS_VALIDATION_SCHEMA_VERSION,
        "generated_at": utc_now_iso(),
        "scope": HARNESS_VALIDATION_SCOPE,
        "status": "harness_validated" if measured else "not_validated",
        "benchmark_id": snapshot.get("benchmark_id"),
        "snapshot_sha256": snapshot.get("snapshot_sha256"),
        "embodiment_family": benchmark.get("embodiment_family"),
        "action_schema_family": benchmark.get("action_schema_family"),
        "task_family": benchmark.get("task_family"),
        "policy_cohort_size": len(_rows(snapshot.get("policy_results"))),
        "headline_metric": headline.get("metric"),
        "headline_value": headline.get("value"),
        "headline_interval_95": headline.get("interval_95"),
        "fidelity_report_status": report_status or None,
        "transfer_blockers": sorted(set(transfer_blockers)),
        "what_this_establishes": (
            "the evaluation harness reproduces an independently measured "
            "real-world policy ordering on the benchmark's own embodiment, "
            "site distribution, and task family"
        ),
        "what_this_does_not_establish": [
            "site_specific_rank_fidelity_for_any_customer_facility",
            "rank_fidelity_for_any_other_embodiment_or_action_schema",
            "physical_task_success_safety_or_deployment_readiness",
            "world_model_quality_independent_of_the_scoring_harness",
        ],
        "claim_boundary": {
            "harness_validation_is_not_site_specific_rank_fidelity": True,
            "harness_validation_does_not_substitute_for_customer_anchors": True,
            "public_rank_fidelity_claim_eligible": False,
            "public_claim_upgrade_allowed": False,
        },
    }


def _command_snapshot(args: argparse.Namespace) -> int:
    payload = _mapping(read_json_any(Path(args.input)))
    snapshot = build_anchor_snapshot(
        benchmark_id=_string(payload.get("benchmark_id")),
        source_uri=_string(payload.get("source_uri")),
        retrieved_at=_string(payload.get("retrieved_at")),
        policy_results=payload.get("policy_results", []) or [],
        acceptance=_mapping(payload.get("acceptance")),
        task_mapping=_mapping(payload.get("task_mapping")),
        terms=_mapping(payload.get("usage_terms")),
    )
    write_json(Path(args.output), snapshot)
    print(
        json.dumps(
            {
                "path": args.output,
                "status": snapshot["status"],
                "snapshot_sha256": snapshot["snapshot_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0 if snapshot["status"] == "snapshot_ready" else 1


def _command_reference(args: argparse.Namespace) -> int:
    snapshot = _mapping(read_json_any(Path(args.input)))
    reference = build_external_reference_results(snapshot)
    write_json(Path(args.output), reference)
    print(json.dumps({"path": args.output, "status": reference["status"]}, sort_keys=True))
    return 0 if reference["status"] == "ready" else 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Produce public real-world benchmark anchors for harness validation"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    snapshot = sub.add_parser("snapshot", help="canonicalise a leaderboard export")
    snapshot.add_argument("--input", required=True)
    snapshot.add_argument("--output", required=True)
    snapshot.set_defaults(func=_command_snapshot)

    reference = sub.add_parser(
        "external-reference", help="emit external_reference_results.v1 from a snapshot"
    )
    reference.add_argument("--input", required=True)
    reference.add_argument("--output", required=True)
    reference.set_defaults(func=_command_reference)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
