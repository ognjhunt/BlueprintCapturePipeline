"""Tests for public real-world benchmark anchors and harness validation."""

from __future__ import annotations

import hashlib

import pytest

from blueprint_pipeline import public_benchmark_anchor as anchor
from blueprint_pipeline.benchmark_protocol import build_external_rank_fidelity_report


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


POLICIES = ("openvla", "pi_zero", "octo_base", "rdt_1b", "smolvla", "gr00t_n1", "dp_baseline", "act_baseline")
TRUE_RATES = (0.62, 0.58, 0.44, 0.41, 0.33, 0.29, 0.18, 0.11)


def _policy_results():
    return [
        {
            "policy_id": policy,
            "checkpoint_sha256": _digest(policy),
            "success_rate": rate,
            "trial_count": 120,
            "policy_source_uri": f"https://example.org/{policy}",
        }
        for policy, rate in zip(POLICIES, TRUE_RATES)
    ]


def _snapshot(**overrides):
    kwargs = {
        "benchmark_id": "roboarena",
        "source_uri": "https://robo-arena.github.io/leaderboard",
        "retrieved_at": "2026-07-24T00:00:00Z",
        "policy_results": _policy_results(),
        "acceptance": {
            "independently_accepted": True,
            "accepted_by": "blueprint-evaluation-authority",
            "accepted_at": "2026-07-24T00:00:00Z",
            "source_artifact_sha256": _digest("leaderboard-export"),
        },
        "task_mapping": {"tabletop_manipulation": "blueprint_manipulation_v1"},
        "terms": {"usage_terms_uri": "https://robo-arena.github.io/terms"},
    }
    kwargs.update(overrides)
    return anchor.build_anchor_snapshot(**kwargs)


def test_snapshot_produces_the_digest_the_admission_checklist_requires() -> None:
    """`roboarena_snapshot_sha256` had no producer anywhere in the repo."""

    snapshot = _snapshot()

    assert snapshot["status"] == "snapshot_ready", snapshot["blockers"]
    assert len(snapshot["snapshot_sha256"]) == 64
    assert snapshot["policy_count"] == 8
    assert snapshot["total_trial_count"] == 960


def test_snapshot_digest_is_deterministic_and_content_bound() -> None:
    first = _snapshot()
    second = _snapshot()
    assert first["snapshot_sha256"] == second["snapshot_sha256"]

    mutated = list(_policy_results())
    mutated[0] = {**mutated[0], "success_rate": 0.99}
    changed = _snapshot(policy_results=mutated)
    assert changed["snapshot_sha256"] != first["snapshot_sha256"]


def test_snapshot_requires_resolved_checkpoint_digests() -> None:
    rows = [{**row} for row in _policy_results()]
    rows[0].pop("checkpoint_sha256")

    snapshot = _snapshot(policy_results=rows)

    assert snapshot["status"] == "blocked"
    assert any(
        item.startswith("public_anchor_checkpoint_digest_missing")
        for item in snapshot["blockers"]
    )


def test_snapshot_refuses_to_assert_independence_on_the_operators_behalf() -> None:
    snapshot = _snapshot(acceptance={"independently_accepted": True})

    assert snapshot["status"] == "blocked"
    assert "public_anchor_acceptance_record_incomplete" in snapshot["blockers"]


def test_duplicate_checkpoints_do_not_inflate_the_cohort() -> None:
    rows = [{**row} for row in _policy_results()]
    rows[1]["checkpoint_sha256"] = rows[0]["checkpoint_sha256"]

    snapshot = _snapshot(policy_results=rows)

    assert snapshot["status"] == "blocked"
    assert any(
        item.startswith("public_anchor_duplicate_checkpoint") for item in snapshot["blockers"]
    )


def test_unregistered_benchmark_is_refused() -> None:
    snapshot = _snapshot(benchmark_id="my_private_leaderboard")
    assert "public_anchor_benchmark_not_registered" in snapshot["blockers"]


def test_external_reference_producer_feeds_the_real_fidelity_report() -> None:
    """The schema had a constant and no producer; this closes the loop."""

    snapshot = _snapshot()
    reference = anchor.build_external_reference_results(snapshot)

    assert reference["status"] == "ready", reference["blockers"]
    assert reference["schema_version"] == "external_reference_results.v1"
    assert reference["independently_accepted"] is True

    # A near-perfect evaluator over the same cohort.
    aggregates = [
        {
            "policy_id": policy,
            "metrics": {"full_task_success": {"estimate": rate + 0.02}},
        }
        for policy, rate in zip(POLICIES, TRUE_RATES)
    ]
    registry = [
        {"policy_id": policy, "checkpoint_sha256": _digest(policy)} for policy in POLICIES
    ]

    report = build_external_rank_fidelity_report(
        reference=reference,
        policy_aggregates=aggregates,
        policy_registry=registry,
        seed=7,
    )

    assert report["status"] == "measured", report["blockers"]
    assert len(report["matched_policies"]) == 8
    assert report["headline"]["metric"] == "pairwise_ordering_accuracy"
    assert report["headline"]["value"] == pytest.approx(1.0)
    # A public leaderboard is never a same-site real-robot anchor.
    assert report["claim_boundary"]["rank_fidelity_result_proven"] is False
    assert report["measurement_scope"] == "cross_site_real_robot_rank_concordance"


def test_public_anchor_cannot_claim_same_site_alignment(monkeypatch) -> None:
    monkeypatch.setitem(
        anchor.PUBLIC_ANCHOR_REGISTRY,
        "roboarena",
        {**anchor.PUBLIC_ANCHOR_REGISTRY["roboarena"], "site_alignment": "same_site"},
    )
    snapshot = _snapshot()
    reference = anchor.build_external_reference_results(snapshot)

    assert reference["status"] == "blocked"
    assert "public_anchor_may_not_claim_same_site_alignment" in reference["blockers"]


def test_harness_validation_scope_refuses_to_transfer_to_a_customer() -> None:
    snapshot = _snapshot()
    reference = anchor.build_external_reference_results(snapshot)
    aggregates = [
        {"policy_id": policy, "metrics": {"full_task_success": {"estimate": rate}}}
        for policy, rate in zip(POLICIES, TRUE_RATES)
    ]
    registry = [
        {"policy_id": policy, "checkpoint_sha256": _digest(policy)} for policy in POLICIES
    ]
    report = build_external_rank_fidelity_report(
        reference=reference,
        policy_aggregates=aggregates,
        policy_registry=registry,
        seed=7,
    )

    scope = anchor.build_harness_validation_scope(
        snapshot=snapshot,
        fidelity_report=report,
        customer_embodiment_id="unitree_g1_whole_body",
        customer_site_id="site-0001",
    )

    assert scope["status"] == "harness_validated"
    assert scope["scope"] == anchor.HARNESS_VALIDATION_SCOPE
    assert scope["claim_boundary"]["public_rank_fidelity_claim_eligible"] is False
    assert (
        "public_anchor_embodiment_differs_from_customer" in scope["transfer_blockers"]
    )
    assert "public_anchor_site_differs_from_customer" in scope["transfer_blockers"]
    assert (
        "site_specific_rank_fidelity_for_any_customer_facility"
        in scope["what_this_does_not_establish"]
    )


def test_blocked_fidelity_report_does_not_validate_the_harness() -> None:
    snapshot = _snapshot()
    scope = anchor.build_harness_validation_scope(
        snapshot=snapshot, fidelity_report={"status": "blocked"}
    )
    assert scope["status"] == "not_validated"


def test_eight_policy_cohort_reports_an_honest_interval_width() -> None:
    """The cohort size, not the trial count, bounds the correlation interval."""

    snapshot = _snapshot()
    reference = anchor.build_external_reference_results(snapshot)
    # Deliberately imperfect: a correlation of exactly 1.0 sits on the unit
    # boundary where the Fisher transform is undefined, which is itself the
    # correct behaviour but is not what this test is about.
    wobble = (0.03, -0.02, 0.04, -0.03, 0.02, -0.04, 0.03, -0.02)
    aggregates = [
        {
            "policy_id": policy,
            "metrics": {"full_task_success": {"estimate": rate + offset}},
        }
        for policy, rate, offset in zip(POLICIES, TRUE_RATES, wobble)
    ]
    registry = [
        {"policy_id": policy, "checkpoint_sha256": _digest(policy)} for policy in POLICIES
    ]
    report = build_external_rank_fidelity_report(
        reference=reference,
        policy_aggregates=aggregates,
        policy_registry=registry,
        seed=7,
    )

    pearson = report["metrics"]["pearson"]
    assert pearson["metric_role"] == "supporting_fragile_at_small_cohorts"
    interval = pearson["fisher_z_interval_95"]
    assert interval["defined"] is True
    assert interval["sample_count"] == 8
    # 960 real trials back these rows, but the correlation still rests on 8
    # points, so the lower bound must stay meaningfully below the estimate.
    assert interval["lower"] < interval["estimate"]
