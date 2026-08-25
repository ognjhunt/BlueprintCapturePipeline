from __future__ import annotations

import json
from pathlib import Path

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.project_spend_reconciliation import (
    materialize_project_spend_reconciliation,
    validate_project_spend_reconciliation,
)


def _authority(
    path: Path,
    *,
    aggregate_before: float = 40.333914,
    hard_cap: float = 0.75,
) -> tuple[Path, dict[str, object]]:
    value: dict[str, object] = {
        "schema_version": "native_task_arena_paid_attempt_authority.v1",
        "provider": "vast",
        "paid_compute_authorized": True,
        "maximum_paid_attempts": 1,
        "maximum_provider_allocations": 1,
        "maximum_automatic_retries": 0,
        "automatic_paid_retry_authorized": False,
        "hard_attempt_spend_cap_usd": hard_cap,
        "aggregate_goal_spend_before_attempt_usd": aggregate_before,
        "authorized_on": "2026-08-25T14:30:00+00:00",
        "authorization_digest": "",
    }
    value["authorization_digest"] = canonical_digest(
        value, digest_field="authorization_digest"
    )
    path.write_text(json.dumps(value), encoding="utf-8")
    return path, value


def test_unposted_attempt_is_counted_at_full_cap_and_reopened(tmp_path: Path) -> None:
    authority_path, authority = _authority(tmp_path / "authority.json")
    output = tmp_path / "project-spend.json"

    receipt = materialize_project_spend_reconciliation(
        baseline_authority_path=authority_path,
        posted_reconciliation_paths=[],
        unposted_authority_paths=[authority_path],
        expected_coverage_ids=[str(authority["authorization_digest"])],
        completeness_reference="fresh authenticated inventory and retained launch queue",
        authorized_by="user",
        authorized_on="2026-08-25T15:15:00+00:00",
        output_path=output,
    )

    assert receipt["baseline_total_cost_usd"] == 40.333914
    assert receipt["unposted_full_cap_total_usd"] == 0.75
    assert receipt["total_cost_usd"] == 41.083914
    assert receipt["continuing_spend_conservatively_counted_at_full_cap"] is True
    reopened, record = validate_project_spend_reconciliation(
        output, expected_total_cost_usd=41.083914
    )
    assert reopened == receipt
    assert record["unposted_authority_count"] == 1


def test_expected_attempt_set_must_exactly_cover_post_baseline_work(
    tmp_path: Path,
) -> None:
    authority_path, _ = _authority(tmp_path / "authority.json")

    with pytest.raises(
        ValueError, match="project_spend_expected_attempt_coverage_mismatch"
    ):
        materialize_project_spend_reconciliation(
            baseline_authority_path=authority_path,
            posted_reconciliation_paths=[],
            unposted_authority_paths=[authority_path],
            expected_coverage_ids=["wrong-attempt"],
            completeness_reference="retained queue inventory",
            authorized_by="user",
            authorized_on="2026-08-25T15:15:00+00:00",
            output_path=tmp_path / "project-spend.json",
        )


def test_reopen_rejects_changed_authority_bytes(tmp_path: Path) -> None:
    authority_path, authority = _authority(tmp_path / "authority.json")
    output = tmp_path / "project-spend.json"
    materialize_project_spend_reconciliation(
        baseline_authority_path=authority_path,
        posted_reconciliation_paths=[],
        unposted_authority_paths=[authority_path],
        expected_coverage_ids=[str(authority["authorization_digest"])],
        completeness_reference="retained queue inventory",
        authorized_by="user",
        authorized_on="2026-08-25T15:15:00+00:00",
        output_path=output,
    )
    authority_path.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="project_spend_baseline_invalid"):
        validate_project_spend_reconciliation(output)


def test_lane_local_receipt_cannot_authorize_a_new_project_lane(tmp_path: Path) -> None:
    old_lane_local = {
        "schema_version": "adp_same_goal_spend_reconciliation.v1",
        "status": "all_same_goal_paid_attempts_terminal_and_provider_zero",
        "goal_id": "arm-decision-proof-v1",
        "entries": [],
        "entry_count": 0,
        "total_cost_usd": 40.333914,
        "receipt_digest": "",
    }
    old_lane_local["receipt_digest"] = canonical_digest(
        old_lane_local, digest_field="receipt_digest"
    )
    path = tmp_path / "lane-local.json"
    path.write_text(json.dumps(old_lane_local), encoding="utf-8")

    with pytest.raises(ValueError, match="project_spend_reconciliation_invalid"):
        validate_project_spend_reconciliation(path)
