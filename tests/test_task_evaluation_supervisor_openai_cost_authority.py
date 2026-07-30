from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import pytest

from blueprint_pipeline.task_evaluation_supervisor.openai_cost_authority import (
    OPENAI_COST_SNAPSHOT_SCHEMA_VERSION,
    OPENAI_COST_SCOPE_ATTESTATION_SCHEMA_VERSION,
    OpenAICostAuthorityError,
    OpenAIOrganizationCostsClient,
    OpenAIProjectCandidateCostAuthority,
    openai_cost_authority_binding_digest,
)
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.common import write_json
from blueprint_pipeline.task_evaluation_supervisor.candidate_policy import (
    CANDIDATE_EVALUATION_EXECUTION_SCHEMA_VERSION,
    reconcile_neutral_candidate_policy_costs,
)


SHA_A = "sha256:" + "a" * 64
SHA_B = "sha256:" + "b" * 64
SHA_C = "sha256:" + "c" * 64


def _admin_key(path: Path) -> Path:
    path.write_text("sk-admin-hermetic-fixture", encoding="utf-8")
    path.chmod(0o600)
    return path


def _page(*, start: int, end: int, cost: float, has_more: bool, next_page=None):
    return {
        "object": "page",
        "data": [
            {
                "object": "bucket",
                "start_time": start,
                "end_time": end,
                "results": [
                    {
                        "object": "organization.costs.result",
                        "amount": {"value": cost, "currency": "usd"},
                        "project_id": "proj_pigey_eval",
                        "api_key_id": "key_pigey_eval",
                    }
                ],
            }
        ],
        "has_more": has_more,
        "next_page": next_page,
    }


def _scope_attestation(
    *,
    issued_by_agent: bool = False,
    exclusive_until: str = "2026-08-02T00:00:00Z",
) -> dict[str, Any]:
    value = {
        "schema_version": OPENAI_COST_SCOPE_ATTESTATION_SCHEMA_VERSION,
        "status": "approved",
        "operator_id": "independent-cost-owner",
        "issued_by_agent": issued_by_agent,
        "provider_id": "pigey_external_candidate",
        "paid_resource_class": "openai_api_candidate",
        "project_id": "proj_pigey_eval",
        "api_key_id": "key_pigey_eval",
        "exclusive_use": True,
        "exclusive_from": "2026-07-30T09:00:00Z",
        "exclusive_until": exclusive_until,
        "candidate_reported_usage_is_authoritative": False,
        "proof_effect": "none",
    }
    value["scope_attestation_digest"] = canonical_digest(
        value,
        digest_field="scope_attestation_digest",
    )
    return value


def test_openai_cost_client_uses_exact_project_key_filters_and_paginates(
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, dict[str, str]]] = []

    def transport(url: str, headers, _timeout: float):
        calls.append((url, dict(headers)))
        query = parse_qs(urlparse(url).query)
        if "page" not in query:
            return _page(start=100, end=150, cost=0.1, has_more=True, next_page="p2")
        assert query["page"] == ["p2"]
        return _page(start=150, end=200, cost=0.2, has_more=False)

    client = OpenAIOrganizationCostsClient(
        project_id="proj_pigey_eval",
        api_key_id="key_pigey_eval",
        admin_api_key_file=_admin_key(tmp_path / "admin-key"),
        transport=transport,
        wall_clock=lambda: datetime(2026, 7, 30, 10, tzinfo=timezone.utc),
    )
    snapshot = client.snapshot(start_time=100, end_time=200)

    assert snapshot["schema_version"] == OPENAI_COST_SNAPSHOT_SCHEMA_VERSION
    assert snapshot["status"] == "complete"
    assert snapshot["total_cost_usd"] == pytest.approx(0.3)
    assert snapshot["page_count"] == 2
    assert snapshot["result_count"] == 2
    assert snapshot["raw_admin_key_recorded"] is False
    assert len(calls) == 2
    first_query = parse_qs(urlparse(calls[0][0]).query)
    assert first_query["project_ids"] == ["proj_pigey_eval"]
    assert first_query["api_key_ids"] == ["key_pigey_eval"]
    assert sorted(first_query["group_by"]) == ["api_key_id", "project_id"]
    assert calls[0][1]["Authorization"] == "Bearer sk-admin-hermetic-fixture"
    assert "sk-admin-hermetic-fixture" not in str(snapshot)


def test_openai_cost_client_rejects_scope_drift_and_weak_key_file(
    tmp_path: Path,
) -> None:
    key = _admin_key(tmp_path / "admin-key")
    key.chmod(0o644)
    client = OpenAIOrganizationCostsClient(
        project_id="proj_pigey_eval",
        api_key_id="key_pigey_eval",
        admin_api_key_file=key,
        transport=lambda *_args: _page(
            start=100,
            end=200,
            cost=0.1,
            has_more=False,
        ),
    )
    with pytest.raises(
        OpenAICostAuthorityError,
        match="openai_admin_key_file_permissions_too_open",
    ):
        client.snapshot(start_time=100, end_time=200)

    key.chmod(0o600)
    wrong_scope = _page(start=100, end=200, cost=0.1, has_more=False)
    wrong_scope["data"][0]["results"][0]["api_key_id"] = "key_other"
    client.transport = lambda *_args: wrong_scope
    with pytest.raises(OpenAICostAuthorityError, match="openai_cost_scope_mismatch"):
        client.snapshot(start_time=100, end_time=200)

    target = _admin_key(tmp_path / "symlink-target")
    link = tmp_path / "admin-key-link"
    link.symlink_to(target)
    symlink_client = OpenAIOrganizationCostsClient(
        project_id="proj_pigey_eval",
        api_key_id="key_pigey_eval",
        admin_api_key_file=link,
        transport=lambda *_args: wrong_scope,
    )
    with pytest.raises(
        OpenAICostAuthorityError,
        match="openai_admin_key_file_missing_or_symlink",
    ):
        symlink_client.snapshot(start_time=100, end_time=200)


class _SnapshotClient:
    project_id = "proj_pigey_eval"
    api_key_id = "key_pigey_eval"

    def __init__(self) -> None:
        self.costs = [0.0, 0.25]
        self.calls: list[tuple[int, int]] = []

    def snapshot(self, *, start_time: int, end_time: int) -> dict[str, Any]:
        self.calls.append((start_time, end_time))
        cost = self.costs.pop(0)
        return {
            "schema_version": OPENAI_COST_SNAPSHOT_SCHEMA_VERSION,
            "status": "complete",
            "project_id": self.project_id,
            "api_key_id": self.api_key_id,
            "start_time": start_time,
            "end_time": end_time,
            "total_cost_usd": cost,
            "openai_cost_snapshot_digest": SHA_C,
        }


def test_openai_cost_authority_reserves_zero_baseline_then_waits_for_provider_costs() -> None:
    current = [datetime(2026, 7, 30, 10, tzinfo=timezone.utc)]
    client = _SnapshotClient()
    authority = OpenAIProjectCandidateCostAuthority(
        client=client,  # type: ignore[arg-type]
        scope_attestation=_scope_attestation(),
        attribution_window_seconds=3_600,
        reconciliation_delay_seconds=86_400,
        wall_clock=lambda: current[0],
    )
    reservation = authority.reserve(
        candidate_id="pigey-verify-recover",
        candidate_evaluation_suite_digest=SHA_B,
        authorization_receipt_digest=SHA_C,
        max_cost_usd=1.0,
    )

    assert reservation["status"] == "reserved"
    assert reservation["baseline_cost_snapshot"]["total_cost_usd"] == 0.0
    assert reservation["candidate_reported_usage_is_authoritative"] is False
    assert reservation["cost_authority_binding_digest"] == (authority.cost_authority_binding_digest)

    pending = authority.settle(
        reservation=reservation,
        runtime_result={"status": "completed", "cost_usd": 999.0},
        runtime_exception_type=None,
    )
    assert pending["status"] == "reconciliation_required"
    assert pending["actual_cost_usd"] is None
    assert pending["cost_is_final"] is False
    assert pending["candidate_reported_cost_accepted"] is False
    assert pending["reconciliation_blocker"] == "openai_cost_reporting_window_open"
    assert len(client.calls) == 1

    current[0] = datetime(2026, 8, 1, 0, 1, tzinfo=timezone.utc)
    reconciled = authority.settle(
        reservation=reservation,
        runtime_result=None,
        runtime_exception_type="RuntimeError",
    )
    assert reconciled["status"] == "reconciled"
    assert reconciled["actual_cost_usd"] == pytest.approx(0.25)
    assert reconciled["cost_is_final"] is True
    assert reconciled["runtime_exception_type"] == "RuntimeError"
    assert reconciled["candidate_reported_cost_accepted"] is False
    assert len(client.calls) == 2


def test_openai_cost_authority_rejects_agent_scope_grant_and_short_window() -> None:
    client = _SnapshotClient()
    with pytest.raises(
        OpenAICostAuthorityError,
        match="openai_cost_scope_attestation_invalid",
    ):
        OpenAIProjectCandidateCostAuthority(
            client=client,  # type: ignore[arg-type]
            scope_attestation=_scope_attestation(issued_by_agent=True),
        )

    authority = OpenAIProjectCandidateCostAuthority(
        client=client,  # type: ignore[arg-type]
        scope_attestation=_scope_attestation(exclusive_until="2026-07-30T10:30:00Z"),
        wall_clock=lambda: datetime(2026, 7, 30, 10, tzinfo=timezone.utc),
    )
    with pytest.raises(
        OpenAICostAuthorityError,
        match="openai_cost_scope_attestation_window_insufficient",
    ):
        authority.reserve(
            candidate_id="pigey-verify-recover",
            candidate_evaluation_suite_digest=SHA_B,
            authorization_receipt_digest=SHA_C,
            max_cost_usd=1.0,
        )


def test_openai_cost_binding_changes_with_key_project_or_attestation() -> None:
    baseline = openai_cost_authority_binding_digest(
        provider_id="pigey_external_candidate",
        paid_resource_class="openai_api_candidate",
        project_id="proj_pigey_eval",
        api_key_id="key_pigey_eval",
        scope_attestation_digest=SHA_A,
    )
    changed = openai_cost_authority_binding_digest(
        provider_id="pigey_external_candidate",
        paid_resource_class="openai_api_candidate",
        project_id="proj_pigey_eval",
        api_key_id="key_other",
        scope_attestation_digest=SHA_A,
    )
    assert baseline.startswith("sha256:")
    assert baseline != changed


def test_delayed_cost_reconciliation_closes_without_rerunning_candidate(
    tmp_path: Path,
) -> None:
    current = [datetime(2026, 7, 30, 10, tzinfo=timezone.utc)]
    client = _SnapshotClient()
    authority = OpenAIProjectCandidateCostAuthority(
        client=client,  # type: ignore[arg-type]
        scope_attestation=_scope_attestation(),
        wall_clock=lambda: current[0],
    )
    reservation = authority.reserve(
        candidate_id="pigey-verify-recover",
        candidate_evaluation_suite_digest=SHA_B,
        authorization_receipt_digest=SHA_C,
        max_cost_usd=1.0,
    )
    cost_dir = tmp_path / "candidates" / "pigey-verify-recover" / "cost_authority"
    write_json(cost_dir / "reservation.json", reservation)
    execution = {
        "schema_version": CANDIDATE_EVALUATION_EXECUTION_SCHEMA_VERSION,
        "status": "partial",
        "candidate_evaluation_suite_digest": SHA_B,
        "execution_started": True,
        "candidate_results": [
            {
                "candidate_id": "pigey-verify-recover",
                "status": "evaluated",
                "proof_effect": "none",
            }
        ],
        "authorization_receipt_digest": SHA_C,
        "authorized_max_cost_usd": 1.0,
        "reported_cost_usd": 0.0,
        "reported_cost_is_final": False,
        "cost_reconciliation_required_candidate_ids": ["pigey-verify-recover"],
        "proof_effect": "none",
    }
    execution["candidate_evaluation_execution_digest"] = canonical_digest(
        execution,
        digest_field="candidate_evaluation_execution_digest",
    )
    write_json(tmp_path / "candidate_evaluation_execution.json", execution)

    current[0] = datetime(2026, 8, 1, 0, 1, tzinfo=timezone.utc)
    report = reconcile_neutral_candidate_policy_costs(
        tmp_path,
        candidate_cost_authorities=[authority],
    )

    assert report["status"] == "reconciled"
    assert report["reported_cost_usd"] == pytest.approx(0.25)
    assert report["reported_cost_is_final"] is True
    assert report["candidate_execution_repeated"] is False
    assert report["candidate_evaluation_repeated"] is False
    assert len(client.calls) == 2
    reconciliation_files = list((cost_dir / "reconciliations").glob("*.json"))
    assert len(reconciliation_files) == 1

    repeated = reconcile_neutral_candidate_policy_costs(
        tmp_path,
        candidate_cost_authorities=[authority],
    )
    assert repeated == report
    assert len(client.calls) == 2
