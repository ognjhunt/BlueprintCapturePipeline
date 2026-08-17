"""Production-shaped tests for exact OpenAI cost attribution."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.openai_official_cost_gate import (
    RUN_COMPLETION_SCHEMA_VERSION,
    RUN_SETTLEMENT_SCHEMA_VERSION,
    OpenAIOfficialCostGateError,
    build_openai_official_cost_run_gate,
)


NOW = datetime(2026, 8, 17, 12, 0, tzinfo=timezone.utc)
PROJECT = "proj_blueprint_scene840920"
KEY = "key_scene840920"
PROVIDER = "openai"
RESOURCE_CLASS = "sam31_ai_visual_review"


def _attestation(tmp_path: Path, **overrides: Any) -> Path:
    value: dict[str, Any] = {
        "schema_version": "openai_candidate_cost_scope_attestation.v1",
        "status": "approved",
        "issued_by_agent": False,
        "operator_id": "independent-cost-owner",
        "provider_id": PROVIDER,
        "paid_resource_class": RESOURCE_CLASS,
        "project_id": PROJECT,
        "api_key_id": KEY,
        "exclusive_use": True,
        "candidate_reported_usage_is_authoritative": False,
        "exclusive_from": (NOW - timedelta(days=1)).isoformat(),
        "exclusive_until": (NOW + timedelta(days=7)).isoformat(),
        "proof_effect": "none",
    }
    value.update(overrides)
    value["scope_attestation_digest"] = canonical_digest(
        value, digest_field="scope_attestation_digest"
    )
    path = tmp_path / "scope-attestation.json"
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return path


def _admin_key(tmp_path: Path, *, mode: int = 0o600) -> Path:
    path = tmp_path / "admin-key"
    path.write_text("sk-admin-0123456789abcdef", encoding="utf-8")
    path.chmod(mode)
    return path


def _transport(costs: list[float]):
    calls = 0

    def transport(url: str, _headers: Any, _timeout: float) -> dict[str, Any]:
        nonlocal calls
        query = parse_qs(urlparse(url).query)
        start = int(query["start_time"][0])
        end = int(query["end_time"][0])
        total = costs[min(calls, len(costs) - 1)]
        calls += 1
        results = []
        if total:
            results.append(
                {
                    "object": "organization.costs.result",
                    "project_id": PROJECT,
                    "api_key_id": KEY,
                    "amount": {"currency": "usd", "value": total},
                }
            )
        return {
            "object": "page",
            "data": [
                {
                    "object": "bucket",
                    "start_time": start,
                    "end_time": end,
                    "results": results,
                }
            ],
            "has_more": False,
        }

    return transport


def _gate(
    tmp_path: Path,
    *,
    transport,
    wall_clock,
    attestation_path: Path | None = None,
    admin_key_path: Path | None = None,
):
    return build_openai_official_cost_run_gate(
        scope_attestation_path=attestation_path or _attestation(tmp_path),
        admin_api_key_file=admin_key_path or _admin_key(tmp_path),
        project_id=PROJECT,
        api_key_id=KEY,
        lane_id="public_scene_sam31_ai_visual_review",
        run_id="sam31-ai-visual-review-abc",
        request_digest=f"sha256:{'c' * 64}",
        candidate_digest=f"sha256:{'d' * 64}",
        authorization_receipt_digest=f"sha256:{'e' * 64}",
        max_cost_usd=1.0,
        output_root=tmp_path / "official-cost",
        provider_id=PROVIDER,
        paid_resource_class=RESOURCE_CLASS,
        transport=transport,
        wall_clock=wall_clock,
    )


def test_run_gate_binds_reservation_completion_and_terminal_settlement(
    tmp_path: Path,
) -> None:
    clock = [NOW]
    gate = _gate(
        tmp_path,
        transport=_transport([0.0, 0.12, 0.12]),
        wall_clock=lambda: clock[0],
    )
    reservation = gate.reserve()
    completion = gate.complete(
        provider_call_performed=True,
        runtime_result_digest=f"sha256:{'f' * 64}",
        runtime_exception_type=None,
    )

    assert reservation["zero_cost_baseline_confirmed"] is True
    assert reservation["candidate_digest"] == f"sha256:{'d' * 64}"
    assert completion["schema_version"] == RUN_COMPLETION_SCHEMA_VERSION
    assert completion["provider_observed_cost_usd"] == pytest.approx(0.12)
    assert completion["cost_is_final"] is False
    assert completion["strict_official_billing_satisfied"] is False

    clock[0] = NOW + timedelta(days=4)
    settlement = gate.settle()
    assert settlement["schema_version"] == RUN_SETTLEMENT_SCHEMA_VERSION
    assert settlement["status"] == "reconciled"
    assert settlement["actual_cost_usd"] == pytest.approx(0.12)
    assert settlement["strict_official_billing_satisfied"] is True


def test_nonzero_baseline_refuses_before_reservation_file(tmp_path: Path) -> None:
    gate = _gate(
        tmp_path,
        transport=_transport([0.01]),
        wall_clock=lambda: NOW,
    )
    with pytest.raises(
        OpenAIOfficialCostGateError,
        match="openai_cost_scope_baseline_not_zero",
    ):
        gate.reserve()
    assert not gate.reservation_path.exists()


@pytest.mark.parametrize(
    ("attestation_overrides", "admin_mode"),
    [
        ({"issued_by_agent": True}, 0o600),
        ({"exclusive_use": False}, 0o600),
        ({}, 0o644),
    ],
)
def test_gate_refuses_untrusted_scope_or_admin_key(
    tmp_path: Path,
    attestation_overrides: dict[str, Any],
    admin_mode: int,
) -> None:
    with pytest.raises(OpenAIOfficialCostGateError):
        gate = _gate(
            tmp_path,
            transport=_transport([0.0]),
            wall_clock=lambda: NOW,
            attestation_path=_attestation(tmp_path, **attestation_overrides),
            admin_key_path=_admin_key(tmp_path, mode=admin_mode),
        )
        gate.reserve()


def test_completion_refuses_tampered_exact_candidate_binding(tmp_path: Path) -> None:
    gate = _gate(
        tmp_path,
        transport=_transport([0.0, 0.1]),
        wall_clock=lambda: NOW,
    )
    gate.reserve()
    value = json.loads(gate.reservation_path.read_text(encoding="utf-8"))
    value["candidate_digest"] = f"sha256:{'9' * 64}"
    value["reservation_receipt_digest"] = canonical_digest(
        value, digest_field="reservation_receipt_digest"
    )
    gate.reservation_path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(
        OpenAIOfficialCostGateError,
        match="openai_official_cost_reservation_invalid",
    ):
        gate.complete(
            provider_call_performed=True,
            runtime_result_digest=f"sha256:{'f' * 64}",
            runtime_exception_type=None,
        )
