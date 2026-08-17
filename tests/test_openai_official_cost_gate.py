"""Contract tests for the fail-closed OpenAI official-cost gate.

Official OpenAI attribution is only possible if a reservation precedes the
spend against an exclusively-scoped project and key whose attribution window is
still zero.  Once money has moved, the window is no longer zero and no official
per-run cost can ever be produced.  These tests pin a gate that answers "will
attribution be possible?" for ``$0`` before a lane spends, and that refuses to
let a production lane spend when the answer is no.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.openai_official_cost_gate import (
    PREFLIGHT_SCHEMA_VERSION,
    OpenAIOfficialCostGateError,
    preflight_official_cost_attribution,
    require_official_cost_reservation,
)


NOW = datetime(2026, 8, 17, 12, 0, tzinfo=timezone.utc)
PROJECT = "proj_blueprint_cad"
KEY = "key_abc123"


def _clock() -> datetime:
    return NOW


def _attestation(tmp_path: Path, *, name: str = "scope", **overrides: Any) -> Path:
    # Distinct filenames: `_kwargs` evaluates its own default after the
    # caller's override is built, so a shared path silently clobbers it.
    value: dict[str, Any] = {
        "schema_version": "openai_candidate_cost_scope_attestation.v1",
        "status": "approved",
        "issued_by_agent": False,
        "operator_id": "ognjhunt",
        "provider_id": "pigey_external_candidate",
        "paid_resource_class": "openai_api_candidate",
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
    path = tmp_path / f"{name}_attestation.json"
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _admin_key(tmp_path: Path, *, mode: int = 0o600, name: str = "admin") -> Path:
    path = tmp_path / f"{name}.key"
    path.write_text("sk-admin-0123456789abcdef", encoding="utf-8")
    path.chmod(mode)
    return path


def _transport(total: float = 0.0):
    """A fake that answers inside the window it was actually asked about.

    The real endpoint only ever returns buckets within the queried range, and
    the client rejects any bucket outside it, so a fixed bucket would fail for
    reasons unrelated to what each test is pinning.
    """

    def transport(url: str, headers: Any, timeout: float) -> dict[str, Any]:
        query = parse_qs(urlparse(url).query)
        start = int(query["start_time"][0])
        end = int(query["end_time"][0])
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
                    "end_time": min(start + 86_400, end),
                    "results": results,
                }
            ],
            "has_more": False,
        }

    return transport


def _kwargs(tmp_path: Path, **overrides: Any) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "scope_attestation_path": _attestation(tmp_path),
        "admin_api_key_file": _admin_key(tmp_path),
        "project_id": PROJECT,
        "api_key_id": KEY,
        "lane_id": "public_scene_sam31_ai_visual_review",
        "transport": _transport(),
        "wall_clock": _clock,
    }
    kwargs.update(overrides)
    return kwargs


# --- preflight: costs nothing, spends nothing -------------------------------


def test_preflight_reports_ready_when_attribution_is_possible(tmp_path: Path) -> None:
    receipt = preflight_official_cost_attribution(**_kwargs(tmp_path))
    assert receipt["schema_version"] == PREFLIGHT_SCHEMA_VERSION
    assert receipt["status"] == "ready"
    assert receipt["blockers"] == []
    assert receipt["baseline_cost_usd"] == 0.0
    assert receipt["provider_mutation_performed"] is False
    assert receipt["spend_incurred_usd"] == 0.0


def test_preflight_blocks_when_the_window_is_already_dirty(tmp_path: Path) -> None:
    """A non-zero baseline is the exact condition that makes attribution impossible."""

    receipt = preflight_official_cost_attribution(
        **_kwargs(tmp_path, transport=_transport(total=0.37))
    )
    assert receipt["status"] == "blocked"
    assert "openai_attribution_window_not_zero" in receipt["blockers"]
    assert receipt["baseline_cost_usd"] == 0.37


def test_preflight_blocks_on_a_world_readable_admin_key(tmp_path: Path) -> None:
    receipt = preflight_official_cost_attribution(
        **_kwargs(tmp_path, admin_api_key_file=_admin_key(tmp_path, mode=0o644, name="loose"))
    )
    assert receipt["status"] == "blocked"
    assert any("permissions" in code for code in receipt["blockers"])


def test_preflight_blocks_an_agent_issued_attestation(tmp_path: Path) -> None:
    """The attestation is an operator act; an agent must not be able to self-authorize."""

    receipt = preflight_official_cost_attribution(
        **_kwargs(
            tmp_path,
            scope_attestation_path=_attestation(tmp_path, name="agentissued", issued_by_agent=True),
        )
    )
    assert receipt["status"] == "blocked"
    assert "openai_cost_scope_attestation_invalid" in receipt["blockers"]


def test_preflight_blocks_a_non_exclusive_scope(tmp_path: Path) -> None:
    receipt = preflight_official_cost_attribution(
        **_kwargs(
            tmp_path, scope_attestation_path=_attestation(tmp_path, name="nonexclusive", exclusive_use=False)
        )
    )
    assert receipt["status"] == "blocked"


def test_preflight_blocks_an_expired_exclusivity_window(tmp_path: Path) -> None:
    receipt = preflight_official_cost_attribution(
        **_kwargs(
            tmp_path,
            scope_attestation_path=_attestation(
                tmp_path,
                name="expired",
                exclusive_until=(NOW - timedelta(hours=1)).isoformat(),
            ),
        )
    )
    assert receipt["status"] == "blocked"


def test_preflight_never_contacts_a_provider_for_mutation(tmp_path: Path) -> None:
    calls: list[str] = []

    def watching(url: str, headers: Any, timeout: float) -> dict[str, Any]:
        calls.append(url)
        return _transport()(url, headers, timeout)

    preflight_official_cost_attribution(**_kwargs(tmp_path, transport=watching))
    assert calls and all("/v1/organization/costs" in url for url in calls)


# --- the gate: refuses unattributable production spend ----------------------


def test_gate_returns_a_reservation_when_attribution_is_possible(
    tmp_path: Path,
) -> None:
    reservation = require_official_cost_reservation(
        **_kwargs(tmp_path),
        candidate_id="sam31-ai-visual-review-abc",
        max_cost_usd=1.0,
        candidate_evaluation_suite_digest=f"sha256:{'d' * 64}",
        authorization_receipt_digest=f"sha256:{'e' * 64}",
    )
    assert reservation["status"] == "reserved"
    assert reservation["reserved_max_cost_usd"] == 1.0
    assert reservation["project_id"] == PROJECT


def test_gate_refuses_to_let_a_lane_spend_unattributably(tmp_path: Path) -> None:
    """This is the whole point: no reservation, no spend."""

    with pytest.raises(OpenAIOfficialCostGateError) as exc:
        require_official_cost_reservation(
            **_kwargs(tmp_path, transport=_transport(total=0.37)),
            candidate_id="sam31-ai-visual-review-abc",
            max_cost_usd=1.0,
            candidate_evaluation_suite_digest=f"sha256:{'d' * 64}",
            authorization_receipt_digest=f"sha256:{'e' * 64}",
        )
    assert "openai_official_cost_attribution_unavailable" in str(exc.value)


def test_gate_refuses_a_missing_attestation(tmp_path: Path) -> None:
    with pytest.raises(OpenAIOfficialCostGateError):
        require_official_cost_reservation(
            **_kwargs(tmp_path, scope_attestation_path=tmp_path / "absent.json"),
            candidate_id="c",
            max_cost_usd=1.0,
            candidate_evaluation_suite_digest=f"sha256:{'d' * 64}",
            authorization_receipt_digest=f"sha256:{'e' * 64}",
        )
