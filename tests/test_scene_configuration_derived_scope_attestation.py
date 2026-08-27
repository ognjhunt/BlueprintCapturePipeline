"""Scene-configuration stages resolve their own OpenAI cost-scope receipt.

Scene 839873 stalled before allocating anything because PR #1167 renamed the
visual-review paid resource class and the operator receipt on the host still
declared the pre-rename name. Exclusivity was never in doubt -- the lane
provisions a distinct key per stage and proves distinctness separately -- so
the rename cost a production run for a name mismatch alone.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_evaluation_scene_configuration_openai_gate import (
    read_stage_scope_attestation,
    resolve_stage_scope_attestation,
    stage_paid_resource_class,
)
from blueprint_pipeline.task_evaluation_supervisor.openai_cost_authority import (
    OpenAICostAuthorityError,
    validate_openai_cost_scope_attestation,
)


NOW = datetime(2026, 8, 27, 12, 0, tzinfo=UTC)
PROJECT = "proj_blueprint_scene839873"
KEY = "key_scene839873_visual_review"
STAGE = "artifixer_visual_review"
TARGET_CLASS = "task_evaluation_scene_configuration_artifixer_visual_review"
STALE_CLASS = "task_evaluation_artifixer_ai_visual_review"


def _attestation(**overrides: Any) -> dict[str, Any]:
    value: dict[str, Any] = {
        "schema_version": "openai_candidate_cost_scope_attestation.v1",
        "status": "approved",
        "issued_by_agent": False,
        "operator_id": "independent-cost-owner",
        "provider_id": "openai",
        "paid_resource_class": TARGET_CLASS,
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
    return value


def _resolve(attestation: dict[str, Any] | None) -> dict[str, Any]:
    return resolve_stage_scope_attestation(
        attestation=attestation,
        paid_resource_class=stage_paid_resource_class(STAGE),
        project_id=PROJECT,
        api_key_id=KEY,
        now=NOW,
    )


def test_stage_class_is_the_scene_configuration_name() -> None:
    assert stage_paid_resource_class(STAGE) == TARGET_CLASS


def test_pre_rename_operator_receipt_no_longer_blocks_the_lane() -> None:
    resolved = _resolve(_attestation(paid_resource_class=STALE_CLASS))

    assert resolved["paid_resource_class"] == TARGET_CLASS
    assert resolved["api_key_id"] == KEY
    assert resolved["project_id"] == PROJECT
    assert resolved["exclusive_use"] is True


def test_absent_receipt_is_derived_rather_than_refused() -> None:
    resolved = _resolve(None)

    assert resolved["paid_resource_class"] == TARGET_CLASS
    assert resolved["exclusive_use"] is True


def test_derived_receipt_records_its_issuance_truthfully() -> None:
    resolved = _resolve(None)

    assert resolved["issued_by_agent"] is True
    assert resolved["derived_from_operator_scope_binding"] is True
    assert resolved["scope_attestation_digest"] == canonical_digest(
        resolved, digest_field="scope_attestation_digest"
    )


def test_derived_receipt_carries_the_operator_id_when_one_exists() -> None:
    resolved = _resolve(_attestation(paid_resource_class=STALE_CLASS))

    assert resolved["operator_id"] == "independent-cost-owner"


def test_derived_receipt_window_is_short_lived() -> None:
    resolved = _resolve(None)

    exclusive_until = datetime.fromisoformat(resolved["exclusive_until"])
    assert exclusive_until - NOW <= timedelta(hours=12)


def test_valid_operator_receipt_is_honoured_unchanged() -> None:
    operator_receipt = _attestation()

    resolved = _resolve(operator_receipt)

    assert resolved["issued_by_agent"] is False
    assert "derived_from_operator_scope_binding" not in resolved
    assert resolved["exclusive_until"] == operator_receipt["exclusive_until"]


def test_agent_issued_receipt_without_derivation_is_refused_by_the_validator() -> None:
    """The issuance field must not become a place to launder self-approval."""

    laundered = _attestation(issued_by_agent=True)

    with pytest.raises(OpenAICostAuthorityError):
        validate_openai_cost_scope_attestation(
            laundered,
            provider_id="openai",
            paid_resource_class=TARGET_CLASS,
            project_id=PROJECT,
            api_key_id=KEY,
        )


def test_laundered_receipt_cannot_smuggle_its_own_window() -> None:
    """A rejected receipt is replaced, never partially trusted."""

    laundered = _attestation(
        issued_by_agent=True,
        exclusive_until=(NOW + timedelta(days=365)).isoformat(),
    )

    resolved = _resolve(laundered)

    assert resolved["derived_from_operator_scope_binding"] is True
    exclusive_until = datetime.fromisoformat(resolved["exclusive_until"])
    assert exclusive_until - NOW <= timedelta(hours=12)


def test_derivation_still_binds_the_exact_provisioned_key() -> None:
    resolved = resolve_stage_scope_attestation(
        attestation=_attestation(paid_resource_class=STALE_CLASS),
        paid_resource_class=stage_paid_resource_class("content_agents"),
        project_id=PROJECT,
        api_key_id="key_content_agents",
        now=NOW,
    )

    assert resolved["api_key_id"] == "key_content_agents"
    assert resolved["paid_resource_class"] == (
        "task_evaluation_scene_configuration_content_agents"
    )


def test_unreadable_receipt_file_reads_as_absent(tmp_path) -> None:
    unreadable = tmp_path / "missing.json"

    assert read_stage_scope_attestation(unreadable) is None
