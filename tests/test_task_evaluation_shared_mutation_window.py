from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_evaluation_shared_mutation_window import (
    TaskEvaluationSharedMutationWindowError,
    validate_shared_mutation_window,
)


def window(now: datetime) -> dict[str, object]:
    value: dict[str, object] = {
        "schema_version": "task_evaluation_shared_mutation_window.v1",
        "status": "released",
        "window_id": "window-scene-841007-001",
        "activation_id": "activation-scene-841007-construction",
        "activation_intent_digest": "sha256:" + "1" * 64,
        "team_namespace": "team-a",
        "expected_production_commit": "a" * 40,
        "allowed_mutations": [
            "profile_publication",
            "catalog_synchronization",
            "standing_authorization",
        ],
        "provider_allowlist": [],
        "maximum_hard_cap_usd": 1.0,
        "issued_at": (now - timedelta(minutes=1)).isoformat(),
        "expires_at": (now + timedelta(minutes=10)).isoformat(),
        "released_by": "policy-lead-001",
        "release_reference": "coordinated release window",
        "provider_resource_allocation_allowed": False,
        "paid_request_allowed": False,
        "window_digest": "",
    }
    value["window_digest"] = canonical_digest(value, digest_field="window_digest")
    return value


def test_exact_window_releases_only_no_execution_mutations() -> None:
    now = datetime(2026, 8, 25, 18, 0, tzinfo=timezone.utc)
    value = window(now)
    assert validate_shared_mutation_window(
        value,
        activation_id="activation-scene-841007-construction",
        activation_intent_digest="sha256:" + "1" * 64,
        team_namespace="team-a",
        expected_production_commit="a" * 40,
        provider_allowlist=[],
        hard_cap_usd=0.75,
        now=now,
    ) == value
    assert value["provider_resource_allocation_allowed"] is False
    assert value["paid_request_allowed"] is False


def test_window_fails_closed_on_stale_commit_cap_or_time() -> None:
    now = datetime(2026, 8, 25, 18, 0, tzinfo=timezone.utc)
    value = window(now)
    with pytest.raises(
        TaskEvaluationSharedMutationWindowError,
        match="shared_mutation_window_binding_mismatch",
    ):
        validate_shared_mutation_window(
            value,
            activation_id="activation-scene-841007-construction",
            activation_intent_digest="sha256:" + "1" * 64,
            team_namespace="team-a",
            expected_production_commit="b" * 40,
            provider_allowlist=[],
            hard_cap_usd=0.75,
            now=now,
        )
    with pytest.raises(
        TaskEvaluationSharedMutationWindowError,
        match="shared_mutation_window_binding_mismatch",
    ):
        validate_shared_mutation_window(
            value,
            activation_id="activation-scene-841007-construction",
            activation_intent_digest="sha256:" + "1" * 64,
            team_namespace="team-a",
            expected_production_commit="a" * 40,
            provider_allowlist=[],
            hard_cap_usd=1.01,
            now=now,
        )
    with pytest.raises(
        TaskEvaluationSharedMutationWindowError,
        match="shared_mutation_window_not_current",
    ):
        validate_shared_mutation_window(
            value,
            activation_id="activation-scene-841007-construction",
            activation_intent_digest="sha256:" + "1" * 64,
            team_namespace="team-a",
            expected_production_commit="a" * 40,
            provider_allowlist=[],
            hard_cap_usd=0.75,
            now=now + timedelta(hours=1),
        )


def test_window_fails_closed_when_activation_intent_changes_after_release() -> None:
    now = datetime(2026, 8, 25, 18, 0, tzinfo=timezone.utc)
    value = window(now)

    with pytest.raises(
        TaskEvaluationSharedMutationWindowError,
        match="shared_mutation_window_binding_mismatch",
    ):
        validate_shared_mutation_window(
            value,
            activation_id="activation-scene-841007-construction",
            activation_intent_digest="sha256:" + "2" * 64,
            team_namespace="team-a",
            expected_production_commit="a" * 40,
            provider_allowlist=[],
            hard_cap_usd=0.75,
            now=now,
        )
