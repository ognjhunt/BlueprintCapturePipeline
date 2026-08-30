from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_evaluation_shared_mutation_window import (
    TaskEvaluationSharedMutationWindowError,
    materialize_shared_mutation_window,
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
        "provider_allowlist": ["vast"],
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


def template() -> dict[str, object]:
    value: dict[str, object] = {
        "schema_version": "task_evaluation_configured_controls_release_window_template.v1",
        "status": "authorized_for_dynamic_release",
        "team_namespace": "team-a",
        "expected_production_commit": "a" * 40,
        "allowed_mutations": [
            "profile_publication",
            "catalog_synchronization",
            "standing_authorization",
        ],
        "provider_allowlist": ["vast"],
        "maximum_hard_cap_usd": 1.0,
        "valid_for_seconds": 3600,
        "released_by": "policy-lead-001",
        "release_reference": "configured-controls dynamic release",
        "provider_resource_allocation_allowed": False,
        "paid_request_allowed": False,
        "template_digest": "",
    }
    value["template_digest"] = canonical_digest(
        value, digest_field="template_digest"
    )
    return value


def activation_request() -> dict[str, object]:
    reference = {
        "uri": "s3://blueprint-production-inputs/placeholder.json",
        "digest": "sha256:" + "0" * 64,
        "size_bytes": 1,
    }
    return {
        "schema_version": "task_evaluation_launch_activation_request.v1",
        "expected_production_commit": "a" * 40,
        "activation_id": "activation-scene-839873-construction",
        "team_namespace": "team-a",
        "lane": "native_task_arena_construction",
        "preparation": {
            "preparation_id": "prep-scene-839873",
            "request_digest": "sha256:" + "1" * 64,
            "result_digest": "sha256:" + "2" * 64,
        },
        "release_window": reference,
        "lineage": {
            "kind": "initial_project",
            "project_spend_reconciliation": reference,
            "initial_provider_zero": reference,
        },
        "authorization": {
            "reference": "ADP-009D Day-28",
            "authorized_by": "policy-lead-001",
            "authorized_on": "2026-08-29T00:00:00+00:00",
            "standing_authorization_expires_at": "2026-09-01T00:00:00+00:00",
            "profile_revision": "scene-839873-r1",
        },
        "requested_mutations": {
            "profile_publication": True,
            "catalog_synchronization": True,
            "standing_authorization": True,
        },
    }


def test_exact_window_releases_only_no_execution_mutations() -> None:
    now = datetime(2026, 8, 25, 18, 0, tzinfo=timezone.utc)
    value = window(now)
    assert validate_shared_mutation_window(
        value,
        activation_id="activation-scene-841007-construction",
        activation_intent_digest="sha256:" + "1" * 64,
        team_namespace="team-a",
        expected_production_commit="a" * 40,
        provider_allowlist=["vast"],
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
            provider_allowlist=["vast"],
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
            provider_allowlist=["vast"],
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
            provider_allowlist=["vast"],
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
            provider_allowlist=["vast"],
            hard_cap_usd=0.75,
            now=now,
        )


def test_dynamic_window_binds_future_preparation_and_precomputed_digest_fails() -> None:
    now = datetime(2026, 8, 29, 18, 0, tzinfo=timezone.utc)
    request = activation_request()
    value = materialize_shared_mutation_window(
        template(),
        activation_request=request,
        provider_allowlist=["vast"],
        hard_cap_usd=0.75,
        now=now,
    )
    from blueprint_pipeline.task_evaluation_launch_activation_contract import (
        launch_activation_intent_digest,
    )

    assert validate_shared_mutation_window(
        value,
        activation_id=str(request["activation_id"]),
        activation_intent_digest=launch_activation_intent_digest(request),
        team_namespace="team-a",
        expected_production_commit="a" * 40,
        provider_allowlist=["vast"],
        hard_cap_usd=0.75,
        now=now,
    ) == value

    future_request = dict(request)
    future_request["preparation"] = {
        **dict(request["preparation"]),
        "result_digest": "sha256:" + "3" * 64,
    }
    with pytest.raises(
        TaskEvaluationSharedMutationWindowError,
        match="shared_mutation_window_binding_mismatch",
    ):
        validate_shared_mutation_window(
            value,
            activation_id=str(request["activation_id"]),
            activation_intent_digest=launch_activation_intent_digest(future_request),
            team_namespace="team-a",
            expected_production_commit="a" * 40,
            provider_allowlist=["vast"],
            hard_cap_usd=0.75,
            now=now,
        )
