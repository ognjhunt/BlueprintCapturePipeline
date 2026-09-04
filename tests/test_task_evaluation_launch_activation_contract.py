from __future__ import annotations

import copy

import pytest

from blueprint_pipeline.task_evaluation_launch_activation_contract import (
    SCHEMA_VERSION,
    TaskEvaluationLaunchActivationContractError,
    launch_activation_intent_digest,
    launch_activation_request_digest,
    validate_launch_activation_request,
)


def ref(index: int) -> dict[str, object]:
    return {
        "uri": f"s3://blueprint-production-inputs/activation-{index}.json",
        "digest": f"sha256:{index:064x}",
        "size_bytes": 1000 + index,
    }


def request(*, lane: str = "native_task_arena_construction") -> dict[str, object]:
    lineage: dict[str, object]
    if lane in {
        "task_evaluation_scene_configuration",
        "native_task_arena_construction",
        "native_task_arena_destination_qualification",
    }:
        lineage = {
            "kind": "initial_project",
            "project_spend_reconciliation": ref(2),
            "initial_provider_zero": ref(3),
        }
    else:
        lineage = {
            "kind": "predecessor",
            "prior_authority": ref(4),
            "prior_result": ref(5),
            "prior_launch_receipt": ref(6),
            "prior_webapp_sync": ref(7),
            "prior_provider_zero": ref(8),
            "prior_spend_reconciliation": ref(9),
            "construction_result": ref(10),
        }
        if lane == "native_task_arena_construction_after_destination":
            lineage["destination_qualification_result"] = lineage.pop(
                "construction_result"
            )
        if lane == "native_task_arena_scripted_positive":
            lineage["zero_action_result"] = ref(11)
        if lane == "native_task_arena_policy_evaluation":
            lineage.update(
                controls_qualification_manifest=ref(13),
            )
    return {
        "schema_version": SCHEMA_VERSION,
        "expected_production_commit": "a" * 40,
        "activation_id": f"activation-scene-841007-{lane}",
        "team_namespace": "team-a",
        "lane": lane,
        "preparation": {
            "preparation_id": "preparation-scene-841007-v1",
            "request_digest": "sha256:" + "b" * 64,
            "result_digest": "sha256:" + "c" * 64,
        },
        "release_window": ref(1),
        "lineage": lineage,
        "authorization": {
            "reference": "user-approved-window-20260825",
            "authorized_by": "founder-001",
            "authorized_on": "2026-08-25T18:00:00+00:00",
            "standing_authorization_expires_at": "2026-08-25T19:30:00+00:00",
            "profile_revision": "r1",
        },
        "requested_mutations": (
            {
                "profile_publication": False,
                "catalog_synchronization": False,
                "standing_authorization": False,
                "policy_campaign_queue": True,
            }
            if lane == "native_task_arena_policy_evaluation"
            else {
                "profile_publication": True,
                "catalog_synchronization": True,
                "standing_authorization": True,
            }
        ),
    }


@pytest.mark.parametrize(
    "lane",
    [
        "native_task_arena_construction",
        "native_task_arena_destination_qualification",
        "native_task_arena_construction_after_destination",
        "task_evaluation_scene_configuration",
        "native_task_arena_controls",
        "native_task_arena_zero_action",
        "native_task_arena_scripted_positive",
        "native_task_arena_policy_evaluation",
    ],
)
def test_accepts_each_explicit_native_arena_stage(lane: str) -> None:
    value = request(lane=lane)
    assert validate_launch_activation_request(value) == value
    assert launch_activation_request_digest(value) == launch_activation_request_digest(
        copy.deepcopy(value)
    )


def test_rejects_wrong_lineage_or_missing_zero_action_predecessor() -> None:
    value = request()
    value["lineage"] = request(lane="native_task_arena_zero_action")["lineage"]
    with pytest.raises(
        TaskEvaluationLaunchActivationContractError,
        match="launch_activation_request_invalid:lineage",
    ):
        validate_launch_activation_request(value)

    value = request(lane="native_task_arena_policy_evaluation")
    del value["lineage"]["controls_qualification_manifest"]
    with pytest.raises(
        TaskEvaluationLaunchActivationContractError,
        match="launch_activation_request_invalid:lineage",
    ):
        validate_launch_activation_request(value)

    value = request(lane="native_task_arena_scripted_positive")
    del value["lineage"]["zero_action_result"]
    with pytest.raises(
        TaskEvaluationLaunchActivationContractError,
        match="launch_activation_request_invalid:lineage",
    ):
        validate_launch_activation_request(value)


def test_intent_digest_excludes_window_reference_but_binds_customer_intent() -> None:
    value = request()
    expected = launch_activation_intent_digest(value)

    value["release_window"] = ref(99)
    assert launch_activation_intent_digest(value) == expected

    value["authorization"]["profile_revision"] = "r2"
    assert launch_activation_intent_digest(value) != expected


def test_rejects_host_paths_secrets_and_invalid_authority_window() -> None:
    value = request()
    value["release_window"]["uri"] = "/etc/blueprint/release-window.json"
    with pytest.raises(
        TaskEvaluationLaunchActivationContractError,
        match="launch_activation_request_invalid:release_window.uri",
    ):
        validate_launch_activation_request(value)

    value = request()
    value["api_key"] = "secret"
    with pytest.raises(
        TaskEvaluationLaunchActivationContractError,
        match=r"launch_activation_request_invalid:\$",
    ):
        validate_launch_activation_request(value)

    value = request()
    value["authorization"]["standing_authorization_expires_at"] = (
        value["authorization"]["authorized_on"]
    )
    with pytest.raises(
        TaskEvaluationLaunchActivationContractError,
        match="launch_activation_authorization_window_invalid",
    ):
        validate_launch_activation_request(value)
