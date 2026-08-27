from __future__ import annotations

from blueprint_pipeline.task_evaluation_scene_configuration_runtime_budget import (
    BOOTSTRAP_TRANSFER_AND_NO_SPEND_RESERVE_SECONDS,
    GPU_STAGE_TIMEOUT_SECONDS,
    MAX_ATTEMPT_SPEND_USD,
    MAX_EXTERNAL_SERVICE_SPEND_USD,
    MAX_HOURLY_RATE_USD,
    MAX_PROVIDER_COMPUTE_SPEND_USD,
    MIN_ARTIFIXER_SEMANTIC_TEACHER_SPEND_USD,
    MIN_ARTIFIXER_VISUAL_REVIEW_SPEND_USD,
    MIN_CONTENT_AGENTS_SPEND_USD,
    MIN_EXTERNAL_SERVICE_SPEND_USD,
    OUTPUT_AND_CLOSURE_RESERVE_SECONDS,
    REQUIRED_PARENT_TTL_SECONDS,
    SERIAL_GPU_STAGE_TIMEOUT_SECONDS,
    ceil_live_minutes,
    parent_runtime_budget_blockers,
    required_remaining_stage_seconds,
)


def test_parent_runtime_policy_covers_serialized_stages_and_named_reserves() -> None:
    assert tuple(GPU_STAGE_TIMEOUT_SECONDS.values()) == (7_800, 7_800, 1_800)
    assert SERIAL_GPU_STAGE_TIMEOUT_SECONDS == 17_400
    assert BOOTSTRAP_TRANSFER_AND_NO_SPEND_RESERVE_SECONDS == 6_000
    assert OUTPUT_AND_CLOSURE_RESERVE_SECONDS == 1_800
    assert REQUIRED_PARENT_TTL_SECONDS == 25_200
    assert ceil_live_minutes(REQUIRED_PARENT_TTL_SECONDS) == 420
    assert MAX_PROVIDER_COMPUTE_SPEND_USD >= (
        MAX_HOURLY_RATE_USD * REQUIRED_PARENT_TTL_SECONDS / 3_600
    )
    assert MAX_PROVIDER_COMPUTE_SPEND_USD == 6.0
    assert MIN_ARTIFIXER_SEMANTIC_TEACHER_SPEND_USD == 2.4
    assert MIN_ARTIFIXER_VISUAL_REVIEW_SPEND_USD == 0.3
    assert MIN_CONTENT_AGENTS_SPEND_USD == 0.2
    assert MIN_EXTERNAL_SERVICE_SPEND_USD == 2.9
    assert MAX_EXTERNAL_SERVICE_SPEND_USD == 3.0
    assert MAX_ATTEMPT_SPEND_USD == 10.0
    assert (
        MAX_ATTEMPT_SPEND_USD
        >= MAX_PROVIDER_COMPUTE_SPEND_USD + MAX_EXTERNAL_SERVICE_SPEND_USD
    )


def test_parent_runtime_authority_refuses_short_ttl_or_compute_budget() -> None:
    assert parent_runtime_budget_blockers(
        ttl_seconds=REQUIRED_PARENT_TTL_SECONDS - 1,
        maximum_hourly_rate_usd=MAX_HOURLY_RATE_USD,
        provider_compute_spend_cap_usd=MAX_PROVIDER_COMPUTE_SPEND_USD,
    ) == [
        "scene_configuration_parent_runtime_budget_insufficient:25200:25199"
    ]
    assert parent_runtime_budget_blockers(
        ttl_seconds=REQUIRED_PARENT_TTL_SECONDS,
        maximum_hourly_rate_usd=MAX_HOURLY_RATE_USD,
        provider_compute_spend_cap_usd=5.59,
    ) == [
        "scene_configuration_provider_compute_budget_insufficient:5.600000:5.590000"
    ]


def test_remaining_stage_budget_is_serial_and_reserves_output_closure() -> None:
    stages = [
        {"adapter": {"id": adapter_id}}
        for adapter_id in GPU_STAGE_TIMEOUT_SECONDS
    ]
    assert required_remaining_stage_seconds(stages, start_index=0) == 19_200
    assert required_remaining_stage_seconds(stages, start_index=1) == 11_400
    assert required_remaining_stage_seconds(stages, start_index=3) == 1_800
