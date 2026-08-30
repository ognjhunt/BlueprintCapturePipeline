from __future__ import annotations

from blueprint_pipeline.task_evaluation_artifixer_ai_visual_review import (
    AI_REVIEW_MAX_INPUT_TOKENS,
    AI_REVIEW_MAX_OUTPUT_TOKENS,
)
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
    diagnostic_parent_runtime_budget_blockers,
    diagnostic_required_parent_ttl_seconds,
    parent_runtime_budget_blockers,
    required_remaining_stage_seconds,
)
from blueprint_pipeline.task_evaluation_supervisor.agents_sdk import (
    OpenAIAgentsSDKConfig,
)


def test_parent_runtime_policy_covers_serialized_stages_and_named_reserves() -> None:
    assert tuple(GPU_STAGE_TIMEOUT_SECONDS.values()) == (12_000, 7_800, 1_800)
    assert SERIAL_GPU_STAGE_TIMEOUT_SECONDS == 21_600
    assert BOOTSTRAP_TRANSFER_AND_NO_SPEND_RESERVE_SECONDS == 3_600
    assert OUTPUT_AND_CLOSURE_RESERVE_SECONDS == 1_800
    assert REQUIRED_PARENT_TTL_SECONDS == 27_000
    assert ceil_live_minutes(REQUIRED_PARENT_TTL_SECONDS) == 450
    assert MAX_PROVIDER_COMPUTE_SPEND_USD >= (
        MAX_HOURLY_RATE_USD * REQUIRED_PARENT_TTL_SECONDS / 3_600
    )
    assert MAX_PROVIDER_COMPUTE_SPEND_USD == 6.0
    assert MIN_ARTIFIXER_SEMANTIC_TEACHER_SPEND_USD == 4.8
    reviewer_costs = OpenAIAgentsSDKConfig()
    fixed_reviewer_reservation = (
        AI_REVIEW_MAX_INPUT_TOKENS
        * reviewer_costs.input_cost_per_million_tokens_usd
        + AI_REVIEW_MAX_OUTPUT_TOKENS
        * reviewer_costs.output_cost_per_million_tokens_usd
    ) / 1_000_000
    assert MIN_ARTIFIXER_VISUAL_REVIEW_SPEND_USD == (
        2 * fixed_reviewer_reservation
    ) == 0.64
    assert MIN_CONTENT_AGENTS_SPEND_USD == 0.2
    assert MIN_EXTERNAL_SERVICE_SPEND_USD == 5.64
    assert MAX_EXTERNAL_SERVICE_SPEND_USD == 6.0
    assert MAX_ATTEMPT_SPEND_USD == 12.0
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
        "scene_configuration_parent_runtime_budget_insufficient:27000:26999"
    ]
    assert parent_runtime_budget_blockers(
        ttl_seconds=REQUIRED_PARENT_TTL_SECONDS,
        maximum_hourly_rate_usd=MAX_HOURLY_RATE_USD,
        provider_compute_spend_cap_usd=5.59,
    ) == [
        "scene_configuration_provider_compute_budget_insufficient:6.000000:5.590000"
    ]


def test_remaining_stage_budget_is_serial_and_reserves_output_closure() -> None:
    stages = [
        {"adapter": {"id": adapter_id}}
        for adapter_id in GPU_STAGE_TIMEOUT_SECONDS
    ]
    assert required_remaining_stage_seconds(stages, start_index=0) == 23_400
    assert required_remaining_stage_seconds(stages, start_index=1) == 11_400
    assert required_remaining_stage_seconds(stages, start_index=3) == 1_800


def test_diagnostic_runtime_budget_drops_only_completed_gpu_allowances() -> None:
    assert diagnostic_required_parent_ttl_seconds(0) == 27_000
    assert diagnostic_required_parent_ttl_seconds(1) == 15_000
    assert diagnostic_required_parent_ttl_seconds(3) == 7_200
    assert diagnostic_required_parent_ttl_seconds(6) == 5_400
    assert diagnostic_required_parent_ttl_seconds(0) == REQUIRED_PARENT_TTL_SECONDS


def test_diagnostic_runtime_budget_refuses_short_remaining_lease_or_cap() -> None:
    assert diagnostic_parent_runtime_budget_blockers(
        completed_stage_prefix_count=3,
        ttl_seconds=7_199,
        maximum_hourly_rate_usd=0.8,
        provider_compute_spend_cap_usd=3.0,
    ) == [
        "scene_configuration_diagnostic_runtime_budget_insufficient:7200:7199"
    ]
    assert diagnostic_parent_runtime_budget_blockers(
        completed_stage_prefix_count=3,
        ttl_seconds=9_600,
        maximum_hourly_rate_usd=0.8,
        provider_compute_spend_cap_usd=2.12,
    ) == [
        "scene_configuration_diagnostic_provider_compute_budget_insufficient:"
        "2.133333:2.120000"
    ]
