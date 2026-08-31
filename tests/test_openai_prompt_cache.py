from __future__ import annotations

from types import SimpleNamespace

import pytest

from blueprint_pipeline.openai_prompt_cache import (
    create_prompt_cache_policy,
    decide_prompt_cache_policy,
    explicit_cache_input,
    explicit_cache_request_kwargs,
    usage_and_cost_receipt,
    worst_case_reservation_usd,
)


def _policy(**overrides):
    values = {
        "model": "gpt-5.6-sol",
        "family": "task_aware_robot_placement_proposal",
        "contract_version": "placement-proposal-v1",
        "stable_prefix": "stable contract " * 700,
        "stable_prefix_tokens": 1_400,
        "tool_schema": [],
        "output_schema": {"type": "object", "required": ["candidate_id"]},
        "reasoning_effort": "high",
        "verbosity": "low",
        "privacy_scope": "internal_rights_admitted",
        "processing_region": "us",
        "expected_reuse_count": 3,
        "expected_reuse_probability": 1.0,
        "dynamic_suffix_fields": ("run_id", "round_index", "native_feedback"),
    }
    values.update(overrides)
    return create_prompt_cache_policy(**values)


def test_economic_decision_disables_one_off_and_below_break_even() -> None:
    one_off = decide_prompt_cache_policy(
        model="gpt-5.6-sol",
        stable_prefix_tokens=2_000,
        expected_reuse_probability=0.0,
        expected_reuse_count=0,
        ttl_compatible=True,
        privacy_compatible=True,
        explicit_breakpoint_available=True,
    )
    below_break_even = decide_prompt_cache_policy(
        model="gpt-5.6-sol",
        stable_prefix_tokens=2_000,
        expected_reuse_probability=0.27,
        expected_reuse_count=1,
        ttl_compatible=True,
        privacy_compatible=True,
        explicit_breakpoint_available=True,
    )
    repeated = decide_prompt_cache_policy(
        model="gpt-5.6-sol",
        stable_prefix_tokens=2_000,
        expected_reuse_probability=0.28,
        expected_reuse_count=1,
        ttl_compatible=True,
        privacy_compatible=True,
        explicit_breakpoint_available=True,
    )

    assert one_off.enabled is False
    assert one_off.reason == "one_off_no_expected_reuse"
    assert below_break_even.enabled is False
    assert below_break_even.reason == "expected_cached_cost_not_lower"
    assert repeated.enabled is True
    assert repeated.economics.break_even_reuse_probability == pytest.approx(0.2777777778)
    assert repeated.economics.expected_savings_usd > 0


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ({"stable_prefix_tokens": 1_023}, "stable_prefix_below_model_minimum"),
        ({"explicit_breakpoint_available": False}, "explicit_stable_breakpoint_missing"),
        ({"ttl_compatible": False}, "ttl_incompatible"),
        ({"privacy_compatible": False}, "privacy_or_region_incompatible"),
    ],
)
def test_economic_decision_fail_closed_gates(changes, reason) -> None:
    values = {
        "model": "gpt-5.6-sol",
        "stable_prefix_tokens": 2_000,
        "expected_reuse_probability": 1.0,
        "expected_reuse_count": 1,
        "ttl_compatible": True,
        "privacy_compatible": True,
        "explicit_breakpoint_available": True,
    }
    values.update(changes)
    decision = decide_prompt_cache_policy(**values)
    assert decision.enabled is False
    assert decision.reason == reason


def test_stable_key_ignores_dynamic_suffix_but_versions_every_stable_contract() -> None:
    first = _policy()
    same = _policy(dynamic_suffix_fields=("different_run_id", "new_feedback"))
    changed_contract = _policy(contract_version="placement-proposal-v2")
    changed_tools = _policy(tool_schema=[{"name": "inspect"}])
    changed_output = _policy(output_schema={"type": "object", "required": ["pose"]})
    changed_model = _policy(model="gpt-5.6-terra")
    changed_effort = _policy(reasoning_effort="xhigh")
    changed_privacy = _policy(privacy_scope="different_tenant_scope")
    changed_region = _policy(processing_region="eu")
    changed_parallel_tools = _policy(parallel_tool_calls=False)
    changed_context_management = _policy(context_management={"type": "compaction"})

    assert first.cache_key == same.cache_key
    for changed in (
        changed_contract,
        changed_tools,
        changed_output,
        changed_model,
        changed_effort,
        changed_privacy,
        changed_region,
        changed_parallel_tools,
        changed_context_management,
    ):
        assert changed.cache_key != first.cache_key
    assert first.cache_key is not None
    assert len(first.cache_key) <= 64
    assert "run" not in first.cache_key
    assert "tenant" not in first.cache_key


def test_explicit_layout_stops_before_dynamic_text_and_images() -> None:
    policy = _policy(explicit_breakpoints=("stable_developer_prefix", "scene_static_prefix"))
    dynamic = [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "run_id=unique round=2"},
                {"type": "input_image", "image_url": "data:image/png;base64,AA=="},
            ],
        }
    ]
    rendered = explicit_cache_input(
        policy=policy,
        stable_developer_prefix="stable developer contract",
        scene_static_prefix="stable scene digest and normalized trajectory",
        dynamic_input=dynamic,
    )

    assert isinstance(rendered, list)
    assert [item["role"] for item in rendered] == ["developer", "developer", "user"]
    assert rendered[0]["content"][0]["prompt_cache_breakpoint"] == {"mode": "explicit"}
    assert rendered[1]["content"][0]["prompt_cache_breakpoint"] == {"mode": "explicit"}
    assert "prompt_cache_breakpoint" not in rendered[2]["content"][0]
    assert explicit_cache_request_kwargs(policy) == {
        "prompt_cache_options": {"mode": "explicit", "ttl": "30m"},
        "prompt_cache_key": policy.cache_key,
    }


def test_one_off_explicit_mode_has_no_key_or_breakpoint() -> None:
    policy = _policy(
        stable_prefix_tokens=0,
        expected_reuse_count=0,
        expected_reuse_probability=0,
        explicit_breakpoint_available=False,
    )
    dynamic = [{"role": "user", "content": "one off"}]
    assert policy.status == "disabled"
    assert explicit_cache_request_kwargs(policy) == {
        "prompt_cache_options": {"mode": "explicit", "ttl": "30m"}
    }
    rendered = explicit_cache_input(
        policy=policy,
        stable_developer_prefix="not sent",
        dynamic_input=dynamic,
    )
    assert isinstance(rendered, list)
    assert [item["role"] for item in rendered] == ["developer", "user"]
    assert "prompt_cache_breakpoint" not in rendered[0]["content"][0]


def test_usage_receipt_prices_write_read_uncached_and_output_separately() -> None:
    response = SimpleNamespace(
        id="resp_test",
        status="completed",
        usage=SimpleNamespace(
            input_tokens=10_000,
            output_tokens=100,
            input_tokens_details=SimpleNamespace(
                cached_tokens=6_000,
                cache_write_tokens=2_000,
            ),
            output_tokens_details=SimpleNamespace(reasoning_tokens=40),
        ),
    )
    receipt = usage_and_cost_receipt(response, model="gpt-5.6-sol")

    assert receipt["uncached_input_tokens"] == 2_000
    assert receipt["cached_read_cost_usd"] == pytest.approx(0.0024)
    assert receipt["cache_write_cost_usd"] == pytest.approx(0.01)
    assert receipt["uncached_input_cost_usd"] == pytest.approx(0.008)
    assert receipt["output_cost_usd"] == pytest.approx(0.002)
    assert receipt["estimated_total_cost_usd"] == pytest.approx(0.0224)
    assert receipt["estimated_cost_without_caching_usd"] == pytest.approx(0.042)
    assert receipt["estimated_savings_usd"] == pytest.approx(0.0196)
    assert receipt["provider_response_id"] == "resp_test"


def test_reservation_assumes_a_write_and_long_context_tier_not_a_hit() -> None:
    policy = _policy(stable_prefix_tokens=2_000)
    cost = worst_case_reservation_usd(
        model="gpt-5.6-sol",
        input_token_ceiling=300_000,
        max_output_tokens=8_000,
        cache_policy=policy,
    )
    expected = (2_000 * 10 + 298_000 * 8 + 8_000 * 30) / 1_000_000
    assert cost == pytest.approx(expected)


def test_aggregate_usage_applies_long_context_pricing_per_request() -> None:
    entries = [
        {
            "input_tokens": 100_000,
            "output_tokens": 0,
            "input_tokens_details": {
                "cached_tokens": 0,
                "cache_write_tokens": 0,
            },
            "output_tokens_details": {"reasoning_tokens": 0},
        }
        for _ in range(3)
    ]
    receipt = usage_and_cost_receipt(
        {
            "input_tokens": 300_000,
            "output_tokens": 0,
            "input_tokens_details": {
                "cached_tokens": 0,
                "cache_write_tokens": 0,
            },
            "output_tokens_details": {"reasoning_tokens": 0},
            "request_usage_entries": entries,
        },
        model="gpt-5.6-sol",
    )

    assert receipt["request_count"] == 3
    assert receipt["uncached_input_cost_usd"] == pytest.approx(1.2)
    assert all(
        row["long_context_pricing_applied"] is False
        for row in receipt["per_request_costs"]
    )
