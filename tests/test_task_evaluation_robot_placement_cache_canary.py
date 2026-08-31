from __future__ import annotations

from collections import defaultdict

from blueprint_pipeline.task_evaluation_robot_placement_agent import (
    RobotPlacementProposalOutput,
    RobotPlacementVisualReviewOutput,
)
from blueprint_pipeline.task_evaluation_supervisor.agents_sdk import (
    AgentsSDKInvocationResult,
)
from blueprint_pipeline import task_evaluation_robot_placement_cache_canary as canary


def test_production_shape_canary_reuses_both_families_without_gpu(
    tmp_path, monkeypatch
) -> None:
    key_file = tmp_path / "openai-key"
    key_file.write_text("test-only-key\n", encoding="utf-8")
    key_file.chmod(0o600)
    counts: defaultdict[str, int] = defaultdict(int)

    class FakeInvoker:
        def invoke(self, spec, _input):
            family = str(spec.capability)
            ordinal = counts[family]
            counts[family] += 1
            written = 2_500 if ordinal == 0 else 0
            cached = 0 if ordinal == 0 else 2_500
            policy = {
                "schema_version": "openai_prompt_cache_policy.v1",
                "status": "enabled",
                "model_family": "gpt-5.6-sol",
                "mode": "explicit",
                "family": family,
                "contract_version": spec.prompt_contract_version,
                "stable_prefix_digest": "sha256:" + "a" * 64,
                "tool_schema_digest": "sha256:" + "b" * 64,
                "output_schema_digest": "sha256:" + "c" * 64,
                "reasoning_effort": "high",
                "verbosity": "low",
                "privacy_scope": "task_evaluation_rights_admitted",
                "processing_region": "default",
                "expected_reuse_count": 3,
                "expected_reuse_probability": 1.0,
                "ttl": "30m",
                "explicit_breakpoints": ["stable_developer_prefix"],
                "dynamic_suffix_fields": list(spec.dynamic_suffix_fields),
                "cache_key_digest": "sha256:" + "d" * 64,
                "decision_reason": "expected_cached_cost_lower",
                "economics": {"stable_prefix_tokens": 2_500},
                "policy_digest": "sha256:" + "e" * 64,
            }
            usage = {
                "model": "gpt-5.6-sol",
                "input_tokens": 3_300,
                "cached_tokens": cached,
                "cache_write_tokens": written,
                "uncached_input_tokens": 800,
                "output_tokens": 20,
                "reasoning_tokens": 5,
                "cache_hit_ratio": cached / 3_300,
                "uncached_input_cost_usd": 0.0032,
                "cache_write_cost_usd": written * 5 / 1_000_000,
                "cached_read_cost_usd": cached * 0.4 / 1_000_000,
                "output_cost_usd": 0.0004,
                "estimated_total_cost_usd": 0.0046,
                "estimated_cost_without_caching_usd": 0.0136,
                "estimated_savings_usd": 0.009,
                "cost_status": "model_pricing_estimate_not_official_billing",
                "provider_response_id": f"resp_{family}_{ordinal}",
                "provider_request_id": f"req_{family}_{ordinal}",
                "usage_receipt_digest": "sha256:" + "f" * 64,
                "breakpoint_digests": ["sha256:" + "a" * 64],
                "cache_policy": policy,
            }
            if family == "task_aware_robot_placement_proposal":
                output = RobotPlacementProposalOutput.model_validate(
                    {
                        "candidate_id": "cache_canary_candidate_0001",
                        "pose": {
                            "position_world_m": [0.0, 0.0, 0.0],
                            "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                        },
                        "support_surface_id": "schematic_floor",
                        "rationale": "Exact immutable candidate.",
                        "addressed_blockers": [],
                        "uncertainty": "visual review remains advisory",
                    }
                )
            else:
                output = RobotPlacementVisualReviewOutput.model_validate(
                    {
                        "status": "passed",
                        "robot_supported_by_declared_surface": True,
                        "robot_not_visibly_clipping_site_geometry": True,
                        "robot_faces_task_workspace": True,
                        "task_workspace_visually_reachable": True,
                        "camera_views_are_sufficient": True,
                        "reason": "The schematic visibly satisfies the advisory checks.",
                        "revision_guidance": [],
                    }
                )
            return AgentsSDKInvocationResult(
                output=output,
                provider="openai",
                model="gpt-5.6-sol",
                sdk_version="0.19.1",
                latency_seconds=0.01,
                usage=usage,
                cost_usd=0.0046,
                cost_status="model_pricing_estimate_not_official_billing",
            )

    fake_invoker = FakeInvoker()
    monkeypatch.setattr(canary, "OpenAIAgentsSDKInvoker", lambda _config: fake_invoker)
    monkeypatch.setattr(
        canary,
        "sync_inference_usage_to_webapp",
        lambda **kwargs: {
            "schema_version": "blueprint_openai_inference_usage_sync_result.v1",
            "status": "succeeded",
            "packet_digest": kwargs["packet"]["packet_digest"],
        },
    )

    report = canary.run_production_shape_canary(
        output_dir=tmp_path / "output",
        api_key_file=key_file,
        source_commit="1" * 40,
        max_total_cost_usd=2.0,
        require_webapp_sync=True,
        verify_source_commit=False,
    )

    assert report["status"] == "passed"
    assert report["call_count"] == 8
    assert report["cache_families"] == [
        "robot_placement_visual_review",
        "task_aware_robot_placement_proposal",
    ]
    assert report["candidate_inventory_membership_preserved"] is True
    assert report["gpu_or_vast_resource_used"] is False
    assert report["robot_motion_performed"] is False
    assert counts["task_aware_robot_placement_proposal"] == 4
    assert counts["robot_placement_visual_review"] == 4
