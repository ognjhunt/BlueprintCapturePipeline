from __future__ import annotations

import json

from blueprint_pipeline.task_evaluation_openai_inference_usage import (
    build_placement_inference_usage_packet,
    sync_inference_usage_to_webapp,
)


def _placement_receipt() -> dict:
    usage = {
        "model": "gpt-5.6-sol",
        "input_tokens": 2_000,
        "cached_tokens": 1_200,
        "cache_write_tokens": 0,
        "uncached_input_tokens": 800,
        "output_tokens": 20,
        "reasoning_tokens": 5,
        "cache_hit_ratio": 0.6,
        "uncached_input_cost_usd": 0.0032,
        "cache_write_cost_usd": 0.0,
        "cached_read_cost_usd": 0.00048,
        "output_cost_usd": 0.0004,
        "estimated_total_cost_usd": 0.00408,
        "estimated_cost_without_caching_usd": 0.0084,
        "estimated_savings_usd": 0.00432,
        "cost_status": "model_pricing_estimate_not_official_billing",
        "provider_response_id": "resp_cache_read",
        "provider_request_id": "req_cache_read",
        "usage_receipt_digest": "sha256:" + "b" * 64,
        "breakpoint_digests": [
            "sha256:" + "c" * 64,
            "sha256:" + "9" * 64,
        ],
        "cache_policy": {
            "status": "enabled",
            "model_family": "gpt-5.6-sol",
            "family": "task_aware_robot_placement_proposal",
            "contract_version": "robot-placement-proposal-v2",
            "stable_prefix_digest": "sha256:" + "c" * 64,
            "policy_digest": "sha256:" + "d" * 64,
            "privacy_scope": "task_evaluation_rights_admitted",
            "processing_region": "default",
            "decision_reason": "expected_cached_cost_lower",
            "cache_key_digest": "sha256:" + "a" * 64,
            "economics": {"stable_prefix_tokens": 1_200},
        },
    }
    return {
        "run_id": "placement-run",
        "receipt_digest": "sha256:" + "e" * 64,
        "rounds": [{"proposal_usage": usage}],
    }


def test_packet_projects_only_digest_and_usage_evidence() -> None:
    packet = build_placement_inference_usage_packet(
        placement_receipt=_placement_receipt(),
        packet_run_id="website-run",
        launch_id="launch-1",
        source_commit="f" * 40,
    )
    call = packet["calls"][0]

    assert call["cached_tokens"] == 1_200
    assert call["cache_write_tokens"] == 0
    assert call["uncached_input_tokens"] == 800
    assert call["cache_key_digest"].startswith("sha256:")
    assert "cache_key" not in call
    assert call["raw_prompt_recorded"] is False
    assert call["dynamic_content_before_breakpoint"] is False
    assert "blueprint:cache:v1" not in json.dumps(packet)


def test_signed_sync_requires_exact_response_binding(monkeypatch) -> None:
    packet = build_placement_inference_usage_packet(
        placement_receipt=_placement_receipt(),
        packet_run_id="website-run",
        launch_id=None,
        source_commit="f" * 40,
    )
    captured: dict = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self):
            return json.dumps(
                {
                    "schema_version": "blueprint_openai_inference_usage_ingest_receipt.v1",
                    "status": "created",
                    "run_id": packet["run_id"],
                    "launch_id": packet["launch_id"],
                    "source_commit": packet["source_commit"],
                    "packet_digest": packet["packet_digest"],
                    "call_count": len(packet["calls"]),
                }
            ).encode()

    def urlopen(request, *, timeout):
        captured["request"] = request
        captured["timeout"] = timeout
        return Response()

    monkeypatch.setattr(
        "blueprint_pipeline.task_evaluation_openai_inference_usage.urllib_request.urlopen",
        urlopen,
    )
    result = sync_inference_usage_to_webapp(
        packet=packet,
        endpoint_url="https://tryblueprint.io/api/internal/pipeline/openai-inference-usage",
        token="test-sync-token",
    )

    assert result["status"] == "succeeded"
    assert captured["request"].headers["X-blueprint-pipeline-signature"].startswith(
        "sha256="
    )
    assert captured["timeout"] == 10.0
