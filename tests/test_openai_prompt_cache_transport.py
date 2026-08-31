from __future__ import annotations

import json

from openai import AsyncOpenAI
from openai.types.responses import Response

from agents import set_default_openai_client
from agents import ModelSettings
from agents.run_internal.prompt_cache_key import PromptCacheKeyResolver

from blueprint_pipeline.task_evaluation_supervisor.agents_sdk import (
    AgentsSDKAgentSpec,
    AgentsSDKCapabilityOutput,
    OpenAIAgentsSDKConfig,
    OpenAIAgentsSDKInvoker,
)


def test_predecessor_sdk_defaults_fragment_by_run_and_leave_implicit_mode() -> None:
    class OfficialModel:
        @staticmethod
        def _supports_default_prompt_cache_key() -> bool:
            return True

    first = PromptCacheKeyResolver().resolve(
        ModelSettings(),
        model=OfficialModel(),
        conversation_id=None,
        session=None,
        group_id="unique-run-a",
    )
    second = PromptCacheKeyResolver().resolve(
        ModelSettings(),
        model=OfficialModel(),
        conversation_id=None,
        session=None,
        group_id="unique-run-b",
    )

    assert first != second
    assert ModelSettings().prompt_cache_options is None


def _response(*, response_id: str, cached: int, written: int) -> Response:
    output = {
        "status": "abstained",
        "summary": "Transport spy completed.",
        "artifact_json": "{}",
        "proposals": [],
        "blockers": [],
        "evidence_refs": [],
        "uncertainty": "none",
    }
    return Response.model_validate(
        {
            "id": response_id,
            "object": "response",
            "created_at": 0,
            "status": "completed",
            "model": "gpt-5.6-sol",
            "output": [
                {
                    "id": f"msg_{response_id}",
                    "type": "message",
                    "status": "completed",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": json.dumps(output),
                            "annotations": [],
                        }
                    ],
                }
            ],
            "usage": {
                "input_tokens": 2_000,
                "input_tokens_details": {
                    "cached_tokens": cached,
                    "cache_write_tokens": written,
                },
                "output_tokens": 20,
                "output_tokens_details": {"reasoning_tokens": 5},
                "total_tokens": 2_020,
            },
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
        }
    )


def test_agents_sdk_serializes_stable_explicit_prefix_and_key(
    monkeypatch,
) -> None:
    requests: list[dict] = []
    responses = [
        _response(response_id="resp_write", cached=0, written=1_200),
        _response(response_id="resp_read", cached=1_200, written=0),
    ]
    client = AsyncOpenAI(api_key="test-only-key")

    async def create(**kwargs):
        requests.append(kwargs)
        return responses.pop(0)

    monkeypatch.setattr(client.responses, "create", create)
    set_default_openai_client(client, use_for_tracing=False)
    monkeypatch.setenv("BLUEPRINT_ALLOW_LIVE_AGENTS_SDK_OPERATORS", "true")
    monkeypatch.delenv("OPENAI_API_KEY_FILE", raising=False)

    stable_prefix = "Stable useful Blueprint authority and output contract. " * 500
    scene_prefix = json.dumps(
        {
            "scene_revision_digest": "sha256:" + "a" * 64,
            "trajectory_digest": "sha256:" + "b" * 64,
        },
        sort_keys=True,
    )
    config = OpenAIAgentsSDKConfig(
        model="gpt-5.6-sol",
        allow_live_invocation=True,
        tracing_disabled=True,
        max_inference_cost_usd=1.0,
    )
    invoker = OpenAIAgentsSDKInvoker(config)

    def invoke(run_id: str):
        return invoker.invoke(
            AgentsSDKAgentSpec(
                run_id=run_id,
                capability="task_aware_robot_placement_proposal",
                name="Blueprint placement transport spy",
                instructions="Return only the declared structured output.",
                model="gpt-5.6-sol",
                max_turns=1,
                max_output_tokens=256,
                max_input_tokens=10_000,
                reasoning_effort="high",
                output_type=AgentsSDKCapabilityOutput,
                stable_developer_prefix=stable_prefix,
                scene_static_prefix=scene_prefix,
                prompt_contract_version="transport-spy-v1",
                stable_prefix_tokens=1_500,
                expected_reuse_count=1,
                expected_reuse_probability=1.0,
                privacy_scope="rights_admitted_test",
                processing_region="default",
                dynamic_suffix_fields=("run_id", "round_index", "image"),
            ),
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": json.dumps(
                                {"run_id": run_id, "round_index": 1},
                                sort_keys=True,
                            ),
                        }
                    ],
                }
            ],
        )

    first = invoke("run-unique-a")
    second = invoke("run-unique-b")

    assert len(requests) == 2
    first_request, second_request = requests
    assert first_request["prompt_cache_key"] == second_request["prompt_cache_key"]
    assert len(first_request["prompt_cache_key"]) <= 64
    options = first_request["prompt_cache_options"]
    options = options.model_dump(mode="json") if hasattr(options, "model_dump") else options
    assert options == {"mode": "explicit", "ttl": "30m"}
    assert [item["role"] for item in first_request["input"]] == [
        "developer",
        "developer",
        "user",
    ]
    assert first_request["input"][0]["content"][0]["prompt_cache_breakpoint"] == {
        "mode": "explicit"
    }
    assert "prompt_cache_breakpoint" not in first_request["input"][1]["content"][0]
    assert "run-unique-a" not in json.dumps(first_request["input"][:2])
    assert "run-unique-a" in json.dumps(first_request["input"][2])
    assert "prompt_cache_breakpoint" not in json.dumps(first_request["input"][2])
    assert first_request["store"] is False
    assert first.usage["cache_write_tokens"] == 1_200
    assert second.usage["cached_tokens"] == 1_200
    assert first.usage["cache_policy"]["family"] == (
        "task_aware_robot_placement_proposal"
    )
    assert first.usage["cache_policy"]["cache_key_digest"].startswith("sha256:")
    assert "cache_key" not in first.usage["cache_policy"]
    assert len(first.usage["breakpoint_digests"]) == 1
