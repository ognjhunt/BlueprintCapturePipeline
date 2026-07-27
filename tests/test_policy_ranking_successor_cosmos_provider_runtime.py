from __future__ import annotations

from pathlib import Path

import pytest

from blueprint_pipeline import policy_ranking_successor_cosmos_provider_runtime as runtime


def _row() -> dict[str, object]:
    return {"seed": 0}


def _stream() -> dict[str, object]:
    return {"actions": [[float(column) for column in range(10)] for _ in range(16)]}


def test_server_command_pins_pipeline_revision_dtype_and_guardrail_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runtime.shutil, "which", lambda name: "/usr/bin/vllm" if name == "vllm" else None)

    command = runtime._server_command()

    assert command[:4] == ["/usr/bin/vllm", "serve", "nvidia/Cosmos3-Nano", "--revision"]
    assert runtime.CHECKPOINT_REVISION in command
    assert command[command.index("--model-class-name") + 1] == "Cosmos3OmniDiffusersPipeline"
    assert command[command.index("--dtype") + 1] == "bfloat16"
    assert "--omni" in command
    assert "--no-guardrails" in command
    assert "--allowed-local-media-path" not in command


def test_direct_and_wrapper_requests_match_exact_pinned_forward_dynamics_contract() -> None:
    direct = runtime._serialize_rollout_request(
        request_row=_row(), action_stream=_stream(), num_inference_steps=4
    )
    wrapper = runtime._serialize_blueprint_wrapper_request(
        request_row=_row(), action_stream=_stream(), num_inference_steps=4
    )

    assert direct == wrapper
    assert direct["model"] == "nvidia/Cosmos3-Nano"
    assert direct["num_frames"] == "17"
    assert direct["num_inference_steps"] == "4"
    extra = runtime.json.loads(direct["extra_params"])
    assert extra["action_mode"] == "forward_dynamics"
    assert extra["domain_name"] == "droid_lerobot"
    assert extra["raw_action_dim"] == 10
    assert extra["action_chunk_size"] == 16
    assert extra["action_space"] == "midtrain"
    assert len(extra["action"]) == 16
    assert all(len(row) == 10 for row in extra["action"])


def test_wrapper_serializer_is_independent_of_direct_serializer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_if_called(**_: object) -> dict[str, object]:
        raise AssertionError("direct serializer must not implement the wrapper boundary")

    monkeypatch.setattr(runtime, "_serialize_rollout_request", fail_if_called)

    wrapper = runtime._serialize_blueprint_wrapper_request(
        request_row=_row(), action_stream=_stream(), num_inference_steps=4
    )

    assert wrapper["model"] == "nvidia/Cosmos3-Nano"
    assert runtime.json.loads(wrapper["extra_params"])["raw_action_dim"] == 10


def test_qualification_and_scientific_request_budgets_are_accounted_separately() -> None:
    assert runtime.QUALIFICATION_CANARY_REQUEST_COUNT == 2
    assert runtime.SCIENTIFIC_MATRIX_REQUEST_COUNT == 10
    assert runtime.TOTAL_INITIAL_GENERATION_REQUEST_COUNT == 12


@pytest.mark.parametrize(
    "actions, expected",
    [
        ([[0.0] * 10] * 15, "action_chunk_row_count_invalid"),
        ([[0.0] * 9] * 16, "action_chunk_raw_dimension_invalid"),
    ],
)
def test_request_serialization_fails_closed_on_wrong_action_shape(
    actions: list[list[float]], expected: str
) -> None:
    with pytest.raises(ValueError, match=expected):
        runtime._serialize_rollout_request(
            request_row=_row(),
            action_stream={"actions": actions},
            num_inference_steps=4,
        )


def test_sync_submit_writes_direct_mp4_response(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    observation = tmp_path / "observation.png"
    observation.write_bytes(b"png")
    output = tmp_path / "output.mp4"
    observed: dict[str, object] = {}

    class Response:
        status_code = 200
        content = b"mp4-payload"
        headers = {"content-type": "video/mp4"}

        @staticmethod
        def raise_for_status() -> None:
            return None

    def fake_post(url: str, **kwargs: object) -> Response:
        observed.update({"url": url, **kwargs})
        return Response()

    monkeypatch.setattr(runtime.requests, "post", fake_post)
    request = runtime._serialize_rollout_request(
        request_row=_row(), action_stream=_stream(), num_inference_steps=4
    )

    result = runtime._submit_rollout(
        serialized_request=request,
        initial_observation=observation,
        output_path=output,
    )

    assert observed["url"] == "http://127.0.0.1:8001/v1/videos/sync"
    assert observed["data"] == request
    assert output.read_bytes() == b"mp4-payload"
    assert result["endpoint"] == "/v1/videos/sync"
    assert result["content_type"] == "video/mp4"
