from __future__ import annotations

from pathlib import Path
import shutil
import subprocess

import pytest

from blueprint_pipeline import policy_ranking_successor_cosmos_provider_runtime as runtime


def _row() -> dict[str, object]:
    return {"seed": 0}


def _stream() -> dict[str, object]:
    return {"actions": [[float(column) for column in range(10)] for _ in range(16)]}


def test_server_command_pins_pipeline_revision_dtype_and_guardrail_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        runtime.shutil, "which", lambda name: "/usr/bin/vllm" if name == "vllm" else None
    )

    command = runtime._server_command()

    assert command[:4] == ["/usr/bin/vllm", "serve", "nvidia/Cosmos3-Nano", "--revision"]
    assert runtime.CHECKPOINT_REVISION in command
    assert command[command.index("--model-class-name") + 1] == "Cosmos3OmniDiffusersPipeline"
    assert command[command.index("--dtype") + 1] == "bfloat16"
    assert "--omni" in command
    assert "--no-guardrails" in command
    assert "--allowed-local-media-path" not in command


def test_retained_server_reuses_exact_healthy_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    retained = tmp_path / "retained"
    retained.mkdir()
    identity = {
        "pid": 321,
        "process_start_ticks": "444",
        "server_port": runtime.SERVER_PORT,
        "pipeline_class": runtime.PIPELINE_CLASS,
        "checkpoint": runtime.CHECKPOINT,
        "checkpoint_revision": runtime.CHECKPOINT_REVISION,
    }
    runtime.write_json(retained / "server_identity.json", identity)

    class Response:
        ok = True

    monkeypatch.setenv(runtime.RETAIN_SERVER_ENV, "true")
    monkeypatch.setenv(runtime.RETAINED_ROOT_ENV, str(retained))
    monkeypatch.setattr(runtime, "_process_start_ticks", lambda pid: "444")
    monkeypatch.setattr(runtime.requests, "get", lambda *args, **kwargs: Response())
    monkeypatch.setattr(
        runtime.subprocess,
        "Popen",
        lambda *args, **kwargs: pytest.fail("healthy retained server must not restart"),
    )

    process, observed, log, reused = runtime._acquire_server(
        output_dir=tmp_path / "output", environment={}
    )

    assert reused is True
    assert process.pid == 321
    assert process.poll() is None
    assert observed["reused_retained_server"] is True
    assert log is None


def test_retained_server_rejects_changed_checkpoint_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    retained = tmp_path / "retained"
    retained.mkdir()
    identity = {
        "pid": 321,
        "process_start_ticks": "444",
        "server_port": runtime.SERVER_PORT,
        "pipeline_class": runtime.PIPELINE_CLASS,
        "checkpoint": runtime.CHECKPOINT,
        "checkpoint_revision": "wrong",
    }
    runtime.write_json(retained / "server_identity.json", identity)

    monkeypatch.setenv(runtime.RETAIN_SERVER_ENV, "true")
    monkeypatch.setenv(runtime.RETAINED_ROOT_ENV, str(retained))
    monkeypatch.setattr(runtime, "_process_start_ticks", lambda pid: "444")
    assert runtime._retained_server_identity_valid(identity) is False


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
    assert direct["size"] == "640x544"
    assert direct["num_inference_steps"] == "4"
    extra = runtime.json.loads(direct["extra_params"])
    assert extra["action_mode"] == "forward_dynamics"
    assert extra["domain_name"] == "droid_lerobot"
    assert extra["raw_action_dim"] == 10
    assert extra["action_chunk_size"] == 16
    assert extra["action_space"] == "midtrain"
    assert len(extra["action"]) == 16
    assert all(len(row) == 10 for row in extra["action"])


def test_static_video_can_pass_structural_canary_without_passing_motion(
    tmp_path: Path,
) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        pytest.skip("ffmpeg unavailable")
    video = tmp_path / "static.mp4"
    completed = subprocess.run(
        [
            ffmpeg,
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=gray:s=640x544:r=15:d=1.133334",
            "-frames:v",
            "17",
            "-pix_fmt",
            "yuv420p",
            str(video),
        ],
        check=False,
        capture_output=True,
    )
    assert completed.returncode == 0, completed.stderr.decode(errors="replace")

    metrics = runtime._decode_video_metrics(video)

    assert metrics["status"] == "passed"
    assert metrics["structural_status"] == "passed"
    assert metrics["motion_status"] == "failed"
    assert metrics["static_detected"] is True


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


def test_frozen_action_condition_validation_accepts_sorted_bundle_key_order() -> None:
    sorted_conditions = {
        condition: {"actions": []} for condition in sorted(runtime.EXPECTED_CONDITIONS)
    }

    assert tuple(sorted_conditions) != runtime.EXPECTED_CONDITIONS
    assert runtime._action_conditions_match_frozen_contract(sorted_conditions)


@pytest.mark.parametrize(
    "conditions",
    [
        None,
        {},
        {condition: {} for condition in runtime.EXPECTED_CONDITIONS[:-1]},
        {
            **{condition: {} for condition in runtime.EXPECTED_CONDITIONS},
            "unexpected": {},
        },
    ],
)
def test_frozen_action_condition_validation_rejects_missing_or_extra_conditions(
    conditions: object,
) -> None:
    assert not runtime._action_conditions_match_frozen_contract(conditions)


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
