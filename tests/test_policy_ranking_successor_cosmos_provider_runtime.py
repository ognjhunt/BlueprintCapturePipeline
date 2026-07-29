from __future__ import annotations

from pathlib import Path
import json
import shutil
import subprocess

import pytest

from blueprint_pipeline import policy_ranking_successor_cosmos_provider_runtime as runtime


def _row() -> dict[str, object]:
    return {"seed": 0}


def _stream() -> dict[str, object]:
    return {"actions": [[float(column) for column in range(10)] for _ in range(16)]}


def _droid_reference_manifest() -> dict[str, object]:
    return {
        "request_contract": {
            "model": runtime.CHECKPOINT,
            "checkpoint_revision": runtime.CHECKPOINT_REVISION,
            "endpoint": "/v1/videos",
            "prompt": " ",
            "num_frames": 17,
            "fps": 15,
            "size": "640x540",
            "num_inference_steps": 30,
            "guidance_scale": 1.0,
            "flow_shift": 10.0,
            "seed": 0,
            "extra_params": {
                "action_mode": "forward_dynamics",
                "domain_name": "droid_lerobot",
                "action_chunk_size": 16,
                "image_size": 480,
                "view_point": "concat_view",
                "guardrails": False,
            },
        }
    }


def test_droid_reference_schema_requires_geometry_amendment_v2() -> None:
    assert runtime.DROID_REFERENCE_SCHEMA_VERSION.endswith(".v2")


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
        request_row=_row(),
        action_stream=_stream(),
        num_inference_steps=4,
        task_instruction="Pick up the bottle.",
    )
    wrapper = runtime._serialize_blueprint_wrapper_request(
        request_row=_row(),
        action_stream=_stream(),
        num_inference_steps=4,
        task_instruction="Pick up the bottle.",
    )

    assert direct == wrapper
    assert direct["model"] == "nvidia/Cosmos3-Nano"
    assert direct["num_frames"] == "17"
    assert direct["size"] == "640x544"
    assert direct["num_inference_steps"] == "4"
    assert direct["prompt"] == "Pick up the bottle."
    extra = runtime.json.loads(direct["extra_params"])
    assert extra["action_mode"] == "forward_dynamics"
    assert extra["domain_name"] == "droid_lerobot"
    assert extra["raw_action_dim"] == 10
    assert extra["action_chunk_size"] == 16
    assert extra["action_space"] == "midtrain"
    assert len(extra["action"]) == 16
    assert all(len(row) == 10 for row in extra["action"])


def test_request_serializer_rejects_generic_or_missing_task_prompt() -> None:
    for prompt in ("", "A robot manipulates an object."):
        with pytest.raises(ValueError, match="task_specific|generic_robot"):
            runtime._serialize_rollout_request(
                request_row=_row(),
                action_stream=_stream(),
                num_inference_steps=4,
                task_instruction=prompt,
            )


def test_positive_control_serializer_matches_frozen_official_contract() -> None:
    action_spec = {
        "prompt": "Pickup items in the supermarket",
        "fps": 10,
        "action_chunk_size": 16,
        "domain_name": "agibotworld",
        "image_size": 480,
        "view_point": "concat_view",
    }
    request = runtime._serialize_positive_control_request(
        action_chunk=[[0.0] * 29 for _ in range(16)],
        action_spec=action_spec,
    )

    assert request["fps"] == "10"
    assert request["size"] == "640x720"
    assert request["num_frames"] == "17"
    assert request["guidance_scale"] == "1.0"
    assert request["flow_shift"] == "10.0"
    extra = runtime.json.loads(request["extra_params"])
    assert extra["domain_name"] == "agibotworld"
    assert extra["action_chunk_size"] == 16
    assert len(extra["action"]) == 16
    assert all(len(row) == 29 for row in extra["action"])


def test_positive_control_serializer_rejects_wrong_action_shape() -> None:
    with pytest.raises(ValueError, match="positive_control_action_dimension_invalid"):
        runtime._serialize_positive_control_request(
            action_chunk=[[0.0] * 10 for _ in range(16)],
            action_spec={
                "prompt": "Pickup items in the supermarket",
                "fps": 10,
                "action_chunk_size": 16,
            },
        )


def test_droid_reference_serializer_matches_current_official_async_contract() -> None:
    request = runtime._serialize_droid_reference_request(
        manifest=_droid_reference_manifest(), action_stream=_stream()
    )

    assert request["model"] == "nvidia/Cosmos3-Nano"
    assert request["prompt"] == " "
    assert request["size"] == "640x540"
    assert request["num_inference_steps"] == "30"
    extra = json.loads(request["extra_params"])
    assert set(extra) == {
        "action_mode",
        "domain_name",
        "action_chunk_size",
        "image_size",
        "view_point",
        "guardrails",
        "action",
    }
    assert "raw_action_dim" not in extra
    assert "action_space" not in extra


def test_droid_reference_serializer_fails_closed_on_contract_drift() -> None:
    manifest = _droid_reference_manifest()
    manifest["request_contract"]["size"] = "640x544"  # type: ignore[index]

    with pytest.raises(ValueError, match="request_contract_changed"):
        runtime._serialize_droid_reference_request(manifest=manifest, action_stream=_stream())


def test_powered_droid_serializer_matches_official_blank_prompt_contract() -> None:
    request = runtime._serialize_powered_droid_request(action_stream=_stream(), seed=1)

    assert request["prompt"] == " "
    assert request["size"] == "640x540"
    assert request["fps"] == "15"
    assert request["num_inference_steps"] == "30"
    assert request["seed"] == "1"
    extra = json.loads(request["extra_params"])
    assert extra["domain_name"] == "droid_lerobot"
    assert extra["action_chunk_size"] == 16
    assert extra["guardrails"] is False
    assert len(extra["action"]) == 16


def test_powered_droid_serializer_rejects_wrong_action_shape() -> None:
    stream = _stream()
    stream["actions"] = stream["actions"][:-1]
    with pytest.raises(ValueError, match="powered_droid_action_shape_invalid"):
        runtime._serialize_powered_droid_request(action_stream=stream, seed=0)


def test_server_environment_disables_xet_for_all_runtime_launches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HF_HUB_DISABLE_XET", "0")

    environment = runtime._server_environment()

    assert environment["HF_HUB_DISABLE_XET"] == "1"
    assert environment["HF_HUB_DISABLE_TELEMETRY"] == "1"


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


def test_pinned_droid_decoded_geometry_passes_manifest_bound_structure(tmp_path: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        pytest.skip("ffmpeg unavailable")
    video = tmp_path / "droid-decoded.mp4"
    completed = subprocess.run(
        [
            ffmpeg,
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=gray:s=640x528:r=15:d=1.133334",
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

    metrics = runtime._decode_video_metrics(
        video,
        expected_width=640,
        expected_height=528,
        expected_frames=17,
        expected_fps=15.0,
    )

    assert metrics["structural_status"] == "passed"
    assert not any("unexpected_video_dimensions" in item for item in metrics["blockers"])


def test_wrong_frame_rate_and_duration_fail_structural_canary(tmp_path: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        pytest.skip("ffmpeg unavailable")
    video = tmp_path / "wrong-rate.mp4"
    completed = subprocess.run(
        [
            ffmpeg,
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=gray:s=640x544:r=12:d=1.416667",
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

    assert metrics["structural_status"] == "blocked"
    assert "unexpected_video_frame_rate:12.0" in metrics["blockers"]
    assert any(item.startswith("unexpected_video_duration:") for item in metrics["blockers"])


def test_wrapper_serializer_is_independent_of_direct_serializer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_if_called(**_: object) -> dict[str, object]:
        raise AssertionError("direct serializer must not implement the wrapper boundary")

    monkeypatch.setattr(runtime, "_serialize_rollout_request", fail_if_called)

    wrapper = runtime._serialize_blueprint_wrapper_request(
        request_row=_row(),
        action_stream=_stream(),
        num_inference_steps=4,
        task_instruction="Pick up the bottle.",
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


def test_phase_b_condition_declaration_requires_shifted_control() -> None:
    inventory = {"required_conditions": list(runtime.PHASE_B_EXPECTED_CONDITIONS)}
    expected = runtime._declared_expected_conditions(inventory)
    assert expected == runtime.PHASE_B_EXPECTED_CONDITIONS
    conditions = {condition: {} for condition in runtime.PHASE_B_EXPECTED_CONDITIONS}
    assert runtime._action_conditions_match_frozen_contract(conditions, expected)
    conditions.pop("shifted")
    assert not runtime._action_conditions_match_frozen_contract(conditions, expected)


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
            task_instruction="Pick up the bottle.",
        )


def test_sync_submit_materializes_parent_and_writes_direct_mp4_response(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    observation = tmp_path / "observation.png"
    observation.write_bytes(b"png")
    output = tmp_path / "fresh" / "condition" / "output.mp4"
    assert not output.parent.exists()
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
        request_row=_row(),
        action_stream=_stream(),
        num_inference_steps=4,
        task_instruction="Pick up the bottle.",
    )

    result = runtime._submit_rollout(
        serialized_request=request,
        initial_observation=observation,
        output_path=output,
    )

    assert observed["url"] == "http://127.0.0.1:8001/v1/videos/sync"
    assert observed["data"] == request
    assert output.parent.is_dir()
    assert output.read_bytes() == b"mp4-payload"
    assert result["endpoint"] == "/v1/videos/sync"
    assert result["content_type"] == "video/mp4"


def test_run_dispatches_reference_bundle_before_legacy_input_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime_root = tmp_path / "provider_runtime"
    reference = runtime_root / runtime.DROID_REFERENCE_DIRECTORY
    reference.mkdir(parents=True)
    (reference / "canary_manifest.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(runtime, "__file__", str(runtime_root / "runner.py"))
    monkeypatch.setenv("BLUEPRINT_VAST_PROVIDER_OUTPUT_DIR", str(tmp_path / "output"))
    observed: dict[str, Path] = {}

    def fake_reference_run(*, runtime_dir: Path, output_dir: Path) -> dict[str, object]:
        observed["runtime_dir"] = runtime_dir
        observed["output_dir"] = output_dir
        return {"status": "reference-dispatched"}

    monkeypatch.setattr(runtime, "_run_droid_reference_only", fake_reference_run)

    result = runtime.run()

    assert result == {"status": "reference-dispatched"}
    assert observed["runtime_dir"] == runtime_root
    assert observed["output_dir"] == (tmp_path / "output").resolve()


def test_run_dispatches_powered_bundle_before_reference_or_legacy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime_root = tmp_path / "provider_runtime"
    powered = runtime_root / runtime.POWERED_DROID_DIRECTORY
    powered.mkdir(parents=True)
    (powered / "packet.json").write_text("{}", encoding="utf-8")
    reference = runtime_root / runtime.DROID_REFERENCE_DIRECTORY
    reference.mkdir()
    (reference / "canary_manifest.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(runtime, "__file__", str(runtime_root / "runner.py"))
    monkeypatch.setenv("BLUEPRINT_VAST_PROVIDER_OUTPUT_DIR", str(tmp_path / "output"))
    observed: dict[str, Path] = {}

    def fake_powered_run(*, runtime_dir: Path, output_dir: Path) -> dict[str, object]:
        observed["runtime_dir"] = runtime_dir
        observed["output_dir"] = output_dir
        return {"status": "powered-dispatched"}

    monkeypatch.setattr(runtime, "_run_powered_droid", fake_powered_run)

    result = runtime.run()

    assert result == {"status": "powered-dispatched"}
    assert observed["runtime_dir"] == runtime_root
