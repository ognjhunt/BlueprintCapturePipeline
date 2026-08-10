from __future__ import annotations

import json
import sys
import types
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from blueprint_pipeline import wam_generated_video_success_label_openai as openai_labeler


RUNTIME_SIGNING_PRIVATE_KEY_FILE_ENV = (
    "BLUEPRINT_WAM_SUCCESS_LABEL_RUNTIME_SIGNING_PRIVATE_KEY_FILE"
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _configure_runtime_signing(tmp_path: Path, monkeypatch) -> None:
    private_key = Ed25519PrivateKey.from_private_bytes(bytes(range(32)))
    private_key_path = tmp_path / "wam-success-label-runtime.pem"
    private_key_path.write_bytes(
        private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    monkeypatch.setenv(
        RUNTIME_SIGNING_PRIVATE_KEY_FILE_ENV,
        str(private_key_path),
    )


def _request(tmp_path: Path) -> Path:
    video = tmp_path / "rollout.mp4"
    video.write_bytes(b"mp4")
    request = tmp_path / "wam_success_label_request.json"
    _write_json(
        request,
        {
            "schema_version": "wam_success_label_request.v1",
            "inference_input_manifest_sha256": "a" * 64,
            "rollouts": [
                {
                    "rollout_id": "rollout_1",
                    "scenario_eval_run_id": "run_1",
                    "policy_id": "policy",
                    "generated_video_path": str(video),
                }
            ],
            "task_prompts": [
                {
                    "scenario_eval_run_id": "run_1",
                    "task_prompt": "Turn the faucet handle on.",
                    "success_check_plan": {
                        "success_requires": [
                            "robot hand reaches the faucet handle",
                            "faucet handle visibly changes state",
                            "target motion is caused by visible robot motion",
                        ],
                        "common_failure_modes": [
                            "target_moves_without_visible_robot_contact",
                            "end_effector_does_not_reach_target",
                        ],
                    },
                },
            ],
        },
    )
    return request


def test_openai_wam_success_labeler_blocks_without_gate_or_key(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.delenv(openai_labeler.GATE_ENV, raising=False)
    monkeypatch.delenv(openai_labeler.SHARED_GATE_ENV, raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY_FILE", str(tmp_path / "missing-openai-key"))
    monkeypatch.setenv("BLUEPRINT_OPENAI_API_KEY_FILE", str(tmp_path / "missing-openai-key"))

    result = openai_labeler.build_openai_wam_success_labels(
        input_path=_request(tmp_path),
        output_path=tmp_path / "out.json",
    )

    assert result["status"] == "blocked"
    assert (
        f"missing_env_{openai_labeler.GATE_ENV}_or_{openai_labeler.SHARED_GATE_ENV}"
        in result["blockers"]
    )
    assert "missing_openai_api_key_or_key_file" in result["blockers"]
    assert "secret-openai-key" not in json.dumps(result, sort_keys=True)


def test_openai_wam_success_labeler_blocks_taskless_generic_request(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv(openai_labeler.GATE_ENV, "true")
    monkeypatch.setenv("OPENAI_API_KEY", "secret-openai-key")
    request = json.loads(_request(tmp_path).read_text(encoding="utf-8"))
    request["task_prompts"] = [{"scenario_eval_run_id": "run_1", "task_prompt": ""}]
    input_path = tmp_path / "taskless_request.json"
    _write_json(input_path, request)

    def fail_sampling(**_kwargs):
        raise AssertionError("taskless request must block before frame sampling")

    monkeypatch.setattr(openai_labeler, "_sample_video_frames", fail_sampling)

    result = openai_labeler.build_openai_wam_success_labels(
        input_path=input_path,
        output_path=tmp_path / "out.json",
    )

    assert result["status"] == "blocked"
    assert (
        "missing_task_prompt_or_task_success_metadata_for_generated_video_success_label"
        in result["blockers"]
    )
    assert result["label_count"] == 0


def test_openai_wam_success_labeler_uses_responses_without_writing_secret(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv(openai_labeler.GATE_ENV, "true")
    monkeypatch.setenv("OPENAI_API_KEY", "secret-openai-key")
    _configure_runtime_signing(tmp_path, monkeypatch)
    monkeypatch.setattr(
        openai_labeler,
        "_sample_video_frames",
        lambda **_kwargs: (
            [
                {
                    "frame_index": 0,
                    "image_url": "data:image/jpeg;base64,ZmFrZQ==",
                    "evidence_ref": "rollout.mp4#frame=0",
                },
                {
                    "frame_index": 8,
                    "image_url": "data:image/jpeg;base64,ZmFrZQ==",
                    "evidence_ref": "rollout.mp4#frame=8",
                },
            ],
            [],
        ),
    )

    class FakeResponses:
        def create(self, *, model, input, max_output_tokens, reasoning):
            assert model == "gpt-test-vision"
            assert max_output_tokens == 900
            assert reasoning == {"effort": "xhigh"}
            content = input[0]["content"]
            prompt = content[0]["text"]
            assert "Turn the faucet handle on" in prompt
            assert "robot_caused_target_motion" in prompt
            assert "target_moves_without_visible_robot_contact" in prompt
            assert "end_effector_does_not_reach_target" in prompt
            assert len([item for item in content if item["type"] == "input_image"]) == 2
            return types.SimpleNamespace(
                output_text=json.dumps(
                    {
                        "success": True,
                        "confidence": 0.82,
                        "rationale": "The hand reaches the target and the handle moves.",
                        "scene_description": "A robot arm manipulates a faucet.",
                        "task_completion_evidence": ["hand on handle", "handle changes angle"],
                        "failure_modes": [],
                        "end_effector_reaches_target": True,
                        "target_state_change_visible": True,
                        "robot_caused_target_motion": True,
                    }
                )
            )

    class FakeOpenAI:
        def __init__(self, *, api_key):
            assert api_key == "secret-openai-key"
            self.responses = FakeResponses()

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=FakeOpenAI))

    output = tmp_path / "wam_success_labels.command.json"
    result = openai_labeler.build_openai_wam_success_labels(
        input_path=_request(tmp_path),
        output_path=output,
        model="gpt-test-vision",
    )

    assert result["status"] == "completed"
    assert result["provider"] == "openai"
    assert result["reasoning_effort"] == "xhigh"
    assert result["label_count"] == 1
    assert result["labels"][0]["success"] is True
    assert result["labels"][0]["label_source"] == "openai_generated_video_frame_judge"
    assert result["labels"][0]["sampled_frame_count"] == 2
    assert result["labels"][0]["end_effector_reaches_target"] is True
    assert result["labels"][0]["target_state_change_visible"] is True
    assert result["labels"][0]["robot_caused_target_motion"] is True
    assert result["labels"][0]["public_claim_upgrade_allowed"] is False
    assert result["inference_input_manifest_sha256"] == "a" * 64
    assert result["inference_attestation"]["signature_verified"] is True
    assert output.is_file()
    assert "secret-openai-key" not in output.read_text(encoding="utf-8")


def test_openai_success_labeler_defaults_to_luna_xhigh() -> None:
    assert openai_labeler.DEFAULT_MODEL == "gpt-5.6-luna"
    assert openai_labeler.OPENAI_REASONING_EFFORT == "xhigh"
