from __future__ import annotations

import json
import sys
import types
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from blueprint_pipeline import wam_generated_video_success_label_gemini as gemini_labeler


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
        gemini_labeler.RUNTIME_SIGNING_PRIVATE_KEY_FILE_ENV,
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
                    "task_prompt": "Open the articulated target.",
                    "success_check_plan": {
                        "success_requires": [
                            "robot hand reaches the specified pull or handle",
                            "articulated target visibly changes from closed to open",
                            "target motion is caused by the visible robot motion",
                        ],
                        "common_failure_modes": [
                            "target_moves_without_visible_robot_contact",
                            "target_already_open_in_first_frame",
                        ],
                    },
                },
            ],
        },
    )
    return request


def test_gemini_wam_success_labeler_blocks_without_gate_or_key(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.delenv(gemini_labeler.GATE_ENV, raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_GENAI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_AI_API_KEY", raising=False)
    # Override every file path so this no-key test remains hermetic on
    # developer machines that intentionally provision a default secret file.
    monkeypatch.setenv("GEMINI_API_KEY_FILE", str(tmp_path / "missing-gemini-key"))
    monkeypatch.setenv("GOOGLE_GENAI_API_KEY_FILE", str(tmp_path / "missing-google-genai-key"))
    monkeypatch.setenv("GOOGLE_AI_API_KEY_FILE", str(tmp_path / "missing-google-ai-key"))

    result = gemini_labeler.build_gemini_wam_success_labels(
        input_path=_request(tmp_path),
        output_path=tmp_path / "out.json",
    )

    assert result["status"] == "blocked"
    assert f"missing_env_{gemini_labeler.GATE_ENV}" in result["blockers"]
    assert "missing_gemini_google_genai_or_google_ai_api_key_or_key_file" in result["blockers"]
    serialized = json.dumps(result, sort_keys=True)
    assert "secret-gemini-key" not in serialized


def test_gemini_task_success_context_requires_prompt_or_metadata(tmp_path: Path) -> None:
    request = {
        "rollouts": [
            {
                "rollout_id": "rollout_1",
                "scenario_eval_run_id": "run_1",
                "generated_video_path": str(tmp_path / "rollout.mp4"),
            }
        ],
        "task_prompts": [{"scenario_eval_run_id": "run_1", "task_prompt": ""}],
    }
    rollout = request["rollouts"][0]
    task_record = gemini_labeler._rollout_task_prompt_record(request, rollout)

    assert (
        gemini_labeler._has_task_success_context(
            request=request,
            rollout=rollout,
            task_record=task_record,
        )
        is False
    )

    request["success_label_contract"] = {
        "strict_task_success_requirements": ["visible_robot_end_effector_reaches_task_target"]
    }
    assert (
        gemini_labeler._has_task_success_context(
            request=request,
            rollout=rollout,
            task_record=task_record,
        )
        is False
    )

    request["task_prompts"][0]["success_check_plan"] = {
        "success_requires": ["robot hand reaches the faucet handle"]
    }
    task_record = gemini_labeler._rollout_task_prompt_record(request, rollout)
    assert (
        gemini_labeler._has_task_success_context(
            request=request,
            rollout=rollout,
            task_record=task_record,
        )
        is True
    )


def test_gemini_provider_error_maps_invalid_api_key() -> None:
    blocker = gemini_labeler._provider_error_blocker(
        RuntimeError("400 INVALID_ARGUMENT API_KEY_INVALID API key not valid")
    )

    assert blocker == "gemini_authentication_failed"


def test_runtime_attestation_requires_digest_and_signing_key(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.delenv(
        gemini_labeler.RUNTIME_SIGNING_PRIVATE_KEY_FILE_ENV,
        raising=False,
    )
    completed = {"status": "completed", "labels": []}

    invalid_digest = gemini_labeler.attach_success_label_runtime_attestation(
        completed,
        inference_input_manifest_sha256="z" * 64,
        output_dir=tmp_path,
    )
    missing_key = gemini_labeler.attach_success_label_runtime_attestation(
        completed,
        inference_input_manifest_sha256="a" * 64,
        output_dir=tmp_path,
    )

    assert invalid_digest["status"] == "blocked"
    assert "success_label_inference_input_manifest_sha256_invalid" in invalid_digest["blockers"]
    assert missing_key["status"] == "blocked"
    assert "success_label_runtime_signing_key_missing" in missing_key["blockers"]


def test_gemini_wam_success_labeler_uses_sdk_without_writing_secret(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv(gemini_labeler.GATE_ENV, "true")
    monkeypatch.setenv("GEMINI_API_KEY", "secret-gemini-key")
    _configure_runtime_signing(tmp_path, monkeypatch)

    google_module = types.ModuleType("google")
    genai_module = types.ModuleType("google.genai")
    types_module = types.ModuleType("google.genai.types")

    class FakePart:
        @staticmethod
        def from_bytes(*, data, mime_type):
            assert data == b"mp4"
            assert mime_type == "video/mp4"
            return {"data": data, "mime_type": mime_type}

    class FakeGenerateContentConfig:
        def __init__(self, *, response_mime_type):
            assert response_mime_type == "application/json"

    class FakeModels:
        def generate_content(self, *, model, contents, config):
            assert model == "gemini-test-flash-model"
            assert "Open the articulated target" in contents[0]
            assert "robot_caused_target_motion" in contents[0]
            assert "target_moves_without_visible_robot_contact" in contents[0]
            assert "target_already_open_in_first_frame" in contents[0]
            assert "fridge_door_moves_without_visible_robot_contact" not in contents[0]
            assert "For faucet tasks" not in contents[0]
            assert contents[1]["mime_type"] == "video/mp4"
            return types.SimpleNamespace(
                text=json.dumps(
                    {
                        "success": True,
                        "confidence": 0.87,
                        "rationale": "The robot reaches the target.",
                        "task_completion_evidence": ["target visible"],
                        "failure_modes": [],
                        "end_effector_reaches_target": True,
                        "target_state_change_visible": True,
                        "robot_caused_target_motion": True,
                    }
                )
            )

    class FakeClient:
        def __init__(self, *, api_key):
            assert api_key == "secret-gemini-key"
            self.models = FakeModels()

    types_module.Part = FakePart
    types_module.GenerateContentConfig = FakeGenerateContentConfig
    genai_module.Client = FakeClient
    genai_module.types = types_module
    google_module.genai = genai_module
    monkeypatch.setitem(sys.modules, "google", google_module)
    monkeypatch.setitem(sys.modules, "google.genai", genai_module)
    monkeypatch.setitem(sys.modules, "google.genai.types", types_module)

    output = tmp_path / "wam_success_labels.command.json"
    result = gemini_labeler.build_gemini_wam_success_labels(
        input_path=_request(tmp_path),
        output_path=output,
        model="gemini-test-flash-model",
    )

    assert result["status"] == "completed"
    assert result["provider"] == "gemini"
    assert result["label_count"] == 1
    assert result["labels"][0]["success"] is True
    assert result["labels"][0]["label_source"] == "gemini_generated_video_judge"
    assert result["labels"][0]["end_effector_reaches_target"] is True
    assert result["labels"][0]["target_state_change_visible"] is True
    assert result["labels"][0]["robot_caused_target_motion"] is True
    assert (
        "target_moves_without_visible_robot_contact"
        in result["labels"][0]["task_success_criteria"]["common_failure_modes"]
    )
    assert (
        result["labels"][0]["task_success_criteria"]["task_specific_rules_source"]
        == "request_or_rollout_metadata"
    )
    assert result["labels"][0]["task_success_criteria"]["hardcoded_task_family_rules_used"] is False
    assert result["labels"][0]["public_claim_upgrade_allowed"] is False
    assert result["inference_input_manifest_sha256"] == "a" * 64
    assert result["inference_attestation"]["signature_verified"] is True
    assert output.is_file()
    serialized = output.read_text(encoding="utf-8")
    assert "secret-gemini-key" not in serialized


def test_gemini_wam_success_labeler_prefers_sampled_video_frames(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv(gemini_labeler.GATE_ENV, "true")
    monkeypatch.setenv("GEMINI_API_KEY", "secret-gemini-key")
    _configure_runtime_signing(tmp_path, monkeypatch)
    monkeypatch.setattr(
        gemini_labeler,
        "_sample_video_frames",
        lambda **_kwargs: (
            [
                {
                    "frame_index": 0,
                    "jpeg_bytes": b"frame-zero",
                    "mime_type": "image/jpeg",
                    "evidence_ref": "rollout.mp4#frame=0",
                },
                {
                    "frame_index": 8,
                    "jpeg_bytes": b"frame-eight",
                    "mime_type": "image/jpeg",
                    "evidence_ref": "rollout.mp4#frame=8",
                },
            ],
            [],
        ),
    )

    google_module = types.ModuleType("google")
    genai_module = types.ModuleType("google.genai")
    types_module = types.ModuleType("google.genai.types")

    class FakePart:
        @staticmethod
        def from_bytes(*, data, mime_type):
            assert data in {b"frame-zero", b"frame-eight"}
            assert mime_type == "image/jpeg"
            return {"data": data, "mime_type": mime_type}

    class FakeGenerateContentConfig:
        def __init__(self, *, response_mime_type):
            assert response_mime_type == "application/json"

    class FakeModels:
        def generate_content(self, *, model, contents, config):
            assert model == "gemini-test-flash-model"
            prompt = contents[0]
            assert "Open the articulated target" in prompt
            assert "sampled_frame_indices" in prompt
            assert (
                len(
                    [
                        item
                        for item in contents
                        if isinstance(item, dict) and item["mime_type"] == "image/jpeg"
                    ]
                )
                == 2
            )
            return types.SimpleNamespace(
                text=json.dumps(
                    {
                        "success": False,
                        "confidence": 0.76,
                        "rationale": "The target motion is not caused by visible robot contact.",
                        "task_completion_evidence": [],
                        "failure_modes": ["target_motion_not_robot_caused"],
                        "end_effector_reaches_target": False,
                        "target_state_change_visible": True,
                        "robot_caused_target_motion": False,
                    }
                )
            )

    class FakeClient:
        def __init__(self, *, api_key):
            assert api_key == "secret-gemini-key"
            self.models = FakeModels()

    types_module.Part = FakePart
    types_module.GenerateContentConfig = FakeGenerateContentConfig
    genai_module.Client = FakeClient
    genai_module.types = types_module
    google_module.genai = genai_module
    monkeypatch.setitem(sys.modules, "google", google_module)
    monkeypatch.setitem(sys.modules, "google.genai", genai_module)
    monkeypatch.setitem(sys.modules, "google.genai.types", types_module)

    result = gemini_labeler.build_gemini_wam_success_labels(
        input_path=_request(tmp_path),
        output_path=tmp_path / "wam_success_labels.command.json",
        model="gemini-test-flash-model",
        max_frames=2,
    )

    assert result["status"] == "completed"
    assert result["label_count"] == 1
    assert result["sampled_rollouts"][0]["sampled_frame_count"] == 2
    assert result["labels"][0]["success"] is False
    assert result["labels"][0]["video_evidence_mode"] == "sampled_frames"
    assert result["labels"][0]["sampled_frame_count"] == 2
    assert result["labels"][0]["sampled_frame_indices"] == [0, 8]
    assert "rollout.mp4#frame=8" in result["labels"][0]["evidence_refs"]
