from __future__ import annotations

import json
import sys
import types
from pathlib import Path

from blueprint_pipeline import wam_generated_video_success_label_gemini as gemini_labeler


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _request(tmp_path: Path) -> Path:
    video = tmp_path / "rollout.mp4"
    video.write_bytes(b"mp4")
    request = tmp_path / "wam_success_label_request.json"
    _write_json(
        request,
        {
            "schema_version": "wam_success_label_request.v1",
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
                    "task_prompt": "Approach the target and stop.",
                }
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
    monkeypatch.delenv("GEMINI_API_KEY_FILE", raising=False)
    monkeypatch.delenv("GOOGLE_GENAI_API_KEY_FILE", raising=False)
    monkeypatch.delenv("GOOGLE_AI_API_KEY_FILE", raising=False)

    result = gemini_labeler.build_gemini_wam_success_labels(
        input_path=_request(tmp_path),
        output_path=tmp_path / "out.json",
    )

    assert result["status"] == "blocked"
    assert f"missing_env_{gemini_labeler.GATE_ENV}" in result["blockers"]
    assert "missing_gemini_google_genai_or_google_ai_api_key_or_key_file" in result["blockers"]
    serialized = json.dumps(result, sort_keys=True)
    assert "secret-gemini-key" not in serialized


def test_gemini_wam_success_labeler_uses_sdk_without_writing_secret(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv(gemini_labeler.GATE_ENV, "true")
    monkeypatch.setenv("GEMINI_API_KEY", "secret-gemini-key")

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
            assert "Approach the target" in contents[0]
            assert contents[1]["mime_type"] == "video/mp4"
            return types.SimpleNamespace(
                text=json.dumps(
                    {
                        "success": True,
                        "confidence": 0.87,
                        "rationale": "The robot reaches the target.",
                        "task_completion_evidence": ["target visible"],
                        "failure_modes": [],
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
    assert result["labels"][0]["public_claim_upgrade_allowed"] is False
    assert output.is_file()
    serialized = output.read_text(encoding="utf-8")
    assert "secret-gemini-key" not in serialized
