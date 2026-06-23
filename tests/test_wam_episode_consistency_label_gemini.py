from __future__ import annotations

import json
import sys
import types
from pathlib import Path

from blueprint_pipeline import wam_episode_consistency_label_gemini as consistency_labeler


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _request(tmp_path: Path) -> Path:
    video = tmp_path / "rollout.mp4"
    video.write_bytes(b"mp4")
    request = tmp_path / "wam_episode_consistency_request.json"
    _write_json(
        request,
        {
            "schema_version": "wam_episode_consistency_request.v1",
            "rollouts": [
                {
                    "rollout_id": "rollout_1",
                    "scenario_eval_run_id": "run_1",
                    "policy_id": "policy",
                    "model_candidate": "oscar_wam",
                    "generated_video_path": str(video),
                }
            ],
            "task_prompts": [
                {
                    "scenario_eval_run_id": "run_1",
                    "task_prompt": "Approach the target and stop.",
                }
            ],
            "trace_summary": {
                "action_row_count": 1,
                "action_type_counts": [{"action_type": "waypoint", "count": 1}],
            },
        },
    )
    return request


def test_gemini_wam_episode_consistency_blocks_without_gate_or_key(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.delenv(consistency_labeler.GATE_ENV, raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_GENAI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_AI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY_FILE", raising=False)
    monkeypatch.delenv("GOOGLE_GENAI_API_KEY_FILE", raising=False)
    monkeypatch.delenv("GOOGLE_AI_API_KEY_FILE", raising=False)

    result = consistency_labeler.build_gemini_wam_episode_consistency_labels(
        input_path=_request(tmp_path),
        output_path=tmp_path / "out.json",
    )

    assert result["status"] == "blocked"
    assert f"missing_env_{consistency_labeler.GATE_ENV}" in result["blockers"]
    assert "missing_gemini_google_genai_or_google_ai_api_key_or_key_file" in result["blockers"]
    serialized = json.dumps(result, sort_keys=True)
    assert "secret-gemini-key" not in serialized


def test_gemini_wam_episode_consistency_uses_sdk_without_writing_secret(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv(consistency_labeler.GATE_ENV, "true")
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
        def generate_content(self, *, model, contents, config=None):
            assert model == "gemini-test-model"
            assert "forward/inverse consistent" in contents[0]
            assert "Approach the target" in contents[0]
            assert contents[1]["mime_type"] == "video/mp4"
            return types.SimpleNamespace(
                text=json.dumps(
                    {
                        "forward_consistent": True,
                        "inverse_consistent": True,
                        "confidence": 0.93,
                        "rationale": "The rollout motion follows the trace context.",
                        "visible_action_alignment_evidence": ["motion follows waypoint"],
                        "inconsistency_evidence": [],
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

    output = tmp_path / "wam_episode_consistency.command.json"
    result = consistency_labeler.build_gemini_wam_episode_consistency_labels(
        input_path=_request(tmp_path),
        output_path=output,
        model="gemini-test-model",
    )

    assert result["status"] == "completed"
    assert result["provider"] == "gemini_wam_episode_consistency_judge"
    assert result["rollout_check_count"] == 1
    assert result["rollout_checks"][0]["forward_consistent"] is True
    assert result["rollout_checks"][0]["inverse_consistent"] is True
    assert result["rollout_checks"][0]["label_source"] == "gemini_wam_episode_consistency_judge"
    assert result["rollout_checks"][0]["public_claim_upgrade_allowed"] is False
    assert output.is_file()
    serialized = output.read_text(encoding="utf-8")
    assert "secret-gemini-key" not in serialized
