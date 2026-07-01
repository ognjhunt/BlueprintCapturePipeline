from __future__ import annotations

import json
import sys
import types
from pathlib import Path

from blueprint_pipeline import wam_episode_consistency_label_openai as consistency_labeler


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


def test_openai_wam_episode_consistency_blocks_without_gate_or_key(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.delenv(consistency_labeler.GATE_ENV, raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY_FILE", raising=False)
    monkeypatch.delenv("BLUEPRINT_OPENAI_API_KEY_FILE", raising=False)

    result = consistency_labeler.build_openai_wam_episode_consistency_labels(
        input_path=_request(tmp_path),
        output_path=tmp_path / "out.json",
    )

    assert result["status"] == "blocked"
    assert f"missing_env_{consistency_labeler.GATE_ENV}" in result["blockers"]
    assert "missing_openai_api_key_or_key_file" in result["blockers"]
    serialized = json.dumps(result, sort_keys=True)
    assert "secret-openai-key" not in serialized


def test_openai_wam_episode_consistency_uses_responses_without_writing_secret(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv(consistency_labeler.GATE_ENV, "true")
    monkeypatch.setenv("OPENAI_API_KEY", "secret-openai-key")
    monkeypatch.setattr(
        consistency_labeler,
        "_sample_video_frames",
        lambda **_: (
            [
                {
                    "frame_index": 0,
                    "image_url": "data:image/jpeg;base64,aaa",
                    "evidence_ref": "rollout.mp4#frame=0",
                },
                {
                    "frame_index": 4,
                    "image_url": "data:image/jpeg;base64,bbb",
                    "evidence_ref": "rollout.mp4#frame=4",
                },
            ],
            [],
        ),
    )

    openai_module = types.ModuleType("openai")

    class FakeResponses:
        def create(self, *, model, input, max_output_tokens):
            assert model == "openai-test-model"
            assert max_output_tokens == 800
            content = input[0]["content"]
            assert content[0]["type"] == "input_text"
            assert "forward/inverse consistent" in content[0]["text"]
            assert "Approach the target" in content[0]["text"]
            assert [item["type"] for item in content[1:]] == ["input_image", "input_image"]
            return types.SimpleNamespace(
                output_text=json.dumps(
                    {
                        "forward_consistent": True,
                        "inverse_consistent": True,
                        "confidence": 0.91,
                        "rationale": "The sampled frames follow the trace context.",
                        "visible_action_alignment_evidence": ["visible motion aligns"],
                        "inconsistency_evidence": [],
                        "task_success_proven": True,
                        "policy_success_proven": True,
                        "rank_fidelity_result_proven": True,
                        "deployment_readiness_proven": True,
                        "sensor_truth_proven": True,
                        "external_validation_proven": True,
                        "public_claim_upgrade_allowed": True,
                    }
                )
            )

    class FakeClient:
        def __init__(self, *, api_key):
            assert api_key == "secret-openai-key"
            self.responses = FakeResponses()

    openai_module.OpenAI = FakeClient
    monkeypatch.setitem(sys.modules, "openai", openai_module)

    output = tmp_path / "wam_episode_consistency.command.json"
    result = consistency_labeler.build_openai_wam_episode_consistency_labels(
        input_path=_request(tmp_path),
        output_path=output,
        model="openai-test-model",
    )

    assert result["status"] == "completed"
    assert result["provider"] == "openai_wam_episode_consistency_judge"
    assert result["rollout_check_count"] == 1
    assert result["rollout_checks"][0]["forward_consistent"] is True
    assert result["rollout_checks"][0]["inverse_consistent"] is True
    assert result["rollout_checks"][0]["label_source"] == "openai_wam_episode_consistency_judge"
    assert result["rollout_checks"][0]["sampled_frame_count"] == 2
    assert result["rollout_checks"][0]["public_claim_upgrade_allowed"] is False
    assert result["rollout_checks"][0]["task_success_proven"] is False
    assert result["rollout_checks"][0]["policy_success_claimed_from_consistency"] is False
    assert result["rollout_checks"][0]["task_success_claimed_from_consistency"] is False
    assert result["rollout_checks"][0]["rank_fidelity_claimed_from_consistency"] is False
    assert result["rollout_checks"][0]["deployment_readiness_claimed_from_consistency"] is False
    assert result["rollout_checks"][0]["sensor_truth_claimed_from_consistency"] is False
    assert result["rollout_checks"][0]["external_validation_claimed_from_consistency"] is False
    assert result["claim_boundary"][
        "forward_inverse_consistency_is_reliability_review_signal_only"
    ] is True
    assert result["claim_boundary"][
        "forward_inverse_consistency_does_not_upgrade_evaluator_bounded_policy_ranking"
    ] is True
    assert result["claim_boundary"]["forward_inverse_consistency_is_not_external_validation"] is True
    assert output.is_file()
    serialized = output.read_text(encoding="utf-8")
    assert "secret-openai-key" not in serialized
