from __future__ import annotations

import base64
import io
import json

import pytest
from PIL import Image

from blueprint_pipeline import haiku_vision_judge as haiku
from blueprint_pipeline import rollout_vision_label_openai as rollout
from blueprint_pipeline import wam_episode_consistency_label_openai as consistency
from blueprint_pipeline import wam_generated_video_success_label_openai as success


def image_url():
    stream = io.BytesIO()
    Image.new("RGB", (16, 16), "red").save(stream, format="JPEG")
    return "data:image/jpeg;base64," + base64.b64encode(stream.getvalue()).decode()


def response(**overrides):
    return {"model": haiku.MODEL, "id": "fixture-response", "stop_reason": "end_turn",
            "content": [{"type": "text", "text": '{"confidence":0.5,"success":null}'}],
            "usage": {"input_tokens": 2000, "output_tokens": 200, "inference_geo": "global"}, **overrides}


def test_native_messages_and_interrupt_safe_reservations(monkeypatch, tmp_path):
    calls = []
    def send(payload, key):
        calls.append(payload)
        assert key == "offline-key"
        assert len(list((tmp_path / "haiku_vision_budget/inference_reservations/reserved").glob("*.json"))) == 1
        return response()
    monkeypatch.setattr(haiku, "_post_message", send)
    content = [{"type": "input_text", "text": "Return visible evidence only."}, {"type": "input_image", "image_url": image_url()}]
    value, receipt = haiku.judge_json(system="JSON only.", content=content, api_key="offline-key", audit_root=tmp_path, capability="test")
    assert value["success"] is None
    assert calls[0]["messages"][0]["content"][1]["source"]["media_type"] == "image/jpeg"
    assert calls[0]["output_config"] == {"effort": "medium"}
    assert not {"reasoning", "input", "temperature", "prompt_cache_options", "tools"} & calls[0].keys()
    assert receipt["usage"]["estimated_total_cost_usd"] == pytest.approx(0.0003)
    with pytest.raises(ValueError, match="prior_inference_reservation"):
        haiku.judge_json(system="JSON only.", content=content, api_key="offline-key", audit_root=tmp_path, capability="test")
    assert len(calls) == 1


@pytest.mark.parametrize("bad", [response(stop_reason="max_tokens"), response(stop_reason="refusal"),
    response(model="claude-opus-5-5"), response(usage={}), response(content=[{"type": "text", "text": "not-json"}])])
def test_invalid_responses_retain_original_evidence_without_retry(monkeypatch, tmp_path, bad):
    calls = []
    monkeypatch.setattr(haiku, "_post_message", lambda *_: calls.append(1) or bad)
    with pytest.raises((RuntimeError, ValueError)):
        haiku.judge_json(system="JSON", content=[{"type": "input_text", "text": "Fixture"}], api_key="offline", audit_root=tmp_path, capability="test")
    assert len(calls) == 1
    saved = list((tmp_path / "haiku_vision_budget/inference_reservations/responses").glob("*.json"))
    assert len(saved) == 1
    assert json.loads(saved[0].read_text()) == bad


def test_budget_and_remote_media_fail_before_request(monkeypatch, tmp_path):
    monkeypatch.setattr(haiku, "_post_message", lambda *_: pytest.fail("must not call provider"))
    for content in [[{"type": "input_text", "text": "x" * 100_000}],
                    [{"type": "input_image", "image_url": "https://example.com/private.jpg"}]]:
        with pytest.raises(RuntimeError):
            haiku.judge_json(system="JSON", content=content, api_key="offline", audit_root=tmp_path, capability="test")


def test_direct_call_sites_use_haiku_and_keep_review_claims(monkeypatch, tmp_path):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "offline-key")
    monkeypatch.setattr(haiku, "_post_message", lambda *_: response())
    frame = tmp_path / "frame.jpg"
    frame.write_bytes(base64.b64decode(image_url().split(",", 1)[1]))
    calls = []
    value = rollout._openai_label(model=haiku.MODEL, label={}, clip={}, keyframe_path=frame, audit_root=tmp_path, provider_calls=calls)
    assert value["success"] is None and calls[0]["provider"] == "anthropic"
    check = consistency._openai_score_one(api_key="offline", model=haiku.MODEL, request={}, rollout={},
        frames=[{"image_url": image_url()}], audit_root=tmp_path, provider_calls=calls)
    assert check["forward_consistent"] is None
    assert check["label_source"] == "anthropic_wam_episode_consistency_judge"
    check = success._openai_label_one(api_key="offline", model=haiku.MODEL, request={}, rollout={}, video_path=tmp_path / "fixture.mp4",
        frames=[{"image_url": image_url()}], audit_root=tmp_path, provider_calls=calls)
    assert check["label_source"] == "anthropic_generated_video_frame_judge"
    assert check["success"] is None
    assert len(calls) == 3


def test_only_luna_is_replaced():
    assert haiku.replacement_model("gpt-6-luna") == haiku.MODEL
    assert haiku.replacement_model("gpt-6-luna-2026-10-01") == haiku.MODEL
    for model in ("gpt-6-sol", "gpt-6-astra", "gpt-5.6-luna", "deepseek-v4-pro"):
        assert haiku.replacement_model(model) == model


def test_proxy_credentials_are_not_forwarded_to_native_anthropic(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "offline-key")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://api.deepseek.com/anthropic")
    assert haiku.anthropic_key() == ("", None)
