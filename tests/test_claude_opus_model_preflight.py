"""Metadata-only model admission; no provider or paid scene is contacted."""
import pytest

from blueprint_pipeline.claude_opus_authoring_invoker import ClaudeAuthoringBlocked
from blueprint_pipeline.claude_opus_model_preflight import preflight


def _model():
    return {"id": "claude-opus-5-5", "max_input_tokens": 1_000_000,
            "max_tokens": 128_000, "capabilities": {
                "image_input": {"supported": True},
                "structured_outputs": {"supported": True},
                "thinking": {"types": {"adaptive": {"supported": True}}},
                "effort": {"medium": {"supported": True}}}}


def test_preflight_requires_scoped_key_and_reports_metadata_only(monkeypatch, tmp_path):
    monkeypatch.delenv("ANTHROPIC_API_KEY_FILE", raising=False)
    with pytest.raises(ClaudeAuthoringBlocked, match="key_file_missing"):
        preflight(fetch=lambda _key: _model())
    key = tmp_path / "scoped-anthropic-key"
    key.write_text("test-only-placeholder")
    key.chmod(0o600)
    monkeypatch.setenv("ANTHROPIC_API_KEY_FILE", str(key))
    observed = []
    result = preflight(fetch=lambda value: observed.append(value) or _model())
    assert observed == ["test-only-placeholder"]
    assert result["status"] == "model_metadata_available_no_inference"
    assert result["maximum_one_call_reservation_usd"] == 4.664
    assert result["provider_calls_performed"] == 0
    assert result["billing_observed"] is False
    assert "key" not in str(result).lower()


@pytest.mark.parametrize("change", [
    {"id": "claude-opus-5"},
    {"max_input_tokens": 200_000},
    {"max_tokens": 8_192},
    {"capabilities": {}},
])
def test_preflight_rejects_missing_model_or_capability(monkeypatch, tmp_path, change):
    key = tmp_path / "scoped-anthropic-key"
    key.write_text("test-only-placeholder")
    key.chmod(0o600)
    monkeypatch.setenv("ANTHROPIC_API_KEY_FILE", str(key))
    with pytest.raises(ClaudeAuthoringBlocked, match="model_capability_unavailable"):
        preflight(fetch=lambda _key: {**_model(), **change})
