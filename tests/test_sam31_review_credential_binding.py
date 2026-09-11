"""The visual review must spend through its metered key, not a shared default."""
import pytest

from blueprint_pipeline.public_scene_sam31_ai_visual_reviewer import (
    _scoped_review_model_provider, Sam31AIVisualReviewError,
)


def test_explicit_review_credential_and_project_override_unrelated_defaults(tmp_path, monkeypatch):
    import openai
    from agents.models import openai_provider
    key = tmp_path / "review.key"
    key.write_text("fixture-scoped-key")
    key.chmod(0o600)
    monkeypatch.setenv("OPENAI_API_KEY", "fixture-unrelated-default")
    monkeypatch.setenv("OPENAI_PROJECT_ID", "unrelated-project")
    monkeypatch.setattr(openai, "AsyncOpenAI", lambda **kwargs: kwargs)
    monkeypatch.setattr(openai_provider, "OpenAIProvider", lambda **kwargs: kwargs)
    provider = _scoped_review_model_provider(key, "metered-project")
    assert provider["openai_client"] == {"api_key": "fixture-scoped-key", "project": "metered-project",
                                         "base_url": "https://api.openai.com/v1"}
    assert provider["use_responses"] is True and provider["use_responses_websocket"] is False
    key.chmod(0o644)
    with pytest.raises(Sam31AIVisualReviewError, match="key_file_invalid"):
        _scoped_review_model_provider(key, "metered-project")
