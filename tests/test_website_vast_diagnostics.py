from blueprint_pipeline import gpu_render_providers, vast_provider_adapter
from blueprint_pipeline.website_vast_diagnostics import worker_log_diagnostic


def test_live_worker_log_retains_only_typed_markers(monkeypatch):
    monkeypatch.setattr(gpu_render_providers, "_read_secret", lambda _name: "test-key")
    monkeypatch.setattr(vast_provider_adapter, "_api_json", lambda **_kwargs: (
        200, {"result_url": "https://objects.example/log?secret=hidden"}))
    monkeypatch.setattr(vast_provider_adapter, "_fetch_text", lambda *_args, **_kwargs: (
        "Traceback with https://objects.example/log?secret=hidden\n"
        "BLUEPRINT_WEBSITE_MAPANYTHING_BOOTSTRAP_FAILURE:model_download:network_or_timeout\n"
        "TimeoutError: https://objects.example/log?secret=hidden\n"))
    result = worker_log_diagnostic("42")
    assert result["typed_markers"] == [
        "BLUEPRINT_WEBSITE_MAPANYTHING_BOOTSTRAP_FAILURE:model_download:network_or_timeout"]
    assert result["exception_types"] == ["TimeoutError"]
    assert "hidden" not in str(result)
