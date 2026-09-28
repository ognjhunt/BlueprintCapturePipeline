from types import SimpleNamespace
from urllib.error import HTTPError

import pytest

from blueprint_pipeline import safe_outbound_http, vast_provider_adapter as adapter
from blueprint_pipeline import vast_inventory_read_retry as retry
from blueprint_pipeline.gpu_render_providers import VastRenderProvider


def install(monkeypatch, responses):
    calls, sleeps = [], []
    monkeypatch.setattr(retry.time, "sleep", sleeps.append)
    def request(req, **kwargs):
        calls.append(req.get_method())
        response = responses[min(len(calls) - 1, len(responses) - 1)]
        if isinstance(response, Exception):
            raise response
        return SimpleNamespace(status=200, body=response)
    monkeypatch.setattr(safe_outbound_http, "open_request", request)
    return calls, sleeps


def error(status=429, headers=None):
    return HTTPError("https://console.vast.ai/api/v0/instances/", status, "fixture", headers or {}, None)


def test_inventory_429_then_zero_retries_only_read_and_retains_safe_evidence(monkeypatch, caplog):
    calls, sleeps = install(monkeypatch, [error(headers={"Retry-After": "2"}), b'{"instances":[]}'])
    monkeypatch.setattr(VastRenderProvider, "_key", lambda self: "secret-fixture")
    result = VastRenderProvider().billable_inventory(name_prefix="")
    assert result["api_confirmed"] is True and result["live_resource_count"] == 0
    assert calls == ["GET", "GET"] and sleeps == [2.0]
    assert '"http_status": 429' in caplog.text
    assert "secret-fixture" not in caplog.text


def test_persistent_rate_limit_is_bounded_and_never_becomes_zero(monkeypatch):
    calls, sleeps = install(monkeypatch, [error()])
    monkeypatch.setattr(VastRenderProvider, "_key", lambda self: "fixture")
    result = VastRenderProvider().billable_inventory(name_prefix="")
    assert calls == ["GET"] * 3 and len(sleeps) == 2
    assert result["api_confirmed"] is False and result["live_resource_count"] is None
    assert result["http"] == 429


@pytest.mark.parametrize("method,path,status,headers", [
    ("POST", "/instances/", 429, {}), ("PUT", "/asks/123/", 429, {}),
    ("DELETE", "/instances/123/", 429, {}), ("GET", "/bundles/", 429, {}),
    ("GET", "/instances/", 401, {}), ("GET", "/instances/", 403, {}),
    ("GET", "/instances/", 429, {"Retry-After": "60"}),
])
def test_mutations_other_paths_auth_failures_and_long_backoff_are_not_retried(monkeypatch, method, path, status, headers):
    calls, sleeps = install(monkeypatch, [error(status, headers)])
    with pytest.raises(HTTPError) as caught:
        adapter._api_json(method=method, path=path, api_key="fixture")
    assert caught.value.code == status
    assert calls == [method] and sleeps == []


def test_rate_limit_followed_by_live_inventory_preserves_nonzero(monkeypatch):
    calls, _ = install(monkeypatch, [error(), b'{"instances":[{"id":42,"label":"other","actual_status":"running"}]}'])
    monkeypatch.setattr(VastRenderProvider, "_key", lambda self: "fixture")
    result = VastRenderProvider().billable_inventory(name_prefix="")
    assert calls == ["GET", "GET"]
    assert result["api_confirmed"] is True and result["live_resource_count"] == 1
