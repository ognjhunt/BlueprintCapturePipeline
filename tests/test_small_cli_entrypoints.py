from __future__ import annotations


from blueprint_pipeline import native_runtime_service


def test_native_runtime_service_main_uses_env_host_port(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    captured: dict[str, object] = {}

    def fake_uvicorn_run(app, *, host: str, port: int) -> None:  # type: ignore[no-untyped-def]
        captured["app"] = app
        captured["host"] = host
        captured["port"] = port

    monkeypatch.setenv("SITE_WORLD_RUNTIME_SERVICE_HOST", "0.0.0.0")
    monkeypatch.setenv("SITE_WORLD_RUNTIME_SERVICE_PORT", "9876")
    monkeypatch.setenv("BLUEPRINT_RUNTIME_AUTH_TOKEN", "runtime-secret")
    monkeypatch.setattr(native_runtime_service.uvicorn, "run", fake_uvicorn_run)

    assert native_runtime_service.main() == 0
    assert captured == {
        "app": native_runtime_service.app,
        "host": "0.0.0.0",
        "port": 9876,
    }
