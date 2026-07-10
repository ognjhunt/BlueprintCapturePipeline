from __future__ import annotations

from types import SimpleNamespace

from blueprint_pipeline import agent_review_cli, native_runtime_service


def test_agent_review_cli_builds_optional_openai_phase2_config(monkeypatch, capsys) -> None:  # type: ignore[no-untyped-def]
    assert agent_review_cli._openai_phase2_config_from_args(SimpleNamespace()) is None

    captured: dict[str, object] = {}

    def fake_run_agent_review(**kwargs):  # type: ignore[no-untyped-def]
        captured.update(kwargs)
        return {
            "provider": kwargs["provider_name"],
            "readiness_state": "review_required",
            "final_memo_path": "pipeline/agent_review/final.md",
            "final_bundle_path": "pipeline/agent_review/bundle.json",
        }

    monkeypatch.setattr(agent_review_cli, "run_agent_review", fake_run_agent_review)
    monkeypatch.setenv("OPENAI_PHASE2_MODE", "disabled")
    monkeypatch.setenv("OPENAI_PHASE2_CODEX_BIN", "codex-from-env")

    result = agent_review_cli.main(
        [
            "--capture-root",
            "/tmp/capture",
            "--provider",
            "openai",
            "--openai-phase2-model",
            "gpt-test",
            "--openai-phase2-timeout-seconds",
            "9",
            "--openai-phase2-reasoning-effort",
            "low",
        ]
    )

    assert result == 0
    config = captured["openai_phase2_config"]
    assert config.mode == "disabled"
    assert config.model == "gpt-test"
    assert config.codex_bin == "codex-from-env"
    assert config.timeout_seconds == 9
    assert config.reasoning_effort == "low"
    stdout = capsys.readouterr().out
    assert "[agent-review] provider=openai readiness=review_required" in stdout
    assert "final_bundle=pipeline/agent_review/bundle.json" in stdout


def test_agent_review_cli_reports_review_failures(monkeypatch, capsys) -> None:  # type: ignore[no-untyped-def]
    def fake_run_agent_review(**_kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("provider unavailable")

    monkeypatch.setattr(agent_review_cli, "run_agent_review", fake_run_agent_review)

    result = agent_review_cli.main(
        ["--capture-root", "/tmp/capture", "--provider", "claude", "--mode", "qualification"]
    )

    assert result == 1
    assert "[agent-review] FAILED: provider unavailable" in capsys.readouterr().out


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
