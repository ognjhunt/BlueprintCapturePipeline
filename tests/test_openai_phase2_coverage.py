from __future__ import annotations

import builtins
import subprocess
import sys
import types
from pathlib import Path

from blueprint_pipeline.agent_runtime import openai_phase2


def test_openai_phase2_env_schema_prompt_and_text_helpers(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_PHASE2_MODE", raising=False)
    assert openai_phase2._string_env("MISSING_PHASE2_ENV", " fallback ") == " fallback "
    assert openai_phase2._int_env("MISSING_PHASE2_INT", 4) == 4
    monkeypatch.setenv("OPENAI_PHASE2_TIMEOUT_SECONDS", "bad")
    assert openai_phase2._int_env("OPENAI_PHASE2_TIMEOUT_SECONDS", 4) == 4
    assert openai_phase2._env_truthy("MISSING_PHASE2_BOOL", default=True) is True
    monkeypatch.setenv("OPENAI_PHASE2_BOOL", "on")
    assert openai_phase2._env_truthy("OPENAI_PHASE2_BOOL") is True
    monkeypatch.setenv("OPENAI_PHASE2_BOOL", "no")
    assert openai_phase2._env_truthy("OPENAI_PHASE2_BOOL") is False

    monkeypatch.setenv("OPENAI_PHASE2_MODE", " SDK ")
    monkeypatch.setenv("OPENAI_PHASE2_MODEL", "")
    monkeypatch.setenv("OPENAI_PHASE2_CODEX_BIN", "")
    monkeypatch.setenv("OPENAI_PHASE2_REASONING_EFFORT", "")
    config = openai_phase2.OpenAIPhase2Config.from_env()
    assert config.mode == "sdk"
    assert config.model == "gpt-5.4"
    assert config.codex_bin == "codex"
    assert config.reasoning_effort == "high"
    assert config.enabled() is True
    assert openai_phase2.OpenAIPhase2Config(mode=" ").normalized_mode() == "codex_cli"
    assert openai_phase2.OpenAIPhase2Config(mode="disabled").enabled() is False

    expected_schema_keys = {
        "intake_normalizer": "capture_modality",
        "evidence_auditor": "evidence_gaps",
        "blocker_taxonomist": "entries",
        "capability_envelope_writer": "bounded_claims",
        "standards_retriever": "source",
        "recapture_planner": "steps",
        "humanoid_site_readiness_reviewer": "summary",
        "humanoid_workcell_risk_reviewer": "summary",
        "humanoid_route_access_reviewer": "summary",
        "oem_handoff_writer": "summary",
        "readiness_report_writer": "memo_markdown",
        "unknown": "additionalProperties",
    }
    for skill_name, key in expected_schema_keys.items():
        schema = openai_phase2._skill_schema(skill_name)
        assert key in schema.get("properties", schema)

    assert "Return a conservative" in openai_phase2._skill_instruction("unknown")
    prompt = openai_phase2._prompt_for_skill("intake_normalizer", {"capture_id": "cap-1"})
    assert "Do not invent physical facts" in prompt
    assert '"capture_id": "cap-1"' in prompt

    assert openai_phase2._extract_openai_text(types.SimpleNamespace(output_text=" text ")) == "text"
    response = types.SimpleNamespace(
        output=[
            types.SimpleNamespace(
                content=[
                    types.SimpleNamespace(text='{"ok":'),
                    types.SimpleNamespace(text=" true}"),
                ]
            )
        ]
    )
    assert openai_phase2._extract_openai_text(response) == '{"ok": true}'
    assert openai_phase2._extract_openai_text(types.SimpleNamespace(output=[])) == ""


def test_openai_phase2_sdk_runner_edges(monkeypatch) -> None:
    runner = openai_phase2._OpenAISDKRunner(
        config=openai_phase2.OpenAIPhase2Config(mode="sdk", model="gpt-test", timeout_seconds=9, reasoning_effort="low")
    )
    assert runner.runtime_metadata() == {
        "openai_phase2_mode": "sdk",
        "openai_phase2_model": "gpt-test",
        "openai_phase2_timeout_seconds": 9,
        "openai_phase2_transport": "openai_sdk",
        "openai_phase2_reasoning_effort": "low",
    }
    assert runner.skill_metadata("skill") == {
        "skill_name": "skill",
        "transport": "openai_sdk",
        "mode": "sdk",
        "model": "gpt-test",
        "reasoning_effort": "low",
    }

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert runner("skill", {}) is None

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    real_import = builtins.__import__

    def import_missing(name, *args, **kwargs):
        if name == "openai":
            raise ImportError("missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_missing)
    assert runner("skill", {}) is None
    monkeypatch.setattr(builtins, "__import__", real_import)

    class FakeResponses:
        payload: object = '{"schema_version": "v1"}'

        def create(self, **kwargs):
            assert kwargs["model"] == "gpt-test"
            if isinstance(self.payload, Exception):
                raise self.payload
            return types.SimpleNamespace(output_text=self.payload)

    class FakeOpenAI:
        def __init__(self, api_key: str) -> None:
            assert api_key == "sk-test"
            self.responses = FakeResponses()

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=FakeOpenAI))
    assert runner("skill", {"a": 1}) == {"schema_version": "v1"}

    FakeResponses.payload = RuntimeError("down")
    assert runner("skill", {}) is None
    FakeResponses.payload = ""
    assert runner("skill", {}) is None
    FakeResponses.payload = "not-json"
    assert runner("skill", {}) is None
    FakeResponses.payload = "[]"
    assert runner("skill", {}) is None


def test_openai_phase2_codex_runner_edges(monkeypatch, tmp_path: Path) -> None:
    fallback_calls: list[tuple[str, dict[str, object]]] = []

    class Fallback:
        def __call__(self, skill_name, payload):
            fallback_calls.append((skill_name, dict(payload)))
            return {"fallback": skill_name}

    config = openai_phase2.OpenAIPhase2Config(
        mode="codex_cli",
        model="gpt-test",
        codex_bin="codex",
        timeout_seconds=0,
        reasoning_effort="medium",
    )
    runner = openai_phase2.CodexOpenAIPhase2Runner(config=config, repo_root=tmp_path, fallback_runner=Fallback())
    assert runner.runtime_metadata()["openai_phase2_transport"] == "codex_exec"
    assert runner.runtime_metadata()["openai_phase2_mode"] == "codex_cli"
    assert runner.skill_metadata("skill") == {
        "skill_name": "skill",
        "transport": "codex_exec",
        "mode": "codex_cli",
        "model": "gpt-test",
        "reasoning_effort": "medium",
    }

    disabled = openai_phase2.CodexOpenAIPhase2Runner(
        config=openai_phase2.OpenAIPhase2Config(mode="sdk"),
        repo_root=tmp_path,
    )
    assert disabled("skill", {}) is None

    monkeypatch.setattr(openai_phase2.shutil, "which", lambda _bin: None)
    assert runner("skill", {"x": 1}) == {"fallback": "skill"}
    assert fallback_calls[-1] == ("skill", {"x": 1})
    no_fallback = openai_phase2.CodexOpenAIPhase2Runner(config=config, repo_root=tmp_path)
    assert no_fallback("skill", {}) is None

    monkeypatch.setattr(openai_phase2.shutil, "which", lambda _bin: "/usr/bin/codex")

    def raise_os_error(*_args, **_kwargs):
        raise OSError("missing")

    monkeypatch.setattr(openai_phase2.subprocess, "run", raise_os_error)
    assert runner("skill", {}) == {"fallback": "skill"}
    assert no_fallback("skill", {}) is None

    def run_without_output(command, **_kwargs):
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(openai_phase2.subprocess, "run", run_without_output)
    assert runner("skill", {}) == {"fallback": "skill"}

    def failing_run(command, **_kwargs):
        return subprocess.CompletedProcess(command, 2)

    monkeypatch.setattr(openai_phase2.subprocess, "run", failing_run)
    assert no_fallback("skill", {}) is None

    def write_payload(payload: str):
        def _run(command, **kwargs):
            assert kwargs["input"].startswith("You are generating")
            assert kwargs["timeout"] == 1
            output_path = Path(command[command.index("--output-last-message") + 1])
            schema_path = Path(command[command.index("--output-schema") + 1])
            assert schema_path.is_file()
            output_path.write_text(payload, encoding="utf-8")
            return subprocess.CompletedProcess(command, 0)

        return _run

    monkeypatch.setattr(openai_phase2.subprocess, "run", write_payload("not-json"))
    assert runner("skill", {}) == {"fallback": "skill"}
    assert no_fallback("skill", {}) is None
    monkeypatch.setattr(openai_phase2.subprocess, "run", write_payload("[]"))
    assert no_fallback("skill", {}) is None
    monkeypatch.setattr(openai_phase2.subprocess, "run", write_payload('{"schema_version": "v1"}'))
    assert runner("intake_normalizer", {"capture_id": "cap-1"}) == {"schema_version": "v1"}


def test_build_openai_skill_runner_modes(monkeypatch, tmp_path: Path) -> None:
    assert openai_phase2.build_openai_skill_runner(
        repo_root=tmp_path,
        config=openai_phase2.OpenAIPhase2Config(mode="disabled"),
    ) is None
    assert isinstance(
        openai_phase2.build_openai_skill_runner(
            repo_root=tmp_path,
            config=openai_phase2.OpenAIPhase2Config(mode="sdk"),
        ),
        openai_phase2._OpenAISDKRunner,
    )

    monkeypatch.setenv("OPENAI_PHASE2_ALLOW_SDK_FALLBACK", "0")
    codex_runner = openai_phase2.build_openai_skill_runner(
        repo_root=tmp_path,
        config=openai_phase2.OpenAIPhase2Config(mode="codex_cli"),
    )
    assert isinstance(codex_runner, openai_phase2.CodexOpenAIPhase2Runner)
    assert codex_runner._fallback_runner is None

    monkeypatch.setenv("OPENAI_PHASE2_ALLOW_SDK_FALLBACK", "1")
    monkeypatch.setattr(openai_phase2.shutil, "which", lambda _bin: "/usr/bin/codex")
    auto_runner = openai_phase2.build_openai_skill_runner(
        repo_root=tmp_path,
        config=openai_phase2.OpenAIPhase2Config(mode="auto"),
    )
    assert isinstance(auto_runner, openai_phase2.CodexOpenAIPhase2Runner)
    assert auto_runner._fallback_runner is not None

    monkeypatch.setattr(openai_phase2.shutil, "which", lambda _bin: None)
    assert isinstance(
        openai_phase2.build_openai_skill_runner(
            repo_root=tmp_path,
            config=openai_phase2.OpenAIPhase2Config(mode="auto"),
        ),
        openai_phase2._OpenAISDKRunner,
    )
    assert openai_phase2.build_openai_skill_runner(
        repo_root=tmp_path,
        config=openai_phase2.OpenAIPhase2Config(mode="unknown"),
    ) is None
