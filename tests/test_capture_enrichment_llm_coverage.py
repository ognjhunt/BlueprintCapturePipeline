from __future__ import annotations

import builtins
import json
import subprocess
import types
from pathlib import Path
from urllib import error as urllib_error


from blueprint_pipeline import capture_enrichment_llm as enrichment


class _Response:
    def __init__(self, payload: object) -> None:
        self._body = payload if isinstance(payload, bytes) else str(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def read(self) -> bytes:
        return self._body


def test_capture_enrichment_env_schema_prompt_and_text_helpers(monkeypatch) -> None:
    monkeypatch.delenv("CAPTURE_ENRICHMENT_LLM_PROVIDER", raising=False)
    assert enrichment._string_env("MISSING_CAPTURE_ENV", "fallback") == "fallback"
    assert enrichment._int_env("MISSING_CAPTURE_INT", 12) == 12

    monkeypatch.setenv("CAPTURE_ENRICHMENT_LLM_PROVIDER", " OPENAI ")
    monkeypatch.setenv("CAPTURE_ENRICHMENT_LLM_MODE", " AUTO ")
    monkeypatch.setenv("CAPTURE_ENRICHMENT_LLM_TIMEOUT_SECONDS", "bad")
    config = enrichment.CaptureEnrichmentConfig.from_env()
    assert config.provider == "openai"
    assert config.model == "gpt-5.1-mini"
    assert config.timeout_seconds == 120
    assert config.enabled() is True
    assert enrichment.CaptureEnrichmentConfig(provider="disabled").enabled() is False

    monkeypatch.setenv("CAPTURE_ENRICHMENT_LLM_PROVIDER", "claude")
    monkeypatch.delenv("CAPTURE_ENRICHMENT_LLM_MODEL", raising=False)
    assert enrichment.CaptureEnrichmentConfig.from_env().model == "claude-3-7-sonnet-latest"

    for skill_name, required_key in [
        ("prompt_bank_expander", "additional_prompts"),
        ("task_relevance_ranker", "scores"),
        ("workflow_target_resolver", "manipulation_candidates"),
        ("articulation_prior_writer", "articulation_priors"),
        ("qualification_weakness_summarizer", "summary"),
        ("recapture_instruction_writer", "instructions"),
        ("unknown", "additionalProperties"),
    ]:
        schema = enrichment._skill_schema(skill_name)
        assert required_key in schema.get("properties", schema)

    assert "Return conservative structured JSON" in enrichment._skill_instruction("unknown")
    prompt = enrichment._prompt_for_skill("prompt_bank_expander", {"task": "open cabinet"})
    assert "Do not invent measurements" in prompt
    assert '"task": "open cabinet"' in prompt

    assert enrichment._extract_openai_text(types.SimpleNamespace(output_text=" {\"ok\": true} ")) == '{"ok": true}'
    response = types.SimpleNamespace(
        output=[
            types.SimpleNamespace(
                content=[
                    types.SimpleNamespace(text='{"a":'),
                    types.SimpleNamespace(text=" 1}"),
                ]
            )
        ]
    )
    assert enrichment._extract_openai_text(response) == '{"a": 1}'
    assert enrichment._extract_openai_text(types.SimpleNamespace(output=[])) == ""


def test_codex_runner_handles_cli_absence_failures_and_success(monkeypatch, tmp_path: Path) -> None:
    runner = enrichment._CodexRunner(
        config=enrichment.CaptureEnrichmentConfig(
            provider="openai",
            mode="codex_cli",
            model="model-a",
            reasoning_effort="high",
            timeout_seconds=5,
        ),
        repo_root=tmp_path,
    )

    monkeypatch.setattr(enrichment.shutil, "which", lambda _bin: None)
    assert runner("prompt_bank_expander", {"task": "x"}) is None

    monkeypatch.setattr(enrichment.shutil, "which", lambda _bin: "/usr/bin/codex")
    monkeypatch.setattr(enrichment.subprocess, "run", lambda *_args, **_kwargs: (_ for _ in ()).throw(TimeoutError()))
    assert runner("prompt_bank_expander", {"task": "x"}) is None

    def no_output_run(*_args, **_kwargs):
        return subprocess.CompletedProcess(["codex"], 0)

    monkeypatch.setattr(enrichment.subprocess, "run", no_output_run)
    assert runner("prompt_bank_expander", {"task": "x"}) is None

    def write_output_run(command, **_kwargs):
        output_path = Path(command[command.index("--output-last-message") + 1])
        output_path.write_text(write_output_run.payload, encoding="utf-8")
        return subprocess.CompletedProcess(command, 0)

    write_output_run.payload = "not-json"
    monkeypatch.setattr(enrichment.subprocess, "run", write_output_run)
    assert runner("prompt_bank_expander", {"task": "x"}) is None

    write_output_run.payload = "[]"
    assert runner("prompt_bank_expander", {"task": "x"}) is None

    write_output_run.payload = json.dumps({"additional_prompts": ["cabinet"], "resolved_task_nouns": [], "notes": "ok"})
    assert runner("prompt_bank_expander", {"task": "x"}) == {
        "additional_prompts": ["cabinet"],
        "resolved_task_nouns": [],
        "notes": "ok",
    }


def test_openai_sdk_runner_handles_import_and_response_edges(monkeypatch) -> None:
    runner = enrichment._OpenAISDKRunner(config=enrichment.CaptureEnrichmentConfig(provider="openai", mode="sdk", model="m"))

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert runner("task_relevance_ranker", {"objects": []}) is None

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    real_import = builtins.__import__

    def import_missing(name, *args, **kwargs):
        if name == "openai":
            raise ImportError("missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_missing)
    assert runner("task_relevance_ranker", {"objects": []}) is None

    def import_broken(name, *args, **kwargs):
        if name == "openai":
            raise RuntimeError("broken import")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_broken)
    assert runner("task_relevance_ranker", {"objects": []}) is None
    monkeypatch.setattr(builtins, "__import__", real_import)

    class FakeResponses:
        payload = '{"scores": []}'

        def create(self, **_kwargs):
            if isinstance(self.payload, Exception):
                raise self.payload
            return types.SimpleNamespace(output_text=self.payload)

    class FakeOpenAI:
        def __init__(self, api_key: str) -> None:
            assert api_key == "sk-test"
            self.responses = FakeResponses()

    monkeypatch.setitem(
        __import__("sys").modules,
        "openai",
        types.SimpleNamespace(OpenAI=FakeOpenAI),
    )
    assert runner("task_relevance_ranker", {"objects": []}) == {"scores": []}

    FakeResponses.payload = RuntimeError("provider down")
    assert runner("task_relevance_ranker", {"objects": []}) is None
    FakeResponses.payload = ""
    assert runner("task_relevance_ranker", {"objects": []}) is None
    FakeResponses.payload = "not-json"
    assert runner("task_relevance_ranker", {"objects": []}) is None
    FakeResponses.payload = "[]"
    assert runner("task_relevance_ranker", {"objects": []}) is None


def test_claude_http_runner_handles_transport_parse_and_success(monkeypatch) -> None:
    runner = enrichment._ClaudeHTTPRunner(config=enrichment.CaptureEnrichmentConfig(provider="claude", model="claude"))

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert runner("workflow_target_resolver", {"workflow": "open door"}) is None

    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-key")

    def urlopen_returning(body: object):
        def _urlopen(request, timeout):
            assert request.full_url == "https://api.anthropic.com/v1/messages"
            assert request.headers["X-api-key"] == "anthropic-key"
            assert timeout == 120
            return _Response(body)

        return _urlopen

    monkeypatch.setattr(enrichment.urllib_request, "urlopen", lambda *_args, **_kwargs: (_ for _ in ()).throw(urllib_error.URLError("down")))
    assert runner("workflow_target_resolver", {"workflow": "open door"}) is None

    for body in [
        "not-json",
        json.dumps({"content": {}}),
        json.dumps({"content": [{"type": "image", "text": "ignored"}]}),
        json.dumps({"content": [{"type": "text", "text": "not-json"}]}),
        json.dumps({"content": [{"type": "text", "text": "[]"}]}),
    ]:
        monkeypatch.setattr(enrichment.urllib_request, "urlopen", urlopen_returning(body))
        assert runner("workflow_target_resolver", {"workflow": "open door"}) is None

    body = json.dumps({"content": [{"type": "text", "text": '{"tasks": [], "open_questions": []}'}]})
    monkeypatch.setattr(enrichment.urllib_request, "urlopen", urlopen_returning(body))
    assert runner("workflow_target_resolver", {"workflow": "open door"}) == {"tasks": [], "open_questions": []}


def test_build_capture_enrichment_runner_selects_provider_modes(monkeypatch, tmp_path: Path) -> None:
    assert enrichment.build_capture_enrichment_runner(repo_root=tmp_path, config=enrichment.CaptureEnrichmentConfig()) is None
    assert isinstance(
        enrichment.build_capture_enrichment_runner(
            repo_root=tmp_path,
            config=enrichment.CaptureEnrichmentConfig(provider="openai", mode="codex_cli"),
        ),
        enrichment._CodexRunner,
    )
    assert isinstance(
        enrichment.build_capture_enrichment_runner(
            repo_root=tmp_path,
            config=enrichment.CaptureEnrichmentConfig(provider="openai", mode="sdk"),
        ),
        enrichment._OpenAISDKRunner,
    )
    monkeypatch.setattr(enrichment.shutil, "which", lambda _bin: "/usr/bin/codex")
    assert isinstance(
        enrichment.build_capture_enrichment_runner(
            repo_root=tmp_path,
            config=enrichment.CaptureEnrichmentConfig(provider="openai", mode="auto"),
        ),
        enrichment._CodexRunner,
    )
    monkeypatch.setattr(enrichment.shutil, "which", lambda _bin: None)
    assert isinstance(
        enrichment.build_capture_enrichment_runner(
            repo_root=tmp_path,
            config=enrichment.CaptureEnrichmentConfig(provider="openai", mode="auto"),
        ),
        enrichment._OpenAISDKRunner,
    )
    assert isinstance(
        enrichment.build_capture_enrichment_runner(
            repo_root=tmp_path,
            config=enrichment.CaptureEnrichmentConfig(provider="claude", mode="http"),
        ),
        enrichment._ClaudeHTTPRunner,
    )

    class UnknownEnabled:
        provider = "other"
        mode = "auto"

        def enabled(self) -> bool:
            return True

    assert enrichment.build_capture_enrichment_runner(repo_root=tmp_path, config=UnknownEnabled()) is None
