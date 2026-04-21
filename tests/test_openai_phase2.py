from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

from blueprint_pipeline.agent_runtime.openai_phase2 import (
    OpenAIPhase2Config,
    build_openai_skill_runner,
)


class _FakeResponses:
    def __init__(self, payload: str) -> None:
        self._payload = payload
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(output_text=self._payload)


class _FakeOpenAIClient:
    def __init__(self, *, api_key: str, payload: str) -> None:
        self.api_key = api_key
        self.responses = _FakeResponses(payload)


def test_openai_phase2_falls_back_to_sdk_when_codex_exec_fails(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_PHASE2_ALLOW_SDK_FALLBACK", "true")
    monkeypatch.setitem(
        sys.modules,
        "openai",
        SimpleNamespace(OpenAI=lambda api_key: _FakeOpenAIClient(api_key=api_key, payload=json.dumps({"memo_markdown": "sdk fallback"}))),
    )
    monkeypatch.setattr("blueprint_pipeline.agent_runtime.openai_phase2.shutil.which", lambda _name: "/usr/bin/codex")

    def _failed_run(*_args, **_kwargs):
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr("blueprint_pipeline.agent_runtime.openai_phase2.subprocess.run", _failed_run)

    runner = build_openai_skill_runner(
        repo_root=tmp_path,
        config=OpenAIPhase2Config(mode="codex_cli", model="gpt-5.4", codex_bin="codex", timeout_seconds=5),
    )

    assert runner is not None
    result = runner(
        "readiness_report_writer",
        {
            "schema_version": "v1",
            "scene_id": "scene-1",
            "capture_id": "capture-1",
        },
    )

    assert result == {"memo_markdown": "sdk fallback"}

