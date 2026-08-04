from __future__ import annotations

import json
import subprocess

import pytest

from blueprint_pipeline import vast_safe_status


def test_safe_status_allowlist_excludes_provider_secrets(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    payload = [
        {
            "id": 123,
            "actual_status": "running",
            "gpu_name": "RTX A6000",
            "dph_total": 0.5,
            "label": "blueprint-test",
            "extra_env": [["OPENAI_API_KEY", "raw-secret"]],
            "jupyter_token": "another-secret",
            "onstart": "command-with-secret",
        }
    ]
    monkeypatch.setattr(
        vast_safe_status.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            [], 0, json.dumps(payload), ""
        ),
    )

    assert vast_safe_status.main([]) == 0
    output = capsys.readouterr().out
    assert "raw-secret" not in output
    assert "another-secret" not in output
    assert "extra_env" not in output
    assert json.loads(output)[0]["id"] == 123
