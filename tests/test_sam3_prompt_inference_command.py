"""Tests for PROMPT_INFERENCE_COMMAND rendering safety in scripts/sam3_detect.py."""

from __future__ import annotations

import importlib.util
import shlex
import sys
import types
from pathlib import Path


def _load_sam3_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "scripts" / "sam3_detect.py"

    if "torch" not in sys.modules:
        sys.modules["torch"] = types.ModuleType("torch")

    spec = importlib.util.spec_from_file_location("sam3_detect_prompt_test_module", str(module_path))
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_prompt_inference_quotes_csv_and_json_placeholders(tmp_path: Path, monkeypatch) -> None:
    module = _load_sam3_module()
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    malicious = frames_dir / "frame1.jpg;touch pwned;.jpg"
    malicious.write_bytes(b"fake")

    monkeypatch.setattr(module, "_PROMPT_INFERENCE_COMMAND", "printf '%s %s' {keyframes_csv} {keyframes_json}")

    observed = {}

    def _fake_run(command, **kwargs):  # type: ignore[no-untyped-def]
        observed["command"] = command

        class _Result:
            returncode = 0
            stdout = ""
            stderr = ""

        return _Result()

    monkeypatch.setattr(module.subprocess, "run", _fake_run)

    prompts, source = module._resolve_detection_prompts(
        environment="bedroom",
        frames_dir=frames_dir,
        all_frames=[malicious],
    )

    assert prompts
    assert source == "environment:bedroom"

    cmd = observed["command"]
    expected_csv = shlex.quote(str(malicious))
    expected_json = shlex.quote(f'["{str(malicious)}"]')
    assert expected_csv in cmd
    assert expected_json in cmd
