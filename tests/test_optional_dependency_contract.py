"""Regression coverage for CPU-safe installs without optional LLM SDKs."""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import patch

from blueprint_pipeline.capture_enrichment_llm import (
    CaptureEnrichmentConfig,
    build_capture_enrichment_runner,
)
from blueprint_pipeline.reference_image_utils import cleanup_crop_with_vlm
from blueprint_pipeline.scene_semantics import _infer_with_gemini


def test_gpt_image_cleanup_missing_llm_extra_logs_actionable_message(
    tmp_path: Path, caplog
) -> None:
    image_path = tmp_path / "crop.png"
    output_path = tmp_path / "cleaned.png"
    image_path.write_bytes(b"image-data")

    with caplog.at_level(logging.WARNING):
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}, clear=False):
            with patch.dict("sys.modules", {"openai": None}):
                result = cleanup_crop_with_vlm(image_path, output_path, provider="gpt_image")

    assert result == image_path
    assert "uv sync --extra llm" in caplog.text
    assert "openai" in caplog.text


def test_gemini_scene_semantics_missing_llm_extra_logs_actionable_message(
    tmp_path: Path, caplog
) -> None:
    frame = tmp_path / "frame_00001.jpg"
    frame.write_bytes(b"frame")

    with caplog.at_level(logging.WARNING):
        with patch.dict("os.environ", {"GOOGLE_GENAI_API_KEY": "test-key"}, clear=False):
            with patch.dict("sys.modules", {"google": None}):
                result = _infer_with_gemini(frames=[frame], timeout_sec=5)

    assert result is None
    assert "uv sync --extra llm" in caplog.text
    assert "google-genai" in caplog.text


def test_openai_sdk_enrichment_missing_llm_extra_logs_actionable_message(
    tmp_path: Path, caplog
) -> None:
    runner = build_capture_enrichment_runner(
        repo_root=tmp_path,
        config=CaptureEnrichmentConfig(
            provider="openai",
            mode="sdk",
            model="gpt-test",
            timeout_seconds=5,
        ),
    )

    assert runner is not None

    with caplog.at_level(logging.WARNING):
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}, clear=False):
            with patch.dict("sys.modules", {"openai": None}):
                result = runner("prompt_bank_expander", {"task": "inspect"})

    assert result is None
    assert "uv sync --extra llm" in caplog.text
    assert "OpenAI SDK capture enrichment" in caplog.text
