"""Tests for the default SAM3 dimension completion command runner."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path


def _load_runner_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "scripts" / "sam3_dimension_completion_runner.py"
    spec = importlib.util.spec_from_file_location("sam3_dim_runner_test_module", str(module_path))
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_sam3_module_with_env():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "scripts" / "sam3_detect.py"
    if "torch" not in sys.modules:
        sys.modules["torch"] = types.ModuleType("torch")
    spec = importlib.util.spec_from_file_location("sam3_detect_dim_cmd_test_module", str(module_path))
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_runner_fallback_uses_image_priors() -> None:
    module = _load_runner_module()
    result = module.run_dimension_completion(
        label="box",
        environment="bedroom",
        observed_extents=[0.25, 0.2, 0.15],
        crop_paths=[],
        provider_mode="image_prior",
        model="unused",
        max_images=1,
    )
    assert result["model"] == "image_prior_v1"
    assert result["confidence"] >= 0.35
    assert result["predicted_extents_m"][0] >= 0.25
    assert result["predicted_extents_m"][1] >= 0.2
    assert result["predicted_extents_m"][2] >= 0.15


def test_runner_gemini_mode_degrades_gracefully() -> None:
    module = _load_runner_module()
    result = module.run_dimension_completion(
        label="unknown_object",
        environment="default",
        observed_extents=[0.4, 0.3, 0.2],
        crop_paths=[],
        provider_mode="gemini",
        model="gemini-2.5-flash",
        max_images=1,
    )
    assert result["model"] == "gemini-2.5-flash"
    assert result["reason"] == "gemini_unavailable"
    assert result["predicted_extents_m"] == [0.4, 0.3, 0.2]


def test_sam3_detect_defaults_to_runner_command(monkeypatch) -> None:
    monkeypatch.delenv("SAM3_DIMENSION_COMPLETION_COMMAND", raising=False)
    monkeypatch.delenv("SAM3_DIMENSION_COMPLETION_PYTHON", raising=False)

    module = _load_sam3_module_with_env()
    command = module._DIM_COMPLETION_COMMAND
    assert command
    assert "sam3_dimension_completion_runner.py" in command
