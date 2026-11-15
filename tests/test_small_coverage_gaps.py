from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest

from blueprint_pipeline import local_capture, model_access_env
from blueprint_pipeline.common import PipelineError
from blueprint_pipeline.synthesis import plucker_rays


def test_model_access_env_and_optional_dependency_messages(
    tmp_path: Path,
    monkeypatch,
    caplog,
) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("HOME", str(tmp_path))
    for key in (
        "HUGGINGFACE_HUB_TOKEN",
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "HUGGINGFACE_HUB_TOKEN_FILE",
        "HUGGINGFACE_TOKEN_FILE",
        "HUGGING_FACE_HUB_TOKEN_FILE",
        "HF_TOKEN_FILE",
        "NGC_API_KEY",
        "NVIDIA_NGC_API_KEY",
        "NGC_API_KEY_FILE",
        "NVIDIA_NGC_API_KEY_FILE",
    ):
        monkeypatch.delenv(key, raising=False)
    model_access_env.normalize_model_access_env()
    assert "HF_TOKEN" not in model_access_env.os.environ

    monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", " hf-value ")
    monkeypatch.setenv("NVIDIA_NGC_API_KEY", " ngc-value ")
    model_access_env.normalize_model_access_env()
    assert model_access_env.os.environ["HF_TOKEN"] == "hf-value"
    assert model_access_env.os.environ["HUGGINGFACE_HUB_TOKEN"] == "hf-value"
    assert model_access_env.os.environ["HUGGING_FACE_HUB_TOKEN"] == "hf-value"
    assert model_access_env.os.environ["NGC_API_KEY"] == "ngc-value"
    assert model_access_env.os.environ["NVIDIA_NGC_API_KEY"] == "ngc-value"

    for key in (
        "HUGGINGFACE_HUB_TOKEN",
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "NGC_API_KEY",
        "NVIDIA_NGC_API_KEY",
    ):
        monkeypatch.delenv(key, raising=False)
    hf_file = tmp_path / "hf-token"
    ngc_file = tmp_path / "ngc-token"
    hf_file.write_text("hf-file-value\n", encoding="utf-8")
    ngc_file.write_text("ngc-file-value\n", encoding="utf-8")
    monkeypatch.setenv("HF_TOKEN_FILE", str(hf_file))
    monkeypatch.setenv("NGC_API_KEY_FILE", str(ngc_file))
    model_access_env.normalize_model_access_env()
    assert model_access_env.os.environ["HF_TOKEN"] == "hf-file-value"
    assert model_access_env.os.environ["NGC_API_KEY"] == "ngc-file-value"
    secret_status = model_access_env.model_access_secret_status()
    assert secret_status["huggingface"]["auth_ready"] is True
    assert secret_status["ngc"]["auth_ready"] is True
    serialized_secret_status = str(secret_status)
    assert "hf-file-value" not in serialized_secret_status
    assert "ngc-file-value" not in serialized_secret_status

    original_read_text = Path.read_text

    def raising_read_text(self: Path, *args, **kwargs):  # type: ignore[no-untyped-def]
        if self == hf_file:
            raise OSError("permission denied")
        return original_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", raising_read_text)
    assert model_access_env._read_first_secret_file(
        file_env_names=("HF_TOKEN_FILE",),
        default_paths=(),
    ) == (None, None, None)

    original_stat = Path.stat

    def raising_stat(self: Path, *args, **kwargs):  # type: ignore[no-untyped-def]
        if self == hf_file:
            raise OSError("stat denied")
        return original_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", original_read_text)
    original_exists = Path.exists
    original_is_file = Path.is_file

    def fake_exists(self: Path) -> bool:
        if self == hf_file:
            return True
        return original_exists(self)

    def fake_is_file(self: Path) -> bool:
        if self == hf_file:
            return True
        return original_is_file(self)

    monkeypatch.setattr(Path, "exists", fake_exists)
    monkeypatch.setattr(Path, "is_file", fake_is_file)
    monkeypatch.setattr(Path, "stat", raising_stat)
    stat_status = model_access_env._file_status(
        path=hf_file,
        env_name="HF_TOKEN_FILE",
        configured_by_env=True,
    )
    assert stat_status["mode"] is None
    assert stat_status["size_bytes"] is None


def test_local_capture_error_branches_and_root_layout(tmp_path: Path) -> None:
    with pytest.raises(PipelineError, match="not inside a scenes"):
        local_capture.resolve_local_capture_context(tmp_path / "plain")
    with pytest.raises(PipelineError, match="does not match"):
        local_capture.resolve_local_capture_context(tmp_path / "bucket" / "scenes" / "scene-1" / "bad" / "cap-1")

    context = local_capture.resolve_local_capture_context(Path("/scenes/scene-1/captures/capture-1"))
    assert context.storage_root == Path("/")
    assert context.bucket == "/"
    assert context.capture_prefix == "scenes/scene-1/captures/capture-1"
    assert context.descriptor_uri == "gs:////scenes/scene-1/captures/capture-1/capture_descriptor.json"


def test_plucker_tensor_and_zero_moment_normalization(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    class FakeTensor:
        def __init__(self, array: np.ndarray) -> None:
            self.array = array
            self.unsqueeze_dim: int | None = None

        def unsqueeze(self, dim: int):
            self.unsqueeze_dim = dim
            return self

    fake_torch = types.SimpleNamespace(from_numpy=lambda array: FakeTensor(array))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    plucker = np.zeros((6, 2, 2), dtype=np.float32)
    tensor = plucker_rays.plucker_to_tensor(plucker)
    assert tensor.array is plucker
    assert tensor.unsqueeze_dim == 0
    normalized = plucker_rays.normalise_plucker(plucker)
    assert normalized.shape == (6, 2, 2)
    assert np.all(normalized == 0.0)


