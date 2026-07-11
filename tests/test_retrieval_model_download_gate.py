"""DINOv3 retrieval must be download-gated, revision-pinned, and remote-code-free."""

from __future__ import annotations

import sys
from types import ModuleType

import pytest

from blueprint_pipeline import retrieval_index_stage as ris


def _install_fake_transformers(monkeypatch: pytest.MonkeyPatch, seen: list[dict]) -> None:
    class _FakeModel:
        def eval(self):
            return None

    class _FakeAuto:
        @staticmethod
        def from_pretrained(model_id: str, **kwargs):
            seen.append(
                {
                    "model_id": model_id,
                    "kwargs": dict(kwargs),
                }
            )
            return _FakeModel()

    class _FakeCuda:
        @staticmethod
        def is_available():
            return False

    fake_torch = ModuleType("torch")
    fake_torch.cuda = _FakeCuda()
    fake_transformers = ModuleType("transformers")
    fake_transformers.AutoImageProcessor = _FakeAuto
    fake_transformers.AutoModel = _FakeAuto
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)


def test_ungated_load_is_pinned_remote_code_free_and_local_cache_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(ris._HF_MODEL_DOWNLOAD_GATE_ENV, raising=False)
    seen: list[dict] = []
    _install_fake_transformers(monkeypatch, seen)

    ris._load_dinov3()

    assert len(seen) == 2
    for call in seen:
        assert call["model_id"] == ris._DINOV3_MODEL_ID
        # Exact immutable commit pin, never a floating tag; remote code disabled.
        assert call["kwargs"]["revision"] == ris._DINOV3_MODEL_REVISION
        assert len(ris._DINOV3_MODEL_REVISION) == 40
        assert all(c in "0123456789abcdef" for c in ris._DINOV3_MODEL_REVISION)
        assert call["kwargs"]["trust_remote_code"] is False
        assert call["kwargs"]["local_files_only"] is True


def test_explicit_download_gate_permits_network_fetch_of_pinned_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(ris._HF_MODEL_DOWNLOAD_GATE_ENV, "1")
    seen: list[dict] = []
    _install_fake_transformers(monkeypatch, seen)

    ris._load_dinov3()

    assert len(seen) == 2
    for call in seen:
        assert call["kwargs"]["revision"] == ris._DINOV3_MODEL_REVISION
        assert call["kwargs"]["trust_remote_code"] is False
        assert call["kwargs"]["local_files_only"] is False


def test_ungated_failure_remains_local_files_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(ris._HF_MODEL_DOWNLOAD_GATE_ENV, raising=False)
    seen: list[dict] = []

    class _ExplodingAuto:
        @staticmethod
        def from_pretrained(*_args, **kwargs):
            seen.append(dict(kwargs))
            raise RuntimeError("model_not_in_local_cache")

    fake_transformers = ModuleType("transformers")
    fake_transformers.AutoImageProcessor = _ExplodingAuto
    fake_transformers.AutoModel = _ExplodingAuto
    fake_torch = ModuleType("torch")
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    with pytest.raises(ris.PipelineError, match="Failed to load DINOv3"):
        ris._load_dinov3()
    assert seen and seen[0]["local_files_only"] is True
