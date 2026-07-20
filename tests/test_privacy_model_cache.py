"""Warm-instance model cache + load timing (SCALE2-06).

The GPU warm-pool economics work depends on two properties of
``privacy_service_runtime``:

* model loads are timed and logged (``gpu_model_load``) so the modeled
  3-minute cold start can be replaced with measured numbers, and
* loaded runtimes are cached in-process so consecutive requests on a warm
  instance do not re-pay the load (without which ``min_instance_count > 0``
  would buy container boot only).
"""

from __future__ import annotations

import logging

import pytest

from blueprint_pipeline import privacy_service_runtime as psr


@pytest.fixture(autouse=True)
def _clean_cache(monkeypatch):
    monkeypatch.setattr(psr, "_MODEL_RUNTIME_CACHE", {})
    monkeypatch.delenv("PRIVACY_RUNNER_MODEL_CACHE", raising=False)


def test_timed_model_load_caches_and_logs(caplog):
    calls = []

    def loader():
        calls.append(1)
        return {"runtime": "fake"}

    with caplog.at_level(logging.INFO):
        first = psr._timed_model_load("sam3", "weights-a", loader)
        second = psr._timed_model_load("sam3", "weights-a", loader)

    assert first is second
    assert len(calls) == 1

    events = [r for r in caplog.records if getattr(r, "blueprint_event", "") == "gpu_model_load"]
    assert len(events) == 2
    cold, warm = events
    assert cold.blueprint_fields["cached"] is False
    assert cold.blueprint_fields["duration_seconds"] >= 0.0
    assert warm.blueprint_fields["cached"] is True
    assert warm.blueprint_fields["duration_seconds"] == 0.0


def test_timed_model_load_distinct_keys_load_separately():
    loads = []

    def loader_for(tag):
        def loader():
            loads.append(tag)
            return {"tag": tag}

        return loader

    a = psr._timed_model_load("sam3", "weights-a", loader_for("a"))
    b = psr._timed_model_load("sam3", "weights-b", loader_for("b"))
    assert a != b
    assert loads == ["a", "b"]


def test_timed_model_load_failure_is_not_cached():
    attempts = []

    def flaky():
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("load failed")
        return {"ok": True}

    with pytest.raises(RuntimeError):
        psr._timed_model_load("sam3", "weights-a", flaky)
    assert psr._MODEL_RUNTIME_CACHE == {}

    result = psr._timed_model_load("sam3", "weights-a", flaky)
    assert result == {"ok": True}
    assert len(attempts) == 2


def test_sam3_backend_keys_cache_by_stable_reference(monkeypatch, tmp_path):
    # execute_privacy_service_request materializes gs://https weights under a
    # fresh TemporaryDirectory per request; keying the cache by that temp path
    # would miss on every warm invocation and accumulate duplicate runtimes.
    # The backend must key by the ORIGINAL model reference instead.
    import inspect

    signature = inspect.signature(psr._run_sam3_backend)
    assert "weights_cache_key" in signature.parameters

    loads = []
    monkeypatch.setattr(
        psr,
        "_timed_model_load",
        lambda kind, key, loader: loads.append((kind, key)) or {"runtime": "fake"},
    )

    class _FailingCv2:
        @staticmethod
        def VideoCapture(_path):
            class _Cap:
                @staticmethod
                def isOpened():
                    return False

            return _Cap()

    import sys
    from types import ModuleType

    for name in ("numpy", "PIL", "PIL.Image"):
        if name not in sys.modules:
            monkeypatch.setitem(sys.modules, name, ModuleType(name))
    monkeypatch.setitem(sys.modules, "cv2", _FailingCv2)
    sam3_pkg = ModuleType("sam3")
    sam3_model_pkg = ModuleType("sam3.model")
    processor_mod = ModuleType("sam3.model.sam3_image_processor")
    builder_mod = ModuleType("sam3.model_builder")
    processor_mod.Sam3Processor = lambda model: {"processor": model}  # type: ignore[attr-defined]
    builder_mod.build_sam3_image_model = lambda **_kwargs: object()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sam3", sam3_pkg)
    monkeypatch.setitem(sys.modules, "sam3.model", sam3_model_pkg)
    monkeypatch.setitem(sys.modules, "sam3.model.sam3_image_processor", processor_mod)
    monkeypatch.setitem(sys.modules, "sam3.model_builder", builder_mod)

    video = tmp_path / "input.mov"
    video.write_bytes(b"video")
    for temp_weights in ("/tmp/req-a/sam3.pt", "/tmp/req-b/sam3.pt"):
        psr._run_sam3_backend(
            input_video=video,
            masks_dir=tmp_path / "masks",
            prompt="person",
            stage_name="",
            weights_path=temp_weights,
            weights_cache_key="gs://models/sam3.pt",
        )
    assert [key for _, key in loads] == ["gs://models/sam3.pt", "gs://models/sam3.pt"]


def test_cache_can_be_disabled_by_env(monkeypatch):
    monkeypatch.setenv("PRIVACY_RUNNER_MODEL_CACHE", "0")
    calls = []

    def loader():
        calls.append(1)
        return object()

    first = psr._timed_model_load("sam3", "weights-a", loader)
    second = psr._timed_model_load("sam3", "weights-a", loader)
    assert first is not second
    assert len(calls) == 2
    assert psr._MODEL_RUNTIME_CACHE == {}
