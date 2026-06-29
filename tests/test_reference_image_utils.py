from __future__ import annotations

import base64
import builtins
import json
import sys
import types
from pathlib import Path

import pytest
pytest.importorskip("PIL")
from PIL import Image

from blueprint_pipeline import reference_image_utils as ref


class _Response:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args) -> None:  # type: ignore[no-untyped-def]
        return None

    def read(self) -> bytes:
        return self._payload


def _png(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (2, 2), color=(10, 20, 30)).save(path)
    return path


def test_reference_image_basic_resolution_env_and_url_helpers(
    tmp_path: Path,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    missing = tmp_path / "missing.png"
    empty = tmp_path / "empty.png"
    empty.write_bytes(b"")
    image = _png(tmp_path / "image.png")
    assert ref.load_reference_image_base64(str(missing)) is None
    assert ref.load_reference_image_base64(str(empty)) is None
    assert base64.b64decode(ref.load_reference_image_base64(str(image))).startswith(b"\x89PNG")

    crop = _png(tmp_path / "crop.png")
    alt = _png(tmp_path / "alt.png")
    storage = tmp_path / "storage"
    storage_ref = _png(storage / "asset-a" / "reference.jpg")
    assert ref.find_best_reference_image({"reference_crop": crop}) == str(crop)
    assert ref.find_best_reference_image({"reference_crop": missing, "all_crops": [missing, alt]}) == str(alt)
    assert ref.find_best_reference_image({"reference_images": [alt]}) == str(alt)
    assert ref.find_best_reference_image({"asset_dir": "asset-a"}, storage) == str(storage_ref)
    assert ref.find_best_reference_image({"asset_dir": "missing"}, storage) is None
    assert ref.find_best_reference_image({}) is None

    monkeypatch.delenv("REF_FLOAT", raising=False)
    monkeypatch.delenv("REF_INT", raising=False)
    monkeypatch.delenv("REF_BOOL", raising=False)
    assert ref._env_float("REF_FLOAT", 1.25) == 1.25
    monkeypatch.setenv("REF_FLOAT", "bad")
    assert ref._env_float("REF_FLOAT", 2.5) == 2.5
    monkeypatch.setenv("REF_FLOAT", "3.5")
    assert ref._env_float("REF_FLOAT", 0.0) == 3.5
    assert ref._env_int("REF_INT", 4) == 4
    monkeypatch.setenv("REF_INT", "bad")
    assert ref._env_int("REF_INT", 5) == 5
    monkeypatch.setenv("REF_INT", "6")
    assert ref._env_int("REF_INT", 0) == 6
    assert ref._is_truthy_env("REF_BOOL", default=True) is True
    monkeypatch.setenv("REF_BOOL", "yes")
    assert ref._is_truthy_env("REF_BOOL") is True
    monkeypatch.setenv("REF_BOOL", "no")
    assert ref._is_truthy_env("REF_BOOL", default=True) is False

    monkeypatch.setenv("TOGETHER_QWEN_IMAGE_EDIT_MODEL", "Qwen/Qwen-Image")
    candidates = ref._together_qwen_model_candidates()
    assert candidates[0] == "Qwen/Qwen-Image"
    assert candidates.count("Qwen/Qwen-Image") == 1
    assert ref._decode_image_b64(base64.b64encode(b"img").decode()) == b"img"
    data_url = "data:image/png;base64," + base64.b64encode(b"img2").decode()
    assert ref._decode_image_b64(data_url) == b"img2"
    assert ref._is_allowed_together_image_url("https://files.together.ai/out.png") is True
    assert ref._is_allowed_together_image_url("http://files.together.ai/out.png") is False
    assert ref._is_allowed_together_image_url("https:///out.png") is False
    assert ref._is_allowed_together_image_url("https://example.com/out.png") is False

    ref._QWEN_EDIT_DISABLE_LOGGED = False
    ref._disable_qwen_for_run("test-disable")
    assert ref._QWEN_EDIT_PIPELINE is None
    assert ref._QWEN_EDIT_DISABLED_REASON == "test-disable"


def test_cleanup_crop_dispatch_and_qwen_vram_guards(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    image = _png(tmp_path / "input.png")
    output = tmp_path / "output.png"
    assert ref.cleanup_crop_with_vlm(image, output, provider="skip") == image
    assert ref.cleanup_crop_with_vlm(tmp_path / "missing.png", output, provider="qwen_image_edit") is None
    assert ref.cleanup_crop_with_vlm(image, output, provider="unknown") == image

    monkeypatch.setattr(ref, "_cleanup_with_together_qwen_image_edit", lambda *_args: image)
    monkeypatch.setattr(ref, "_cleanup_with_qwen_image_edit", lambda *_args: image)
    monkeypatch.setattr(ref, "_cleanup_with_nano_banana", lambda *_args: image)
    monkeypatch.setattr(ref, "_cleanup_with_gpt_image", lambda *_args: output)
    assert ref.cleanup_crop_with_vlm(image, output, provider="auto") == output
    monkeypatch.setattr(ref, "_cleanup_with_together_qwen_image_edit", lambda *_args: output)
    assert ref.cleanup_crop_with_vlm(image, output, provider="auto") == output
    monkeypatch.setattr(ref, "_cleanup_with_together_qwen_image_edit", lambda *_args: image)
    monkeypatch.setattr(ref, "_cleanup_with_qwen_image_edit", lambda *_args: output)
    assert ref.cleanup_crop_with_vlm(image, output, provider="auto") == output
    monkeypatch.setattr(ref, "_cleanup_with_qwen_image_edit", lambda *_args: image)
    monkeypatch.setattr(ref, "_cleanup_with_nano_banana", lambda *_args: output)
    assert ref.cleanup_crop_with_vlm(image, output, provider="auto") == output
    monkeypatch.setattr(ref, "_cleanup_with_nano_banana", lambda *_args: image)
    assert ref.cleanup_crop_with_vlm(image, output, provider="together_qwen_image_edit_api") == image
    assert ref.cleanup_crop_with_vlm(image, output, provider="qwen_image_edit") == image
    assert ref.cleanup_crop_with_vlm(image, output, provider="nano_banana") == image
    assert ref.cleanup_crop_with_vlm(image, output, provider="gpt_image") == output

    class Props:
        total_memory = 24 * 1024**3

    class Cuda:
        def __init__(self, *, total_error: bool = False, total_gb: int = 24, free_gb: int = 8) -> None:
            self.total_error = total_error
            self.total_gb = total_gb
            self.free_gb = free_gb

        def get_device_properties(self, _device: int):  # type: ignore[no-untyped-def]
            if self.total_error:
                raise RuntimeError("cuda unavailable")
            return types.SimpleNamespace(total_memory=self.total_gb * 1024**3)

        def mem_get_info(self, _device: int):  # type: ignore[no-untyped-def]
            return (self.free_gb * 1024**3, self.total_gb * 1024**3)

    fake_torch = types.SimpleNamespace(cuda=Cuda())
    monkeypatch.setenv("QWEN_IMAGE_EDIT_FORCE", "true")
    assert ref._qwen_vram_check(fake_torch) == (True, "")
    monkeypatch.setenv("QWEN_IMAGE_EDIT_FORCE", "false")
    assert ref._qwen_vram_check(types.SimpleNamespace(cuda=Cuda(total_gb=12)))[0] is False
    assert ref._qwen_vram_check(types.SimpleNamespace(cuda=Cuda(free_gb=2)))[0] is False
    monkeypatch.setenv("QWEN_IMAGE_EDIT_CUDA_DEVICE", "not-int")
    assert ref._qwen_vram_check(fake_torch) == (True, "")
    ok, reason = ref._qwen_vram_check(types.SimpleNamespace(cuda=Cuda(total_error=True)))
    assert ok is False
    assert "unable to query CUDA memory" in reason


def _fake_torch_module(*, cuda_available: bool = True):
    class Cuda:
        emptied = False

        @staticmethod
        def is_available() -> bool:
            return cuda_available

        @staticmethod
        def get_device_properties(_device: int):  # type: ignore[no-untyped-def]
            return types.SimpleNamespace(total_memory=24 * 1024**3)

        @staticmethod
        def mem_get_info(_device: int):  # type: ignore[no-untyped-def]
            return (8 * 1024**3, 24 * 1024**3)

        @classmethod
        def empty_cache(cls) -> None:
            cls.emptied = True

    module = types.ModuleType("torch")
    module.cuda = Cuda
    module.bfloat16 = object()
    return module


def _fake_diffusers_module(*, pipeline):
    module = types.ModuleType("diffusers")

    class FakePipelineClass:
        @staticmethod
        def from_pretrained(*_args, **_kwargs):  # type: ignore[no-untyped-def]
            return pipeline

    module.QwenImageEditPlusPipeline = FakePipelineClass
    return module


class _SavedImage:
    def save(self, path: str) -> None:
        Path(path).write_bytes(b"qwen-image")


class _QwenPipeline:
    def __init__(self, *, error: Exception | None = None) -> None:
        self.error = error
        self.offloaded = False

    def enable_model_cpu_offload(self) -> None:
        self.offloaded = True

    def __call__(self, **_kwargs):  # type: ignore[no-untyped-def]
        if self.error:
            raise self.error
        return types.SimpleNamespace(images=[_SavedImage()])


def test_qwen_cleanup_success_and_failure_paths(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    image = _png(tmp_path / "input.png")
    output = tmp_path / "qwen.png"
    fake_torch = _fake_torch_module()
    pipeline = _QwenPipeline()
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "diffusers", _fake_diffusers_module(pipeline=pipeline))
    ref._QWEN_EDIT_PIPELINE = None
    ref._QWEN_EDIT_DISABLED_REASON = None
    ref._QWEN_EDIT_DISABLE_LOGGED = False
    assert ref._cleanup_with_qwen_image_edit(image, output) == output
    assert output.read_bytes() == b"qwen-image"
    assert pipeline.offloaded is True

    ref._QWEN_EDIT_DISABLED_REASON = "disabled"
    assert ref._cleanup_with_qwen_image_edit(image, tmp_path / "disabled.png") == image
    ref._QWEN_EDIT_DISABLED_REASON = None

    monkeypatch.setitem(sys.modules, "torch", _fake_torch_module(cuda_available=False))
    assert ref._cleanup_with_qwen_image_edit(image, tmp_path / "no-cuda.png") == image

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(ref, "_qwen_vram_check", lambda _torch: (False, "too small"))
    ref._QWEN_EDIT_PIPELINE = None
    assert ref._cleanup_with_qwen_image_edit(image, tmp_path / "vram.png") == image
    assert ref._QWEN_EDIT_DISABLED_REASON == "too small"
    ref._QWEN_EDIT_DISABLED_REASON = None

    checks = iter([(True, ""), (False, "post-load fail")])
    monkeypatch.setattr(ref, "_qwen_vram_check", lambda _torch: next(checks))
    ref._QWEN_EDIT_PIPELINE = None
    assert ref._cleanup_with_qwen_image_edit(image, tmp_path / "post.png") == image
    ref._QWEN_EDIT_DISABLED_REASON = None

    fake_torch_bad_cache = _fake_torch_module()
    fake_torch_bad_cache.cuda.empty_cache = staticmethod(
        lambda: (_ for _ in ()).throw(RuntimeError("cache failed"))
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch_bad_cache)
    checks = iter([(True, ""), (False, "post-load cache fail")])
    monkeypatch.setattr(ref, "_qwen_vram_check", lambda _torch: next(checks))
    ref._QWEN_EDIT_PIPELINE = None
    assert ref._cleanup_with_qwen_image_edit(image, tmp_path / "post-cache.png") == image
    ref._QWEN_EDIT_DISABLED_REASON = None

    monkeypatch.setattr(ref, "_qwen_vram_check", lambda _torch: (True, ""))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    ref._QWEN_EDIT_PIPELINE = _QwenPipeline(error=RuntimeError("CUDA out of memory"))
    assert ref._cleanup_with_qwen_image_edit(image, tmp_path / "oom.png") == image
    assert fake_torch.cuda.emptied is True
    ref._QWEN_EDIT_DISABLED_REASON = None
    monkeypatch.setitem(sys.modules, "torch", fake_torch_bad_cache)
    ref._QWEN_EDIT_PIPELINE = _QwenPipeline(error=RuntimeError("CUDA memory exhausted"))
    assert ref._cleanup_with_qwen_image_edit(image, tmp_path / "oom-cache.png") == image
    ref._QWEN_EDIT_DISABLED_REASON = None
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    ref._QWEN_EDIT_PIPELINE = _QwenPipeline(error=RuntimeError("other runtime"))
    assert ref._cleanup_with_qwen_image_edit(image, tmp_path / "runtime.png") == image
    ref._QWEN_EDIT_DISABLED_REASON = None
    ref._QWEN_EDIT_PIPELINE = _QwenPipeline(error=ValueError("generic"))
    assert ref._cleanup_with_qwen_image_edit(image, tmp_path / "generic.png") == image

    real_import = builtins.__import__

    def missing_torch(name, *args, **kwargs):  # type: ignore[no-untyped-def]
        if name == "torch":
            raise ImportError("missing torch")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing_torch)
    assert ref._cleanup_with_qwen_image_edit(image, tmp_path / "import.png") == image


def test_together_qwen_cleanup_and_extractors(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    image = _png(tmp_path / "input.png")
    output = tmp_path / "together.png"
    monkeypatch.delenv("TOGETHER_API_KEY", raising=False)
    assert ref._cleanup_with_together_qwen_image_edit(image, output) == image

    body = {"data": [{"b64_json": base64.b64encode(b"edited").decode()}]}

    def urlopen_success(request, timeout):  # type: ignore[no-untyped-def]
        assert request.full_url == "https://api.together.xyz/v1/images/generations"
        assert timeout == 2.0
        return _Response(json.dumps(body).encode("utf-8"))

    monkeypatch.setenv("TOGETHER_API_KEY", "key")
    monkeypatch.setenv("TOGETHER_QWEN_IMAGE_EDIT_TIMEOUT_SECONDS", "2")
    monkeypatch.setenv("TOGETHER_QWEN_IMAGE_EDIT_WIDTH", "64")
    monkeypatch.setenv("TOGETHER_QWEN_IMAGE_EDIT_HEIGHT", "65")
    monkeypatch.setenv("TOGETHER_QWEN_IMAGE_EDIT_STEPS", "1")
    monkeypatch.setenv("TOGETHER_QWEN_IMAGE_EDIT_OUTPUT_FORMAT", "webp")
    monkeypatch.setattr(ref.urllib_request, "urlopen", urlopen_success)
    assert ref._cleanup_with_together_qwen_image_edit(image, output) == output
    assert output.read_bytes() == b"edited"

    assert ref._extract_together_image_bytes(body, timeout_seconds=1) == b"edited"
    monkeypatch.setattr(ref.urllib_request, "urlopen", lambda _url, timeout: _Response(b"url-bytes"))
    assert (
        ref._extract_together_image_bytes(
            {"data": [{"url": "https://cdn.together.xyz/out.png"}]},
            timeout_seconds=1,
        )
        == b"url-bytes"
    )
    assert ref._extract_together_image_bytes({"data": [{"url": "https://example.com/out.png"}]}, timeout_seconds=1) is None
    assert ref._extract_together_image_bytes({"data": []}, timeout_seconds=1) is None

    monkeypatch.setattr(ref.urllib_request, "urlopen", lambda *_args, **_kwargs: _Response(b'{"data":[{}]}'))
    assert ref._cleanup_with_together_qwen_image_edit(image, tmp_path / "missing-payload.png") == image

    class FakeHTTPError(Exception):
        code = 429

        def read(self) -> bytes:
            return b"provider detail"

    def urlopen_http_error(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        raise FakeHTTPError("rate")

    monkeypatch.setattr(ref.urllib_error, "HTTPError", FakeHTTPError)
    monkeypatch.setattr(ref, "_together_qwen_model_candidates", lambda: ["model-http"])
    monkeypatch.setattr(ref.urllib_request, "urlopen", urlopen_http_error)
    assert ref._cleanup_with_together_qwen_image_edit(image, tmp_path / "http.png") == image

    class FakeHTTPErrorBadBody(Exception):
        code = 500

        def read(self) -> bytes:
            raise RuntimeError("read failed")

    monkeypatch.setattr(ref.urllib_error, "HTTPError", FakeHTTPErrorBadBody)
    def urlopen_http_error_bad_body(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        raise FakeHTTPErrorBadBody("bad")

    monkeypatch.setattr(ref.urllib_request, "urlopen", urlopen_http_error_bad_body)
    assert ref._cleanup_with_together_qwen_image_edit(image, tmp_path / "http-bad-body.png") == image
    monkeypatch.setattr(ref, "_together_qwen_model_candidates", lambda: ["model-generic"])
    monkeypatch.setattr(ref.urllib_request, "urlopen", lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("network")))
    assert ref._cleanup_with_together_qwen_image_edit(image, tmp_path / "generic.png") == image
    monkeypatch.setattr(ref, "_together_qwen_model_candidates", lambda: [])
    assert ref._cleanup_with_together_qwen_image_edit(image, tmp_path / "unknown.png") == image


def test_nano_banana_and_gpt_image_cleanup_paths(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    image = _png(tmp_path / "input.png")
    nano_output = tmp_path / "nano.png"
    gpt_output = tmp_path / "gpt.png"
    monkeypatch.delenv("GOOGLE_GENAI_API_KEY", raising=False)
    assert ref._cleanup_with_nano_banana(image, nano_output) == image

    real_import = builtins.__import__

    def missing_google(name, *args, **kwargs):  # type: ignore[no-untyped-def]
        if name == "google":
            raise ImportError("missing google")
        return real_import(name, *args, **kwargs)

    monkeypatch.setenv("GOOGLE_GENAI_API_KEY", "google-key")
    monkeypatch.setattr(builtins, "__import__", missing_google)
    assert ref._cleanup_with_nano_banana(image, nano_output) == image
    monkeypatch.setattr(builtins, "__import__", real_import)

    class Part:
        inline_data = types.SimpleNamespace(data=base64.b64encode(b"nano").decode())

    class NanoModels:
        def __init__(self, response) -> None:  # type: ignore[no-untyped-def]
            self.response = response

        def generate_content(self, **_kwargs):  # type: ignore[no-untyped-def]
            if isinstance(self.response, Exception):
                raise self.response
            return self.response

    class NanoClient:
        def __init__(self, api_key: str) -> None:
            assert api_key == "google-key"
            self.models = NanoModels(types.SimpleNamespace(candidates=[types.SimpleNamespace(content=types.SimpleNamespace(parts=[Part()]))]))

    google_module = types.ModuleType("google")
    google_module.genai = types.SimpleNamespace(Client=NanoClient)
    monkeypatch.setitem(sys.modules, "google", google_module)
    assert ref._cleanup_with_nano_banana(image, nano_output) == nano_output
    assert nano_output.read_bytes() == b"nano"

    class NoImageClient:
        def __init__(self, api_key: str) -> None:
            self.models = NanoModels(types.SimpleNamespace(candidates=[]))

    google_module.genai = types.SimpleNamespace(Client=NoImageClient)
    assert ref._cleanup_with_nano_banana(image, tmp_path / "nano-empty.png") == image

    class ErrorClient:
        def __init__(self, api_key: str) -> None:
            self.models = NanoModels(RuntimeError("nano failed"))

    google_module.genai = types.SimpleNamespace(Client=ErrorClient)
    assert ref._cleanup_with_nano_banana(image, tmp_path / "nano-error.png") == image

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert ref._cleanup_with_gpt_image(image, gpt_output) == image

    def missing_openai(name, *args, **kwargs):  # type: ignore[no-untyped-def]
        if name == "openai":
            raise ImportError("missing openai")
        return real_import(name, *args, **kwargs)

    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.setattr(builtins, "__import__", missing_openai)
    assert ref._cleanup_with_gpt_image(image, gpt_output) == image
    monkeypatch.setattr(builtins, "__import__", real_import)

    class Images:
        def __init__(self, response) -> None:  # type: ignore[no-untyped-def]
            self.response = response

        def edit(self, **_kwargs):  # type: ignore[no-untyped-def]
            if isinstance(self.response, Exception):
                raise self.response
            return self.response

    class OpenAIClient:
        def __init__(self, api_key: str) -> None:
            assert api_key == "openai-key"
            self.images = Images(types.SimpleNamespace(data=[types.SimpleNamespace(b64_json=base64.b64encode(b"gpt").decode())]))

    openai_module = types.ModuleType("openai")
    openai_module.OpenAI = OpenAIClient
    monkeypatch.setitem(sys.modules, "openai", openai_module)
    assert ref._cleanup_with_gpt_image(image, gpt_output) == gpt_output
    assert gpt_output.read_bytes() == b"gpt"

    class EmptyOpenAIClient:
        def __init__(self, api_key: str) -> None:
            self.images = Images(types.SimpleNamespace(data=[]))

    openai_module.OpenAI = EmptyOpenAIClient
    assert ref._cleanup_with_gpt_image(image, tmp_path / "gpt-empty.png") == image

    class ErrorOpenAIClient:
        def __init__(self, api_key: str) -> None:
            self.images = Images(RuntimeError("gpt failed"))

    openai_module.OpenAI = ErrorOpenAIClient
    assert ref._cleanup_with_gpt_image(image, tmp_path / "gpt-error.png") == image
