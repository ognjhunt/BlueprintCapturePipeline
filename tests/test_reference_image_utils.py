"""Tests for reference image utilities (crop loading, selection, VLM cleanup)."""

import base64
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import blueprint_pipeline.reference_image_utils as ref_utils
from blueprint_pipeline.reference_image_utils import (
    cleanup_crop_with_vlm,
    find_best_reference_image,
    load_reference_image_base64,
)


def _reset_qwen_state() -> None:
    ref_utils._QWEN_EDIT_PIPELINE = None
    ref_utils._QWEN_EDIT_DISABLED_REASON = None
    ref_utils._QWEN_EDIT_DISABLE_LOGGED = False


def test_load_reference_image_base64_happy_path(tmp_path: Path) -> None:
    img = tmp_path / "ref.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)

    result = load_reference_image_base64(str(img))
    assert result is not None
    assert isinstance(result, str)
    assert len(result) > 0


def test_load_reference_image_base64_missing_file() -> None:
    result = load_reference_image_base64("/nonexistent/path/ref.png")
    assert result is None


def test_load_reference_image_base64_empty_file(tmp_path: Path) -> None:
    img = tmp_path / "empty.png"
    img.write_bytes(b"")

    result = load_reference_image_base64(str(img))
    assert result is None


def test_find_best_reference_image_primary_crop(tmp_path: Path) -> None:
    crop = tmp_path / "crop.png"
    crop.write_bytes(b"image data")

    candidate = {"reference_crop": str(crop)}
    result = find_best_reference_image(candidate)
    assert result == str(crop)


def test_find_best_reference_image_all_crops_fallback(tmp_path: Path) -> None:
    crop1 = tmp_path / "crop1.png"
    crop1.write_bytes(b"image data")

    candidate = {"reference_crop": "/nonexistent.png", "all_crops": [str(crop1)]}
    result = find_best_reference_image(candidate)
    assert result == str(crop1)


def test_find_best_reference_image_asset_dir_fallback(tmp_path: Path) -> None:
    asset_dir = tmp_path / "obj_drawer_1"
    asset_dir.mkdir()
    ref = asset_dir / "reference.png"
    ref.write_bytes(b"image data")

    candidate = {"asset_dir": "obj_drawer_1"}
    result = find_best_reference_image(candidate, storage_root=tmp_path)
    assert result == str(ref)


def test_find_best_reference_image_none_when_nothing_exists() -> None:
    candidate = {"reference_crop": "/nonexistent.png"}
    result = find_best_reference_image(candidate)
    assert result is None


def test_cleanup_crop_skip_returns_original(tmp_path: Path) -> None:
    img = tmp_path / "crop.png"
    img.write_bytes(b"image data")
    out = tmp_path / "cleaned.png"

    result = cleanup_crop_with_vlm(img, out, provider="skip")
    assert result == img


def test_cleanup_crop_missing_file_returns_none(tmp_path: Path) -> None:
    img = tmp_path / "nonexistent.png"
    out = tmp_path / "cleaned.png"

    result = cleanup_crop_with_vlm(img, out, provider="nano_banana")
    assert result is None


def test_cleanup_crop_unknown_provider_returns_original(tmp_path: Path) -> None:
    img = tmp_path / "crop.png"
    img.write_bytes(b"image data")
    out = tmp_path / "cleaned.png"

    result = cleanup_crop_with_vlm(img, out, provider="unknown_provider")
    assert result == img


def test_cleanup_crop_nano_banana_no_api_key_returns_original(tmp_path: Path) -> None:
    img = tmp_path / "crop.png"
    img.write_bytes(b"image data")
    out = tmp_path / "cleaned.png"

    with patch.dict("os.environ", {"GOOGLE_GENAI_API_KEY": ""}, clear=False):
        result = cleanup_crop_with_vlm(img, out, provider="nano_banana")
    assert result == img


def test_cleanup_crop_gpt_image_no_api_key_returns_original(tmp_path: Path) -> None:
    img = tmp_path / "crop.png"
    img.write_bytes(b"image data")
    out = tmp_path / "cleaned.png"

    with patch.dict("os.environ", {"OPENAI_API_KEY": ""}, clear=False):
        result = cleanup_crop_with_vlm(img, out, provider="gpt_image")
    assert result == img


def test_cleanup_crop_together_qwen_no_api_key_returns_original(tmp_path: Path) -> None:
    img = tmp_path / "crop.png"
    img.write_bytes(b"image data")
    out = tmp_path / "cleaned.png"

    with patch.dict("os.environ", {"TOGETHER_API_KEY": ""}, clear=False):
        result = cleanup_crop_with_vlm(img, out, provider="together_qwen_image_edit")
    assert result == img


def test_cleanup_crop_together_qwen_success(tmp_path: Path) -> None:
    img = tmp_path / "crop.png"
    img.write_bytes(b"image data")
    out = tmp_path / "cleaned.png"
    cleaned_bytes = b"cleaned-image"

    mock_response = MagicMock()
    mock_response.read.return_value = json.dumps(
        {"data": [{"b64_json": base64.b64encode(cleaned_bytes).decode("utf-8")}]}
    ).encode("utf-8")

    mock_context = MagicMock()
    mock_context.__enter__.return_value = mock_response
    mock_context.__exit__.return_value = False

    with patch.dict("os.environ", {"TOGETHER_API_KEY": "tok_test"}, clear=False):
        with patch.object(ref_utils.urllib_request, "urlopen", return_value=mock_context):
            result = cleanup_crop_with_vlm(img, out, provider="together_qwen_image_edit")

    assert result == out
    assert out.read_bytes() == cleaned_bytes


def test_extract_together_image_bytes_rejects_non_https_url() -> None:
    response_json = {"data": [{"url": "file:///tmp/secret.txt"}]}

    with patch.object(ref_utils.urllib_request, "urlopen") as mocked_urlopen:
        result = ref_utils._extract_together_image_bytes(response_json, timeout_seconds=1.0)

    assert result is None
    mocked_urlopen.assert_not_called()


def test_extract_together_image_bytes_rejects_non_together_host() -> None:
    response_json = {"data": [{"url": "https://169.254.169.254/latest/meta-data"}]}

    with patch.object(ref_utils.urllib_request, "urlopen") as mocked_urlopen:
        result = ref_utils._extract_together_image_bytes(response_json, timeout_seconds=1.0)

    assert result is None
    mocked_urlopen.assert_not_called()


def test_extract_together_image_bytes_accepts_https_together_host() -> None:
    response_json = {"data": [{"url": "https://cdn.together.xyz/generated/image.png"}]}
    expected_bytes = b"image-bytes"

    mock_response = MagicMock()
    mock_response.read.return_value = expected_bytes
    mock_context = MagicMock()
    mock_context.__enter__.return_value = mock_response
    mock_context.__exit__.return_value = False

    with patch.object(ref_utils.urllib_request, "urlopen", return_value=mock_context) as mocked_urlopen:
        result = ref_utils._extract_together_image_bytes(response_json, timeout_seconds=2.5)

    assert result == expected_bytes
    mocked_urlopen.assert_called_once_with("https://cdn.together.xyz/generated/image.png", timeout=2.5)


def test_cleanup_crop_qwen_success(tmp_path: Path) -> None:
    """Qwen-Image-Edit produces a cleaned output image."""
    img = tmp_path / "crop.png"
    # Write a minimal valid PNG so PIL.Image.open works
    img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)
    out = tmp_path / "cleaned.png"

    mock_result_img = MagicMock()
    mock_pipeline_instance = MagicMock()
    mock_pipeline_instance.return_value.images = [mock_result_img]
    mock_pipeline_instance.to.return_value = mock_pipeline_instance

    mock_pipe_cls = MagicMock()
    mock_pipe_cls.from_pretrained.return_value = mock_pipeline_instance

    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = True
    mock_torch.cuda.get_device_properties.return_value = MagicMock(total_memory=32 * 1024 ** 3)
    mock_torch.cuda.mem_get_info.return_value = (24 * 1024 ** 3, 32 * 1024 ** 3)
    mock_torch.bfloat16 = "bfloat16"

    mock_diffusers = MagicMock()
    mock_diffusers.QwenImageEditPlusPipeline = mock_pipe_cls

    mock_pil = MagicMock()
    mock_pil_image = MagicMock()
    mock_pil.Image.open.return_value.convert.return_value = mock_pil_image

    _reset_qwen_state()
    try:
        with patch.dict("sys.modules", {
            "torch": mock_torch,
            "PIL": mock_pil,
            "PIL.Image": mock_pil.Image,
            "diffusers": mock_diffusers,
        }):
            result = cleanup_crop_with_vlm(img, out, provider="qwen_image_edit")

        assert result == out
        mock_result_img.save.assert_called_once_with(str(out))
    finally:
        _reset_qwen_state()


def test_cleanup_crop_qwen_no_cuda_returns_original(tmp_path: Path) -> None:
    """Qwen-Image-Edit returns original when CUDA is unavailable."""
    img = tmp_path / "crop.png"
    img.write_bytes(b"image data")
    out = tmp_path / "cleaned.png"

    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = False

    _reset_qwen_state()
    try:
        with patch.dict("sys.modules", {"torch": mock_torch}):
            result = cleanup_crop_with_vlm(img, out, provider="qwen_image_edit")
        assert result == img
    finally:
        _reset_qwen_state()


def test_cleanup_crop_qwen_import_error_returns_original(tmp_path: Path) -> None:
    """Qwen-Image-Edit returns original when torch is not installed."""
    img = tmp_path / "crop.png"
    img.write_bytes(b"image data")
    out = tmp_path / "cleaned.png"

    _reset_qwen_state()
    try:
        # Setting a sys.modules entry to None makes import raise ImportError
        with patch.dict("sys.modules", {"torch": None}):
            result = cleanup_crop_with_vlm(img, out, provider="qwen_image_edit")
        assert result == img
    finally:
        _reset_qwen_state()


def test_cleanup_crop_qwen_exception_returns_original(tmp_path: Path) -> None:
    """Qwen-Image-Edit returns original on pipeline exception."""
    img = tmp_path / "crop.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)
    out = tmp_path / "cleaned.png"

    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = True
    mock_torch.cuda.get_device_properties.return_value = MagicMock(total_memory=32 * 1024 ** 3)
    mock_torch.cuda.mem_get_info.return_value = (24 * 1024 ** 3, 32 * 1024 ** 3)
    mock_torch.bfloat16 = "bfloat16"

    mock_pipe_cls = MagicMock()
    mock_pipe_cls.from_pretrained.side_effect = RuntimeError("CUDA OOM")

    mock_diffusers = MagicMock()
    mock_diffusers.QwenImageEditPlusPipeline = mock_pipe_cls

    _reset_qwen_state()
    try:
        with patch.dict("sys.modules", {
            "torch": mock_torch,
            "diffusers": mock_diffusers,
        }):
            result = cleanup_crop_with_vlm(img, out, provider="qwen_image_edit")
        assert result == img
    finally:
        _reset_qwen_state()
