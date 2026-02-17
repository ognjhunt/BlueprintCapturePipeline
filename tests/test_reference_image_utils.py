"""Tests for reference image utilities (crop loading, selection, VLM cleanup)."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import blueprint_pipeline.reference_image_utils as ref_utils
from blueprint_pipeline.reference_image_utils import (
    cleanup_crop_with_vlm,
    find_best_reference_image,
    load_reference_image_base64,
)


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
    mock_torch.bfloat16 = "bfloat16"

    mock_diffusers = MagicMock()
    mock_diffusers.QwenImageEditPlusPipeline = mock_pipe_cls

    mock_pil = MagicMock()
    mock_pil_image = MagicMock()
    mock_pil.Image.open.return_value.convert.return_value = mock_pil_image

    ref_utils._QWEN_EDIT_PIPELINE = None
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
        ref_utils._QWEN_EDIT_PIPELINE = None


def test_cleanup_crop_qwen_no_cuda_returns_original(tmp_path: Path) -> None:
    """Qwen-Image-Edit returns original when CUDA is unavailable."""
    img = tmp_path / "crop.png"
    img.write_bytes(b"image data")
    out = tmp_path / "cleaned.png"

    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = False

    ref_utils._QWEN_EDIT_PIPELINE = None
    try:
        with patch.dict("sys.modules", {"torch": mock_torch}):
            result = cleanup_crop_with_vlm(img, out, provider="qwen_image_edit")
        assert result == img
    finally:
        ref_utils._QWEN_EDIT_PIPELINE = None


def test_cleanup_crop_qwen_import_error_returns_original(tmp_path: Path) -> None:
    """Qwen-Image-Edit returns original when torch is not installed."""
    img = tmp_path / "crop.png"
    img.write_bytes(b"image data")
    out = tmp_path / "cleaned.png"

    ref_utils._QWEN_EDIT_PIPELINE = None
    try:
        # Setting a sys.modules entry to None makes import raise ImportError
        with patch.dict("sys.modules", {"torch": None}):
            result = cleanup_crop_with_vlm(img, out, provider="qwen_image_edit")
        assert result == img
    finally:
        ref_utils._QWEN_EDIT_PIPELINE = None


def test_cleanup_crop_qwen_exception_returns_original(tmp_path: Path) -> None:
    """Qwen-Image-Edit returns original on pipeline exception."""
    img = tmp_path / "crop.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)
    out = tmp_path / "cleaned.png"

    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = True
    mock_torch.bfloat16 = "bfloat16"

    mock_pipe_cls = MagicMock()
    mock_pipe_cls.from_pretrained.side_effect = RuntimeError("CUDA OOM")

    mock_diffusers = MagicMock()
    mock_diffusers.QwenImageEditPlusPipeline = mock_pipe_cls

    ref_utils._QWEN_EDIT_PIPELINE = None
    try:
        with patch.dict("sys.modules", {
            "torch": mock_torch,
            "diffusers": mock_diffusers,
        }):
            result = cleanup_crop_with_vlm(img, out, provider="qwen_image_edit")
        assert result == img
    finally:
        ref_utils._QWEN_EDIT_PIPELINE = None
