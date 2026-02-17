"""Utilities for preparing object reference images for 3D generation APIs."""

from __future__ import annotations

import base64
import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def load_reference_image_base64(image_path: str) -> Optional[str]:
    """Load a reference image and return it as base64-encoded string.

    Returns None if the file does not exist or is empty.
    """
    path = Path(image_path)
    if not path.is_file() or path.stat().st_size == 0:
        return None
    data = path.read_bytes()
    return base64.b64encode(data).decode("utf-8")


def find_best_reference_image(
    candidate: dict,
    storage_root: Optional[Path] = None,
) -> Optional[str]:
    """Resolve the best available reference image path for a candidate.

    Checks in order:
    1. candidate["reference_crop"] - best SAM3 crop
    2. candidate["all_crops"][0] - first alternate crop
    3. asset_dir/reference.png in storage_root - manually placed reference

    Returns absolute path string or None.
    """
    crop = candidate.get("reference_crop")
    if crop and Path(str(crop)).is_file():
        return str(crop)

    all_crops = candidate.get("all_crops") or candidate.get("reference_images")
    if isinstance(all_crops, list):
        for c in all_crops:
            if c and Path(str(c)).is_file():
                return str(c)

    if storage_root:
        asset_dir = candidate.get("asset_dir", "")
        if asset_dir:
            for ext in ("png", "jpg", "jpeg"):
                ref = storage_root / asset_dir / f"reference.{ext}"
                if ref.is_file():
                    return str(ref)

    return None


def cleanup_crop_with_vlm(
    image_path: Path,
    output_path: Path,
    *,
    provider: str = "skip",
) -> Optional[Path]:
    """Optionally clean up a reference crop using a VLM image editing model.

    Providers:
      - "skip": Return original path unchanged (default, no API call)
      - "qwen_image_edit": Use Qwen-Image-Edit-2511 (local GPU, free, open-source)
      - "nano_banana": Use Google Gemini 3 Pro Image (Nano Banana Pro)
      - "gpt_image": Use OpenAI GPT Image 1.5
      - "auto": Try qwen_image_edit first, then nano_banana, then gpt_image

    Returns the cleaned image path, or the original path if cleanup fails.
    """
    provider = (provider or "skip").strip().lower()

    if provider == "skip":
        return image_path

    if not image_path.is_file():
        return None

    if provider == "auto":
        result = cleanup_crop_with_vlm(image_path, output_path, provider="qwen_image_edit")
        if result is not None and result != image_path:
            return result
        result = cleanup_crop_with_vlm(image_path, output_path, provider="nano_banana")
        if result is not None and result != image_path:
            return result
        return cleanup_crop_with_vlm(image_path, output_path, provider="gpt_image")

    if provider == "qwen_image_edit":
        return _cleanup_with_qwen_image_edit(image_path, output_path)

    if provider == "nano_banana":
        return _cleanup_with_nano_banana(image_path, output_path)

    if provider == "gpt_image":
        return _cleanup_with_gpt_image(image_path, output_path)

    logger.warning("Unknown crop cleanup provider: %s, skipping", provider)
    return image_path


_CLEANUP_PROMPT = (
    "Remove the background from this object image. Keep only the object "
    "on a clean transparent background. Preserve the object's exact shape, "
    "color, texture, and details. Output a PNG with transparency."
)


_QWEN_EDIT_PIPELINE = None
_QWEN_EDIT_DISABLED_REASON: Optional[str] = None
_QWEN_EDIT_DISABLE_LOGGED = False


def _env_float(name: str, default: float) -> float:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _is_truthy_env(name: str, default: bool = False) -> bool:
    raw = (os.getenv(name) or "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def _disable_qwen_for_run(reason: str) -> None:
    global _QWEN_EDIT_PIPELINE  # noqa: PLW0603
    global _QWEN_EDIT_DISABLED_REASON  # noqa: PLW0603
    global _QWEN_EDIT_DISABLE_LOGGED  # noqa: PLW0603
    _QWEN_EDIT_PIPELINE = None
    _QWEN_EDIT_DISABLED_REASON = reason
    if not _QWEN_EDIT_DISABLE_LOGGED:
        logger.warning("Disabling Qwen-Image-Edit for this run: %s", reason)
        _QWEN_EDIT_DISABLE_LOGGED = True


def _qwen_vram_check(torch_module) -> tuple[bool, str]:
    if _is_truthy_env("QWEN_IMAGE_EDIT_FORCE", default=False):
        return True, ""

    min_total_gb = _env_float("QWEN_IMAGE_EDIT_MIN_TOTAL_VRAM_GB", 20.0)
    min_free_gb = _env_float("QWEN_IMAGE_EDIT_MIN_FREE_VRAM_GB", 6.0)
    cuda_device_raw = (os.getenv("QWEN_IMAGE_EDIT_CUDA_DEVICE") or "0").strip()
    try:
        cuda_device = int(cuda_device_raw)
    except ValueError:
        cuda_device = 0

    try:
        props = torch_module.cuda.get_device_properties(cuda_device)
        total_gb = float(props.total_memory) / float(1024 ** 3)
        if total_gb < min_total_gb:
            return (
                False,
                f"device VRAM {total_gb:.1f}GB below required {min_total_gb:.1f}GB",
            )
        free_bytes, _ = torch_module.cuda.mem_get_info(cuda_device)
        free_gb = float(free_bytes) / float(1024 ** 3)
        if free_gb < min_free_gb:
            return (
                False,
                f"free VRAM {free_gb:.1f}GB below required {min_free_gb:.1f}GB",
            )
        return True, ""
    except Exception as exc:
        return False, f"unable to query CUDA memory ({exc})"


def _cleanup_with_qwen_image_edit(image_path: Path, output_path: Path) -> Optional[Path]:
    """Clean up crop using Qwen-Image-Edit-2511 (local GPU, free, open-source)."""
    global _QWEN_EDIT_PIPELINE  # noqa: PLW0603
    global _QWEN_EDIT_DISABLED_REASON  # noqa: PLW0603

    try:
        import torch
        from PIL import Image as PILImage

        if _QWEN_EDIT_DISABLED_REASON:
            return image_path

        if not torch.cuda.is_available():
            logger.warning("CUDA not available, skipping Qwen-Image-Edit cleanup")
            return image_path

        can_run, reason = _qwen_vram_check(torch)
        if not can_run:
            _disable_qwen_for_run(reason)
            return image_path

        if _QWEN_EDIT_PIPELINE is None:
            from diffusers import QwenImageEditPlusPipeline  # type: ignore

            model_path = (
                os.getenv("QWEN_IMAGE_EDIT_MODEL_PATH") or "Qwen/Qwen-Image-Edit-2511"
            ).strip()
            logger.info("Loading Qwen-Image-Edit from %s ...", model_path)
            _QWEN_EDIT_PIPELINE = QwenImageEditPlusPipeline.from_pretrained(
                model_path, torch_dtype=torch.bfloat16
            )
            _QWEN_EDIT_PIPELINE.enable_model_cpu_offload()
            logger.info("Qwen-Image-Edit loaded with CPU offload")

            can_run, reason = _qwen_vram_check(torch)
            if not can_run:
                _disable_qwen_for_run(f"post-load guard: {reason}")
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                return image_path

        img = PILImage.open(image_path).convert("RGB")
        result = _QWEN_EDIT_PIPELINE(
            image=[img],
            prompt=_CLEANUP_PROMPT,
            negative_prompt=" ",
            num_inference_steps=28,
            guidance_scale=1.0,
            true_cfg_scale=4.0,
            num_images_per_prompt=1,
        ).images[0]

        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.save(str(output_path))
        logger.info("Qwen-Image-Edit cleanup saved to %s", output_path)
        return output_path

    except RuntimeError as exc:
        message = str(exc).lower()
        if "out of memory" in message or ("cuda" in message and "memory" in message):
            _disable_qwen_for_run(f"CUDA OOM during inference: {exc}")
            try:
                import torch  # type: ignore

                torch.cuda.empty_cache()
            except Exception:
                pass
            return image_path
        logger.warning("Qwen-Image-Edit cleanup failed: %s, using original crop", exc)
        return image_path
    except ImportError:
        logger.warning("diffusers not installed, skipping Qwen-Image-Edit cleanup")
        return image_path
    except Exception as exc:
        logger.warning("Qwen-Image-Edit cleanup failed: %s, using original crop", exc)
        return image_path


def _cleanup_with_nano_banana(image_path: Path, output_path: Path) -> Optional[Path]:
    """Clean up crop using Google Gemini 3 Pro Image (Nano Banana Pro)."""
    api_key = (os.getenv("GOOGLE_GENAI_API_KEY") or "").strip()
    if not api_key:
        logger.warning("GOOGLE_GENAI_API_KEY not set, skipping Nano Banana cleanup")
        return image_path

    try:
        from google import genai  # type: ignore

        client = genai.Client(api_key=api_key)

        image_bytes = image_path.read_bytes()

        response = client.models.generate_content(
            model="gemini-2.0-flash-exp",
            contents=[
                {
                    "parts": [
                        {"inline_data": {"mime_type": "image/png", "data": base64.b64encode(image_bytes).decode()}},
                        {"text": _CLEANUP_PROMPT},
                    ]
                }
            ],
            config={
                "response_modalities": ["IMAGE"],
            },
        )

        if response.candidates and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if hasattr(part, "inline_data") and part.inline_data:
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    output_path.write_bytes(base64.b64decode(part.inline_data.data))
                    logger.info("Nano Banana cleanup saved to %s", output_path)
                    return output_path

        logger.warning("Nano Banana returned no image, using original crop")
        return image_path

    except Exception as exc:
        logger.warning("Nano Banana cleanup failed: %s, using original crop", exc)
        return image_path


def _cleanup_with_gpt_image(image_path: Path, output_path: Path) -> Optional[Path]:
    """Clean up crop using OpenAI GPT Image 1.5."""
    api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if not api_key:
        logger.warning("OPENAI_API_KEY not set, skipping GPT Image cleanup")
        return image_path

    try:
        from openai import OpenAI  # type: ignore

        client = OpenAI(api_key=api_key)

        image_bytes = image_path.read_bytes()
        image_b64 = base64.b64encode(image_bytes).decode()

        response = client.images.edit(
            model="gpt-image-1.5",
            image=f"data:image/png;base64,{image_b64}",
            prompt=_CLEANUP_PROMPT,
            size="512x512",
        )

        if response.data and response.data[0].b64_json:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(base64.b64decode(response.data[0].b64_json))
            logger.info("GPT Image cleanup saved to %s", output_path)
            return output_path

        logger.warning("GPT Image returned no data, using original crop")
        return image_path

    except Exception as exc:
        logger.warning("GPT Image cleanup failed: %s, using original crop", exc)
        return image_path
