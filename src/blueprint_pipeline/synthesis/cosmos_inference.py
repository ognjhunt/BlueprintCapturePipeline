"""Cosmos-Predict2.5-2B Image2World wrapper (zero-FT synthesis).

This module wraps NVIDIA's Cosmos-Predict2.5-2B in Image2World mode.
Given a splatted reference frame as the conditioning image, Cosmos generates
a plausible short video from that viewpoint.

The zero-FT path (no Blueprint fine-tuning) works because:
  - Depth splatting provides geometrically correct structure as the conditioning image
  - Cosmos I2W was trained to synthesise coherent continuations from any image
  - Warehouse/facility interiors share enough structure (flat floors, straight walls,
    overhead lighting) with training distribution for reasonable zero-FT quality

Fine-tuned quality (Phase 4B) will require:
  - A dataset of aligned Blueprint captures with paired reference/target frames
  - Fine-tuning Cosmos-Predict2.5-2B with Plücker ray map conditioning injected
    as residuals into video token sequence (per SWM training recipe)

For Phase 4A, use generate_view() with mode="splat_only" or mode="cosmos_i2w".

Cosmos model access:
  pip install cosmos-predict2-5  (NVIDIA's package, requires NGC token)
  or via HuggingFace: nvidia/Cosmos-Predict2.5-2B

Environment variable COSMOS_MODEL_ID overrides the default model path.
NGC_API_KEY is required for downloading model weights from NVIDIA NGC.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, List, Optional

import numpy as np

from ..model_access_env import normalize_model_access_env


normalize_model_access_env()


_DEFAULT_COSMOS_MODEL_ID = os.getenv(
    "COSMOS_MODEL_ID",
    "nvidia/Cosmos-Predict2.5-2B",
)

# Cosmos generation defaults — tuned for Blueprint facility captures
_DEFAULT_NUM_FRAMES = 57      # ~2 seconds at ~28fps; matches Cosmos training length
_DEFAULT_WIDTH = 1280
_DEFAULT_HEIGHT = 720
_DEFAULT_GUIDANCE_SCALE = 7.0
_DEFAULT_NUM_STEPS = 35


def generate_view(
    *,
    splatted_image: np.ndarray,          # [H, W, 3] uint8 RGB — depth-splatted reference
    coverage_mask: np.ndarray,           # [H, W] bool — True = valid pixels in splatted_image
    target_plucker_map: Optional[np.ndarray] = None,   # [6, H, W] float32 (for future conditioning)
    output_path: Path,
    mode: str = "splat_only",            # "splat_only" | "cosmos_i2w"
    cosmos_model: Optional[Any] = None,  # pre-loaded model; loads if None
    num_frames: int = _DEFAULT_NUM_FRAMES,
    width: int = _DEFAULT_WIDTH,
    height: int = _DEFAULT_HEIGHT,
    guidance_scale: float = _DEFAULT_GUIDANCE_SCALE,
    num_steps: int = _DEFAULT_NUM_STEPS,
) -> Path:
    """
    Generate a view from the splatted conditioning image.

    mode="splat_only":
      Saves splatted_image as a JPEG. No generative model invoked.
      This is the fastest path and geometrically exact. Holes and low-coverage
      regions will be visible. Use for debugging and as a quality floor.

    mode="cosmos_i2w":
      Passes splatted_image to Cosmos-Predict2.5-2B Image2World.
      Cosmos generates a short video starting from this frame.
      First frame of the video is saved as a JPEG; full video as MP4 alongside.
      Requires Cosmos installed and NGC credentials.

    Returns path to the output image (JPEG).
    """
    from PIL import Image

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if mode == "splat_only":
        img = Image.fromarray(splatted_image)
        img.save(output_path)
        return output_path

    if mode == "cosmos_i2w":
        return _cosmos_image_to_world(
            conditioning_image=splatted_image,
            output_path=output_path,
            cosmos_model=cosmos_model,
            num_frames=num_frames,
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            num_steps=num_steps,
        )

    raise ValueError(f"Unknown generation mode: {mode!r}. Use 'splat_only' or 'cosmos_i2w'.")


def load_cosmos_model(model_id: Optional[str] = None) -> Any:
    """
    Load Cosmos-Predict2.5-2B for inference.

    Tries two backends in order:
      1. cosmos_predict2_5 (NVIDIA's official package)
      2. HuggingFace diffusers pipeline (community wrapper)

    Returns a loaded model object. Pass this to generate_view(cosmos_model=...)
    to avoid reloading on every call.
    """
    mid = model_id or _DEFAULT_COSMOS_MODEL_ID
    model = _try_load_cosmos_official(mid)
    if model is not None:
        return model
    model = _try_load_cosmos_diffusers(mid)
    if model is not None:
        return model
    raise ImportError(
        "Could not load Cosmos-Predict2.5-2B. Install the cosmos-predict2-5 package "
        f"(pip install cosmos-predict2-5) or set COSMOS_MODEL_ID to a valid HuggingFace "
        f"model path. Tried: {mid}"
    )


# ---------------------------------------------------------------------------
# Internal: Cosmos I2W generation
# ---------------------------------------------------------------------------


def _cosmos_image_to_world(
    *,
    conditioning_image: np.ndarray,
    output_path: Path,
    cosmos_model: Optional[Any],
    num_frames: int,
    width: int,
    height: int,
    guidance_scale: float,
    num_steps: int,
) -> Path:
    from PIL import Image

    model = cosmos_model or load_cosmos_model()
    pil_image = Image.fromarray(conditioning_image).resize(
        (width, height), Image.LANCZOS
    )

    video_frames = _invoke_cosmos(
        model=model,
        conditioning_image=pil_image,
        num_frames=num_frames,
        width=width,
        height=height,
        guidance_scale=guidance_scale,
        num_steps=num_steps,
    )

    # Save first frame as JPEG (the synthesised view)
    first_frame = video_frames[0]
    if isinstance(first_frame, np.ndarray):
        Image.fromarray(first_frame).save(output_path)
    else:
        first_frame.save(output_path)

    # Save full video as MP4 alongside
    mp4_path = output_path.with_suffix(".mp4")
    _save_video(video_frames, mp4_path, fps=28)

    return output_path


def _invoke_cosmos(
    *,
    model: Any,
    conditioning_image: Any,     # PIL Image
    num_frames: int,
    width: int,
    height: int,
    guidance_scale: float,
    num_steps: int,
) -> List[Any]:
    """
    Dispatch to the loaded Cosmos model's generation interface.
    Handles both the official NVIDIA API and the HuggingFace diffusers API.
    """
    # --- NVIDIA official cosmos-predict2-5 package ---
    if hasattr(model, "generate") and hasattr(model, "image_to_world"):
        result = model.image_to_world(
            image=conditioning_image,
            num_frames=num_frames,
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            num_inference_steps=num_steps,
        )
        return _extract_frames(result)

    # --- HuggingFace diffusers pipeline (community wrapper) ---
    if hasattr(model, "__call__"):
        result = model(
            image=conditioning_image,
            num_frames=num_frames,
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            num_inference_steps=num_steps,
        )
        frames = getattr(result, "frames", None) or getattr(result, "images", None)
        if frames is not None:
            return list(frames[0]) if isinstance(frames[0], list) else list(frames)

    raise RuntimeError(
        f"Unrecognised Cosmos model interface: {type(model)}. "
        "Expected .image_to_world() or callable pipeline with frames/images output."
    )


def _extract_frames(result: Any) -> List[Any]:
    """Extract frame list from various Cosmos output formats."""
    if isinstance(result, (list, tuple)):
        return list(result)
    frames = getattr(result, "frames", None)
    if frames is not None:
        return list(frames[0]) if hasattr(frames[0], "__iter__") else list(frames)
    images = getattr(result, "images", None)
    if images is not None:
        return list(images)
    raise RuntimeError(f"Cannot extract frames from Cosmos output: {type(result)}")


def _save_video(frames: List[Any], path: Path, fps: int = 28) -> None:
    """Save a list of PIL Images or numpy arrays as an MP4 using FFmpeg via imageio."""
    try:
        import imageio
        import numpy as np
        from PIL import Image

        np_frames = []
        for f in frames:
            if isinstance(f, np.ndarray):
                np_frames.append(f)
            else:
                np_frames.append(np.array(f))

        writer = imageio.get_writer(str(path), fps=fps, codec="libx264", quality=8)
        for frame in np_frames:
            writer.append_data(frame)
        writer.close()
    except Exception:
        pass  # Video saving is best-effort; the frame JPEG is the primary output


# ---------------------------------------------------------------------------
# Backend loaders
# ---------------------------------------------------------------------------


def _try_load_cosmos_official(model_id: str) -> Optional[Any]:
    try:
        # NVIDIA's official cosmos-predict2-5 package
        from cosmos_predict2_5 import CosmosPredict25  # type: ignore[import]
        model = CosmosPredict25.from_pretrained(model_id)
        model.eval()
        return model
    except (ImportError, Exception):
        return None


def _try_load_cosmos_diffusers(model_id: str) -> Optional[Any]:
    try:
        import torch
        from diffusers import DiffusionPipeline  # type: ignore[import]
        pipe = DiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
        )
        if torch.cuda.is_available():
            pipe = pipe.to("cuda")
        return pipe
    except (ImportError, Exception):
        return None
