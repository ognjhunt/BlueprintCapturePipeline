"""Helpers for normalizing model-access environment variables."""

from __future__ import annotations

import os


def normalize_model_access_env() -> None:
    """Bridge common auth env aliases used across VM bootstrap and runtime paths."""
    hf_token = str(
        os.getenv("HUGGINGFACE_HUB_TOKEN")
        or os.getenv("HF_TOKEN")
        or os.getenv("HUGGING_FACE_HUB_TOKEN")
        or ""
    ).strip()
    if hf_token:
        os.environ.setdefault("HF_TOKEN", hf_token)
        os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", hf_token)

    ngc_token = str(
        os.getenv("NGC_API_KEY")
        or os.getenv("NVIDIA_NGC_API_KEY")
        or ""
    ).strip()
    if ngc_token:
        os.environ.setdefault("NGC_API_KEY", ngc_token)
