"""Typed SDK image tools with conservative, independent context accounting.

The September 11, 2026 OpenAI images/vision table assigns these patch models
1.2 tokens per 32px patch. We count the unresized image plus rounding/framing
headroom, never base64 bytes as tokens. Unknown models and remote image URLs
are refused. See https://developers.openai.com/api/docs/guides/images-vision.
"""
from __future__ import annotations

import base64
import binascii
import io
import json
import math
from collections.abc import Mapping

from PIL import Image

MODELS = frozenset({"gpt-5.6-terra", "gpt-5.6-sol", "gpt-5.6-luna", "gpt-6-astra"})


def encode_tool_output(value, *, model):
    """Return SDK content objects and a cumulative context charge, before send."""
    from agents import ToolOutputImage, ToolOutputText

    if isinstance(value, Mapping):
        text = json.dumps(dict(value), sort_keys=True, allow_nan=False)
        return None, len(text.encode("utf-8")) + 256
    if not isinstance(value, list) or not value or model not in MODELS:
        raise ValueError("agents_sdk_image_tool_model_or_content_invalid")
    result, tokens = [], 0
    for part in value:
        if not isinstance(part, dict):
            raise ValueError("agents_sdk_image_tool_content_invalid")
        if set(part) == {"type", "text"} and part["type"] == "input_text" and isinstance(part["text"], str):
            result.append(ToolOutputText(text=part["text"]))
            tokens += len(part["text"].encode("utf-8")) + 256
        elif set(part) == {"type", "image_url"} and part["type"] == "input_image":
            url = part["image_url"]
            prefix = "data:image/png;base64,"
            if not isinstance(url, str) or not url.startswith(prefix) or len(url) > 64_000_000:
                raise ValueError("agents_sdk_image_tool_source_invalid")
            try:
                raw = base64.b64decode(url[len(prefix):], validate=True)
                with Image.open(io.BytesIO(raw)) as picture:
                    width, height = picture.size
                    if picture.format != "PNG" or getattr(picture, "n_frames", 1) != 1:
                        raise ValueError("agents_sdk_image_tool_format_invalid")
                    picture.verify()
            except (binascii.Error, OSError, Image.DecompressionBombError) as exc:
                raise ValueError("agents_sdk_image_tool_decode_failed") from exc
            patches = math.ceil(width / 32) * math.ceil(height / 32)
            if not 0 < patches <= 30_000 or max(width, height) > 65_535:
                raise ValueError("agents_sdk_image_tool_size_not_qualified")
            # Explicit auto preserves the source tool's existing provider sizing.
            result.append(ToolOutputImage(image_url=url, detail="auto"))
            tokens += math.ceil(patches * 1.2) + 1 + 256
        else:
            raise ValueError("agents_sdk_image_tool_content_invalid")
    return result, tokens
