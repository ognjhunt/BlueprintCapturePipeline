"""Resident Cosmos worker that keeps the model loaded across chunk requests."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np
from PIL import Image

from .cosmos_inference import _DEFAULT_COSMOS_MODEL_ID, _cosmos_image_to_world, describe_cosmos_model, load_cosmos_model


def _emit(payload: Mapping[str, Any]) -> None:
    sys.stdout.write(json.dumps(dict(payload)) + "\n")
    sys.stdout.flush()


def _generate(model: Any, request: Mapping[str, Any]) -> Dict[str, Any]:
    request_id = str(request.get("request_id") or "")
    input_path = Path(str(request.get("input_path") or "")).expanduser().resolve()
    output_path = Path(str(request.get("output_path") or "")).expanduser().resolve()
    if not input_path.is_file():
        raise RuntimeError(f"conditioning_image_missing:{input_path}")

    started_at = time.monotonic()
    conditioning_image = np.array(Image.open(input_path).convert("RGB"))
    _cosmos_image_to_world(
        conditioning_image=conditioning_image,
        output_path=output_path,
        cosmos_model=model,
        num_frames=int(request.get("num_frames") or 57),
        width=int(request.get("width") or conditioning_image.shape[1]),
        height=int(request.get("height") or conditioning_image.shape[0]),
        guidance_scale=float(request.get("guidance_scale") or 7.0),
        num_steps=int(request.get("num_steps") or 35),
    )
    video_path = output_path.with_suffix(".mp4")
    return {
        "type": "result",
        "request_id": request_id,
        "ok": output_path.is_file(),
        "output_path": str(output_path),
        "video_path": str(video_path) if video_path.is_file() else None,
        "generation_ms": int(round((time.monotonic() - started_at) * 1000.0)),
    }


def main() -> int:
    try:
        model = load_cosmos_model(model_id=_DEFAULT_COSMOS_MODEL_ID)
    except Exception as exc:
        _emit({"type": "error", "stage": "startup", "error": str(exc)})
        return 1

    description = describe_cosmos_model(model)
    _emit(
        {
            "type": "ready",
            "backend": str(description.get("backend") or description.get("worker_backend") or "unknown"),
            "model_id": _DEFAULT_COSMOS_MODEL_ID,
        }
    )

    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError as exc:
            _emit({"type": "error", "stage": "decode", "error": str(exc)})
            continue
        if not isinstance(request, dict):
            _emit({"type": "error", "stage": "request", "error": "request_must_be_object"})
            continue
        request_type = str(request.get("type") or "")
        if request_type == "ping":
            _emit({"type": "pong"})
            continue
        if request_type != "generate":
            _emit({"type": "error", "stage": "request", "error": f"unsupported_request:{request_type}"})
            continue
        try:
            _emit(_generate(model, request))
        except Exception as exc:
            _emit(
                {
                    "type": "result",
                    "request_id": str(request.get("request_id") or ""),
                    "ok": False,
                    "error": str(exc),
                }
            )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
