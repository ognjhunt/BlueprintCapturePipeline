"""Inference-backed admission for hosted image-generation models."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import struct
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Mapping

from .common import utc_now_iso, write_json
from .decision_evidence_contracts import canonical_digest
from .openai_successor_models import OPENAI_IMAGE_MODEL


SCHEMA_VERSION = "hosted_image_generation_preflight.v1"
PROBE_PROFILE = "text_to_png_generation.v1"
MODEL = OPENAI_IMAGE_MODEL
ENDPOINT = "https://api.openai.com/v1/images/generations"
SIZE = "1024x1024"
QUALITY = "low"
ESTIMATED_OUTPUT_COST_USD = 0.006


def _secret() -> tuple[str, str]:
    value = str(os.getenv("OPENAI_API_KEY") or "").strip()
    if value:
        return value, "OPENAI_API_KEY"
    path = Path("~/.blueprint-secrets/openai_api_key").expanduser()
    if path.is_file() and (value := path.read_text(encoding="utf-8").strip()):
        return value, path.name
    return "", "missing"


def _post(
    endpoint: str, headers: Mapping[str, str], payload: bytes
) -> tuple[int, bytes]:
    request = urllib.request.Request(
        endpoint, headers=dict(headers), data=payload, method="POST"
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        return int(response.status), response.read()


def _safe_provider_error(body: bytes) -> dict[str, str | None]:
    try:
        document = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return {}
    raw = document.get("error") if isinstance(document, Mapping) else None
    error = raw if isinstance(raw, Mapping) else {}
    return {
        key: str(error[key]) if error.get(key) is not None else None
        for key in ("type", "code", "param")
    }


def _png_dimensions(value: bytes) -> tuple[int, int] | None:
    if len(value) < 24 or value[:8] != b"\x89PNG\r\n\x1a\n" or value[12:16] != b"IHDR":
        return None
    return struct.unpack(">II", value[16:24])


def _usage(document: Mapping[str, Any]) -> dict[str, int | None]:
    raw = document.get("usage")
    usage = raw if isinstance(raw, Mapping) else {}
    return {
        key: int(usage[key]) if isinstance(usage.get(key), int) else None
        for key in ("input_tokens", "output_tokens", "total_tokens")
    }


def materialize_hosted_image_generation_preflight(
    *,
    output_path: str | Path,
    model: str = MODEL,
    endpoint: str = ENDPOINT,
    generated_at: str | None = None,
    secret_loader: Callable[[], tuple[str, str]] = _secret,
    http_post: Callable[[str, Mapping[str, str], bytes], tuple[int, bytes]] = _post,
) -> dict[str, Any]:
    """Generate and validate one bounded first-party PNG with the exact model."""

    if not str(model).strip():
        raise ValueError("hosted_image_generation_model_missing")
    secret, credential_source = secret_loader()
    blockers: list[str] = []
    status: int | None = None
    image_bytes = b""
    usage = {"input_tokens": None, "output_tokens": None, "total_tokens": None}
    provider_error: dict[str, str | None] = {}
    performed = False
    prompt = (
        "A single solid red square centered on a plain white background, "
        "flat geometric icon, no text, no logos."
    )
    request_document = {
        "model": model,
        "prompt": prompt,
        "n": 1,
        "size": SIZE,
        "quality": QUALITY,
    }
    if not secret:
        blockers.append("hosted_image_generation_api_key_missing")
    else:
        try:
            performed = True
            status, body = http_post(
                endpoint,
                {
                    "Authorization": f"Bearer {secret}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "User-Agent": "BlueprintCapturePipeline-ADP/1",
                },
                json.dumps(request_document, separators=(",", ":")).encode("utf-8"),
            )
            document = json.loads(body.decode("utf-8"))
            usage = _usage(document)
            data = document.get("data") if isinstance(document, Mapping) else None
            encoded = (
                data[0].get("b64_json")
                if isinstance(data, list) and data and isinstance(data[0], Mapping)
                else None
            )
            image_bytes = base64.b64decode(encoded, validate=True) if isinstance(encoded, str) else b""
            if status != 200:
                blockers.append("hosted_image_generation_response_invalid")
            elif _png_dimensions(image_bytes) != (1024, 1024):
                blockers.append("hosted_image_generation_png_invalid")
        except urllib.error.HTTPError as exc:
            status = int(exc.code)
            provider_error = _safe_provider_error(exc.read())
            blockers.append("hosted_image_generation_http_error")
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
            blockers.append("hosted_image_generation_request_failed")

    dimensions = _png_dimensions(image_bytes)
    receipt: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at or utc_now_iso(),
        "status": "qualified" if not blockers else "blocked",
        "provider": "openai",
        "endpoint": endpoint,
        "model": model,
        "credential_source": credential_source,
        "request_method": "POST",
        "probe_profile": PROBE_PROFILE,
        "prompt_sha256": "sha256:" + hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "requested_output": {"count": 1, "size": SIZE, "quality": QUALITY, "format": "png"},
        "inference_http_status": status,
        "output": {
            "count": 1 if image_bytes else 0,
            "width": dimensions[0] if dimensions else None,
            "height": dimensions[1] if dimensions else None,
            "size_bytes": len(image_bytes),
            "sha256": "sha256:" + hashlib.sha256(image_bytes).hexdigest() if image_bytes else None,
            "bytes_retained": False,
        },
        "usage": usage,
        "estimated_output_cost_usd": ESTIMATED_OUTPUT_COST_USD if image_bytes else 0.0,
        "provider_error": provider_error,
        "inference_probe_performed": performed,
        "uploaded_scene_bytes": False,
        "raw_secret_values_recorded": False,
        "blockers": sorted(set(blockers)),
        "receipt_digest": "",
    }
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    write_json(Path(output_path).expanduser().resolve(), receipt)
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--endpoint", default=ENDPOINT)
    args = parser.parse_args(argv)
    receipt = materialize_hosted_image_generation_preflight(
        output_path=args.output, model=args.model, endpoint=args.endpoint
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if receipt["status"] == "qualified" else 2


if __name__ == "__main__":
    raise SystemExit(main())
