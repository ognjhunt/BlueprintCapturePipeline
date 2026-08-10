"""Minimal inference-backed admission for hosted construction models."""

from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import json
import os
import struct
import urllib.error
import urllib.request
import zlib
from pathlib import Path
from typing import Any, Callable, Mapping

from .common import utc_now_iso, write_json
from .decision_evidence_contracts import canonical_digest
from .openai_successor_models import OPENAI_REASONING_EFFORT, OPENAI_TEXT_MODEL


SCHEMA_VERSION = "hosted_model_inference_preflight.v2"
LEGACY_SCHEMA_VERSION = "hosted_model_inference_preflight.v1"
PROBE_PROFILE = "multimodal_structured_json.v1"
REQUIRED_CAPABILITIES = ("image_input", "structured_json")
REASONING_EFFORTS = frozenset({"none", "low", "medium", "high", "xhigh", "max"})
BACKENDS = {
    "openai": {
        "endpoint": "https://api.openai.com/v1/chat/completions",
        "model": OPENAI_TEXT_MODEL,
        "legacy_model": "gpt-4.1",
        "default_reasoning_effort": OPENAI_REASONING_EFFORT,
        "completion_token_field": "max_completion_tokens",
        "env": "OPENAI_API_KEY",
        "secret_file": "openai_api_key",
    },
    "nvidia_nim": {
        "endpoint": "https://integrate.api.nvidia.com/v1/chat/completions",
        "model": "meta/llama-3.2-11b-vision-instruct",
        "completion_token_field": "max_tokens",
        "env": "NVIDIA_API_KEY",
        "secret_file": "nvidia_nim_api_key",
        "legacy_secret_file": "ngc_api_key",
    },
}


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    body = kind + payload
    return struct.pack(">I", len(payload)) + body + struct.pack(">I", binascii.crc32(body))


def _probe_png() -> bytes:
    """Return a generated 16x16 red PNG with no third-party source bytes."""

    width = height = 16
    rows = b"".join(b"\x00" + (b"\xff\x00\x00" * width) for _ in range(height))
    return b"".join(
        (
            b"\x89PNG\r\n\x1a\n",
            _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)),
            _png_chunk(b"IDAT", zlib.compress(rows, level=9)),
            _png_chunk(b"IEND", b""),
        )
    )


def _choice_content(document: Mapping[str, Any]) -> str:
    choices = document.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    choice = choices[0]
    if not isinstance(choice, Mapping):
        return ""
    message = choice.get("message")
    if not isinstance(message, Mapping):
        return ""
    content = message.get("content")
    return content if isinstance(content, str) else ""


def _usage(document: Mapping[str, Any]) -> dict[str, int | None]:
    raw = document.get("usage")
    usage = raw if isinstance(raw, Mapping) else {}
    return {
        "input_tokens": (
            int(usage["prompt_tokens"])
            if isinstance(usage.get("prompt_tokens"), int)
            else None
        ),
        "output_tokens": (
            int(usage["completion_tokens"])
            if isinstance(usage.get("completion_tokens"), int)
            else None
        ),
        "total_tokens": (
            int(usage["total_tokens"])
            if isinstance(usage.get("total_tokens"), int)
            else None
        ),
    }


def _provider_error(body: bytes) -> dict[str, str | None]:
    """Retain only provider error routing fields, never echoed request data."""

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


def _secret(backend: str) -> tuple[str, str]:
    contract = BACKENDS[backend]
    env_name = str(contract["env"])
    value = str(os.environ.get(env_name) or "").strip()
    if value:
        return value, env_name
    root = Path("~/.blueprint-secrets").expanduser()
    for field in ("secret_file", "legacy_secret_file"):
        filename = str(contract.get(field) or "")
        path = root / filename
        if (
            filename
            and path.is_file()
            and (secret := path.read_text(encoding="utf-8").strip())
        ):
            return secret, filename
    return "", "missing"


def _default_post(
    endpoint: str, headers: Mapping[str, str], payload: bytes
) -> tuple[int, bytes]:
    request = urllib.request.Request(
        endpoint, headers=dict(headers), data=payload, method="POST"
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return int(response.status), response.read()


def materialize_hosted_model_inference_preflight(
    *,
    output_path: str | Path,
    backend: str,
    model: str | None = None,
    endpoint: str | None = None,
    reasoning_effort: str | None = None,
    generated_at: str | None = None,
    secret_loader: Callable[[str], tuple[str, str]] = _secret,
    http_post: Callable[[str, Mapping[str, str], bytes], tuple[int, bytes]] = (
        _default_post
    ),
) -> dict[str, Any]:
    """Require an image-grounded strict-JSON response from the exact model."""

    if backend not in BACKENDS:
        raise ValueError("hosted_model_preflight_backend_invalid")
    contract = BACKENDS[backend]
    resolved_reasoning_effort = reasoning_effort or contract.get(
        "default_reasoning_effort"
    )
    if (
        resolved_reasoning_effort is not None
        and resolved_reasoning_effort not in REASONING_EFFORTS
    ):
        raise ValueError("hosted_model_preflight_reasoning_effort_invalid")
    resolved_model = model or str(contract["model"])
    resolved_endpoint = endpoint or str(contract["endpoint"])
    key, key_source = secret_loader(backend)
    blockers: list[str] = []
    http_status: int | None = None
    response_model: str | None = None
    choice_count = 0
    probe_response_validated = False
    verified_capabilities: list[str] = []
    usage = {"input_tokens": None, "output_tokens": None, "total_tokens": None}
    provider_error: dict[str, str | None] = {}
    probe_performed = False
    probe_png = _probe_png()
    probe_image_sha256 = "sha256:" + hashlib.sha256(probe_png).hexdigest()
    if not key:
        blockers.append("hosted_model_api_key_missing")
    else:
        request_document: dict[str, Any] = {
            "model": resolved_model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Inspect the image and report its dominant color. "
                                "Return only the required JSON object."
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/png;base64,"
                                + base64.b64encode(probe_png).decode("ascii"),
                                "detail": "low",
                            },
                        },
                    ],
                }
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "hosted_model_capability_probe",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "dominant_color": {
                                "type": "string",
                                "enum": ["red", "green", "blue"],
                            }
                        },
                        "required": ["dominant_color"],
                        "additionalProperties": False,
                    },
                },
            },
            str(contract["completion_token_field"]): 256,
        }
        if resolved_reasoning_effort is None:
            request_document["temperature"] = 0
        else:
            request_document["reasoning_effort"] = resolved_reasoning_effort
        payload = json.dumps(request_document, separators=(",", ":")).encode("utf-8")
        try:
            probe_performed = True
            http_status, body = http_post(
                resolved_endpoint,
                {
                    "Authorization": f"Bearer {key}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "User-Agent": "BlueprintCapturePipeline-ADP/1",
                },
                payload,
            )
            document = json.loads(body.decode("utf-8"))
            response_model = str(document.get("model") or "") or None
            choices = document.get("choices")
            choice_count = len(choices) if isinstance(choices, list) else 0
            usage = _usage(document)
            try:
                probe_response = json.loads(_choice_content(document))
            except json.JSONDecodeError:
                probe_response = None
            probe_response_validated = (
                isinstance(probe_response, Mapping)
                and dict(probe_response) == {"dominant_color": "red"}
            )
            if http_status != 200 or choice_count < 1:
                blockers.append("hosted_model_inference_response_invalid")
            elif not probe_response_validated:
                blockers.append("hosted_model_capability_response_invalid")
            else:
                verified_capabilities.extend(REQUIRED_CAPABILITIES)
        except urllib.error.HTTPError as exc:
            http_status = int(exc.code)
            provider_error = _provider_error(exc.read())
            blockers.append("hosted_model_inference_http_error")
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
            blockers.append("hosted_model_inference_request_failed")
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at or utc_now_iso(),
        "status": "qualified" if not blockers else "blocked",
        "backend": backend,
        "endpoint": resolved_endpoint,
        "model": resolved_model,
        "credential_source": key_source,
        "credential_validated": http_status == 200 and choice_count >= 1,
        "inference_http_status": http_status,
        "response_model": response_model,
        "choice_count": choice_count,
        "request_method": "POST",
        "probe_profile": PROBE_PROFILE,
        "required_capabilities": list(REQUIRED_CAPABILITIES),
        "verified_capabilities": sorted(verified_capabilities),
        "probe_response_validated": probe_response_validated,
        "probe_image": {
            "authority": "blueprint_generated_primary_color_fixture",
            "mime_type": "image/png",
            "width": 16,
            "height": 16,
            "sha256": probe_image_sha256,
            "uploaded_scene_bytes": False,
        },
        "reasoning_effort": resolved_reasoning_effort,
        "max_output_tokens": 256,
        "usage": usage,
        "provider_error": provider_error,
        "inference_probe_performed": probe_performed,
        "provider_mutations_performed": 0,
        "raw_secret_values_recorded": False,
        "blockers": sorted(set(blockers)),
        "receipt_digest": "",
    }
    receipt["receipt_digest"] = canonical_digest(
        receipt, digest_field="receipt_digest"
    )
    write_json(Path(output_path).expanduser().resolve(), receipt)
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--backend", required=True, choices=sorted(BACKENDS))
    parser.add_argument("--model")
    parser.add_argument("--endpoint")
    parser.add_argument("--reasoning-effort", choices=sorted(REASONING_EFFORTS))
    args = parser.parse_args(argv)
    result = materialize_hosted_model_inference_preflight(
        output_path=args.output,
        backend=args.backend,
        model=args.model,
        endpoint=args.endpoint,
        reasoning_effort=args.reasoning_effort,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "qualified" else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BACKENDS",
    "LEGACY_SCHEMA_VERSION",
    "PROBE_PROFILE",
    "REQUIRED_CAPABILITIES",
    "REASONING_EFFORTS",
    "SCHEMA_VERSION",
    "materialize_hosted_model_inference_preflight",
]
