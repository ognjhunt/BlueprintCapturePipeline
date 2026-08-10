"""Minimal inference-backed admission for hosted construction models."""

from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Mapping

from .common import utc_now_iso, write_json
from .decision_evidence_contracts import canonical_digest


SCHEMA_VERSION = "hosted_model_inference_preflight.v1"
BACKENDS = {
    "openai": {
        "endpoint": "https://api.openai.com/v1/chat/completions",
        "model": "gpt-4.1",
        "env": "OPENAI_API_KEY",
        "secret_file": "openai_api_key",
    },
    "nvidia_nim": {
        "endpoint": "https://integrate.api.nvidia.com/v1/chat/completions",
        "model": "meta/llama-3.2-11b-vision-instruct",
        "env": "NVIDIA_API_KEY",
        "secret_file": "nvidia_nim_api_key",
        "legacy_secret_file": "ngc_api_key",
    },
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
    generated_at: str | None = None,
    secret_loader: Callable[[str], tuple[str, str]] = _secret,
    http_post: Callable[[str, Mapping[str, str], bytes], tuple[int, bytes]] = (
        _default_post
    ),
) -> dict[str, Any]:
    """Require one bounded response; catalog visibility is not credential proof."""

    if backend not in BACKENDS:
        raise ValueError("hosted_model_preflight_backend_invalid")
    contract = BACKENDS[backend]
    resolved_model = model or str(contract["model"])
    resolved_endpoint = endpoint or str(contract["endpoint"])
    key, key_source = secret_loader(backend)
    blockers: list[str] = []
    http_status: int | None = None
    response_model: str | None = None
    choice_count = 0
    probe_performed = False
    if not key:
        blockers.append("hosted_model_api_key_missing")
    else:
        payload = json.dumps(
            {
                "model": resolved_model,
                "messages": [{"role": "user", "content": "Reply OK"}],
                "max_tokens": 1,
                "temperature": 0,
            },
            separators=(",", ":"),
        ).encode("utf-8")
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
            if http_status != 200 or choice_count < 1:
                blockers.append("hosted_model_inference_response_invalid")
        except urllib.error.HTTPError as exc:
            http_status = int(exc.code)
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
        "max_output_tokens": 1,
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
    args = parser.parse_args(argv)
    result = materialize_hosted_model_inference_preflight(
        output_path=args.output,
        backend=args.backend,
        model=args.model,
        endpoint=args.endpoint,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "qualified" else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BACKENDS",
    "SCHEMA_VERSION",
    "materialize_hosted_model_inference_preflight",
]
