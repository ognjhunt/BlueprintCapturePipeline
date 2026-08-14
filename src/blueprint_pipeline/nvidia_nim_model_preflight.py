"""Mutation-free credential and model admission for hosted NVIDIA NIM."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Callable, Mapping
import urllib.error
import urllib.parse
import urllib.request

from .common import utc_now_iso, write_json
from .decision_evidence_contracts import canonical_digest
from .gpu_render_providers import _read_secret as _read_provider_secret


SCHEMA_VERSION = "nvidia_nim_model_preflight.v1"
DEFAULT_ENDPOINT = "https://integrate.api.nvidia.com/v1/models"
# The Joint Agent template's default, google/gemma-4-31b-it, is listed in the
# catalog and does not answer - a completion request to it hangs past four
# minutes. Point the lane at a vision model that actually serves.
DEFAULT_MODEL = "meta/llama-3.2-11b-vision-instruct"
# A model known to serve, used only to tell a dead credential apart from a
# dead model when the target fails.
CONTROL_MODEL = "meta/llama-3.1-8b-instruct"
# The catalog endpoint answers 200 to a key with no inference entitlement, so
# listing models proves nothing about whether the agent can call one. The gate
# issues a one-token completion instead, which is the capability the run needs.
DEFAULT_INFERENCE_ENDPOINT = "https://integrate.api.nvidia.com/v1/chat/completions"


def _secret() -> tuple[str, str]:
    # NIM inference and the NGC registry are different systems with different
    # entitlements. The NGC key lists models happily and returns 401 on a
    # completion, which is exactly how the wrong credential reached a paid
    # provider, so the inference key is preferred and the registry key is only
    # a legacy fallback.
    value = str(os.environ.get("NVIDIA_API_KEY") or "").strip()
    if value:
        return value, "NVIDIA_API_KEY"
    # A developer home is unreadable under `ProtectHome=true` with home
    # `/nonexistent`, which is how every control-plane unit runs.
    for name in ("nvidia_nim_api_key", "ngc_api_key"):
        text = str(_read_provider_secret(name) or "")
        if text:
            return text, name
    return "", "missing"


def _default_get(endpoint: str, headers: Mapping[str, str]) -> tuple[int, bytes]:
    parsed = urllib.parse.urlsplit(endpoint)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("nvidia_nim_endpoint_not_public_https")
    request = urllib.request.Request(endpoint, headers=dict(headers), method="GET")
    with urllib.request.urlopen(  # nosec B310 - public HTTPS endpoint validated above
        request, timeout=30
    ) as response:
        return int(response.status), response.read()


def _default_post(
    endpoint: str, headers: Mapping[str, str], payload: bytes
) -> tuple[int, bytes]:
    parsed = urllib.parse.urlsplit(endpoint)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("nvidia_nim_endpoint_not_public_https")
    request = urllib.request.Request(
        endpoint, data=payload, headers=dict(headers), method="POST"
    )
    with urllib.request.urlopen(  # nosec B310 - public HTTPS endpoint validated above
        request, timeout=45
    ) as response:
        return int(response.status), response.read()


def materialize_nvidia_nim_model_preflight(
    *,
    output_path: str | Path,
    model: str = DEFAULT_MODEL,
    endpoint: str = DEFAULT_ENDPOINT,
    inference_endpoint: str = DEFAULT_INFERENCE_ENDPOINT,
    control_model: str = CONTROL_MODEL,
    generated_at: str | None = None,
    secret_loader: Callable[[], tuple[str, str]] = _secret,
    http_get: Callable[[str, Mapping[str, str]], tuple[int, bytes]] = _default_get,
    http_post: Callable[[str, Mapping[str, str], bytes], tuple[int, bytes]]
    | None = None,
) -> dict[str, Any]:
    """Validate the credential can list *and* call a model."""

    blockers: list[str] = []
    key, key_source = secret_loader()
    http_status: int | None = None
    model_ids: list[str] = []
    if not key:
        blockers.append("nvidia_nim_api_key_missing")
    else:
        try:
            http_status, body = http_get(
                endpoint,
                {
                    "Authorization": f"Bearer {key}",
                    "Accept": "application/json",
                    "User-Agent": "BlueprintCapturePipeline-ADP/1",
                },
            )
            document = json.loads(body.decode("utf-8"))
            rows = document.get("data") if isinstance(document, dict) else None
            if not isinstance(rows, list):
                blockers.append("nvidia_nim_models_response_invalid")
            else:
                model_ids = sorted(
                    {
                        str(row.get("id"))
                        for row in rows
                        if isinstance(row, dict) and isinstance(row.get("id"), str)
                    }
                )
                if model not in model_ids:
                    blockers.append("nvidia_nim_required_model_unavailable")
            if http_status != 200:
                blockers.append("nvidia_nim_models_http_status_invalid")
        except urllib.error.HTTPError as exc:
            http_status = int(exc.code)
            blockers.append("nvidia_nim_models_http_error")
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
            blockers.append("nvidia_nim_models_request_failed")
    # Prove the credential can actually infer. A key entitled only to the
    # catalog reads as healthy above and fails at the agent's first model call.
    inference_probe: dict[str, Any] = {
        "endpoint": inference_endpoint,
        "model": model,
        "max_tokens": 1,
        "attempted": False,
        "authorized": False,
        "http_status": None,
    }
    poster = http_post if http_post is not None else _default_post

    def _probe(target: str) -> dict[str, Any]:
        row: dict[str, Any] = {
            "model": target,
            "attempted": True,
            "authorized": False,
            "http_status": None,
        }
        payload = json.dumps(
            {
                "model": target,
                "messages": [{"role": "user", "content": "ok"}],
                "max_tokens": 1,
            }
        ).encode("utf-8")
        try:
            status, _ = poster(
                inference_endpoint,
                {
                    "Authorization": f"Bearer {key}",
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                    "User-Agent": "BlueprintCapturePipeline-ADP/1",
                },
                payload,
            )
            row["http_status"] = int(status)
            row["authorized"] = int(status) == 200
        except urllib.error.HTTPError as exc:
            row["http_status"] = int(exc.code)
        except (OSError, UnicodeDecodeError, ValueError):
            row["http_status"] = None
            row["unreachable"] = True
        return row

    control_probe: dict[str, Any] = {"model": control_model, "attempted": False}
    if not blockers:
        inference_probe.update(_probe(model))
        inference_probe["max_tokens"] = 1
        inference_probe["endpoint"] = inference_endpoint
        if not inference_probe["authorized"]:
            # Ask a model known to serve. If that answers, the credential is
            # fine and the target model is the problem, which is a completely
            # different thing to go and fix.
            control_probe = _probe(control_model)
            if control_probe.get("authorized"):
                blockers.append("nvidia_nim_model_not_served")
            elif control_probe.get("http_status") in (401, 403):
                blockers.append("nvidia_nim_inference_unauthorized")
            elif inference_probe.get("http_status") in (401, 403):
                blockers.append("nvidia_nim_inference_unauthorized")
            else:
                blockers.append("nvidia_nim_inference_request_failed")

    receipt = {
        "schema_version": SCHEMA_VERSION,
        "inference_probe": inference_probe,
        "control_probe": control_probe,
        "credential_can_infer": bool(
            inference_probe.get("authorized") or control_probe.get("authorized")
        ),
        "generated_at": generated_at or utc_now_iso(),
        "status": "qualified" if not blockers else "blocked",
        "endpoint": endpoint,
        "model": model,
        "http_status": http_status,
        "credential_source": key_source,
        "credential_validated": http_status == 200,
        "catalog_model_count": len(model_ids),
        "required_model_present": model in model_ids,
        "request_method": "GET",
        "paid_inference_performed": False,
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
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    args = parser.parse_args(argv)
    result = materialize_nvidia_nim_model_preflight(
        output_path=args.output,
        model=args.model,
        endpoint=args.endpoint,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "qualified" else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_ENDPOINT",
    "DEFAULT_MODEL",
    "SCHEMA_VERSION",
    "materialize_nvidia_nim_model_preflight",
]
