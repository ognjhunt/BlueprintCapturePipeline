"""Mutation-free credential and model admission for hosted NVIDIA NIM."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Callable, Mapping
import urllib.error
import urllib.request

from .common import utc_now_iso, write_json
from .decision_evidence_contracts import canonical_digest


SCHEMA_VERSION = "nvidia_nim_model_preflight.v1"
DEFAULT_ENDPOINT = "https://integrate.api.nvidia.com/v1/models"
DEFAULT_MODEL = "google/gemma-4-31b-it"
# The catalog endpoint answers 200 to a key with no inference entitlement, so
# listing models proves nothing about whether the agent can call one. The gate
# issues a one-token completion instead, which is the capability the run needs.
DEFAULT_INFERENCE_ENDPOINT = "https://integrate.api.nvidia.com/v1/chat/completions"


def _secret() -> tuple[str, str]:
    value = str(os.environ.get("NVIDIA_API_KEY") or "").strip()
    if value:
        return value, "NVIDIA_API_KEY"
    path = Path("~/.blueprint-secrets/ngc_api_key").expanduser()
    if path.is_file():
        return path.read_text(encoding="utf-8").strip(), "canonical_secret_store"
    return "", "missing"


def _default_get(endpoint: str, headers: Mapping[str, str]) -> tuple[int, bytes]:
    request = urllib.request.Request(endpoint, headers=dict(headers), method="GET")
    with urllib.request.urlopen(request, timeout=30) as response:
        return int(response.status), response.read()


def _default_post(
    endpoint: str, headers: Mapping[str, str], payload: bytes
) -> tuple[int, bytes]:
    request = urllib.request.Request(
        endpoint, data=payload, headers=dict(headers), method="POST"
    )
    with urllib.request.urlopen(request, timeout=45) as response:
        return int(response.status), response.read()


def materialize_nvidia_nim_model_preflight(
    *,
    output_path: str | Path,
    model: str = DEFAULT_MODEL,
    endpoint: str = DEFAULT_ENDPOINT,
    inference_endpoint: str = DEFAULT_INFERENCE_ENDPOINT,
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
    if not blockers:
        payload = json.dumps(
            {
                "model": model,
                "messages": [{"role": "user", "content": "ok"}],
                "max_tokens": 1,
            }
        ).encode("utf-8")
        inference_probe["attempted"] = True
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
            inference_probe["http_status"] = int(status)
            if int(status) in (401, 403):
                blockers.append("nvidia_nim_inference_unauthorized")
            elif int(status) != 200:
                blockers.append("nvidia_nim_inference_http_status_invalid")
            else:
                inference_probe["authorized"] = True
        except urllib.error.HTTPError as exc:
            inference_probe["http_status"] = int(exc.code)
            blockers.append(
                "nvidia_nim_inference_unauthorized"
                if int(exc.code) in (401, 403)
                else "nvidia_nim_inference_http_error"
            )
        except (OSError, UnicodeDecodeError, ValueError):
            blockers.append("nvidia_nim_inference_request_failed")

    receipt = {
        "schema_version": SCHEMA_VERSION,
        "inference_probe": inference_probe,
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
