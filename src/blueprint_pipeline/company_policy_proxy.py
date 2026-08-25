"""Blueprint-owned Unix-socket proxy for an isolated company policy server.

The proxy accepts only ``POST /v1/actions`` on a Unix domain socket and sends
the exact JSON body to one fixed loopback upstream.  Caller headers, URLs,
redirects, streaming responses and arbitrary methods are never forwarded.
The surrounding sandbox gives this sidecar and the policy a network namespace
with no external interfaces, so loopback is their only IP path.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import urllib.error
import urllib.request
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse


ACTION_ROUTE = "/v1/actions"
MAX_REQUEST_BYTES_ENV = "BLUEPRINT_COMPANY_POLICY_PROXY_MAX_REQUEST_BYTES"
MAX_RESPONSE_BYTES_ENV = "BLUEPRINT_COMPANY_POLICY_PROXY_MAX_RESPONSE_BYTES"
UPSTREAM_PORT_ENV = "BLUEPRINT_COMPANY_POLICY_PROXY_UPSTREAM_PORT"
REQUEST_TIMEOUT_MS_ENV = "BLUEPRINT_COMPANY_POLICY_PROXY_REQUEST_TIMEOUT_MS"
DEFAULT_MAX_REQUEST_BYTES = 8 * 1024 * 1024
DEFAULT_MAX_RESPONSE_BYTES = 4 * 1024 * 1024
DEFAULT_REQUEST_TIMEOUT_MS = 5_000
_SECRET_KEYS = frozenset(
    {
        "authorization",
        "credential",
        "docker_config_json",
        "password",
        "registry_secret",
        "secret",
        "token",
    }
)


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
        return None


def _bounded_int_env(name: str, default: int, *, minimum: int, maximum: int) -> int:
    try:
        value = int(str(os.getenv(name) or ""))
    except ValueError:
        return default
    return value if minimum <= value <= maximum else default


def _contains_secret_carrier(value: Any) -> bool:
    if isinstance(value, dict):
        return any(
            str(key).strip().lower() in _SECRET_KEYS or _contains_secret_carrier(nested)
            for key, nested in value.items()
        )
    if isinstance(value, list):
        return any(_contains_secret_carrier(item) for item in value)
    return False


def _upstream_port() -> int:
    raw = str(os.getenv(UPSTREAM_PORT_ENV) or "").strip()
    if not re.fullmatch(r"[0-9]{4,5}", raw):
        raise RuntimeError("company_policy_proxy_upstream_port_invalid")
    port = int(raw)
    if not 1024 <= port <= 65535:
        raise RuntimeError("company_policy_proxy_upstream_port_invalid")
    return port


def forward_action_json(
    *,
    payload: dict[str, Any],
    upstream_port: int,
    timeout_ms: int,
    max_response_bytes: int,
) -> dict[str, Any]:
    """Forward one validated JSON object to the fixed loopback action route."""

    if _contains_secret_carrier(payload):
        raise ValueError("company_policy_proxy_secret_carrier_forbidden")
    encoded = json.dumps(payload, separators=(",", ":"), allow_nan=False).encode("utf-8")
    request = urllib.request.Request(
        f"http://127.0.0.1:{upstream_port}{ACTION_ROUTE}",
        data=encoded,
        method="POST",
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
    opener = urllib.request.build_opener(_NoRedirect)
    try:
        with opener.open(request, timeout=timeout_ms / 1000.0) as response:  # nosec B310
            content_type = str(response.headers.get("Content-Type") or "").lower()
            if "application/json" not in content_type:
                raise ValueError("company_policy_proxy_response_content_type_invalid")
            body = response.read(max_response_bytes + 1)
    except urllib.error.HTTPError as exc:
        if 300 <= exc.code < 400:
            raise ValueError("company_policy_proxy_redirect_refused") from exc
        raise ValueError(f"company_policy_proxy_upstream_http_error:{exc.code}") from exc
    except urllib.error.URLError as exc:
        raise ValueError("company_policy_proxy_upstream_unreachable") from exc
    if len(body) > max_response_bytes:
        raise ValueError("company_policy_proxy_response_too_large")
    try:
        result = json.loads(body)
    except json.JSONDecodeError as exc:
        raise ValueError("company_policy_proxy_response_not_json") from exc
    if not isinstance(result, dict):
        raise ValueError("company_policy_proxy_response_not_object")
    return result


def create_app() -> FastAPI:
    app = FastAPI(
        title="Blueprint Company Policy Proxy",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "ok": True,
            "schema_version": "company_policy_proxy.v1",
            "external_network_required": False,
            "raw_logging_enabled": False,
        }

    @app.post(ACTION_ROUTE)
    async def actions(request: Request) -> JSONResponse:
        body = await request.body()
        max_request = _bounded_int_env(
            MAX_REQUEST_BYTES_ENV,
            DEFAULT_MAX_REQUEST_BYTES,
            minimum=1_024,
            maximum=64 * 1024 * 1024,
        )
        if len(body) > max_request:
            raise HTTPException(status_code=413, detail="company_policy_proxy_request_too_large")
        try:
            payload = json.loads(body)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail="company_policy_proxy_request_not_json") from exc
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="company_policy_proxy_request_not_object")
        try:
            result = forward_action_json(
                payload=payload,
                upstream_port=_upstream_port(),
                timeout_ms=_bounded_int_env(
                    REQUEST_TIMEOUT_MS_ENV,
                    DEFAULT_REQUEST_TIMEOUT_MS,
                    minimum=1,
                    maximum=120_000,
                ),
                max_response_bytes=_bounded_int_env(
                    MAX_RESPONSE_BYTES_ENV,
                    DEFAULT_MAX_RESPONSE_BYTES,
                    minimum=1_024,
                    maximum=64 * 1024 * 1024,
                ),
            )
        except (RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        return JSONResponse(content=result, headers={"Cache-Control": "no-store"})

    return app


app = create_app()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unix-socket", required=True)
    parser.add_argument("--upstream", required=True)
    parser.add_argument("--route", required=True)
    args = parser.parse_args()
    if args.route != ACTION_ROUTE:
        raise SystemExit("company_policy_proxy_action_route_invalid")
    match = re.fullmatch(r"http://127\.0\.0\.1:([0-9]{4,5})", args.upstream)
    if not match or not 1024 <= int(match.group(1)) <= 65535:
        raise SystemExit("company_policy_proxy_upstream_invalid")
    os.environ[UPSTREAM_PORT_ENV] = match.group(1)
    import uvicorn

    uvicorn.run(app, uds=args.unix_socket, access_log=False, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
