"""ADP-011/day 7: serve an approved model inside the isolated policy worker.

The trusted builder supplies fixed files. HTTP callers can supply observations
only; they cannot select model files, import Python, or invoke other routes.
Run behind the company-policy Unix proxy in a network-isolated runsc sandbox.
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from .policy_model_onnx import OnnxStatePolicy

MAX_REQUEST_BYTES = 8 * 1024 * 1024
WIRE_KEYS = frozenset({"schema_version", "request_id", "synthetic", "prompt", "cameras", "state"})


def create_model_policy_app(policy: OnnxStatePolicy) -> FastAPI:
    app = FastAPI(openapi_url=None, docs_url=None, redoc_url=None)

    @app.post("/v1/actions")
    async def actions(request: Request) -> JSONResponse:
        if request.headers.get("content-type", "").split(";")[0].strip() != "application/json":
            raise HTTPException(415, "policy_model_json_required")
        body = bytearray()
        async for chunk in request.stream():
            body.extend(chunk)
            if len(body) > MAX_REQUEST_BYTES:
                raise HTTPException(413, "policy_model_observation_too_large")
        try:
            def reject_constant(_value: str) -> None:
                raise ValueError("nonfinite_json")

            wire = json.loads(body, parse_constant=reject_constant)
            if (not isinstance(wire, dict) or set(wire) != WIRE_KEYS
                    or wire.get("schema_version") != "blueprint_company_policy_observation.v1"
                    or not isinstance(wire.get("state"), dict)
                    or set(wire["state"]) != {field["name"] for field in policy.fields}):
                raise ValueError("observation_interface_invalid")
            response = policy.infer(wire)
            if len(json.dumps(response, allow_nan=False).encode()) > 65_536:
                raise ValueError("response_too_large")
        except (ValueError, TypeError, RecursionError, KeyError):
            # Never return untrusted graph metadata, state values or raw errors.
            raise HTTPException(400, "policy_model_observation_or_action_invalid") from None
        return JSONResponse(response)

    return app


def main() -> None:
    import uvicorn

    # Fixed image paths, supplied at build time; no caller-controlled filename.
    artifact = json.loads(Path("/opt/policy-model/artifact.json").read_text())
    policy = OnnxStatePolicy(model_path=Path("/opt/policy-model/policy.onnx"), artifact=artifact)
    uvicorn.run(create_model_policy_app(policy), host="127.0.0.1", port=8600,
                access_log=False, log_level="critical")


if __name__ == "__main__":
    main()
