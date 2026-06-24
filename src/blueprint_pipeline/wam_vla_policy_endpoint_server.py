"""Local HTTP wrapper for a WAM/VLA policy command."""

from __future__ import annotations

import argparse
import hmac
import json
import os
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from fastapi import FastAPI, Header, HTTPException, Request

from .provider_worker_contract import (
    HEALTHZ_PATH,
    INFER_PATH,
    LEGACY_HEALTH_PATH,
    LEGACY_POLICY_ACTION_PATH,
    PROVIDER_WORKER_CONTRACT_SCHEMA_VERSION,
    READYZ_PATH,
    SHUTDOWN_PATH,
    classify_policy_worker_command,
)


COMMAND_ENV = "BLUEPRINT_WAM_VLA_POLICY_COMMAND"
AUTH_TOKEN_FILE_ENV = "BLUEPRINT_WAM_VLA_POLICY_AUTH_TOKEN_FILE"
BUILTIN_REFERENCE_ADAPTER_COMMAND = "builtin:g1_endpoint_reference_adapter"
BUILTIN_REFERENCE_ADAPTER_ALIASES = frozenset(
    {
        BUILTIN_REFERENCE_ADAPTER_COMMAND,
        "builtin:blueprint_g1_endpoint_reference_adapter",
        "builtin:reference_g1_policy",
    }
)


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _read_token(path: str | None) -> str | None:
    if not path:
        return None
    token_path = Path(path).expanduser()
    if not token_path.is_file():
        return None
    return token_path.read_text(encoding="utf-8").strip()


def _redact(value: Any) -> Any:
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, child in value.items():
            key_text = str(key)
            if any(marker in key_text.lower() for marker in ("token", "secret", "password", "key")):
                result[key_text] = "<redacted>"
            else:
                result[key_text] = _redact(child)
        return result
    if isinstance(value, list):
        return [_redact(item) for item in value]
    return value


def _check_auth(*, authorization: str | None, token_file: str | None) -> None:
    token = _read_token(token_file)
    if not token:
        return
    prefix = "Bearer "
    supplied = authorization or ""
    if not supplied.startswith(prefix):
        raise HTTPException(status_code=401, detail="missing_bearer_token")
    supplied_token = supplied[len(prefix) :].strip()
    if not hmac.compare_digest(supplied_token, token):
        raise HTTPException(status_code=403, detail="invalid_bearer_token")


def _policy_adapter_invocation_mode(command: str) -> str:
    if command.strip() in BUILTIN_REFERENCE_ADAPTER_ALIASES:
        return "in_process_builtin"
    if command.strip():
        return "subprocess"
    return "missing"


def _run_builtin_reference_adapter(
    *, command: str, payload: Mapping[str, Any], started: float
) -> tuple[dict[str, Any], dict[str, Any]]:
    from .g1_endpoint_reference_adapter import build_response

    response = build_response(payload)
    encoded = json.dumps(response, sort_keys=True)
    return response, {
        "command_exit_code": 0,
        "duration_seconds": round(time.monotonic() - started, 6),
        "stderr_size_bytes": 0,
        "stderr_omitted_to_avoid_secret_leakage": False,
        "stdout_size_bytes": len(encoded),
        "policy_adapter_invocation_mode": "in_process_builtin",
        "subprocess_spawned": False,
        "policy_command_alias": command.strip(),
    }


def run_policy_command(
    *,
    command: str,
    payload: Mapping[str, Any],
    timeout_seconds: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    command = command.strip()
    if not command:
        raise RuntimeError("missing_policy_command")
    started = time.monotonic()
    if _policy_adapter_invocation_mode(command) == "in_process_builtin":
        return _run_builtin_reference_adapter(command=command, payload=payload, started=started)
    result = subprocess.run(
        shlex.split(command),
        input=json.dumps(dict(payload)),
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout_seconds,
    )
    meta = {
        "command_exit_code": result.returncode,
        "duration_seconds": round(time.monotonic() - started, 6),
        "stderr_size_bytes": len(result.stderr or ""),
        "stderr_omitted_to_avoid_secret_leakage": bool(result.stderr),
        "stdout_size_bytes": len(result.stdout or ""),
        "policy_adapter_invocation_mode": "subprocess",
        "subprocess_spawned": True,
    }
    if result.returncode != 0:
        raise RuntimeError(f"policy_command_failed:{json.dumps(meta, sort_keys=True)}")
    response = json.loads(result.stdout or "{}")
    if not isinstance(response, Mapping):
        raise RuntimeError("policy_command_stdout_not_json_object")
    return dict(response), meta


def _worker_health_payload(
    *,
    command: str,
    token_file: str | None,
    shutdown_requested: bool,
) -> dict[str, Any]:
    invocation_mode = _policy_adapter_invocation_mode(command)
    command_classification = classify_policy_worker_command(command)
    status = (
        "blocked_shutdown_requested"
        if shutdown_requested
        else ("ready" if command else "blocked_missing_policy_command")
    )
    return {
        "schema_version": "wam_vla_policy_endpoint_worker_health.v1",
        "provider_worker_contract_schema_version": PROVIDER_WORKER_CONTRACT_SCHEMA_VERSION,
        "status": status,
        "policy_command_configured": bool(command),
        "policy_adapter_invocation_mode": invocation_mode,
        "subprocess_spawned_per_request": invocation_mode == "subprocess",
        "auth_token_file_configured": bool(token_file),
        "shutdown_requested": shutdown_requested,
        "canonical_http_contract": {
            "healthz": HEALTHZ_PATH,
            "readyz": READYZ_PATH,
            "infer": INFER_PATH,
            "shutdown": SHUTDOWN_PATH,
        },
        "legacy_http_contract": {
            "health": LEGACY_HEALTH_PATH,
            "policy_action": LEGACY_POLICY_ACTION_PATH,
        },
        "policy_worker_contract": command_classification,
        "raw_token_values_returned": False,
    }


def create_app(
    *,
    policy_command: str | None = None,
    auth_token_file: str | None = None,
    timeout_seconds: float = 8.0,
) -> FastAPI:
    command = policy_command if policy_command is not None else os.getenv(COMMAND_ENV, "")
    token_file = auth_token_file if auth_token_file is not None else (
        os.getenv(AUTH_TOKEN_FILE_ENV)
        or os.getenv("TEAM_POLICY_AUTH_TOKEN_FILE")
        or os.getenv("WAM_POLICY_AUTH_TOKEN_FILE")
        or os.getenv("VLA_POLICY_AUTH_TOKEN_FILE")
    )
    app = FastAPI(title="Blueprint WAM/VLA Policy Endpoint")
    state: dict[str, Any] = {"shutdown_requested": False}

    @app.get(LEGACY_HEALTH_PATH)
    async def health() -> dict[str, Any]:
        return _worker_health_payload(
            command=command,
            token_file=token_file,
            shutdown_requested=bool(state.get("shutdown_requested")),
        )

    @app.get(HEALTHZ_PATH)
    async def healthz() -> dict[str, Any]:
        return _worker_health_payload(
            command=command,
            token_file=token_file,
            shutdown_requested=bool(state.get("shutdown_requested")),
        )

    @app.get(READYZ_PATH)
    async def readyz() -> dict[str, Any]:
        health_payload = _worker_health_payload(
            command=command,
            token_file=token_file,
            shutdown_requested=bool(state.get("shutdown_requested")),
        )
        return {
            **health_payload,
            "schema_version": "wam_vla_policy_endpoint_worker_ready.v1",
            "model_ready": bool(command) and not bool(state.get("shutdown_requested")),
            "ready_for_inference": bool(command) and not bool(state.get("shutdown_requested")),
        }

    async def _handle_policy_action_request(
        request: Request,
        authorization: str | None,
    ) -> dict[str, Any]:
        _check_auth(authorization=authorization, token_file=token_file)
        if state.get("shutdown_requested"):
            raise HTTPException(status_code=503, detail="shutdown_requested")
        request_payload = await request.json()
        observation = _mapping(_mapping(request_payload).get("observation"))
        if not observation:
            raise HTTPException(status_code=422, detail="missing_observation")
        if not command:
            raise HTTPException(status_code=503, detail="missing_policy_command")
        try:
            response, meta = run_policy_command(
                command=command,
                payload={"observation": observation},
                timeout_seconds=timeout_seconds,
            )
        except Exception as exc:
            raise HTTPException(status_code=502, detail=str(exc)[:800]) from exc
        action = response.get("action") or response.get("policy_action") or response.get("decision")
        if not isinstance(action, Mapping):
            raise HTTPException(status_code=502, detail="policy_response_missing_action")
        return {
            "policy_id": str(response.get("policy_id") or "local_wam_vla_policy_command"),
            "action": dict(action),
            "endpoint_metadata": {
                **meta,
                "raw_response_redacted": _redact(response),
                "raw_token_values_returned": False,
                "canonical_infer_path": INFER_PATH,
                "legacy_policy_action_path": LEGACY_POLICY_ACTION_PATH,
            },
        }

    @app.post(LEGACY_POLICY_ACTION_PATH)
    async def policy_action(
        request: Request,
        authorization: str | None = Header(default=None),
    ) -> dict[str, Any]:
        return await _handle_policy_action_request(
            request=request,
            authorization=authorization,
        )

    @app.post(INFER_PATH)
    async def infer(
        request: Request,
        authorization: str | None = Header(default=None),
    ) -> dict[str, Any]:
        return await _handle_policy_action_request(
            request=request,
            authorization=authorization,
        )

    @app.post(SHUTDOWN_PATH)
    async def shutdown(authorization: str | None = Header(default=None)) -> dict[str, Any]:
        _check_auth(authorization=authorization, token_file=token_file)
        state["shutdown_requested"] = True
        return {
            "schema_version": "wam_vla_policy_endpoint_worker_shutdown.v1",
            "status": "shutdown_requested",
            "shutdown_acknowledged": True,
            "process_shutdown_performed": False,
            "provider_adapter_must_record_teardown": True,
            "raw_token_values_returned": False,
            "claim_boundary": {
                "shutdown_response_is_not_provider_cost_or_teardown_proof": True,
                "provider_adapter_teardown_artifact_required": True,
            },
        }

    return app


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--policy-command")
    parser.add_argument("--auth-token-file")
    parser.add_argument("--timeout-seconds", type=float, default=8.0)
    args = parser.parse_args(argv)
    import uvicorn

    app = create_app(
        policy_command=args.policy_command,
        auth_token_file=args.auth_token_file,
        timeout_seconds=args.timeout_seconds,
    )
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
