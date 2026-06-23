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

    @app.get("/health")
    async def health() -> dict[str, Any]:
        invocation_mode = _policy_adapter_invocation_mode(command)
        return {
            "status": "ready" if command else "blocked_missing_policy_command",
            "policy_command_configured": bool(command),
            "policy_adapter_invocation_mode": invocation_mode,
            "subprocess_spawned_per_request": invocation_mode == "subprocess",
            "auth_token_file_configured": bool(token_file),
            "raw_token_values_returned": False,
        }

    @app.post("/policy/action")
    async def policy_action(
        request: Request,
        authorization: str | None = Header(default=None),
    ) -> dict[str, Any]:
        _check_auth(authorization=authorization, token_file=token_file)
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
