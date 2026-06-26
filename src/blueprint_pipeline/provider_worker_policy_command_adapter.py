"""Policy command adapter for an already-running provider worker.

This adapter lets the existing policy-command loop call a provider-neutral HTTP
worker without allocating a provider instance for each action. The provider
adapter owns worker allocation, endpoint discovery, and teardown; this command
only checks readiness and posts one inference request.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Mapping, Sequence

from .provider_worker_contract import INFER_PATH, READYZ_PATH


SCHEMA_VERSION = "provider_worker_policy_command_adapter.v1"
WORKER_URL_ENVS = (
    "BLUEPRINT_PROVIDER_POLICY_WORKER_URL",
    "BLUEPRINT_POLICY_WORKER_URL",
    "TEAM_POLICY_WORKER_URL",
    "WAM_POLICY_WORKER_URL",
    "VLA_POLICY_WORKER_URL",
)
READY_URL_ENVS = (
    "BLUEPRINT_PROVIDER_POLICY_WORKER_READY_URL",
    "BLUEPRINT_POLICY_WORKER_READY_URL",
    "TEAM_POLICY_WORKER_READY_URL",
    "WAM_POLICY_WORKER_READY_URL",
    "VLA_POLICY_WORKER_READY_URL",
)
AUTH_TOKEN_FILE_ENVS = (
    "BLUEPRINT_PROVIDER_POLICY_WORKER_AUTH_TOKEN_FILE",
    "BLUEPRINT_POLICY_WORKER_AUTH_TOKEN_FILE",
    "TEAM_POLICY_AUTH_TOKEN_FILE",
    "WAM_POLICY_AUTH_TOKEN_FILE",
    "VLA_POLICY_AUTH_TOKEN_FILE",
)
TIMEOUT_SECONDS_ENV = "BLUEPRINT_PROVIDER_POLICY_WORKER_TIMEOUT_SECONDS"
READY_TIMEOUT_SECONDS_ENV = "BLUEPRINT_PROVIDER_POLICY_WORKER_READY_TIMEOUT_SECONDS"
READY_POLL_SECONDS_ENV = "BLUEPRINT_PROVIDER_POLICY_WORKER_READY_POLL_SECONDS"


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _string(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _env_first(names: Sequence[str]) -> str:
    for name in names:
        value = os.getenv(name, "").strip()
        if value:
            return value
    return ""


def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return float(default)
    try:
        value = float(raw)
    except ValueError:
        return float(default)
    return value if value > 0.0 else float(default)


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


def _redact_url(url: str) -> str:
    text = _string(url)
    if not text:
        return ""
    try:
        parsed = urllib.parse.urlsplit(text)
    except ValueError:
        return "<invalid_url>"
    netloc = parsed.hostname or ""
    if parsed.port:
        netloc = f"{netloc}:{parsed.port}"
    return urllib.parse.urlunsplit(
        (parsed.scheme, netloc, parsed.path or "/", "", "")
    )


def _read_payload() -> dict[str, Any]:
    input_path = os.getenv("BLUEPRINT_POLICY_ACTION_INPUT", "").strip()
    if input_path:
        value = json.loads(Path(input_path).expanduser().read_text(encoding="utf-8"))
    else:
        raw = sys.stdin.read().strip()
        value = json.loads(raw) if raw else {}
    if not isinstance(value, Mapping):
        raise ValueError("provider worker policy input must be a JSON object")
    return dict(value)


def _write_payload(payload: Mapping[str, Any]) -> None:
    encoded = json.dumps(dict(payload), sort_keys=True)
    output_path = os.getenv("BLUEPRINT_POLICY_ACTION_OUTPUT", "").strip()
    if output_path:
        path = Path(output_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


def _read_token(token_file: str | None) -> str | None:
    if not token_file:
        return None
    path = Path(token_file).expanduser()
    if not path.is_file():
        return None
    value = path.read_text(encoding="utf-8").strip()
    return value or None


def _headers(token_file: str | None) -> dict[str, str]:
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "BlueprintProviderWorkerPolicyCommandAdapter/1.0",
    }
    token = _read_token(token_file)
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _derive_ready_url(worker_url: str) -> str:
    parsed = urllib.parse.urlsplit(worker_url)
    path = parsed.path or INFER_PATH
    if path.rstrip("/").endswith(INFER_PATH):
        ready_path = path[: -len(INFER_PATH)] + READYZ_PATH
    else:
        ready_path = READYZ_PATH
    return urllib.parse.urlunsplit(
        (parsed.scheme, parsed.netloc, ready_path, "", "")
    )


def _json_request(
    *,
    url: str,
    method: str,
    payload: Mapping[str, Any] | None,
    token_file: str | None,
    timeout_seconds: float,
) -> tuple[dict[str, Any], int]:
    data = json.dumps(dict(payload or {})).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(
        url,
        data=data,
        headers=_headers(token_file),
        method=method,
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:  # noqa: S310
        text = response.read().decode("utf-8")
        status = int(getattr(response, "status", 200) or 200)
    parsed = json.loads(text or "{}")
    return _mapping(parsed), status


def _action_from_worker_response(response: Mapping[str, Any]) -> dict[str, Any] | None:
    action = (
        response.get("action")
        or response.get("normalized_action")
        or response.get("policy_action")
        or response.get("decision")
    )
    if isinstance(action, Mapping):
        return dict(action)
    for key in ("action_chunk", "actions", "action_vector", "joint_targets", "joint_positions"):
        value = response.get(key)
        if isinstance(value, Mapping):
            return {
                "action_type": "manipulation_contact",
                "unitree_action_chunk_present": True,
                "unitree_raw_action_key": key,
                key: dict(value),
            }
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return {
                "action_type": "manipulation_contact",
                "unitree_action_chunk_present": True,
                "unitree_raw_action_key": key,
                key: list(value),
            }
    return None


def _blocked_payload(
    *,
    blockers: Sequence[str],
    worker_url: str | None = None,
    ready_url: str | None = None,
    ready_response: Mapping[str, Any] | None = None,
    error_type: str | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "blocked",
        "policy_id": "provider_worker_policy",
        "provider_worker_policy_command_ran": False,
        "provider_worker_readyz_checked": bool(ready_url),
        "provider_worker_infer_completed": False,
        "model_ran": False,
        "worker_url_redacted": _redact_url(worker_url or ""),
        "ready_url_redacted": _redact_url(ready_url or ""),
        "ready_response_redacted": _redact(ready_response or {}),
        "blockers": sorted(set(str(item) for item in blockers if str(item))),
        "error_type": error_type,
        "error": error[:800] if error else None,
        "raw_credentials_written_to_artifacts": False,
        "secret_hashes_written_to_artifacts": False,
        "claim_boundary": {
            "provider_worker_command_adapter_does_not_allocate_provider": True,
            "provider_worker_command_adapter_does_not_prove_provider_teardown": True,
            "readyz_required_before_infer": True,
            "generated_world_rank_fidelity_result_proven": False,
            "generated_world_policy_evaluation_scope_proven": False,
            "non_ranking_operational_claim_proven": False,
        },
    }


def _ready_response_is_ready(payload: Mapping[str, Any]) -> bool:
    if payload.get("ready_for_inference") is True:
        return True
    if payload.get("model_ready") is True and _string(payload.get("status")) in {
        "",
        "ready",
        "completed",
        "ok",
    }:
        return True
    return False


def wait_for_ready(
    *,
    ready_url: str,
    token_file: str | None,
    timeout_seconds: float,
    poll_seconds: float,
) -> tuple[dict[str, Any], bool, str | None]:
    deadline = time.monotonic() + max(0.001, timeout_seconds)
    last_payload: dict[str, Any] = {}
    last_error: str | None = None
    while True:
        try:
            payload, _ = _json_request(
                url=ready_url,
                method="GET",
                payload=None,
                token_file=token_file,
                timeout_seconds=max(0.001, min(5.0, timeout_seconds)),
            )
            last_payload = payload
            if _ready_response_is_ready(payload):
                return payload, True, None
            last_error = _string(payload.get("status")) or "worker_not_ready"
        except Exception as exc:
            last_error = f"{type(exc).__name__}:{str(exc)[:300]}"
        if time.monotonic() >= deadline:
            return last_payload, False, last_error
        time.sleep(max(0.05, min(poll_seconds, 2.0)))


def run_provider_worker_policy_command(
    *,
    payload: Mapping[str, Any],
    worker_url: str | None = None,
    ready_url: str | None = None,
    auth_token_file: str | None = None,
    timeout_seconds: float | None = None,
    ready_timeout_seconds: float | None = None,
    ready_poll_seconds: float | None = None,
) -> tuple[dict[str, Any], int]:
    worker = _string(worker_url) or _env_first(WORKER_URL_ENVS)
    if not worker:
        return _blocked_payload(blockers=["missing_provider_policy_worker_url"]), 2
    ready = _string(ready_url) or _env_first(READY_URL_ENVS) or _derive_ready_url(worker)
    token_file = _string(auth_token_file) or _env_first(AUTH_TOKEN_FILE_ENVS)
    timeout = float(timeout_seconds) if timeout_seconds is not None else _float_env(
        TIMEOUT_SECONDS_ENV,
        30.0,
    )
    ready_timeout = (
        float(ready_timeout_seconds)
        if ready_timeout_seconds is not None
        else _float_env(READY_TIMEOUT_SECONDS_ENV, 30.0)
    )
    ready_poll = (
        float(ready_poll_seconds)
        if ready_poll_seconds is not None
        else _float_env(READY_POLL_SECONDS_ENV, 0.25)
    )
    ready_payload, ready_ok, ready_error = wait_for_ready(
        ready_url=ready,
        token_file=token_file,
        timeout_seconds=ready_timeout,
        poll_seconds=ready_poll,
    )
    if not ready_ok:
        return (
            _blocked_payload(
                blockers=["provider_policy_worker_not_ready"],
                worker_url=worker,
                ready_url=ready,
                ready_response=ready_payload,
                error=ready_error,
            ),
            2,
        )
    observation = _mapping(_mapping(payload).get("observation"))
    if not observation:
        return (
            _blocked_payload(
                blockers=["missing_policy_observation"],
                worker_url=worker,
                ready_url=ready,
                ready_response=ready_payload,
            ),
            2,
        )
    try:
        response, status = _json_request(
            url=worker,
            method="POST",
            payload={"observation": observation},
            token_file=token_file,
            timeout_seconds=timeout,
        )
    except Exception as exc:
        return (
            _blocked_payload(
                blockers=["provider_policy_worker_infer_failed"],
                worker_url=worker,
                ready_url=ready,
                ready_response=ready_payload,
                error_type=type(exc).__name__,
                error=str(exc),
            ),
            1,
        )
    action = _action_from_worker_response(response)
    if action is None:
        return (
            _blocked_payload(
                blockers=["provider_policy_worker_response_missing_action"],
                worker_url=worker,
                ready_url=ready,
                ready_response=ready_payload,
            ),
            1,
        )
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed",
        "policy_id": str(response.get("policy_id") or "provider_worker_policy"),
        "action": action,
        "provider_worker_policy_command_ran": True,
        "provider_worker_readyz_checked": True,
        "provider_worker_readyz_response_redacted": _redact(ready_payload),
        "provider_worker_infer_completed": True,
        "provider_worker_http_status": status,
        "provider_worker_url_redacted": _redact_url(worker),
        "provider_worker_ready_url_redacted": _redact_url(ready),
        "worker_response_redacted": _redact(response),
        "model_ran": bool(response.get("model_ran", True)),
        "raw_credentials_written_to_artifacts": False,
        "secret_hashes_written_to_artifacts": False,
        "claim_boundary": {
            "provider_worker_command_adapter_does_not_allocate_provider": True,
            "provider_worker_command_adapter_does_not_prove_provider_teardown": True,
            "provider_worker_response_is_not_generated_world_rank_fidelity": True,
            "generated_world_rank_fidelity_result_proven": False,
            "generated_world_policy_evaluation_scope_proven": False,
            "non_ranking_operational_claim_proven": False,
        },
    }
    for key in (
        "unitree_policy_action_command_ran",
        "unitree_lerobot_policy_action_command_ran",
        "unitree_unifolm_policy_action_command_ran",
        "unitree_groot_n17_sonic_policy_action_command_ran",
        "unitree_manipulation_policy_action_command_ran",
        "provider_output_replay_used",
        "fresh_policy_action_model_executed_this_invocation",
    ):
        if key in response:
            result[key] = bool(response[key])
    return result, 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-url")
    parser.add_argument("--ready-url")
    parser.add_argument("--auth-token-file")
    parser.add_argument("--timeout-seconds", type=float)
    parser.add_argument("--ready-timeout-seconds", type=float)
    parser.add_argument("--ready-poll-seconds", type=float)
    args = parser.parse_args(argv)
    payload = _read_payload()
    response, exit_code = run_provider_worker_policy_command(
        payload=payload,
        worker_url=args.worker_url,
        ready_url=args.ready_url,
        auth_token_file=args.auth_token_file,
        timeout_seconds=args.timeout_seconds,
        ready_timeout_seconds=args.ready_timeout_seconds,
        ready_poll_seconds=args.ready_poll_seconds,
    )
    _write_payload(response)
    return exit_code


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
