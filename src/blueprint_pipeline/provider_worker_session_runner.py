"""Run a batch of policy inferences against one provider worker session."""

from __future__ import annotations

import argparse
import json
import urllib.parse
from pathlib import Path
from typing import Any, Mapping, Sequence

from .common import ensure_dir, utc_now_iso, write_json
from .provider_worker_contract import INFER_PATH, SHUTDOWN_PATH
from .provider_worker_policy_command_adapter import (
    _action_from_worker_response,
    _derive_ready_url,
    _env_first,
    _json_request,
    _mapping,
    _redact,
    _redact_url,
    _string,
    AUTH_TOKEN_FILE_ENVS,
    READY_URL_ENVS,
    WORKER_URL_ENVS,
    wait_for_ready,
)


SCHEMA_VERSION = "provider_worker_session_run.v1"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.expanduser().open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise ValueError("provider worker session input rows must be JSON objects")
            rows.append(dict(value))
    return rows


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")


def _derive_shutdown_url(worker_url: str) -> str:
    parsed = urllib.parse.urlsplit(worker_url)
    path = parsed.path or INFER_PATH
    if path.rstrip("/").endswith(INFER_PATH):
        shutdown_path = path[: -len(INFER_PATH)] + SHUTDOWN_PATH
    else:
        shutdown_path = SHUTDOWN_PATH
    return urllib.parse.urlunsplit(
        (parsed.scheme, parsed.netloc, shutdown_path, "", "")
    )


def _observation_from_row(row: Mapping[str, Any]) -> dict[str, Any]:
    observation = _mapping(row.get("observation"))
    return observation or dict(row)


def run_provider_worker_session(
    *,
    observations: Sequence[Mapping[str, Any]],
    output_dir: str | Path,
    worker_url: str | None = None,
    ready_url: str | None = None,
    shutdown_url: str | None = None,
    auth_token_file: str | None = None,
    timeout_seconds: float = 30.0,
    ready_timeout_seconds: float = 30.0,
    ready_poll_seconds: float = 0.25,
    request_shutdown: bool = False,
    stop_on_error: bool = True,
    generated_at: str | None = None,
) -> dict[str, Any]:
    generated = generated_at or utc_now_iso()
    output = Path(output_dir).expanduser()
    ensure_dir(output)
    worker = _string(worker_url) or _env_first(WORKER_URL_ENVS)
    if not worker:
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "generated_at": generated,
            "status": "blocked",
            "blockers": ["missing_provider_policy_worker_url"],
            "provider_worker_session_reused": False,
            "raw_credentials_written_to_artifacts": False,
            "secret_hashes_written_to_artifacts": False,
        }
        write_json(output / "provider_worker_session_run.json", manifest)
        return manifest
    ready = _string(ready_url) or _env_first(READY_URL_ENVS) or _derive_ready_url(worker)
    shutdown = _string(shutdown_url) or _derive_shutdown_url(worker)
    token_file = _string(auth_token_file) or _env_first(AUTH_TOKEN_FILE_ENVS)
    ready_payload, ready_ok, ready_error = wait_for_ready(
        ready_url=ready,
        token_file=token_file,
        timeout_seconds=ready_timeout_seconds,
        poll_seconds=ready_poll_seconds,
    )
    rows: list[dict[str, Any]] = []
    blockers: list[str] = []
    if not ready_ok:
        blockers.append("provider_policy_worker_not_ready")
    if ready_ok:
        for index, row in enumerate(observations):
            observation = _observation_from_row(row)
            try:
                response, status = _json_request(
                    url=worker,
                    method="POST",
                    payload={"observation": observation},
                    token_file=token_file,
                    timeout_seconds=timeout_seconds,
                )
                action = _action_from_worker_response(response)
                completed = action is not None
                row_blockers = [] if completed else [
                    "provider_policy_worker_response_missing_action"
                ]
                if row_blockers:
                    blockers.extend(row_blockers)
                rows.append(
                    {
                        "step_index": index,
                        "status": "completed" if completed else "blocked",
                        "provider_worker_http_status": status,
                        "policy_id": response.get("policy_id"),
                        "action": action,
                        "worker_response_redacted": _redact(response),
                        "blockers": row_blockers,
                    }
                )
                if row_blockers and stop_on_error:
                    break
            except Exception as exc:
                blocker = "provider_policy_worker_infer_failed"
                blockers.append(blocker)
                rows.append(
                    {
                        "step_index": index,
                        "status": "blocked",
                        "error_type": type(exc).__name__,
                        "error": str(exc)[:800],
                        "blockers": [blocker],
                    }
                )
                if stop_on_error:
                    break
    shutdown_response: dict[str, Any] = {}
    shutdown_acknowledged = False
    if request_shutdown and ready_ok:
        try:
            shutdown_response, _ = _json_request(
                url=shutdown,
                method="POST",
                payload={},
                token_file=token_file,
                timeout_seconds=timeout_seconds,
            )
            shutdown_acknowledged = bool(
                shutdown_response.get("shutdown_acknowledged")
                or _string(shutdown_response.get("status")) in {"shutdown_requested", "ok"}
            )
        except Exception as exc:
            blockers.append("provider_policy_worker_shutdown_request_failed")
            shutdown_response = {
                "status": "blocked",
                "error_type": type(exc).__name__,
                "error": str(exc)[:800],
            }
    completed_count = sum(1 for row in rows if row.get("status") == "completed")
    status = (
        "completed"
        if ready_ok and completed_count == len(observations) and not blockers
        else "blocked"
    )
    outputs_path = output / "provider_worker_session_outputs.jsonl"
    _write_jsonl(outputs_path, rows)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated,
        "status": status,
        "provider_worker_session_reused": bool(ready_ok and rows),
        "worker_url_redacted": _redact_url(worker),
        "ready_url_redacted": _redact_url(ready),
        "shutdown_url_redacted": _redact_url(shutdown),
        "readyz_checked_before_batch": True,
        "ready_response_redacted": _redact(ready_payload),
        "requested_infer_count": len(observations),
        "attempted_infer_count": len(rows),
        "completed_infer_count": completed_count,
        "outputs_jsonl": str(outputs_path),
        "request_shutdown": request_shutdown,
        "shutdown_acknowledged": shutdown_acknowledged,
        "shutdown_response_redacted": _redact(shutdown_response),
        "provider_shutdown_proven": False,
        "blockers": sorted(set(blockers)),
        "raw_credentials_written_to_artifacts": False,
        "secret_hashes_written_to_artifacts": False,
        "claim_boundary": {
            "session_runner_does_not_allocate_provider": True,
            "shutdown_acknowledged_is_not_provider_teardown_or_cost_proof": True,
            "provider_adapter_teardown_artifact_required": True,
            "physical_robot_readiness_proven": False,
            "deployment_readiness_proven": False,
            "safety_validation_proven": False,
        },
    }
    write_json(output / "provider_worker_session_run.json", manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--worker-url")
    parser.add_argument("--ready-url")
    parser.add_argument("--shutdown-url")
    parser.add_argument("--auth-token-file")
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument("--ready-timeout-seconds", type=float, default=30.0)
    parser.add_argument("--ready-poll-seconds", type=float, default=0.25)
    parser.add_argument("--request-shutdown", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args(argv)
    manifest = run_provider_worker_session(
        observations=_read_jsonl(Path(args.input_jsonl)),
        output_dir=args.output_dir,
        worker_url=args.worker_url,
        ready_url=args.ready_url,
        shutdown_url=args.shutdown_url,
        auth_token_file=args.auth_token_file,
        timeout_seconds=args.timeout_seconds,
        ready_timeout_seconds=args.ready_timeout_seconds,
        ready_poll_seconds=args.ready_poll_seconds,
        request_shutdown=args.request_shutdown,
        stop_on_error=not args.continue_on_error,
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0 if manifest.get("status") == "completed" else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
