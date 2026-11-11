"""Start a session-served policy's contract server on the rented GPU, prove it ready, or fail the attempt.

Plan 2.1. Runs in the policy's own interpreter after the native provisioning
(the same venv, source and verified checkpoint a native candidate gets). It
starts the contract server named by the sealed spec, waits a bounded time for a
readiness handshake made by the episode's own client (an identity read, never
an inference), and writes the same ready-server receipt the native server
worker writes, so the entrypoint's existing teardown stops it by its exact
process identity at the end of the session.

A server that does not start, exits, or never answers within its bound fails
this step: the entrypoint records the candidate's provisioning stage
(``policy_canary_policy_provisioning_failed``), an infrastructure failure of the
attempt that scores no episode. A server that dies while serving an inference
fails that try as the policy's, as the same crash does on the native path; each
later request finds nothing listening (``session_policy_server_unavailable``) and
is void as infrastructure (``session_policy_client``).

The token goes only to the process this step started. The server binds its
port before it loads the checkpoint and records its own pid in its identity
file; no readiness request (each carries the token) is sent until that record
names the started process, which is still alive and so still holds the port.
Another local process that took the port first makes the server's bind fail,
and the step fails before any token is sent.

The bearer token is generated on the session by the entrypoint into a private
file outside the output directory; it is never in the bundle, the receipt or
any log.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from collections.abc import Callable, Mapping
from typing import Any

RECEIPT_SCHEMA = "adp009d_policy_server_receipt.v1"
POLL_SECONDS = 2.0
TERMINATE_SECONDS = 10.0
_STRIPPED_ENV = frozenset({"HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN", "PYTHONHOME"})


def _token(environment: Mapping[str, str]) -> str:
    from .openpi_droid_contract_server import read_token
    from .session_served_policy import TOKEN_ENV

    path = environment.get(TOKEN_ENV)
    if not path:
        raise ValueError("session_policy_token_unavailable")
    return read_token(path)


def listener_is_started_process(identity_out: Path, *, pid: int, port: int) -> bool:
    """The server's own record says the started process bound the port (it writes it only after binding)."""
    try:
        record = json.loads(identity_out.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return (isinstance(record, dict) and record.get("pid") == pid and record.get("port") == port
            and record.get("host") == "127.0.0.1" and record.get("status") in {"loading", "serving"})


def server_command(*, python: str, spec_path: Path, checkpoint_dir: str, inventory: str, token_file: str,
                   port: int, identity_out: Path, module: str) -> list[str]:
    return [python, "-m", module, "--execution-spec", str(spec_path), "--checkpoint-dir", checkpoint_dir,
            "--checkpoint-inventory", inventory, "--token-file", token_file, "--port", str(port),
            "--identity-out", str(identity_out)]


def _stop(process: subprocess.Popen | None) -> bool:
    if process is None or process.poll() is not None:
        return True
    process.terminate()
    try:
        process.wait(timeout=TERMINATE_SECONDS)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=TERMINATE_SECONDS)
    return process.poll() is not None


def start(*, spec: Mapping[str, Any], spec_path: Path, python: str, runtime: Path, output: Path, checkpoint_dir: str,
          inventory: str, env: Mapping[str, str] | None = None, popen: Callable[..., Any] = subprocess.Popen,
          client_factory: Callable[..., Any] | None = None, monotonic: Callable[[], float] = time.monotonic,
          sleep: Callable[[float], None] = time.sleep) -> dict[str, Any]:
    """Start the server, wait for a handshake within the spec's bound, and retain the receipt."""

    from .session_policy_client import session_policy_client
    from .session_served_policy import TOKEN_ENV, validate_session_served_spec

    spec = validate_session_served_spec(spec)
    block = spec["session_policy_server"]
    environment = dict(os.environ if env is None else env)
    token = _token(environment)
    candidate = spec["candidate_id"]
    output.mkdir(parents=True, exist_ok=True)
    identity_out = output / f"session_policy_server_identity.{candidate}.json"
    receipt_path = output / f"adp009d_policy_server_receipt.{candidate}.json"
    command = server_command(python=python, spec_path=spec_path, checkpoint_dir=checkpoint_dir, inventory=inventory,
                             token_file=environment[TOKEN_ENV], port=block["port"], identity_out=identity_out,
                             module=block["module"])
    child_env = {key: value for key, value in environment.items() if key not in _STRIPPED_ENV}
    child_env.update(PYTHONPATH=str(runtime), PYTHONDONTWRITEBYTECODE="1")
    receipt: dict[str, Any] = {"schema_version": RECEIPT_SCHEMA, "candidate_id": candidate, "status": "starting",
        "transport": "openpi_droid_https_contract_loopback", "host": "127.0.0.1", "port": block["port"],
        "command": command, "server": block["server"], "native_candidate_id": block["native_candidate_id"],
        "candidate_policy_queried": False, "candidate_inference_performed": False, "outbound_calls": "none"}
    process = None
    factory = client_factory or session_policy_client
    started = monotonic()
    try:
        # An identity record left by an earlier server on this output must never pass the readiness check:
        # only the record the process started here writes may say it bound the port.
        identity_out.unlink(missing_ok=True)
        with (output / f"adp009d_policy_server.{candidate}.log").open("wb") as log:
            process = popen(command, stdout=log, stderr=subprocess.STDOUT, env=child_env)
        receipt["server_pid"] = process.pid
        receipt_path.write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")
        attempts, last = 0, None
        while True:
            if process.poll() is not None:
                raise RuntimeError(f"session_policy_server_exited_before_ready:{process.returncode}")
            if monotonic() - started > block["startup_timeout_seconds"]:
                raise RuntimeError(f"session_policy_server_not_ready_within_bound:{last or 'no_answer'}")
            if not listener_is_started_process(identity_out, pid=process.pid, port=block["port"]):
                last = "session_policy_server_not_bound"  # no token is sent to a listener not proven ours
                sleep(POLL_SECONDS)
                continue
            attempts += 1
            try:
                readiness = factory(spec, token).preflight_readiness()
                break
            except ValueError as exc:  # Not listening yet, or still loading the checkpoint.
                last = str(exc).split(":", 1)[0]
                sleep(POLL_SECONDS)
        identity = json.loads(identity_out.read_text(encoding="utf-8")) if identity_out.exists() else None
        receipt.update(status="ready", handshake_completed=True, readiness_attempts=attempts,
                       readiness_seconds=round(monotonic() - started, 3), server_metadata=readiness["server_metadata"],
                       server_identity=identity)
        receipt_path.write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")
        return receipt
    except BaseException as exc:
        stopped = _stop(process)
        receipt.update(status="blocked", handshake_completed=False, blocker=str(exc).split(":", 1)[0][:128],
                       server_exit_code=None if process is None else process.poll(), process_terminated=stopped)
        receipt_path.write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", required=True)
    parser.add_argument("--runtime", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--execution-spec", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--checkpoint-inventory", required=True)
    args = parser.parse_args(argv)
    spec_path = Path(args.execution_spec)
    try:
        receipt = start(spec=json.loads(spec_path.read_text(encoding="utf-8")), spec_path=spec_path,
                        python=args.python, runtime=Path(args.runtime), output=Path(args.output),
                        checkpoint_dir=args.checkpoint_dir, inventory=args.checkpoint_inventory)
    except Exception as exc:  # noqa: BLE001 - the receipt holds the typed reason
        print("BLUEPRINT_SESSION_POLICY_SERVER:blocked:" + str(exc).split(":", 1)[0][:128], file=sys.stderr)
        return 2
    print("BLUEPRINT_SESSION_POLICY_SERVER:" + receipt["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["listener_is_started_process", "server_command", "start"]
