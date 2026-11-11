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
attempt that scores no episode. A server that dies after it is ready makes each
later request ``session_policy_server_unavailable``, which the episode evidence
records as infrastructure, never as the policy failing.

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


def provisioning_script(spec: Mapping[str, Any]) -> str:
    """The native candidate's provisioning, ending with this contract server in place of its own."""

    from .adp009d_policy_provisioning import (
        CHECKPOINT_INVENTORY_STAGED_NAME, CHECKPOINT_ROOT, POLICY_EXECUTION_SPEC_STAGED_NAME, build_provisioning_script,
        policy_venv_root,
    )
    from .session_served_policy import validate_session_served_spec

    spec = validate_session_served_spec(spec)
    native = spec["session_policy_server"]["native_candidate_id"]
    candidate = spec["candidate_id"]
    python = f"{policy_venv_root(native)}/bin/python"
    block = f'''echo "BLUEPRINT_WAM_RUNTIME_PHASE:adp009d:provision_{candidate}_contract_server:started"
PYTHONPATH="$RUNTIME_DIR" PYTHONDONTWRITEBYTECODE=1 "{python}" -m blueprint_pipeline.session_policy_server_start \\
  --python "{python}" --runtime "$RUNTIME_DIR" --output "$OUT_DIR" \\
  --execution-spec "$RUNTIME_DIR/{POLICY_EXECUTION_SPEC_STAGED_NAME}" \\
  --checkpoint-dir "{CHECKPOINT_ROOT}/{native}" \\
  --checkpoint-inventory "$RUNTIME_DIR/{CHECKPOINT_INVENTORY_STAGED_NAME}"
echo "BLUEPRINT_WAM_RUNTIME_PHASE:adp009d:provision_{candidate}_contract_server:completed"'''
    return build_provisioning_script(native, server_block=block)


def entrypoint_setup() -> str:
    """Entrypoint lines: a private per-session token file outside every uploaded directory."""

    from .session_served_policy import TOKEN_ENV

    return f'''SESSION_POLICY_DIR="$(mktemp -d /tmp/blueprint-session-policy.XXXXXX)"
chmod 700 "$SESSION_POLICY_DIR"
(umask 077 && /isaac-sim/python.sh -c 'import secrets; print(secrets.token_hex(32))' >"$SESSION_POLICY_DIR/token")
token_rc=$?
if [ $token_rc -ne 0 ] || [ ! -s "$SESSION_POLICY_DIR/token" ]; then
  write_fallback_result session_policy_token_unavailable 86
  exit 86
fi
export {TOKEN_ENV}="$SESSION_POLICY_DIR/token"'''


def bundle_contract(spec: Mapping[str, Any]) -> dict[str, Any]:
    """What the sealed provider manifest records about the session-served candidate."""

    from .session_served_policy import validate_session_served_spec

    block = validate_session_served_spec(spec)["session_policy_server"]
    return {"server": block["server"], "module": block["module"], "native_candidate_id": block["native_candidate_id"],
            "port": block["port"], "startup_timeout_seconds": block["startup_timeout_seconds"],
            "token_env": block["token_env"], "token_origin": "generated_on_session", "outbound_calls": "none"}


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


__all__ = ["bundle_contract", "entrypoint_setup", "provisioning_script", "server_command", "start"]
