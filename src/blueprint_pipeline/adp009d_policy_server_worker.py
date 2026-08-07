"""Start the policy server on the worker and prove it answers before declaring it up.

A listening socket is not readiness.  One shipped server in this repo writes
``model_loaded_ready_to_serve`` before it calls ``serve_forever``, so a port
check would report a server that cannot yet serve.  Loading a 12.4 GB checkpoint
also takes far longer than binding a port, and an episode that starts too early
fails in a way that looks like a policy problem rather than a startup race.

So readiness here means one completed inference round trip returning a
well-formed action chunk.  Nothing weaker is accepted, and the observation used
to probe it is built by the same adapter the episode will use, so a shape
mismatch surfaces here rather than mid-episode.

Runs under the policy venv's interpreter, never Isaac's.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

SCHEMA_VERSION = "adp009d_policy_server_worker.v1"

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
# Loading 12.4 GB of weights dominates this; a port appears long before.
READINESS_TIMEOUT_SECONDS = 900.0
READINESS_POLL_SECONDS = 10.0
DROID_ACTION_WIDTH = 8
DROID_OPEN_LOOP_HORIZON = 8


def _probe_observation() -> dict:
    """The observation the episode will send, so a shape error surfaces now."""

    import numpy as np

    frame = np.zeros((224, 224, 3), dtype=np.uint8)
    return {
        "observation/exterior_image_1_left": frame,
        "observation/wrist_image_left": frame,
        "observation/joint_position": np.zeros(7, dtype=float),
        "observation/gripper_position": np.zeros(1, dtype=float),
        "prompt": "pick up the can",
    }


def attempt_round_trip(host: str, port: int) -> dict:
    """One real inference.  Returns the observed chunk shape, or raises."""

    import numpy as np
    from openpi_client import websocket_client_policy

    client = websocket_client_policy.WebsocketClientPolicy(host=host, port=int(port))
    response = client.infer(_probe_observation())
    actions = response["actions"] if isinstance(response, dict) else response
    chunk = np.asarray(actions, dtype=float)
    if chunk.ndim != 2 or chunk.shape[1] != DROID_ACTION_WIDTH:
        raise RuntimeError(f"policy_round_trip_chunk_shape_invalid:{chunk.shape}")
    if chunk.shape[0] < DROID_OPEN_LOOP_HORIZON:
        raise RuntimeError(f"policy_round_trip_chunk_too_short:{chunk.shape[0]}")
    if not np.isfinite(chunk).all():
        raise RuntimeError("policy_round_trip_chunk_nonfinite")
    return {
        "action_chunk_rows": int(chunk.shape[0]),
        "action_chunk_width": int(chunk.shape[1]),
        "server_metadata": (
            client.get_server_metadata()
            if hasattr(client, "get_server_metadata")
            else None
        ),
    }


def wait_for_round_trip(
    *, host: str, port: int, timeout_seconds: float, process: subprocess.Popen | None
) -> dict:
    """Poll until one inference succeeds, the process dies, or time runs out."""

    started = time.monotonic()
    attempts = 0
    last_error: str | None = None
    while time.monotonic() - started < timeout_seconds:
        if process is not None and process.poll() is not None:
            raise RuntimeError(
                f"policy_server_exited_before_ready:{process.returncode}"
            )
        attempts += 1
        try:
            result = attempt_round_trip(host, port)
            result["readiness_attempts"] = attempts
            result["readiness_seconds"] = round(time.monotonic() - started, 3)
            return result
        except Exception as exc:  # noqa: BLE001 - not-yet-ready is the normal case
            last_error = f"{type(exc).__name__}: {exc}"
            time.sleep(READINESS_POLL_SECONDS)
    raise RuntimeError(f"policy_server_never_answered:{last_error}")


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--log", required=True)
    parser.add_argument("--receipt", required=True)
    parser.add_argument(
        "--timeout-seconds", type=float, default=READINESS_TIMEOUT_SECONDS
    )
    args = parser.parse_args(list(sys.argv[1:] if argv is None else argv))

    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        raise SystemExit("policy_server_must_be_loopback_only")

    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    serve_script = Path(args.source_root, "scripts", "serve_policy.py")

    command = [
        args.python,
        str(serve_script),
        "--port",
        str(args.port),
        "policy:checkpoint",
        "--policy.config=pi05_droid",
        f"--policy.dir={args.checkpoint_root}",
    ]

    receipt: dict = {
        "schema_version": SCHEMA_VERSION,
        "candidate_id": args.candidate_id,
        "host": args.host,
        "port": int(args.port),
        "command": command,
        "policy_interpreter": args.python,
    }

    process: subprocess.Popen | None = None
    try:
        with log_path.open("wb") as log_handle:
            process = subprocess.Popen(  # noqa: S603
                command, stdout=log_handle, stderr=subprocess.STDOUT
            )
            round_trip = wait_for_round_trip(
                host=args.host,
                port=int(args.port),
                timeout_seconds=float(args.timeout_seconds),
                process=process,
            )
        receipt.update(
            {
                "status": "ready",
                "server_pid": process.pid,
                "round_trip_completed": True,
                **round_trip,
            }
        )
        exit_code = 0
    except Exception as exc:  # noqa: BLE001 - the failure is the evidence
        receipt.update(
            {
                "status": "blocked",
                "round_trip_completed": False,
                "error": f"{type(exc).__name__}: {exc}",
                "server_pid": process.pid if process else None,
                "server_exit_code": process.poll() if process else None,
            }
        )
        exit_code = 1

    Path(args.receipt).parent.mkdir(parents=True, exist_ok=True)
    Path(args.receipt).write_text(
        json.dumps(receipt, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    print(f"BLUEPRINT_ADP009D_POLICY_SERVER:{receipt['status']}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
