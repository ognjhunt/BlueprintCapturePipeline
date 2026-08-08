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

import hashlib
import json
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "adp009d_policy_server_worker.v1"

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
# Loading 12.4 GB of weights dominates this; a port appears long before.
READINESS_TIMEOUT_SECONDS = 900.0
READINESS_POLL_SECONDS = 10.0
# A readiness deadline is meaningless if one import, client constructor, ping,
# or inference can occupy the polling thread forever.  This bound is shorter
# than the overall model-load allowance and leaves the outer loop free to
# observe process exit and its own deadline.
ROUND_TRIP_ATTEMPT_TIMEOUT_SECONDS = 30.0
# Enough consecutive timeouts to distinguish a slow load from a wedged
# client, without letting a genuinely blocked client run to the deadline.
MAX_CONSECUTIVE_ATTEMPT_TIMEOUTS = 6
SERVER_LOG_TAIL_BYTES = 32_768
DROID_ACTION_WIDTH = 8
DROID_OPEN_LOOP_HORIZON = 8

TRANSPORT_OPENPI_WEBSOCKET = "openpi_websocket"
TRANSPORT_GROOT_ZMQ = "groot_zmq"

# Read from each vendor runtime rather than assumed: openpi serves over a
# websocket on 8000 via scripts/serve_policy.py, while GR00T's client is
# gr00t.policy.server_client.PolicyClient over ZMQ, defaulting to 5555.  They
# share neither a transport nor a launch command, so a single hardcoded form
# can only ever serve one of them.
CANDIDATE_TRANSPORTS = {
    "pi05_droid": TRANSPORT_OPENPI_WEBSOCKET,
    "groot_n17_droid": TRANSPORT_GROOT_ZMQ,
    "groot_n16_droid": TRANSPORT_GROOT_ZMQ,
    "cosmos3_edge_policy_droid": TRANSPORT_OPENPI_WEBSOCKET,
}
CANDIDATE_DEFAULT_PORTS = {
    TRANSPORT_OPENPI_WEBSOCKET: 8000,
    TRANSPORT_GROOT_ZMQ: 5555,
}


class RoundTripAttemptTimeout(TimeoutError):
    """One readiness attempt starved the poll loop past its own bound."""


def transport_for(candidate_id: str) -> str:
    if candidate_id not in CANDIDATE_TRANSPORTS:
        raise RuntimeError(f"policy_server_unknown_candidate:{candidate_id}")
    return CANDIDATE_TRANSPORTS[candidate_id]


def build_serve_command(
    *, candidate_id: str, python: str, source_root: str, checkpoint_root: str, port: int
) -> list[str]:
    """The launch command for this candidate's own server.

    openpi's form is verified against scripts/serve_policy.py at the frozen
    revision: tyro.cli over Args{port, policy: Checkpoint|Default}, so the union
    subcommand is policy:checkpoint with --policy.config and --policy.dir, and
    its DEFAULT_CHECKPOINT names config "pi05_droid".
    """

    transport = transport_for(candidate_id)
    if transport == TRANSPORT_OPENPI_WEBSOCKET:
        config = "pi05_droid" if candidate_id == "pi05_droid" else candidate_id
        return [
            python,
            str(Path(source_root, "scripts", "serve_policy.py")),
            "--port",
            str(port),
            "policy:checkpoint",
            f"--policy.config={config}",
            f"--policy.dir={checkpoint_root}",
        ]
    return [
        python,
        "-m",
        "gr00t.policy.server",
        "--model-path",
        checkpoint_root,
        "--port",
        str(port),
    ]


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


def attempt_round_trip(host: str, port: int, transport: str = TRANSPORT_OPENPI_WEBSOCKET) -> dict:
    """One real inference over this candidate's transport, or raises."""

    import numpy as np

    if transport == TRANSPORT_GROOT_ZMQ:
        from gr00t.policy.server_client import PolicyClient

        client = PolicyClient(host=host, port=int(port), timeout_ms=15000, strict=False)
        if client.ping() is not True:
            raise RuntimeError("policy_server_ping_failed")
        response = client.get_action(_probe_observation())
    else:
        from openpi_client import websocket_client_policy

        client = websocket_client_policy.WebsocketClientPolicy(
            host=host, port=int(port)
        )
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


def _attempt_round_trip_with_timeout(
    *,
    host: str,
    port: int,
    transport: str,
    timeout_seconds: float,
) -> dict:
    """Run one vendor attempt in a daemon thread with a real wall deadline.

    Python cannot safely kill a blocked import or third-party constructor.  A
    daemon thread lets the worker emit its failure receipt and exit instead of
    joining the stuck call during interpreter shutdown.  After a timeout the
    caller fails immediately; it never accumulates more abandoned attempts.
    """

    timeout = float(timeout_seconds)
    if timeout <= 0:
        raise RoundTripAttemptTimeout("policy_server_round_trip_attempt_timeout_invalid")

    result_queue: queue.Queue[tuple[bool, Any]] = queue.Queue(maxsize=1)

    def _run() -> None:
        try:
            result_queue.put((True, attempt_round_trip(host, port, transport)))
        except BaseException as exc:  # noqa: BLE001 - re-raised on the polling thread
            result_queue.put((False, exc))

    thread = threading.Thread(
        target=_run,
        name=f"policy-round-trip-{transport}",
        daemon=True,
    )
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        raise RoundTripAttemptTimeout(
            f"policy_server_round_trip_attempt_timed_out:{timeout:.3f}s"
        )

    succeeded, payload = result_queue.get_nowait()
    if succeeded:
        return dict(payload)
    if isinstance(payload, BaseException):
        raise payload
    raise RuntimeError("policy_server_round_trip_attempt_returned_invalid_result")


def _server_log_summary(path: Path) -> dict[str, Any]:
    """Retain bounded diagnostics in the receipt as well as the output file."""

    if not path.is_file():
        return {
            "path": str(path),
            "present": False,
            "size_bytes": 0,
            "sha256": None,
            "tail": None,
        }
    data = path.read_bytes()
    return {
        "path": str(path),
        "present": True,
        "size_bytes": len(data),
        "sha256": "sha256:" + hashlib.sha256(data).hexdigest(),
        "tail": data[-SERVER_LOG_TAIL_BYTES:].decode("utf-8", errors="replace"),
        "tail_bytes_limit": SERVER_LOG_TAIL_BYTES,
    }


def wait_for_round_trip(
    *,
    host: str,
    port: int,
    timeout_seconds: float,
    process: subprocess.Popen | None,
    transport: str = TRANSPORT_OPENPI_WEBSOCKET,
) -> dict:
    """Poll until one inference succeeds, the process dies, or time runs out."""

    started = time.monotonic()
    attempts = 0
    timed_out_attempts = 0
    last_error: str | None = None
    while time.monotonic() - started < timeout_seconds:
        if process is not None and process.poll() is not None:
            raise RuntimeError(
                f"policy_server_exited_before_ready:{process.returncode}"
            )
        attempts += 1
        remaining = float(timeout_seconds) - (time.monotonic() - started)
        attempt_timeout = min(ROUND_TRIP_ATTEMPT_TIMEOUT_SECONDS, remaining)
        try:
            result = _attempt_round_trip_with_timeout(
                host=host,
                port=port,
                transport=transport,
                timeout_seconds=attempt_timeout,
            )
            result["readiness_attempts"] = attempts
            result["readiness_seconds"] = round(time.monotonic() - started, 3)
            return result
        except RoundTripAttemptTimeout as exc:
            # Retried, not terminal.  Ending readiness on the first timed-out
            # attempt makes any server slower than the attempt bound
            # unreachable: pi05_droid loads 12.4 GB in about 53 seconds against
            # a 30 second bound, so a policy with four clean prior receipts
            # failed every time.
            #
            # The original concern -- not starting another vendor call while a
            # timed-out daemon thread may still hold the socket -- is real, so
            # consecutive timeouts are capped rather than ignored.  A server
            # that never answers still fails, and now says which way it failed.
            timed_out_attempts += 1
            last_error = f"{type(exc).__name__}: {exc}"
            if timed_out_attempts >= MAX_CONSECUTIVE_ATTEMPT_TIMEOUTS:
                raise RuntimeError(
                    f"policy_client_blocked_repeatedly:{timed_out_attempts}:{exc}"
                ) from exc
            time.sleep(min(READINESS_POLL_SECONDS, max(0.0, timeout_seconds - (time.monotonic() - started))))
        except Exception as exc:  # noqa: BLE001 - not-yet-ready is the normal case
            last_error = f"{type(exc).__name__}: {exc}"
            remaining = float(timeout_seconds) - (time.monotonic() - started)
            if remaining > 0:
                time.sleep(min(READINESS_POLL_SECONDS, remaining))
    raise RuntimeError(f"policy_server_never_answered:{last_error}")


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=0)
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

    transport = transport_for(args.candidate_id)
    port = int(args.port or CANDIDATE_DEFAULT_PORTS[transport])
    command = build_serve_command(
        candidate_id=args.candidate_id,
        python=args.python,
        source_root=args.source_root,
        checkpoint_root=args.checkpoint_root,
        port=port,
    )

    receipt: dict = {
        "schema_version": SCHEMA_VERSION,
        "candidate_id": args.candidate_id,
        "host": args.host,
        "port": port,
        "transport": transport,
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
                port=port,
                timeout_seconds=float(args.timeout_seconds),
                process=process,
                transport=transport,
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

    # The provider output zipper retains every bounded file in OUT_DIR, so the
    # log itself survives.  Embed its digest and tail in the receipt too: even
    # a future packaging regression cannot erase the diagnosis silently.
    receipt["server_log"] = _server_log_summary(log_path)

    Path(args.receipt).parent.mkdir(parents=True, exist_ok=True)
    Path(args.receipt).write_text(
        json.dumps(receipt, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    print(f"BLUEPRINT_ADP009D_POLICY_SERVER:{receipt['status']}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
