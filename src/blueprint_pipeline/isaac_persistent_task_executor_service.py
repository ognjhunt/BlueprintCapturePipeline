"""HTTP service owning one persistent Isaac stage for a complete task episode."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

from .oscar_isaac_closed_loop_eval import build_sc3_runtime_attestation


def _computed(criterion: dict[str, Any], before: float, after: float) -> bool:
    comparison = str(criterion.get("comparison") or "")
    tolerance = float(criterion.get("tolerance") or 0.0)
    if comparison == "increase_at_least":
        return after - before >= tolerance
    if comparison == "decrease_at_least":
        return before - after >= tolerance
    if comparison == "absolute_change_at_least":
        return abs(after - before) >= tolerance
    target = float(criterion.get("target_value") or 0.0)
    if comparison == "within_tolerance":
        return abs(after - target) <= tolerance
    if comparison == "at_or_above":
        return after >= target - tolerance
    if comparison == "at_or_below":
        return after <= target + tolerance
    raise ValueError("persistent_isaac_completion_comparison_unsupported")


def serve(
    *,
    backend: Any,
    host: str,
    port: int,
    signing_key_file: str,
    attempt_input_manifest_sha256: str,
) -> None:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):  # noqa: A002
            return

        def do_POST(self):  # noqa: N802
            if self.path != "/apply-and-measure":
                self.send_error(404)
                return
            try:
                length = int(self.headers.get("Content-Length") or 0)
                request = json.loads(self.rfile.read(length).decode("utf-8"))
                result = dict(backend.apply_and_measure(request))
                contract = dict(request.get("task_success_contract") or {})
                criteria = list(contract.get("registered_criteria") or contract.get("criteria") or [])
                criterion = dict(criteria[0])
                result.update(
                    {
                        "status": "completed",
                        "comparison": criterion.get("comparison"),
                        "tolerance": criterion.get("tolerance"),
                        "passed": _computed(
                            criterion,
                            float(result["before_value"]),
                            float(result["after_value"]),
                        ),
                        "attempt_input_manifest_sha256": attempt_input_manifest_sha256,
                    }
                )
                result["evaluator_attestation"] = build_sc3_runtime_attestation(
                    result,
                    private_key_file=signing_key_file,
                    report_path=Path(backend.evidence_dir)
                    / f"task_measurement_{int(request.get('step_index') or 0):04d}_signature.json",
                    signer_key_id="persistent-isaac-task-executor",
                    verifier_id="blueprint-task-transition-verifier",
                )
                body = json.dumps(result, sort_keys=True).encode("utf-8")
                self.send_response(200)
            except Exception as exc:  # noqa: BLE001
                body = json.dumps(
                    {"status": "blocked", "error_type": type(exc).__name__, "error": str(exc)}
                ).encode("utf-8")
                self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    # Isaac/Omniverse stage and dynamic-control interfaces are thread-affine.
    # A single request loop preserves one ordered timeline and prevents two
    # completion calls from mutating or measuring the articulation concurrently.
    server = HTTPServer((host, int(port)), Handler)
    try:
        server.serve_forever()
    finally:
        server.server_close()
        backend.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--robot-prim-path", default="/World/G1")
    parser.add_argument("--evidence-dir", default="/workspace/closed_loop_out/isaac_task_state")
    parser.add_argument("--initial-state-output", default="/workspace/initial_g1_sonic_state.json")
    parser.add_argument("--attempt-input-manifest", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    attempt_path = Path(args.attempt_input_manifest)
    attempt_sha = hashlib.sha256(attempt_path.read_bytes()).hexdigest()
    signing_key = os.environ.get("BLUEPRINT_SC3_TASK_COMPLETION_PRIVATE_KEY_FILE", "")
    if not signing_key:
        raise SystemExit("BLUEPRINT_SC3_TASK_COMPLETION_PRIVATE_KEY_FILE required")
    from .isaac_runtime_task_backend import create_backend

    backend = create_backend(
        stage_path=args.stage,
        robot_prim_path=args.robot_prim_path,
        evidence_dir=args.evidence_dir,
    )
    initial = backend.initial_policy_state()
    Path(args.initial_state_output).write_text(
        json.dumps(initial, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    serve(
        backend=backend,
        host=args.host,
        port=args.port,
        signing_key_file=signing_key,
        attempt_input_manifest_sha256=attempt_sha,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
