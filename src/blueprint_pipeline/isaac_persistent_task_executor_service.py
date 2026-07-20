"""HTTP service owning one persistent Isaac stage for a complete task episode."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

from .oscar_isaac_closed_loop_eval import build_sc3_runtime_attestation
from .task_episode_baseline import evaluate_task_criterion


def _write_json_atomic(path: str | Path, payload: dict[str, Any]) -> None:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _clear_readiness_output(path: str | Path) -> None:
    destination = Path(path).expanduser().resolve()
    if destination.exists() or destination.is_symlink():
        if destination.is_dir() and not destination.is_symlink():
            raise SystemExit("persistent_isaac_readiness_output_is_directory")
        destination.unlink()


def _evaluate_completion_request(
    *,
    backend: Any,
    request: dict[str, Any],
    signing_key_file: str,
    attempt_input_manifest_sha256: str,
) -> dict[str, Any]:
    result = dict(backend.apply_and_measure(request))
    contract = dict(request.get("task_success_contract") or {})
    criteria = list(contract.get("registered_criteria") or contract.get("criteria") or [])
    criterion = dict(criteria[0])
    episode_fields = ("episode_initial_value", "step_before", "step_after")
    missing = sorted(field for field in episode_fields if result.get(field) is None)
    if missing:
        raise RuntimeError(
            "persistent_isaac_task_result_episode_fields_missing:" + ",".join(missing)
        )
    evaluation = evaluate_task_criterion(
        criterion,
        episode_initial_value=float(result["episode_initial_value"]),
        step_before=float(result["step_before"]),
        step_after=float(result["step_after"]),
    )
    result.update(
        {
            "status": "completed",
            "comparison": criterion.get("comparison"),
            "tolerance": criterion.get("tolerance"),
            "passed": evaluation["passed"],
            "evaluation_basis": evaluation["evaluation_basis"],
            "step_delta": evaluation["step_delta"],
            "episode_delta": evaluation["episode_delta"],
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
    return result


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
                result = _evaluate_completion_request(
                    backend=backend,
                    request=request,
                    signing_key_file=signing_key_file,
                    attempt_input_manifest_sha256=attempt_input_manifest_sha256,
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
        _serve_with_live_state_heartbeat(server=server, backend=backend)
    finally:
        server.server_close()
        backend.close()


def _serve_with_live_state_heartbeat(
    *, server: Any, backend: Any, max_iterations: int | None = None
) -> None:
    """Serve serial requests while keeping the same Isaac state fresh for DDS."""

    refresh = getattr(backend, "refresh_live_state_snapshot", None)
    if not callable(refresh):
        raise RuntimeError("persistent_isaac_live_state_heartbeat_method_missing")
    # ``handle_request`` returns after this interval even when idle. The bridge
    # rejects source samples older than 500 ms, so 20 ms leaves ample margin
    # without touching Isaac from a background thread.
    server.timeout = 0.02
    iterations = 0
    while max_iterations is None or iterations < int(max_iterations):
        server.handle_request()
        refresh()
        iterations += 1


def _load_attempt_bound_task_contract(
    attempt_manifest: dict[str, Any],
    attempt_path: Path,
    explicit_contract_path: str,
) -> dict[str, Any]:
    ref = dict((attempt_manifest.get("artifacts") or {}).get("task_success_contract") or {})
    expected_sha = str(ref.get("sha256") or "").lower()
    if len(expected_sha) != 64:
        raise SystemExit("persistent_isaac_task_contract_ref_missing")
    candidates: list[Path] = []
    if explicit_contract_path:
        candidates.append(Path(explicit_contract_path))
    ref_path = str(ref.get("path") or "")
    if ref_path:
        candidates.append(Path(ref_path))
        candidates.append(attempt_path.parent / Path(ref_path).name)
    contract_file = next((path for path in candidates if path.is_file()), None)
    if contract_file is None:
        raise SystemExit("persistent_isaac_task_contract_file_missing")
    if hashlib.sha256(contract_file.read_bytes()).hexdigest() != expected_sha:
        raise SystemExit("persistent_isaac_task_contract_sha256_mismatch")
    return json.loads(contract_file.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--robot-prim-path", default="/World/G1")
    parser.add_argument(
        "--g1-usd",
        default=os.environ.get(
            "BLUEPRINT_ISAAC_UNITREE_G1_USD",
            "/isaac-sim/Isaac/Robots/Unitree/G1/g1.usd",
        ),
    )
    parser.add_argument("--route-file", required=True)
    parser.add_argument("--evidence-dir", default="/workspace/closed_loop_out/isaac_task_state")
    parser.add_argument("--initial-state-output", default="/workspace/initial_g1_sonic_state.json")
    parser.add_argument(
        "--initial-frame-output",
        default="/workspace/initial_policy_frame.png",
    )
    parser.add_argument(
        "--camera-projection-context-output",
        default="/workspace/controller_fk_camera_projection_context.json",
    )
    parser.add_argument("--attempt-input-manifest", required=True)
    parser.add_argument("--task-success-contract", default="")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    for output_path in (
        args.initial_state_output,
        args.initial_frame_output,
        args.camera_projection_context_output,
    ):
        _clear_readiness_output(output_path)
    attempt_path = Path(args.attempt_input_manifest)
    attempt_sha = hashlib.sha256(attempt_path.read_bytes()).hexdigest()
    attempt_manifest = json.loads(attempt_path.read_text(encoding="utf-8"))
    attempt_id = str(attempt_manifest.get("attempt_id") or "")
    launch_nonce = str(attempt_manifest.get("launch_nonce") or "")
    if not attempt_id or not launch_nonce:
        raise SystemExit("persistent_isaac_attempt_identity_missing")
    allocation_launch_session_id = str(
        attempt_manifest.get("allocation_launch_session_id") or launch_nonce
    )
    qualification_attempt_bound = (
        attempt_manifest.get("qualification_attempt_bound") is True
    )
    qualification_attempt_sequence = attempt_manifest.get(
        "qualification_attempt_sequence"
    )
    qualification_attempt_nonce = str(
        attempt_manifest.get("qualification_attempt_nonce") or ""
    )
    qualification_attempt_nonce_sha256 = str(
        attempt_manifest.get("qualification_attempt_nonce_sha256") or ""
    ).lower()
    if qualification_attempt_bound:
        try:
            qualification_attempt_sequence = int(qualification_attempt_sequence)
        except (TypeError, ValueError) as exc:
            raise SystemExit("persistent_isaac_qualification_attempt_sequence_invalid") from exc
        if (
            qualification_attempt_sequence < 1
            or qualification_attempt_nonce != launch_nonce
            or hashlib.sha256(qualification_attempt_nonce.encode("utf-8")).hexdigest()
            != qualification_attempt_nonce_sha256
        ):
            raise SystemExit("persistent_isaac_qualification_attempt_identity_invalid")
    elif any(
        value not in (None, "")
        for value in (
            qualification_attempt_sequence,
            qualification_attempt_nonce,
            qualification_attempt_nonce_sha256,
        )
    ):
        raise SystemExit("persistent_isaac_qualification_attempt_identity_unexpected")
    task_contract = _load_attempt_bound_task_contract(
        attempt_manifest, attempt_path, args.task_success_contract
    )
    task_contract_artifact_sha256 = str(
        dict((attempt_manifest.get("artifacts") or {}).get("task_success_contract") or {}).get(
            "sha256"
        )
        or ""
    ).lower()
    signing_key = os.environ.get("BLUEPRINT_SC3_TASK_COMPLETION_PRIVATE_KEY_FILE", "")
    if not signing_key:
        raise SystemExit("BLUEPRINT_SC3_TASK_COMPLETION_PRIVATE_KEY_FILE required")
    from .isaac_runtime_task_backend import create_backend

    backend = create_backend(
        stage_path=args.stage,
        robot_prim_path=args.robot_prim_path,
        evidence_dir=args.evidence_dir,
        g1_usd_path=args.g1_usd,
        route_file=args.route_file,
    )
    backend.allocation_launch_session_id = allocation_launch_session_id
    backend.qualification_attempt_bound = qualification_attempt_bound
    backend.qualification_attempt_sequence = qualification_attempt_sequence
    backend.qualification_attempt_nonce_sha256 = qualification_attempt_nonce_sha256 or None
    baseline = backend.capture_episode_baseline(
        task_success_contract=task_contract,
        attempt_id=attempt_id,
        launch_nonce=launch_nonce,
        task_contract_artifact_sha256=task_contract_artifact_sha256,
    )
    attestation = build_sc3_runtime_attestation(
        baseline,
        private_key_file=signing_key,
        report_path=Path(backend.evidence_dir) / "task_episode_baseline_signature.json",
        signer_key_id="persistent-isaac-task-executor",
        verifier_id="blueprint-task-episode-baseline-verifier",
    )
    (Path(backend.evidence_dir) / "task_episode_baseline_attestation.json").write_text(
        json.dumps(attestation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    backend.install_episode_baseline_attestation(attestation)
    target_prim_path = str(baseline.get("articulation_prim_path") or "").strip()
    if not target_prim_path:
        raise SystemExit("persistent_isaac_initial_policy_target_prim_missing")
    initial_observation = backend.capture_initial_policy_observation(
        target_prim_path=target_prim_path,
        frame_output_path=args.initial_frame_output,
        projection_context_output_path=args.camera_projection_context_output,
    )
    if (
        not isinstance(initial_observation, dict)
        or initial_observation.get("status") != "completed"
    ):
        raise SystemExit("persistent_isaac_initial_policy_observation_not_completed")
    baseline_guard = backend.verify_initial_observation_preserved_episode_baseline(
        task_success_contract=task_contract,
    )
    if baseline_guard.get("status") != "passed":
        raise SystemExit("persistent_isaac_initial_observation_baseline_guard_failed")
    # This is the readiness marker consumed by the worker.  Write it last and
    # atomically so its existence proves the live frame/context and the
    # post-render baseline guard all completed in this process.
    initial = backend.initial_policy_state()
    _write_json_atomic(args.initial_state_output, initial)
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
