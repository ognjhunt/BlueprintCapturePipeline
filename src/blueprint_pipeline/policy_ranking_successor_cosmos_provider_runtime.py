"""Standalone provider runtime embedded in the successor Cosmos3 bundle.

The bundle builder copies this file into ``provider_runtime``.  It intentionally
uses no Blueprint imports so the pinned public vLLM-Omni image can execute it.
It is not a provider launcher: allocation and teardown remain owned by the
canonical paid-resource allocator and Vast adapter.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import requests


CHECKPOINT = "nvidia/Cosmos3-Nano"
CHECKPOINT_REVISION = "411f42a8fdfb8c5b2583cb8786e0938f49796eaa"
EXPERIMENT_ID = "policy_ranking_successor_experiment_20260727"
SERVER_PORT = 8001
SERVER_BASE_URL = f"http://127.0.0.1:{SERVER_PORT}"
SERVER_START_TIMEOUT_SECONDS = 3600
REQUEST_POLL_TIMEOUT_SECONDS = 1200
INFRASTRUCTURE_RETRY_LIMIT = 1
EXPECTED_CONDITIONS = ("recorded", "zero", "shuffled", "reversed", "policy_swapped")
EXPECTED_SEEDS = (0, 1)


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


class HashChainedJournal:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.previous = "0" * 64
        self.sequence = 0
        if path.is_file():
            for raw in path.read_text(encoding="utf-8").splitlines():
                row = json.loads(raw)
                self.previous = str(row["event_sha256"])
                self.sequence = int(row["sequence"]) + 1

    def append(self, event: dict[str, Any]) -> dict[str, Any]:
        row = {
            "sequence": self.sequence,
            "recorded_at_epoch": time.time(),
            "previous_event_sha256": self.previous,
            **event,
        }
        row["event_sha256"] = canonical_sha256(row)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self.previous = row["event_sha256"]
        self.sequence += 1
        return row


def _decode_video_metrics(path: Path) -> dict[str, Any]:
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        return {"status": "blocked", "blockers": ["ffmpeg_or_ffprobe_missing"]}
    probe = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,nb_frames,r_frame_rate",
            "-of",
            "json",
            str(path),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if probe.returncode != 0:
        return {"status": "blocked", "blockers": ["ffprobe_failed"]}
    try:
        probe_payload = json.loads(probe.stdout)
    except json.JSONDecodeError:
        return {"status": "blocked", "blockers": ["ffprobe_json_invalid"]}
    decoded = subprocess.run(
        [
            ffmpeg,
            "-v",
            "error",
            "-i",
            str(path),
            "-vf",
            "scale=64:54,format=gray",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "gray",
            "pipe:1",
        ],
        check=False,
        capture_output=True,
        timeout=180,
    )
    frame_size = 64 * 54
    if decoded.returncode != 0 or len(decoded.stdout) < frame_size:
        return {"status": "blocked", "blockers": ["video_decode_failed"]}
    usable = len(decoded.stdout) // frame_size * frame_size
    frames = np.frombuffer(decoded.stdout[:usable], dtype=np.uint8).reshape(-1, 54, 64)
    frame_means = frames.astype(np.float32).mean(axis=(1, 2))
    frame_stds = frames.astype(np.float32).std(axis=(1, 2))
    temporal = (
        np.abs(np.diff(frames.astype(np.float32), axis=0)).mean(axis=(1, 2))
        if len(frames) > 1
        else np.asarray([], dtype=np.float32)
    )
    blank = bool(float(frame_stds.max(initial=0.0)) < 2.0)
    static = bool(float(temporal.max(initial=0.0)) < 0.5)
    return {
        "status": "passed" if not blank and not static else "scientific_failure",
        "frame_count_decoded": int(len(frames)),
        "mean_luma_min": float(frame_means.min(initial=0.0)),
        "mean_luma_max": float(frame_means.max(initial=0.0)),
        "spatial_std_max": float(frame_stds.max(initial=0.0)),
        "temporal_absolute_difference_mean": float(temporal.mean()) if len(temporal) else 0.0,
        "temporal_absolute_difference_max": float(temporal.max(initial=0.0)),
        "blank_detected": blank,
        "static_detected": static,
        "ffprobe": probe_payload,
    }


def _server_command() -> list[str]:
    executable = shutil.which("vllm")
    if not executable:
        raise RuntimeError("vllm_cli_missing")
    return [
        executable,
        "serve",
        CHECKPOINT,
        "--revision",
        CHECKPOINT_REVISION,
        "--omni",
        "--host",
        "127.0.0.1",
        "--port",
        str(SERVER_PORT),
        "--init-timeout",
        "1800",
        "--dtype",
        "bfloat16",
    ]


def _wait_for_server(process: subprocess.Popen[bytes]) -> None:
    deadline = time.monotonic() + SERVER_START_TIMEOUT_SECONDS
    last_error = ""
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"vllm_server_exited:{process.returncode}")
        try:
            response = requests.get(f"{SERVER_BASE_URL}/health", timeout=10)
            if response.ok:
                return
            last_error = f"http_{response.status_code}"
        except requests.RequestException as exc:
            last_error = type(exc).__name__
        time.sleep(5)
    raise RuntimeError(f"vllm_server_start_timeout:{last_error}")


def _submit_rollout(
    *,
    request_row: dict[str, Any],
    action_stream: dict[str, Any],
    initial_observation: Path,
    output_path: Path,
) -> dict[str, Any]:
    extra_params = {
        "action_mode": "forward_dynamics",
        "domain_name": "droid_lerobot",
        "action_chunk_size": 16,
        "image_size": 480,
        "view_point": "concat_view",
        "action": action_stream["actions"],
        # The pinned NVIDIA DROID example disables guardrails. This bundle is
        # restricted to that public robotics sample and records the exception.
        "guardrails": False,
    }
    form = {
        "prompt": "A robot manipulates an object.",
        "num_frames": "17",
        "fps": "15",
        "size": "640x540",
        "num_inference_steps": "30",
        "guidance_scale": "1.0",
        "flow_shift": "10.0",
        "seed": str(request_row["seed"]),
        "extra_params": json.dumps(extra_params, separators=(",", ":")),
    }
    with initial_observation.open("rb") as image_file:
        response = requests.post(
            f"{SERVER_BASE_URL}/v1/videos",
            data=form,
            files={"input_reference": (initial_observation.name, image_file, "image/png")},
            timeout=180,
        )
    response.raise_for_status()
    initial = response.json()
    job_id = str(initial["id"])
    deadline = time.monotonic() + REQUEST_POLL_TIMEOUT_SECONDS
    final: dict[str, Any] = {}
    while time.monotonic() < deadline:
        polled = requests.get(f"{SERVER_BASE_URL}/v1/videos/{job_id}", timeout=30)
        polled.raise_for_status()
        final = dict(polled.json())
        if final.get("status") == "completed":
            break
        if final.get("status") in {"failed", "cancelled"}:
            raise RuntimeError(f"vllm_job_{final.get('status')}:{job_id}")
        time.sleep(2)
    else:
        raise TimeoutError(f"vllm_job_timeout:{job_id}")
    content = requests.get(
        f"{SERVER_BASE_URL}/v1/videos/{job_id}/content", timeout=300
    )
    content.raise_for_status()
    output_path.write_bytes(content.content)
    return {
        "provider_job_id": job_id,
        "initial_response": initial,
        "final_response": final,
        "output_sha256": sha256_file(output_path),
        "output_size_bytes": output_path.stat().st_size,
    }


def _classify_retryable(exc: BaseException) -> bool:
    return isinstance(
        exc,
        (
            requests.ConnectionError,
            requests.Timeout,
            TimeoutError,
        ),
    )


def run() -> dict[str, Any]:
    runtime_dir = Path(__file__).resolve().parent
    input_dir = runtime_dir / "cosmos3_input"
    output_dir = Path(
        os.environ.get("BLUEPRINT_VAST_PROVIDER_OUTPUT_DIR", runtime_dir.parent / "runtime_output")
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    journal = HashChainedJournal(output_dir / "immutable_request_journal.jsonl")
    result_path = output_dir / "wam_runtime_result.json"
    initial_observation = input_dir / "initial_observation.png"
    inventory = json.loads((input_dir / "smoke_request_inventory.json").read_text())
    action_streams = json.loads((input_dir / "action_streams.json").read_text())
    input_checks = {
        "initial_observation_sha256": sha256_file(initial_observation),
        "inventory_sha256": canonical_sha256(inventory),
        "action_streams_sha256": canonical_sha256(action_streams),
    }
    blockers: list[str] = []
    if input_checks["initial_observation_sha256"] != inventory.get(
        "initial_observation_sha256"
    ):
        blockers.append("initial_observation_sha256_mismatch")
    conditions = action_streams.get("conditions")
    if not isinstance(conditions, dict) or tuple(conditions) != EXPECTED_CONDITIONS:
        blockers.append("action_stream_condition_order_invalid")
    expected_pairs = {
        (condition, seed) for condition in EXPECTED_CONDITIONS for seed in EXPECTED_SEEDS
    }
    rows = inventory.get("requests") if isinstance(inventory.get("requests"), list) else []
    observed_pairs = {(row.get("condition"), row.get("seed")) for row in rows}
    if observed_pairs != expected_pairs or len(rows) != 10:
        blockers.append("smoke_request_matrix_invalid")
    if blockers:
        result = {
            "schema_version": "policy_ranking_successor_cosmos_runtime.v1",
            "status": "blocked",
            "failure_class": "adapter_action_semantics_failure",
            "blockers": blockers,
            "provider_requests_submitted": 0,
            "action_conditioned_video_rollout_generated": False,
        }
        write_json(result_path, result)
        return result

    command = _server_command()
    environment = dict(os.environ)
    environment.update(
        {
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "DO_NOT_TRACK": "1",
            "TRANSFORMERS_NO_ADVISORY_WARNINGS": "1",
        }
    )
    server_log = (output_dir / "vllm_server.log").open("wb")
    process = subprocess.Popen(command, stdout=server_log, stderr=subprocess.STDOUT, env=environment)
    started_at = time.monotonic()
    journal.append(
        {
            "event": "runtime_started",
            "checkpoint": CHECKPOINT,
            "checkpoint_revision": CHECKPOINT_REVISION,
            "command": command,
            "trust_remote_code": False,
            "precision": "bf16",
            "guardrail_exception": "public_nvidia_droid_example_only",
        }
    )
    rollout_records: list[dict[str, Any]] = []
    accepted_request_ids: set[str] = set()
    exact_stack_preflight = "pending"
    failure_class: str | None = None
    try:
        _wait_for_server(process)
        journal.append({"event": "exact_model_server_ready"})
        for row in rows:
            request_id = str(row["request_id"])
            condition = str(row["condition"])
            if request_id in accepted_request_ids:
                continue
            action_stream = conditions[condition]
            if canonical_sha256(action_stream["actions"]) != row["action_sha256"]:
                raise ValueError(f"action_hash_mismatch:{request_id}")
            request_artifact = {
                "request_id": request_id,
                "condition": condition,
                "seed": int(row["seed"]),
                "initial_observation_sha256": input_checks["initial_observation_sha256"],
                "task_instruction": "A robot manipulates an object.",
                "action_sha256": row["action_sha256"],
                "checkpoint": CHECKPOINT,
                "checkpoint_revision": CHECKPOINT_REVISION,
                "accepted_first_valid_only": True,
                "policy_identity_present": False,
            }
            write_json(output_dir / "requests" / f"{request_id}.json", request_artifact)
            attempt = 0
            while True:
                attempt += 1
                journal.append(
                    {
                        "event": "provider_request_started",
                        "request_id": request_id,
                        "attempt": attempt,
                        "retry_kind": "initial" if attempt == 1 else "infrastructure",
                    }
                )
                output_path = output_dir / "videos" / f"{request_id}.mp4"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                request_started = time.monotonic()
                try:
                    response = _submit_rollout(
                        request_row=row,
                        action_stream=action_stream,
                        initial_observation=initial_observation,
                        output_path=output_path,
                    )
                    metrics = _decode_video_metrics(output_path)
                    valid = metrics.get("status") in {"passed", "scientific_failure"}
                    record = {
                        **request_artifact,
                        "attempt": attempt,
                        "elapsed_seconds": time.monotonic() - request_started,
                        "response": response,
                        "deterministic_video_metrics": metrics,
                        "generated_media_valid": valid,
                        "accepted_first_valid": valid,
                        "causal_validity": {"status": "pending_cross_condition_analysis"},
                    }
                    write_json(output_dir / "responses" / f"{request_id}.json", record)
                    journal.append(
                        {
                            "event": "provider_response_recorded",
                            "request_id": request_id,
                            "attempt": attempt,
                            "output_sha256": response["output_sha256"],
                            "validation_status": metrics.get("status"),
                            "accepted_first_valid": valid,
                        }
                    )
                    if valid:
                        accepted_request_ids.add(request_id)
                        rollout_records.append(record)
                        if exact_stack_preflight == "pending":
                            exact_stack_preflight = "passed"
                            journal.append(
                                {
                                    "event": "blackwell_exact_action_stack_preflight_passed",
                                    "request_id": request_id,
                                }
                            )
                        break
                    raise RuntimeError(f"generated_media_invalid:{request_id}")
                except BaseException as exc:  # noqa: BLE001
                    retryable = _classify_retryable(exc)
                    journal.append(
                        {
                            "event": "provider_request_failed",
                            "request_id": request_id,
                            "attempt": attempt,
                            "error_type": type(exc).__name__,
                            "retryable_infrastructure_failure": retryable,
                        }
                    )
                    if retryable and attempt <= INFRASTRUCTURE_RETRY_LIMIT:
                        continue
                    failure_class = (
                        "infrastructure_failure" if retryable else "wam_scientific_or_runtime_failure"
                    )
                    raise
        hashes = [row["response"]["output_sha256"] for row in rollout_records]
        duplicate_output_hashes = len(hashes) - len(set(hashes))
        runtime_status = "completed" if len(rollout_records) == 10 else "blocked"
        result = {
            "schema_version": "policy_ranking_successor_cosmos_runtime.v1",
            "experiment_id": EXPERIMENT_ID,
            "status": runtime_status,
            "failure_class": failure_class,
            "checkpoint": CHECKPOINT,
            "checkpoint_revision": CHECKPOINT_REVISION,
            "precision": "bf16",
            "exact_action_conditioned_stack_preflight": exact_stack_preflight,
            "request_count_frozen": 10,
            "provider_requests_submitted_valid": len(rollout_records),
            "accepted_first_valid_count": len(accepted_request_ids),
            "output_duplicate_hash_count": duplicate_output_hashes,
            "action_conditioned_video_rollout_generated": len(rollout_records) > 0,
            "complete_action_condition_seed_matrix_generated": len(rollout_records) == 10,
            "runtime_seconds": time.monotonic() - started_at,
            "immutable_request_journal_tail_sha256": journal.previous,
            "guardrail_exception": "public_nvidia_droid_example_only",
            "causal_validity": "pending_deterministic_cross_condition_adjudication",
            "evaluator_eligible": False,
            "claims": {
                "runtime": True,
                "generated_media": len(rollout_records) > 0,
                "wam_causal_validity": False,
                "evaluator_validity": False,
                "ranking_fidelity": False,
                "physical_performance": False,
            },
        }
        write_json(result_path, result)
        return result
    except BaseException as exc:  # noqa: BLE001
        result = {
            "schema_version": "policy_ranking_successor_cosmos_runtime.v1",
            "experiment_id": EXPERIMENT_ID,
            "status": "blocked",
            "failure_class": failure_class or "infrastructure_failure",
            "blockers": [f"{type(exc).__name__}:{str(exc)[:300]}"],
            "exact_action_conditioned_stack_preflight": exact_stack_preflight,
            "provider_requests_submitted_valid": len(rollout_records),
            "action_conditioned_video_rollout_generated": len(rollout_records) > 0,
            "runtime_seconds": time.monotonic() - started_at,
            "immutable_request_journal_tail_sha256": journal.previous,
            "evaluator_eligible": False,
        }
        write_json(result_path, result)
        return result
    finally:
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=30)
        server_log.close()


if __name__ == "__main__":
    outcome = run()
    print(json.dumps({"status": outcome.get("status")}, sort_keys=True))
    raise SystemExit(0 if outcome.get("status") == "completed" else 2)
