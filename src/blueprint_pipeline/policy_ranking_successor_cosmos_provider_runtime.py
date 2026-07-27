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
PIPELINE_CLASS = "Cosmos3OmniDiffusersPipeline"
VLLM_OMNI_SOURCE_REVISION = "9c1b7504b178afcf541867c1a2d30db48c69cda8"
RAW_ACTION_DIM = 10
ACTION_CHUNK_SIZE = 16
ACTION_SPACE = "midtrain"
CANARY_INFERENCE_STEPS = 4
PUBLICATION_INFERENCE_STEPS = 30
EXPECTED_CONDITIONS = ("recorded", "zero", "shuffled", "reversed", "policy_swapped")
EXPECTED_SEEDS = (0, 1)
QUALIFICATION_CANARY_REQUEST_COUNT = 2
SCIENTIFIC_MATRIX_REQUEST_COUNT = 10
TOTAL_INITIAL_GENERATION_REQUEST_COUNT = (
    QUALIFICATION_CANARY_REQUEST_COUNT + SCIENTIFIC_MATRIX_REQUEST_COUNT
)


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
    streams = probe_payload.get("streams") if isinstance(probe_payload, dict) else None
    stream = streams[0] if isinstance(streams, list) and streams else {}
    width = int(stream.get("width") or 0)
    height = int(stream.get("height") or 0)
    declared_frames = int(stream.get("nb_frames") or 0)
    structural_blockers: list[str] = []
    if (width, height) != (640, 540):
        structural_blockers.append(f"unexpected_video_dimensions:{width}x{height}")
    if declared_frames != 17 or len(frames) != 17:
        structural_blockers.append(
            f"unexpected_video_frame_count:declared={declared_frames}:decoded={len(frames)}"
        )
    blank = bool(float(frame_stds.max(initial=0.0)) < 2.0)
    static = bool(float(temporal.max(initial=0.0)) < 0.5)
    return {
        "status": (
            "blocked"
            if structural_blockers
            else ("passed" if not blank and not static else "scientific_failure")
        ),
        "blockers": structural_blockers,
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
        "--model-class-name",
        PIPELINE_CLASS,
        "--host",
        "127.0.0.1",
        "--port",
        str(SERVER_PORT),
        "--init-timeout",
        "1800",
        "--dtype",
        "bfloat16",
        "--no-guardrails",
    ]


def _run_cuda_preflight() -> dict[str, Any]:
    commands = {
        "nvidia_smi": [
            "nvidia-smi",
            "--query-gpu=timestamp,name,uuid,driver_version,memory.total,memory.free,memory.used",
            "--format=csv,noheader,nounits",
        ],
        "container_cuda": ["nvcc", "--version"],
    }
    command_results: dict[str, Any] = {}
    for name, command in commands.items():
        started = time.time()
        completed = subprocess.run(
            command, check=False, capture_output=True, text=True, timeout=60
        )
        command_results[name] = {
            "command": command,
            "started_at_epoch": started,
            "finished_at_epoch": time.time(),
            "exit_code": completed.returncode,
            "stdout": completed.stdout[-12000:],
            "stderr": completed.stderr[-12000:],
        }
    import torch
    import vllm
    import vllm_omni
    from vllm_omni.diffusion.models.cosmos3.pipeline_cosmos3 import (
        Cosmos3OmniDiffusersPipeline,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("torch_cuda_unavailable")
    capability = tuple(int(value) for value in torch.cuda.get_device_capability())
    if capability != (12, 0):
        raise RuntimeError(f"unexpected_cuda_capability:{capability}")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("cuda_bf16_unsupported")
    torch.cuda.reset_peak_memory_stats()
    left = torch.randn((256, 256), device="cuda", dtype=torch.bfloat16)
    right = torch.randn((256, 256), device="cuda", dtype=torch.bfloat16)
    product = left @ right
    torch.cuda.synchronize()
    if product.dtype != torch.bfloat16 or not bool(torch.isfinite(product).all().item()):
        raise RuntimeError("bf16_cuda_matrix_operation_invalid")
    memory = torch.cuda.mem_get_info()
    return {
        "status": "passed",
        "commands": command_results,
        "torch_version": torch.__version__,
        "vllm_version": getattr(vllm, "__version__", "unknown"),
        "vllm_omni_version": getattr(vllm_omni, "__version__", "unknown"),
        "vllm_omni_source_revision": VLLM_OMNI_SOURCE_REVISION,
        "pipeline_import": Cosmos3OmniDiffusersPipeline.__name__,
        "torch_cuda_available": True,
        "cuda_device_capability": list(capability),
        "cuda_bf16_supported": True,
        "bf16_cuda_matrix_operation": True,
        "bf16_result_shape": list(product.shape),
        "gpu_memory_bytes": {
            "free": int(memory[0]),
            "total": int(memory[1]),
            "allocated": int(torch.cuda.memory_allocated()),
            "reserved": int(torch.cuda.memory_reserved()),
            "peak_allocated": int(torch.cuda.max_memory_allocated()),
            "peak_reserved": int(torch.cuda.max_memory_reserved()),
        },
    }


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


def _serialize_rollout_request(
    *,
    request_row: dict[str, Any],
    action_stream: dict[str, Any],
    num_inference_steps: int,
) -> dict[str, Any]:
    actions = action_stream.get("actions")
    if not isinstance(actions, list) or len(actions) != ACTION_CHUNK_SIZE:
        raise ValueError("action_chunk_row_count_invalid")
    if any(not isinstance(row, list) or len(row) != RAW_ACTION_DIM for row in actions):
        raise ValueError("action_chunk_raw_dimension_invalid")
    extra_params = {
        "action_mode": "forward_dynamics",
        "domain_name": "droid_lerobot",
        "raw_action_dim": RAW_ACTION_DIM,
        "action_chunk_size": ACTION_CHUNK_SIZE,
        "action_space": ACTION_SPACE,
        "image_size": 480,
        "view_point": "concat_view",
        "action": actions,
        # The pinned NVIDIA DROID example disables guardrails. This bundle is
        # restricted to that public robotics sample and records the exception.
        "guardrails": False,
    }
    return {
        "model": CHECKPOINT,
        "prompt": "A robot manipulates an object.",
        "num_frames": "17",
        "fps": "15",
        "size": "640x540",
        "num_inference_steps": str(num_inference_steps),
        "guidance_scale": "1.0",
        "flow_shift": "10.0",
        "seed": str(request_row["seed"]),
        "extra_params": json.dumps(extra_params, separators=(",", ":")),
    }


def _serialize_blueprint_wrapper_request(
    *,
    request_row: dict[str, Any],
    action_stream: dict[str, Any],
    num_inference_steps: int,
) -> dict[str, Any]:
    """Serialize the Blueprint wrapper contract without calling the direct serializer."""
    actions = action_stream.get("actions")
    if not isinstance(actions, list) or len(actions) != ACTION_CHUNK_SIZE:
        raise ValueError("action_chunk_row_count_invalid")
    if any(not isinstance(row, list) or len(row) != RAW_ACTION_DIM for row in actions):
        raise ValueError("action_chunk_raw_dimension_invalid")
    wrapper_extra_params = {
        "action_mode": "forward_dynamics",
        "domain_name": "droid_lerobot",
        "raw_action_dim": 10,
        "action_chunk_size": 16,
        "action_space": "midtrain",
        "image_size": 480,
        "view_point": "concat_view",
        "action": actions,
        "guardrails": False,
    }
    return {
        "model": "nvidia/Cosmos3-Nano",
        "prompt": "A robot manipulates an object.",
        "num_frames": "17",
        "fps": "15",
        "size": "640x540",
        "num_inference_steps": str(num_inference_steps),
        "guidance_scale": "1.0",
        "flow_shift": "10.0",
        "seed": str(request_row["seed"]),
        "extra_params": json.dumps(wrapper_extra_params, separators=(",", ":")),
    }


def _submit_rollout(
    *,
    serialized_request: dict[str, Any],
    initial_observation: Path,
    output_path: Path,
) -> dict[str, Any]:
    with initial_observation.open("rb") as image_file:
        response = requests.post(
            f"{SERVER_BASE_URL}/v1/videos/sync",
            headers={"Accept": "video/mp4"},
            data=serialized_request,
            files={"input_reference": (initial_observation.name, image_file, "image/png")},
            timeout=REQUEST_POLL_TIMEOUT_SECONDS,
        )
    response.raise_for_status()
    output_path.write_bytes(response.content)
    if not response.content:
        raise RuntimeError("vllm_sync_video_response_empty")
    return {
        "endpoint": "/v1/videos/sync",
        "http_status_code": response.status_code,
        "content_type": response.headers.get("content-type"),
        "serialized_request_sha256": canonical_sha256(serialized_request),
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
    if observed_pairs != expected_pairs or len(rows) != SCIENTIFIC_MATRIX_REQUEST_COUNT:
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

    try:
        cuda_preflight = _run_cuda_preflight()
    except Exception as exc:
        cuda_preflight = {
            "status": "blocked",
            "failure_class": "cuda_driver_or_runtime_failure",
            "blockers": [f"{type(exc).__name__}:{str(exc)[:300]}"],
        }
        write_json(output_dir / "blackwell_cuda_canary.json", cuda_preflight)
        result = {
            "schema_version": "policy_ranking_successor_cosmos_runtime.v1",
            "experiment_id": EXPERIMENT_ID,
            "status": "blocked",
            "failure_class": "cuda_driver_or_runtime_failure",
            "blockers": cuda_preflight["blockers"],
            "provider_requests_submitted_valid": 0,
            "action_conditioned_video_rollout_generated": False,
            "evaluator_eligible": False,
        }
        write_json(result_path, result)
        return result
    write_json(output_dir / "blackwell_cuda_canary.json", cuda_preflight)
    journal.append(
        {
            "event": "blackwell_cuda_canary_passed",
            "cuda_device_capability": cuda_preflight["cuda_device_capability"],
            "bf16_cuda_matrix_operation": True,
        }
    )
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
    server_identity = {
        "pid": process.pid,
        "server_port": SERVER_PORT,
        "pipeline_class": PIPELINE_CLASS,
        "checkpoint": CHECKPOINT,
        "checkpoint_revision": CHECKPOINT_REVISION,
    }
    started_at = time.monotonic()
    journal.append(
        {
            "event": "runtime_started",
            "checkpoint": CHECKPOINT,
            "checkpoint_revision": CHECKPOINT_REVISION,
            "command": command,
            "trust_remote_code": False,
            "precision": "bf16",
            "server_identity": server_identity,
            "guardrail_exception": "public_nvidia_droid_example_only",
        }
    )
    rollout_records: list[dict[str, Any]] = []
    accepted_request_ids: set[str] = set()
    exact_stack_preflight = "pending"
    failure_class: str | None = None
    provider_generation_requests_attempted = 0
    qualification_canary_responses_valid = 0
    try:
        _wait_for_server(process)
        model_load_seconds = time.monotonic() - started_at
        health = requests.get(f"{SERVER_BASE_URL}/health", timeout=10)
        health.raise_for_status()
        journal.append(
            {
                "event": "exact_model_server_ready",
                "server_identity": server_identity,
                "model_load_seconds": model_load_seconds,
                "health_status_code": health.status_code,
            }
        )
        canary_row = next(
            row for row in rows if row.get("condition") == "recorded" and row.get("seed") == 0
        )
        canary_action = conditions["recorded"]
        direct_request = _serialize_rollout_request(
            request_row=canary_row,
            action_stream=canary_action,
            num_inference_steps=CANARY_INFERENCE_STEPS,
        )
        wrapper_request = _serialize_blueprint_wrapper_request(
            request_row=canary_row,
            action_stream=canary_action,
            num_inference_steps=CANARY_INFERENCE_STEPS,
        )
        if direct_request != wrapper_request:
            raise RuntimeError("blueprint_wrapper_direct_request_mismatch")
        canary_records: dict[str, Any] = {}
        for canary_kind, serialized_request in (
            ("direct", direct_request),
            ("blueprint_wrapper", wrapper_request),
        ):
            canary_path = output_dir / "canary" / f"{canary_kind}.mp4"
            canary_path.parent.mkdir(parents=True, exist_ok=True)
            canary_started = time.monotonic()
            provider_generation_requests_attempted += 1
            response = _submit_rollout(
                serialized_request=serialized_request,
                initial_observation=initial_observation,
                output_path=canary_path,
            )
            metrics = _decode_video_metrics(canary_path)
            if metrics.get("status") not in {"passed", "scientific_failure"}:
                raise RuntimeError(f"{canary_kind}_video_validation_failed")
            if process.poll() is not None:
                raise RuntimeError(f"server_exited_during_{canary_kind}_canary")
            qualification_canary_responses_valid += 1
            canary_records[canary_kind] = {
                "request": serialized_request,
                "request_sha256": canonical_sha256(serialized_request),
                "response": response,
                "metrics": metrics,
                "elapsed_seconds": time.monotonic() - canary_started,
                "server_identity": server_identity,
            }
            write_json(output_dir / "canary" / f"{canary_kind}.json", canary_records[canary_kind])
            journal.append(
                {
                    "event": f"{canary_kind}_canary_passed",
                    "request_sha256": canonical_sha256(serialized_request),
                    "output_sha256": response["output_sha256"],
                    "server_pid": process.pid,
                }
            )
        write_json(
            output_dir / "canary" / "same_process_handoff.json",
            {
                "status": "passed",
                "request_payloads_equal": True,
                "request_sha256": canonical_sha256(direct_request),
                "server_identity_before_and_after": server_identity,
                "server_process_unchanged": process.poll() is None,
                "model_reloaded": False,
            },
        )
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
                    serialized_request = _serialize_blueprint_wrapper_request(
                        request_row=row,
                        action_stream=action_stream,
                        num_inference_steps=PUBLICATION_INFERENCE_STEPS,
                    )
                    provider_generation_requests_attempted += 1
                    response = _submit_rollout(
                        serialized_request=serialized_request,
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
                except Exception as exc:
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
        runtime_status = (
            "completed"
            if len(rollout_records) == SCIENTIFIC_MATRIX_REQUEST_COUNT
            else "blocked"
        )
        result = {
            "schema_version": "policy_ranking_successor_cosmos_runtime.v1",
            "experiment_id": EXPERIMENT_ID,
            "status": runtime_status,
            "failure_class": failure_class,
            "checkpoint": CHECKPOINT,
            "checkpoint_revision": CHECKPOINT_REVISION,
            "precision": "bf16",
            "pipeline_class": PIPELINE_CLASS,
            "vllm_omni_source_revision": VLLM_OMNI_SOURCE_REVISION,
            "server_identity": server_identity,
            "model_load_seconds": model_load_seconds,
            "direct_canary_passed": True,
            "blueprint_wrapper_same_process_canary_passed": True,
            "exact_action_conditioned_stack_preflight": exact_stack_preflight,
            "qualification_canary_request_count_frozen": QUALIFICATION_CANARY_REQUEST_COUNT,
            "qualification_canary_responses_valid": qualification_canary_responses_valid,
            "scientific_matrix_request_count_frozen": SCIENTIFIC_MATRIX_REQUEST_COUNT,
            "total_initial_generation_request_count_frozen": (
                TOTAL_INITIAL_GENERATION_REQUEST_COUNT
            ),
            "provider_generation_requests_attempted_total": (
                provider_generation_requests_attempted
            ),
            "provider_scientific_matrix_responses_valid": len(rollout_records),
            "accepted_first_valid_count": len(accepted_request_ids),
            "output_duplicate_hash_count": duplicate_output_hashes,
            "action_conditioned_video_rollout_generated": bool(rollout_records),
            "complete_action_condition_seed_matrix_generated": (
                len(rollout_records) == SCIENTIFIC_MATRIX_REQUEST_COUNT
            ),
            "runtime_seconds": time.monotonic() - started_at,
            "immutable_request_journal_tail_sha256": journal.previous,
            "guardrail_exception": "public_nvidia_droid_example_only",
            "causal_validity": "pending_deterministic_cross_condition_adjudication",
            "evaluator_eligible": False,
            "claims": {
                "runtime": True,
                "generated_media": bool(rollout_records),
                "wam_causal_validity": False,
                "evaluator_validity": False,
                "ranking_fidelity": False,
                "physical_performance": False,
            },
        }
        write_json(result_path, result)
        return result
    except Exception as exc:
        result = {
            "schema_version": "policy_ranking_successor_cosmos_runtime.v1",
            "experiment_id": EXPERIMENT_ID,
            "status": "blocked",
            "failure_class": failure_class or "infrastructure_failure",
            "blockers": [f"{type(exc).__name__}:{str(exc)[:300]}"],
            "exact_action_conditioned_stack_preflight": exact_stack_preflight,
            "qualification_canary_responses_valid": qualification_canary_responses_valid,
            "provider_generation_requests_attempted_total": (
                provider_generation_requests_attempted
            ),
            "provider_scientific_matrix_responses_valid": len(rollout_records),
            "action_conditioned_video_rollout_generated": bool(rollout_records),
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
