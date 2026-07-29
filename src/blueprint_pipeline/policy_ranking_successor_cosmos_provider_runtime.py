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
import signal
import shutil
import subprocess
import time
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import requests


CHECKPOINT = "nvidia/Cosmos3-Nano"
CHECKPOINT_REVISION = "411f42a8fdfb8c5b2583cb8786e0938f49796eaa"
EXPERIMENT_ID = os.environ.get(
    "BLUEPRINT_COSMOS_EXPERIMENT_ID", "policy_ranking_successor_experiment_20260727"
)
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
PHASE_B_EXPECTED_CONDITIONS = (*EXPECTED_CONDITIONS, "shifted")
EXPECTED_SEEDS = (0, 1)
QUALIFICATION_CANARY_REQUEST_COUNT = 2
SCIENTIFIC_MATRIX_REQUEST_COUNT = 10
TOTAL_INITIAL_GENERATION_REQUEST_COUNT = (
    QUALIFICATION_CANARY_REQUEST_COUNT + SCIENTIFIC_MATRIX_REQUEST_COUNT
)
RETAIN_SERVER_ENV = "BLUEPRINT_RETAIN_COSMOS_SERVER"
RETAINED_ROOT_ENV = "BLUEPRINT_COSMOS_RETAINED_ROOT"
DEFAULT_RETAINED_ROOT = "/workspace/blueprint_vast_probe/cosmos3_retained"
EXPECTED_VIDEO_WIDTH = 640
EXPECTED_VIDEO_HEIGHT = 544
POSITIVE_CONTROL_DIRECTORY = "cosmos3_positive_control"
POSITIVE_CONTROL_REQUEST_COUNT = 4
DROID_REFERENCE_DIRECTORY = "cosmos3_droid_reference"
DROID_REFERENCE_SCHEMA_VERSION = "policy_ranking_cosmos3_official_droid_reference_canary.v2"


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


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _process_start_ticks(pid: int) -> str | None:
    try:
        fields = Path(f"/proc/{int(pid)}/stat").read_text(encoding="utf-8").split()
    except (OSError, UnicodeError, ValueError):
        return None
    return fields[21] if len(fields) > 21 else None


class ServerProcess:
    """Small Popen-compatible handle that can adopt a detached retained process."""

    def __init__(
        self,
        *,
        pid: int,
        start_ticks: str | None,
        process: subprocess.Popen[bytes] | None = None,
    ) -> None:
        self.pid = int(pid)
        self.start_ticks = start_ticks
        self.process = process

    def poll(self) -> int | None:
        if self.process is not None:
            return self.process.poll()
        observed = _process_start_ticks(self.pid)
        if observed is None or (self.start_ticks is not None and observed != self.start_ticks):
            return 1
        return None

    def terminate(self) -> None:
        if self.process is not None:
            self.process.terminate()
            return
        os.killpg(self.pid, signal.SIGTERM)

    def kill(self) -> None:
        if self.process is not None:
            self.process.kill()
            return
        os.killpg(self.pid, signal.SIGKILL)

    def wait(self, timeout: float) -> int:
        if self.process is not None:
            return self.process.wait(timeout=timeout)
        deadline = time.monotonic() + timeout
        while self.poll() is None and time.monotonic() < deadline:
            time.sleep(0.1)
        if self.poll() is None:
            raise subprocess.TimeoutExpired("retained_cosmos_server", timeout)
        return 0


def _retained_server_identity_valid(identity: dict[str, Any]) -> bool:
    required = {
        "checkpoint": CHECKPOINT,
        "checkpoint_revision": CHECKPOINT_REVISION,
        "pipeline_class": PIPELINE_CLASS,
        "server_port": SERVER_PORT,
    }
    if any(identity.get(key) != value for key, value in required.items()):
        return False
    try:
        pid = int(identity.get("pid"))
    except (TypeError, ValueError):
        return False
    start_ticks = identity.get("process_start_ticks")
    return bool(start_ticks and _process_start_ticks(pid) == start_ticks)


def _acquire_server(
    *, output_dir: Path, environment: dict[str, str]
) -> tuple[ServerProcess, dict[str, Any], Any, bool]:
    retain = _env_truthy(RETAIN_SERVER_ENV)
    retained_root = Path(os.environ.get(RETAINED_ROOT_ENV, DEFAULT_RETAINED_ROOT)).resolve()
    identity_path = retained_root / "server_identity.json"
    if retain and identity_path.is_file():
        try:
            identity = json.loads(identity_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            identity = {}
        if isinstance(identity, dict) and _retained_server_identity_valid(identity):
            try:
                health = requests.get(f"{SERVER_BASE_URL}/health", timeout=10)
                if health.ok:
                    handle = ServerProcess(
                        pid=int(identity["pid"]),
                        start_ticks=str(identity["process_start_ticks"]),
                    )
                    return handle, {**identity, "reused_retained_server": True}, None, True
            except requests.RequestException:
                pass
    command = _server_command()
    log_root = retained_root if retain else output_dir
    log_root.mkdir(parents=True, exist_ok=True)
    server_log = (log_root / "vllm_server.log").open("ab" if retain else "wb")
    process = subprocess.Popen(
        command,
        stdout=server_log,
        stderr=subprocess.STDOUT,
        env=environment,
        start_new_session=retain,
    )
    start_ticks = _process_start_ticks(process.pid)
    identity = {
        "pid": process.pid,
        "process_start_ticks": start_ticks,
        "server_port": SERVER_PORT,
        "pipeline_class": PIPELINE_CLASS,
        "checkpoint": CHECKPOINT,
        "checkpoint_revision": CHECKPOINT_REVISION,
        "command_sha256": canonical_sha256(command),
        "started_at_epoch": time.time(),
        "reused_retained_server": False,
        "retention_enabled": retain,
    }
    if retain:
        write_json(identity_path, identity)
    return (
        ServerProcess(pid=process.pid, start_ticks=start_ticks, process=process),
        identity,
        server_log,
        False,
    )


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


def _decode_video_metrics(
    path: Path,
    *,
    expected_width: int = EXPECTED_VIDEO_WIDTH,
    expected_height: int = EXPECTED_VIDEO_HEIGHT,
    expected_frames: int = 17,
    expected_fps: float = 15.0,
) -> dict[str, Any]:
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
            "stream=width,height,nb_frames,r_frame_rate:format=duration",
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
            f"scale=64:{max(1, round(64 * expected_height / expected_width))},format=gray",
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
    decoded_height = max(1, round(64 * expected_height / expected_width))
    frame_size = 64 * decoded_height
    if decoded.returncode != 0 or len(decoded.stdout) < frame_size:
        return {"status": "blocked", "blockers": ["video_decode_failed"]}
    usable = len(decoded.stdout) // frame_size * frame_size
    frames = np.frombuffer(decoded.stdout[:usable], dtype=np.uint8).reshape(-1, decoded_height, 64)
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
    try:
        frame_rate = float(Fraction(str(stream.get("r_frame_rate") or "0/1")))
    except (ValueError, ZeroDivisionError):
        frame_rate = 0.0
    format_payload = probe_payload.get("format") if isinstance(probe_payload, dict) else None
    try:
        duration_seconds = float(
            format_payload.get("duration") if isinstance(format_payload, dict) else 0.0
        )
    except (TypeError, ValueError):
        duration_seconds = 0.0
    structural_blockers: list[str] = []
    if (width, height) != (expected_width, expected_height):
        structural_blockers.append(f"unexpected_video_dimensions:{width}x{height}")
    if declared_frames != expected_frames or len(frames) != expected_frames:
        structural_blockers.append(
            f"unexpected_video_frame_count:declared={declared_frames}:decoded={len(frames)}"
        )
    if not np.isclose(frame_rate, expected_fps, rtol=0.0, atol=1e-6):
        structural_blockers.append(f"unexpected_video_frame_rate:{frame_rate}")
    expected_duration_seconds = expected_frames / expected_fps
    if not np.isclose(duration_seconds, expected_duration_seconds, rtol=0.0, atol=0.05):
        structural_blockers.append(f"unexpected_video_duration:{duration_seconds}")
    blank = bool(float(frame_stds.max(initial=0.0)) < 2.0)
    static = bool(float(temporal.max(initial=0.0)) < 0.5)
    return {
        # Four-step canaries are structural only. Motion remains a separate
        # scientific observation and cannot make a decodable response fail its
        # structural contract.
        "status": "blocked" if structural_blockers else "passed",
        "structural_status": "blocked" if structural_blockers else "passed",
        "motion_status": "failed" if blank or static else "passed",
        "blockers": structural_blockers,
        "frame_count_decoded": int(len(frames)),
        "frame_rate": frame_rate,
        "duration_seconds": duration_seconds,
        "expected_duration_seconds": expected_duration_seconds,
        "mean_luma_min": float(frame_means.min(initial=0.0)),
        "mean_luma_max": float(frame_means.max(initial=0.0)),
        "spatial_std_max": float(frame_stds.max(initial=0.0)),
        "temporal_absolute_difference_mean": float(temporal.mean()) if len(temporal) else 0.0,
        "temporal_absolute_difference_max": float(temporal.max(initial=0.0)),
        "first_to_last_absolute_difference_mean": float(
            np.abs(frames[-1].astype(np.float32) - frames[0].astype(np.float32)).mean()
        ),
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


def _server_environment() -> dict[str, str]:
    """Return the fail-closed environment shared by every Cosmos server launch.

    The Xet transfer client can leave a healthy paid worker indefinitely waiting
    on partial model shards without exercising the GPU.  Use the Hub's ordinary
    HTTP download path instead.  Keep this in the runtime module rather than an
    allocation-specific launch command so retained refreshes and future runs
    receive the same protection.
    """

    environment = dict(os.environ)
    environment.update(
        {
            "HF_HUB_DISABLE_XET": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "DO_NOT_TRACK": "1",
            "TRANSFORMERS_NO_ADVISORY_WARNINGS": "1",
        }
    )
    return environment


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
        completed = subprocess.run(command, check=False, capture_output=True, text=True, timeout=60)
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


def _wait_for_server(process: ServerProcess) -> None:
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
    task_instruction: str,
) -> dict[str, Any]:
    task_instruction = str(task_instruction).strip()
    if not task_instruction:
        raise ValueError("task_specific_instruction_missing")
    if task_instruction == "A robot manipulates an object.":
        raise ValueError("generic_robot_manipulation_prompt_forbidden")
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
        "prompt": task_instruction,
        "num_frames": "17",
        "fps": "15",
        "size": f"{EXPECTED_VIDEO_WIDTH}x{EXPECTED_VIDEO_HEIGHT}",
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
    task_instruction: str,
) -> dict[str, Any]:
    """Serialize the Blueprint wrapper contract without calling the direct serializer."""
    task_instruction = str(task_instruction).strip()
    if not task_instruction:
        raise ValueError("task_specific_instruction_missing")
    if task_instruction == "A robot manipulates an object.":
        raise ValueError("generic_robot_manipulation_prompt_forbidden")
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
        "prompt": task_instruction,
        "num_frames": "17",
        "fps": "15",
        "size": f"{EXPECTED_VIDEO_WIDTH}x{EXPECTED_VIDEO_HEIGHT}",
        "num_inference_steps": str(num_inference_steps),
        "guidance_scale": "1.0",
        "flow_shift": "10.0",
        "seed": str(request_row["seed"]),
        "extra_params": json.dumps(wrapper_extra_params, separators=(",", ":")),
    }


def _serialize_positive_control_request(
    *, action_chunk: list[list[float]], action_spec: dict[str, Any]
) -> dict[str, Any]:
    action_chunk_size = int(action_spec.get("action_chunk_size") or 0)
    if action_chunk_size != 16 or len(action_chunk) != action_chunk_size:
        raise ValueError("positive_control_action_chunk_length_invalid")
    if any(not isinstance(row, list) or len(row) != 29 for row in action_chunk):
        raise ValueError("positive_control_action_dimension_invalid")
    prompt = str(action_spec.get("prompt") or "").strip()
    if not prompt:
        raise ValueError("positive_control_prompt_missing")
    extra_params = {
        "action_mode": "forward_dynamics",
        "domain_name": str(action_spec.get("domain_name") or "agibotworld"),
        "action_chunk_size": action_chunk_size,
        "image_size": int(action_spec.get("image_size") or 480),
        "view_point": str(action_spec.get("view_point") or "concat_view"),
        "action": action_chunk,
        # The frozen public-robotics runtime starts with --no-guardrails.  This
        # is recorded as the sole model-card request deviation; the visual and
        # action conditioning path is otherwise the published example.
        "guardrails": False,
    }
    return {
        "model": CHECKPOINT,
        "prompt": prompt,
        "num_frames": "17",
        "fps": str(int(action_spec.get("fps") or 10)),
        "size": "640x720",
        "num_inference_steps": "30",
        "guidance_scale": "1.0",
        "flow_shift": "10.0",
        "seed": "0",
        "extra_params": json.dumps(extra_params, separators=(",", ":")),
    }


def _extract_last_frame(video_path: Path, output_path: Path, *, frame_index: int = 16) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("positive_control_ffmpeg_missing")
    completed = subprocess.run(
        [
            ffmpeg,
            "-v",
            "error",
            "-i",
            str(video_path),
            "-vf",
            f"select=eq(n\\,{frame_index})",
            "-vsync",
            "0",
            "-frames:v",
            "1",
            "-y",
            str(output_path),
        ],
        check=False,
        capture_output=True,
        timeout=120,
    )
    if completed.returncode != 0 or not output_path.is_file():
        raise RuntimeError("positive_control_last_frame_extract_failed")


def _run_positive_control(
    *,
    input_dir: Path,
    output_dir: Path,
    journal: HashChainedJournal,
    attempt_counter: list[int] | None = None,
) -> dict[str, Any] | None:
    control_dir = input_dir.parent / POSITIVE_CONTROL_DIRECTORY
    manifest_path = control_dir / "manifest.json"
    if not manifest_path.is_file():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_sha256 = canonical_sha256(
        {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    )
    if manifest.get("manifest_sha256") != manifest_sha256:
        raise ValueError("positive_control_manifest_sha256_mismatch")
    if (
        manifest.get("schema_version") != "policy_ranking_cosmos_official_positive_control.v1"
        or manifest.get("request_count") != POSITIVE_CONTROL_REQUEST_COUNT
    ):
        raise ValueError("positive_control_manifest_contract_invalid")
    action_path = control_dir / "action_chunks.json"
    first_frame = control_dir / "first_frame.png"
    reference_output = control_dir / "reference_output.mp4"
    files = {
        "action_chunks": action_path,
        "first_frame": first_frame,
        "reference_output": reference_output,
    }
    blockers: list[str] = []
    for name, path in files.items():
        expected = str((manifest.get("asset_sha256") or {}).get(name) or "")
        if not path.is_file() or not expected or sha256_file(path) != expected:
            blockers.append(f"positive_control_asset_hash_mismatch:{name}")
    if blockers:
        raise ValueError(";".join(blockers))
    action_spec = json.loads(action_path.read_text(encoding="utf-8"))
    chunks = action_spec.get("action_chunks")
    if not isinstance(chunks, list) or len(chunks) != POSITIVE_CONTROL_REQUEST_COUNT:
        raise ValueError("positive_control_action_chunk_count_invalid")
    reference_metrics = _decode_video_metrics(
        reference_output,
        expected_width=640,
        expected_height=720,
        expected_frames=64,
        expected_fps=10.0,
    )
    if reference_metrics.get("structural_status") != "passed":
        raise ValueError("positive_control_reference_media_invalid")
    gates = manifest.get("frozen_gates") if isinstance(manifest.get("frozen_gates"), dict) else {}
    required_gates = {
        "chunk_temporal_absolute_difference_mean_minimum",
        "chunk_first_to_last_absolute_difference_mean_minimum",
        "minimum_dynamic_chunks",
    }
    if set(gates) != required_gates or any(float(gates[key]) <= 0 for key in required_gates):
        raise ValueError("positive_control_frozen_gates_invalid")
    generated_dir = output_dir / "positive_control"
    generated_dir.mkdir(parents=True, exist_ok=True)
    current_frame = first_frame
    records: list[dict[str, Any]] = []
    for chunk_index, action_chunk in enumerate(chunks):
        request = _serialize_positive_control_request(
            action_chunk=action_chunk,
            action_spec=action_spec,
        )
        output_path = generated_dir / f"chunk_{chunk_index:02d}.mp4"
        started = time.monotonic()
        if attempt_counter is not None:
            attempt_counter[0] += 1
        response = _submit_rollout(
            serialized_request=request,
            initial_observation=current_frame,
            output_path=output_path,
        )
        metrics = _decode_video_metrics(
            output_path,
            expected_width=640,
            expected_height=720,
            expected_frames=17,
            expected_fps=10.0,
        )
        next_frame = generated_dir / f"conditioning_{chunk_index + 1:02d}.png"
        _extract_last_frame(output_path, next_frame)
        record = {
            "chunk_index": chunk_index,
            "request": request,
            "request_sha256": canonical_sha256(request),
            "response": response,
            "metrics": metrics,
            "elapsed_seconds": time.monotonic() - started,
            "next_conditioning_frame_sha256": sha256_file(next_frame),
        }
        records.append(record)
        write_json(generated_dir / f"chunk_{chunk_index:02d}.json", record)
        journal.append(
            {
                "event": "official_positive_control_chunk_recorded",
                "chunk_index": chunk_index,
                "request_sha256": record["request_sha256"],
                "output_sha256": response["output_sha256"],
            }
        )
        current_frame = next_frame
    structural_pass = all(
        record["metrics"].get("structural_status") == "passed" for record in records
    )
    dynamic_chunks = sum(
        1
        for record in records
        if float(record["metrics"].get("temporal_absolute_difference_mean") or 0.0)
        >= float(gates.get("chunk_temporal_absolute_difference_mean_minimum") or 0.0)
        and float(record["metrics"].get("first_to_last_absolute_difference_mean") or 0.0)
        >= float(gates.get("chunk_first_to_last_absolute_difference_mean_minimum") or 0.0)
    )
    passed = structural_pass and dynamic_chunks >= int(gates.get("minimum_dynamic_chunks") or 4)
    result = {
        "schema_version": "policy_ranking_cosmos_official_positive_control_result.v1",
        "status": "passed" if passed else "failed",
        "model": CHECKPOINT,
        "checkpoint_revision": CHECKPOINT_REVISION,
        "source_manifest_sha256": manifest_sha256,
        "published_reference_metrics": reference_metrics,
        "records": records,
        "structural_pass": structural_pass,
        "dynamic_chunk_count": dynamic_chunks,
        "request_count": len(records),
        "frozen_gates": gates,
        "droid_matrix_admitted": passed,
        "claim_boundary": (
            "A pass proves only that the pinned deployment can reproduce visible "
            "action-conditioned motion on NVIDIA's AgiBotWorld example; it does "
            "not qualify DROID, policy ranking, or physical task success."
        ),
    }
    result["result_sha256"] = canonical_sha256(result)
    write_json(generated_dir / "positive_control_result.json", result)
    return result


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


def _serialize_droid_reference_request(
    *, manifest: dict[str, Any], action_stream: dict[str, Any]
) -> dict[str, Any]:
    """Match NVIDIA's published DROID vLLM-Omni request exactly."""

    request = manifest.get("request_contract")
    if not isinstance(request, dict):
        raise ValueError("official_droid_reference_request_contract_missing")
    actions = action_stream.get("actions")
    if (
        not isinstance(actions, list)
        or len(actions) != 16
        or any(not isinstance(row, list) or len(row) != 10 for row in actions)
    ):
        raise ValueError("official_droid_reference_action_shape_invalid")
    extra = request.get("extra_params")
    expected_extra = {
        "action_mode": "forward_dynamics",
        "domain_name": "droid_lerobot",
        "action_chunk_size": 16,
        "image_size": 480,
        "view_point": "concat_view",
        "guardrails": False,
    }
    if extra != expected_extra:
        raise ValueError("official_droid_reference_extra_params_invalid")
    expected = {
        "model": CHECKPOINT,
        "checkpoint_revision": CHECKPOINT_REVISION,
        "endpoint": "/v1/videos",
        "prompt": " ",
        "num_frames": 17,
        "fps": 15,
        "size": "640x540",
        "num_inference_steps": 30,
        "guidance_scale": 1.0,
        "flow_shift": 10.0,
        "seed": 0,
        "extra_params": expected_extra,
    }
    if request != expected:
        raise ValueError("official_droid_reference_request_contract_changed")
    return {
        "model": CHECKPOINT,
        "prompt": " ",
        "num_frames": "17",
        "fps": "15",
        "size": "640x540",
        "num_inference_steps": "30",
        "guidance_scale": "1.0",
        "flow_shift": "10.0",
        "seed": "0",
        "extra_params": json.dumps({**expected_extra, "action": actions}, separators=(",", ":")),
    }


def _submit_rollout_async(
    *, serialized_request: dict[str, Any], initial_observation: Path, output_path: Path
) -> dict[str, Any]:
    """Submit and collect the structured asynchronous NVIDIA cookbook endpoint."""

    with initial_observation.open("rb") as image_file:
        response = requests.post(
            f"{SERVER_BASE_URL}/v1/videos",
            data=serialized_request,
            files={"input_reference": (initial_observation.name, image_file, "image/png")},
            timeout=120,
        )
    response.raise_for_status()
    initial = response.json()
    request_id = str(initial.get("id") or "") if isinstance(initial, dict) else ""
    if not request_id:
        raise RuntimeError("official_droid_reference_response_id_missing")
    deadline = time.monotonic() + REQUEST_POLL_TIMEOUT_SECONDS
    final: dict[str, Any] = {}
    while time.monotonic() < deadline:
        status_response = requests.get(f"{SERVER_BASE_URL}/v1/videos/{request_id}", timeout=30)
        status_response.raise_for_status()
        payload = status_response.json()
        if not isinstance(payload, dict):
            raise RuntimeError("official_droid_reference_status_not_object")
        final = payload
        status = str(final.get("status") or "")
        if status == "completed":
            break
        if status in {"failed", "cancelled"}:
            raise RuntimeError(f"official_droid_reference_terminal_status:{status}")
        time.sleep(2)
    else:
        raise TimeoutError("official_droid_reference_poll_timeout")
    content = requests.get(f"{SERVER_BASE_URL}/v1/videos/{request_id}/content", timeout=300)
    content.raise_for_status()
    if not content.content:
        raise RuntimeError("official_droid_reference_video_empty")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(content.content)
    return {
        "endpoint": "/v1/videos",
        "provider_response_id": request_id,
        "initial_response": initial,
        "terminal_response": final,
        "terminal_status": final.get("status"),
        "output_sha256": sha256_file(output_path),
        "output_size_bytes": output_path.stat().st_size,
    }


def _run_droid_reference_only(*, runtime_dir: Path, output_dir: Path) -> dict[str, Any]:
    """Run the prospectively frozen official DROID reference and nothing else."""

    control = runtime_dir / DROID_REFERENCE_DIRECTORY
    manifest = json.loads((control / "canary_manifest.json").read_text(encoding="utf-8"))
    recorded_digest = str(manifest.get("manifest_sha256") or "")
    computed_digest = canonical_sha256(
        {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    )
    if not recorded_digest or recorded_digest != computed_digest:
        raise ValueError("official_droid_reference_manifest_sha256_mismatch")
    if manifest.get("schema_version") != DROID_REFERENCE_SCHEMA_VERSION:
        raise ValueError("official_droid_reference_manifest_schema_invalid")
    initial_observation = control / "initial_observation.png"
    actions_path = control / "action_streams.json"
    actions = json.loads(actions_path.read_text(encoding="utf-8"))
    provider_inputs = manifest.get("provider_inputs") or {}
    if sha256_file(initial_observation) != provider_inputs.get("initial_observation_sha256"):
        raise ValueError("official_droid_reference_initial_sha256_mismatch")
    if canonical_sha256(actions) != provider_inputs.get("action_streams_sha256"):
        raise ValueError("official_droid_reference_actions_sha256_mismatch")

    journal = HashChainedJournal(output_dir / "immutable_request_journal.jsonl")
    result_path = output_dir / "wam_runtime_result.json"
    try:
        cuda_preflight = _run_cuda_preflight()
    except Exception as exc:
        result = {
            "schema_version": "policy_ranking_cosmos3_droid_reference_runtime.v1",
            "status": "blocked",
            "failure_class": "cuda_driver_or_runtime_failure",
            "blockers": [f"{type(exc).__name__}:{str(exc)[:300]}"],
            "provider_generation_requests_attempted": 0,
        }
        write_json(result_path, result)
        return result
    write_json(output_dir / "gpu_preflight.json", cuda_preflight)

    process, server_identity, server_log, reused = _acquire_server(
        output_dir=output_dir, environment=_server_environment()
    )
    started = time.monotonic()
    attempted = 0
    records: list[dict[str, Any]] = []
    force_shutdown = False
    try:
        _wait_for_server(process)
        model_load_seconds = 0.0 if reused else time.monotonic() - started
        gates = (manifest.get("frozen_gates") or {}).get("structured_canary") or {}
        expected_width = int(gates.get("output_width") or 0)
        expected_height = int(gates.get("output_height") or 0)
        expected_frames = int(gates.get("output_frames") or 0)
        expected_fps = float(gates.get("output_fps") or 0.0)
        if min(expected_width, expected_height, expected_frames) <= 0 or expected_fps <= 0:
            raise ValueError("official_droid_reference_output_geometry_gate_invalid")
        for name in ("recorded", "no_motion"):
            if name == "no_motion" and not records[0]["gate_passed"]:
                break
            serialized = _serialize_droid_reference_request(
                manifest=manifest, action_stream=actions[name]
            )
            attempted += 1
            output_path = output_dir / "reference_canary" / f"{name}.mp4"
            response = _submit_rollout_async(
                serialized_request=serialized,
                initial_observation=initial_observation,
                output_path=output_path,
            )
            metrics = _decode_video_metrics(
                output_path,
                expected_width=expected_width,
                expected_height=expected_height,
                expected_frames=expected_frames,
                expected_fps=expected_fps,
            )
            motion_pass = float(metrics.get("temporal_absolute_difference_mean") or 0.0) >= float(
                gates.get("temporal_absolute_difference_mean_minimum_gray_0_255") or 0.0
            ) and float(metrics.get("first_to_last_absolute_difference_mean") or 0.0) >= float(
                gates.get("first_to_last_absolute_difference_mean_minimum_gray_0_255") or 0.0
            )
            gate_passed = metrics.get("structural_status") == "passed" and (
                motion_pass if name == "recorded" else True
            )
            record = {
                "name": name,
                "request": serialized,
                "request_sha256": canonical_sha256(serialized),
                "response": response,
                "metrics": metrics,
                "gate_passed": gate_passed,
            }
            records.append(record)
            write_json(output_dir / "reference_canary" / f"{name}.json", record)
            journal.append(
                {
                    "event": "official_droid_reference_response_recorded",
                    "name": name,
                    "provider_response_id": response["provider_response_id"],
                    "output_sha256": response["output_sha256"],
                    "gate_passed": gate_passed,
                }
            )
        recorded_pass = bool(records and records[0]["gate_passed"])
        paired_complete = len(records) == 2 and records[1]["gate_passed"]
        result = {
            "schema_version": "policy_ranking_cosmos3_droid_reference_runtime.v1",
            "status": "completed",
            "experiment_id": manifest.get("experiment_id"),
            "manifest_sha256": recorded_digest,
            "checkpoint": CHECKPOINT,
            "checkpoint_revision": CHECKPOINT_REVISION,
            "server_identity": server_identity,
            "model_load_seconds": model_load_seconds,
            "reused_retained_server": reused,
            "provider_generation_requests_attempted": attempted,
            "records": records,
            "structured_canary_passed": recorded_pass,
            "paired_reference_complete": paired_complete,
            "causal_adjudication": "pending_offline_tier1_and_timing_analysis",
            "untouched_data_admitted": False,
            "evaluator_eligible": False,
            "runtime_seconds": time.monotonic() - started,
            "claim_boundary": manifest.get("claim_boundary"),
        }
        result["result_sha256"] = canonical_sha256(result)
        write_json(result_path, result)
        return result
    except Exception as exc:
        force_shutdown = True
        result = {
            "schema_version": "policy_ranking_cosmos3_droid_reference_runtime.v1",
            "status": "blocked",
            "failure_class": "reference_canary_runtime_or_transport_failure",
            "blockers": [f"{type(exc).__name__}:{str(exc)[:300]}"],
            "provider_generation_requests_attempted": attempted,
            "untouched_data_admitted": False,
        }
        write_json(result_path, result)
        return result
    finally:
        retained = _env_truthy(RETAIN_SERVER_ENV) and not force_shutdown and process.poll() is None
        write_json(
            output_dir / "cosmos_server_retention.json",
            {
                "status": "retained_loaded" if retained else "terminal_shutdown",
                "server_identity": server_identity,
                "process_alive": process.poll() is None,
                "server_remained_loaded": retained,
                "reused_retained_server": reused,
            },
        )
        if not retained:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=30)
        if server_log is not None:
            server_log.close()


def _classify_retryable(exc: BaseException) -> bool:
    return isinstance(
        exc,
        (
            requests.ConnectionError,
            requests.Timeout,
            TimeoutError,
        ),
    )


def _declared_expected_conditions(inventory: dict[str, Any]) -> tuple[str, ...] | None:
    declared = inventory.get("required_conditions")
    if declared is None:
        return EXPECTED_CONDITIONS
    if not isinstance(declared, list):
        return None
    normalized = tuple(str(value) for value in declared)
    if normalized not in {EXPECTED_CONDITIONS, PHASE_B_EXPECTED_CONDITIONS}:
        return None
    return normalized


def _action_conditions_match_frozen_contract(
    value: Any, expected_conditions: tuple[str, ...] = EXPECTED_CONDITIONS
) -> bool:
    return (
        isinstance(value, dict)
        and len(value) == len(expected_conditions)
        and frozenset(value) == frozenset(expected_conditions)
    )


def run() -> dict[str, Any]:
    runtime_dir = Path(__file__).resolve().parent
    output_dir = Path(
        os.environ.get("BLUEPRINT_VAST_PROVIDER_OUTPUT_DIR", runtime_dir.parent / "runtime_output")
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if (runtime_dir / DROID_REFERENCE_DIRECTORY / "canary_manifest.json").is_file():
        return _run_droid_reference_only(runtime_dir=runtime_dir, output_dir=output_dir)

    input_dir = runtime_dir / "cosmos3_input"
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
    if input_checks["initial_observation_sha256"] != inventory.get("initial_observation_sha256"):
        blockers.append("initial_observation_sha256_mismatch")
    task_instruction = str(inventory.get("task_instruction") or "").strip()
    if not task_instruction:
        blockers.append("task_specific_instruction_missing")
    if task_instruction == "A robot manipulates an object.":
        blockers.append("generic_robot_manipulation_prompt_forbidden")
    conditions = action_streams.get("conditions")
    expected_conditions = _declared_expected_conditions(inventory)
    if expected_conditions is None:
        blockers.append("required_conditions_invalid")
        expected_conditions = EXPECTED_CONDITIONS
    if not _action_conditions_match_frozen_contract(conditions, expected_conditions):
        blockers.append("action_stream_condition_order_invalid")
    expected_pairs = {
        (condition, seed) for condition in expected_conditions for seed in EXPECTED_SEEDS
    }
    scientific_matrix_request_count = len(expected_conditions) * len(EXPECTED_SEEDS)
    total_initial_generation_request_count = (
        QUALIFICATION_CANARY_REQUEST_COUNT
        + scientific_matrix_request_count
        + (
            POSITIVE_CONTROL_REQUEST_COUNT
            if (input_dir.parent / POSITIVE_CONTROL_DIRECTORY / "manifest.json").is_file()
            else 0
        )
    )
    rows = inventory.get("requests") if isinstance(inventory.get("requests"), list) else []
    observed_pairs = {(row.get("condition"), row.get("seed")) for row in rows}
    if observed_pairs != expected_pairs or len(rows) != scientific_matrix_request_count:
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
    environment = _server_environment()
    process, server_identity, server_log, reused_retained_server = _acquire_server(
        output_dir=output_dir,
        environment=environment,
    )
    command = _server_command()
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
    positive_control_attempt_counter = [0]
    positive_control: dict[str, Any] | None = None
    force_server_shutdown = False
    try:
        _wait_for_server(process)
        model_load_seconds = 0.0 if reused_retained_server else time.monotonic() - started_at
        health = requests.get(f"{SERVER_BASE_URL}/health", timeout=10)
        health.raise_for_status()
        journal.append(
            {
                "event": "exact_model_server_ready",
                "server_identity": server_identity,
                "model_load_seconds": model_load_seconds,
                "health_status_code": health.status_code,
                "reused_retained_server": reused_retained_server,
            }
        )
        positive_control = _run_positive_control(
            input_dir=input_dir,
            output_dir=output_dir,
            journal=journal,
            attempt_counter=positive_control_attempt_counter,
        )
        provider_generation_requests_attempted += positive_control_attempt_counter[0]
        if positive_control is not None:
            journal.append(
                {
                    "event": "official_positive_control_completed",
                    "status": positive_control["status"],
                    "request_count": positive_control["request_count"],
                    "result_sha256": positive_control["result_sha256"],
                    "droid_matrix_admitted": positive_control["droid_matrix_admitted"],
                }
            )
            if positive_control["status"] != "passed":
                force_server_shutdown = True
                result = {
                    "schema_version": "policy_ranking_successor_cosmos_runtime.v1",
                    "experiment_id": EXPERIMENT_ID,
                    "status": "completed",
                    "failure_class": "official_positive_control_scientific_failure",
                    "checkpoint": CHECKPOINT,
                    "checkpoint_revision": CHECKPOINT_REVISION,
                    "precision": "bf16",
                    "pipeline_class": PIPELINE_CLASS,
                    "vllm_omni_source_revision": VLLM_OMNI_SOURCE_REVISION,
                    "server_identity": server_identity,
                    "model_load_seconds": model_load_seconds,
                    "reused_retained_server": reused_retained_server,
                    "official_positive_control": positive_control,
                    "positive_control_request_count_frozen": POSITIVE_CONTROL_REQUEST_COUNT,
                    "positive_control_responses_valid": positive_control["request_count"],
                    "qualification_canary_responses_valid": 0,
                    "provider_generation_requests_attempted_total": (
                        provider_generation_requests_attempted
                    ),
                    "provider_scientific_matrix_responses_valid": 0,
                    "droid_matrix_admitted": False,
                    "action_conditioned_video_rollout_generated": True,
                    "droid_action_conditioned_video_rollout_generated": False,
                    "runtime_seconds": time.monotonic() - started_at,
                    "immutable_request_journal_tail_sha256": journal.previous,
                    "evaluator_eligible": False,
                    "claims": {
                        "runtime": True,
                        "generated_media": True,
                        "official_positive_control": False,
                        "wam_causal_validity": False,
                        "evaluator_validity": False,
                        "ranking_fidelity": False,
                        "physical_performance": False,
                    },
                }
                write_json(result_path, result)
                return result
        canary_row = next(
            row for row in rows if row.get("condition") == "recorded" and row.get("seed") == 0
        )
        canary_action = conditions["recorded"]
        direct_request = _serialize_rollout_request(
            request_row=canary_row,
            action_stream=canary_action,
            num_inference_steps=CANARY_INFERENCE_STEPS,
            task_instruction=task_instruction,
        )
        wrapper_request = _serialize_blueprint_wrapper_request(
            request_row=canary_row,
            action_stream=canary_action,
            num_inference_steps=CANARY_INFERENCE_STEPS,
            task_instruction=task_instruction,
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
                "experiment_id": EXPERIMENT_ID,
                "request_id": request_id,
                "condition": condition,
                "seed": int(row["seed"]),
                "initial_observation_sha256": input_checks["initial_observation_sha256"],
                "task_instruction": task_instruction,
                "vision_conditioning_mode": "first_pixel_frame_only",
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
                        task_instruction=task_instruction,
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
                        "infrastructure_failure"
                        if retryable
                        else "wam_scientific_or_runtime_failure"
                    )
                    raise
        hashes = [row["response"]["output_sha256"] for row in rollout_records]
        duplicate_output_hashes = len(hashes) - len(set(hashes))
        runtime_status = (
            "completed" if len(rollout_records) == scientific_matrix_request_count else "blocked"
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
            "reused_retained_server": reused_retained_server,
            "official_positive_control": positive_control,
            "positive_control_request_count_frozen": (
                POSITIVE_CONTROL_REQUEST_COUNT if positive_control is not None else 0
            ),
            "positive_control_responses_valid": (
                positive_control["request_count"] if positive_control is not None else 0
            ),
            "direct_canary_passed": True,
            "blueprint_wrapper_same_process_canary_passed": True,
            "exact_action_conditioned_stack_preflight": exact_stack_preflight,
            "qualification_canary_request_count_frozen": QUALIFICATION_CANARY_REQUEST_COUNT,
            "qualification_canary_responses_valid": qualification_canary_responses_valid,
            "scientific_matrix_request_count_frozen": scientific_matrix_request_count,
            "total_initial_generation_request_count_frozen": (
                total_initial_generation_request_count
            ),
            "provider_generation_requests_attempted_total": (
                provider_generation_requests_attempted
            ),
            "provider_scientific_matrix_responses_valid": len(rollout_records),
            "accepted_first_valid_count": len(accepted_request_ids),
            "output_duplicate_hash_count": duplicate_output_hashes,
            "action_conditioned_video_rollout_generated": bool(rollout_records),
            "complete_action_condition_seed_matrix_generated": (
                len(rollout_records) == scientific_matrix_request_count
            ),
            "runtime_seconds": time.monotonic() - started_at,
            "immutable_request_journal_tail_sha256": journal.previous,
            "guardrail_exception": "public_nvidia_droid_example_only",
            "causal_validity": "pending_deterministic_cross_condition_adjudication",
            "evaluator_eligible": False,
            "claims": {
                "runtime": True,
                "generated_media": bool(rollout_records),
                "official_positive_control": (
                    positive_control is None or positive_control["status"] == "passed"
                ),
                "wam_causal_validity": False,
                "evaluator_validity": False,
                "ranking_fidelity": False,
                "physical_performance": False,
            },
        }
        write_json(result_path, result)
        return result
    except Exception as exc:
        provider_generation_requests_attempted = max(
            provider_generation_requests_attempted,
            positive_control_attempt_counter[0],
        )
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
        retained = (
            _env_truthy(RETAIN_SERVER_ENV) and not force_server_shutdown and process.poll() is None
        )
        write_json(
            output_dir / "cosmos_server_retention.json",
            {
                "status": "retained_loaded" if retained else "terminal_shutdown",
                "server_identity": server_identity,
                "process_alive": process.poll() is None,
                "server_remained_loaded": retained,
                "reused_retained_server": reused_retained_server,
            },
        )
        if not retained:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=30)
        if server_log is not None:
            server_log.close()


if __name__ == "__main__":
    outcome = run()
    print(json.dumps({"status": outcome.get("status")}, sort_keys=True))
    raise SystemExit(0 if outcome.get("status") == "completed" else 2)
