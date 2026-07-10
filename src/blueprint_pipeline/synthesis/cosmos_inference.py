"""Cosmos-Predict2.5-2B Image2World wrapper (zero-FT synthesis).

This module wraps NVIDIA's Cosmos-Predict2.5-2B in Image2World mode.
Given a splatted reference frame as the conditioning image, Cosmos generates
a plausible short video from that viewpoint.

The zero-FT path (no Blueprint fine-tuning) works because:
  - Depth splatting provides geometrically correct structure as the conditioning image
  - Cosmos I2W was trained to synthesise coherent continuations from any image
  - Warehouse/facility interiors share enough structure (flat floors, straight walls,
    overhead lighting) with training distribution for reasonable zero-FT quality

Fine-tuned quality (Phase 4B) will require:
  - A dataset of aligned Blueprint captures with paired reference/target frames
  - Fine-tuning Cosmos-Predict2.5-2B with Plücker ray map conditioning injected
    as residuals into video token sequence (per SWM training recipe)

For Phase 4A, use generate_view() with mode="splat_only" or mode="cosmos_i2w".

Cosmos model access:
  pip install cosmos-predict2-5  (NVIDIA's package, requires NGC token)
  or via HuggingFace: nvidia/Cosmos-Predict2.5-2B

Environment variable COSMOS_MODEL_ID overrides the default model path.
NGC_API_KEY is required for downloading model weights from NVIDIA NGC.
"""

from __future__ import annotations

import atexit
import json
import os
from pathlib import Path
import queue
import re
import subprocess
import threading
import time
import uuid
from typing import Any, Dict, List, Mapping, Optional

import numpy as np

from ..model_access_env import normalize_model_access_env


normalize_model_access_env()


_DEFAULT_COSMOS_MODEL_ID = os.getenv(
    "COSMOS_MODEL_ID",
    "nvidia/Cosmos-Predict2.5-2B",
)
# Hugging Face revision ``diffusers/base/post-trained`` resolved to this
# immutable commit when the source pin was audited.  Branch names and caller-
# supplied hashes are deliberately not accepted: adding a new model revision
# requires a reviewed source change that extends this allowlist.
_COSMOS_PREDICT25_2B_DIFFUSERS_REVISION = "0d37c7498f54cee3c599d438d895a0a4a8608064"
_APPROVED_COSMOS_MODEL_REVISIONS = {
    "nvidia/Cosmos-Predict2.5-2B": frozenset({_COSMOS_PREDICT25_2B_DIFFUSERS_REVISION}),
}
_IMMUTABLE_REVISION_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_DEFAULT_COSMOS_MODEL_REVISION = (
    os.getenv(
        "COSMOS_MODEL_REVISION",
        _COSMOS_PREDICT25_2B_DIFFUSERS_REVISION,
    )
    .strip()
    .lower()
)
_DEFAULT_COSMOS_MODEL_VARIANT = os.getenv(
    "COSMOS_MODEL_VARIANT",
    "post-trained",
).strip()
# NVIDIA's official-repository Inference API selects a mutable model variant
# and currently exposes no immutable checkpoint revision argument.  Keep those
# backends unavailable until that API can be bound to the approved model commit.
_OFFICIAL_REPO_IMMUTABLE_MODEL_REVISION_SUPPORTED = False
_DEFAULT_COSMOS_OFFICIAL_REPO_ROOT = os.getenv("COSMOS_OFFICIAL_REPO_ROOT", "").strip()
_DEFAULT_COSMOS_DISABLE_GUARDRAILS = str(
    os.getenv("COSMOS_DISABLE_GUARDRAILS") or ""
).strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
_DEFAULT_COSMOS_PROMPT = os.getenv(
    "COSMOS_DEFAULT_PROMPT",
    (
        "First-person camera moving through the same real indoor site shown in "
        "the conditioning image. Preserve the existing room layout, furniture, "
        "lighting, textures, and site identity. Continue the scene naturally "
        "without changing it into a different building type."
    ),
)

# Cosmos generation defaults — tuned for Blueprint facility captures
_DEFAULT_NUM_FRAMES = 57  # ~2 seconds at ~28fps; matches Cosmos training length
_DEFAULT_WIDTH = 1280
_DEFAULT_HEIGHT = 720
_DEFAULT_GUIDANCE_SCALE = 7.0
_DEFAULT_NUM_STEPS = 35
_OFFICIAL_REPO_INFERENCE_LOCK = threading.Lock()
_LOADED_MODELS: Dict[str, Any] = {}
_LOADED_MODELS_LOCK = threading.Lock()


def _approved_cosmos_model_revision(
    model_id: str,
    revision: str | None = None,
) -> str | None:
    """Return a reviewed immutable revision, or fail closed with ``None``."""

    normalized_model_id = str(model_id or "").strip()
    normalized_revision = (
        str(revision if revision is not None else _DEFAULT_COSMOS_MODEL_REVISION).strip().lower()
    )
    if _IMMUTABLE_REVISION_PATTERN.fullmatch(normalized_revision) is None:
        return None
    approved = _APPROVED_COSMOS_MODEL_REVISIONS.get(normalized_model_id, frozenset())
    return normalized_revision if normalized_revision in approved else None


def _env_truthy(name: str, *, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _persistent_worker_enabled() -> bool:
    return _env_truthy("COSMOS_ENABLE_PERSISTENT_WORKER", default=True) and not _env_truthy(
        "COSMOS_DISABLE_PERSISTENT_WORKER",
        default=False,
    )


def _skip_official_repo_script() -> bool:
    return _env_truthy("COSMOS_SKIP_OFFICIAL_REPO_SCRIPT", default=False)


def _cold_subprocess_fallback_enabled() -> bool:
    return _env_truthy("COSMOS_ALLOW_COLD_SUBPROCESS_FALLBACK", default=True)


class PersistentCosmosWorkerClient:
    """Resident Cosmos worker backed by a long-lived Python subprocess."""

    def __init__(
        self,
        *,
        repo_root: Path,
        python_bin: Path,
        worker_env: Mapping[str, str],
        model_id: str,
        model_variant: str,
    ) -> None:
        self.repo_root = repo_root.resolve()
        self.python_bin = python_bin.expanduser()
        self.worker_env = _normalized_subprocess_env(worker_env)
        self.model_id = model_id
        self.model_variant = model_variant
        self.startup_timeout_s = max(
            30.0, float(os.getenv("COSMOS_WORKER_STARTUP_TIMEOUT_S", "900"))
        )
        self.request_timeout_s = max(
            30.0, float(os.getenv("COSMOS_WORKER_REQUEST_TIMEOUT_S", "900"))
        )
        self._process: Optional[subprocess.Popen[str]] = None
        self._stdout_queue: queue.Queue[Dict[str, Any]] = queue.Queue()
        self._reader_thread: Optional[threading.Thread] = None
        self._log_handle: Optional[Any] = None
        self._io_lock = threading.Lock()
        self._backend_name = "persistent_worker"
        self._ready_payload: Dict[str, Any] = {}

    @property
    def backend_name(self) -> str:
        return self._backend_name

    def describe(self) -> Dict[str, Any]:
        return {
            "backend": "persistent_worker",
            "repo_root": str(self.repo_root),
            "python_bin": str(self.python_bin),
            "model_id": self.model_id,
            "model_variant": self.model_variant,
            "worker_backend": self.backend_name,
            "ready": bool(self._ready_payload),
        }

    def prewarm(self) -> Dict[str, Any]:
        with self._io_lock:
            self._ensure_started()
            self._send_message({"type": "ping"})
            response = self._await_message(message_type="pong", timeout_s=10.0)
            return dict(response)

    def close(self) -> None:
        process = self._process
        self._process = None
        if process is not None:
            try:
                if process.stdin is not None:
                    process.stdin.close()
            except Exception:
                pass
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
            if process.stdout is not None:
                try:
                    process.stdout.close()
                except Exception:
                    pass
        if self._log_handle is not None:
            try:
                self._log_handle.close()
            except Exception:
                pass
            self._log_handle = None
        self._reader_thread = None
        self._ready_payload = {}

    def generate_image_to_world(
        self,
        *,
        conditioning_image: Any,
        output_path: Path,
        num_frames: int,
        width: int,
        height: int,
        guidance_scale: float,
        num_steps: int,
    ) -> Dict[str, Any]:
        from PIL import Image

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        conditioning_path = output_path.with_stem(output_path.stem + "_conditioning")
        Image.fromarray(np.array(conditioning_image)).save(conditioning_path)

        with self._io_lock:
            self._ensure_started()
            request_id = uuid.uuid4().hex
            self._send_message(
                {
                    "type": "generate",
                    "request_id": request_id,
                    "input_path": str(conditioning_path),
                    "output_path": str(output_path),
                    "num_frames": int(num_frames),
                    "width": int(width),
                    "height": int(height),
                    "guidance_scale": float(guidance_scale),
                    "num_steps": int(num_steps),
                }
            )
            response = self._await_message(
                message_type="result",
                request_id=request_id,
                timeout_s=self.request_timeout_s,
            )
        if not bool(response.get("ok", False)):
            raise RuntimeError(str(response.get("error") or "persistent_worker_generation_failed"))
        return dict(response)

    def _ensure_started(self) -> None:
        if self._process is not None and self._process.poll() is None:
            return

        self.close()
        log_path = Path(
            str(
                self.worker_env.get("COSMOS_WORKER_LOG_PATH")
                or (self.repo_root / "assets" / "outputs" / "blueprint_cosmos_worker.log")
            )
        )
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_handle = log_path.open("a", encoding="utf-8")
        self._process = subprocess.Popen(
            [str(self.python_bin), "-m", "blueprint_pipeline.synthesis.cosmos_worker"],
            cwd=str(self.repo_root),
            env=self.worker_env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._log_handle,
            text=True,
            bufsize=1,
        )
        self._stdout_queue = queue.Queue()
        self._reader_thread = threading.Thread(target=self._drain_stdout, daemon=True)
        self._reader_thread.start()
        try:
            ready = self._await_message(message_type="ready", timeout_s=self.startup_timeout_s)
        except Exception:
            self.close()
            raise
        self._backend_name = str(ready.get("backend") or "persistent_worker")
        self._ready_payload = dict(ready)

    def _drain_stdout(self) -> None:
        process = self._process
        if process is None or process.stdout is None:
            return
        for raw_line in process.stdout:
            text = raw_line.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, Mapping):
                self._stdout_queue.put(dict(payload))

    def _send_message(self, payload: Mapping[str, Any]) -> None:
        process = self._process
        if process is None or process.stdin is None or process.poll() is not None:
            raise RuntimeError("persistent_worker_not_running")
        process.stdin.write(json.dumps(dict(payload)) + "\n")
        process.stdin.flush()

    def _await_message(
        self,
        *,
        message_type: str,
        timeout_s: float,
        request_id: str | None = None,
    ) -> Dict[str, Any]:
        deadline = time.monotonic() + timeout_s
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError(f"persistent_worker_timeout:{message_type}")
            process = self._process
            if process is not None and process.poll() is not None and self._stdout_queue.empty():
                raise RuntimeError(f"persistent_worker_exited:{process.returncode}")
            try:
                message = dict(self._stdout_queue.get(timeout=min(remaining, 0.25)))
            except queue.Empty as exc:
                process = self._process
                if process is not None and process.poll() is not None:
                    raise RuntimeError(f"persistent_worker_exited:{process.returncode}") from exc
                continue
            msg_type = str(message.get("type") or "")
            if msg_type == "error":
                stage = str(message.get("stage") or "worker")
                detail = str(message.get("error") or "persistent_worker_failed")
                raise RuntimeError(f"{stage}:{detail}")
            if msg_type == "protocol_error":
                raise RuntimeError(f"persistent_worker_protocol_error:{message.get('raw')}")
            if msg_type != message_type:
                continue
            if request_id is not None and str(message.get("request_id") or "") != request_id:
                continue
            return message


def _close_loaded_models() -> None:
    with _LOADED_MODELS_LOCK:
        loaded_models = list(_LOADED_MODELS.values())
        _LOADED_MODELS.clear()
    for model in loaded_models:
        if isinstance(model, PersistentCosmosWorkerClient):
            model.close()


atexit.register(_close_loaded_models)


def generate_view(
    *,
    splatted_image: np.ndarray,  # [H, W, 3] uint8 RGB — depth-splatted reference
    coverage_mask: np.ndarray,  # [H, W] bool — True = valid pixels in splatted_image
    target_plucker_map: Optional[np.ndarray] = None,  # [6, H, W] float32 (for future conditioning)
    output_path: Path,
    mode: str = "splat_only",  # "splat_only" | "cosmos_i2w"
    cosmos_model: Optional[Any] = None,  # pre-loaded model; loads if None
    num_frames: int = _DEFAULT_NUM_FRAMES,
    width: int = _DEFAULT_WIDTH,
    height: int = _DEFAULT_HEIGHT,
    guidance_scale: float = _DEFAULT_GUIDANCE_SCALE,
    num_steps: int = _DEFAULT_NUM_STEPS,
) -> Path:
    """
    Generate a view from the splatted conditioning image.

    mode="splat_only":
      Saves splatted_image as a JPEG. No generative model invoked.
      This is the fastest path and geometrically exact. Holes and low-coverage
      regions will be visible. Use for debugging and as a quality floor.

    mode="cosmos_i2w":
      Passes splatted_image to Cosmos-Predict2.5-2B Image2World.
      Cosmos generates a short video starting from this frame.
      First frame of the video is saved as a JPEG; full video as MP4 alongside.
      Requires Cosmos installed and NGC credentials.

    Returns path to the output image (JPEG).
    """
    from PIL import Image

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if mode == "splat_only":
        img = Image.fromarray(splatted_image)
        img.save(output_path)
        return output_path

    if mode == "cosmos_i2w":
        return _cosmos_image_to_world(
            conditioning_image=splatted_image,
            output_path=output_path,
            cosmos_model=cosmos_model,
            num_frames=num_frames,
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            num_steps=num_steps,
        )

    raise ValueError(f"Unknown generation mode: {mode!r}. Use 'splat_only' or 'cosmos_i2w'.")


def load_cosmos_model(model_id: Optional[str] = None) -> Any:
    """
    Load Cosmos-Predict2.5-2B for inference.

    Tries two backends in order:
      1. cosmos_predict2_5 (NVIDIA's official package)
      2. HuggingFace diffusers pipeline (community wrapper)

    Returns a loaded model object. Pass this to generate_view(cosmos_model=...)
    to avoid reloading on every call.
    """
    mid = model_id or _DEFAULT_COSMOS_MODEL_ID
    revision = _approved_cosmos_model_revision(mid)
    if revision is None:
        raise ValueError(
            "cosmos_model_revision_not_approved: model downloads require a "
            "reviewed immutable commit"
        )
    cache_key = f"{mid}@{revision}"
    with _LOADED_MODELS_LOCK:
        cached = _LOADED_MODELS.get(cache_key)
        if cached is not None:
            return cached

        model = _try_load_cosmos_official(mid)
        if model is None:
            model = _try_load_cosmos_official_repo_direct(mid)
        if model is None:
            model = _try_load_cosmos_diffusers(mid)
        if model is None:
            model = _try_load_cosmos_official_repo_worker(mid)
        if model is None:
            model = _try_load_cosmos_official_repo(mid)
        if model is not None:
            _LOADED_MODELS[cache_key] = model
            return model
    raise ImportError(
        "Could not load Cosmos-Predict2.5-2B. Tried the deprecated cosmos-predict2-5 wheel path, "
        "official in-process loaders, resident worker fallback, and cold official NVIDIA repo fallback. "
        f"Tried model path: {mid}"
    )


def prewarm_cosmos_model(model_id: Optional[str] = None) -> Dict[str, Any]:
    started_at = time.monotonic()
    model = load_cosmos_model(model_id=model_id)
    payload = describe_cosmos_model(model)
    if isinstance(model, PersistentCosmosWorkerClient):
        payload.update(model.prewarm())
    payload["prewarm_ms"] = int(round((time.monotonic() - started_at) * 1000.0))
    payload["model_id"] = model_id or _DEFAULT_COSMOS_MODEL_ID
    payload["model_revision"] = _approved_cosmos_model_revision(
        model_id or _DEFAULT_COSMOS_MODEL_ID
    )
    return payload


def describe_cosmos_model(model: Any) -> Dict[str, Any]:
    if isinstance(model, PersistentCosmosWorkerClient):
        return model.describe()
    if isinstance(model, Mapping):
        return {str(key): value for key, value in model.items()}
    return {"backend": type(model).__name__}


# ---------------------------------------------------------------------------
# Internal: Cosmos I2W generation
# ---------------------------------------------------------------------------


def _cosmos_image_to_world(
    *,
    conditioning_image: np.ndarray,
    output_path: Path,
    cosmos_model: Optional[Any],
    num_frames: int,
    width: int,
    height: int,
    guidance_scale: float,
    num_steps: int,
) -> Path:
    from PIL import Image

    model = cosmos_model or load_cosmos_model()
    if isinstance(model, PersistentCosmosWorkerClient):
        model.generate_image_to_world(
            conditioning_image=conditioning_image,
            output_path=output_path,
            num_frames=num_frames,
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            num_steps=num_steps,
        )
        return output_path
    pil_image = Image.fromarray(conditioning_image).resize((width, height), Image.LANCZOS)

    video_frames = _invoke_cosmos(
        model=model,
        conditioning_image=pil_image,
        num_frames=num_frames,
        width=width,
        height=height,
        guidance_scale=guidance_scale,
        num_steps=num_steps,
    )

    # Save first frame as JPEG (the synthesised view)
    first_frame = video_frames[0]
    if isinstance(first_frame, np.ndarray):
        Image.fromarray(first_frame).save(output_path)
    else:
        first_frame.save(output_path)

    # Save full video as MP4 alongside
    mp4_path = output_path.with_suffix(".mp4")
    _save_video(video_frames, mp4_path, fps=28)

    return output_path


def _invoke_cosmos(
    *,
    model: Any,
    conditioning_image: Any,  # PIL Image
    num_frames: int,
    width: int,
    height: int,
    guidance_scale: float,
    num_steps: int,
) -> List[Any]:
    """
    Dispatch to the loaded Cosmos model's generation interface.
    Handles both the official NVIDIA API and the HuggingFace diffusers API.
    """
    # --- NVIDIA official cosmos-predict2-5 package ---
    if hasattr(model, "generate") and hasattr(model, "image_to_world"):
        result = model.image_to_world(
            image=conditioning_image,
            num_frames=num_frames,
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            num_inference_steps=num_steps,
        )
        return _extract_frames(result)

    # --- HuggingFace diffusers pipeline (community wrapper) ---
    if hasattr(model, "__call__"):
        result = model(
            image=conditioning_image,
            num_frames=num_frames,
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            num_inference_steps=num_steps,
        )
        frames = getattr(result, "frames", None) or getattr(result, "images", None)
        if frames is not None:
            return list(frames[0]) if isinstance(frames[0], list) else list(frames)

    # --- NVIDIA official repo loaded directly in-process ---
    if isinstance(model, dict) and model.get("backend") == "official_repo_direct":
        return _invoke_cosmos_official_repo_direct(
            model=model,
            conditioning_image=conditioning_image,
            num_frames=num_frames,
            guidance_scale=guidance_scale,
            num_steps=num_steps,
        )

    # --- NVIDIA official repo examples/inference.py wrapper ---
    if isinstance(model, dict) and model.get("backend") == "official_repo_script":
        return _invoke_cosmos_official_repo_script(
            model=model,
            conditioning_image=conditioning_image,
        )

    raise RuntimeError(
        f"Unrecognised Cosmos model interface: {type(model)}. "
        "Expected .image_to_world() or callable pipeline with frames/images output."
    )


def _extract_frames(result: Any) -> List[Any]:
    """Extract frame list from various Cosmos output formats."""
    if isinstance(result, (list, tuple)):
        return list(result)
    frames = getattr(result, "frames", None)
    if frames is not None:
        return list(frames[0]) if hasattr(frames[0], "__iter__") else list(frames)
    images = getattr(result, "images", None)
    if images is not None:
        return list(images)
    raise RuntimeError(f"Cannot extract frames from Cosmos output: {type(result)}")


def _save_video(frames: List[Any], path: Path, fps: int = 28) -> None:
    """Save a list of PIL Images or numpy arrays as an MP4 using FFmpeg via imageio."""
    try:
        import imageio
        import numpy as np

        np_frames = []
        for f in frames:
            if isinstance(f, np.ndarray):
                np_frames.append(f)
            else:
                np_frames.append(np.array(f))

        writer = imageio.get_writer(str(path), fps=fps, codec="libx264", quality=8)
        for frame in np_frames:
            writer.append_data(frame)
        writer.close()
    except Exception:
        pass  # Video saving is best-effort; the frame JPEG is the primary output


# ---------------------------------------------------------------------------
# Backend loaders
# ---------------------------------------------------------------------------


def _try_load_cosmos_official(model_id: str) -> Optional[Any]:
    revision = _approved_cosmos_model_revision(model_id)
    if revision is None:
        return None
    try:
        # NVIDIA's official cosmos-predict2-5 package
        from cosmos_predict2_5 import CosmosPredict25  # type: ignore[import]

        model = CosmosPredict25.from_pretrained(model_id, revision=revision)
        model.eval()
        return model
    except (ImportError, Exception):
        return None


def _try_load_cosmos_diffusers(model_id: str) -> Optional[Any]:
    revision = _approved_cosmos_model_revision(model_id)
    if revision is None:
        return None
    try:
        import torch
        from diffusers import DiffusionPipeline  # type: ignore[import]

        pipe = DiffusionPipeline.from_pretrained(
            model_id,
            revision=revision,
            torch_dtype=torch.bfloat16,
        )
        if torch.cuda.is_available():
            pipe = pipe.to("cuda")
        return pipe
    except (ImportError, Exception):
        return None


def _try_load_cosmos_official_repo_direct(model_id: str) -> Optional[Dict[str, Any]]:
    if (
        not _OFFICIAL_REPO_IMMUTABLE_MODEL_REVISION_SUPPORTED
        or _approved_cosmos_model_revision(model_id) is None
        or not _DEFAULT_COSMOS_OFFICIAL_REPO_ROOT
    ):
        return None
    repo_root = Path(_DEFAULT_COSMOS_OFFICIAL_REPO_ROOT).expanduser()
    model_variant = _official_repo_model_variant(model_id, _DEFAULT_COSMOS_MODEL_VARIANT)
    if model_variant is None:
        return None
    try:
        from cosmos_oss.init import init_environment, init_output_dir
        from cosmos_predict2.config import SetupArguments
        from cosmos_predict2.inference import Inference

        output_root = (
            repo_root / "assets" / "outputs" / "blueprint_cosmos_resident_worker"
        ).resolve()
        output_root.mkdir(parents=True, exist_ok=True)
        init_environment()
        init_output_dir(output_root, profile=False)
        setup = SetupArguments(
            output_dir=output_root,
            model=model_variant,
            disable_guardrails=bool(_DEFAULT_COSMOS_DISABLE_GUARDRAILS),
            keep_going=False,
        )
        return {
            "backend": "official_repo_direct",
            "repo_root": str(repo_root.resolve()),
            "output_root": str(output_root),
            "model_variant": model_variant,
            "disable_guardrails": bool(_DEFAULT_COSMOS_DISABLE_GUARDRAILS),
            "inference": Inference(setup),
        }
    except Exception:
        return None


def _try_load_cosmos_official_repo_worker(model_id: str) -> Optional[PersistentCosmosWorkerClient]:
    if (
        not _OFFICIAL_REPO_IMMUTABLE_MODEL_REVISION_SUPPORTED
        or _approved_cosmos_model_revision(model_id) is None
        or not _persistent_worker_enabled()
        or not _DEFAULT_COSMOS_OFFICIAL_REPO_ROOT
    ):
        return None
    repo_root = Path(_DEFAULT_COSMOS_OFFICIAL_REPO_ROOT).expanduser()
    inference_entrypoint = repo_root / "examples" / "inference.py"
    venv_python = repo_root / ".venv" / "bin" / "python"
    if not inference_entrypoint.is_file() or not venv_python.is_file():
        return None
    model_variant = _official_repo_model_variant(model_id, _DEFAULT_COSMOS_MODEL_VARIANT)
    if model_variant is None:
        return None
    worker_python_raw = str(os.getenv("COSMOS_WORKER_PYTHON_BIN") or "").strip()
    worker_python = Path(worker_python_raw).expanduser() if worker_python_raw else venv_python
    if not worker_python.is_file():
        worker_python = venv_python
    return PersistentCosmosWorkerClient(
        repo_root=repo_root,
        python_bin=worker_python,
        worker_env=_official_repo_worker_env(repo_root=repo_root, python_bin=worker_python),
        model_id=model_id,
        model_variant=model_variant,
    )


def _try_load_cosmos_official_repo(model_id: str) -> Optional[Dict[str, Any]]:
    if (
        not _OFFICIAL_REPO_IMMUTABLE_MODEL_REVISION_SUPPORTED
        or _approved_cosmos_model_revision(model_id) is None
        or _skip_official_repo_script()
        or not _cold_subprocess_fallback_enabled()
    ):
        return None
    if not _DEFAULT_COSMOS_OFFICIAL_REPO_ROOT:
        return None
    repo_root = Path(_DEFAULT_COSMOS_OFFICIAL_REPO_ROOT).expanduser()
    inference_entrypoint = repo_root / "examples" / "inference.py"
    venv_python = repo_root / ".venv" / "bin" / "python"
    if not inference_entrypoint.is_file() or not venv_python.is_file():
        return None
    model_variant = _official_repo_model_variant(model_id, _DEFAULT_COSMOS_MODEL_VARIANT)
    if model_variant is None:
        return None
    return {
        "backend": "official_repo_script",
        "repo_root": str(repo_root.resolve()),
        "python_bin": str(venv_python),
        "model_variant": model_variant,
        "disable_guardrails": _DEFAULT_COSMOS_DISABLE_GUARDRAILS,
        "subprocess_env": _official_repo_subprocess_env(
            repo_root=repo_root, python_bin=venv_python
        ),
    }


def _official_repo_model_variant(model_id: str, variant: str) -> str | None:
    text = str(model_id or "").strip()
    if not text.startswith("nvidia/Cosmos-Predict2.5-"):
        return None
    size = "2B" if text.endswith("2B") else ("14B" if text.endswith("14B") else None)
    if size is None:
        return None
    suffix = str(variant or "").strip()
    if suffix not in {"pre-trained", "post-trained"}:
        return None
    return f"{size}/{suffix}"


def _invoke_cosmos_official_repo_script(
    *,
    model: Mapping[str, Any],
    conditioning_image: Any,
) -> List[Any]:
    from PIL import Image

    repo_root = Path(str(model["repo_root"])).resolve()
    python_bin = Path(str(model["python_bin"]))
    output_root = (
        repo_root / "assets" / "outputs" / f"blueprint_cosmos_official_{uuid.uuid4().hex[:8]}"
    ).resolve()
    output_root.mkdir(parents=True, exist_ok=False)
    sample_name = output_root.name
    input_path = output_root / f"{sample_name}.jpg"
    asset_path = output_root / f"{sample_name}.json"
    output_dir = output_root
    output_video_path = output_dir / f"{sample_name}.mp4"
    subprocess_log_path = output_dir / "official_repo_subprocess.log"

    Image.fromarray(np.array(conditioning_image)).save(input_path)
    asset_path.write_text(
        json.dumps(
            {
                "inference_type": "image2world",
                "name": sample_name,
                "input_path": str(input_path),
                "prompt": _DEFAULT_COSMOS_PROMPT,
            }
        ),
        encoding="utf-8",
    )

    python_command = [
        str(python_bin),
        "examples/inference.py",
        "-i",
        str(asset_path),
        "-o",
        str(output_dir),
        f"--model={model['model_variant']}",
    ]
    if bool(model.get("disable_guardrails")):
        python_command.append("--disable-guardrails")

    # The official repo path cold-loads a large model into a separate Python
    # process. Running more than one at once on a single H100 quickly exhausts
    # VRAM and makes every session slower, so serialize these launches.
    with _OFFICIAL_REPO_INFERENCE_LOCK:
        with subprocess_log_path.open("w", encoding="utf-8") as log_handle:
            result = subprocess.run(
                python_command,
                cwd=str(repo_root),
                text=True,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                check=False,
                env=_normalized_subprocess_env(model.get("subprocess_env")),
            )
    if result.returncode != 0:
        failure_detail = ""
        if subprocess_log_path.is_file():
            failure_detail = subprocess_log_path.read_text(encoding="utf-8").strip()
        raise RuntimeError(
            "official_repo_inference_failed: "
            + (failure_detail or f"exit_code={result.returncode}")
        )
    if not output_video_path.is_file():
        raise RuntimeError(f"official_repo_output_missing:{output_video_path}")

    try:
        import imageio.v3 as iio

        frames = list(iio.imiter(output_video_path))
    except Exception as exc:  # pragma: no cover - best effort decoding
        raise RuntimeError(f"official_repo_decode_failed:{exc}") from exc
    if not frames:
        raise RuntimeError(f"official_repo_output_empty:{output_video_path}")
    return frames


def _invoke_cosmos_official_repo_direct(
    *,
    model: Mapping[str, Any],
    conditioning_image: Any,
    num_frames: int,
    guidance_scale: float,
    num_steps: int,
) -> List[Any]:
    from PIL import Image

    output_root = Path(str(model["output_root"])).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    sample_name = f"blueprint_cosmos_resident_{uuid.uuid4().hex[:8]}"
    input_path = output_root / f"{sample_name}.jpg"
    Image.fromarray(np.array(conditioning_image)).save(input_path)

    inference = model.get("inference")
    if inference is None:
        raise RuntimeError("official_repo_direct_inference_missing")

    try:
        from cosmos_predict2.config import InferenceArguments
    except Exception as exc:
        raise RuntimeError(f"official_repo_direct_import_failed:{exc}") from exc

    sample = InferenceArguments(
        inference_type="image2world",
        name=sample_name,
        input_path=input_path,
        prompt=_DEFAULT_COSMOS_PROMPT,
        num_output_frames=int(num_frames),
        guidance=max(0, min(7, int(round(guidance_scale)))),
        num_steps=int(num_steps),
    )
    output_paths = inference.generate([sample], output_dir=output_root)
    if not output_paths:
        raise RuntimeError("official_repo_direct_output_missing")
    output_video_path = Path(str(output_paths[0])).resolve()
    if not output_video_path.is_file():
        raise RuntimeError(f"official_repo_direct_output_missing:{output_video_path}")

    try:
        import imageio.v3 as iio

        frames = list(iio.imiter(output_video_path))
    except Exception as exc:
        raise RuntimeError(f"official_repo_direct_decode_failed:{exc}") from exc
    if not frames:
        raise RuntimeError(f"official_repo_direct_output_empty:{output_video_path}")
    return frames


def _official_repo_subprocess_env(*, repo_root: Path, python_bin: Path) -> Dict[str, str]:
    env = _select_official_repo_env_vars()
    env["PATH"] = _prepend_search_paths(
        [
            str((repo_root / ".venv" / "bin").resolve()),
            str((Path.home() / ".local" / "bin").resolve()),
        ],
        env.get("PATH", ""),
    )
    return env


def _official_repo_worker_env(*, repo_root: Path, python_bin: Path) -> Dict[str, str]:
    env = _official_repo_subprocess_env(repo_root=repo_root, python_bin=python_bin)
    src_root = Path(__file__).resolve().parents[2]
    env["PYTHONPATH"] = _prepend_search_paths(
        [
            str(src_root.resolve()),
            str(repo_root.resolve()),
            str((repo_root / "packages" / "cosmos-oss").resolve()),
        ],
        os.getenv("PYTHONPATH", ""),
    )
    env["PYTHONUNBUFFERED"] = "1"
    env["COSMOS_OFFICIAL_REPO_ROOT"] = str(repo_root.resolve())
    env["COSMOS_MODEL_ID"] = _DEFAULT_COSMOS_MODEL_ID
    env["COSMOS_MODEL_REVISION"] = _DEFAULT_COSMOS_MODEL_REVISION
    env["COSMOS_MODEL_VARIANT"] = _DEFAULT_COSMOS_MODEL_VARIANT
    env["COSMOS_DISABLE_GUARDRAILS"] = "1" if _DEFAULT_COSMOS_DISABLE_GUARDRAILS else "0"
    env["COSMOS_DISABLE_PERSISTENT_WORKER"] = "1"
    env["COSMOS_SKIP_OFFICIAL_REPO_SCRIPT"] = "1"
    env["COSMOS_ALLOW_COLD_SUBPROCESS_FALLBACK"] = "0"
    return env


def _prepend_search_paths(paths: List[str], existing_path: str) -> str:
    merged: List[str] = []
    for value in [*paths, *existing_path.split(":")]:
        text = str(value or "").strip()
        if not text or text in merged:
            continue
        merged.append(text)
    return ":".join(merged)


def _normalized_subprocess_env(env: object) -> Dict[str, str]:
    if not isinstance(env, Mapping):
        return {key: value for key, value in os.environ.items() if isinstance(value, str)}
    normalized: Dict[str, str] = {}
    for key, value in env.items():
        if not isinstance(key, str) or not isinstance(value, str):
            continue
        normalized[key] = value
    return normalized


def _select_official_repo_env_vars() -> Dict[str, str]:
    selected: Dict[str, str] = {}
    allowed_exact = {
        "HOME",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "LD_LIBRARY_PATH",
        "LIBRARY_PATH",
        "PATH",
        "TZ",
    }
    allowed_prefixes = (
        "HF_",
        "HUGGINGFACE_",
        "NGC_",
    )
    for key, value in os.environ.items():
        if not isinstance(value, str):
            continue
        if key in allowed_exact or key.startswith(allowed_prefixes):
            selected[key] = value
    return selected
