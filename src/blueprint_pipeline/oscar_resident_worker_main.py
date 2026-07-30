"""Worker entrypoint for the resident OSCAR generation process.

Runs ON the GPU pod. Loads the OSCAR pipeline exactly once, announces readiness
with its load time and device identity, then serves line-delimited JSON generate
requests until told to shut down or its stdin closes.

The heavy imports are deliberately deferred into :func:`load_pipeline` so this
module can be imported, unit-tested, and protocol-checked on a machine with no
CUDA, no torch, and no OSCAR checkout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from collections.abc import Mapping, Sequence
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Callable

from .oscar_resident_worker import (
    READY_SCHEMA_VERSION,
    REQUEST_SCHEMA_VERSION,
    RESPONSE_SCHEMA_VERSION,
)


TAIL_LIMIT = 4000


def _string(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _emit(stream: Any, payload: Mapping[str, Any]) -> None:
    stream.write(json.dumps(dict(payload), sort_keys=True) + "\n")
    stream.flush()


def checkpoint_digest(checkpoint: str | Path) -> str:
    """Digest of the checkpoint the worker actually opened.

    Directory checkpoints are digested over their file names and sizes rather
    than their bytes: the identity must be cheap enough to compute at startup,
    and a multi-gigabyte byte digest would reintroduce the very startup cost the
    resident worker exists to remove.
    """

    path = Path(checkpoint).expanduser()
    hasher = hashlib.sha256()
    if path.is_file():
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    if path.is_dir():
        for entry in sorted(path.rglob("*")):
            if entry.is_file():
                hasher.update(str(entry.relative_to(path)).encode("utf-8"))
                hasher.update(str(entry.stat().st_size).encode("utf-8"))
        return hasher.hexdigest()
    return ""


class _OfficialOscarPipeline:
    """Small resident wrapper around OSCAR's pinned public inference primitives."""

    def __init__(
        self,
        *,
        model: Any,
        load_first_frame: Callable[..., Any],
        load_video: Callable[..., Any],
        run_inference: Callable[..., Any],
        save_video: Callable[..., Any],
        rearrange: Callable[..., Any],
        numpy: Any,
    ) -> None:
        self.model = model
        self._load_first_frame = load_first_frame
        self._load_video = load_video
        self._run_inference = run_inference
        self._save_video = save_video
        self._rearrange = rearrange
        self._numpy = numpy

    def generate(
        self,
        *,
        first_frame: str,
        prompt: str,
        negative_prompt: str | None,
        skeleton_video: str,
        num_frames: int,
        num_steps: int,
        guidance: float,
        shift: float,
        seed: int,
        height: int,
        width: int,
        fps: float,
        output: str,
    ) -> None:
        first = self._load_first_frame(Path(first_frame), height, width)
        skeleton = self._load_video(Path(skeleton_video), 0, num_frames, height, width)
        rgb = self._numpy.tile(first[None], (num_frames, 1, 1, 1))
        sample = self._run_inference(
            self.model,
            rgb_frames=rgb,
            condition_frames=skeleton,
            prompt=prompt,
            negative_prompt=negative_prompt,
            fps=fps,
            num_frames=num_frames,
            height=height,
            width=width,
            num_steps=num_steps,
            guidance=guidance,
            shift=shift,
            seed=seed,
        )
        destination = Path(output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        base = str(destination)[:-4] if str(destination).endswith(".mp4") else str(destination)
        frames = self._rearrange(sample.float().clamp(-1, 1).cpu(), "b c t h w -> c t h (b w)")
        self._save_video(frames * 0.5 + 0.5, base, fps=fps)


def load_pipeline(*, oscar_repo: str | Path, checkpoint: str | Path) -> tuple[Any, str]:
    """Load OSCAR's official model once and return its resident public wrapper."""

    repo = str(Path(oscar_repo).expanduser())
    if repo not in sys.path:
        sys.path.insert(0, repo)

    import numpy as np  # noqa: PLC0415 - deferred with the OSCAR dependency graph
    import torch  # noqa: PLC0415 - deferred so this module imports without CUDA
    import torch.distributed as dist  # noqa: PLC0415

    device_name = ""
    if torch.cuda.is_available():  # pragma: no cover - requires a GPU
        device_name = str(torch.cuda.get_device_name(0))

    from einops import rearrange  # noqa: PLC0415
    from inference._core import (  # type: ignore # noqa: PLC0415
        load_first_frame_np,
        load_video_np,
        run_inference,
        setup_backends,
    )
    import worldsim._ext.imaginaire.utils.distributed as oscar_distributed  # type: ignore # noqa: PLC0415
    from worldsim._ext.imaginaire.visualize.video import (  # type: ignore # noqa: PLC0415
        save_img_or_video,
    )
    from worldsim._src.utils.model_loader import (  # type: ignore # noqa: PLC0415
        load_model_from_checkpoint,
    )

    setup_backends()
    for key, value in {
        "RANK": "0",
        "LOCAL_RANK": "0",
        "WORLD_SIZE": "1",
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": "29500",
    }.items():
        os.environ.setdefault(key, value)
    if not dist.is_initialized():
        oscar_distributed.init()
    os.chdir(repo)
    model, _ = load_model_from_checkpoint(
        experiment_name="cosmos2_robot_plus_human_v2_70f",
        checkpoint_path=str(checkpoint),
        enable_fsdp=False,
        config_file="worldsim/_src/configs/agibot_control/config.py",
    )
    pipeline = _OfficialOscarPipeline(
        model=model,
        load_first_frame=load_first_frame_np,
        load_video=load_video_np,
        run_inference=run_inference,
        save_video=save_img_or_video,
        rearrange=rearrange,
        numpy=np,
    )
    return pipeline, device_name


def serve(
    *,
    stdin: Any,
    stdout: Any,
    generate: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    ready_payload: Mapping[str, Any],
    monotonic: Callable[[], float] = time.monotonic,
) -> int:
    """Serve the resident protocol over the supplied streams.

    Split out from :func:`main` so the protocol is testable with an in-memory
    ``generate`` and no model at all.
    """

    _emit(stdout, ready_payload)
    request_counter = 0
    for line in stdin:
        text = line.strip()
        if not text:
            continue
        try:
            request = json.loads(text)
        except ValueError:
            _emit(
                stdout,
                {
                    "schema_version": RESPONSE_SCHEMA_VERSION,
                    "request_id": "",
                    "status": "error",
                    "blockers": ["oscar_resident_worker_request_not_json"],
                },
            )
            continue
        if not isinstance(request, Mapping):
            continue
        if request.get("schema_version") != REQUEST_SCHEMA_VERSION:
            _emit(
                stdout,
                {
                    "schema_version": RESPONSE_SCHEMA_VERSION,
                    "request_id": _string(request.get("request_id")),
                    "status": "error",
                    "blockers": ["oscar_resident_worker_request_schema_invalid"],
                },
            )
            continue
        operation = _string(request.get("op"))
        if operation == "shutdown":
            return 0
        request_id = _string(request.get("request_id"))
        if operation != "generate":
            _emit(
                stdout,
                {
                    "schema_version": RESPONSE_SCHEMA_VERSION,
                    "request_id": request_id,
                    "status": "error",
                    "blockers": [f"oscar_resident_worker_unsupported_op:{operation}"],
                },
            )
            continue

        request_counter += 1
        started = monotonic()
        try:
            # OSCAR and several of its transitive libraries print progress to
            # stdout. Keep the line-delimited JSON control channel pure.
            with redirect_stdout(sys.stderr):
                outcome = dict(generate(request))
            blockers = [
                _string(item) for item in outcome.get("blockers", []) or [] if _string(item)
            ]
            status = "ok" if not blockers else "error"
        except Exception as error:  # noqa: BLE001 - a worker crash must be a protocol error
            outcome = {
                "stderr_tail": traceback.format_exc()[-TAIL_LIMIT:],
            }
            blockers = [f"oscar_resident_worker_generate_raised:{type(error).__name__}"]
            status = "error"
        elapsed = monotonic() - started
        _emit(
            stdout,
            {
                "schema_version": RESPONSE_SCHEMA_VERSION,
                "request_id": request_id,
                "status": status,
                "blockers": sorted(set(blockers)),
                "generate_seconds": round(elapsed, 6),
                # Distinct per response so the client can detect a replayed or
                # reordered stream rather than trusting request ids alone.
                "runtime_result_id": hashlib.sha256(
                    f"{ready_payload.get('worker_session_id')}:{request_counter}".encode("utf-8")
                ).hexdigest(),
                "stdout_tail": _string(outcome.get("stdout_tail"))[-TAIL_LIMIT:],
                "stderr_tail": _string(outcome.get("stderr_tail"))[-TAIL_LIMIT:],
                "output_video": _string(outcome.get("output_video")),
            },
        )
    return 0


def _pipeline_generate(
    pipeline: Any,
    *,
    num_steps: int,
    guidance: float,
    shift: float,
    height: int,
    width: int,
    fps: float,
) -> Callable[[Mapping[str, Any]], Mapping[str, Any]]:
    def _generate(request: Mapping[str, Any]) -> Mapping[str, Any]:  # pragma: no cover - GPU only
        output_video = _string(request.get("output_video"))
        skeleton_video = _string(request.get("skeleton_video"))
        pipeline.generate(
            first_frame=_string(request.get("reference_frame_path")),
            prompt=_string(request.get("task_prompt")),
            negative_prompt=_string(request.get("negative_prompt")) or None,
            skeleton_video=skeleton_video,
            num_frames=int(request.get("num_frames") or 8),
            num_steps=int(num_steps),
            guidance=float(guidance),
            shift=float(shift),
            seed=int(request.get("seed") or 42),
            height=int(height),
            width=int(width),
            fps=float(fps),
            output=output_video,
        )
        blockers = [] if Path(output_video).is_file() else ["oscar_resident_worker_output_missing"]
        return {"output_video": output_video, "blockers": blockers}

    return _generate


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - GPU entrypoint
    parser = argparse.ArgumentParser(description="Resident OSCAR generation worker")
    parser.add_argument("--oscar-repo", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--num-steps", type=int, default=35)
    parser.add_argument("--guidance", type=float, default=6.0)
    parser.add_argument("--shift", type=float, default=5.0)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--fps", type=float, default=15.0)
    args = parser.parse_args(argv)

    started = time.monotonic()
    try:
        with redirect_stdout(sys.stderr):
            pipeline, device_name = load_pipeline(
                oscar_repo=args.oscar_repo, checkpoint=args.checkpoint
            )
    except Exception:
        _emit(
            sys.stdout,
            {
                "schema_version": READY_SCHEMA_VERSION,
                "status": "failed",
                "blockers": ["oscar_resident_worker_model_load_failed"],
                "stderr_tail": traceback.format_exc()[-TAIL_LIMIT:],
            },
        )
        return 1

    ready = {
        "schema_version": READY_SCHEMA_VERSION,
        "status": "ready",
        "model_load_seconds": round(time.monotonic() - started, 6),
        "cuda_device_name": device_name,
        "checkpoint_sha256": checkpoint_digest(args.checkpoint),
        "worker_session_id": hashlib.sha256(os.urandom(32)).hexdigest(),
        "worker_pid": os.getpid(),
    }
    return serve(
        stdin=sys.stdin,
        stdout=sys.stdout,
        generate=_pipeline_generate(
            pipeline,
            num_steps=args.num_steps,
            guidance=args.guidance,
            shift=args.shift,
            height=args.height,
            width=args.width,
            fps=args.fps,
        ),
        ready_payload=ready,
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
