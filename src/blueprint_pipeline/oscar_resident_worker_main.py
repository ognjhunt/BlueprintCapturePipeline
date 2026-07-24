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


def load_pipeline(*, oscar_repo: str | Path, checkpoint: str | Path) -> tuple[Any, str]:
    """Import OSCAR and construct its pipeline once. Returns (pipeline, device name)."""

    repo = str(Path(oscar_repo).expanduser())
    if repo not in sys.path:
        sys.path.insert(0, repo)

    import torch  # noqa: PLC0415 - deferred so this module imports without CUDA

    device_name = ""
    if torch.cuda.is_available():  # pragma: no cover - requires a GPU
        device_name = str(torch.cuda.get_device_name(0))

    from inference.inference_oscar import build_pipeline  # type: ignore # noqa: PLC0415

    pipeline = build_pipeline(checkpoint=str(checkpoint))
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
            skeleton_video=skeleton_video or None,
            num_frames=int(request.get("num_frames") or 8),
            num_steps=int(num_steps),
            guidance=float(guidance),
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
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--fps", type=float, default=15.0)
    args = parser.parse_args(argv)

    started = time.monotonic()
    try:
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
            height=args.height,
            width=args.width,
            fps=args.fps,
        ),
        ready_payload=ready,
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
