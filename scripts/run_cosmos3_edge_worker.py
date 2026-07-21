#!/usr/bin/env python3
"""Isolated offline Cosmos 3 Edge worker using NVIDIA's cosmos-framework CLI."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import threading
import time
from typing import Any


MODEL_ID = "nvidia/Cosmos3-Edge"
MODE_OUTPUT = {
    "forward_dynamics": ("generated_video", "vision.mp4"),
    "inverse_dynamics": ("action_inference", "sample_outputs.json"),
    "reasoning": ("reasoning_result", "reasoner_text.txt"),
}


def _sha(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _contains_remote_url(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower().startswith(("http://", "https://"))
    if isinstance(value, dict):
        return any(_contains_remote_url(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_remote_url(item) for item in value)
    return False


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for distribution in ("cosmos-framework", "torch", "transformers"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = "not-installed"
    return versions


def _gpu_sampler(stop: threading.Event, samples: list[int], identity: dict[str, Any]) -> None:
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        identity.update(
            {
                "name": str(pynvml.nvmlDeviceGetName(handle)),
                "uuid": str(pynvml.nvmlDeviceGetUUID(handle)),
                "memory_total_bytes": int(pynvml.nvmlDeviceGetMemoryInfo(handle).total),
                "driver_version": str(pynvml.nvmlSystemGetDriverVersion()),
            }
        )
        while not stop.wait(0.05):
            samples.append(int(pynvml.nvmlDeviceGetMemoryInfo(handle).used))
    except Exception as exc:  # noqa: BLE001 - preserved as runtime evidence
        identity["query_error"] = type(exc).__name__


def _mode_spec(payload: dict[str, Any], mode: str) -> dict[str, Any]:
    raw_modes = payload.get("mode_inputs")
    if not isinstance(raw_modes, dict) or not isinstance(raw_modes.get(mode), dict):
        raise ValueError(f"input cell must contain mode_inputs.{mode}")
    spec = dict(raw_modes[mode])
    if _contains_remote_url(spec):
        raise ValueError("offline Edge worker prohibits HTTP(S) conditioning inputs")
    if not str(spec.get("name") or "").strip():
        spec["name"] = mode
    expected_modes = {
        "forward_dynamics": {"forward_dynamics"},
        "inverse_dynamics": {"inverse_dynamics"},
        "reasoning": {"reasoning", "reasoner", "understanding"},
    }[mode]
    if str(spec.get("model_mode") or "").strip().lower() not in expected_modes:
        raise ValueError(f"mode_inputs.{mode}.model_mode is incompatible")
    return spec


def _find_output(root: Path, filename: str, *, reasoning: bool = False) -> Path | None:
    matches = sorted(root.rglob(filename))
    if not matches and reasoning:
        matches = sorted(root.rglob("sample_outputs.json"))
    return matches[0] if len(matches) == 1 else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mode", choices=tuple(MODE_OUTPUT), required=True)
    parser.add_argument("--cell-id", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--code-revision", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--configuration-sha256", required=True)
    args = parser.parse_args()
    started = time.monotonic()
    blockers: list[str] = []
    outputs: list[dict[str, Any]] = []
    stdout = ""
    stderr = ""
    returncode: int | None = None
    gpu_samples: list[int] = []
    gpu_identity: dict[str, Any] = {}
    stop = threading.Event()
    sampler = threading.Thread(
        target=_gpu_sampler, args=(stop, gpu_samples, gpu_identity), daemon=True
    )
    sampler.start()
    try:
        config = json.loads(args.config.read_text(encoding="utf-8"))
        cell = json.loads(args.input.read_text(encoding="utf-8"))
        if _sha(args.checkpoint) != args.checkpoint_sha256:
            raise ValueError("checkpoint SHA-256 mismatch")
        spec = _mode_spec(cell, args.mode)
        spec_path = args.output_dir / f"{args.cell_id}_{args.mode}_cosmos_input.json"
        spec_path.parent.mkdir(parents=True, exist_ok=True)
        spec_path.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        inference_root = args.output_dir / "cosmos_framework_output"
        command = [
            sys.executable,
            "-m",
            "cosmos_framework.scripts.inference",
            "--parallelism-preset=latency",
            "-i",
            str(spec_path),
            "-o",
            str(inference_root),
            "--checkpoint-path",
            str(args.checkpoint),
            "--seed",
            str(int(config.get("seed", 0))),
        ]
        if args.mode == "forward_dynamics":
            resolution = str(config.get("size", "480p")).lower().removesuffix("p")
            command.extend(
                [
                    "--resolution",
                    resolution,
                    "--num-frames",
                    str(int(config["num_frames"])),
                    "--fps",
                    str(float(config["fps"])),
                    "--num-steps",
                    str(int(config.get("inference_steps", 50))),
                    "--guidance",
                    str(float(config.get("guidance", 5.0))),
                    "--shift",
                    str(float(config.get("shift", 3.0))),
                    "--sampler",
                    str(config.get("sampler", "unipc")),
                ]
            )
        completed = subprocess.run(command, check=False, text=True, capture_output=True)
        returncode = completed.returncode
        stdout = completed.stdout or ""
        stderr = completed.stderr or ""
        if completed.returncode != 0:
            blockers.append("cosmos_framework_inference_failed")
        kind, filename = MODE_OUTPUT[args.mode]
        artifact = _find_output(
            inference_root,
            filename,
            reasoning=args.mode == "reasoning",
        )
        if artifact is None or not artifact.is_file() or artifact.stat().st_size == 0:
            blockers.append(f"cosmos_framework_required_output_missing:{kind}")
        else:
            outputs.append(
                {
                    "kind": kind,
                    "path": str(artifact.resolve()),
                    "metadata": {
                        "cell_id": args.cell_id,
                        "mode": args.mode,
                        "cosmos_input_sha256": _sha(spec_path),
                    },
                }
            )
    except Exception as exc:  # noqa: BLE001 - worker must retain failure evidence
        blockers.append(f"cosmos3_edge_worker_exception:{type(exc).__name__}:{exc}")
    finally:
        stop.set()
        sampler.join(timeout=1.0)
    report = {
        "schema_version": "cosmos3_edge_worker_result.v1",
        "status": "completed" if not blockers else "blocked",
        "mode": args.mode,
        "model_id": MODEL_ID,
        "parameter_count_billion": 4,
        "model_revision": args.model_revision,
        "code_revision": args.code_revision,
        "checkpoint_sha256": args.checkpoint_sha256,
        "configuration_sha256": args.configuration_sha256,
        "input_sha256": _sha(args.input),
        "runtime": {
            "python_version": sys.version.split()[0],
            "package_versions": _package_versions(),
            "gpu_identity": gpu_identity,
        },
        "outputs": outputs,
        "metrics": {
            "wall_seconds": round(time.monotonic() - started, 6),
            "peak_vram_bytes": max(gpu_samples, default=None),
            "gpu_memory_samples": len(gpu_samples),
            "framework_returncode": returncode,
        },
        "grounding": {"status": "not_measured_requires_blueprint_evaluator"},
        "abstention": {"status": "not_measured_requires_blueprint_evaluator"},
        "stdout_tail": stdout[-4000:],
        "stderr_tail": stderr[-4000:],
        "blockers": blockers,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if not blockers else 2


if __name__ == "__main__":
    raise SystemExit(main())
