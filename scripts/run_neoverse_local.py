#!/usr/bin/env python3
"""Dispatch NeoVerse local GPU jobs and normalize outputs."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = REPO_ROOT / "src"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from neoverse_contract_adapter import normalize_backend_manifest


REQUIRED_STAGE1_ARTIFACTS = (
    "export_last.usdz",
    "nvblox_mesh.ply",
    "visual_mesh.glb",
    "mesh_manifest.json",
    "occupancy.bin",
    "object_point_cloud_index.json",
    "capture_quality_report.json",
)


def _load_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return dict(payload) if isinstance(payload, dict) else {}


def _env(name: str) -> str:
    return (os.getenv(name) or "").strip()


def _build_local_command(
    *,
    input_video: Path,
    output_dir: Path,
    scene_id: str,
    capture_id: str,
    job_spec_path: Path,
    result_manifest_path: Path,
) -> tuple[str | list[str] | None, bool, str]:
    template = _env("NEOVERSE_CMD_TEMPLATE")
    if template:
        try:
            command = template.format(
                INPUT_VIDEO=str(input_video),
                OUTPUT_DIR=str(output_dir),
                SCENE_ID=scene_id,
                CAPTURE_ID=capture_id,
                JOB_SPEC_PATH=str(job_spec_path),
                RESULT_MANIFEST_PATH=str(result_manifest_path),
            )
        except KeyError as exc:
            raise RuntimeError(
                f"invalid NEOVERSE_CMD_TEMPLATE: missing placeholder {exc}. "
                "Expected INPUT_VIDEO, OUTPUT_DIR, SCENE_ID, CAPTURE_ID, JOB_SPEC_PATH, RESULT_MANIFEST_PATH"
            ) from exc
        return command, True, "command_template"

    executable = _env("NEOVERSE_EXECUTABLE")
    if not executable:
        return None, False, ""
    try:
        command = shlex.split(executable)
    except ValueError as exc:
        raise RuntimeError(f"invalid NEOVERSE_EXECUTABLE shell syntax: {executable}") from exc
    if not command:
        return None, False, ""
    command.extend(
        [
            "--input-video",
            str(input_video),
            "--output-dir",
            str(output_dir),
            "--scene-id",
            scene_id,
            "--capture-id",
            capture_id,
            "--job-spec",
            str(job_spec_path),
            "--result-manifest",
            str(result_manifest_path),
        ]
    )
    return command, False, "executable"


def _run_local_command(command: str | list[str], *, use_shell: bool) -> int:
    try:
        completed = subprocess.run(command, shell=use_shell, text=True)
    except OSError as exc:
        raise RuntimeError(f"failed to launch NeoVerse local runtime: {exc}") from exc
    return int(completed.returncode)


def _manifest_candidates(output_dir: Path, explicit_path: Path) -> list[Path]:
    candidates = [explicit_path]
    for name in (
        "neoverse_result_manifest.json",
        "result_manifest.json",
        "native_result_manifest.json",
    ):
        candidate = output_dir / name
        if candidate not in candidates:
            candidates.append(candidate)
    return candidates


def _first_manifest_path(output_dir: Path, explicit_path: Path) -> Path | None:
    for candidate in _manifest_candidates(output_dir, explicit_path):
        if candidate.is_file():
            return candidate
    return None


def _missing_required_artifacts(output_dir: Path) -> list[str]:
    missing: list[str] = []
    for artifact_name in REQUIRED_STAGE1_ARTIFACTS:
        path = output_dir / artifact_name
        if not path.is_file() or path.stat().st_size <= 0:
            missing.append(artifact_name)
    return missing


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run NeoVerse local GPU runtime")
    parser.add_argument("--job-spec", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--capture-id", required=True)
    args = parser.parse_args(argv)

    job_spec_path = Path(args.job_spec)
    job_spec = _load_json(job_spec_path)
    capture = job_spec.get("capture")
    if not isinstance(capture, dict):
        raise SystemExit("NeoVerse job spec is missing capture payload")

    input_video = Path(str(capture.get("raw_video_path") or "").strip())
    if not input_video.is_file():
        raise SystemExit(f"NeoVerse raw video is missing: {input_video}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    backend_report_path = output_dir / "neoverse_backend_report.json"
    result_manifest_path = output_dir / "neoverse_result_manifest.json"

    command, use_shell, command_mode = _build_local_command(
        input_video=input_video,
        output_dir=output_dir,
        scene_id=args.scene_id,
        capture_id=args.capture_id,
        job_spec_path=job_spec_path,
        result_manifest_path=result_manifest_path,
    )
    if command is None:
        raise SystemExit(
            "NeoVerse local runtime is not configured; set NEOVERSE_CMD_TEMPLATE or NEOVERSE_EXECUTABLE"
        )

    return_code = _run_local_command(command, use_shell=use_shell)
    if return_code != 0:
        return return_code

    manifest_path = _first_manifest_path(output_dir, result_manifest_path)
    backend_report = {
        "schema_version": "v1",
        "backend": "neoverse",
        "status": "completed",
        "execution_mode": "local_gpu_runtime",
        "runtime_contract_version": "stage1_world_model_local_v1",
        "conditioning_used": ["rgb_video"],
        "runtime_invocation_mode": command_mode,
        "native_artifact_manifest_location": str(manifest_path) if manifest_path else None,
    }
    backend_report_path.write_text(json.dumps(backend_report, indent=2), encoding="utf-8")

    if manifest_path is not None:
        normalize_backend_manifest(
            result_manifest=_load_json(manifest_path),
            output_dir=output_dir,
            backend_report_path=backend_report_path,
        )

    missing = _missing_required_artifacts(output_dir)
    if missing:
        missing_text = ", ".join(missing)
        raise SystemExit(
            "NeoVerse local runtime completed without the required Stage 1 artifacts. "
            f"Missing: {missing_text}. Ensure the local wrapper writes the normalized contract directly "
            "or emits neoverse_result_manifest.json / result_manifest.json."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
