"""Standalone NuRec worker process.

This worker consumes ``nurec_job_spec.json`` and is responsible for:
1) executing the NuRec reconstruction command,
2) validating required artifacts,
3) writing ``.nurec_complete`` or ``.nurec_failed`` marker payloads.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from .common import ensure_dir, has_nonempty_file, resolve_gs_uri_to_path


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected object payload in {path}, got {type(payload).__name__}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(dict(payload), indent=2), encoding="utf-8")


def _build_paths(spec: Mapping[str, Any], storage_root: Path) -> Dict[str, Path]:
    outputs = spec.get("outputs") if isinstance(spec.get("outputs"), Mapping) else {}
    nurec_prefix = str(outputs.get("nurec_prefix") or "").strip()
    complete_marker_uri = str(outputs.get("complete_marker") or "").strip()
    failed_marker_uri = str(outputs.get("failed_marker") or "").strip()

    if not nurec_prefix or not complete_marker_uri or not failed_marker_uri:
        raise ValueError("job spec outputs must include nurec_prefix, complete_marker, failed_marker")

    nurec_dir = resolve_gs_uri_to_path(nurec_prefix, storage_root)
    complete_marker = resolve_gs_uri_to_path(complete_marker_uri, storage_root)
    failed_marker = resolve_gs_uri_to_path(failed_marker_uri, storage_root)

    return {
        "nurec_dir": nurec_dir,
        "complete_marker": complete_marker,
        "failed_marker": failed_marker,
    }


def _render_command(template: str, *, spec_path: Path, spec: Mapping[str, Any], nurec_dir: Path) -> str:
    capture = spec.get("capture") if isinstance(spec.get("capture"), Mapping) else {}
    return (
        template.replace("{JOB_SPEC_PATH}", str(spec_path))
        .replace("{NUREC_OUTPUT_DIR}", str(nurec_dir))
        .replace("{RAW_PREFIX_URI}", str(capture.get("raw_prefix_uri") or ""))
        .replace("{FRAMES_INDEX_URI}", str(capture.get("frames_index_uri") or ""))
        .replace("{ARKIT_POSES_URI}", str(capture.get("arkit_poses_uri") or ""))
        .replace("{ARKIT_INTRINSICS_URI}", str(capture.get("arkit_intrinsics_uri") or ""))
        .replace("{SCENE_ID}", str(spec.get("scene_id") or ""))
        .replace("{CAPTURE_ID}", str(spec.get("capture_id") or ""))
    )


def _run_nurec_pipeline(*, spec_path: Path, spec: Mapping[str, Any], nurec_dir: Path) -> Dict[str, Any]:
    command_template = (os.getenv("NUREC_PIPELINE_COMMAND") or "").strip()
    skip_command = (os.getenv("NUREC_SKIP_PIPELINE_COMMAND") or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    if not command_template and not skip_command:
        raise RuntimeError(
            "NUREC_PIPELINE_COMMAND is required (or set NUREC_SKIP_PIPELINE_COMMAND=true if artifacts are pre-generated)"
        )

    if skip_command:
        return {
            "executed": False,
            "command": "",
            "return_code": 0,
            "stdout": "",
            "stderr": "",
        }

    command = _render_command(command_template, spec_path=spec_path, spec=spec, nurec_dir=nurec_dir)

    proc = subprocess.run(
        command,
        shell=True,
        cwd=str(spec_path.parent),
        check=False,
        text=True,
        capture_output=True,
    )

    if proc.returncode != 0:
        raise RuntimeError(
            f"NuRec command failed with code {proc.returncode}: {proc.stderr[-1200:] or proc.stdout[-1200:]}"
        )

    return {
        "executed": True,
        "command": command,
        "return_code": proc.returncode,
        "stdout": proc.stdout[-4000:],
        "stderr": proc.stderr[-4000:],
    }


def _validate_outputs(nurec_dir: Path) -> Dict[str, Any]:
    export_usdz = nurec_dir / "export_last.usdz"
    mesh_ply = nurec_dir / "nvblox_mesh.ply"
    occupancy = sorted(nurec_dir.glob("occupancy*"))

    if not has_nonempty_file(export_usdz):
        raise RuntimeError(f"Missing required NuRec visual artifact: {export_usdz}")
    if not has_nonempty_file(mesh_ply):
        raise RuntimeError(f"Missing required NuRec collision mesh: {mesh_ply}")
    if not occupancy:
        raise RuntimeError(f"Missing required occupancy artifacts in {nurec_dir}")

    return {
        "visual_usdz": str(export_usdz),
        "collision_mesh_ply": str(mesh_ply),
        "occupancy": [str(path) for path in occupancy],
    }


def _marker_payload(
    *,
    spec: Mapping[str, Any],
    status: str,
    nurec_dir: Path,
    command_meta: Mapping[str, Any],
    outputs: Optional[Mapping[str, Any]] = None,
    error: Optional[str] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "schema_version": "v1",
        "scene_id": str(spec.get("scene_id") or ""),
        "capture_id": str(spec.get("capture_id") or ""),
        "status": status,
        "generated_at": _utc_now_iso(),
        "nurec_dir": str(nurec_dir),
        "command": dict(command_meta),
    }
    if outputs is not None:
        payload["outputs"] = dict(outputs)
    if error is not None:
        payload["error"] = error
    return payload


def run_job_spec(job_spec_path: Path, *, storage_root: Path) -> int:
    spec = _read_json(job_spec_path)
    paths = _build_paths(spec, storage_root)
    nurec_dir = paths["nurec_dir"]
    complete_marker = paths["complete_marker"]
    failed_marker = paths["failed_marker"]

    ensure_dir(nurec_dir)
    ensure_dir(complete_marker.parent)
    ensure_dir(failed_marker.parent)

    try:
        command_meta = _run_nurec_pipeline(spec_path=job_spec_path, spec=spec, nurec_dir=nurec_dir)
        outputs = _validate_outputs(nurec_dir)

        payload = _marker_payload(
            spec=spec,
            status="completed",
            nurec_dir=nurec_dir,
            command_meta=command_meta,
            outputs=outputs,
        )
        _write_json(complete_marker, payload)
        return 0

    except Exception as exc:
        command_meta = {
            "executed": False,
            "command": "",
            "return_code": 1,
            "stdout": "",
            "stderr": "",
        }
        payload = _marker_payload(
            spec=spec,
            status="failed",
            nurec_dir=nurec_dir,
            command_meta=command_meta,
            error=f"{exc}\n{traceback.format_exc(limit=20)}",
        )
        _write_json(failed_marker, payload)
        return 1


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="NuRec worker")
    parser.add_argument("--job-spec", required=True, help="Path to nurec_job_spec.json")
    parser.add_argument(
        "--storage-root",
        default=os.getenv("GCS_ROOT", "/mnt/gcs"),
        help="Mounted storage root used to resolve gs:// URIs",
    )
    args = parser.parse_args(argv)

    return run_job_spec(Path(args.job_spec), storage_root=Path(args.storage_root))


if __name__ == "__main__":
    raise SystemExit(main())
