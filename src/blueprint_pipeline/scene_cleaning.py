"""Candidate-scoped scene cleaning orchestration (Inpaint360GS)."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from .common import relative_scene_path, resolve_gs_uri_to_path, utc_now_iso

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER_SCRIPT = REPO_ROOT / "scripts" / "inpaint360gs_runner.py"


def _env_int(name: str, default: int) -> int:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _normalize_mode(raw: str) -> str:
    mode = (raw or "").strip().lower()
    if mode in {"off", "auto", "force"}:
        return mode
    return "off"


def _stage_result(
    *,
    mode: str,
    status: str,
    reason: str,
    target_object_ids: Sequence[str],
    target_instance_ids: Sequence[int],
    details: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "schema_version": "v1",
        "generated_at": utc_now_iso(),
        "mode": mode,
        "status": status,
        "reason": reason,
        "target_object_ids": list(target_object_ids),
        "target_instance_ids": [int(v) for v in target_instance_ids],
    }
    if details:
        payload["details"] = dict(details)
    return payload


def _auto_or_force(
    *,
    mode: str,
    reason: str,
    target_object_ids: Sequence[str],
    target_instance_ids: Sequence[int],
    details: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    status = "failed" if mode == "force" else "skipped"
    return _stage_result(
        mode=mode,
        status=status,
        reason=reason,
        target_object_ids=target_object_ids,
        target_instance_ids=target_instance_ids,
        details=details,
    )


def _resolve_required_artifacts(
    *,
    storage_root: Path,
    nurec_outputs: Mapping[str, Any],
) -> Dict[str, Path]:
    artifacts = nurec_outputs.get("artifacts") if isinstance(nurec_outputs.get("artifacts"), Mapping) else {}
    if not isinstance(artifacts, Mapping):
        return {}

    out: Dict[str, Path] = {}
    for key in (
        "visual_usdz",
        "sam3_instance_masks_dir",
        "colmap_undistorted_sparse_dir",
        "colmap_undistorted_images_dir",
    ):
        uri = str(artifacts.get(key) or "").strip()
        if not uri:
            continue
        try:
            out[key] = resolve_gs_uri_to_path(uri, storage_root)
        except Exception:
            continue
    return out


def _resolve_target_instance_ids(
    *,
    object_index_path: Path,
    swap_candidates: Sequence[Mapping[str, Any]],
) -> tuple[list[str], list[int]]:
    payload = json.loads(object_index_path.read_text(encoding="utf-8"))
    objects = payload.get("objects") if isinstance(payload.get("objects"), list) else []
    by_object_id: Dict[str, int] = {}
    for obj in objects:
        if not isinstance(obj, Mapping):
            continue
        object_id = str(obj.get("id") or "").strip()
        if not object_id:
            continue
        try:
            instance_mask_id = int(obj.get("instance_mask_id"))
        except (TypeError, ValueError):
            continue
        if instance_mask_id > 0:
            by_object_id[object_id] = instance_mask_id

    target_object_ids: list[str] = []
    target_instance_ids: list[int] = []
    for candidate in swap_candidates:
        if not isinstance(candidate, Mapping):
            continue
        object_id = str(candidate.get("object_id") or "").strip()
        if not object_id:
            continue
        target_object_ids.append(object_id)
        instance_mask_id = by_object_id.get(object_id)
        if instance_mask_id is not None:
            target_instance_ids.append(int(instance_mask_id))

    # Deduplicate while preserving deterministic order.
    dedup_object_ids = list(dict.fromkeys(target_object_ids))
    dedup_instance_ids = sorted(set(target_instance_ids))
    return dedup_object_ids, dedup_instance_ids


def run_scene_cleaning(
    *,
    storage_root: Path,
    bucket: str,
    pipeline_dir: Path,
    nurec_outputs: Mapping[str, Any],
    swap_candidates: Sequence[Mapping[str, Any]],
    mode: str,
    resume: bool,
) -> Dict[str, Any]:
    """Run candidate-scoped Inpaint360GS scene cleaning.

    Returns a structured report with status=ok|skipped|failed.
    """
    resolved_mode = _normalize_mode(mode)
    if resolved_mode == "off":
        return _stage_result(
            mode=resolved_mode,
            status="skipped",
            reason="scene_cleaning_mode_off",
            target_object_ids=[],
            target_instance_ids=[],
        )

    resolved = _resolve_required_artifacts(storage_root=storage_root, nurec_outputs=nurec_outputs)
    missing = [
        key
        for key in ("visual_usdz", "sam3_instance_masks_dir", "colmap_undistorted_sparse_dir", "colmap_undistorted_images_dir")
        if key not in resolved
    ]
    if missing:
        return _auto_or_force(
            mode=resolved_mode,
            reason=f"missing_required_artifacts:{','.join(missing)}",
            target_object_ids=[],
            target_instance_ids=[],
        )

    visual_usdz = resolved["visual_usdz"]
    nurec_dir = visual_usdz.parent
    object_index_path = nurec_dir / "object_point_cloud_index.json"
    if not object_index_path.is_file():
        return _auto_or_force(
            mode=resolved_mode,
            reason=f"missing_object_index:{object_index_path}",
            target_object_ids=[],
            target_instance_ids=[],
        )

    target_object_ids, target_instance_ids = _resolve_target_instance_ids(
        object_index_path=object_index_path,
        swap_candidates=swap_candidates,
    )
    if not target_instance_ids:
        return _auto_or_force(
            mode=resolved_mode,
            reason="no_candidate_instance_mask_ids",
            target_object_ids=target_object_ids,
            target_instance_ids=target_instance_ids,
        )

    if not RUNNER_SCRIPT.is_file():
        return _auto_or_force(
            mode=resolved_mode,
            reason=f"runner_script_missing:{RUNNER_SCRIPT}",
            target_object_ids=target_object_ids,
            target_instance_ids=target_instance_ids,
        )

    python_bin = os.getenv("INPAINT360GS_PYTHON", "python3.10").strip() or "python3.10"
    probe_cmd = [python_bin, str(RUNNER_SCRIPT), "--probe"]
    probe = subprocess.run(probe_cmd, check=False, text=True, capture_output=True)
    if probe.returncode != 0:
        return _auto_or_force(
            mode=resolved_mode,
            reason="inpaint360gs_probe_failed",
            target_object_ids=target_object_ids,
            target_instance_ids=target_instance_ids,
            details={
                "stdout_tail": (probe.stdout or "")[-1200:],
                "stderr_tail": (probe.stderr or "")[-1200:],
            },
        )

    resolution = max(1, _env_int("INPAINT360GS_RESOLUTION", 2))
    target_arg = ",".join(str(v) for v in sorted(set(target_instance_ids)))
    cmd = [
        python_bin,
        str(RUNNER_SCRIPT),
        "--colmap-sparse-dir",
        str(resolved["colmap_undistorted_sparse_dir"]),
        "--images-dir",
        str(resolved["colmap_undistorted_images_dir"]),
        "--instance-masks-dir",
        str(resolved["sam3_instance_masks_dir"]),
        "--object-index",
        str(object_index_path),
        "--output-dir",
        str(nurec_dir),
        "--resolution",
        str(resolution),
        "--target-instance-ids",
        target_arg,
    ]
    if resume:
        cmd.append("--resume")

    run_proc = subprocess.run(cmd, check=False, text=True, capture_output=True)
    report_path = nurec_dir / "scene_cleaning_report.json"
    report_payload: Dict[str, Any] = {}
    if report_path.is_file():
        try:
            report_payload = json.loads(report_path.read_text(encoding="utf-8"))
        except Exception:
            report_payload = {}

    if run_proc.returncode != 0:
        return _auto_or_force(
            mode=resolved_mode,
            reason="scene_cleaning_runner_failed",
            target_object_ids=target_object_ids,
            target_instance_ids=target_instance_ids,
            details={
                "stdout_tail": (run_proc.stdout or "")[-1200:],
                "stderr_tail": (run_proc.stderr or "")[-1200:],
                "runner_report": report_payload,
            },
        )

    cleaned_glb = nurec_dir / "inpainted_visual_mesh.glb"
    if not cleaned_glb.is_file() or cleaned_glb.stat().st_size <= 0:
        return _auto_or_force(
            mode=resolved_mode,
            reason="scene_cleaning_missing_output_glb",
            target_object_ids=target_object_ids,
            target_instance_ids=target_instance_ids,
            details={"runner_report": report_payload},
        )

    cleaned_uri = f"gs://{bucket}/{relative_scene_path(cleaned_glb, storage_root)}"
    scene_report_uri = f"gs://{bucket}/{relative_scene_path(pipeline_dir / 'scene_cleaning_report.json', storage_root)}"

    # Inpainted Gaussian PLY (optional — present if runner copied it successfully)
    cleaned_ply = nurec_dir / "inpainted_gaussian_splat.ply"
    cleaned_ply_uri = (
        f"gs://{bucket}/{relative_scene_path(cleaned_ply, storage_root)}"
        if cleaned_ply.is_file() and cleaned_ply.stat().st_size > 0
        else None
    )

    details_dict: Dict[str, Any] = {
        "inpainted_visual_mesh_glb": str(cleaned_glb),
        "inpainted_visual_mesh_glb_uri": cleaned_uri,
        "runner_report": report_payload,
        "scene_cleaning_report_uri": scene_report_uri,
        "runner_stdout_tail": (run_proc.stdout or "")[-1200:],
        "runner_stderr_tail": (run_proc.stderr or "")[-1200:],
    }
    if cleaned_ply.is_file():
        details_dict["inpainted_gaussian_ply"] = str(cleaned_ply)
    if cleaned_ply_uri:
        details_dict["inpainted_gaussian_ply_uri"] = cleaned_ply_uri

    return _stage_result(
        mode=resolved_mode,
        status="ok",
        reason="scene_cleaning_completed",
        target_object_ids=target_object_ids,
        target_instance_ids=target_instance_ids,
        details=details_dict,
    )
