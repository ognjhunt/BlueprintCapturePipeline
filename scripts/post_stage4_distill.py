#!/usr/bin/env python3
"""Post-Stage-4 pseudo-view distillation into refined Gaussian outputs."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _safe_unlink(path: Path) -> None:
    try:
        path.unlink()
    except Exception:
        pass


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except Exception:
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _build_render_index_to_frame_map(undistorted_images_dir: Path) -> Dict[str, str]:
    """Build mapping from 3DGRUT render index names to original frame filenames.

    3DGRUT renders are 0-indexed PNGs (00000.png, 00001.png, ...) ordered by
    the same training image order that COLMAP uses during reconstruction.
    """
    frame_files = sorted(
        f.name
        for f in undistorted_images_dir.iterdir()
        if f.is_file() and f.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")
    )
    mapping: Dict[str, str] = {}
    for idx, frame_name in enumerate(frame_files):
        render_name = f"{idx:05d}.png"
        mapping[render_name] = frame_name
    return mapping


def _build_render_index_to_frame_map_with_colmap(
    undistorted_images_dir: Path,
    sparse_dir: Path,
) -> Dict[str, str] | None:
    """Prefer mapping using COLMAP image registration order when available."""
    images_txt = sparse_dir / "images.txt"
    if not images_txt.is_file():
        return None

    try:
        lines = images_txt.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return None

    registered_images: list[str] = []
    idx = 0
    while idx < len(lines):
        text = lines[idx].strip()
        idx += 1
        if not text or text.startswith("#"):
            continue
        parts = text.split()
        if len(parts) >= 10:
            registered_images.append(parts[9])
            # Skip the following 2D-point line.
            idx += 1

    if not registered_images:
        return None

    mapping: Dict[str, str] = {}
    available = {f.name: f.name for f in undistorted_images_dir.iterdir() if f.is_file()}
    for idx, source_name in enumerate(registered_images):
        render_name = f"{idx:05d}.png"
        source_name = source_name.strip()
        if not source_name:
            continue
        basename = Path(source_name).name
        frame_name = available.get(basename)
        if frame_name is not None:
            mapping[render_name] = frame_name

    return mapping if mapping else None


def _copy_matching_repaired_views(
    *,
    undistorted_images_dir: Path,
    repaired_views_dir: Path,
    accepted_views_jsonl: Path,
    sparse_dir: Path | None = None,
) -> Tuple[int, List[str]]:
    accepted = _load_jsonl(accepted_views_jsonl)
    replaced: List[str] = []

    # Prefer COLMAP-ordered image mapping to avoid index drift on filtered input.
    render_to_frame: Dict[str, str] = {}
    if sparse_dir is not None:
        render_to_frame.update(_build_render_index_to_frame_map_with_colmap(undistorted_images_dir, sparse_dir) or {})
    if not render_to_frame:
        render_to_frame = _build_render_index_to_frame_map(undistorted_images_dir)

    seen_sources: set = set()
    for row in accepted:
        if bool(row.get("is_virtual")):
            # Virtual repaired views are appended as new cameras, not overlaid onto real captures.
            continue
        source_name = str(row.get("source_image") or "").strip()
        repaired_path = Path(str(row.get("repaired_image") or "").strip())
        if not source_name or not repaired_path.is_file():
            continue

        # Skip duplicates — multiple pseudo-views can share the same source.
        if source_name in seen_sources:
            continue
        seen_sources.add(source_name)

        # Translate render index name to original frame filename.
        source_basename = Path(source_name).name
        frame_name = render_to_frame.get(source_name) or render_to_frame.get(source_basename, source_name)

        dst = undistorted_images_dir / frame_name
        if not dst.is_file():
            # Fallback: try the original source_name directly.
            dst = undistorted_images_dir / source_name
            if not dst.is_file():
                matches = list(undistorted_images_dir.rglob(source_name))
                if matches:
                    dst = matches[0]
                else:
                    continue
        shutil.copy2(repaired_path, dst)
        replaced.append(str(dst))
    return len(replaced), replaced


def _read_metrics(result_dir: Path) -> Dict[str, Any]:
    metrics_path = result_dir / "metrics.json"
    if not metrics_path.is_file():
        return {}
    try:
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _run(cmd: List[str], *, cwd: Path | None = None, timeout_sec: int | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd is not None else None,
        text=True,
        capture_output=True,
        check=False,
        timeout=timeout_sec,
    )


def _run_colmap_model_converter(*, input_path: Path, output_path: Path, output_type: str) -> Tuple[bool, str]:
    cmd = [
        "colmap",
        "model_converter",
        "--input_path",
        str(input_path),
        "--output_path",
        str(output_path),
        "--output_type",
        str(output_type).upper(),
    ]
    try:
        proc = _run(cmd)
    except Exception as exc:
        return False, f"colmap_model_converter_exception:{exc}"
    if proc.returncode != 0:
        tail = ((proc.stderr or "")[-300:] or (proc.stdout or "")[-300:]).strip().replace("\n", " ")
        return False, f"colmap_model_converter_failed:{tail or 'unknown'}"
    return True, ""


def _ensure_sparse_text_model(sparse_dir: Path) -> Tuple[bool, str]:
    """Ensure sparse_dir has text model files, converting from BIN when needed."""
    required_txt = ("images.txt", "cameras.txt", "points3D.txt")
    if all((sparse_dir / name).is_file() for name in required_txt):
        return True, ""

    required_bin = ("images.bin", "cameras.bin", "points3D.bin")
    if not all((sparse_dir / name).is_file() for name in required_bin):
        return False, "sparse_model_missing_text_and_binary"

    tmp_txt = sparse_dir / "_tmp_txt_model"
    if tmp_txt.exists():
        shutil.rmtree(tmp_txt, ignore_errors=True)
    tmp_txt.mkdir(parents=True, exist_ok=True)
    try:
        ok, reason = _run_colmap_model_converter(input_path=sparse_dir, output_path=tmp_txt, output_type="TXT")
        if not ok:
            return False, reason
        for name in required_txt:
            src = tmp_txt / name
            if not src.is_file():
                return False, f"model_converter_missing_{name}"
            shutil.copy2(src, sparse_dir / name)
    finally:
        shutil.rmtree(tmp_txt, ignore_errors=True)
    return True, ""


def _regenerate_sparse_bin_model(sparse_dir: Path) -> Tuple[bool, str]:
    """Regenerate binary sparse model files from updated TXT files."""
    required_txt = ("images.txt", "cameras.txt", "points3D.txt")
    if not all((sparse_dir / name).is_file() for name in required_txt):
        return False, "sparse_text_model_missing_for_bin_regen"

    tmp_bin = sparse_dir / "_tmp_bin_model"
    if tmp_bin.exists():
        shutil.rmtree(tmp_bin, ignore_errors=True)
    tmp_bin.mkdir(parents=True, exist_ok=True)
    try:
        ok, reason = _run_colmap_model_converter(input_path=sparse_dir, output_path=tmp_bin, output_type="BIN")
        if not ok:
            return False, reason
        for name in ("images.bin", "cameras.bin", "points3D.bin"):
            src = tmp_bin / name
            if not src.is_file():
                return False, f"model_converter_missing_{name}"
            shutil.copy2(src, sparse_dir / name)
    finally:
        shutil.rmtree(tmp_bin, ignore_errors=True)
    return True, ""


def _read_first_camera_id_from_cameras_bin(path: Path) -> int | None:
    if not path.is_file():
        return None
    data = path.read_bytes()
    if len(data) < 12:
        return None
    try:
        (num_cameras,) = struct.unpack_from("<Q", data, 0)
        if num_cameras < 1:
            return None
        (camera_id,) = struct.unpack_from("<I", data, 8)
        return int(camera_id)
    except Exception:
        return None


def _resolve_primary_camera_id(sparse_dir: Path) -> int | None:
    cameras_txt = sparse_dir / "cameras.txt"
    if cameras_txt.is_file():
        for line in cameras_txt.read_text(encoding="utf-8", errors="ignore").splitlines():
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            parts = text.split()
            if not parts:
                continue
            try:
                return int(parts[0])
            except Exception:
                continue
    return _read_first_camera_id_from_cameras_bin(sparse_dir / "cameras.bin")


def _coerce_float_vec(value: Any, expected_len: int) -> List[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < expected_len:
        return None
    out: List[float] = []
    try:
        for idx in range(expected_len):
            out.append(float(value[idx]))
    except Exception:
        return None
    return out


def _read_max_image_id_from_images_txt(path: Path) -> int:
    """Find the highest IMAGE_ID in an images.txt file."""
    max_id = 0
    if not path.is_file():
        return max_id
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 10:
            try:
                max_id = max(max_id, int(parts[0]))
            except ValueError:
                pass
    return max_id


def _append_virtual_images_to_colmap(
    sparse_dir: Path,
    virtual_candidates: List[Dict[str, Any]],
    virtual_renders_dir: Path,
    images_dir: Path,
    starting_id: int,
    *,
    default_camera_id: int,
) -> Tuple[int, int]:
    """Append new virtual camera entries to images.txt and copy renders to images/.

    Legacy fallback path for backwards compatibility. Returns:
      (appended_count, skipped_missing_required_count)
    """
    images_txt = sparse_dir / "images.txt"
    if not images_txt.is_file():
        return 0, 0

    appended = 0
    skipped_missing_required = 0
    with images_txt.open("a", encoding="utf-8") as f:
        next_id = int(starting_id)
        for idx, cand in enumerate(virtual_candidates):
            render_name = str(cand.get("render_name") or f"{idx:05d}.png").strip()
            render_src_str = str(cand.get("render_image") or "").strip()
            render_src = Path(render_src_str) if render_src_str else (virtual_renders_dir / render_name)
            if not render_src.is_file():
                skipped_missing_required += 1
                continue

            qvec = _coerce_float_vec(cand.get("qvec"), 4)
            tvec = _coerce_float_vec(cand.get("tvec"), 3)
            if qvec is None or tvec is None:
                skipped_missing_required += 1
                continue

            next_id += 1
            dst_name = f"virtual_{idx:05d}.png"
            dst_path = images_dir / dst_name
            shutil.copy2(render_src, dst_path)

            camera_id = int(cand.get("camera_id", default_camera_id) or default_camera_id)
            f.write(
                f"{next_id} "
                f"{qvec[0]:.10f} {qvec[1]:.10f} {qvec[2]:.10f} {qvec[3]:.10f} "
                f"{tvec[0]:.10f} {tvec[1]:.10f} {tvec[2]:.10f} "
                f"{camera_id} {dst_name}\n"
            )
            f.write("\n")  # Empty 2D points line
            appended += 1

    return appended, skipped_missing_required


def _append_virtual_from_accepted_rows(
    *,
    sparse_dir: Path,
    accepted_views_jsonl: Path,
    images_dir: Path,
    starting_id: int,
    default_camera_id: int,
) -> Tuple[int, int]:
    """Append virtual camera rows from accepted repaired rows.

    Returns:
      (appended_count, missing_required_count)
    """
    images_txt = sparse_dir / "images.txt"
    if not images_txt.is_file():
        return 0, 0

    rows = _load_jsonl(accepted_views_jsonl)
    virtual_rows = [row for row in rows if bool(row.get("is_virtual"))]
    if not virtual_rows:
        return 0, 0

    appended = 0
    missing_required = 0
    next_id = int(starting_id)
    with images_txt.open("a", encoding="utf-8") as f:
        for idx, row in enumerate(virtual_rows):
            repaired_path = Path(str(row.get("repaired_image") or "").strip())
            qvec = _coerce_float_vec(row.get("qvec"), 4)
            tvec = _coerce_float_vec(row.get("tvec"), 3)
            if not repaired_path.is_file() or qvec is None or tvec is None:
                missing_required += 1
                continue

            next_id += 1
            dst_name = f"virtual_{idx:05d}.png"
            dst_path = images_dir / dst_name
            shutil.copy2(repaired_path, dst_path)

            camera_id = int(row.get("camera_id", default_camera_id) or default_camera_id)
            f.write(
                f"{next_id} "
                f"{qvec[0]:.10f} {qvec[1]:.10f} {qvec[2]:.10f} {qvec[3]:.10f} "
                f"{tvec[0]:.10f} {tvec[1]:.10f} {tvec[2]:.10f} "
                f"{camera_id} {dst_name}\n"
            )
            f.write("\n")
            appended += 1

    return appended, missing_required


def _find_baseline_checkpoint(output_dir: Path) -> Path | None:
    """Find the baseline 3DGRUT checkpoint from Stage 4 to resume from."""
    grut_dir = output_dir / "3dgrut"
    if not grut_dir.is_dir():
        return None
    candidates = sorted(grut_dir.rglob("ckpt_last.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def _build_default_distill_cmd(
    *,
    threedgrut_python: str,
    threedgrut_dir: Path,
    dataset_dir: Path,
    out_dir: Path,
    distill_iters: int,
    max_n_gaussians: int,
    baseline_checkpoint: Path | None = None,
) -> List[str]:
    train_script = threedgrut_dir / "train.py"
    cmd: List[str] = [
        threedgrut_python,
        str(train_script),
        "--config-name",
        "apps/colmap_3dgut_mcmc",
        f"path={dataset_dir}/",
        f"out_dir={out_dir}/",
        "experiment_name=post_stage4_refine",
        "export_usdz.enabled=true",
        "export_usdz.apply_normalizing_transform=true",
        "export_ply.enabled=true",
        f"n_iterations={max(1, distill_iters)}",
        "with_gui=false",
        "with_viser_gui=false",
        "num_workers=4",
    ]
    if max_n_gaussians > 0:
        cmd.append(f"strategy.add.max_n_gaussians={int(max_n_gaussians)}")
    if baseline_checkpoint is not None and baseline_checkpoint.is_file():
        cmd.append(f"resume={baseline_checkpoint}")
    return cmd


def _find_latest_result(result_root: Path) -> Path | None:
    candidates = sorted(
        result_root.rglob("export_last.ply"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        return None
    return candidates[0].parent


def run_post_stage4_distill(
    *,
    output_dir: Path,
    undistorted_dir: Path,
    base_usdz: Path,
    base_ply: Path,
    base_ingp: Path | None,
    accepted_views_jsonl: Path,
    repaired_views_dir: Path,
    distill_iters: int,
    max_n_gaussians: int,
    time_budget_min: int,
    threedgrut_python: str,
    threedgrut_dir: Path,
    virtual_renders_dir: Path | None = None,
    virtual_candidates_jsonl: Path | None = None,
) -> Dict[str, Any]:
    started = time.time()
    work_dir = output_dir / "post_stage4_distill"
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    dataset_dir = work_dir / "undistorted_refine"
    shutil.copytree(undistorted_dir, dataset_dir)

    undistorted_images_dir = dataset_dir / "images"
    replaced_count, replaced_paths = _copy_matching_repaired_views(
        undistorted_images_dir=undistorted_images_dir,
        repaired_views_dir=repaired_views_dir,
        accepted_views_jsonl=accepted_views_jsonl,
        sparse_dir=undistorted_dir / "sparse" / "0",
    )

    # Append virtual camera entries (prefer accepted repaired virtual rows).
    virtual_appended = 0
    virtual_append_failed_reason = ""
    resolved_primary_camera_id: int | None = None
    used_legacy_virtual_fallback = False
    has_legacy_virtual_inputs = (
        virtual_renders_dir is not None
        and virtual_candidates_jsonl is not None
        and virtual_renders_dir.is_dir()
        and virtual_candidates_jsonl.is_file()
    )
    accepted_rows = _load_jsonl(accepted_views_jsonl)
    accepted_virtual_rows = [row for row in accepted_rows if bool(row.get("is_virtual"))]
    if accepted_virtual_rows or has_legacy_virtual_inputs:
        sparse_copy = dataset_dir / "sparse" / "0"
        ok_text, text_reason = _ensure_sparse_text_model(sparse_copy)
        if not ok_text:
            virtual_append_failed_reason = text_reason
        else:
            images_txt = sparse_copy / "images.txt"
            max_id = _read_max_image_id_from_images_txt(images_txt)
            resolved_primary_camera_id = _resolve_primary_camera_id(sparse_copy)
            default_camera_id = int(resolved_primary_camera_id or 1)

            accepted_appended = 0
            accepted_missing_required = 0
            if accepted_virtual_rows:
                accepted_appended, accepted_missing_required = _append_virtual_from_accepted_rows(
                    sparse_dir=sparse_copy,
                    accepted_views_jsonl=accepted_views_jsonl,
                    images_dir=undistorted_images_dir,
                    starting_id=max_id,
                    default_camera_id=default_camera_id,
                )
                virtual_appended += int(accepted_appended)
                max_id += int(accepted_appended)

            if (
                virtual_appended == 0
                and accepted_virtual_rows
                and accepted_missing_required > 0
                and has_legacy_virtual_inputs
                and virtual_renders_dir is not None
                and virtual_candidates_jsonl is not None
            ):
                used_legacy_virtual_fallback = True
                virtual_cands = _load_jsonl(virtual_candidates_jsonl)
                virtual_cands = [c for c in virtual_cands if c.get("is_virtual")]
                legacy_appended, _legacy_missing_required = _append_virtual_images_to_colmap(
                    sparse_copy,
                    virtual_cands,
                    virtual_renders_dir,
                    undistorted_images_dir,
                    max_id,
                    default_camera_id=default_camera_id,
                )
                virtual_appended += int(legacy_appended)
                max_id += int(legacy_appended)
                if legacy_appended == 0:
                    virtual_append_failed_reason = (
                        "virtual_accepted_rows_missing_required_fields_and_legacy_fallback_failed"
                    )
            elif virtual_appended == 0 and accepted_virtual_rows and accepted_missing_required > 0:
                virtual_append_failed_reason = "virtual_accepted_rows_missing_required_fields"

            if virtual_appended > 0:
                ok_bin, bin_reason = _regenerate_sparse_bin_model(sparse_copy)
                if not ok_bin:
                    virtual_append_failed_reason = bin_reason
                    virtual_appended = 0

    refined_usdz = output_dir / "export_last_refined.usdz"
    refined_ply = output_dir / "export_last_refined.ply"
    refined_ingp = output_dir / "export_last_refined.ingp"
    for target in (refined_usdz, refined_ply, refined_ingp):
        _safe_unlink(target)

    report: Dict[str, Any] = {
        "schema_version": "v1",
        "generated_at": _utc_now_iso(),
        "status": "",
        "distill_ok": False,
        "work_dir": str(work_dir),
        "overlay_replaced_count": int(replaced_count),
        "overlay_replaced_paths": replaced_paths[:200],
        "virtual_appended_count": int(virtual_appended),
        "virtual_append_failed_reason": str(virtual_append_failed_reason or ""),
        "used_legacy_virtual_fallback": bool(used_legacy_virtual_fallback),
        "resume_disabled_reason": "",
        "resolved_primary_camera_id": int(resolved_primary_camera_id) if resolved_primary_camera_id is not None else None,
        "distill_iters": int(distill_iters),
        "max_n_gaussians": int(max_n_gaussians),
        "time_budget_min": int(time_budget_min),
    }
    refined_metrics: Dict[str, Any] = {}

    total_new_data = replaced_count + virtual_appended
    if virtual_append_failed_reason:
        shutil.copy2(base_usdz, refined_usdz)
        shutil.copy2(base_ply, refined_ply)
        if base_ingp is not None and base_ingp.is_file():
            shutil.copy2(base_ingp, refined_ingp)
        report["status"] = "fallback_baseline_copy_virtual_append_failed"
        report["result_dir"] = ""
    elif total_new_data <= 0:
        shutil.copy2(base_usdz, refined_usdz)
        shutil.copy2(base_ply, refined_ply)
        if base_ingp is not None and base_ingp.is_file():
            shutil.copy2(base_ingp, refined_ingp)
        report["status"] = "skipped_no_matching_repaired_views"
        report["result_dir"] = ""
    else:
        out_dir = work_dir / "distill_run"
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd_template = os.getenv("POST_STAGE4_DISTILL_COMMAND", "").strip()

        baseline_ckpt = _find_baseline_checkpoint(output_dir)
        # When virtual views are added, camera count changes — can't resume from checkpoint
        if virtual_appended > 0:
            baseline_ckpt = None
            distill_iters = max(distill_iters, 5000)
            report["resume_disabled_reason"] = "virtual_views_appended_camera_count_changed"
        report["baseline_checkpoint"] = str(baseline_ckpt) if baseline_ckpt else None

        if cmd_template:
            cmd = cmd_template.format(
                dataset_dir=str(dataset_dir),
                out_dir=str(out_dir),
                n_iterations=str(int(distill_iters)),
                max_n_gaussians=str(int(max_n_gaussians)),
            )
            proc = subprocess.run(cmd, shell=True, text=True, capture_output=True, check=False)
            report["command"] = cmd
        else:
            cmd_list = _build_default_distill_cmd(
                threedgrut_python=threedgrut_python,
                threedgrut_dir=threedgrut_dir,
                dataset_dir=dataset_dir,
                out_dir=out_dir,
                distill_iters=distill_iters,
                max_n_gaussians=max_n_gaussians,
                baseline_checkpoint=baseline_ckpt,
            )
            proc = _run(cmd_list, cwd=threedgrut_dir, timeout_sec=max(60, int(time_budget_min) * 60))
            report["command"] = " ".join(cmd_list)

        report["command_return_code"] = int(proc.returncode)
        report["command_stdout_tail"] = (proc.stdout or "")[-4000:]
        report["command_stderr_tail"] = (proc.stderr or "")[-4000:]

        result_dir = _find_latest_result(out_dir)
        if proc.returncode == 0 and result_dir is not None:
            usdz_src = result_dir / "export_last.usdz"
            ply_src = result_dir / "export_last.ply"
            ingp_src = result_dir / "export_last.ingp"
            if usdz_src.is_file() and ply_src.is_file():
                shutil.copy2(usdz_src, refined_usdz)
                shutil.copy2(ply_src, refined_ply)
                if ingp_src.is_file():
                    shutil.copy2(ingp_src, refined_ingp)
                report["status"] = "ok"
                report["distill_ok"] = True
                report["result_dir"] = str(result_dir)
                refined_metrics = _read_metrics(result_dir)
            else:
                report["status"] = "fallback_baseline_copy_missing_distill_exports"
                report["result_dir"] = str(result_dir)
        else:
            report["status"] = "fallback_baseline_copy_distill_failed"
            report["result_dir"] = str(result_dir) if result_dir is not None else ""

        if not refined_usdz.is_file() or not refined_ply.is_file():
            shutil.copy2(base_usdz, refined_usdz)
            shutil.copy2(base_ply, refined_ply)
            if base_ingp is not None and base_ingp.is_file():
                shutil.copy2(base_ingp, refined_ingp)

    report["refined_usdz"] = str(refined_usdz)
    report["refined_ply"] = str(refined_ply)
    report["refined_ingp"] = str(refined_ingp) if refined_ingp.is_file() else ""
    report["refined_metrics"] = refined_metrics
    report["refined_usdz_bytes"] = int(refined_usdz.stat().st_size) if refined_usdz.is_file() else 0
    report["refined_ply_bytes"] = int(refined_ply.stat().st_size) if refined_ply.is_file() else 0
    report["distill_ok"] = bool(str(report.get("status", "")).strip().lower() == "ok")
    report["elapsed_sec"] = float(time.time() - started)

    report_path = output_dir / "post_stage4_distill_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Distill repaired pseudo-views into refined 3DGRUT outputs")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--undistorted-dir", required=True)
    parser.add_argument("--base-usdz", required=True)
    parser.add_argument("--base-ply", required=True)
    parser.add_argument("--base-ingp", default="")
    parser.add_argument("--accepted-views-jsonl", required=True)
    parser.add_argument("--repaired-views-dir", required=True)
    parser.add_argument("--distill-iters", type=int, default=int(os.getenv("POST_STAGE4_DISTILL_ITERS", "3000")))
    parser.add_argument("--max-n-gaussians", type=int, default=int(os.getenv("MAX_N_GAUSSIANS", "0")))
    parser.add_argument("--time-budget-min", type=int, default=int(os.getenv("POST_STAGE4_TIME_BUDGET_MIN", "90")))
    parser.add_argument("--threedgrut-python", default=os.getenv("THREEDGRUT_PYTHON", "python3.11"))
    parser.add_argument("--threedgrut-dir", default=os.getenv("THREEDGRUT_DIR", "/opt/3dgrut"))
    parser.add_argument("--virtual-renders-dir", default="", help="Directory with virtual camera renders (from post_stage4_virtual_render.py)")
    parser.add_argument("--virtual-candidates-jsonl", default="", help="JSONL with virtual candidates (is_virtual=True entries)")
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_ingp_path = Path(args.base_ingp) if str(args.base_ingp).strip() else None

    virtual_renders = Path(args.virtual_renders_dir) if str(args.virtual_renders_dir).strip() else None
    virtual_cands = Path(args.virtual_candidates_jsonl) if str(args.virtual_candidates_jsonl).strip() else None

    run_post_stage4_distill(
        output_dir=output_dir,
        undistorted_dir=Path(args.undistorted_dir),
        base_usdz=Path(args.base_usdz),
        base_ply=Path(args.base_ply),
        base_ingp=base_ingp_path,
        accepted_views_jsonl=Path(args.accepted_views_jsonl),
        repaired_views_dir=Path(args.repaired_views_dir),
        distill_iters=max(1, int(args.distill_iters)),
        max_n_gaussians=max(0, int(args.max_n_gaussians)),
        time_budget_min=max(1, int(args.time_budget_min)),
        threedgrut_python=str(args.threedgrut_python),
        threedgrut_dir=Path(args.threedgrut_dir),
        virtual_renders_dir=virtual_renders,
        virtual_candidates_jsonl=virtual_cands,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
