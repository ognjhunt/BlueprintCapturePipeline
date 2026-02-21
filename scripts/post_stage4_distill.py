#!/usr/bin/env python3
"""Post-Stage-4 pseudo-view distillation into refined Gaussian outputs."""

from __future__ import annotations

import argparse
import json
import os
import shutil
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

    refined_usdz = output_dir / "export_last_refined.usdz"
    refined_ply = output_dir / "export_last_refined.ply"
    refined_ingp = output_dir / "export_last_refined.ingp"
    for target in (refined_usdz, refined_ply, refined_ingp):
        _safe_unlink(target)

    report: Dict[str, Any] = {
        "schema_version": "v1",
        "generated_at": _utc_now_iso(),
        "status": "",
        "work_dir": str(work_dir),
        "overlay_replaced_count": int(replaced_count),
        "overlay_replaced_paths": replaced_paths[:200],
        "distill_iters": int(distill_iters),
        "max_n_gaussians": int(max_n_gaussians),
        "time_budget_min": int(time_budget_min),
    }
    refined_metrics: Dict[str, Any] = {}

    if replaced_count <= 0:
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
    parser.add_argument("--distill-iters", type=int, default=int(os.getenv("POST_STAGE4_DISTILL_ITERS", "1600")))
    parser.add_argument("--max-n-gaussians", type=int, default=int(os.getenv("MAX_N_GAUSSIANS", "0")))
    parser.add_argument("--time-budget-min", type=int, default=int(os.getenv("POST_STAGE4_TIME_BUDGET_MIN", "90")))
    parser.add_argument("--threedgrut-python", default=os.getenv("THREEDGRUT_PYTHON", "python3.11"))
    parser.add_argument("--threedgrut-dir", default=os.getenv("THREEDGRUT_DIR", "/opt/3dgrut"))
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_ingp_path = Path(args.base_ingp) if str(args.base_ingp).strip() else None

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
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
