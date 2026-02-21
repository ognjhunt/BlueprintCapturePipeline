#!/usr/bin/env python3
"""Post-Stage-4 pseudo-view repair using Fixer/GSFix3D backends.

This stage consumes candidate pseudo-views, creates repair masks, runs image-space
repair, and filters outputs with conservative acceptance gates.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _pil_image_module():
    try:
        from PIL import Image
    except Exception as exc:
        raise RuntimeError("Pillow is required for post-stage4 view-repair image IO") from exc
    return Image


def _load_rgb_alpha(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    Image = _pil_image_module()
    img = Image.open(path)
    alpha = None
    if "A" in img.getbands():
        alpha = np.asarray(img.getchannel("A"), dtype=np.uint8)
    rgb = np.asarray(img.convert("RGB"), dtype=np.uint8)
    return rgb, alpha


def _save_rgb(path: Path, rgb: np.ndarray) -> None:
    Image = _pil_image_module()
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb.astype(np.uint8), mode="RGB").save(path)


def _load_mask_bool(path: Path) -> np.ndarray:
    Image = _pil_image_module()
    mask = Image.open(path).convert("L")
    return np.asarray(mask, dtype=np.uint8) >= 128


def _compose_repair_with_mask(
    *,
    input_path: Path,
    fixed_path: Path,
    mask_path: Path,
    output_path: Path,
) -> tuple[bool, str]:
    try:
        original_rgb, _ = _load_rgb_alpha(input_path)
        fixed_rgb, _ = _load_rgb_alpha(fixed_path)
        mask = _load_mask_bool(mask_path)
    except Exception:
        return False, "mask_compose_io_error"

    if original_rgb.shape != fixed_rgb.shape:
        return False, "mask_compose_shape_mismatch"
    if mask.shape != original_rgb.shape[:2]:
        return False, "mask_compose_mask_shape_mismatch"

    composed = original_rgb.copy()
    if np.any(mask):
        composed[mask] = fixed_rgb[mask]
    _save_rgb(output_path, composed)
    return True, "mask_applied"


def _gray(rgb: np.ndarray) -> np.ndarray:
    rgb_f = rgb.astype(np.float32)
    return 0.299 * rgb_f[..., 0] + 0.587 * rgb_f[..., 1] + 0.114 * rgb_f[..., 2]


def _laplacian_variance(gray: np.ndarray) -> float:
    if gray.shape[0] < 3 or gray.shape[1] < 3:
        return 0.0
    center = gray[1:-1, 1:-1]
    up = gray[:-2, 1:-1]
    down = gray[2:, 1:-1]
    left = gray[1:-1, :-2]
    right = gray[1:-1, 2:]
    lap = (4.0 * center) - up - down - left - right
    return float(np.var(lap)) if lap.size else 0.0


def _hole_ratio(rgb: np.ndarray, alpha: np.ndarray | None = None) -> float:
    gray = _gray(rgb)
    dark = gray <= 18.0
    contrast = np.max(rgb, axis=2).astype(np.int16) - np.min(rgb, axis=2).astype(np.int16)
    hole = np.logical_and(dark, contrast <= 8)
    if alpha is not None:
        hole = np.logical_or(hole, alpha <= 8)
    return float(hole.sum()) / float(max(1, hole.size))


def build_repair_mask(
    rgb: np.ndarray,
    *,
    alpha: np.ndarray | None = None,
    depth: np.ndarray | None = None,
    alpha_threshold: int = 8,
    dark_threshold: int = 18,
) -> np.ndarray:
    """Build repair mask from low-alpha, invalid-depth, and dark-flat pixels."""
    gray = _gray(rgb)
    mask = gray <= float(dark_threshold)
    contrast = np.max(rgb, axis=2).astype(np.int16) - np.min(rgb, axis=2).astype(np.int16)
    mask = np.logical_and(mask, contrast <= 8)

    if alpha is not None:
        mask = np.logical_or(mask, alpha <= int(alpha_threshold))
    if depth is not None:
        depth_invalid = ~np.isfinite(depth) | (depth <= 0.0)
        mask = np.logical_or(mask, depth_invalid)
    return mask


def compute_photometric_drift_outside_mask(before_rgb: np.ndarray, after_rgb: np.ndarray, mask: np.ndarray) -> float:
    """Mean normalized pixel drift on observed (non-masked) regions."""
    if before_rgb.shape != after_rgb.shape:
        return 1.0
    observed = ~mask
    if not np.any(observed):
        return 1.0
    diff = np.abs(before_rgb.astype(np.float32) - after_rgb.astype(np.float32)) / 255.0
    per_pixel = np.mean(diff, axis=2)
    return float(np.mean(per_pixel[observed]))


def apply_acceptance_gate(
    rows: Sequence[Mapping[str, Any]],
    *,
    max_reprojection_error_px: float = 2.5,
    max_photometric_drift: float = 0.08,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split repaired views into accepted/rejected by hard quality thresholds."""
    accepted: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []

    for row in rows:
        reproj = float(row.get("cross_view_reprojection_error_px", 0.0))
        drift = float(row.get("photometric_drift_outside_mask", 1.0))
        existing_reasons = row.get("gate_reasons")
        reasons: List[str] = []
        if isinstance(existing_reasons, list):
            reasons.extend(str(item) for item in existing_reasons if str(item).strip())
        if reproj > float(max_reprojection_error_px):
            if "reprojection_error" not in reasons:
                reasons.append("reprojection_error")
        if drift > float(max_photometric_drift):
            if "outside_mask_drift" not in reasons:
                reasons.append("outside_mask_drift")
        if str(row.get("backend_mode", "")).strip().lower() == "passthrough":
            if "backend_passthrough" not in reasons:
                reasons.append("backend_passthrough")

        out = dict(row)
        out["gate_reasons"] = reasons
        out["accepted"] = len(reasons) == 0
        if out["accepted"]:
            accepted.append(out)
        else:
            rejected.append(out)
    return accepted, rejected


def _load_candidate_views(path: Path) -> List[Dict[str, Any]]:
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


def _load_virtual_render_mapping(path: Path | None) -> Dict[str, Dict[str, Any]]:
    rows: Dict[str, Dict[str, Any]] = {}
    if path is None or not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        candidate_id = str(payload.get("candidate_id") or "").strip()
        if candidate_id:
            rows[candidate_id] = payload
    return rows


def _run_template_command(template: str, *, input_path: Path, mask_path: Path, output_path: Path) -> bool:
    cmd_str = template.format(input=str(input_path), mask=str(mask_path), output=str(output_path))
    proc = subprocess.run(cmd_str, shell=True, text=True, capture_output=True)
    return proc.returncode == 0 and output_path.is_file() and output_path.stat().st_size > 0


def _run_fixer_native(
    *,
    input_path: Path,
    mask_path: Path,
    output_path: Path,
    fixer_dir: str = "",
    fixer_weights_dir: str = "",
    fixer_python: str = "",
    fixer_timestep: int = 250,
    fixer_resolution: int = 1024,
) -> tuple[bool, str]:
    """Run NVIDIA Fixer natively via inference_pretrained_model.py."""
    fixer_root = Path(fixer_dir or os.getenv("FIXER_DIR", "/opt/Fixer"))
    weights_root = Path(fixer_weights_dir or os.getenv("FIXER_WEIGHTS_DIR", "/opt/fixer_weights"))
    python_bin = fixer_python or os.getenv("FIXER_PYTHON", "python3")
    inference_script = fixer_root / "src" / "inference_pretrained_model.py"
    pretrained_path = weights_root / "pretrained" / "pretrained_fixer.pkl"

    if not inference_script.is_file() or not pretrained_path.is_file():
        return False, "fixer_missing"

    # Fixer expects a directory of images as input; create a temp single-image dir.
    tmp_in = output_path.parent / ".fixer_tmp_in"
    tmp_out = output_path.parent / ".fixer_tmp_out"
    tmp_in.mkdir(parents=True, exist_ok=True)
    tmp_out.mkdir(parents=True, exist_ok=True)
    try:
        # Clear previous temp files.
        for f in tmp_in.iterdir():
            f.unlink(missing_ok=True)
        for f in tmp_out.iterdir():
            f.unlink(missing_ok=True)

        shutil.copy2(input_path, tmp_in / input_path.name)

        cmd = [
            python_bin,
            str(inference_script),
            "--model", str(pretrained_path),
            "--input", str(tmp_in),
            "--output", str(tmp_out),
            "--timestep", str(fixer_timestep),
            "--resolution", str(fixer_resolution),
        ]
        proc = subprocess.run(cmd, text=True, capture_output=True, check=False,
                              cwd=str(fixer_root / "src"))
        if proc.returncode != 0:
            return False, f"fixer_error_rc{proc.returncode}"

        # Find output image and blend only masked regions into the source frame.
        fixed_candidates = list(tmp_out.iterdir())
        if not fixed_candidates:
            return False, "fixer_no_output"
        fixed_path = next((p for p in fixed_candidates if p.name == input_path.name), fixed_candidates[0])
        composed_ok, composed_mode = _compose_repair_with_mask(
            input_path=input_path,
            fixed_path=fixed_path,
            mask_path=mask_path,
            output_path=output_path,
        )
        if not composed_ok:
            return False, composed_mode
        return True, "fixer_native"
    finally:
        shutil.rmtree(tmp_in, ignore_errors=True)
        shutil.rmtree(tmp_out, ignore_errors=True)


def _run_backend(
    *,
    backend: str,
    input_path: Path,
    mask_path: Path,
    output_path: Path,
) -> tuple[bool, str]:
    # First try env-var template command (user override).
    if backend == "fixer":
        template = os.getenv("POST_STAGE4_FIXER_IMAGE_COMMAND", "").strip()
    elif backend == "gsfix3d":
        template = os.getenv("POST_STAGE4_GSFIX3D_IMAGE_COMMAND", "").strip()
    else:
        template = ""

    if template:
        ok = _run_template_command(template, input_path=input_path, mask_path=mask_path, output_path=output_path)
        if ok:
            return True, "command"

    # Try native Fixer runtime when backend is "fixer".
    if backend == "fixer":
        ok, mode = _run_fixer_native(
            input_path=input_path,
            mask_path=mask_path,
            output_path=output_path,
        )
        if ok:
            return True, mode

    # Fallback: pass-through copy to keep pipeline deterministic when model runtime is unavailable.
    shutil.copy2(input_path, output_path)
    return True, "passthrough"


def repair_candidate_views(
    *,
    renders_dir: Path,
    candidate_views_path: Path,
    output_dir: Path,
    model_mode: str,
    max_reprojection_error_px: float,
    max_photometric_drift: float,
    virtual_render_mapping_path: Path | None = None,
) -> Dict[str, Any]:
    candidates = _load_candidate_views(candidate_views_path)
    virtual_render_mapping = _load_virtual_render_mapping(virtual_render_mapping_path)

    masks_dir = output_dir / "gap_mask_preview"
    masks_dir.mkdir(parents=True, exist_ok=True)
    repaired_dir = output_dir / "post_stage4_repaired_views"
    repaired_dir.mkdir(parents=True, exist_ok=True)

    use_gsfix3d = model_mode.strip().lower() == "fixer+gsfix3d"
    rows: List[Dict[str, Any]] = []

    for idx, cand in enumerate(candidates):
        is_virtual = bool(cand.get("is_virtual"))
        candidate_id = str(cand.get("id") or f"candidate_{idx:04d}")
        mapping_row = virtual_render_mapping.get(candidate_id, {})
        render_image = str(cand.get("render_image") or "").strip()
        source_image = str(cand.get("source_image") or "").strip()
        if render_image:
            render_path = Path(render_image)
        elif is_virtual:
            mapped_render = str(mapping_row.get("render_image") or "").strip()
            if mapped_render:
                render_path = Path(mapped_render)
            else:
                rows.append(
                    {
                        "id": candidate_id,
                        "candidate_id": candidate_id,
                        "is_virtual": True,
                        "source_image": source_image,
                        "render_image": "",
                        "repaired_image": "",
                        "mask_image": "",
                        "backend": "fixer",
                        "backend_mode": "unresolved",
                        "cross_view_reprojection_error_px": float(cand.get("cross_view_reprojection_error_px", 0.0)),
                        "photometric_drift_outside_mask": 1.0,
                        "gate_reasons": ["virtual_render_mapping_missing"],
                        "qvec": cand.get("qvec", mapping_row.get("qvec")),
                        "tvec": cand.get("tvec", mapping_row.get("tvec")),
                        "camera_id": cand.get("camera_id", mapping_row.get("camera_id")),
                        "predicted_hole_ratio": cand.get("predicted_hole_ratio", mapping_row.get("predicted_hole_ratio")),
                    }
                )
                continue
        elif source_image:
            render_path = renders_dir / source_image
        else:
            continue
        if not render_path.is_file():
            if is_virtual:
                rows.append(
                    {
                        "id": candidate_id,
                        "candidate_id": candidate_id,
                        "is_virtual": True,
                        "source_image": source_image,
                        "render_image": str(render_path),
                        "repaired_image": "",
                        "mask_image": "",
                        "backend": "fixer",
                        "backend_mode": "unresolved",
                        "cross_view_reprojection_error_px": float(cand.get("cross_view_reprojection_error_px", 0.0)),
                        "photometric_drift_outside_mask": 1.0,
                        "gate_reasons": ["virtual_render_missing"],
                        "qvec": cand.get("qvec", mapping_row.get("qvec")),
                        "tvec": cand.get("tvec", mapping_row.get("tvec")),
                        "camera_id": cand.get("camera_id", mapping_row.get("camera_id")),
                        "predicted_hole_ratio": cand.get("predicted_hole_ratio", mapping_row.get("predicted_hole_ratio")),
                    }
                )
            continue

        view_id = candidate_id
        original_rgb, original_alpha = _load_rgb_alpha(render_path)
        mask = build_repair_mask(original_rgb, alpha=original_alpha)

        mask_path = masks_dir / f"{view_id}_mask.png"
        Image = _pil_image_module()
        Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(mask_path)

        repaired_path = repaired_dir / render_path.name
        backend_used = "fixer"
        backend_mode = ""
        ok, backend_mode = _run_backend(
            backend="fixer",
            input_path=render_path,
            mask_path=mask_path,
            output_path=repaired_path,
        )
        if not ok:
            continue

        repaired_rgb, _ = _load_rgb_alpha(repaired_path)
        drift = compute_photometric_drift_outside_mask(original_rgb, repaired_rgb, mask)
        reproj = float(cand.get("cross_view_reprojection_error_px", 0.0))

        row: Dict[str, Any] = {
            "id": view_id,
            "candidate_id": candidate_id,
            "is_virtual": is_virtual,
            "source_image": source_image or render_path.name,
            "render_image": str(render_path),
            "repaired_image": str(repaired_path),
            "mask_image": str(mask_path),
            "backend": backend_used,
            "backend_mode": backend_mode,
            "mask_ratio": float(mask.sum()) / float(max(1, mask.size)),
            "cross_view_reprojection_error_px": reproj,
            "photometric_drift_outside_mask": float(drift),
            "pre_hole_ratio": _hole_ratio(original_rgb, original_alpha),
            "post_hole_ratio": _hole_ratio(repaired_rgb, None),
            "pre_sharpness": _laplacian_variance(_gray(original_rgb)),
            "post_sharpness": _laplacian_variance(_gray(repaired_rgb)),
            "qvec": cand.get("qvec", mapping_row.get("qvec")),
            "tvec": cand.get("tvec", mapping_row.get("tvec")),
            "camera_id": cand.get("camera_id", mapping_row.get("camera_id")),
            "predicted_hole_ratio": cand.get("predicted_hole_ratio", mapping_row.get("predicted_hole_ratio")),
        }

        accepted, rejected = apply_acceptance_gate(
            [row],
            max_reprojection_error_px=max_reprojection_error_px,
            max_photometric_drift=max_photometric_drift,
        )
        if rejected and use_gsfix3d:
            backend_used = "gsfix3d"
            ok2, backend_mode2 = _run_backend(
                backend="gsfix3d",
                input_path=render_path,
                mask_path=mask_path,
                output_path=repaired_path,
            )
            if ok2:
                repaired_rgb, _ = _load_rgb_alpha(repaired_path)
                row["backend"] = backend_used
                row["backend_mode"] = backend_mode2
                row["photometric_drift_outside_mask"] = compute_photometric_drift_outside_mask(
                    original_rgb,
                    repaired_rgb,
                    mask,
                )
                row["post_hole_ratio"] = _hole_ratio(repaired_rgb, None)
                row["post_sharpness"] = _laplacian_variance(_gray(repaired_rgb))
                accepted, rejected = apply_acceptance_gate(
                    [row],
                    max_reprojection_error_px=max_reprojection_error_px,
                    max_photometric_drift=max_photometric_drift,
                )

        if accepted:
            rows.append(accepted[0])
        else:
            try:
                repaired_path.unlink()
            except Exception:
                pass
            rows.append(rejected[0])

    accepted_rows, rejected_rows = apply_acceptance_gate(
        rows,
        max_reprojection_error_px=max_reprojection_error_px,
        max_photometric_drift=max_photometric_drift,
    )

    accepted_jsonl = output_dir / "accepted_repaired_views.jsonl"
    with accepted_jsonl.open("w", encoding="utf-8") as f:
        for row in accepted_rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")

    pre_holes = [float(r.get("pre_hole_ratio", 0.0)) for r in rows]
    post_holes = [float(r.get("post_hole_ratio", 0.0)) for r in accepted_rows]
    pre_sharp = [float(r.get("pre_sharpness", 0.0)) for r in rows]
    post_sharp = [float(r.get("post_sharpness", 0.0)) for r in accepted_rows]
    pre_holes_mean = float(np.mean(pre_holes)) if pre_holes else 0.0
    pre_sharp_mean = float(np.mean(pre_sharp)) if pre_sharp else 0.0

    report = {
        "schema_version": "v1",
        "generated_at": _utc_now_iso(),
        "candidate_views_path": str(candidate_views_path),
        "model_mode": model_mode,
        "accepted_count": int(len(accepted_rows)),
        "rejected_count": int(len(rejected_rows)),
        "accepted_views_path": str(accepted_jsonl),
        "repaired_views_dir": str(repaired_dir),
        "max_reprojection_error_px": float(max_reprojection_error_px),
        "max_photometric_drift": float(max_photometric_drift),
        "pre_repair_hole_ratio_mean": pre_holes_mean,
        "post_repair_hole_ratio_mean": float(np.mean(post_holes)) if post_holes else pre_holes_mean,
        "pre_sharpness_mean": pre_sharp_mean,
        "post_sharpness_mean": float(np.mean(post_sharp)) if post_sharp else pre_sharp_mean,
        "backend_passthrough_rejected_count": int(
            sum(
                1
                for row in rejected_rows
                if "backend_passthrough" in list(row.get("gate_reasons", []))
            )
        ),
        "rows": rows,
    }
    report_path = output_dir / "view_repair_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Repair pseudo-views with Fixer/GSFix3D and quality gating")
    parser.add_argument("--renders-dir", required=True)
    parser.add_argument("--candidate-views", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--virtual-render-mapping", default="", help="Optional JSONL mapping from virtual candidate_id -> rendered image metadata")
    parser.add_argument(
        "--model",
        default=os.getenv("POST_STAGE4_REFINE_MODEL", "fixer+gsfix3d"),
        choices=["fixer", "fixer+gsfix3d"],
    )
    parser.add_argument(
        "--max-reprojection-error-px",
        type=float,
        default=float(os.getenv("POST_STAGE4_MAX_REPROJ_ERROR_PX", "2.5")),
    )
    parser.add_argument(
        "--max-photometric-drift",
        type=float,
        default=float(os.getenv("POST_STAGE4_MAX_PHOTOMETRIC_DRIFT", "0.08")),
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    repair_candidate_views(
        renders_dir=Path(args.renders_dir),
        candidate_views_path=Path(args.candidate_views),
        output_dir=output_dir,
        model_mode=str(args.model),
        max_reprojection_error_px=float(args.max_reprojection_error_px),
        max_photometric_drift=float(args.max_photometric_drift),
        virtual_render_mapping_path=Path(args.virtual_render_mapping) if str(args.virtual_render_mapping).strip() else None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
