"""Digest-bound outside-mask locality measurements for public-scene edits."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image, ImageFilter

from .decision_evidence_contracts import canonical_digest, canonical_json
from .heldout_appearance_evaluation_v2 import _LpipsRuntime, _global_ssim, windowed_ssim


SCHEMA_VERSION = "public_scene_inpainting_locality_measurement.v1"


class PublicSceneInpaintingLocalityError(ValueError):
    def __init__(self, codes: Sequence[str]) -> None:
        self.codes = tuple(sorted(set(str(code) for code in codes if str(code))))
        super().__init__("; ".join(self.codes))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _approved(path: str | Path, roots: Sequence[str | Path]) -> Path:
    resolved = Path(path).expanduser().resolve()
    approved_roots = [Path(root).expanduser().resolve() for root in roots]
    if not approved_roots or not any(resolved.is_relative_to(root) for root in approved_roots):
        raise PublicSceneInpaintingLocalityError(["locality_path_outside_approved_roots"])
    if resolved.is_symlink():
        raise PublicSceneInpaintingLocalityError(["locality_symlink_path_unsupported"])
    return resolved


def _rgb(path: Path) -> np.ndarray:
    if not path.is_file() or path.is_symlink():
        raise PublicSceneInpaintingLocalityError(["locality_rgb_missing_or_symlink"])
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def _mask(path: Path, *, dilation_pixels: int) -> np.ndarray:
    if not path.is_file() or path.is_symlink():
        raise PublicSceneInpaintingLocalityError(["locality_mask_missing_or_symlink"])
    with Image.open(path) as image:
        mask = image.convert("L")
        if dilation_pixels:
            mask = mask.filter(ImageFilter.MaxFilter(2 * dilation_pixels + 1))
        return np.asarray(mask) > 0


def measure_inpainting_locality(
    *,
    before_dir: str | Path,
    mask_dir: str | Path,
    after_render_manifest: str | Path,
    output_path: str | Path,
    approved_roots: Sequence[str | Path],
    dilation_pixels: int = 16,
    lpips_model: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not 0 <= dilation_pixels <= 128:
        raise PublicSceneInpaintingLocalityError(["locality_dilation_pixels_invalid"])
    before = _approved(before_dir, approved_roots)
    masks = _approved(mask_dir, approved_roots)
    manifest_path = _approved(after_render_manifest, approved_roots)
    output = _approved(output_path, approved_roots)
    if not before.is_dir() or not masks.is_dir() or not manifest_path.is_file():
        raise PublicSceneInpaintingLocalityError(["locality_input_path_missing"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != "sealed_camera_render_manifest.v1"
        or manifest.get("status") != "rendered_exact_cameras"
    ):
        raise PublicSceneInpaintingLocalityError(["locality_after_manifest_invalid"])
    render_rows = manifest.get("renders")
    if not isinstance(render_rows, list) or not render_rows:
        raise PublicSceneInpaintingLocalityError(["locality_after_renders_missing"])
    lpips_runtime = None
    if lpips_model is not None:
        lpips_runtime = _LpipsRuntime(
            str(lpips_model.get("model_id") or ""),
            str(lpips_model.get("checkpoint_digest") or ""),
            str(lpips_model.get("backbone_digest") or "") or None,
        )
    rows: list[dict[str, Any]] = []
    after_root = manifest_path.parent
    for render in render_rows:
        camera_id = str(render.get("camera_id") or "")
        if not camera_id or "/" in camera_id or ".." in camera_id:
            raise PublicSceneInpaintingLocalityError(["locality_camera_id_invalid"])
        before_path = before / f"{camera_id}.png"
        mask_path = masks / f"{camera_id}.png"
        after_path = after_root / str(render.get("relative_path") or "")
        if not after_path.resolve().is_relative_to(after_root.resolve()):
            raise PublicSceneInpaintingLocalityError(["locality_after_path_escapes_manifest"])
        if _sha256(after_path) != render.get("digest"):
            raise PublicSceneInpaintingLocalityError(
                [f"locality_after_digest_mismatch:{camera_id}"]
            )
        left = _rgb(before_path)
        right = _rgb(after_path)
        target = _mask(mask_path, dilation_pixels=dilation_pixels)
        if left.shape != right.shape or target.shape != left.shape[:2]:
            raise PublicSceneInpaintingLocalityError(
                [f"locality_shape_mismatch:{camera_id}"]
            )
        outside = ~target
        if not bool(np.any(target)) or not bool(np.any(outside)):
            raise PublicSceneInpaintingLocalityError(
                [f"locality_mask_empty_or_full:{camera_id}"]
            )
        difference = left - right
        outside_difference = difference[outside]
        mse = float(np.mean(np.square(outside_difference)))
        psnr = float("inf") if mse == 0.0 else 10.0 * math.log10(1.0 / mse)
        locality_only = right.copy()
        locality_only[target] = left[target]
        row = {
            "camera_id": camera_id,
            "before_sha256": _sha256(before_path),
            "mask_sha256": _sha256(mask_path),
            "after_sha256": _sha256(after_path),
            "width": int(left.shape[1]),
            "height": int(left.shape[0]),
            "dilated_mask_pixel_count": int(np.count_nonzero(target)),
            "outside_mask_pixel_count": int(np.count_nonzero(outside)),
            "outside_mask_psnr_db": "infinity" if math.isinf(psnr) else round(psnr, 6),
            "outside_mask_mean_absolute_error": round(
                float(np.mean(np.abs(outside_difference))), 8
            ),
            "outside_mask_fraction_max_channel_delta_gt_20_255": round(
                float(np.mean(np.max(np.abs(outside_difference), axis=-1) > (20.0 / 255.0))),
                8,
            ),
            "outside_mask_global_ssim": round(_global_ssim(left, locality_only), 8),
            "outside_mask_windowed_ssim": round(windowed_ssim(left, locality_only), 8),
        }
        if lpips_runtime is not None:
            row["outside_mask_lpips"] = round(
                lpips_runtime.distance(left, locality_only), 8
            )
        rows.append(row)
    finite_psnr = [
        float(row["outside_mask_psnr_db"])
        for row in rows
        if row["outside_mask_psnr_db"] != "infinity"
    ]
    aggregate = {
        "view_count": len(rows),
        "mean_outside_mask_psnr_db": (
            "infinity"
            if len(finite_psnr) != len(rows)
            else round(float(np.mean(finite_psnr)), 6)
        ),
        "mean_outside_mask_mean_absolute_error": round(
            float(np.mean([row["outside_mask_mean_absolute_error"] for row in rows])), 8
        ),
        "mean_outside_mask_fraction_max_channel_delta_gt_20_255": round(
            float(
                np.mean(
                    [
                        row["outside_mask_fraction_max_channel_delta_gt_20_255"]
                        for row in rows
                    ]
                )
            ),
            8,
        ),
        "mean_outside_mask_global_ssim": round(
            float(np.mean([row["outside_mask_global_ssim"] for row in rows])), 8
        ),
        "mean_outside_mask_windowed_ssim": round(
            float(np.mean([row["outside_mask_windowed_ssim"] for row in rows])), 8
        ),
        "mean_outside_mask_lpips": (
            round(float(np.mean([row["outside_mask_lpips"] for row in rows])), 8)
            if lpips_runtime is not None
            else None
        ),
    }
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "status": "measured_no_admission_effect",
        "after_render_manifest_sha256": _sha256(manifest_path),
        "after_render_manifest_digest": manifest.get(
            "sealed_camera_render_manifest_digest"
        ),
        "dilation_pixels": dilation_pixels,
        "metric_contract": (
            "compare_before_to_after_only_outside_dilated_target_mask; "
            "replace_target_pixels_with_before_pixels for SSIM and LPIPS"
        ),
        "lpips_runtime": (
            {
                "model_id": "lpips_alex_v0.1",
                "checkpoint_digest": lpips_runtime.checkpoint_digest,
                "backbone_digest": lpips_runtime.backbone_digest,
                "torch_version": lpips_runtime.torch_version,
            }
            if lpips_runtime is not None
            else None
        ),
        "rows": rows,
        "aggregate": aggregate,
        "thresholds_frozen_before_evaluation": False,
        "quality_pass_claimed": False,
        "admission_effect": "none",
        "claim_ceiling": "outside_mask_edit_locality_measurement_only",
        "raw_secret_values_recorded": False,
    }
    receipt["locality_measurement_digest"] = canonical_digest(
        receipt, digest_field="locality_measurement_digest"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(canonical_json(receipt) + "\n", encoding="utf-8")
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--before-dir", required=True)
    parser.add_argument("--mask-dir", required=True)
    parser.add_argument("--after-render-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--approved-root", action="append", required=True)
    parser.add_argument("--dilation-pixels", type=int, default=16)
    parser.add_argument("--lpips-checkpoint-digest")
    parser.add_argument("--lpips-backbone-digest")
    args = parser.parse_args(argv)
    lpips_model = None
    if args.lpips_checkpoint_digest:
        lpips_model = {
            "model_id": "lpips_alex_v0.1",
            "checkpoint_digest": args.lpips_checkpoint_digest,
            "backbone_digest": args.lpips_backbone_digest,
        }
    measure_inpainting_locality(
        before_dir=args.before_dir,
        mask_dir=args.mask_dir,
        after_render_manifest=args.after_render_manifest,
        output_path=args.output,
        approved_roots=args.approved_root,
        dilation_pixels=args.dilation_pixels,
        lpips_model=lpips_model,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
