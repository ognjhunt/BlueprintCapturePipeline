"""Evaluate a frozen Gaussian-excision split without outcome leakage.

The renderer does not expose a trustworthy alpha attachment.  Two otherwise
identical exact-camera renders over black and white backgrounds recover the
composited alpha independently of splat colour.  This module binds those
renders, evaluates the frozen held-out cameras, and creates review contact
sheets without changing ownership labels.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .decision_evidence_contracts import canonical_digest, canonical_json


HELDOUT_AUDIT_SCHEMA = "adp009b_gaussian_excision_heldout_audit.v1"
OWNERSHIP_REPLAY_SCHEMA = "adp009b_gaussian_excision_ownership_replay.v1"


class GaussianExcisionHeldoutError(ValueError):
    """Stable fail-closed held-out audit errors."""

    def __init__(self, codes: Sequence[str]) -> None:
        self.codes = tuple(sorted(set(str(code) for code in codes if str(code))))
        super().__init__(";".join(self.codes))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _record(path: Path, root: Path) -> dict[str, Any]:
    return {
        "relative_path": path.relative_to(root).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _read_json(path: Path, code: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GaussianExcisionHeldoutError([code]) from exc
    if not isinstance(value, dict):
        raise GaussianExcisionHeldoutError([code])
    return value


def derive_alpha_from_background_pair(black_rgb: np.ndarray, white_rgb: np.ndarray) -> np.ndarray:
    """Recover alpha from identical RGB-over-black and RGB-over-white renders."""

    black = np.asarray(black_rgb)
    white = np.asarray(white_rgb)
    if (
        black.shape != white.shape
        or black.ndim != 3
        or black.shape[2] != 3
        or black.dtype != np.uint8
        or white.dtype != np.uint8
    ):
        raise GaussianExcisionHeldoutError(["heldout_background_pair_shape_invalid"])
    transmission_channels = (white.astype(np.float32) - black.astype(np.float32)) / 255.0
    # Quantization and renderer rounding can move a channel by one code value.
    transmission = np.median(np.clip(transmission_channels, 0.0, 1.0), axis=2)
    return np.clip(1.0 - transmission, 0.0, 1.0)


def _largest_component(mask: np.ndarray) -> int:
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
        np.asarray(mask, dtype=np.uint8), 8
    )
    return int(stats[1:, cv2.CC_STAT_AREA].max()) if count > 1 else 0


def evaluate_alpha_layer(
    alpha: np.ndarray,
    exact_mask: np.ndarray,
    *,
    significant_alpha_threshold: float,
    rasterization_band_pixels: int,
) -> dict[str, Any]:
    """Measure silhouette coverage and protected contribution for one layer."""

    alpha = np.asarray(alpha, dtype=np.float32)
    mask = np.asarray(exact_mask, dtype=bool)
    if alpha.shape != mask.shape or alpha.ndim != 2 or not np.isfinite(alpha).all():
        raise GaussianExcisionHeldoutError(["heldout_alpha_or_mask_invalid"])
    if not 0.0 < significant_alpha_threshold <= 1.0 or rasterization_band_pixels < 0:
        raise GaussianExcisionHeldoutError(["heldout_alpha_policy_invalid"])
    significant = alpha > float(significant_alpha_threshold)
    kernel_size = 2 * int(rasterization_band_pixels) + 1
    envelope = cv2.dilate(mask.astype(np.uint8), np.ones((kernel_size, kernel_size), np.uint8)) > 0
    missed = mask & ~significant
    protected = significant & ~envelope
    intersection = int((significant & mask).sum())
    union = int((significant | mask).sum())
    return {
        "mask_pixel_count": int(mask.sum()),
        "significant_pixel_count": int(significant.sum()),
        "silhouette_intersection_over_union": intersection / union if union else 1.0,
        "silhouette_recall": intersection / int(mask.sum()) if mask.any() else 1.0,
        "silhouette_missing_pixel_count": int(missed.sum()),
        "silhouette_largest_missing_component_pixels": _largest_component(missed),
        "protected_significant_pixel_count": int(protected.sum()),
        "protected_alpha_sum": float(alpha[~envelope].sum()),
        "protected_maximum_alpha": float(alpha[~envelope].max()) if np.any(~envelope) else 0.0,
        "inside_mask_alpha_sum": float(alpha[mask].sum()),
        "inside_mask_maximum_alpha": float(alpha[mask].max()) if mask.any() else 0.0,
        "inside_mask_significant_pixel_count": intersection,
        "inside_mask_largest_significant_component_pixels": _largest_component(significant & mask),
    }


def _render_frames(manifest_path: Path, expected_background: str) -> dict[str, Path]:
    manifest = _read_json(manifest_path, "heldout_render_manifest_unreadable")
    expected_digest = canonical_digest(
        manifest, digest_field="sealed_camera_render_manifest_digest"
    )
    if (
        manifest.get("schema_version") != "sealed_camera_render_manifest.v1"
        or manifest.get("status") != "rendered_exact_cameras"
        or manifest.get("sealed_camera_render_manifest_digest") != expected_digest
        or (manifest.get("renderer_identity") or {}).get("background_rgb") != expected_background
    ):
        raise GaussianExcisionHeldoutError(["heldout_render_manifest_invalid"])
    frames: dict[str, Path] = {}
    for row in manifest.get("renders") or []:
        camera_id = str(row.get("camera_id") or "")
        frame = manifest_path.parent / str(row.get("relative_path") or "")
        if (
            not camera_id
            or camera_id in frames
            or not frame.is_file()
            or _sha256(frame) != row.get("digest")
        ):
            raise GaussianExcisionHeldoutError(["heldout_render_frame_invalid"])
        frames[camera_id] = frame
    return frames


def _load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _heatmap(alpha: np.ndarray) -> Image.Image:
    values = np.clip(np.rint(alpha * 255.0), 0, 255).astype(np.uint8)
    heat = cv2.applyColorMap(values, cv2.COLORMAP_INFERNO)
    return Image.fromarray(cv2.cvtColor(heat, cv2.COLOR_BGR2RGB))


def _contact_sheet(
    columns: Sequence[tuple[str, Image.Image]], output: Path, *, width: int = 384
) -> None:
    header = 34
    resized: list[tuple[str, Image.Image]] = []
    for label, image in columns:
        height = max(1, round(image.height * width / image.width))
        resized.append((label, image.resize((width, height), Image.Resampling.LANCZOS)))
    canvas = Image.new(
        "RGB", (width * len(resized), header + max(image.height for _, image in resized)), "white"
    )
    draw = ImageDraw.Draw(canvas)
    for column, (label, image) in enumerate(resized):
        x = column * width
        draw.text((x + 8, 10), label, fill="black")
        canvas.paste(image, (x, header))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, format="PNG", optimize=False)


def materialize_gaussian_excision_heldout_audit(
    *,
    freeze_path: str | Path,
    ownership_receipt_path: str | Path,
    ownership_replay_receipt_path: str | Path,
    obb_black_manifest_path: str | Path,
    obb_white_manifest_path: str | Path,
    owned_black_manifest_path: str | Path,
    owned_white_manifest_path: str | Path,
    ambiguous_black_manifest_path: str | Path,
    ambiguous_white_manifest_path: str | Path,
    retained_scene_manifest_path: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    """Evaluate all frozen cameras and admit or abstain on the held-out pair."""

    freeze_file = Path(freeze_path).expanduser().resolve()
    ownership_file = Path(ownership_receipt_path).expanduser().resolve()
    replay_file = Path(ownership_replay_receipt_path).expanduser().resolve()
    output = Path(output_root).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    freeze = _read_json(freeze_file, "heldout_freeze_unreadable")
    ownership = _read_json(ownership_file, "heldout_ownership_receipt_unreadable")
    replay = _read_json(replay_file, "heldout_ownership_replay_unreadable")
    if (
        freeze.get("schema_version") != "adp009b_gaussian_excision_audit_freeze.v1"
        or freeze.get("freeze_digest") != canonical_digest(freeze, digest_field="freeze_digest")
        or ownership.get("schema_version") != "adp009b_gaussian_excision_ownership_receipt.v1"
        or ownership.get("receipt_digest")
        != canonical_digest(ownership, digest_field="receipt_digest")
        or ownership.get("freeze_digest") != freeze.get("freeze_digest")
        or ownership.get("heldout_cameras_accessed_for_classification") is not False
    ):
        raise GaussianExcisionHeldoutError(["heldout_freeze_or_ownership_join_invalid"])
    ownership_counts = ownership.get("ownership") or {}
    baseline_count = (freeze.get("historical_baseline") or {}).get("selected_gaussian_count")
    if (
        ownership_counts.get("exhaustive") is not True
        or ownership_counts.get("pairwise_disjoint") is not True
        or ownership_counts.get("historical_obb_count") != baseline_count
    ):
        raise GaussianExcisionHeldoutError(["heldout_ownership_partition_invalid"])
    if (
        replay.get("schema_version") != OWNERSHIP_REPLAY_SCHEMA
        or replay.get("replay_digest") != canonical_digest(replay, digest_field="replay_digest")
        or replay.get("execution_count") != 2
        or replay.get("freeze_digest") != freeze.get("freeze_digest")
        or replay.get("ownership_receipt_digest") != ownership.get("receipt_digest")
        or replay.get("canonical_manifests_identical") is not True
        or replay.get("receipt_files_byte_identical") is not True
        or replay.get("output_digests_identical") is not True
        or replay.get("index_sets_identical") is not True
        or replay.get("protected_source_records_byte_identical") is not True
        or replay.get("gate_passed") is not True
    ):
        raise GaussianExcisionHeldoutError(["heldout_ownership_replay_invalid"])

    manifests = {
        "obb_black": Path(obb_black_manifest_path).resolve(),
        "obb_white": Path(obb_white_manifest_path).resolve(),
        "owned_black": Path(owned_black_manifest_path).resolve(),
        "owned_white": Path(owned_white_manifest_path).resolve(),
        "ambiguous_black": Path(ambiguous_black_manifest_path).resolve(),
        "ambiguous_white": Path(ambiguous_white_manifest_path).resolve(),
        "retained_scene": Path(retained_scene_manifest_path).resolve(),
    }
    frames = {
        "obb_black": _render_frames(manifests["obb_black"], "#000000"),
        "obb_white": _render_frames(manifests["obb_white"], "#ffffff"),
        "owned_black": _render_frames(manifests["owned_black"], "#000000"),
        "owned_white": _render_frames(manifests["owned_white"], "#ffffff"),
        "ambiguous_black": _render_frames(manifests["ambiguous_black"], "#000000"),
        "ambiguous_white": _render_frames(manifests["ambiguous_white"], "#ffffff"),
        "retained_scene": _render_frames(manifests["retained_scene"], "#000000"),
    }
    camera_ids = [str(row["camera_id"]) for row in freeze.get("masks") or []]
    if any(set(layer) != set(camera_ids) for layer in frames.values()):
        raise GaussianExcisionHeldoutError(["heldout_render_camera_set_mismatch"])
    source_images = {
        str(row["camera_id"]): Path(row["path"]).resolve()
        for row in freeze.get("source_images") or []
    }
    masks = {
        str(row["camera_id"]): Path(row["historical_outer_mask"]["path"]).resolve()
        for row in freeze.get("masks") or []
    }
    policy = freeze.get("policy") or {}
    threshold = float(policy["heldout_significant_alpha_threshold"])
    band = int(policy["heldout_rasterization_band_pixels"])
    maximum_component = int(policy["heldout_maximum_residual_connected_component_pixels"])
    maximum_protected = int(policy["heldout_maximum_protected_significant_pixels"])
    heldout = set(freeze["camera_split"]["heldout_camera_ids"])
    rows: list[dict[str, Any]] = []
    sheets: list[Path] = []
    for camera_id in camera_ids:
        with Image.open(masks[camera_id]) as mask_image:
            mask = np.asarray(mask_image.convert("L"), dtype=np.uint8) >= 128
            mask_review = Image.fromarray((mask.astype(np.uint8) * 255), mode="L").convert("RGB")
        alphas = {}
        metrics = {}
        for layer in ("obb", "owned", "ambiguous"):
            alphas[layer] = derive_alpha_from_background_pair(
                _load_rgb(frames[f"{layer}_black"][camera_id]),
                _load_rgb(frames[f"{layer}_white"][camera_id]),
            )
            metrics[layer] = evaluate_alpha_layer(
                alphas[layer],
                mask,
                significant_alpha_threshold=threshold,
                rasterization_band_pixels=band,
            )
        gates = {
            "removed_only_reproduces_silhouette": metrics["owned"]["silhouette_missing_pixel_count"]
            == 0,
            "removed_only_protected_spill_within_limit": metrics["owned"][
                "protected_significant_pixel_count"
            ]
            <= maximum_protected,
            "residual_alpha_at_most_threshold": metrics["ambiguous"][
                "inside_mask_significant_pixel_count"
            ]
            == 0,
            "residual_component_within_limit": metrics["ambiguous"][
                "inside_mask_largest_significant_component_pixels"
            ]
            <= maximum_component,
            "residue_at_least_as_good_as_obb": (
                metrics["owned"]["silhouette_missing_pixel_count"]
                <= metrics["obb"]["silhouette_missing_pixel_count"]
                and metrics["owned"]["silhouette_largest_missing_component_pixels"]
                <= metrics["obb"]["silhouette_largest_missing_component_pixels"]
            ),
            "protected_contribution_strictly_less_than_obb": metrics["owned"]["protected_alpha_sum"]
            < metrics["obb"]["protected_alpha_sum"],
        }
        owned_pass = all(
            gates[name]
            for name in (
                "removed_only_reproduces_silhouette",
                "removed_only_protected_spill_within_limit",
                "residual_alpha_at_most_threshold",
                "residual_component_within_limit",
            )
        )
        baseline_comparison_pass = all(
            gates[name]
            for name in (
                "residue_at_least_as_good_as_obb",
                "protected_contribution_strictly_less_than_obb",
            )
        )
        sheet = output / "contact_sheets" / f"{camera_id}.png"
        _contact_sheet(
            [
                ("original", Image.open(source_images[camera_id]).convert("RGB")),
                ("exact mask", mask_review),
                ("OBB removed-only", Image.open(frames["obb_black"][camera_id]).convert("RGB")),
                (
                    "contribution removed-only",
                    Image.open(frames["owned_black"][camera_id]).convert("RGB"),
                ),
                ("retained scene", Image.open(frames["retained_scene"][camera_id]).convert("RGB")),
                ("ambiguity heatmap", _heatmap(alphas["ambiguous"])),
            ],
            sheet,
        )
        sheets.append(sheet)
        rows.append(
            {
                "camera_id": camera_id,
                "split": "heldout" if camera_id in heldout else "calibration_review_only",
                "layers": metrics,
                "gates": gates,
                "ownership_gate_passed": owned_pass,
                "at_least_as_good_as_obb_passed": baseline_comparison_pass,
                "residual_upper_bound_layer": "ambiguous",
                "protected_contribution_comparison": "strict_alpha_sum",
                "contact_sheet": _record(sheet, output),
            }
        )

    heldout_rows = [row for row in rows if row["split"] == "heldout"]
    deterministic = bool(replay.get("gate_passed"))
    partition_passed = bool(
        ownership_counts.get("exhaustive") and ownership_counts.get("pairwise_disjoint")
    )
    protected_records_passed = bool(replay.get("protected_source_records_byte_identical"))
    passed = bool(
        deterministic
        and partition_passed
        and protected_records_passed
        and heldout_rows
        and all(
            row["ownership_gate_passed"] and row["at_least_as_good_as_obb_passed"]
            for row in heldout_rows
        )
    )
    receipt: dict[str, Any] = {
        "schema_version": HELDOUT_AUDIT_SCHEMA,
        "program_id": "arm-decision-proof-v1",
        "adp_item": "ADP-009B",
        "status": (
            "heldout_gaussian_ownership_gate_passed"
            if passed
            else "abstained_calibrated_gaussian_ownership_separation_insufficient"
        ),
        "freeze_digest": freeze["freeze_digest"],
        "ownership_receipt_digest": ownership.get("receipt_digest"),
        "ownership_replay_digest": replay.get("replay_digest"),
        "heldout_camera_ids": sorted(heldout),
        "policy": {
            "significant_alpha_threshold": threshold,
            "rasterization_band_pixels": band,
            "maximum_protected_significant_pixels": maximum_protected,
            "maximum_residual_connected_component_pixels": maximum_component,
            "alpha_method": "paired_identical_rgb_over_black_and_white.v1",
        },
        "render_manifests": {
            name: {"path": str(path), "sha256": _sha256(path)} for name, path in manifests.items()
        },
        "camera_results": rows,
        "determinism_gate_passed": deterministic,
        "partition_gate_passed": partition_passed,
        "protected_records_gate_passed": protected_records_passed,
        "determinism": {
            "two_materializations_identical": deterministic,
            "canonical_manifests_identical": replay.get("canonical_manifests_identical"),
            "receipt_files_byte_identical": replay.get("receipt_files_byte_identical"),
            "output_digests_identical": replay.get("output_digests_identical"),
            "index_sets_identical": replay.get("index_sets_identical"),
            "raw_gpu_contribution_arrays_identical": replay.get(
                "raw_gpu_contribution_arrays_identical"
            ),
        },
        "heldout_gate_passed": passed,
        "replacement_coverage_sweep_authorized": passed,
        "smallest_missing_capability": (
            None
            if passed
            else "calibrated_gaussian_ownership_separation_without_protected_scene_deletion"
        ),
        "claim_ceiling": "public_scene_gaussian_ownership_heldout_audit_not_physical_truth",
    }
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    receipt_path = output / f"{HELDOUT_AUDIT_SCHEMA}.json"
    receipt_path.write_text(canonical_json(receipt) + "\n", encoding="utf-8")
    return receipt


__all__ = [
    "GaussianExcisionHeldoutError",
    "HELDOUT_AUDIT_SCHEMA",
    "derive_alpha_from_background_pair",
    "evaluate_alpha_layer",
    "materialize_gaussian_excision_heldout_audit",
]
