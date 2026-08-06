"""Compare a SimReady replacement with the retained multi-view target evidence.

The target-only Gaussian renders provide a synthetic, view-consistent silhouette
and pixel-selection support.  They are not measured geometry or material truth.
This seam therefore diagnoses scale, silhouette, colour, and visual-detail match
without admitting the replacement or overriding human review.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np

from .common import utc_now_iso
from .decision_evidence_contracts import canonical_digest


SCHEMA_VERSION = "adp009b_simready_replacement_match_review.v1"
PRIMARY_SUPPORT_THRESHOLD = 64
SUPPORT_THRESHOLD_SENSITIVITY = (8, 32, 64, 128)
REPLACEMENT_MASK_THRESHOLD = 128
GEOMETRY_GATES = {
    "minimum_median_silhouette_iou": 0.85,
    "minimum_camera_silhouette_iou": 0.75,
    "maximum_median_bbox_dimension_relative_error": 0.10,
    "maximum_median_centroid_error_bbox_diagonal_fraction": 0.08,
}
APPEARANCE_GATES = {"maximum_median_delta_e76": 12.0}


class SimReadyMatchReviewError(ValueError):
    """The replacement or retained reference evidence is invalid."""


def _read(path: Path, *, error: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SimReadyMatchReviewError(error) from exc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _record(path: Path, root: Path) -> dict[str, Any]:
    return {
        "relative_path": path.resolve().relative_to(root.resolve()).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _mask_geometry(reference: np.ndarray, replacement: np.ndarray) -> dict[str, float]:
    if reference.dtype != np.bool_ or replacement.dtype != np.bool_:
        raise SimReadyMatchReviewError("boolean_masks_required")
    if reference.shape != replacement.shape:
        raise SimReadyMatchReviewError("mask_resolution_mismatch")
    reference_y, reference_x = np.nonzero(reference)
    replacement_y, replacement_x = np.nonzero(replacement)
    if not len(reference_x):
        raise SimReadyMatchReviewError("reference_target_support_empty")
    if not len(replacement_x):
        raise SimReadyMatchReviewError("replacement_mask_empty")
    intersection = int(np.count_nonzero(reference & replacement))
    union = int(np.count_nonzero(reference | replacement))
    reference_width = int(reference_x.max() - reference_x.min() + 1)
    reference_height = int(reference_y.max() - reference_y.min() + 1)
    replacement_width = int(replacement_x.max() - replacement_x.min() + 1)
    replacement_height = int(replacement_y.max() - replacement_y.min() + 1)
    reference_center = np.asarray([reference_x.mean(), reference_y.mean()])
    replacement_center = np.asarray([replacement_x.mean(), replacement_y.mean()])
    diagonal = float(np.hypot(reference_width, reference_height))
    return {
        "silhouette_iou": intersection / union,
        "reference_width_px": float(reference_width),
        "reference_height_px": float(reference_height),
        "replacement_width_px": float(replacement_width),
        "replacement_height_px": float(replacement_height),
        "width_relative_error": abs(replacement_width - reference_width) / reference_width,
        "height_relative_error": abs(replacement_height - reference_height) / reference_height,
        "centroid_error_px": float(np.linalg.norm(replacement_center - reference_center)),
        "centroid_error_bbox_diagonal_fraction": float(
            np.linalg.norm(replacement_center - reference_center) / diagonal
        ),
    }


def _opencv_lab_pixels(image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    encoded = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float64)[mask]
    if not len(encoded):
        raise SimReadyMatchReviewError("appearance_mask_empty")
    encoded[:, 0] *= 100.0 / 255.0
    encoded[:, 1:] -= 128.0
    return encoded


def _appearance(
    original_bgr: np.ndarray,
    replacement_bgr: np.ndarray,
    reference: np.ndarray,
    replacement: np.ndarray,
) -> dict[str, Any]:
    reference_lab = _opencv_lab_pixels(original_bgr, reference)
    replacement_lab = _opencv_lab_pixels(replacement_bgr, replacement)
    reference_median = np.median(reference_lab, axis=0)
    replacement_median = np.median(replacement_lab, axis=0)
    reference_luma = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2GRAY)[reference]
    replacement_luma = cv2.cvtColor(replacement_bgr, cv2.COLOR_BGR2GRAY)[replacement]
    return {
        "reference_median_lab": [round(float(value), 4) for value in reference_median],
        "replacement_median_lab": [round(float(value), 4) for value in replacement_median],
        "median_delta_e76": float(np.linalg.norm(reference_median - replacement_median)),
        "reference_luma_standard_deviation": float(np.std(reference_luma)),
        "replacement_luma_standard_deviation": float(np.std(replacement_luma)),
    }


def _annotated_contact_sheet(
    *,
    original: np.ndarray,
    replacement: np.ndarray,
    reference_mask: np.ndarray,
    replacement_mask: np.ndarray,
) -> np.ndarray:
    union_y, union_x = np.nonzero(reference_mask | replacement_mask)
    pad = max(40, int(max(np.ptp(union_x), np.ptp(union_y)) * 0.45))
    x0 = max(0, int(union_x.min()) - pad)
    x1 = min(original.shape[1], int(union_x.max()) + pad + 1)
    y0 = max(0, int(union_y.min()) - pad)
    y1 = min(original.shape[0], int(union_y.max()) + pad + 1)
    original_crop = original[y0:y1, x0:x1].copy()
    replacement_crop = replacement[y0:y1, x0:x1].copy()
    overlay = replacement_crop.copy()
    reference_crop = reference_mask[y0:y1, x0:x1].astype(np.uint8) * 255
    replacement_mask_crop = replacement_mask[y0:y1, x0:x1].astype(np.uint8) * 255
    reference_contours, _ = cv2.findContours(
        reference_crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    replacement_contours, _ = cv2.findContours(
        replacement_mask_crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(overlay, reference_contours, -1, (255, 255, 0), 2)
    cv2.drawContours(overlay, replacement_contours, -1, (255, 0, 255), 2)
    panels = [original_crop, replacement_crop, overlay]
    labels = ["ORIGINAL TARGET", "REPLACEMENT", "CYAN ORIGINAL / MAGENTA REPLACEMENT"]
    for panel, label in zip(panels, labels, strict=True):
        cv2.rectangle(panel, (0, 0), (panel.shape[1], 30), (0, 0, 0), -1)
        cv2.putText(
            panel,
            label,
            (8, 21),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return np.concatenate(panels, axis=1)


def materialize_match_review(
    *,
    edit_input_receipt_path: str | Path,
    edit_input_root: str | Path,
    visual_review_receipt_path: str | Path,
    visual_review_root: str | Path,
    evidence_root: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    evidence = Path(evidence_root).expanduser().resolve()
    output = Path(output_root).expanduser().resolve()
    if output != evidence and evidence not in output.parents:
        raise SimReadyMatchReviewError("match_review_output_outside_evidence_root")
    edit_root = Path(edit_input_root).expanduser().resolve()
    visual_root = Path(visual_review_root).expanduser().resolve()
    edit_receipt_path = Path(edit_input_receipt_path).expanduser().resolve()
    visual_receipt_path = Path(visual_review_receipt_path).expanduser().resolve()
    edit_receipt = _read(edit_receipt_path, error="edit_input_receipt_invalid")
    visual_receipt = _read(visual_receipt_path, error="visual_review_receipt_invalid")
    for value, field, error in (
        (edit_receipt, "receipt_digest", "edit_input_receipt_digest_invalid"),
        (visual_receipt, "receipt_digest", "visual_review_receipt_digest_invalid"),
    ):
        if value.get(field) != canonical_digest(value, digest_field=field):
            raise SimReadyMatchReviewError(error)
    if edit_receipt.get("status") != "render_derived_input_packet_materialized":
        raise SimReadyMatchReviewError("render_derived_edit_input_required")
    if visual_receipt.get("status") != "rendered_visual_review_candidate":
        raise SimReadyMatchReviewError("rendered_visual_review_candidate_required")
    if any(
        key in edit_receipt or key in visual_receipt
        for key in ("admitted", "acceptance", "match_accepted")
    ):
        raise SimReadyMatchReviewError("caller_asserted_acceptance_forbidden")
    derived = edit_receipt.get("derived_artifacts")
    if not isinstance(derived, Mapping) or not isinstance(derived.get("images"), list):
        raise SimReadyMatchReviewError("edit_input_images_missing")
    image_by_camera = {
        str(row.get("camera_id")): row
        for row in derived["images"]
        if isinstance(row, Mapping)
    }
    visual_rows = visual_receipt.get("artifacts")
    if not isinstance(visual_rows, list):
        raise SimReadyMatchReviewError("visual_review_artifacts_missing")
    visual_by_camera = {
        str(row.get("camera_id")): row for row in visual_rows if isinstance(row, Mapping)
    }
    if not image_by_camera or set(image_by_camera) != set(visual_by_camera):
        raise SimReadyMatchReviewError("match_review_camera_identity_mismatch")
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for camera_id in sorted(image_by_camera):
        image_record = image_by_camera[camera_id]
        visual_record = visual_by_camera[camera_id]
        original_path = edit_root / str(image_record.get("relative_path") or "")
        support_path = edit_root / "target_support" / f"{camera_id}.png"
        after_record = visual_record.get("after")
        mask_record = visual_record.get("replacement_mask")
        if not isinstance(after_record, Mapping) or not isinstance(mask_record, Mapping):
            raise SimReadyMatchReviewError(f"visual_review_artifact_record_missing:{camera_id}")
        replacement_path = visual_root / str(after_record.get("relative_path") or "")
        replacement_mask_path = visual_root / str(mask_record.get("relative_path") or "")
        expected = (
            (original_path, image_record.get("sha256"), "original_frame_digest_mismatch"),
            (replacement_path, after_record.get("sha256"), "replacement_frame_digest_mismatch"),
            (
                replacement_mask_path,
                mask_record.get("sha256"),
                "replacement_mask_digest_mismatch",
            ),
        )
        for path, digest, error in expected:
            if not path.is_file() or _sha256(path) != digest:
                raise SimReadyMatchReviewError(f"{error}:{camera_id}")
        if not support_path.is_file():
            raise SimReadyMatchReviewError(f"target_support_render_missing:{camera_id}")
        original = cv2.imread(str(original_path), cv2.IMREAD_COLOR)
        target_support = cv2.imread(str(support_path), cv2.IMREAD_UNCHANGED)
        replacement_image = cv2.imread(str(replacement_path), cv2.IMREAD_COLOR)
        replacement_alpha = cv2.imread(str(replacement_mask_path), cv2.IMREAD_GRAYSCALE)
        if (
            original is None
            or target_support is None
            or replacement_image is None
            or replacement_alpha is None
            or target_support.ndim != 3
            or target_support.shape[2] < 3
        ):
            raise SimReadyMatchReviewError(f"match_review_image_invalid:{camera_id}")
        if not (
            original.shape[:2]
            == target_support.shape[:2]
            == replacement_image.shape[:2]
            == replacement_alpha.shape[:2]
        ):
            raise SimReadyMatchReviewError(f"match_review_resolution_mismatch:{camera_id}")
        # The target-support renderer stores Gaussian opacity in grayscale RGB;
        # its alpha channel is an opaque file container and is intentionally ignored.
        support_intensity = np.max(target_support[:, :, :3], axis=2)
        replacement_mask = replacement_alpha >= REPLACEMENT_MASK_THRESHOLD
        sensitivity: dict[str, Any] = {}
        for threshold in SUPPORT_THRESHOLD_SENSITIVITY:
            sensitivity[str(threshold)] = _mask_geometry(
                support_intensity >= threshold, replacement_mask
            )
        reference_mask = support_intensity >= PRIMARY_SUPPORT_THRESHOLD
        geometry = sensitivity[str(PRIMARY_SUPPORT_THRESHOLD)]
        appearance = _appearance(
            original, replacement_image, reference_mask, replacement_mask
        )
        softness_band = (support_intensity >= SUPPORT_THRESHOLD_SENSITIVITY[0]) & (
            support_intensity < SUPPORT_THRESHOLD_SENSITIVITY[-1]
        )
        support_extent = support_intensity >= SUPPORT_THRESHOLD_SENSITIVITY[0]
        replacement_softness = (replacement_alpha > 0) & (
            replacement_alpha < REPLACEMENT_MASK_THRESHOLD
        )
        sheet = _annotated_contact_sheet(
            original=original,
            replacement=replacement_image,
            reference_mask=reference_mask,
            replacement_mask=replacement_mask,
        )
        sheet_path = output / f"{camera_id}.original_replacement_match.png"
        if not cv2.imwrite(str(sheet_path), sheet):
            raise SimReadyMatchReviewError(f"match_contact_sheet_write_failed:{camera_id}")
        rows.append(
            {
                "camera_id": camera_id,
                "original_frame": _record(original_path, edit_root),
                "target_support_render": _record(support_path, edit_root),
                "replacement_frame": _record(replacement_path, visual_root),
                "replacement_mask": _record(replacement_mask_path, visual_root),
                "geometry_at_primary_threshold": geometry,
                "threshold_sensitivity": sensitivity,
                "appearance": appearance,
                "edge_softness_diagnostic": {
                    "reference_transition_fraction_of_support_extent": float(
                        np.count_nonzero(softness_band) / np.count_nonzero(support_extent)
                    ),
                    "replacement_antialias_fraction_of_nonzero_extent": float(
                        np.count_nonzero(replacement_softness)
                        / max(1, np.count_nonzero(replacement_alpha))
                    ),
                    "is_admission_gate": False,
                },
                "contact_sheet": _record(sheet_path, output),
            }
        )
    median_iou = float(
        np.median([row["geometry_at_primary_threshold"]["silhouette_iou"] for row in rows])
    )
    minimum_iou = float(
        min(row["geometry_at_primary_threshold"]["silhouette_iou"] for row in rows)
    )
    median_width_error = float(
        np.median(
            [row["geometry_at_primary_threshold"]["width_relative_error"] for row in rows]
        )
    )
    median_height_error = float(
        np.median(
            [row["geometry_at_primary_threshold"]["height_relative_error"] for row in rows]
        )
    )
    median_centroid_error = float(
        np.median(
            [
                row["geometry_at_primary_threshold"][
                    "centroid_error_bbox_diagonal_fraction"
                ]
                for row in rows
            ]
        )
    )
    median_delta_e = float(np.median([row["appearance"]["median_delta_e76"] for row in rows]))
    geometry_pass = (
        median_iou >= GEOMETRY_GATES["minimum_median_silhouette_iou"]
        and minimum_iou >= GEOMETRY_GATES["minimum_camera_silhouette_iou"]
        and max(median_width_error, median_height_error)
        <= GEOMETRY_GATES["maximum_median_bbox_dimension_relative_error"]
        and median_centroid_error
        <= GEOMETRY_GATES["maximum_median_centroid_error_bbox_diagonal_fraction"]
    )
    appearance_pass = median_delta_e <= APPEARANCE_GATES["maximum_median_delta_e76"]
    blockers = ["human_multiview_identity_review_pending", "native_ovrtx_material_render_missing"]
    if not geometry_pass:
        blockers.append("replacement_multiview_silhouette_match_below_threshold")
    if not appearance_pass:
        blockers.append("replacement_color_material_match_below_threshold")
    receipt: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": utc_now_iso(),
        "status": "diagnosed_match_candidate" if geometry_pass and appearance_pass else "diagnosed_mismatch",
        "edit_input_receipt_digest": edit_receipt["receipt_digest"],
        "visual_review_receipt_digest": visual_receipt["receipt_digest"],
        "frozen_thresholds": {
            "primary_target_support_intensity": PRIMARY_SUPPORT_THRESHOLD,
            "target_support_sensitivity": list(SUPPORT_THRESHOLD_SENSITIVITY),
            "replacement_mask_intensity": REPLACEMENT_MASK_THRESHOLD,
            "geometry": GEOMETRY_GATES,
            "appearance": APPEARANCE_GATES,
        },
        "aggregate": {
            "camera_count": len(rows),
            "median_silhouette_iou": median_iou,
            "minimum_camera_silhouette_iou": minimum_iou,
            "median_width_relative_error": median_width_error,
            "median_height_relative_error": median_height_error,
            "median_centroid_error_bbox_diagonal_fraction": median_centroid_error,
            "median_delta_e76": median_delta_e,
            "projected_scale_and_pose_gate_passed": geometry_pass,
            "colour_appearance_gate_passed": appearance_pass,
        },
        "camera_results": rows,
        "human_multiview_identity_review": "pending",
        "blockers": blockers,
        "proof_boundaries": [
            "target support is synthetic InteriorGS Gaussian evidence, not measured geometry",
            "colour comparison is diagnostic and may include reconstruction blur or view-dependent effects",
            "a projected scale pass does not establish detailed geometry or material identity",
            "this receipt cannot admit the SimReady replacement or prove dynamics",
        ],
        "claim_ceiling": "synthetic_multiview_replacement_matching_diagnostic",
    }
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    receipt_path = output / "adp009b_simready_replacement_match_review.v1.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edit-input-receipt", type=Path, required=True)
    parser.add_argument("--edit-input-root", type=Path, required=True)
    parser.add_argument("--visual-review-receipt", type=Path, required=True)
    parser.add_argument("--visual-review-root", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    materialize_match_review(
        edit_input_receipt_path=args.edit_input_receipt,
        edit_input_root=args.edit_input_root,
        visual_review_receipt_path=args.visual_review_receipt,
        visual_review_root=args.visual_review_root,
        evidence_root=args.evidence_root,
        output_root=args.output_root,
    )


if __name__ == "__main__":
    main()
