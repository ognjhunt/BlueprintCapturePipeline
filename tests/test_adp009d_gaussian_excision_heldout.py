from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from blueprint_pipeline.decision_evidence_contracts import canonical_digest, canonical_json
from blueprint_pipeline.public_scene_gaussian_excision_heldout import (
    GaussianExcisionHeldoutError,
    HELDOUT_AUDIT_SCHEMA,
    derive_alpha_from_background_pair,
    evaluate_alpha_layer,
    materialize_gaussian_excision_heldout_audit,
)


def _composite(foreground: np.ndarray, alpha: np.ndarray, background: int) -> np.ndarray:
    value = foreground * alpha[..., None] + background * (1.0 - alpha[..., None])
    return np.rint(value).clip(0, 255).astype(np.uint8)


def test_background_pair_recovers_colour_independent_alpha() -> None:
    foreground = np.array([[[20.0, 200.0, 80.0], [250.0, 10.0, 120.0]]], dtype=np.float32)
    alpha = np.array([[0.25, 0.75]], dtype=np.float32)

    recovered = derive_alpha_from_background_pair(
        _composite(foreground, alpha, 0), _composite(foreground, alpha, 255)
    )

    assert recovered == pytest.approx(alpha, abs=1.0 / 255.0)


def test_background_pair_rejects_mismatched_or_non_rgb_inputs() -> None:
    with pytest.raises(GaussianExcisionHeldoutError) as exc:
        derive_alpha_from_background_pair(
            np.zeros((2, 2, 3), dtype=np.uint8), np.zeros((2, 2, 4), dtype=np.uint8)
        )
    assert exc.value.codes == ("heldout_background_pair_shape_invalid",)


def test_alpha_layer_excludes_two_pixel_band_and_measures_components() -> None:
    mask = np.zeros((11, 11), dtype=bool)
    mask[4:7, 4:7] = True
    alpha = np.zeros((11, 11), dtype=np.float32)
    alpha[4:7, 4:7] = 1.0
    alpha[5, 5] = 0.0
    alpha[5, 8] = 1.0  # one pixel beyond the mask, but inside the two-pixel band
    alpha[0:2, 0:2] = 1.0  # protected connected component

    result = evaluate_alpha_layer(
        alpha,
        mask,
        significant_alpha_threshold=1.0 / 255.0,
        rasterization_band_pixels=2,
    )

    assert result["silhouette_missing_pixel_count"] == 1
    assert result["silhouette_largest_missing_component_pixels"] == 1
    assert result["protected_significant_pixel_count"] == 4
    assert result["inside_mask_significant_pixel_count"] == 8
    assert result["inside_mask_largest_significant_component_pixels"] == 8


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(value) + "\n", encoding="utf-8")


def _render_manifest(
    root: Path,
    name: str,
    camera_ids: list[str],
    images: dict[str, np.ndarray],
    *,
    background: str,
) -> Path:
    output = root / name
    rows = []
    for camera_id in camera_ids:
        frame = output / "frames" / f"{camera_id}.png"
        frame.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(images[camera_id]).save(frame)
        rows.append(
            {
                "camera_id": camera_id,
                "relative_path": f"frames/{camera_id}.png",
                "digest": _sha256(frame),
                "width": images[camera_id].shape[1],
                "height": images[camera_id].shape[0],
                "pixel_std": float(images[camera_id].std()),
            }
        )
    value: dict[str, object] = {
        "schema_version": "sealed_camera_render_manifest.v1",
        "status": "rendered_exact_cameras",
        "renderer_identity": {"background_rgb": background},
        "renders": rows,
    }
    value["sealed_camera_render_manifest_digest"] = canonical_digest(
        value, digest_field="sealed_camera_render_manifest_digest"
    )
    path = output / "sealed_camera_render_manifest.v1.json"
    _write_json(path, value)
    return path


@pytest.mark.parametrize(("scene_id", "should_pass"), [("840313", True), ("840796", False)])
def test_materialize_heldout_audit_is_scene_neutral_and_fail_closed(
    tmp_path: Path, scene_id: str, should_pass: bool
) -> None:
    camera_ids = ["calibration", "far_left", "far_right"]
    mask = np.zeros((12, 12), dtype=np.uint8)
    mask[3:9, 3:9] = 255
    mask_paths = {}
    source_paths = {}
    for camera_id in camera_ids:
        mask_path = tmp_path / "inputs" / "masks" / f"{camera_id}.png"
        source_path = tmp_path / "inputs" / "images" / f"{camera_id}.png"
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        source_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(mask).save(mask_path)
        Image.fromarray(np.full((12, 12, 3), 127, dtype=np.uint8)).save(source_path)
        mask_paths[camera_id] = mask_path
        source_paths[camera_id] = source_path
    freeze: dict[str, object] = {
        "schema_version": "adp009b_gaussian_excision_audit_freeze.v1",
        "scene": {"publisher_scene_id": scene_id},
        "camera_split": {
            "calibration_camera_ids": ["calibration"],
            "heldout_camera_ids": ["far_left", "far_right"],
        },
        "source_images": [
            {"camera_id": camera_id, "path": str(source_paths[camera_id])}
            for camera_id in camera_ids
        ],
        "masks": [
            {
                "camera_id": camera_id,
                "historical_outer_mask": {"path": str(mask_paths[camera_id])},
            }
            for camera_id in camera_ids
        ],
        "policy": {
            "heldout_significant_alpha_threshold": 1.0 / 255.0,
            "heldout_rasterization_band_pixels": 2,
            "heldout_maximum_residual_connected_component_pixels": 4,
            "heldout_maximum_protected_significant_pixels": 0,
        },
        "historical_baseline": {"selected_gaussian_count": 4},
    }
    freeze["freeze_digest"] = canonical_digest(freeze, digest_field="freeze_digest")
    freeze_path = tmp_path / "freeze.json"
    _write_json(freeze_path, freeze)
    ownership: dict[str, object] = {
        "schema_version": "adp009b_gaussian_excision_ownership_receipt.v1",
        "freeze_digest": freeze["freeze_digest"],
        "heldout_cameras_accessed_for_classification": False,
        "ownership": {
            "source_gaussian_count": 12,
            "owned_count": 4,
            "retained_count": 4,
            "ambiguous_count": 4,
            "historical_obb_count": 4,
            "exhaustive": True,
            "pairwise_disjoint": True,
        },
        "determinism": {"quantized_contribution_arrays_identical": should_pass},
    }
    ownership["receipt_digest"] = canonical_digest(ownership, digest_field="receipt_digest")
    ownership_path = tmp_path / "ownership.json"
    _write_json(ownership_path, ownership)
    replay: dict[str, object] = {
        "schema_version": "adp009b_gaussian_excision_ownership_replay.v1",
        "execution_count": 2,
        "freeze_digest": freeze["freeze_digest"],
        "ownership_receipt_digest": ownership["receipt_digest"],
        "canonical_manifests_identical": True,
        "receipt_files_byte_identical": True,
        "output_digests_identical": True,
        "index_sets_identical": True,
        "protected_source_records_byte_identical": True,
        "raw_gpu_contribution_arrays_identical": False,
        "gate_passed": True,
    }
    replay["replay_digest"] = canonical_digest(replay, digest_field="replay_digest")
    replay_path = tmp_path / "ownership-replay.json"
    _write_json(replay_path, replay)

    exact_alpha = mask.astype(np.float32) / 255.0
    owned_alpha = exact_alpha.copy()
    ambiguous_alpha = np.zeros_like(exact_alpha)
    if not should_pass:
        owned_alpha[3:9, 6:9] = 0.0
        ambiguous_alpha[3:9, 3:9] = 1.0
    obb_alpha = exact_alpha.copy()
    obb_alpha[0, 0] = 1.0
    foreground = np.full((12, 12, 3), 180.0, dtype=np.float32)

    def pair(alpha: np.ndarray, label: str) -> tuple[Path, Path]:
        black = {camera_id: _composite(foreground, alpha, 0) for camera_id in camera_ids}
        white = {camera_id: _composite(foreground, alpha, 255) for camera_id in camera_ids}
        return (
            _render_manifest(tmp_path, f"{label}-black", camera_ids, black, background="#000000"),
            _render_manifest(tmp_path, f"{label}-white", camera_ids, white, background="#ffffff"),
        )

    obb_black, obb_white = pair(obb_alpha, "obb")
    owned_black, owned_white = pair(owned_alpha, "owned")
    ambiguous_black, ambiguous_white = pair(ambiguous_alpha, "ambiguous")
    retained_images = {
        camera_id: np.full((12, 12, 3), 90, dtype=np.uint8) for camera_id in camera_ids
    }
    retained = _render_manifest(
        tmp_path, "retained", camera_ids, retained_images, background="#000000"
    )

    receipt = materialize_gaussian_excision_heldout_audit(
        freeze_path=freeze_path,
        ownership_receipt_path=ownership_path,
        ownership_replay_receipt_path=replay_path,
        obb_black_manifest_path=obb_black,
        obb_white_manifest_path=obb_white,
        owned_black_manifest_path=owned_black,
        owned_white_manifest_path=owned_white,
        ambiguous_black_manifest_path=ambiguous_black,
        ambiguous_white_manifest_path=ambiguous_white,
        retained_scene_manifest_path=retained,
        output_root=tmp_path / "audit",
    )

    assert receipt["schema_version"] == HELDOUT_AUDIT_SCHEMA
    assert receipt["heldout_gate_passed"] is should_pass
    assert receipt["replacement_coverage_sweep_authorized"] is should_pass
    assert receipt["determinism_gate_passed"] is True
    assert receipt["determinism"]["raw_gpu_contribution_arrays_identical"] is False
    assert len(receipt["camera_results"]) == 3
    assert all(
        (tmp_path / "audit" / row["contact_sheet"]["relative_path"]).is_file()
        for row in receipt["camera_results"]
    )
    index_path = tmp_path / "audit" / receipt["contact_sheet_index"]["relative_path"]
    assert index_path.is_file()
    index = index_path.read_text(encoding="utf-8")
    assert "original | exact mask | OBB removed-only" in index
    assert all(camera_id in index for camera_id in camera_ids)
