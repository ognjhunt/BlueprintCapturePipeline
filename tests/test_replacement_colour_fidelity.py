from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from blueprint_pipeline.replacement_colour_fidelity import (
    COLOUR_FIDELITY_SCHEMA_VERSION,
    ReplacementColourFidelityError,
    correct_albedo_to_target,
    measure_observed_target_colour,
    evaluate_colour_fidelity,
)


def _png(path: Path, rgb, size=(64, 64), noise=0.0) -> Path:
    array = np.tile(np.asarray(rgb, dtype=np.float64), (size[1], size[0], 1))
    if noise:
        rng = np.random.default_rng(0)
        array = np.clip(array + rng.normal(0, noise, array.shape), 0, 1)
    Image.fromarray((array * 255).astype(np.uint8)).save(path)
    return path


def _mask(path: Path, size=(64, 64), fill=True) -> Path:
    value = 255 if fill else 0
    array = np.zeros((size[1], size[0]), dtype=np.uint8)
    array[16:48, 16:48] = value
    Image.fromarray(array).save(path)
    return path


def test_the_target_is_measured_from_the_scan_not_from_prose(tmp_path: Path) -> None:
    """Prose cannot carry a colour; the observed pixels can."""

    scan = _png(tmp_path / "scan.png", (0.823, 0.774, 0.740))
    mask = _mask(tmp_path / "mask.png")

    target = measure_observed_target_colour(
        observed_image_paths=[scan], region_mask_paths=[mask]
    )

    # an 8-bit PNG round-trip quantises to 1/255, so compare within that
    assert target["target_srgb"] == pytest.approx([0.823, 0.774, 0.740], abs=1 / 255)
    assert target["sampled_pixel_count"] == 32 * 32
    assert target["warmth_r_minus_b"] == pytest.approx(0.083, abs=2 / 255)
    assert target["source"] == "observed_scan_pixels_inside_region_mask"


def test_measuring_with_an_empty_mask_fails_closed(tmp_path: Path) -> None:
    scan = _png(tmp_path / "scan.png", (0.8, 0.8, 0.8))
    mask = _mask(tmp_path / "mask.png", fill=False)

    with pytest.raises(ReplacementColourFidelityError) as excinfo:
        measure_observed_target_colour(
            observed_image_paths=[scan], region_mask_paths=[mask]
        )

    assert any("region_mask_empty" in error for error in excinfo.value.errors)


def test_correction_matches_the_target_while_keeping_texture_detail(
    tmp_path: Path,
) -> None:
    """The generated map keeps its grain; only its colour is pulled to observation."""

    albedo = _png(tmp_path / "albedo.png", (0.904, 0.879, 0.818), noise=0.05)
    before = np.asarray(Image.open(albedo).convert("RGB"), dtype=np.float64) / 255.0

    receipt = correct_albedo_to_target(
        albedo_path=albedo,
        target_srgb=[0.823, 0.774, 0.740],
        destination=tmp_path / "albedo_corrected.png",
    )

    after = (
        np.asarray(Image.open(receipt["corrected_path"]).convert("RGB"), dtype=np.float64)
        / 255.0
    )
    assert np.allclose(after.reshape(-1, 3).mean(0), [0.823, 0.774, 0.740], atol=0.01)
    # detail survives: per-pixel variation is preserved up to the applied gain
    assert after.std() > 0.5 * before.std()
    assert receipt["gain"] == pytest.approx(
        [0.823 / 0.904, 0.774 / 0.879, 0.740 / 0.818], rel=0.02
    )
    assert receipt["detail_preserved"] is True


def test_fidelity_gate_passes_when_the_rendered_result_matches(
    tmp_path: Path,
) -> None:
    observed = _png(tmp_path / "observed.png", (0.823, 0.774, 0.740))
    rendered = _png(tmp_path / "rendered.png", (0.826, 0.771, 0.744))
    mask = _mask(tmp_path / "mask.png")

    receipt = evaluate_colour_fidelity(
        observed_image_paths=[observed],
        rendered_image_paths=[rendered],
        region_mask_paths=[mask],
        maximum_delta=3.0,
    )

    assert receipt["schema_version"] == COLOUR_FIDELITY_SCHEMA_VERSION
    assert receipt["status"] == "colour_fidelity_passed"
    assert receipt["passed"] is True
    assert receipt["worst_delta"] < 3.0
    assert receipt["blockers"] == []


def test_fidelity_gate_fails_a_twin_that_is_too_bright(tmp_path: Path) -> None:
    """The exact defect seen on 840796: right hue, too light."""

    observed = _png(tmp_path / "observed.png", (0.823, 0.774, 0.740))
    rendered = _png(tmp_path / "rendered.png", (0.904, 0.879, 0.818))
    mask = _mask(tmp_path / "mask.png")

    receipt = evaluate_colour_fidelity(
        observed_image_paths=[observed],
        rendered_image_paths=[rendered],
        region_mask_paths=[mask],
        maximum_delta=3.0,
    )

    assert receipt["passed"] is False
    assert "replacement_colour_delta_above_ceiling" in receipt["blockers"]
    assert receipt["worst_delta"] > 3.0
    # hue was fine; the gate should say so rather than blaming the colour cast
    assert abs(receipt["cameras"][0]["warmth_delta"]) < 0.01


def test_fidelity_gate_reports_every_camera(tmp_path: Path) -> None:
    observed = [
        _png(tmp_path / "o0.png", (0.82, 0.77, 0.74)),
        _png(tmp_path / "o1.png", (0.80, 0.75, 0.72)),
    ]
    rendered = [
        _png(tmp_path / "r0.png", (0.82, 0.77, 0.74)),
        _png(tmp_path / "r1.png", (0.95, 0.95, 0.95)),
    ]
    masks = [_mask(tmp_path / "m0.png"), _mask(tmp_path / "m1.png")]

    receipt = evaluate_colour_fidelity(
        observed_image_paths=observed,
        rendered_image_paths=rendered,
        region_mask_paths=masks,
        maximum_delta=3.0,
    )

    assert len(receipt["cameras"]) == 2
    assert receipt["cameras"][0]["delta"] < receipt["cameras"][1]["delta"]
    assert receipt["passed"] is False


def test_mismatched_input_counts_fail_closed(tmp_path: Path) -> None:
    observed = _png(tmp_path / "o.png", (0.8, 0.8, 0.8))
    mask = _mask(tmp_path / "m.png")

    with pytest.raises(ReplacementColourFidelityError) as excinfo:
        evaluate_colour_fidelity(
            observed_image_paths=[observed],
            rendered_image_paths=[],
            region_mask_paths=[mask],
        )

    assert any("input_counts_mismatch" in error for error in excinfo.value.errors)


def test_receipt_round_trips(tmp_path: Path) -> None:
    observed = _png(tmp_path / "observed.png", (0.82, 0.77, 0.74))
    rendered = _png(tmp_path / "rendered.png", (0.82, 0.77, 0.74))
    mask = _mask(tmp_path / "mask.png")

    receipt = evaluate_colour_fidelity(
        observed_image_paths=[observed],
        rendered_image_paths=[rendered],
        region_mask_paths=[mask],
        destination=tmp_path / "fidelity.json",
    )

    stored = json.loads((tmp_path / "fidelity.json").read_text(encoding="utf-8"))
    assert stored == receipt
