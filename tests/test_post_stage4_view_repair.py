"""Tests for post_stage4_view_repair.py."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from post_stage4_view_repair import (
    apply_acceptance_gate,
    build_repair_mask,
    compute_photometric_drift_outside_mask,
    _run_backend,
    _run_fixer_native,
    repair_candidate_views,
)


# ---------------------------------------------------------------------------
# Test: Repair mask
# ---------------------------------------------------------------------------

class TestBuildRepairMask:
    def test_dark_pixels_masked(self) -> None:
        rgb = np.zeros((64, 64, 3), dtype=np.uint8)
        mask = build_repair_mask(rgb)
        assert mask.all()

    def test_bright_pixels_not_masked(self) -> None:
        rgb = np.full((64, 64, 3), 200, dtype=np.uint8)
        mask = build_repair_mask(rgb)
        assert not mask.any()

    def test_alpha_zero_masked(self) -> None:
        rgb = np.full((64, 64, 3), 200, dtype=np.uint8)
        alpha = np.zeros((64, 64), dtype=np.uint8)
        mask = build_repair_mask(rgb, alpha=alpha)
        assert mask.all()


# ---------------------------------------------------------------------------
# Test: Photometric drift
# ---------------------------------------------------------------------------

class TestPhotometricDrift:
    def test_identical_images(self) -> None:
        rgb = np.full((64, 64, 3), 128, dtype=np.uint8)
        mask = np.zeros((64, 64), dtype=bool)
        assert compute_photometric_drift_outside_mask(rgb, rgb, mask) == pytest.approx(0.0)

    def test_all_masked(self) -> None:
        before = np.full((64, 64, 3), 128, dtype=np.uint8)
        after = np.full((64, 64, 3), 0, dtype=np.uint8)
        mask = np.ones((64, 64), dtype=bool)
        # All pixels masked → drift = 1.0 (no observable region)
        assert compute_photometric_drift_outside_mask(before, after, mask) == 1.0

    def test_shape_mismatch(self) -> None:
        before = np.zeros((64, 64, 3), dtype=np.uint8)
        after = np.zeros((32, 32, 3), dtype=np.uint8)
        mask = np.zeros((64, 64), dtype=bool)
        assert compute_photometric_drift_outside_mask(before, after, mask) == 1.0


# ---------------------------------------------------------------------------
# Test: Acceptance gate
# ---------------------------------------------------------------------------

class TestAcceptanceGate:
    def test_accept_good_view(self) -> None:
        rows = [{
            "cross_view_reprojection_error_px": 0.5,
            "photometric_drift_outside_mask": 0.02,
        }]
        accepted, rejected = apply_acceptance_gate(rows)
        assert len(accepted) == 1
        assert len(rejected) == 0
        assert accepted[0]["accepted"] is True

    def test_reject_high_drift(self) -> None:
        rows = [{
            "cross_view_reprojection_error_px": 0.5,
            "photometric_drift_outside_mask": 0.5,  # Way above threshold
        }]
        accepted, rejected = apply_acceptance_gate(rows)
        assert len(accepted) == 0
        assert len(rejected) == 1
        assert "outside_mask_drift" in rejected[0]["gate_reasons"]

    def test_reject_high_reprojection(self) -> None:
        rows = [{
            "cross_view_reprojection_error_px": 10.0,  # Way above threshold
            "photometric_drift_outside_mask": 0.01,
        }]
        accepted, rejected = apply_acceptance_gate(rows)
        assert len(accepted) == 0
        assert len(rejected) == 1
        assert "reprojection_error" in rejected[0]["gate_reasons"]

    def test_reject_passthrough_backend(self) -> None:
        rows = [{
            "cross_view_reprojection_error_px": 0.0,
            "photometric_drift_outside_mask": 0.0,
            "backend_mode": "passthrough",
        }]
        accepted, rejected = apply_acceptance_gate(rows)
        assert len(accepted) == 0
        assert len(rejected) == 1
        assert "backend_passthrough" in rejected[0]["gate_reasons"]


# ---------------------------------------------------------------------------
# Test: _run_fixer_native
# ---------------------------------------------------------------------------

class TestRunFixerNative:
    def test_missing_fixer_returns_false(self, tmp_path: Path) -> None:
        input_path = tmp_path / "input.png"
        input_path.write_bytes(b"fake")
        mask_path = tmp_path / "mask.png"
        mask_path.write_bytes(b"fake")
        output_path = tmp_path / "output.png"

        ok, mode = _run_fixer_native(
            input_path=input_path,
            mask_path=mask_path,
            output_path=output_path,
            fixer_dir="/nonexistent/fixer",
            fixer_weights_dir="/nonexistent/weights",
        )
        assert ok is False
        assert mode == "fixer_missing"

    def test_native_fixer_composites_only_masked_pixels(self, tmp_path: Path) -> None:
        PILImage = pytest.importorskip("PIL.Image")
        input_path = tmp_path / "input.png"
        mask_path = tmp_path / "mask.png"
        output_path = tmp_path / "output.png"

        src = np.full((8, 8, 3), 64, dtype=np.uint8)
        PILImage.fromarray(src).save(input_path)
        mask = np.zeros((8, 8), dtype=np.uint8)
        mask[:, :4] = 255
        PILImage.fromarray(mask).save(mask_path)

        fixer_root = tmp_path / "Fixer"
        fixer_src = fixer_root / "src"
        fixer_src.mkdir(parents=True)
        inference = fixer_src / "inference_pretrained_model.py"
        inference.write_text(
            "\n".join(
                [
                    "import argparse",
                    "from pathlib import Path",
                    "from PIL import Image",
                    "p = argparse.ArgumentParser()",
                    "p.add_argument('--model')",
                    "p.add_argument('--input')",
                    "p.add_argument('--output')",
                    "p.add_argument('--timestep')",
                    "p.add_argument('--resolution')",
                    "a = p.parse_args()",
                    "inp = sorted(Path(a.input).iterdir())[0]",
                    "out_dir = Path(a.output)",
                    "out_dir.mkdir(parents=True, exist_ok=True)",
                    "img = Image.open(inp).convert('RGB')",
                    "Image.new('RGB', img.size, (255, 255, 255)).save(out_dir / inp.name)",
                ]
            ),
            encoding="utf-8",
        )
        weights = tmp_path / "weights" / "pretrained"
        weights.mkdir(parents=True)
        (weights / "pretrained_fixer.pkl").write_bytes(b"x")

        ok, mode = _run_fixer_native(
            input_path=input_path,
            mask_path=mask_path,
            output_path=output_path,
            fixer_dir=str(fixer_root),
            fixer_weights_dir=str(tmp_path / "weights"),
            fixer_python=sys.executable,
        )
        assert ok is True
        assert mode == "fixer_native"
        out = np.asarray(PILImage.open(output_path).convert("RGB"))
        assert np.all(out[:, :4, :] == 255)
        assert np.all(out[:, 4:, :] == 64)


# ---------------------------------------------------------------------------
# Test: _run_backend
# ---------------------------------------------------------------------------

class TestRunBackend:
    def test_passthrough_fallback(self, tmp_path: Path) -> None:
        """Without Fixer or template command, backend falls through to passthrough."""
        input_path = tmp_path / "input.png"
        PILImage = pytest.importorskip("PIL.Image")
        PILImage.fromarray(np.full((32, 32, 3), 128, dtype=np.uint8)).save(input_path)

        mask_path = tmp_path / "mask.png"
        PILImage.fromarray(np.zeros((32, 32), dtype=np.uint8)).save(mask_path)
        output_path = tmp_path / "output.png"

        # No env vars set, no Fixer installed → passthrough
        with mock.patch.dict("os.environ", {
            "POST_STAGE4_FIXER_IMAGE_COMMAND": "",
            "FIXER_DIR": "/nonexistent",
            "FIXER_WEIGHTS_DIR": "/nonexistent",
        }):
            ok, mode = _run_backend(
                backend="fixer",
                input_path=input_path,
                mask_path=mask_path,
                output_path=output_path,
            )
        assert ok is True
        assert mode == "passthrough"
        assert output_path.is_file()

    def test_gsfix3d_passthrough(self, tmp_path: Path) -> None:
        input_path = tmp_path / "input.png"
        PILImage = pytest.importorskip("PIL.Image")
        PILImage.fromarray(np.full((32, 32, 3), 128, dtype=np.uint8)).save(input_path)

        mask_path = tmp_path / "mask.png"
        PILImage.fromarray(np.zeros((32, 32), dtype=np.uint8)).save(mask_path)
        output_path = tmp_path / "output.png"

        ok, mode = _run_backend(
            backend="gsfix3d",
            input_path=input_path,
            mask_path=mask_path,
            output_path=output_path,
        )
        assert ok is True
        assert mode == "passthrough"


# ---------------------------------------------------------------------------
# Test: E2E repair_candidate_views
# ---------------------------------------------------------------------------

class TestRepairCandidateViews:
    def test_no_candidates(self, tmp_path: Path) -> None:
        renders_dir = tmp_path / "renders"
        renders_dir.mkdir()
        candidates_path = tmp_path / "candidates.jsonl"
        candidates_path.write_text("")
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        report = repair_candidate_views(
            renders_dir=renders_dir,
            candidate_views_path=candidates_path,
            output_dir=output_dir,
            model_mode="fixer",
            max_reprojection_error_px=2.5,
            max_photometric_drift=0.08,
        )
        assert report["accepted_count"] == 0
        assert report["rejected_count"] == 0

    def test_candidate_with_render_image(self, tmp_path: Path) -> None:
        renders_dir = tmp_path / "renders"
        renders_dir.mkdir()
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Create a render image (half dark).
        PILImage = pytest.importorskip("PIL.Image")
        img = np.full((64, 64, 3), 200, dtype=np.uint8)
        img[:32, :, :] = 0
        render_path = renders_dir / "frame_0001.png"
        PILImage.fromarray(img).save(render_path)

        candidates_path = tmp_path / "candidates.jsonl"
        candidates_path.write_text(
            json.dumps({
                "id": "test_01",
                "source_image": "frame_0001.png",
                "render_image": str(render_path),
                "cross_view_reprojection_error_px": 0.0,
            }) + "\n"
        )

        with mock.patch.dict("os.environ", {
            "POST_STAGE4_FIXER_IMAGE_COMMAND": "",
            "FIXER_DIR": "/nonexistent",
            "FIXER_WEIGHTS_DIR": "/nonexistent",
        }):
            report = repair_candidate_views(
                renders_dir=renders_dir,
                candidate_views_path=candidates_path,
                output_dir=output_dir,
                model_mode="fixer",
                max_reprojection_error_px=2.5,
                max_photometric_drift=0.08,
            )

        assert report["accepted_count"] == 0
        assert report["rejected_count"] == 1
        assert report["rows"][0]["backend_mode"] == "passthrough"
        assert "backend_passthrough" in report["rows"][0]["gate_reasons"]
