"""Tests for post_stage4_view_repair.py."""

from __future__ import annotations

import json
import shutil
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
    _build_parser,
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

    def test_worldforge_template_command_success(self, tmp_path: Path) -> None:
        input_path = tmp_path / "input.png"
        PILImage = pytest.importorskip("PIL.Image")
        PILImage.fromarray(np.full((16, 16, 3), 128, dtype=np.uint8)).save(input_path)
        mask_path = tmp_path / "mask.png"
        PILImage.fromarray(np.zeros((16, 16), dtype=np.uint8)).save(mask_path)
        output_path = tmp_path / "output.png"

        with mock.patch.dict("os.environ", {"POST_STAGE4_WORLDFORGE_IMAGE_COMMAND": "template"}):
            with mock.patch("post_stage4_view_repair._run_template_command", return_value=True) as run_template:
                with mock.patch("post_stage4_view_repair._run_worldforge_native") as run_native:
                    ok, mode = _run_backend(
                        backend="worldforge",
                        input_path=input_path,
                        mask_path=mask_path,
                        output_path=output_path,
                    )
        assert ok is True
        assert mode == "command"
        run_template.assert_called_once()
        run_native.assert_not_called()

    def test_worldforge_native_fallback_when_template_unset(self, tmp_path: Path) -> None:
        input_path = tmp_path / "input.png"
        PILImage = pytest.importorskip("PIL.Image")
        PILImage.fromarray(np.full((16, 16, 3), 128, dtype=np.uint8)).save(input_path)
        mask_path = tmp_path / "mask.png"
        PILImage.fromarray(np.zeros((16, 16), dtype=np.uint8)).save(mask_path)
        output_path = tmp_path / "output.png"

        with mock.patch.dict("os.environ", {"POST_STAGE4_WORLDFORGE_IMAGE_COMMAND": ""}):
            with mock.patch(
                "post_stage4_view_repair._run_worldforge_native",
                return_value=(True, "worldforge_native_longcat"),
            ) as run_native:
                ok, mode = _run_backend(
                    backend="worldforge",
                    input_path=input_path,
                    mask_path=mask_path,
                    output_path=output_path,
                )
        assert ok is True
        assert mode == "worldforge_native_longcat"
        run_native.assert_called_once()

    def test_worldforge_missing_falls_back_to_passthrough(self, tmp_path: Path) -> None:
        input_path = tmp_path / "input.png"
        PILImage = pytest.importorskip("PIL.Image")
        source = np.full((16, 16, 3), 96, dtype=np.uint8)
        PILImage.fromarray(source).save(input_path)
        mask_path = tmp_path / "mask.png"
        PILImage.fromarray(np.zeros((16, 16), dtype=np.uint8)).save(mask_path)
        output_path = tmp_path / "output.png"

        with mock.patch.dict("os.environ", {"POST_STAGE4_WORLDFORGE_IMAGE_COMMAND": ""}):
            with mock.patch(
                "post_stage4_view_repair._run_worldforge_native",
                return_value=(False, "worldforge_missing"),
            ):
                ok, mode = _run_backend(
                    backend="worldforge",
                    input_path=input_path,
                    mask_path=mask_path,
                    output_path=output_path,
                )
        assert ok is True
        assert mode == "passthrough"
        out = np.asarray(PILImage.open(output_path).convert("RGB"))
        assert np.all(out == source)


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

    def test_virtual_candidate_resolves_render_from_mapping(self, tmp_path: Path) -> None:
        renders_dir = tmp_path / "renders"
        renders_dir.mkdir()
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        PILImage = pytest.importorskip("PIL.Image")
        img = np.full((32, 32, 3), 180, dtype=np.uint8)
        mapped_render = renders_dir / "00000.png"
        PILImage.fromarray(img).save(mapped_render)

        candidates_path = tmp_path / "candidates.jsonl"
        candidates_path.write_text(
            json.dumps(
                {
                    "id": "virtual_01",
                    "is_virtual": True,
                    "source_image": "00000.png",
                    "render_image": "",
                    "qvec": [1.0, 0.0, 0.0, 0.0],
                    "tvec": [0.0, 0.0, 0.0],
                    "camera_id": 7,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        mapping_path = tmp_path / "mapping.jsonl"
        mapping_path.write_text(
            json.dumps(
                {
                    "candidate_id": "virtual_01",
                    "render_image": str(mapped_render),
                    "qvec": [1.0, 0.0, 0.0, 0.0],
                    "tvec": [0.0, 0.0, 0.0],
                    "camera_id": 7,
                    "predicted_hole_ratio": 0.2,
                }
            )
            + "\n",
            encoding="utf-8",
        )

        with mock.patch.dict("os.environ", {"POST_STAGE4_FIXER_IMAGE_COMMAND": ""}):
            report = repair_candidate_views(
                renders_dir=renders_dir,
                candidate_views_path=candidates_path,
                output_dir=output_dir,
                model_mode="fixer",
                max_reprojection_error_px=2.5,
                max_photometric_drift=0.08,
                virtual_render_mapping_path=mapping_path,
            )

        assert report["rejected_count"] == 1
        row = report["rows"][0]
        assert row["candidate_id"] == "virtual_01"
        assert row["is_virtual"] is True
        assert row["render_image"] == str(mapped_render)
        assert row["camera_id"] == 7
        assert float(row["predicted_hole_ratio"]) == pytest.approx(0.2)
        assert row["qvec"] == [1.0, 0.0, 0.0, 0.0]
        assert row["tvec"] == [0.0, 0.0, 0.0]

    def test_virtual_candidate_missing_mapping_is_rejected_explicitly(self, tmp_path: Path) -> None:
        renders_dir = tmp_path / "renders"
        renders_dir.mkdir()
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        candidates_path = tmp_path / "candidates.jsonl"
        candidates_path.write_text(
            json.dumps(
                {
                    "id": "virtual_missing",
                    "is_virtual": True,
                    "source_image": "00000.png",
                    "render_image": "",
                }
            )
            + "\n",
            encoding="utf-8",
        )

        report = repair_candidate_views(
            renders_dir=renders_dir,
            candidate_views_path=candidates_path,
            output_dir=output_dir,
            model_mode="fixer",
            max_reprojection_error_px=2.5,
            max_photometric_drift=0.08,
            virtual_render_mapping_path=None,
        )
        assert report["accepted_count"] == 0
        assert report["rejected_count"] == 1
        assert "virtual_render_mapping_missing" in report["rows"][0]["gate_reasons"]

    def test_accepted_virtual_rows_preserve_metadata(self, tmp_path: Path) -> None:
        renders_dir = tmp_path / "renders"
        renders_dir.mkdir()
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        PILImage = pytest.importorskip("PIL.Image")
        img = np.full((32, 32, 3), 200, dtype=np.uint8)
        mapped_render = renders_dir / "00001.png"
        PILImage.fromarray(img).save(mapped_render)

        candidates_path = tmp_path / "candidates.jsonl"
        candidates_path.write_text(
            json.dumps(
                {
                    "id": "virtual_ok",
                    "is_virtual": True,
                    "source_image": "00001.png",
                    "render_image": "",
                    "qvec": [0.7071, 0.0, 0.7071, 0.0],
                    "tvec": [1.0, 2.0, 3.0],
                    "camera_id": 5,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        mapping_path = tmp_path / "mapping.jsonl"
        mapping_path.write_text(
            json.dumps(
                {
                    "candidate_id": "virtual_ok",
                    "render_image": str(mapped_render),
                    "qvec": [0.7071, 0.0, 0.7071, 0.0],
                    "tvec": [1.0, 2.0, 3.0],
                    "camera_id": 5,
                    "predicted_hole_ratio": 0.1,
                }
            )
            + "\n",
            encoding="utf-8",
        )

        def _fake_backend(*, input_path: Path, output_path: Path, **kwargs):  # type: ignore[no-untyped-def]
            del kwargs
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(input_path.read_bytes())
            return True, "fixer_native"

        with mock.patch("post_stage4_view_repair._run_backend", side_effect=_fake_backend):
            report = repair_candidate_views(
                renders_dir=renders_dir,
                candidate_views_path=candidates_path,
                output_dir=output_dir,
                model_mode="fixer",
                max_reprojection_error_px=2.5,
                max_photometric_drift=0.08,
                virtual_render_mapping_path=mapping_path,
            )

        assert report["accepted_count"] == 1
        row = report["rows"][0]
        assert row["accepted"] is True
        assert row["is_virtual"] is True
        assert row["candidate_id"] == "virtual_ok"
        assert row["camera_id"] == 5
        assert row["qvec"] == [0.7071, 0.0, 0.7071, 0.0]
        assert row["tvec"] == [1.0, 2.0, 3.0]

    def test_worldforge_plus_gsfix3d_fallback_on_rejection(self, tmp_path: Path) -> None:
        renders_dir = tmp_path / "renders"
        renders_dir.mkdir()
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        PILImage = pytest.importorskip("PIL.Image")
        img = np.full((64, 64, 3), 200, dtype=np.uint8)
        img[:32, :, :] = 0
        render_path = renders_dir / "frame_0001.png"
        PILImage.fromarray(img).save(render_path)

        candidates_path = tmp_path / "candidates.jsonl"
        candidates_path.write_text(
            json.dumps(
                {
                    "id": "wf_01",
                    "source_image": "frame_0001.png",
                    "render_image": str(render_path),
                    "cross_view_reprojection_error_px": 0.0,
                }
            )
            + "\n",
            encoding="utf-8",
        )

        def _fake_backend(*, backend: str, input_path: Path, output_path: Path, **kwargs):  # type: ignore[no-untyped-def]
            del kwargs
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if backend == "worldforge":
                PILImage.fromarray(np.zeros((64, 64, 3), dtype=np.uint8)).save(output_path)
                return True, "worldforge_native_longcat"
            if backend == "gsfix3d":
                shutil.copy2(input_path, output_path)
                return True, "command"
            raise AssertionError(f"unexpected backend {backend}")

        with mock.patch("post_stage4_view_repair._run_backend", side_effect=_fake_backend):
            report = repair_candidate_views(
                renders_dir=renders_dir,
                candidate_views_path=candidates_path,
                output_dir=output_dir,
                model_mode="worldforge+gsfix3d",
                max_reprojection_error_px=2.5,
                max_photometric_drift=0.08,
            )

        assert report["accepted_count"] == 1
        assert report["rejected_count"] == 0
        row = report["rows"][0]
        assert row["backend"] == "gsfix3d"
        assert row["backend_mode"] == "command"
        assert row["accepted"] is True


def test_parser_accepts_worldforge_modes() -> None:
    parser = _build_parser()
    args = parser.parse_args(
        [
            "--renders-dir",
            "renders",
            "--candidate-views",
            "candidates.jsonl",
            "--output-dir",
            "out",
            "--model",
            "worldforge+gsfix3d",
        ]
    )
    assert args.model == "worldforge+gsfix3d"
