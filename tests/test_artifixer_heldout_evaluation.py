from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from blueprint_pipeline.artifixer_heldout_evaluation import evaluate_artifixer_heldout_views


def _image(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((16, 16, 3), value, dtype=np.uint8)).save(path)


def _manifest(path: Path, *, view_id: str, real: Path, generated: Path, heldout: bool) -> None:
    path.write_text(
        json.dumps(
            {
                "frozen": True,
                "training_view_ids": [] if heldout else [view_id],
                "pairs": [
                    {
                        "view_id": view_id,
                        "real_view_path": str(real),
                        "generated_view_path": str(generated),
                        "excluded_from_training": heldout,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def test_artifixer_evaluation_requires_real_heldout_views_and_keeps_claim_boundary(
    tmp_path: Path,
) -> None:
    real = tmp_path / "capture" / "heldout.png"
    generated = tmp_path / "generated" / "heldout.png"
    _image(real, 100)
    _image(generated, 100)
    manifest = tmp_path / "manifest.json"
    _manifest(manifest, view_id="view-1", real=real, generated=generated, heldout=True)
    result = evaluate_artifixer_heldout_views(
        manifest_path=manifest,
        generated_root=generated.parent,
        output_path=tmp_path / "result.json",
    )
    assert result["status"] == "passed_advisory"
    assert result["aggregate"]["thresholds_passed"] is True
    assert result["claim_boundary"]["generated_pixels_are_capture_truth"] is False
    assert result["claim_boundary"]["generated_geometry_is_collision_truth"] is False

    _manifest(manifest, view_id="view-1", real=real, generated=generated, heldout=False)
    leaked = evaluate_artifixer_heldout_views(
        manifest_path=manifest,
        generated_root=generated.parent,
        output_path=tmp_path / "leaked.json",
    )
    assert leaked["status"] == "blocked_or_failed"
    assert "artifixer_view_not_proven_held_out:view-1" in leaked["blockers"]


def test_artifixer_evaluation_fails_bad_generated_view(tmp_path: Path) -> None:
    real = tmp_path / "capture" / "heldout.png"
    generated = tmp_path / "generated" / "heldout.png"
    _image(real, 0)
    _image(generated, 255)
    manifest = tmp_path / "manifest.json"
    _manifest(manifest, view_id="view-2", real=real, generated=generated, heldout=True)
    result = evaluate_artifixer_heldout_views(
        manifest_path=manifest,
        generated_root=generated.parent,
        output_path=tmp_path / "result.json",
    )
    assert result["status"] == "blocked_or_failed"
    assert "artifixer_heldout_real_view_thresholds_not_met" in result["blockers"]
