from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.heldout_appearance_evaluation_v2 import (
    HELDOUT_V2_REQUEST_SCHEMA_VERSION,
    HeldoutAppearanceV2Error,
    build_heldout_appearance_evaluation_request_v2,
    build_visual_heldout_evaluation_report_v2,
    evaluate_heldout_appearance_v2,
    windowed_ssim,
)


D = ["sha256:" + character * 64 for character in "abcdef"]
COMMIT = "9" * 40
LPIPS_AVAILABLE = importlib.util.find_spec("lpips") is not None
LPIPS_CHECKPOINT = "sha256:df73285e35b22355a2df87cdb6b70b343713b667eddbda73e1977e0c860835c0"


def _digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _image(rng: np.random.Generator) -> np.ndarray:
    base = rng.integers(0, 256, size=(48, 64, 3)).astype(np.float64)
    for _ in range(2):
        base = (
            base
            + np.roll(base, 1, 0)
            + np.roll(base, -1, 0)
            + np.roll(base, 1, 1)
            + np.roll(base, -1, 1)
        ) / 5.0
    return base


def test_windowed_ssim_properties() -> None:
    rng = np.random.default_rng(5)
    image = _image(rng) / 255.0
    assert windowed_ssim(image, image) == pytest.approx(1.0, abs=1e-9)
    noisy = np.clip(image + rng.normal(0, 0.08, size=image.shape), 0, 1)
    very_noisy = np.clip(image + rng.normal(0, 0.3, size=image.shape), 0, 1)
    score_noisy = windowed_ssim(image, noisy)
    score_very_noisy = windowed_ssim(image, very_noisy)
    assert 0.0 < score_very_noisy < score_noisy < 1.0
    assert windowed_ssim(noisy, image) == pytest.approx(score_noisy, abs=1e-12)
    with pytest.raises(HeldoutAppearanceV2Error, match="too_small"):
        windowed_ssim(image[:8, :8], image[:8, :8])


def _write_pairs(tmp_path: Path, *, degrade_short: float = 0.02) -> tuple[dict, Path, Path]:
    rng = np.random.default_rng(7)
    evaluator_root = tmp_path / "evaluator"
    candidate_root = tmp_path / "candidate"
    pairs = []
    for trajectory, count, noise in (
        ("author_heldout", 3, 0.02),
        ("independent_short", 3, degrade_short),
    ):
        for index in range(count):
            real = _image(rng)
            render = np.clip(real + rng.normal(0, noise * 255, size=real.shape), 0, 255)
            view_id = f"{trajectory}-{index:03d}"
            real_rel = f"hidden/{view_id}.png"
            render_rel = f"renders/{view_id}.png"
            real_path = evaluator_root / real_rel
            render_path = candidate_root / render_rel
            real_path.parent.mkdir(parents=True, exist_ok=True)
            render_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(real.astype(np.uint8)).save(real_path)
            Image.fromarray(render.astype(np.uint8)).save(render_path)
            pairs.append(
                {
                    "view_id": view_id,
                    "trajectory": trajectory,
                    "split": "held_out",
                    "excluded_from_training": True,
                    "real_view_relative_path": real_rel,
                    "real_view_digest": _digest(real_path),
                    "candidate_render_relative_path": render_rel,
                    "candidate_render_digest": _digest(render_path),
                }
            )
    request = {
        "schema_version": HELDOUT_V2_REQUEST_SCHEMA_VERSION,
        "stable_run_identity": "heldout-v2-fixture",
        "source_capture_identity": "fixture-capture",
        "source_capture_digest": D[0],
        "reconstruction_dataset_digest": D[1],
        "frozen_split_digest": D[2],
        "candidate_reconstruction_result_digest": D[3],
        "evaluator_implementation_digest": D[4],
        "source_commit_sha": COMMIT,
        "candidate_method_id": "teleport_modelv3",
        "candidate_provider_identity": "teleport",
        "evaluator_identity": "blueprint_heldout_evaluator_v2",
        "evaluator_provider_identity": "blueprint_local",
        "candidate_root": str(candidate_root),
        "evaluator_root": str(evaluator_root),
        "coordinate_frame_declaration": {"declaration": "fixture"},
        "authority_used": {"local_processing_authorized": True},
        "split_frozen_before_training": True,
        "thresholds_frozen_before_evaluation": True,
        "candidate_had_hidden_access": False,
        "candidate_selected_heldout": False,
        "candidate_self_grading": False,
        "lpips_required": False,
        "lpips_model": None,
        "thresholds": {
            "minimum_mean_psnr_db": 20.0,
            "minimum_mean_global_ssim": 0.5,
            "minimum_mean_windowed_ssim": 0.5,
            "maximum_mean_absolute_error": 0.1,
            "maximum_mean_lpips": None,
        },
        "pairs": pairs,
        "timestamp": "2026-08-01T00:00:00Z",
    }
    return request, evaluator_root, candidate_root


def test_evaluation_reports_trajectories_separately_and_replays(tmp_path: Path) -> None:
    request, _evaluator_root, _candidate_root = _write_pairs(tmp_path)
    report = evaluate_heldout_appearance_v2(source_artifact=request, output_root=tmp_path)
    assert report["status"] == "passed_appearance_only"
    assert report["by_trajectory"]["author_heldout"]["view_count"] == 3
    assert report["by_trajectory"]["independent_short"]["view_count"] == 3
    assert report["by_trajectory"]["author_heldout"]["thresholds_passed"] is True
    assert report["measured_trajectories"] == ["author_heldout", "independent_short"]
    assert report["metric_definitions"]["windowed_ssim"].startswith("wang2004")
    assert build_visual_heldout_evaluation_report_v2(report) == report

    tampered = dict(report)
    tampered["status"] = "passed_appearance_only"
    tampered["by_trajectory"] = dict(report["by_trajectory"])
    weakened = dict(tampered["by_trajectory"]["independent_short"])
    weakened["mean_absolute_error"] = 0.0
    tampered["by_trajectory"]["independent_short"] = weakened
    tampered["visual_heldout_evaluation_report_digest"] = canonical_digest(
        {k: v for k, v in tampered.items() if k != "visual_heldout_evaluation_report_digest"},
        digest_field="visual_heldout_evaluation_report_digest",
    )
    with pytest.raises(HeldoutAppearanceV2Error, match="recomputation_mismatch"):
        build_visual_heldout_evaluation_report_v2(tampered)


def test_one_bad_trajectory_rejects_without_averaging(tmp_path: Path) -> None:
    request, _evaluator_root, _candidate_root = _write_pairs(tmp_path, degrade_short=0.6)
    report = evaluate_heldout_appearance_v2(source_artifact=request, output_root=tmp_path)
    assert report["status"] == "rejected_appearance_quality"
    assert report["by_trajectory"]["author_heldout"]["thresholds_passed"] is True
    assert report["by_trajectory"]["independent_short"]["thresholds_passed"] is False
    assert report["blockers"] == ["heldout_appearance_thresholds_not_met"]


def test_digest_mismatch_and_blank_render_fail_closed(tmp_path: Path) -> None:
    request, _evaluator_root, candidate_root = _write_pairs(tmp_path)
    request["pairs"][0]["candidate_render_digest"] = "sha256:" + "f" * 64
    with pytest.raises(HeldoutAppearanceV2Error, match="candidate_render_digest_mismatch"):
        evaluate_heldout_appearance_v2(source_artifact=request, output_root=tmp_path)

    request2, _evaluator_root2, candidate_root2 = _write_pairs(tmp_path / "blank")
    blank_rel = request2["pairs"][0]["candidate_render_relative_path"]
    blank_path = candidate_root2 / blank_rel
    Image.new("RGB", (64, 48), (7, 7, 7)).save(blank_path)
    request2["pairs"][0]["candidate_render_digest"] = _digest(blank_path)
    with pytest.raises(HeldoutAppearanceV2Error, match="candidate_render_blank"):
        evaluate_heldout_appearance_v2(source_artifact=request2, output_root=tmp_path)

    request3, _e, _c = _write_pairs(tmp_path / "same-provider")
    request3["evaluator_provider_identity"] = "teleport"
    with pytest.raises(HeldoutAppearanceV2Error, match="evaluator_not_independent"):
        build_heldout_appearance_evaluation_request_v2(request3)


@pytest.mark.skipif(not LPIPS_AVAILABLE, reason="lpips runtime not installed")
def test_lpips_lane_pins_checkpoint_and_records_values(tmp_path: Path) -> None:
    request, _evaluator_root, _candidate_root = _write_pairs(tmp_path)
    request["lpips_required"] = True
    request["lpips_model"] = {
        "model_id": "lpips_alex_v0.1",
        "checkpoint_digest": LPIPS_CHECKPOINT,
    }
    request["thresholds"]["maximum_mean_lpips"] = 0.5
    report = evaluate_heldout_appearance_v2(source_artifact=request, output_root=tmp_path)
    assert report["lpips_runtime"]["checkpoint_digest"] == LPIPS_CHECKPOINT
    assert all(0.0 <= row["lpips"] <= 1.5 for row in report["rows"])
    assert report["by_trajectory"]["author_heldout"]["mean_lpips"] is not None
    assert build_visual_heldout_evaluation_report_v2(report) == report

    request_bad = dict(request)
    request_bad["lpips_model"] = {
        "model_id": "lpips_alex_v0.1",
        "checkpoint_digest": "sha256:" + "0" * 64,
    }
    with pytest.raises(HeldoutAppearanceV2Error, match="checkpoint_digest_mismatch"):
        evaluate_heldout_appearance_v2(source_artifact=request_bad, output_root=tmp_path)
