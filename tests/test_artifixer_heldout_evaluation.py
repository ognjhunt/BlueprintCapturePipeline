from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
from jsonschema import Draft202012Validator

from blueprint_pipeline.artifixer_heldout_evaluation import (
    MANIFEST_SCHEMA,
    evaluate_artifixer_heldout_views,
)
from blueprint_pipeline.common import sha256_file
from blueprint_pipeline.decision_evidence_contracts import canonical_digest


def _image(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((16, 16, 3), value, dtype=np.uint8)).save(path)


def _manifest(
    path: Path,
    *,
    heldout_root: Path,
    baseline_root: Path,
    generated_root: Path,
    heldout: bool = True,
    baseline_value: int = 80,
    generated_value: int = 100,
) -> None:
    pairs = []
    for index in range(3):
        filename = f"view-{index}.png"
        real = heldout_root / filename
        baseline = baseline_root / filename
        generated = generated_root / filename
        _image(real, 100)
        _image(baseline, baseline_value)
        _image(generated, generated_value)
        pairs.append(
            {
                "view_id": f"view-{index}",
                "real_view_reference": filename,
                "real_view_sha256": sha256_file(real),
                "baseline_view_reference": filename,
                "baseline_view_sha256": sha256_file(baseline),
                "generated_view_reference": filename,
                "generated_view_sha256": sha256_file(generated),
                "excluded_from_candidate_training": heldout,
            }
        )
    value = {
        "schema_version": MANIFEST_SCHEMA,
        "frozen": True,
        "frozen_before_candidate_execution": True,
        "source_capture_digest": "sha256:" + "1" * 64,
        "frozen_split_digest": "sha256:" + "2" * 64,
        "baseline_reconstruction_digest": "sha256:" + "3" * 64,
        "enhancement_method_audit_digest": "sha256:" + "4" * 64,
        "training_view_ids": [] if heldout else ["view-0"],
        "pairs": pairs,
        "timestamp": "2026-08-01T12:00:00Z",
    }
    value["manifest_digest"] = canonical_digest(value, digest_field="manifest_digest")
    path.write_text(json.dumps(value), encoding="utf-8")


def _evaluate(tmp_path: Path, **manifest_kwargs) -> dict:
    heldout_root = tmp_path / "heldout"
    baseline_root = tmp_path / "baseline"
    generated_root = tmp_path / "generated"
    manifest = tmp_path / "manifest.json"
    _manifest(
        manifest,
        heldout_root=heldout_root,
        baseline_root=baseline_root,
        generated_root=generated_root,
        **manifest_kwargs,
    )
    return evaluate_artifixer_heldout_views(
        manifest_path=manifest,
        heldout_root=heldout_root,
        baseline_root=baseline_root,
        generated_root=generated_root,
        output_path=tmp_path / "result.json",
    )


def test_artifixer_evaluation_requires_and_measures_baseline_improvement(
    tmp_path: Path,
) -> None:
    result = _evaluate(tmp_path)

    assert result["status"] == "passed_advisory"
    assert result["heldout_view_count"] == 3
    assert result["aggregate"]["thresholds_passed"] is True
    assert result["claim_boundary"]["baseline_improvement_established"] is True
    assert result["claim_boundary"]["generated_pixels_are_capture_truth"] is False
    assert result["claim_boundary"]["generated_geometry_is_collision_truth"] is False
    assert result["claim_boundary"]["metric_or_collision_qualification_changed"] is False
    assert result["result_digest"] == canonical_digest(result, digest_field="result_digest")
    schema = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "docs/schemas/artifixer_heldout_evaluation.v2.schema.json"
        ).read_text(encoding="utf-8")
    )
    Draft202012Validator(schema).validate(result)


def test_artifixer_evaluation_rejects_training_leakage(tmp_path: Path) -> None:
    result = _evaluate(tmp_path, heldout=False)

    assert result["status"] == "blocked_or_failed"
    assert "artifixer_view_not_proven_held_out:view-0" in result["blockers"]


def test_artifixer_evaluation_rejects_worse_generated_views(tmp_path: Path) -> None:
    result = _evaluate(tmp_path, generated_value=0)

    assert result["status"] == "blocked_or_failed"
    assert "artifixer_heldout_baseline_improvement_not_established" in result["blockers"]


def test_artifixer_evaluation_serializes_perfect_baseline_regression(tmp_path: Path) -> None:
    result = _evaluate(tmp_path, baseline_value=100, generated_value=90)

    assert result["status"] == "blocked_or_failed"
    assert result["rows"][0]["psnr_improvement_db"] == "negative_infinity"
    json.dumps(result, allow_nan=False)


def test_artifixer_evaluation_rejects_real_view_digest_tamper(tmp_path: Path) -> None:
    heldout_root = tmp_path / "heldout"
    baseline_root = tmp_path / "baseline"
    generated_root = tmp_path / "generated"
    manifest = tmp_path / "manifest.json"
    _manifest(
        manifest,
        heldout_root=heldout_root,
        baseline_root=baseline_root,
        generated_root=generated_root,
    )
    _image(heldout_root / "view-0.png", 101)

    result = evaluate_artifixer_heldout_views(
        manifest_path=manifest,
        heldout_root=heldout_root,
        baseline_root=baseline_root,
        generated_root=generated_root,
        output_path=tmp_path / "result.json",
    )
    assert "artifixer_real_view:view-0_digest_mismatch" in result["blockers"]


def test_artifixer_evaluation_rejects_manifest_path_escape(tmp_path: Path) -> None:
    heldout_root = tmp_path / "heldout"
    baseline_root = tmp_path / "baseline"
    generated_root = tmp_path / "generated"
    manifest = tmp_path / "manifest.json"
    _manifest(
        manifest,
        heldout_root=heldout_root,
        baseline_root=baseline_root,
        generated_root=generated_root,
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["pairs"][0]["real_view_reference"] = "../outside.png"
    payload["manifest_digest"] = canonical_digest(payload, digest_field="manifest_digest")
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    result = evaluate_artifixer_heldout_views(
        manifest_path=manifest,
        heldout_root=heldout_root,
        baseline_root=baseline_root,
        generated_root=generated_root,
        output_path=tmp_path / "result.json",
    )
    assert "artifixer_real_view:view-0_reference_unsafe" in result["blockers"]
