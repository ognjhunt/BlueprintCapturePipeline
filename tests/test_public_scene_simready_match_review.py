from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.public_scene_simready_match_review import (
    SimReadyMatchReviewError,
    materialize_match_review,
)


def _sha(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _write_receipt(path: Path, value: dict[str, object]) -> None:
    value["receipt_digest"] = canonical_digest(value, digest_field="receipt_digest")
    path.write_text(json.dumps(value), encoding="utf-8")


def _fixture(tmp_path: Path, *, shifted: bool = False, dark: bool = False) -> dict[str, Path]:
    evidence = tmp_path / "evidence"
    edit_root = evidence / "edit"
    visual_root = evidence / "visual"
    output = evidence / "match"
    (edit_root / "images").mkdir(parents=True)
    (edit_root / "target_support").mkdir()
    visual_root.mkdir()

    original = np.full((120, 160, 3), 238, dtype=np.uint8)
    original[35:95, 65:95] = (165, 213, 121)
    original_path = edit_root / "images" / "view.png"
    cv2.imwrite(str(original_path), original)

    support = np.zeros((120, 160, 4), dtype=np.uint8)
    support[:, :, 3] = 255
    support[35:95, 65:95, :3] = 255
    support_path = edit_root / "target_support" / "view.png"
    cv2.imwrite(str(support_path), support)

    replacement = np.full((120, 160, 3), 238, dtype=np.uint8)
    replacement_mask = np.zeros((120, 160), dtype=np.uint8)
    x0 = 78 if shifted else 65
    replacement[x0 - 30 : x0 + 30, 65:95] = (70, 162, 19) if dark else (165, 213, 121)
    replacement_mask[x0 - 30 : x0 + 30, 65:95] = 255
    replacement_path = visual_root / "view.after.png"
    replacement_mask_path = visual_root / "view.replacement_mask.png"
    cv2.imwrite(str(replacement_path), replacement)
    cv2.imwrite(str(replacement_mask_path), replacement_mask)

    edit_receipt_path = edit_root / "edit.json"
    _write_receipt(
        edit_receipt_path,
        {
            "status": "render_derived_input_packet_materialized",
            "derived_artifacts": {
                "images": [
                    {
                        "camera_id": "view",
                        "relative_path": "images/view.png",
                        "sha256": _sha(original_path),
                        "size_bytes": original_path.stat().st_size,
                    }
                ]
            },
        },
    )
    visual_receipt_path = visual_root / "visual.json"
    _write_receipt(
        visual_receipt_path,
        {
            "status": "rendered_visual_review_candidate",
            "artifacts": [
                {
                    "camera_id": "view",
                    "source_frame_sha256": _sha(original_path),
                    "after": {
                        "relative_path": "view.after.png",
                        "sha256": _sha(replacement_path),
                        "size_bytes": replacement_path.stat().st_size,
                    },
                    "replacement_mask": {
                        "relative_path": "view.replacement_mask.png",
                        "sha256": _sha(replacement_mask_path),
                        "size_bytes": replacement_mask_path.stat().st_size,
                    },
                }
            ],
        },
    )
    return {
        "evidence": evidence,
        "edit_root": edit_root,
        "visual_root": visual_root,
        "edit_receipt": edit_receipt_path,
        "visual_receipt": visual_receipt_path,
        "original": original_path,
        "output": output,
    }


def _run(paths: dict[str, Path]) -> dict[str, object]:
    return materialize_match_review(
        edit_input_receipt_path=paths["edit_receipt"],
        edit_input_root=paths["edit_root"],
        visual_review_receipt_path=paths["visual_receipt"],
        visual_review_root=paths["visual_root"],
        evidence_root=paths["evidence"],
        output_root=paths["output"],
    )


def test_match_review_derives_scale_and_colour_from_observed_pixels(tmp_path: Path) -> None:
    receipt = _run(_fixture(tmp_path))

    assert receipt["status"] == "diagnosed_match_candidate"
    assert receipt["aggregate"]["projected_scale_and_pose_gate_passed"] is True
    assert receipt["aggregate"]["colour_appearance_gate_passed"] is True
    assert receipt["aggregate"]["median_silhouette_iou"] == pytest.approx(1.0)
    assert receipt["human_multiview_identity_review"] == "pending"
    assert "human_multiview_identity_review_pending" in receipt["blockers"]


def test_match_review_reports_shifted_dark_replacement_without_caller_override(
    tmp_path: Path,
) -> None:
    receipt = _run(_fixture(tmp_path, shifted=True, dark=True))

    assert receipt["status"] == "diagnosed_mismatch"
    assert receipt["aggregate"]["projected_scale_and_pose_gate_passed"] is False
    assert receipt["aggregate"]["colour_appearance_gate_passed"] is False
    assert "replacement_multiview_silhouette_match_below_threshold" in receipt["blockers"]
    assert "replacement_color_material_match_below_threshold" in receipt["blockers"]


def test_match_review_rejects_changed_source_bytes(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    paths["original"].write_bytes(b"changed")

    with pytest.raises(SimReadyMatchReviewError, match="original_frame_digest_mismatch:view"):
        _run(paths)


def test_match_review_rejects_caller_asserted_acceptance(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    visual = json.loads(paths["visual_receipt"].read_text(encoding="utf-8"))
    visual.pop("receipt_digest")
    visual["match_accepted"] = True
    _write_receipt(paths["visual_receipt"], visual)

    with pytest.raises(SimReadyMatchReviewError, match="caller_asserted_acceptance_forbidden"):
        _run(paths)
