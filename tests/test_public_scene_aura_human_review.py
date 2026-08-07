from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.public_scene_aura_human_review import (
    AuraHumanReviewError,
    materialize_aura_human_review,
)


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _file_record(path: Path, *, relative_to: Path) -> dict[str, object]:
    return {
        "relative_path": path.relative_to(relative_to).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    repo_root = tmp_path / "repo"
    data_root = tmp_path / "data"
    repo_root.mkdir()
    data_root.mkdir()

    aura_frames: list[dict[str, object]] = []
    artifacts: list[dict[str, object]] = []
    native_root = data_root / "native_review"
    native_root.mkdir()
    for index in range(8):
        camera_id = f"camera_{index}"
        source_bytes = f"source-frame-{index}".encode()
        source_sha = "sha256:" + hashlib.sha256(source_bytes).hexdigest()
        aura_frames.append({"camera_id": camera_id, "sha256": source_sha})
        row: dict[str, object] = {
            "camera_id": camera_id,
            "source_frame_sha256": source_sha,
        }
        for role in ("before", "after", "before_after", "contact_crop_before_after"):
            path = native_root / f"{camera_id}.{role}.png"
            path.write_bytes(f"{camera_id}:{role}".encode())
            row[role] = _file_record(path, relative_to=native_root)
        artifacts.append(row)

    aura: dict[str, object] = {
        "schema_version": "adp009b_aurafusion360_execution_receipt.v1",
        "status": "executed_candidate",
        "scene": {"publisher_scene_id": "840313", "target_instance_id": "ins160"},
        "claim_boundary": {"successful_inpainting_admitted": False},
        "execution": {"final_frames": aura_frames},
        "receipt_digest": "",
    }
    aura["receipt_digest"] = canonical_digest(aura, digest_field="receipt_digest")
    _write_json(repo_root / "aura.json", aura)

    locality: dict[str, object] = {
        "schema_version": "public_scene_inpainting_locality_measurement.v1",
        "status": "measured_no_admission_effect",
        "admission_effect": "none",
        "quality_pass_claimed": False,
        "thresholds_frozen_before_evaluation": False,
        "aggregate": {
            "view_count": 8,
            "mean_outside_mask_psnr_db": 39.3,
            "mean_outside_mask_windowed_ssim": 0.992,
            "mean_outside_mask_lpips": 0.014,
        },
        "locality_measurement_digest": "",
    }
    locality["locality_measurement_digest"] = canonical_digest(
        locality, digest_field="locality_measurement_digest"
    )
    _write_json(data_root / "locality.json", locality)

    native: dict[str, object] = {
        "schema_version": "adp009b_simready_native_visual_review_receipt.v1",
        "status": "rendered_native_visual_review_candidate",
        "renderer_is_native_ovrtx": True,
        "background_renderer": "aurafusion360_native_2d_gaussian_rasterizer",
        "human_visual_acceptance": "pending",
        "technical_admission": False,
        "artifacts": artifacts,
        "receipt_digest": "",
    }
    native["receipt_digest"] = canonical_digest(native, digest_field="receipt_digest")
    _write_json(data_root / "native.json", native)

    request: dict[str, object] = {
        "schema_version": "adp009b_aura_human_visual_review_request.v1",
        "reviewer_role": "project_owner",
        "decision": "accept_for_internal_hybrid_replacement_control",
        "approval_statement": "Good enough for the bounded internal hybrid control.",
        "aura_execution_receipt_path": "aura.json",
        "locality_measurement_path": "locality.json",
        "native_visual_review_receipt_path": "native.json",
        "native_visual_review_root": "native_review",
    }
    request_path = repo_root / "request.json"
    _write_json(request_path, request)
    return repo_root, data_root, request_path, repo_root / "receipt.json"


def test_materializes_human_acceptance_without_technical_admission(tmp_path: Path) -> None:
    repo_root, data_root, request_path, output_path = _fixture(tmp_path)

    receipt = materialize_aura_human_review(
        request_path=request_path,
        repo_root=repo_root,
        data_root=data_root,
        output_path=output_path,
    )

    assert receipt["status"] == "human_accepted_visual_candidate_for_internal_hybrid_control"
    assert receipt["technical_admission"] is False
    assert receipt["successful_inpainting_admitted"] is False
    assert receipt["hidden_background_truth_available"] is False
    assert len(receipt["bindings"]["review_files"]) == 32
    assert receipt["receipt_digest"] == canonical_digest(
        receipt, digest_field="receipt_digest"
    )


def test_rejects_changed_review_image(tmp_path: Path) -> None:
    repo_root, data_root, request_path, output_path = _fixture(tmp_path)
    (data_root / "native_review" / "camera_3.after.png").write_bytes(b"changed")

    with pytest.raises(AuraHumanReviewError, match="native_visual_review_file_changed"):
        materialize_aura_human_review(
            request_path=request_path,
            repo_root=repo_root,
            data_root=data_root,
            output_path=output_path,
        )


def test_rejects_caller_asserted_technical_admission(tmp_path: Path) -> None:
    repo_root, data_root, request_path, output_path = _fixture(tmp_path)
    request = json.loads(request_path.read_text(encoding="utf-8"))
    request["technical_admission"] = True
    _write_json(request_path, request)

    with pytest.raises(
        AuraHumanReviewError, match="caller_asserted_technical_admission_forbidden"
    ):
        materialize_aura_human_review(
            request_path=request_path,
            repo_root=repo_root,
            data_root=data_root,
            output_path=output_path,
        )


def test_rejects_broken_aura_to_native_frame_join(tmp_path: Path) -> None:
    repo_root, data_root, request_path, output_path = _fixture(tmp_path)
    native_path = data_root / "native.json"
    native = json.loads(native_path.read_text(encoding="utf-8"))
    native["artifacts"][0]["source_frame_sha256"] = "sha256:" + "0" * 64
    native["receipt_digest"] = canonical_digest(native, digest_field="receipt_digest")
    _write_json(native_path, native)

    with pytest.raises(AuraHumanReviewError, match="aura_native_camera_frame_join_invalid"):
        materialize_aura_human_review(
            request_path=request_path,
            repo_root=repo_root,
            data_root=data_root,
            output_path=output_path,
        )
