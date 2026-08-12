from __future__ import annotations

import json
from pathlib import Path

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.public_scene_aura_exact_residual_visual_abstention import (
    AuraExactResidualVisualAbstentionError,
    materialize_aura_exact_residual_visual_abstention,
)


def _write(path: Path, value: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _sha(path: Path) -> str:
    import hashlib

    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _relative(path: Path, root: Path) -> dict[str, object]:
    return {
        "relative_path": path.relative_to(root).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": _sha(path),
    }


def _absolute(path: Path) -> dict[str, object]:
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": _sha(path)}


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    composite_root = tmp_path / "composite"
    native_root = tmp_path / "native"
    frames = []
    manifests = []
    for task_id, camera_id in (("task_a", "camera_a"), ("task_b", "camera_b")):
        task_root = composite_root / task_id
        before = task_root / "before" / f"{camera_id}.png"
        mask = task_root / "masks" / f"{camera_id}.png"
        after = task_root / "frames" / f"{camera_id}.png"
        native = native_root / task_id / f"{camera_id}.png"
        for path, content in (
            (before, b"before"),
            (mask, b"mask"),
            (after, b"after"),
            (native, b"native"),
        ):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        frames.append(
            {
                "task_id": task_id,
                "camera_id": camera_id,
                "retained_scene_before": _relative(before, task_root),
                "exact_residual_mask": _relative(mask, task_root),
                "native_aura_frame": _absolute(native),
                "exact_mask_composited_frame": _relative(after, task_root),
            }
        )
        manifests.append({"task_id": task_id})
    composite = {
        "schema_version": "public_scene_aura_exact_residual_composite.v1",
        "status": "exact_mask_composites_materialized_unqualified",
        "replacement_object_count": 2,
        "outside_mask_changed_pixels_total": 0,
        "multi_view_consistency_required": True,
        "multi_view_consistency_measurement": {
            "visual_semantic_consistency_passed": False,
            "tasks": [
                {
                    "task_id": task_id,
                    "exact_camera_count": 1,
                    "exact_camera_ids": [camera_id],
                    "all_raw_frames_bind_same_native_aura_point_cloud": True,
                    "visual_semantic_consistency_passed": False,
                }
                for task_id, camera_id in (
                    ("task_a", "camera_a"),
                    ("task_b", "camera_b"),
                )
            ],
        },
        "claim_boundary": {"inpainting_result_qualified": False},
        "provider_closeout": {
            "actual_cost_usd": 0.25,
            "destroyed_vast_instance_ids": [123],
        },
        "frames": frames,
        "task_render_manifests": manifests,
        "composite_digest": "",
    }
    composite["composite_digest"] = canonical_digest(
        composite, digest_field="composite_digest"
    )
    composite_path = _write(
        composite_root / "public_scene_aura_exact_residual_composite.v1.json",
        composite,
    )
    request = {
        "schema_version": "public_scene_aura_exact_residual_visual_abstention_request.v1",
        "decision": "reject_or_block_shared_scene_visual_candidate",
        "reviewer_role": "evidence_operator",
        "learned_policy_outcomes_accessed": False,
        "findings": [
            {
                "task_id": "task_a",
                "disposition": "blocked_by_shared_scene_dependency",
                "observed_artifact_codes": ["shared_scene_dependency_failed"],
                "reviewed_camera_ids": ["camera_a"],
            },
            {
                "task_id": "task_b",
                "disposition": "reject_visual_candidate",
                "observed_artifact_codes": [
                    "semantic_hallucination_in_exact_residual",
                    "multiview_background_inconsistency",
                ],
                "reviewed_camera_ids": ["camera_b"],
            },
        ],
    }
    return composite_path, _write(tmp_path / "request.json", request)


def test_seals_reject_only_two_task_visual_abstention(tmp_path: Path) -> None:
    composite, request = _fixture(tmp_path)
    receipt = materialize_aura_exact_residual_visual_abstention(
        request_path=request,
        composite_receipt_path=composite,
        output_path=tmp_path / "abstention.json",
    )

    assert receipt["status"] == "typed_visual_abstention_before_native_simulator"
    assert receipt["replacement_object_count"] == 2
    assert receipt["controls_executed"] is False
    assert receipt["learned_candidate_episodes_executed"] is False
    assert receipt["automatic_paid_retry_executed"] is False
    assert len(receipt["bindings"]["reviewed_frames"]) == 2
    assert receipt["receipt_digest"] == canonical_digest(
        receipt, digest_field="receipt_digest"
    )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda request: request.update({"decision": "accept_visual_candidate"}),
        lambda request: request["findings"][1].update(
            {"disposition": "blocked_by_shared_scene_dependency"}
        ),
        lambda request: request["findings"].pop(),
    ],
)
def test_rejects_acceptance_or_incomplete_dispositions(tmp_path: Path, mutation) -> None:
    composite, request_path = _fixture(tmp_path)
    request = json.loads(request_path.read_text())
    mutation(request)
    _write(request_path, request)
    with pytest.raises(AuraExactResidualVisualAbstentionError):
        materialize_aura_exact_residual_visual_abstention(
            request_path=request_path,
            composite_receipt_path=composite,
            output_path=tmp_path / "abstention.json",
        )


def test_rejects_changed_reviewed_frame(tmp_path: Path) -> None:
    composite, request = _fixture(tmp_path)
    (composite.parent / "task_b/frames/camera_b.png").write_bytes(b"changed")
    with pytest.raises(
        AuraExactResidualVisualAbstentionError,
        match="visual_composite_frame_changed",
    ):
        materialize_aura_exact_residual_visual_abstention(
            request_path=request,
            composite_receipt_path=composite,
            output_path=tmp_path / "abstention.json",
        )


def test_rejects_incomplete_exact_camera_set(tmp_path: Path) -> None:
    composite_path, request = _fixture(tmp_path)
    composite = json.loads(composite_path.read_text())
    composite["frames"].pop()
    composite["composite_digest"] = canonical_digest(
        composite, digest_field="composite_digest"
    )
    _write(composite_path, composite)
    with pytest.raises(
        AuraExactResidualVisualAbstentionError,
        match="visual_frame_set_incomplete",
    ):
        materialize_aura_exact_residual_visual_abstention(
            request_path=request,
            composite_receipt_path=composite_path,
            output_path=tmp_path / "abstention.json",
        )
