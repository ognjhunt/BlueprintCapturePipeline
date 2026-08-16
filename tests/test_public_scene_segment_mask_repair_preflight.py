from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.gaussian_splat_decode import write_standard_3dgs_ply_subset_exact
from blueprint_pipeline.public_scene_artifixer3d_candidate_inputs import (
    materialize_artifixer3d_candidate_inputs,
)
from blueprint_pipeline.public_scene_segment_mask_repair_preflight import (
    SegmentMaskRepairPreflightError,
    materialize_segment_mask_repair_preflight,
)
from tests.test_adp_retained_scene_render_packet import (
    _absolute_record,
    _authority,
    _relative_record,
    _source_ply,
    _task_freeze,
    _write_json,
)


def _fixture(
    tmp_path: Path,
    *,
    task_count: int = 2,
    relative_mask_records: bool = False,
    source_anchored_mask_records: bool = False,
) -> tuple[Path, Path, list[Path]]:
    root = tmp_path / "segment_set"
    source = _source_ply(root / "source.ply")
    shared = root / "shared_scene_union"
    shared.mkdir(parents=True)
    deleted_indices = np.arange(task_count, dtype=np.int64)
    retained_indices = np.setdiff1d(
        np.arange(10, dtype=np.int64), deleted_indices, assume_unique=True
    )
    retained = write_standard_3dgs_ply_subset_exact(
        source, shared / "retained_scene_gaussians.ply", retained_indices
    )
    tasks: list[dict[str, object]] = []
    mask_paths: list[Path] = []
    for slot in range(1, task_count + 1):
        task = _task_freeze(f"task_{slot}", slot)
        freeze_path = tmp_path / "freezes" / f"task_{slot}.json"
        _write_json(freeze_path, task)
        camera_root = tmp_path / "task_inputs" / f"task_{slot}"
        camera_rows: list[dict[str, object]] = []
        masks: list[dict[str, object]] = []
        source_images: list[dict[str, object]] = []
        mask_record_root = (
            tmp_path / "task_inputs" / f"task_{slot}_excision"
            if source_anchored_mask_records
            else camera_root
        )
        for camera_slot in range(2):
            camera_id = f"camera_{camera_slot}"
            camera_rows.append(
                {
                    "camera_id": camera_id,
                    "T_world_camera_provider_frame": [
                        [1.0, 0.0, 0.0, float(camera_slot)],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ],
                    "intrinsics": {
                        "model": "PINHOLE",
                        "fx": 4.0,
                        "fy": 4.0,
                        "cx": 2.0,
                        "cy": 2.0,
                        "width": 4,
                        "height": 4,
                    },
                }
            )
            image_path = camera_root / "images" / f"{camera_id}.png"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(
                np.full((4, 4, 3), 30 + slot + camera_slot, dtype=np.uint8),
                mode="RGB",
            ).save(image_path)
            mask_pixels = np.zeros((4, 4), dtype=np.uint8)
            mask_pixels[1:3, 1:3] = 255
            mask_path = mask_record_root / "masks" / f"{camera_id}.png"
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(mask_pixels, mode="L").save(mask_path)
            mask_paths.append(mask_path)
            masks.append(
                {
                    "camera_id": camera_id,
                    "historical_outer_mask": (
                        _relative_record(mask_record_root, mask_path)
                        if relative_mask_records or source_anchored_mask_records
                        else _absolute_record(mask_path)
                    ),
                }
            )
            source_images.append({"camera_id": camera_id, **_absolute_record(image_path)})
        camera_path = camera_root / "cameras.json"
        camera_path.write_text(json.dumps(camera_rows) + "\n", encoding="utf-8")
        contribution: dict[str, object] | None = None
        if source_anchored_mask_records:
            source_freeze: dict[str, object] = {
                "scene": {"publisher_scene_id": "840920", "task_id": f"task_{slot}"},
                "freeze_digest": "",
            }
            source_freeze["freeze_digest"] = canonical_digest(
                source_freeze, digest_field="freeze_digest"
            )
            source_freeze_path = mask_record_root / "source_freeze.json"
            _write_json(source_freeze_path, source_freeze)
            contribution = {
                "source_excision_freeze": {
                    **_absolute_record(source_freeze_path),
                    "freeze_digest": source_freeze["freeze_digest"],
                }
            }
        sweep: dict[str, object] = {
            "scene": {"publisher_scene_id": "840920", "task_id": f"task_{slot}"},
            "camera_contract": _absolute_record(camera_path),
            "masks": masks,
            "source_images": source_images,
            "learned_policy_outcomes_observed": False,
            "replacement_usd_inserted": False,
            "freeze_digest": "",
        }
        if contribution is not None:
            sweep["segment_contribution_sweep"] = contribution
        sweep["freeze_digest"] = canonical_digest(sweep, digest_field="freeze_digest")
        sweep_path = camera_root / "sweep.json"
        _write_json(sweep_path, sweep)
        tasks.append(
            {
                "task_id": f"task_{slot}",
                "task_freeze_digest": task["task_freeze_digest"],
                "task_freeze": _absolute_record(freeze_path),
                "sweep_freeze": {
                    **_absolute_record(sweep_path),
                    "freeze_digest": sweep["freeze_digest"],
                },
            }
        )
    candidate: dict[str, object] = {
        "schema_version": "adp009d_segment_contribution_cutout_set.v1",
        "task_candidates": tasks,
        "shared_scene_union": {
            "counts": {
                "source": 10,
                "deleted_total": task_count,
                "retained_total": 10 - task_count,
            },
            "outputs": {"retained_scene_gaussians": _relative_record(root, retained)},
        },
        "receipt_digest": "",
    }
    candidate["receipt_digest"] = canonical_digest(candidate, digest_field="receipt_digest")
    candidate_path = root / "candidate.json"
    _write_json(candidate_path, candidate)
    return candidate_path, _authority(tmp_path / "authority.json"), mask_paths


def test_segment_masks_are_the_only_artifixer_generated_support(tmp_path: Path) -> None:
    candidate, authority, _masks = _fixture(tmp_path, task_count=2)
    preflight_path = tmp_path / "preflight.json"

    preflight = materialize_segment_mask_repair_preflight(
        segment_cutout_set_path=candidate,
        execution_authority_path=authority,
        output_path=preflight_path,
    )
    inputs = materialize_artifixer3d_candidate_inputs(
        calibrated_residual_preflight_path=preflight_path,
        output_root=tmp_path / "artifixer_inputs",
    )

    assert preflight["repair_authority"] == {
        "generated_pixel_support": "historical_outer_object_segment_exact_binary_mask",
        "mask_dilation_pixels": 0,
        "full_deleted_gaussian_projection_is_diagnostic_only": True,
        "full_deleted_gaussian_projection_is_generated_edit_support": False,
        "canonical_source_frame_is_outside_mask_authority": True,
    }
    assert len(preflight["camera_inputs"]) == 4
    assert all(row["exact_residual_mask"]["pixel_count"] == 4 for row in preflight["camera_inputs"])
    for task in inputs["tasks"]:
        assert task["repair_support_coverage"] == {
            "minimum_fraction": 0.25,
            "mean_fraction": 0.25,
            "maximum_fraction": 0.25,
            "interpretation": (
                "pre_execution_large_hole_risk_metric_not_method_quality_or_qualification_verdict"
            ),
        }
        assert all(frame["outside_support_changed_pixels"] == 0 for frame in task["frames"])


def test_segment_masks_accept_digest_bound_paths_relative_to_sweep(tmp_path: Path) -> None:
    candidate, authority, _masks = _fixture(
        tmp_path, task_count=2, relative_mask_records=True
    )

    preflight = materialize_segment_mask_repair_preflight(
        segment_cutout_set_path=candidate,
        execution_authority_path=authority,
        output_path=tmp_path / "preflight.json",
    )

    assert len(preflight["camera_inputs"]) == 4
    assert all(row["exact_residual_mask"]["pixel_count"] == 4 for row in preflight["camera_inputs"])


def test_segment_masks_use_digest_bound_source_excision_freeze_as_relative_root(
    tmp_path: Path,
) -> None:
    candidate, authority, _masks = _fixture(
        tmp_path, task_count=2, source_anchored_mask_records=True
    )

    preflight = materialize_segment_mask_repair_preflight(
        segment_cutout_set_path=candidate,
        execution_authority_path=authority,
        output_path=tmp_path / "preflight.json",
    )

    assert len(preflight["camera_inputs"]) == 4
    assert all(row["exact_residual_mask"]["pixel_count"] == 4 for row in preflight["camera_inputs"])


def test_segment_mask_digest_tamper_is_rejected(tmp_path: Path) -> None:
    candidate, authority, masks = _fixture(tmp_path, task_count=1)
    Image.fromarray(np.full((4, 4), 255, dtype=np.uint8), mode="L").save(masks[0])

    with pytest.raises(
        SegmentMaskRepairPreflightError,
        match="segment_repair_exact_segment_mask_invalid",
    ):
        materialize_segment_mask_repair_preflight(
            segment_cutout_set_path=candidate,
            execution_authority_path=authority,
            output_path=tmp_path / "preflight.json",
        )
