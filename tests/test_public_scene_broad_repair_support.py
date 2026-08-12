import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from blueprint_pipeline.decision_evidence_contracts import canonical_digest, canonical_json
from blueprint_pipeline.public_scene_broad_repair_support import (
    BroadRepairSupportError,
    materialize_broad_repair_support,
)
from tests.test_adp_retained_scene_render_packet import _task_freeze


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(value) + "\n", encoding="utf-8")
    return path


def _absolute_record(path: Path) -> dict[str, object]:
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": _sha256(path)}


def _relative_record(root: Path, path: Path) -> dict[str, object]:
    return {
        "relative_path": path.relative_to(root).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _image(path: Path, values: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(values.astype(np.uint8), mode="RGB").save(path)
    return path


def _render_manifest(
    path: Path,
    *,
    task_id: str,
    layer: str,
    background: str,
    splat_digest: str,
    splat_count: int,
    candidate_digest: str,
    values: np.ndarray,
) -> Path:
    frame = _image(path.parent / "frames/camera.png", values)
    manifest: dict[str, object] = {
        "schema_version": "sealed_camera_render_manifest.v1",
        "status": "rendered_exact_cameras",
        "authorization_class": "method_input",
        "rendered_by_gpu": True,
        "candidate_set_digest": candidate_digest,
        "splat_digest": splat_digest,
        "source_splat": {
            "digest": splat_digest,
            **(
                {"retained_gaussian_count": splat_count}
                if layer == "shared_deleted_source_layer"
                else {"retained_count_source": "verified_standard_ply_header"}
            ),
        },
        "source_layer_role": layer,
        "render_settings": {
            "dimensions": {"width": 2, "height": 2},
            "background_rgb": background,
        },
        "calibrated_cameras": [{"id": "camera", "spec": {}}],
        "renders": [
            {
                "camera_id": "camera",
                "relative_path": "frames/camera.png",
                "size_bytes": frame.stat().st_size,
                "digest": _sha256(frame),
                "width": 2,
                "height": 2,
            }
        ],
        "sealed_camera_render_manifest_digest": "",
    }
    manifest["sealed_camera_render_manifest_digest"] = canonical_digest(
        manifest,
        digest_field="sealed_camera_render_manifest_digest",
    )
    return _write(path, manifest)


def _fixture(
    tmp_path: Path,
    *,
    task_count: int = 5,
    candidate_schema: str = "adp009b_ownership_coverage_cutout_set.v1",
) -> tuple[Path, Path, Path]:
    candidate_root = tmp_path / "candidate"
    deleted = candidate_root / "shared/deleted.ply"
    retained = candidate_root / "shared/retained.ply"
    deleted.parent.mkdir(parents=True)
    deleted.write_bytes(b"deleted-derived-ply")
    retained.write_bytes(b"retained-derived-ply")
    task_ids = [f"task_{index}" for index in range(task_count)]
    task_rows = []
    for index, task_id in enumerate(task_ids, start=1):
        freeze = _task_freeze(task_id, index)
        freeze_path = _write(candidate_root / "freezes" / f"{task_id}.json", freeze)
        task_rows.append(
            {
                "task_id": task_id,
                "task_freeze_digest": freeze["task_freeze_digest"],
                "task_freeze": _absolute_record(freeze_path),
            }
        )
    candidate: dict[str, object] = {
        "schema_version": candidate_schema,
        "task_candidates": task_rows,
        "shared_scene_union": {
            "counts": {"deleted_total": 2, "retained_total": 8},
            "outputs": {
                "deleted_source_gaussians": _relative_record(candidate_root, deleted),
                "retained_scene_gaussians": _relative_record(candidate_root, retained),
            },
        },
        "claim_boundary": {"candidate_derived_layers_only": True},
        "receipt_digest": "",
    }
    candidate["receipt_digest"] = canonical_digest(candidate, digest_field="receipt_digest")
    candidate_path = _write(candidate_root / "candidate.json", candidate)
    black = np.zeros((2, 2, 3), dtype=np.uint8)
    white = np.full((2, 2, 3), 255, dtype=np.uint8)
    black[0, 0] = [100, 50, 25]
    white[0, 0] = [100, 50, 25]
    white[1, 1] = [254, 254, 254]
    retained_rgb = np.full((2, 2, 3), [40, 80, 120], dtype=np.uint8)
    immutable = tmp_path / "immutable"
    render_rows: list[dict[str, object]] = []
    relocation_rows: list[dict[str, object]] = []
    for task_id in task_ids:
        variants = (
            ("shared_deleted_source_layer", "#000000", "black", black, deleted, 2),
            ("shared_deleted_source_layer", "#ffffff", "white", white, deleted, 2),
            ("shared_retained_scene", "#000000", "retained", retained_rgb, retained, 8),
        )
        for layer, background, label, values, splat, count in variants:
            manifest = _render_manifest(
                immutable / "renders" / task_id / label / "manifest.json",
                task_id=task_id,
                layer=layer,
                background=background,
                splat_digest=_sha256(splat),
                splat_count=count,
                candidate_digest=str(candidate["receipt_digest"]),
                values=values,
            )
            manifest_value = json.loads(manifest.read_text(encoding="utf-8"))
            render_rows.append(
                {
                    "task_id": task_id,
                    "layer": layer,
                    "background_rgb": background,
                    "manifest_path": f"/remote/{task_id}/{label}/manifest.json",
                    "manifest_digest": manifest_value[
                        "sealed_camera_render_manifest_digest"
                    ],
                }
            )
            relocation_rows.append(
                {
                    "task_id": task_id,
                    "layer": layer,
                    "background_rgb": background,
                    "remote_manifest_path": f"/remote/{task_id}/{label}/manifest.json",
                    "local_manifest": {
                        **_absolute_record(manifest),
                        "manifest_digest": manifest_value[
                            "sealed_camera_render_manifest_digest"
                        ],
                    },
                }
            )
    result: dict[str, object] = {
        "schema_version": "adp009d_retained_scene_gpu_render_result.v1",
        "status": "completed",
        "released_renderer_executed": True,
        "gpu_runtime_started": True,
        "paid_inference_performed": False,
        "provider_mutations_performed": 0,
        "candidate_set_digest": candidate["receipt_digest"],
        "shared_deleted_source_layer_digest": _sha256(deleted),
        "shared_retained_scene_digest": _sha256(retained),
        "render_manifests": render_rows,
        "blockers": [],
    }
    result_path = _write(immutable / "result.json", result)
    relocation: dict[str, object] = {
        "schema_version": "adp009d_retained_scene_gpu_render_output_relocation.v1",
        "status": "extracted_manifest_paths_verified",
        "provider_result": _absolute_record(result_path),
        "render_manifests": relocation_rows,
        "receipt_digest": "",
    }
    relocation["receipt_digest"] = canonical_digest(relocation, digest_field="receipt_digest")
    relocation_path = _write(immutable / "relocation.json", relocation)
    return candidate_path, result_path, relocation_path


@pytest.mark.parametrize(
    "candidate_schema",
    [
        "adp009b_ownership_coverage_cutout_set.v1",
        "adp009d_segment_contribution_cutout_set.v1",
    ],
)
def test_materializes_full_detectable_support_for_five_tasks(
    tmp_path: Path, candidate_schema: str
) -> None:
    candidate, result, relocation = _fixture(
        tmp_path, task_count=5, candidate_schema=candidate_schema
    )

    packet = materialize_broad_repair_support(
        candidate_set_path=candidate,
        renderer_result_path=result,
        output_relocation_receipt_path=relocation,
        output_root=tmp_path / "output",
        repair_support_dilation_pixels=1,
    )

    assert packet["status"] == "full_deleted_projection_repair_support_materialized"
    assert len(packet["task_lanes"]) == 5
    for lane in packet["task_lanes"]:
        assert lane["camera_count"] == 1
        frame = lane["frames"][0]
        assert frame["detectable_alpha_pixel_count"] == 2
        assert frame["repair_support_pixel_count"] == 4
        mask = np.asarray(
            Image.open(tmp_path / "output" / frame["repair_support_mask"]["relative_path"])
        )
        assert np.count_nonzero(mask) == 4
    assert packet["claim_boundary"]["outside_repair_support_change_authorized"] is False
    assert packet["claim_boundary"]["repaired_pixels_are_observed_physical_evidence"] is False


def test_rejects_changed_relocation_receipt(tmp_path: Path) -> None:
    candidate, result, relocation = _fixture(tmp_path, task_count=1)
    value = json.loads(relocation.read_text(encoding="utf-8"))
    value["status"] = "tampered"
    _write(relocation, value)

    with pytest.raises(BroadRepairSupportError, match="broad_repair_relocation_invalid"):
        materialize_broad_repair_support(
            candidate_set_path=candidate,
            renderer_result_path=result,
            output_relocation_receipt_path=relocation,
            output_root=tmp_path / "output",
        )
