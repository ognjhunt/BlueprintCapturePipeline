from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import jsonschema
import numpy as np
import pytest
from PIL import Image

from blueprint_pipeline.gaussian_object_partition import (
    GaussianObjectPartitionError,
    partition_gaussian_object,
    select_gaussians_from_multiview_masks,
    selection_from_semantic_lifting,
    validate_gaussian_object_selection,
    verify_gaussian_object_partition_files,
)
from blueprint_pipeline.gaussian_splat_decode import (
    SplatData,
    read_standard_3dgs_ply,
    write_standard_3dgs_ply,
)
from blueprint_pipeline.scene_placement.semantic_gaussian_lifting import (
    RESULT_SCHEMA_VERSION,
    canonical_json_digest,
)


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _f_dc(rgb: tuple[float, float, float]) -> np.ndarray:
    sh_c0 = 0.28209479177387814
    return np.asarray([(channel - 0.5) / sh_c0 for channel in rgb], dtype=np.float32)


def _scene() -> tuple[SplatData, np.ndarray]:
    background = np.asarray(
        [[x, y, 3.0] for y in np.linspace(-0.7, 0.7, 15) for x in np.linspace(-0.9, 0.9, 19)],
        dtype=np.float32,
    )
    object_xyz = np.asarray(
        [[x, y, 2.0] for y in np.linspace(-0.16, 0.16, 7) for x in np.linspace(-0.12, 0.12, 5)],
        dtype=np.float32,
    )
    xyz = np.concatenate([background, object_xyz], axis=0)
    count = xyz.shape[0]
    object_indices = np.arange(background.shape[0], count, dtype=np.int64)
    colors = np.repeat(_f_dc((0.12, 0.25, 0.82))[None, :], count, axis=0)
    colors[object_indices] = _f_dc((0.92, 0.08, 0.06))
    sh_rest = np.zeros((count, 45), dtype=np.float32)
    sh_rest[object_indices, 0] = 0.15
    return (
        SplatData(
            count=count,
            xyz=xyz,
            opacity=np.full(count, 8.0, dtype=np.float32),
            f_dc=colors,
            scales=np.full((count, 3), np.log(0.045), dtype=np.float32),
            quats=np.tile(np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (count, 1)),
            properties=(),
            sh_rest=sh_rest,
        ),
        object_indices,
    )


def _write_mask(
    path: Path,
    object_xyz: np.ndarray,
    *,
    camera_x: float,
    width: int = 320,
    height: int = 240,
) -> dict:
    fx = fy = 250.0
    cx, cy = width / 2.0, height / 2.0
    u = np.rint(fx * (object_xyz[:, 0] - camera_x) / object_xyz[:, 2] + cx).astype(int)
    v = np.rint(fy * object_xyz[:, 1] / object_xyz[:, 2] + cy).astype(int)
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[max(0, v.min() - 6) : min(height, v.max() + 7), max(0, u.min() - 6) : min(width, u.max() + 7)] = 255
    Image.fromarray(mask).save(path)
    transform = np.eye(4, dtype=np.float64)
    transform[0, 3] = camera_x
    return {
        "view_id": path.stem,
        "mask_path": str(path),
        "mask_digest": _sha256(path),
        "intrinsics": {"fx": fx, "fy": fy, "cx": cx, "cy": cy, "width": width, "height": height},
        "T_world_camera_opencv": transform.tolist(),
    }


def _selection_fixture(tmp_path: Path) -> tuple[Path, dict, np.ndarray]:
    splat, expected_object_indices = _scene()
    source = write_standard_3dgs_ply(splat, tmp_path / "captured-kitchen.ply")
    views = [
        _write_mask(tmp_path / "mask-left.png", splat.xyz[expected_object_indices], camera_x=-0.25),
        _write_mask(tmp_path / "mask-center.png", splat.xyz[expected_object_indices], camera_x=0.0),
        _write_mask(tmp_path / "mask-right.png", splat.xyz[expected_object_indices], camera_x=0.25),
    ]
    selection = select_gaussians_from_multiview_masks(
        source,
        object_id="counter-mug",
        views=views,
        min_positive_views=2,
        foreground_probability_threshold=0.75,
        depth_tolerance_m=0.02,
    )
    return source, selection, expected_object_indices


def test_multiview_masks_select_and_partition_exact_mug_gaussians(tmp_path: Path) -> None:
    source, selection, expected_object_indices = _selection_fixture(tmp_path)
    assert selection["selected_gaussian_ids"] == expected_object_indices.tolist()
    assert selection["semantic_completeness_validated"] is False
    assert selection["physics_authority_granted"] is False

    partition = partition_gaussian_object(
        source,
        selection,
        output_dir=tmp_path / "partition",
    )
    schema = json.loads(
        (
            Path(__file__).parents[1]
            / "docs/schemas/gaussian_object_partition.v1.schema.json"
        ).read_text(encoding="utf-8")
    )
    jsonschema.validate(selection, schema)
    jsonschema.validate(partition, schema)
    report = verify_gaussian_object_partition_files(partition)
    assert report["status"] == "passed"
    assert report["exact_mechanical_partition_verified"] is True
    assert report["semantic_completeness_verified"] is False
    assert partition["counts"] == {
        "source": 320,
        "background": 285,
        "object": 35,
    }
    background = read_standard_3dgs_ply(partition["artifacts"]["background"]["path"])
    object_splat = read_standard_3dgs_ply(partition["artifacts"]["object"]["path"])
    assert background.sh_rest is not None
    assert object_splat.sh_rest is not None
    np.testing.assert_array_equal(object_splat.sh_rest[:, 0], np.full(35, 0.15, dtype=np.float32))
    np.testing.assert_allclose(object_splat.xyz.mean(axis=0), np.zeros(3), atol=1e-7)
    assert partition["partition"]["background_excludes_selected_object"] is True
    assert partition["background_completion"]["generated_gaussians_added"] is False


def test_selection_and_partition_tampering_fail_closed(tmp_path: Path) -> None:
    source, selection, _ = _selection_fixture(tmp_path)
    tampered = copy.deepcopy(selection)
    tampered["selected_gaussian_ids"].append(0)
    with pytest.raises(GaussianObjectPartitionError, match="ids_not_unique_sorted|ids_digest"):
        validate_gaussian_object_selection(tampered)

    wrong_source = tmp_path / "wrong-source.ply"
    wrong_source.write_bytes(source.read_bytes() + b"tamper")
    with pytest.raises(GaussianObjectPartitionError, match="source_digest_mismatch"):
        partition_gaussian_object(
            wrong_source,
            selection,
            output_dir=tmp_path / "bad-partition",
        )

    partition = partition_gaussian_object(
        source,
        selection,
        output_dir=tmp_path / "partition",
    )
    background_path = Path(partition["artifacts"]["background"]["path"])
    background = read_standard_3dgs_ply(background_path)
    background.xyz[0] = background.xyz[1]
    write_standard_3dgs_ply(background, background_path)
    forged = copy.deepcopy(partition)
    forged.pop("gaussian_object_partition_digest")
    forged["artifacts"]["background"]["digest"] = _sha256(background_path)
    verification = verify_gaussian_object_partition_files(forged)
    assert verification["status"] == "blocked"
    assert "gaussian_object_partition_background_rows_mismatch_source" in (
        verification["errors"]
    )


def test_semantic_lifting_adapter_maps_gaussian_ids_to_source_rows(tmp_path: Path) -> None:
    splat, _ = _scene()
    source = write_standard_3dgs_ply(splat, tmp_path / "source.ply")
    mapping = [
        {"gaussian_id": index + 1000, "source_index": index, "source_class": "observed"}
        for index in range(splat.count)
    ]
    selected = [1000 + splat.count - 2, 1000 + splat.count - 1]
    lifting = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "completed",
        "bindings": {
            "analysis_splat_digest": _sha256(source),
            "gaussian_mapping_digest": canonical_json_digest(mapping),
        },
        "tracks": [
            {
                "track_id": "counter-mug",
                "label": "mug",
                "status": "qualified_semantic_support_candidate",
                "selected_gaussian_ids": selected,
                "supporting_view_ids": ["left", "right"],
                "angular_diversity_degrees": 35.0,
            }
        ],
    }
    lifting["result_digest"] = canonical_json_digest(lifting)
    selection = selection_from_semantic_lifting(
        lifting,
        gaussian_mapping=mapping,
        source_splat_path=source,
        track_id="counter-mug",
    )
    assert selection["selected_gaussian_ids"] == [splat.count - 2, splat.count - 1]
    assert selection["method"]["method_id"] == "blueprint.semantic_gaussian_lifting"


def test_multiview_mask_digest_and_calibration_fail_closed(tmp_path: Path) -> None:
    splat, object_indices = _scene()
    source = write_standard_3dgs_ply(splat, tmp_path / "source.ply")
    views = [
        _write_mask(tmp_path / "a.png", splat.xyz[object_indices], camera_x=-0.2),
        _write_mask(tmp_path / "b.png", splat.xyz[object_indices], camera_x=0.2),
    ]
    views[0]["mask_digest"] = "sha256:" + "0" * 64
    with pytest.raises(GaussianObjectPartitionError, match="mask_digest_mismatch"):
        select_gaussians_from_multiview_masks(source, object_id="mug", views=views)
    views[0]["mask_digest"] = _sha256(Path(views[0]["mask_path"]))
    views[0]["T_world_camera_opencv"][0][0] = 2.0
    with pytest.raises(GaussianObjectPartitionError, match="rotation_not_orthonormal"):
        select_gaussians_from_multiview_masks(source, object_id="mug", views=views)
