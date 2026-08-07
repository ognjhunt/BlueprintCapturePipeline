from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

from blueprint_pipeline.adp009d_aura_task_volume_exclusion import (
    CAN_AXIS_XY_M,
    EXCLUSION_RADIUS_M,
    SUPPORT_HEIGHT_M,
    AuraTaskVolumeExclusionError,
    materialize_aura_task_volume_exclusion,
    validate_aura_task_volume_exclusion_receipt,
)

PROPERTIES = ["x", "y", "z", "opacity", "scale_0", "scale_1"]


def _write_ply(path: Path, rows: np.ndarray) -> None:
    header = ["ply", "format binary_little_endian 1.0", f"element vertex {len(rows)}"]
    header += [f"property float {name}" for name in PROPERTIES]
    header.append("end_header")
    path.write_bytes(
        ("\n".join(header) + "\n").encode("latin-1")
        + np.ascontiguousarray(rows, dtype="<f4").tobytes()
    )


def _fixture(tmp_path: Path) -> tuple[Path, np.ndarray]:
    """Ghost splats inside the can volume, shelf below it, scene around it."""

    cx, cy = CAN_AXIS_XY_M
    rows = []
    # 3 ghost splats inside the cylinder, above the support plane -> removed.
    for height in (0.05, 0.10, 0.15):
        rows.append([cx, cy, SUPPORT_HEIGHT_M + height, 5.0, -3.0, -3.0])
    # 2 shelf splats inside the footprint at/below the support plane -> kept.
    rows.append([cx, cy, SUPPORT_HEIGHT_M, 5.0, -3.0, -3.0])
    rows.append([cx + 0.01, cy, SUPPORT_HEIGHT_M - 0.005, 5.0, -3.0, -3.0])
    # 2 surrounding splats outside the radius at can height -> kept.
    rows.append([cx + EXCLUSION_RADIUS_M + 0.01, cy, SUPPORT_HEIGHT_M + 0.10, 5.0, -3.0, -3.0])
    rows.append([cx, cy + 0.25, SUPPORT_HEIGHT_M + 0.10, 5.0, -3.0, -3.0])
    # 1 splat above the ceiling -> kept.
    rows.append([cx, cy, SUPPORT_HEIGHT_M + 0.30, 5.0, -3.0, -3.0])
    array = np.asarray(rows, dtype="<f4")
    path = tmp_path / "scene.ply"
    _write_ply(path, array)
    return path, array


def _materialize(tmp_path: Path, source: Path, **kwargs) -> dict:
    return materialize_aura_task_volume_exclusion(
        source_ply_path=source,
        output_ply_path=tmp_path / "excluded.ply",
        receipt_path=tmp_path / "receipt.json",
        expected_source_sha256="",
        expected_removed_vertex_count=kwargs.pop("expected_removed_vertex_count", 3),
        # The tiny fixture is 8 rows; the production 0.5% ceiling is exercised
        # separately by test_default_removed_fraction_ceiling_rejects_broad_removal.
        removed_fraction_ceiling=kwargs.pop("removed_fraction_ceiling", 0.5),
        **kwargs,
    )


def test_default_removed_fraction_ceiling_rejects_broad_removal(
    tmp_path: Path,
) -> None:
    """A removal that eats a meaningful share of the scene must fail closed."""

    source, _ = _fixture(tmp_path)

    with pytest.raises(
        AuraTaskVolumeExclusionError, match="aura_exclusion_removed_fraction_exceeded"
    ):
        materialize_aura_task_volume_exclusion(
            source_ply_path=source,
            output_ply_path=tmp_path / "excluded.ply",
            expected_source_sha256="",
            expected_removed_vertex_count=3,
        )


def test_removes_only_centre_inside_ghosts_and_copies_rows_verbatim(
    tmp_path: Path,
) -> None:
    source, array = _fixture(tmp_path)
    receipt = _materialize(tmp_path, source)

    assert receipt["removed_vertex_count"] == 3
    assert receipt["output_vertex_count"] == 5
    assert receipt["support_surface_disturbed"] is False

    # Retained rows must be byte-identical to their source rows, in order.
    body = (tmp_path / "excluded.ply").read_bytes()
    kept = np.frombuffer(body[body.index(b"end_header\n") + 11 :], dtype="<f4").reshape(
        5, len(PROPERTIES)
    )
    expected = array[3:]
    assert np.array_equal(kept, expected)
    # The shelf splats inside the footprint survived.
    assert float(kept[0][2]) == pytest.approx(SUPPORT_HEIGHT_M, abs=1e-6)


def test_rejects_source_that_is_not_the_sealed_asset(tmp_path: Path) -> None:
    source, _ = _fixture(tmp_path)

    with pytest.raises(
        AuraTaskVolumeExclusionError, match="aura_exclusion_source_digest_mismatch"
    ):
        materialize_aura_task_volume_exclusion(
            source_ply_path=source,
            output_ply_path=tmp_path / "excluded.ply",
            expected_source_sha256="sha256:" + "0" * 64,
        )


def test_rejects_removed_count_drift(tmp_path: Path) -> None:
    """The removal is preregistered; a different count means the rule drifted."""

    source, _ = _fixture(tmp_path)

    with pytest.raises(
        AuraTaskVolumeExclusionError, match="aura_exclusion_removed_count_unexpected"
    ):
        _materialize(tmp_path, source, expected_removed_vertex_count=2)


def test_receipt_validator_rejects_unsealed_or_tampered_evidence(
    tmp_path: Path,
) -> None:
    source, _ = _fixture(tmp_path)
    receipt = _materialize(tmp_path, source)

    with pytest.raises(
        AuraTaskVolumeExclusionError,
        match="aura_exclusion_receipt_source_not_sealed_asset",
    ):
        validate_aura_task_volume_exclusion_receipt(receipt)

    tampered = copy.deepcopy(receipt)
    tampered["removed_vertex_count"] = 1
    with pytest.raises(
        AuraTaskVolumeExclusionError, match="aura_exclusion_receipt_digest_mismatch"
    ):
        validate_aura_task_volume_exclusion_receipt(tampered)


def _confinement_inputs(tmp_path: Path):
    """Camera looking down -Z world at the can, so camera_z maps to height."""

    import numpy as np

    height = width = 9
    intrinsic = [[10.0, 0.0, 4.0], [0.0, 10.0, 4.0], [0.0, 0.0, 1.0]]
    # Camera 1 m above the can, looking straight down.
    world_from_camera = [
        [1.0, 0.0, 0.0, CAN_AXIS_XY_M[0]],
        [0.0, -1.0, 0.0, CAN_AXIS_XY_M[1]],
        [0.0, 0.0, -1.0, SUPPORT_HEIGHT_M + 1.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    baseline = np.zeros((height, width, 3), dtype=np.uint8)
    # Depth chosen so every pixel's surface sits 0.10 m above the support.
    depth = np.full((height, width), 0.90, dtype=np.float64)
    return baseline, depth, intrinsic, world_from_camera


def test_render_confinement_accepts_changes_inside_the_task_volume(
    tmp_path: Path,
) -> None:
    from blueprint_pipeline.adp009d_aura_task_volume_exclusion import (
        verify_exclusion_render_confined,
    )

    baseline, depth, intrinsic, pose = _confinement_inputs(tmp_path)
    derivative = baseline.copy()
    derivative[4, 4] = (255, 0, 0)  # principal ray -> can axis, inside the cylinder

    report = verify_exclusion_render_confined(
        baseline_rgb=baseline,
        derivative_rgb=derivative,
        baseline_depth_camera_z=depth,
        intrinsic_matrix=intrinsic,
        world_from_camera=pose,
    )
    assert report["changed_pixel_count"] == 1
    assert report["changed_pixels_not_viewing_exclusion_volume"] == 0
    assert report["confined"] is True


def test_render_confinement_rejects_changes_beyond_the_task_volume(
    tmp_path: Path,
) -> None:
    """An edit that altered the wider scene must fail closed."""

    from blueprint_pipeline.adp009d_aura_task_volume_exclusion import (
        verify_exclusion_render_confined,
    )

    baseline, depth, intrinsic, pose = _confinement_inputs(tmp_path)
    derivative = baseline.copy()
    derivative[0, 0] = (255, 0, 0)  # corner ray -> far outside the 6 cm cylinder

    with pytest.raises(
        AuraTaskVolumeExclusionError,
        match="aura_exclusion_render_changed_outside_task_volume",
    ):
        verify_exclusion_render_confined(
            baseline_rgb=baseline,
            derivative_rgb=derivative,
            baseline_depth_camera_z=depth,
            intrinsic_matrix=intrinsic,
            world_from_camera=pose,
        )
