from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from blueprint_pipeline.gaussian_splat_decode import (
    SplatData,
    read_standard_3dgs_ply,
    write_standard_3dgs_ply,
)
from blueprint_pipeline.public_scene_infusion_compose import (
    InFusionCompositionError,
    compose_infusion_supplement,
)


def _original(path: Path, *, sh_degree: int = 3) -> Path:
    grid = np.asarray(
        [[x, y, z] for x in np.linspace(-0.3, 0.3, 7) for y in np.linspace(-0.3, 0.3, 7) for z in (0.0, 0.05)],
        dtype=np.float32,
    )
    rest_count = 3 * ((sh_degree + 1) ** 2 - 1)
    return write_standard_3dgs_ply(
        SplatData(
            count=len(grid),
            xyz=grid,
            opacity=np.linspace(-1.0, 1.0, len(grid), dtype=np.float32),
            f_dc=np.arange(len(grid) * 3, dtype=np.float32).reshape(-1, 3) / 100.0,
            scales=np.full((len(grid), 3), -2.0, dtype=np.float32),
            quats=np.tile(np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (len(grid), 1)),
            properties=(),
            sh_rest=(
                None
                if rest_count == 0
                else np.arange(len(grid) * rest_count, dtype=np.float32).reshape(len(grid), rest_count) / 1000.0
            ),
        ),
        path,
    )


def _supplement(path: Path) -> Path:
    rows = [
        (0.01, 0.01, 0.02, 255, 0, 0),
        (0.02, 0.01, 0.02, 0, 255, 0),
        (0.01, 0.02, 0.02, 0, 0, 255),
        (0.02, 0.02, 0.02, 128, 128, 128),
        (0.015, 0.015, 0.03, 64, 96, 128),
    ]
    path.write_text(
        "ply\nformat ascii 1.0\nelement vertex 5\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
        + "".join("{} {} {} {} {} {}\n".format(*row) for row in rows),
        encoding="ascii",
    )
    return path


def test_compose_preserves_publisher_degree_three_fields(tmp_path: Path) -> None:
    original_path = _original(tmp_path / "background.ply")
    supplement_path = _supplement(tmp_path / "predicted_mask.ply")
    output_path = tmp_path / "composed.ply"

    receipt = compose_infusion_supplement(
        original_ply=original_path,
        supplement_ply=supplement_path,
        output_ply=output_path,
        data_root=tmp_path,
        similarity_threshold_m=0.001,
        radius_m=0.2,
        radius_min_neighbors=1,
    )

    original = read_standard_3dgs_ply(original_path)
    output = read_standard_3dgs_ply(output_path)
    assert receipt["status"] == "completed"
    assert receipt["spherical_harmonics"] == {
        "publisher_degree": 3,
        "publisher_f_rest_fields_preserved": 45,
        "supplement_f_rest_initialization": "zero",
    }
    assert receipt["counts"]["original_retained"] == original.count
    assert output.count == original.count + receipt["counts"]["supplement_retained"]
    np.testing.assert_array_equal(output.sh_rest[: original.count], original.sh_rest)
    np.testing.assert_array_equal(output.sh_rest[original.count :], 0.0)
    assert receipt["proof_boundaries"]["composition_is_not_metric_surface_truth"] is True


def test_compose_rejects_degree_zero_original(tmp_path: Path) -> None:
    with pytest.raises(InFusionCompositionError, match="infusion_original_higher_order_sh_missing"):
        compose_infusion_supplement(
            original_ply=_original(tmp_path / "degree0.ply", sh_degree=0),
            supplement_ply=_supplement(tmp_path / "supplement.ply"),
            output_ply=tmp_path / "output.ply",
            data_root=tmp_path,
        )


def test_compose_refuses_output_outside_data_root(tmp_path: Path) -> None:
    root = tmp_path / "allowed"
    root.mkdir()
    with pytest.raises(InFusionCompositionError, match="infusion_output_outside_data_root"):
        compose_infusion_supplement(
            original_ply=_original(root / "background.ply"),
            supplement_ply=_supplement(root / "supplement.ply"),
            output_ply=tmp_path / "escape.ply",
            data_root=root,
        )


def test_compose_refuses_conflicting_existing_output(tmp_path: Path) -> None:
    original_path = _original(tmp_path / "background.ply")
    supplement_path = _supplement(tmp_path / "supplement.ply")
    output_path = tmp_path / "output.ply"
    output_path.write_bytes(b"unrelated")
    with pytest.raises(InFusionCompositionError, match="infusion_composition_output_conflict"):
        compose_infusion_supplement(
            original_ply=original_path,
            supplement_ply=supplement_path,
            output_ply=output_path,
            data_root=tmp_path,
            similarity_threshold_m=0.001,
            radius_m=0.2,
            radius_min_neighbors=1,
        )
