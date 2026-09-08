import json
from pathlib import Path

import numpy as np
import pytest

from blueprint_pipeline.artifixer_source_geometry_admission import admit_source_geometry
from blueprint_pipeline.gaussian_splat_decode import (
    SplatData,
    read_standard_3dgs_ply,
    write_standard_3dgs_ply,
)


def _source(tmp_path, *, outliers=1, visible=False):
    rng = np.random.default_rng(1)
    xyz = rng.uniform(-1, 1, (2000, 3)).astype(np.float32)
    xyz[:, 2] += 4
    if outliers:
        xyz[-outliers:] = [0, 0, 10000] if visible else [10000, 10000, 4]
    if outliers == 3:
        xyz[-3:] = [[10000, 0, 4], [-10000, 0, 4], [0, 10000, 4]]
    n = len(xyz)
    splat = SplatData(
        count=n,
        properties=(),
        xyz=xyz,
        f_dc=np.zeros((n, 3), np.float32),
        sh_rest=np.zeros((n, 45), np.float32),
        opacity=np.zeros(n, np.float32),
        scales=np.full((n, 3), -4, np.float32),
        quats=np.tile([1.0, 0, 0, 0], (n, 1)).astype(np.float32),
    )
    path = tmp_path / "source.ply"
    write_standard_3dgs_ply(splat, path)
    return path


def _cameras():
    return [
        {
            "camera_id": "camera",
            "T_world_camera_opencv": np.eye(4),
            "intrinsics": {"fl_x": 500, "fl_y": 500, "cx": 320, "cy": 240, "w": 640, "h": 480},
        }
    ]


def test_quarantine_keeps_exact_original_rows_and_quality_guard(tmp_path):
    source = _source(tmp_path)
    original = source.read_bytes()
    path, receipt = admit_source_geometry(
        source=source, cameras=_cameras(), output_root=tmp_path / "admit"
    )
    assert source.read_bytes() == original
    assert (
        path.read_bytes().split(b"end_header\n", 1)[1]
        == original.split(b"end_header\n", 1)[1][
            : -(len(original.split(b"end_header\n", 1)[1]) // 2000)
        ]
    )
    assert receipt["initialization_quality"]["status"] == "qualified"
    assert receipt["quarantined_gaussian_count"] == 1
    assert read_standard_3dgs_ply(path).count == 1999
    assert json.loads(Path(receipt["quarantine_indices"]["path"]).read_text())["indices"] == [1999]
    assert receipt["pixel_equality_claimed"] is False


def test_qualified_source_is_byte_identical(tmp_path):
    source = _source(tmp_path, outliers=0)
    path, receipt = admit_source_geometry(
        source=source, cameras=_cameras(), output_root=tmp_path / "admit"
    )
    assert path == source
    assert receipt["quarantined_gaussian_count"] == 0


@pytest.mark.parametrize(
    "outliers,visible,error", [(1, True, "camera_support_overlap"), (3, False, "fraction_exceeded")]
)
def test_refuses_visible_or_excessive_quarantine(tmp_path, outliers, visible, error):
    source = _source(tmp_path, outliers=outliers, visible=visible)
    with pytest.raises(ValueError, match=error):
        admit_source_geometry(source=source, cameras=_cameras(), output_root=tmp_path / "admit")
    receipt = json.loads((tmp_path / "admit/source_geometry_admission.json").read_text())
    assert receipt["status"] == "blocked"
    assert not (tmp_path / "admit/initialization.ply").exists()


def test_actual_export_adapter_accepts_conditioned_identity_and_preserves_refusal_metrics(tmp_path):
    from tests.test_public_scene_artifixer3d_export_position_range import (
        _checkpoint,
        _load_real_provider_runner,
        _ValueTensor,
    )

    source = _source(tmp_path)
    runner = _load_real_provider_runner()
    original = read_standard_3dgs_ply(source)

    def model(splat):
        checkpoint = _checkpoint(splat.xyz)
        checkpoint["rotation"] = _ValueTensor(splat.quats)
        checkpoint["scale"] = _ValueTensor(splat.scales)
        return runner._CheckpointExportModel(
            checkpoint, reference_splat=splat, geometry_policy=runner.RETAINED_GEOMETRY_POLICY
        )

    with pytest.raises(ValueError) as failure:
        model(original)
    assert failure.value.geometry_quality["status"] == "blocked"
    path, _ = admit_source_geometry(
        source=source, cameras=_cameras(), output_root=tmp_path / "admit"
    )
    admitted = model(read_standard_3dgs_ply(path))
    assert admitted.gaussian_field_drift_quality["status"] == "qualified"
