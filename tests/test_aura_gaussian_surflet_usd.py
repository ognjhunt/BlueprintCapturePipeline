"""Strict AuraFusion360 2DGS to OpenUSD Gaussian-surflet coverage."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from blueprint_pipeline.common import sha256_file
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.gaussian_splat_decode import (
    GaussianSurfelData,
    read_aura_2dgs_surfel_ply,
)
from blueprint_pipeline.particlefield_usd import (
    GAUSSIAN_SURFLET_SCHEMA,
    build_gaussian_surflet_arrays,
    write_gaussian_surflet_particlefield_usd,
)

_HAS_PXR = importlib.util.find_spec("pxr") is not None


def _surfel(count: int = 3) -> GaussianSurfelData:
    return GaussianSurfelData(
        count=count,
        xyz=np.arange(count * 3, dtype=np.float32).reshape(count, 3) / 10.0,
        opacity=np.linspace(-1.0, 1.0, count, dtype=np.float32),
        f_dc=np.arange(count * 3, dtype=np.float32).reshape(count, 3),
        scales=np.log(
            np.arange(1, count * 2 + 1, dtype=np.float32).reshape(count, 2) / 100.0
        ),
        quats=np.tile(np.asarray([[2.0, 0.0, 0.0, 0.0]], dtype=np.float32), (count, 1)),
        sh_rest=np.tile(np.arange(1, 46, dtype=np.float32), (count, 1)),
        mask_logits=np.zeros((count, 3), dtype=np.float32),
        properties=(),
    )


def _write_fixture(path: Path, surfel: GaussianSurfelData, *, add_scale_2: bool = False) -> None:
    names = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"]
    names.extend(f"f_rest_{index}" for index in range(45))
    names.extend(["opacity", "scale_0", "scale_1"])
    if add_scale_2:
        names.append("scale_2")
    names.extend(["rot_0", "rot_1", "rot_2", "rot_3"])
    names.extend(["is_masked_0", "is_masked_1", "is_masked_2"])
    columns = [surfel.xyz[:, index] for index in range(3)]
    columns.extend(np.zeros(surfel.count, dtype=np.float32) for _ in range(3))
    columns.extend(surfel.f_dc[:, index] for index in range(3))
    columns.extend(surfel.sh_rest[:, index] for index in range(45))
    columns.append(surfel.opacity)
    columns.extend(surfel.scales[:, index] for index in range(2))
    if add_scale_2:
        columns.append(np.zeros(surfel.count, dtype=np.float32))
    columns.extend(surfel.quats[:, index] for index in range(4))
    columns.extend(surfel.mask_logits[:, index] for index in range(3))
    header = ["ply", "format binary_little_endian 1.0", f"element vertex {surfel.count}"]
    header.extend(f"property float {name}" for name in names)
    header.append("end_header\n")
    path.write_bytes(
        "\n".join(header).encode("ascii")
        + np.stack(columns, axis=1).astype("<f4").tobytes(order="C")
    )


def test_aura_reader_requires_exact_two_scale_layout(tmp_path: Path) -> None:
    source = _surfel()
    source.opacity[-1] = np.inf
    path = tmp_path / "aura.ply"
    _write_fixture(path, source)
    observed = read_aura_2dgs_surfel_ply(path)
    assert observed.count == source.count
    assert observed.scales.shape == (source.count, 2)
    np.testing.assert_array_equal(observed.sh_rest, source.sh_rest)
    assert np.isposinf(observed.opacity[-1])

    ellipsoid = tmp_path / "ellipsoid.ply"
    _write_fixture(ellipsoid, source, add_scale_2=True)
    with pytest.raises(ValueError, match="aura_2dgs_ellipsoid_scale_forbidden"):
        read_aura_2dgs_surfel_ply(ellipsoid)


def test_aura_activations_and_channel_major_sh_are_exact() -> None:
    source = _surfel(2)
    source.opacity[-1] = np.inf
    arrays = build_gaussian_surflet_arrays(source)
    np.testing.assert_allclose(arrays["scales"][:, :2], np.exp(source.scales))
    np.testing.assert_array_equal(arrays["scales"][:, 2], np.ones(2, dtype=np.float32))
    np.testing.assert_allclose(
        arrays["opacities"], 1.0 / (1.0 + np.exp(-source.opacity)), rtol=1e-6
    )
    assert arrays["opacities"][-1] == 1.0
    assert arrays["positive_infinite_opacity_logit_count"] == 1
    np.testing.assert_array_equal(arrays["orientations"][:, 0], np.ones(2))
    coefficients = arrays["sh_coefficients"].reshape(2, 16, 3)
    np.testing.assert_array_equal(coefficients[:, 0], source.f_dc)
    np.testing.assert_array_equal(coefficients[0, 1], [1.0, 16.0, 31.0])
    np.testing.assert_array_equal(coefficients[0, 15], [15.0, 30.0, 45.0])
    assert arrays["sh_degree"] == 3


def test_aura_zero_quaternion_fails_closed() -> None:
    source = _surfel()
    source.quats[1] = 0.0
    with pytest.raises(ValueError, match="aura_2dgs_zero_quaternion"):
        build_gaussian_surflet_arrays(source)


def test_aura_nonfinite_value_outside_positive_infinite_opacity_fails(tmp_path: Path) -> None:
    source = _surfel()
    source.scales[0, 0] = np.nan
    path = tmp_path / "nan_scale.ply"
    _write_fixture(path, source)
    with pytest.raises(ValueError, match="aura_2dgs_nonfinite_input"):
        read_aura_2dgs_surfel_ply(path)


def test_aura_file_conversion_is_digest_bound_and_uses_surflet_api(tmp_path: Path) -> None:
    source = tmp_path / "aura.ply"
    _write_fixture(source, _surfel())
    expected = f"sha256:{sha256_file(source)}"
    output = tmp_path / "aura.usdc"
    receipt_path = tmp_path / "aura_ovrtx_particlefield_receipt.v1.json"

    missing_digest = write_gaussian_surflet_particlefield_usd(source, output)
    if not _HAS_PXR:
        assert missing_digest["status"] == "blocked"
        return
    assert missing_digest["blockers"] == ["aura_2dgs_expected_source_sha256_missing"]
    mismatch = write_gaussian_surflet_particlefield_usd(
        source, output, expected_source_sha256="sha256:" + "0" * 64
    )
    assert mismatch["blockers"] == ["aura_2dgs_source_sha256_mismatch"]

    result = write_gaussian_surflet_particlefield_usd(
        source,
        output,
        expected_source_sha256=expected,
        receipt_path=receipt_path,
    )
    assert result["status"] == "completed"
    assert result["schema"] == GAUSSIAN_SURFLET_SCHEMA
    assert result["source_sha256"] == expected
    assert result["sealed_source_mutated"] is False
    assert result["learned_scale_components"] == 2
    assert receipt_path.is_file()
    assert result["receipt_digest"] == canonical_digest(result, digest_field="receipt_digest")

    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(str(output))
    prim = stage.GetPrimAtPath("/World/AuraAppearance/GaussianSurflets")
    assert prim.GetTypeName() == "ParticleField"
    assert "ParticleFieldKernelGaussianSurfletAPI" in prim.GetAppliedSchemas()
    assert "ParticleField3DGaussianSplat" not in str(prim.GetTypeName())
    assert UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.z
    assert UsdGeom.GetStageMetersPerUnit(stage) == 1.0
