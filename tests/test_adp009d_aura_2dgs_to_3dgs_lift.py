from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from blueprint_pipeline.adp009d_aura_2dgs_to_3dgs_lift import (
    DEFAULT_THICKNESS_RATIO,
    MIN_LOG_SCALE,
    OPACITY_LOGIT_CEILING,
    AuraLiftError,
    lift_aura_2dgs_ply_to_3dgs,
    read_binary_ply,
    write_binary_ply,
)

_SH_REST = [f"f_rest_{index}" for index in range(45)]
_AURA_PROPERTIES = [
    "x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2",
    *_SH_REST,
    "opacity", "scale_0", "scale_1", "rot_0", "rot_1", "rot_2", "rot_3",
    "is_masked_0", "is_masked_1", "is_masked_2",
]


def _write_aura_ply(path: Path, rows: list[dict]) -> None:
    dtype = np.dtype([(name, "<f4") for name in _AURA_PROPERTIES])
    vertices = np.zeros(len(rows), dtype=dtype)
    for index, row in enumerate(rows):
        for name, value in row.items():
            vertices[name][index] = value
    write_binary_ply(path, _AURA_PROPERTIES, vertices)


def test_lift_adds_a_third_axis_thinner_than_both_in_plane_axes(tmp_path: Path) -> None:
    source = tmp_path / "aura.ply"
    _write_aura_ply(
        source,
        [
            {"x": 1.0, "scale_0": -6.0, "scale_1": -5.0, "opacity": 2.0, "rot_0": 1.0},
            {"x": 2.0, "scale_0": -3.0, "scale_1": -9.0, "opacity": -1.0, "rot_0": 1.0},
        ],
    )
    destination = tmp_path / "lifted.ply"

    receipt = lift_aura_2dgs_ply_to_3dgs(source, destination)

    assert receipt["status"] == "lifted"
    assert receipt["vertex_count"] == 2
    properties, vertices = read_binary_ply(destination)
    assert "scale_2" in properties
    assert properties.index("scale_2") == properties.index("scale_1") + 1
    # Third axis sits ln(ratio) below the smaller in-plane axis.
    expected = math.log(DEFAULT_THICKNESS_RATIO)
    assert vertices["scale_2"][0] == pytest.approx(-6.0 + expected, abs=1e-5)
    assert vertices["scale_2"][1] == pytest.approx(-9.0 + expected, abs=1e-5)
    # A surfel must stay flat: the new axis is never the largest.
    for index in range(2):
        assert vertices["scale_2"][index] <= min(
            vertices["scale_0"][index], vertices["scale_1"][index]
        )


def test_infinite_opacity_becomes_finite_without_changing_rendered_opacity(
    tmp_path: Path,
) -> None:
    """63% of the sealed opacities are +inf; infinity poisons any downstream math."""

    source = tmp_path / "aura.ply"
    _write_aura_ply(
        source,
        [
            {"opacity": np.inf, "scale_0": -6.0, "scale_1": -6.0, "rot_0": 1.0},
            {"opacity": -np.inf, "scale_0": -6.0, "scale_1": -6.0, "rot_0": 1.0},
            {"opacity": 3.5, "scale_0": -6.0, "scale_1": -6.0, "rot_0": 1.0},
        ],
    )
    destination = tmp_path / "lifted.ply"

    receipt = lift_aura_2dgs_ply_to_3dgs(source, destination)

    assert receipt["non_finite_opacity_count"] == 2
    _, vertices = read_binary_ply(destination)
    assert np.isfinite(vertices["opacity"]).all()
    assert vertices["opacity"][0] == pytest.approx(OPACITY_LOGIT_CEILING)
    assert vertices["opacity"][1] == pytest.approx(-OPACITY_LOGIT_CEILING)
    assert vertices["opacity"][2] == pytest.approx(3.5)
    # The clamp must be invisible after the sigmoid the renderer applies.
    sigmoid = 1.0 / (1.0 + np.exp(-np.float32(OPACITY_LOGIT_CEILING)))
    assert np.float32(sigmoid) == np.float32(1.0)


def test_lift_drops_the_mask_channels_and_emits_standard_3dgs_layout(
    tmp_path: Path,
) -> None:
    source = tmp_path / "aura.ply"
    _write_aura_ply(source, [{"scale_0": -6.0, "scale_1": -6.0, "rot_0": 1.0}])
    destination = tmp_path / "lifted.ply"

    receipt = lift_aura_2dgs_ply_to_3dgs(source, destination)

    properties, _ = read_binary_ply(destination)
    assert not [name for name in properties if name.startswith("is_masked")]
    assert receipt["dropped_properties"] == ["is_masked_0", "is_masked_1", "is_masked_2"]
    # Exactly the standard 3DGS layout: 3 pos + 3 normal + 3 DC + 45 rest
    # + opacity + 3 scales + 4 rotation.
    assert len(properties) == 62
    assert properties[:6] == ["x", "y", "z", "nx", "ny", "nz"]
    assert properties[-4:] == ["rot_0", "rot_1", "rot_2", "rot_3"]


def test_lift_refuses_a_ply_that_is_not_the_sealed_aura_layout(tmp_path: Path) -> None:
    """An unrelated PLY must never be silently reinterpreted as the appearance."""

    already_3dgs = tmp_path / "standard.ply"
    properties = [name for name in _AURA_PROPERTIES if not name.startswith("is_masked")]
    properties.insert(properties.index("scale_1") + 1, "scale_2")
    dtype = np.dtype([(name, "<f4") for name in properties])
    write_binary_ply(already_3dgs, properties, np.zeros(1, dtype=dtype))

    with pytest.raises(AuraLiftError) as excinfo:
        lift_aura_2dgs_ply_to_3dgs(already_3dgs, tmp_path / "out.ply")
    assert "source_already_has_three_scale_axes" in excinfo.value.errors
    assert not (tmp_path / "out.ply").exists()

    truncated = tmp_path / "short.ply"
    write_binary_ply(truncated, ["x", "y", "z"], np.zeros(1, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4")]))
    with pytest.raises(AuraLiftError):
        lift_aura_2dgs_ply_to_3dgs(truncated, tmp_path / "out2.ply")


def test_extreme_scales_are_floored_without_inverting_the_surfel(tmp_path: Path) -> None:
    """The sealed file reaches log-scale -31.2; flooring must not fatten the disc."""

    source = tmp_path / "aura.ply"
    _write_aura_ply(
        source,
        [
            {"scale_0": -31.2159, "scale_1": -22.0989, "opacity": 1.0, "rot_0": 1.0},
            {"scale_0": -59.5, "scale_1": -59.5, "opacity": 1.0, "rot_0": 1.0},
        ],
    )
    destination = tmp_path / "lifted.ply"

    receipt = lift_aura_2dgs_ply_to_3dgs(source, destination)

    _, vertices = read_binary_ply(destination)
    assert np.isfinite(vertices["scale_2"]).all()
    for index in range(2):
        assert vertices["scale_2"][index] >= MIN_LOG_SCALE
        # Even when floored, the third axis never exceeds the in-plane axes.
        assert vertices["scale_2"][index] <= min(
            vertices["scale_0"][index], vertices["scale_1"][index]
        )
    assert receipt["third_axis_floored_count"] == 1


def test_receipt_binds_both_digests_and_never_claims_equivalence(tmp_path: Path) -> None:
    source = tmp_path / "aura.ply"
    _write_aura_ply(source, [{"scale_0": -6.0, "scale_1": -6.0, "rot_0": 1.0}])
    destination = tmp_path / "lifted.ply"

    receipt = lift_aura_2dgs_ply_to_3dgs(source, destination)

    assert receipt["source_sha256"].startswith("sha256:")
    assert receipt["destination_sha256"].startswith("sha256:")
    assert receipt["source_sha256"] != receipt["destination_sha256"]
    assert receipt["source_mutated"] is False
    # The lift is an approximation; it must not assert the appearance survived.
    assert receipt["appearance_equivalence_established"] is False


def test_thickness_ratio_is_rejected_outside_the_open_unit_interval(tmp_path: Path) -> None:
    source = tmp_path / "aura.ply"
    _write_aura_ply(source, [{"scale_0": -6.0, "scale_1": -6.0, "rot_0": 1.0}])

    for ratio in (0.0, 1.0, -0.5, 2.0):
        with pytest.raises(AuraLiftError):
            lift_aura_2dgs_ply_to_3dgs(source, tmp_path / "out.ply", thickness_ratio=ratio)
