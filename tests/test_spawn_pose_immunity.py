"""A placed asset must survive the spawner's zero-set, or refuse to launch.

rt51 and rt52 each spent a paid GPU run to learn the same fact: Isaac's spawn
authors a local (0,0,0) opinion on the referencing prim's transform, and a
local opinion always beats the referenced asset's own. An asset that carries
its placement on the prim the spawner controls is an asset that spawns at the
origin, every time, no matter what the USD file says when opened alone.

The whole class dies here: reference the asset, set the spawner's zero the way
Isaac does, and read where the root body actually lands. Composition is
deterministic, so the laptop answer is the GPU answer.
"""

from __future__ import annotations

import textwrap

import pytest

pytest.importorskip("pxr")

from blueprint_pipeline.spawn_pose_immunity import (  # noqa: E402
    SpawnPoseImmunityError,
    probe_spawn_pose_immunity,
)


PLACEMENT = (1.9742142, 1.4792181, 0.0)


def _write(path, body):
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    return path


@pytest.fixture
def defeatable_asset(tmp_path):
    """Placement on the default prim - the prim the spawner overwrites."""

    return _write(
        tmp_path / "defeatable.usda",
        f"""\
        #usda 1.0
        (
            defaultPrim = "Asset"
            metersPerUnit = 1.0
            upAxis = "Z"
        )

        def Xform "Asset"
        {{
            double3 xformOp:translate = ({PLACEMENT[0]}, {PLACEMENT[1]}, {PLACEMENT[2]})
            uniform token[] xformOpOrder = ["xformOp:translate"]

            def Xform "cabinet"
            {{
            }}
        }}
        """,
    )


@pytest.fixture
def immune_asset(tmp_path):
    """Placement on the body prim - below anything the spawner touches."""

    return _write(
        tmp_path / "immune.usda",
        f"""\
        #usda 1.0
        (
            defaultPrim = "Asset"
            metersPerUnit = 1.0
            upAxis = "Z"
        )

        def Xform "Asset"
        {{
            def Xform "cabinet"
            {{
                double3 xformOp:translate = ({PLACEMENT[0]}, {PLACEMENT[1]}, {PLACEMENT[2]})
                uniform token[] xformOpOrder = ["xformOp:translate"]
            }}
        }}
        """,
    )


def test_the_v18_pattern_is_refused(defeatable_asset):
    """The asset that cost rt51 and rt52 must fail this gate on the laptop."""

    with pytest.raises(SpawnPoseImmunityError) as excinfo:
        probe_spawn_pose_immunity(
            asset_path=defeatable_asset,
            root_body_prim_name="cabinet",
            expected_world_position_m=PLACEMENT,
        )

    assert any("spawn_zero_defeats_placement" in e for e in excinfo.value.errors)


def test_body_level_placement_survives_the_spawner(immune_asset):
    receipt = probe_spawn_pose_immunity(
        asset_path=immune_asset,
        root_body_prim_name="cabinet",
        expected_world_position_m=PLACEMENT,
    )

    assert receipt["immune"] is True
    assert receipt["position_after_spawner_zero_m"] == pytest.approx(
        PLACEMENT, abs=1e-4
    )


def test_an_asset_that_never_reaches_the_expected_pose_is_refused(immune_asset):
    """Immunity to the override is not enough; it must land where we said."""

    with pytest.raises(SpawnPoseImmunityError) as excinfo:
        probe_spawn_pose_immunity(
            asset_path=immune_asset,
            root_body_prim_name="cabinet",
            expected_world_position_m=(9.0, 9.0, 9.0),
        )

    assert any("expected_position_not_reached" in e for e in excinfo.value.errors)


def test_a_missing_root_body_is_refused(immune_asset):
    with pytest.raises(SpawnPoseImmunityError) as excinfo:
        probe_spawn_pose_immunity(
            asset_path=immune_asset,
            root_body_prim_name="fridge_root",
            expected_world_position_m=PLACEMENT,
        )

    assert any("root_body_prim_missing" in e for e in excinfo.value.errors)


def test_the_receipt_records_both_measurements(immune_asset):
    """The receipt must show the probe actually ran both compositions."""

    receipt = probe_spawn_pose_immunity(
        asset_path=immune_asset,
        root_body_prim_name="cabinet",
        expected_world_position_m=PLACEMENT,
    )

    assert receipt["position_unauthored_m"] == pytest.approx(PLACEMENT, abs=1e-4)
    assert receipt["claim_boundary"][
        "composition_on_the_laptop_is_composition_on_the_gpu"
    ]
