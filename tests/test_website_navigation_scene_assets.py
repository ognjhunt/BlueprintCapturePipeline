"""A scene kept as filmed gets its Isaac files on the controller, in the compiled spec's own frame."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
from pxr import Usd, UsdGeom

from blueprint_pipeline.gaussian_splat_decode import write_standard_3dgs_ply
from blueprint_pipeline.local_reconstruction_adapters import _sha256_file
from blueprint_pipeline.website_locomotion_task import scene_facts
from blueprint_pipeline.website_navigation_scene_assets import navigation_frame, navigation_scene_assets
from tests.test_particlefield_usd import _splat
from tests.test_website_locomotion_task import _corridor


def _base(tmp_path):
    base, path = _corridor(tmp_path)
    data, _ = _splat(32)
    data.scales[:] = -5
    splat = tmp_path / "scene.ply"
    write_standard_3dgs_ply(data, splat)
    base.update(splat_path=str(splat), splat_digest=_sha256_file(splat), up_axis="Y")
    return base, path, data


def test_the_scene_files_are_authored_with_no_spend_in_the_compilers_frame(tmp_path):
    base, path, data = _base(tmp_path)
    assets = navigation_scene_assets(base_scene=base, output_root=tmp_path / "assets")
    assert assets["paid_construction_performed"] is False and assets["scene_edited"] is False
    assert assets["claim_ceiling"] == "development_only"
    for role in ("appearance", "collision"):
        file = Path(assets[role]["path"])
        assert file.is_file() and "sha256:" + hashlib.sha256(file.read_bytes()).hexdigest() == assets[role]["sha256"]
    # The collider's floor sits where the compiler put it (Z up, metres): the same frame.
    stage = Usd.Stage.Open(assets["collision"]["path"])
    assert UsdGeom.GetStageUpAxis(stage) == "Z"
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    bounds = cache.ComputeWorldBound(stage.GetPseudoRoot()).ComputeAlignedRange()
    facts = scene_facts(base_scene=base, camera_path_levelled_m=path)
    assert bounds.GetMin()[2] == pytest.approx(facts["floor_height_world_m"] - 0.02, abs=0.03)
    # The appearance carries the same transform the frame declares, Gaussians untouched.
    field_stage = Usd.Stage.Open(assets["appearance"]["path"])
    field = next(prim for prim in field_stage.Traverse() if prim.GetTypeName() == "ParticleField3DGaussianSplat")
    np.testing.assert_allclose(np.asarray(field.GetAttribute("positions").Get()), data.xyz)
    # Idempotent: the same inputs give the same record.
    assert navigation_scene_assets(base_scene=base, output_root=tmp_path / "assets") == assets


def test_an_unknown_frame_is_refused(tmp_path):
    base, _, _ = _base(tmp_path)
    for bad in ({"up_axis": "X"}, {"meters_per_unit": 0}, {"meters_per_unit": float("nan")}):
        with pytest.raises(ValueError, match="website_navigation_scene_frame_invalid"):
            navigation_frame({**base, **bad})
